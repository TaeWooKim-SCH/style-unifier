"""정량 메트릭 모듈 (C1).

CLIP Style Similarity, LPIPS Structure, DINOv2 Identity,
Palette Distance, Gram Matrix Distance 5가지 메트릭을 제공한다.

모든 모델 로드는 lazy + 모듈 레벨 캐시로 처리한다.
import 시 모델이 로드되지 않는다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.preprocessing.palette import extract_palette
from src.utils.device import get_device, get_dtype
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 모듈 레벨 캐시 — lazy 로딩. import 시 초기화 금지.
# ---------------------------------------------------------------------------

# key: (model_name, pretrained) → (model, preprocess_fn)
_CLIP_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}

# key: net 이름 → lpips.LPIPS 인스턴스
_LPIPS_CACHE: dict[str, Any] = {}

# key: model_name → (processor, model)
_DINO_CACHE: dict[str, tuple[Any, Any]] = {}

# key: "vgg19" → torchvision VGG feature extractor
_VGG_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _load_clip(
    model_name: str,
    pretrained: str,
) -> tuple[Any, Any]:
    """CLIP 모델과 전처리 함수를 lazy 로드하고 캐시한다."""
    key = (model_name, pretrained)
    if key not in _CLIP_CACHE:
        logger.info("Loading CLIP model: %s / %s", model_name, pretrained)
        try:
            import open_clip  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required. Install with: pip install open-clip-torch"
            ) from exc

        device = get_device()
        dtype = get_dtype(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        model = model.to(device=device, dtype=dtype)  # type: ignore[attr-defined]
        model.eval()  # type: ignore[attr-defined]
        _CLIP_CACHE[key] = (model, preprocess)
        logger.info("CLIP model loaded: %s / %s", model_name, pretrained)

    return _CLIP_CACHE[key]


def _load_lpips(net: str) -> Any:
    """LPIPS 모델을 lazy 로드하고 캐시한다."""
    if net not in _LPIPS_CACHE:
        logger.info("Loading LPIPS model (net=%s)", net)
        try:
            import lpips  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("lpips is required. Install with: pip install lpips") from exc

        device = get_device()
        loss_fn = lpips.LPIPS(net=net).to(device)  # type: ignore[attr-defined]
        loss_fn.eval()  # type: ignore[attr-defined]
        _LPIPS_CACHE[net] = loss_fn
        logger.info("LPIPS model loaded (net=%s)", net)

    return _LPIPS_CACHE[net]


def _load_dino(model_name: str) -> tuple[Any, Any]:
    """DINOv2 processor와 모델을 lazy 로드하고 캐시한다."""
    if model_name not in _DINO_CACHE:
        logger.info("Loading DINOv2 model: %s", model_name)
        try:
            from transformers import AutoImageProcessor, AutoModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            ) from exc

        device = get_device()
        dtype = get_dtype(device)
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()  # type: ignore[attr-defined]
        _DINO_CACHE[model_name] = (processor, model)
        logger.info("DINOv2 model loaded: %s", model_name)

    return _DINO_CACHE[model_name]


def _load_vgg19_features(
    layers: tuple[int, ...],
) -> Any:
    """VGG-19 feature extractor를 lazy 로드하고 캐시한다."""
    cache_key = "vgg19"
    if cache_key not in _VGG_CACHE:
        logger.info("Loading VGG-19 model for Gram matrix computation")
        try:
            from torchvision import models  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "torchvision is required. Install with: pip install torchvision"
            ) from exc

        device = get_device()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        vgg = vgg.features  # type: ignore[attr-defined]
        vgg = vgg.to(device)
        vgg.eval()  # type: ignore[attr-defined]
        _VGG_CACHE[cache_key] = vgg
        logger.info("VGG-19 loaded and cached")

    return _VGG_CACHE[cache_key]


def _pil_to_clip_tensor(image: Image.Image, preprocess: Any) -> Any:
    """PIL Image를 CLIP 전처리 텐서로 변환한다."""
    device = get_device()
    tensor = preprocess(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def _pil_to_lpips_tensor(image: Image.Image) -> Any:
    """PIL Image를 LPIPS용 [-1, 1] float BCHW 텐서로 변환한다."""
    import torchvision.transforms.functional as tvf  # type: ignore[import-untyped]

    device = get_device()
    tensor = tvf.to_tensor(image.convert("RGB"))  # [0, 1], (C, H, W)
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    return tensor.unsqueeze(0).to(device)


def _pil_to_vgg_tensor(image: Image.Image, size: int = 256) -> Any:
    """PIL Image를 VGG-19 입력용 텐서로 변환한다 (ImageNet 정규화 포함)."""
    import torchvision.transforms as tvt  # type: ignore[import-untyped]

    device = get_device()
    transform = tvt.Compose(
        [
            tvt.Resize((size, size)),
            tvt.ToTensor(),
            tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor: Any = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device)


def _cosine_similarity(a: Any, b: Any) -> float:
    """두 1D 또는 2D(B, D) 텐서의 cosine similarity를 반환한다."""
    import torch.nn.functional as nn_fn

    a_t = a.flatten()
    b_t = b.flatten()
    return float(nn_fn.cosine_similarity(a_t.unsqueeze(0), b_t.unsqueeze(0)).item())


def _gram_matrix(features: Any) -> Any:
    """(B, C, H, W) 피처 맵에서 Gram matrix (C, C)를 계산한다."""
    import torch

    b, c, h, w = features.shape
    feat = features.view(b, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))  # (B, C, C)
    return gram / (c * h * w)


def _rgb_uint8_to_lab(rgb_palette: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """(k, 3) uint8 RGB palette를 (k, 3) float32 LAB로 변환한다.

    skimage.color.rgb2lab는 입력이 (H, W, 3) float 또는 uint8을 기대하므로
    (k, 1, 3)으로 reshape 후 변환하고 (k, 3)으로 복원한다.
    """
    from skimage.color import rgb2lab  # type: ignore[import-untyped]

    rgb_f32 = rgb_palette.astype(np.float32) / 255.0  # (k, 3) float [0,1]
    rgb_hw3 = rgb_f32.reshape(-1, 1, 3)  # (k, 1, 3)
    lab_hw3 = rgb2lab(rgb_hw3)  # (k, 1, 3)
    return lab_hw3.reshape(-1, 3).astype(np.float32)  # (k, 3)


# ---------------------------------------------------------------------------
# 공개 메트릭 함수
# ---------------------------------------------------------------------------


def clip_style_similarity(
    generated: Image.Image,
    reference: Image.Image,
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
) -> float:
    """CLIP image embedding cosine similarity.

    높을수록 generated가 reference와 스타일적으로 유사하다.

    Args:
        generated: 변환된 출력 이미지 (RGBA 또는 RGB).
        reference: 스타일 reference 이미지 (RGBA 또는 RGB).
        model_name: open_clip 모델 이름. 기본값 "ViT-L-14".
        pretrained: 사전학습 가중치 태그. 기본값 "openai".
            TODO(Linux): ``open_clip.list_pretrained()`` 에서 ``("ViT-L-14", "openai")``
            조합이 실제로 존재하는지 확인. 부재 시 ``"laion2b_s34b_b79k"`` 로 대체 검토.

    Returns:
        cosine similarity float. 범위 -1 ~ 1. 실제로는 0.2 ~ 0.95.

    Raises:
        ImportError: open-clip-torch가 설치되지 않은 경우.
    """
    import torch

    model, preprocess = _load_clip(model_name, pretrained)

    with torch.inference_mode():
        gen_tensor = _pil_to_clip_tensor(generated, preprocess)
        ref_tensor = _pil_to_clip_tensor(reference, preprocess)
        gen_emb = model.encode_image(gen_tensor)  # type: ignore[union-attr]
        ref_emb = model.encode_image(ref_tensor)  # type: ignore[union-attr]

    return _cosine_similarity(gen_emb, ref_emb)


def lpips_structure(
    source: Image.Image,
    generated: Image.Image,
    net: str = "vgg",
) -> float:
    """LPIPS perceptual distance.

    낮을수록 generated가 source의 구조를 잘 보존하고 있다.
    두 이미지는 동일한 해상도여야 한다 (resize는 호출자 책임).

    Args:
        source: 원본 에셋 이미지.
        generated: 변환된 출력 이미지.
        net: LPIPS backbone. "vgg" | "alex" | "squeeze".

    Returns:
        LPIPS distance float. 범위 0 ~ 1+. 보통 0.1 ~ 0.5.

    Raises:
        ValueError: source와 generated의 크기가 다를 때.
        ImportError: lpips가 설치되지 않은 경우.
    """
    import torch

    if source.size != generated.size:
        msg = (
            f"source and generated must have the same size."
            f" Got source={source.size}, generated={generated.size}"
        )
        raise ValueError(msg)

    loss_fn = _load_lpips(net)

    with torch.inference_mode():
        src_tensor = _pil_to_lpips_tensor(source)
        gen_tensor = _pil_to_lpips_tensor(generated)
        dist = loss_fn(src_tensor, gen_tensor)  # type: ignore[operator]

    return float(dist.item())  # type: ignore[union-attr]


def dino_identity(
    source: Image.Image,
    generated: Image.Image,
    model_name: str = "facebook/dinov2-large",
) -> float:
    """DINOv2 CLS token embedding cosine similarity.

    높을수록 generated가 source 객체의 identity를 잘 유지한다.
    캐릭터의 경우 정체성 유지를, 사물·아이템의 경우 인식 가능성 유지를 측정한다.
    카테고리 분기 없이 동일 메트릭을 사용한다.

    Args:
        source: 원본 에셋 이미지.
        generated: 변환된 출력 이미지.
        model_name: HuggingFace DINOv2 모델 식별자. 기본값 "facebook/dinov2-large".

    Returns:
        cosine similarity float. 범위 -1 ~ 1.

    Raises:
        ImportError: transformers가 설치되지 않은 경우.
    """
    import torch

    processor, model = _load_dino(model_name)
    device = get_device()

    with torch.inference_mode():
        src_inputs = processor(images=source.convert("RGB"), return_tensors="pt")  # type: ignore[operator]
        gen_inputs = processor(images=generated.convert("RGB"), return_tensors="pt")  # type: ignore[operator]

        src_inputs = {k: v.to(device) for k, v in src_inputs.items()}
        gen_inputs = {k: v.to(device) for k, v in gen_inputs.items()}

        src_out = model(**src_inputs)  # type: ignore[operator]
        gen_out = model(**gen_inputs)  # type: ignore[operator]

        # CLS token: last_hidden_state[:, 0, :]
        src_cls = src_out.last_hidden_state[:, 0]  # (1, D)
        gen_cls = gen_out.last_hidden_state[:, 0]  # (1, D)

    return _cosine_similarity(src_cls, gen_cls)


def palette_distance(
    reference: Image.Image,
    generated: Image.Image,
    k: int = 12,
    random_state: int = 42,
) -> float:
    """Reference와 generated의 k-color palette 간 근사 EMD (LAB 공간).

    간이 EMD: 각 generated palette 점에서 reference palette 가장 가까운 점까지
    거리의 평균과, 반대 방향 평균의 합. 진짜 EMD는 아니지만 monotonic이고 빠르다.
    낮을수록 두 이미지의 색 분포가 유사하다.

    Args:
        reference: 스타일 reference 이미지.
        generated: 변환된 출력 이미지.
        k: 팔레트 색 수. 기본값 12.
        random_state: KMeans seed. 기본값 42.

    Returns:
        근사 EMD float (LAB 공간 유클리드 거리 기반). 낮을수록 유사.

    Raises:
        ImportError: scikit-image 또는 scikit-learn이 설치되지 않은 경우.
    """
    ref_palette = extract_palette(reference, k=k, random_state=random_state)
    gen_palette = extract_palette(generated, k=k, random_state=random_state)

    ref_lab = _rgb_uint8_to_lab(ref_palette)  # (k, 3)
    gen_lab = _rgb_uint8_to_lab(gen_palette)  # (k, 3)

    # gen → ref 방향: 각 gen 점에 대해 ref에서 가장 가까운 점까지 거리
    gen_to_ref = _mean_nearest_distance(gen_lab, ref_lab)
    # ref → gen 방향: 반대 방향
    ref_to_gen = _mean_nearest_distance(ref_lab, gen_lab)

    return float(gen_to_ref + ref_to_gen)


def _mean_nearest_distance(
    query: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
) -> float:
    """각 query 점에서 target 내 가장 가까운 점까지 거리의 평균."""
    # (n_query, n_target) pairwise distance matrix
    diff = query[:, np.newaxis, :] - target[np.newaxis, :, :]  # (Q, T, 3)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))  # (Q, T)
    min_dists = dist_matrix.min(axis=1)  # (Q,)
    return float(np.mean(min_dists))


def gram_matrix_distance(
    reference: Image.Image,
    generated: Image.Image,
    vgg_layers: tuple[int, ...] = (0, 5, 10, 19, 28),
) -> float:
    """Neural style transfer의 Gram matrix L2 거리 평균.

    각 지정 VGG-19 layer에서 feature map의 Gram matrix를 계산하고,
    두 이미지 사이의 L2 거리를 평균한다. 낮을수록 스타일 통계가 유사하다.
    입력은 256x256으로 resize된다.

    Args:
        reference: 스타일 reference 이미지.
        generated: 변환된 출력 이미지.
        vgg_layers: Gram matrix를 추출할 VGG-19 feature layer 인덱스.

    Returns:
        layer별 Gram matrix L2 거리의 평균 float.

    Raises:
        ImportError: torchvision이 설치되지 않은 경우.
    """
    import torch

    vgg = _load_vgg19_features(vgg_layers)

    with torch.inference_mode():
        ref_tensor = _pil_to_vgg_tensor(reference, size=256)
        gen_tensor = _pil_to_vgg_tensor(generated, size=256)

        distances: list[float] = []
        ref_feat = ref_tensor
        gen_feat = gen_tensor
        prev_idx = -1

        for layer_idx in sorted(vgg_layers):
            # 이전 layer 이후부터 현재 layer까지 순차 통과
            for sub_idx in range(prev_idx + 1, layer_idx + 1):
                layer = vgg[sub_idx]  # type: ignore[index]
                ref_feat = layer(ref_feat)  # type: ignore[operator]
                gen_feat = layer(gen_feat)  # type: ignore[operator]
            prev_idx = layer_idx

            ref_gram: torch.Tensor = _gram_matrix(ref_feat)  # (1, C, C)
            gen_gram: torch.Tensor = _gram_matrix(gen_feat)  # (1, C, C)
            dist = float(torch.nn.functional.mse_loss(ref_gram, gen_gram).item())
            distances.append(dist)
            logger.debug("Gram distance at layer %d: %.6f", layer_idx, dist)

    return float(np.mean(distances))
