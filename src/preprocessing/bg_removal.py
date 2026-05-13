"""배경 분리 모듈.

게임 에셋의 전경을 분리해 RGBA 이미지와 alpha mask를 반환한다.
입력이 이미 RGBA이고 투명 영역이 있으면 모델 호출 없이 빠른 경로를 사용한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from src.utils.device import get_device, get_dtype
from src.utils.logging import get_logger

logger = get_logger(__name__)

# 모델 캐시: model_name → (model, transform) 쌍
_RMBG_CACHE: dict[str, tuple[object, object]] = {}

# RMBG-1.4 전처리 입력 해상도
_RMBG_INPUT_SIZE = 1024


def _load_rmbg_model(model_name: str) -> tuple[object, object]:
    """RMBG 모델과 전처리 transform을 lazy-load해 캐시에 저장한다.

    Args:
        model_name: HuggingFace 모델 식별자.

    Returns:
        (model, transform) 튜플.

    Raises:
        ImportError: transformers가 설치되어 있지 않을 때.
        OSError: 모델 다운로드 또는 로딩에 실패했을 때.
    """
    if model_name in _RMBG_CACHE:
        return _RMBG_CACHE[model_name]

    try:
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise ImportError(
            "transformers and torchvision are required for bg_removal."
            + " Install them with: pip install transformers torchvision"
        ) from exc

    # RMBG는 CPU에서 fp32로 실행한다. RTX 4070 12GB는 SDXL+ControlNet+IP-Adapter (cpu_offload
    # 활성)로도 빠듯하여, 1024×1024 source upscale 시 RMBG GPU 상주 시 OOM이 발생한다.
    # RMBG는 transform당 1회만 호출되므로 CPU 5-10초 추가 비용은 수용 가능하다.
    device = torch.device("cpu")
    dtype = torch.float32

    logger.info("Loading RMBG model: %s (device=%s, dtype=%s)", model_name, device, dtype)

    model = AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model = model.to(device=device, dtype=dtype)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((_RMBG_INPUT_SIZE, _RMBG_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    _RMBG_CACHE[model_name] = (model, transform)
    logger.info("RMBG model loaded and cached: %s", model_name)
    return model, transform


def _alpha_to_mask(image: Image.Image) -> np.ndarray:
    """RGBA 이미지의 alpha 채널을 [0, 1] float32 mask로 변환한다."""
    alpha = np.array(image.getchannel("A"), dtype=np.float32)
    return alpha / 255.0


def _is_fully_opaque(mask: np.ndarray) -> bool:
    """mask가 모두 1.0(완전 불투명)이면 True를 반환한다."""
    return bool(np.allclose(mask, 1.0))


def _run_rmbg_inference(
    rgb_image: Image.Image,
    model: object,
    transform: object,
) -> np.ndarray:
    """RMBG 모델로 배경 분리 추론을 실행해 (H, W) float32 mask를 반환한다.

    Args:
        rgb_image: RGB PIL Image.
        model: 로드된 AutoModelForImageSegmentation 인스턴스.
        transform: torchvision 전처리 transform.

    Returns:
        (H, W) float32 mask, 값 범위 [0, 1].
    """
    orig_w, orig_h = rgb_image.size

    # CRITICAL: input device/dtype을 model parameter device/dtype에 맞춰야 한다.
    # RMBG는 OOM 회피를 위해 CPU에서 실행되지만, model을 다른 곳으로 옮긴 경우에도 자동 정합.
    model_device: torch.device = torch.device("cpu")
    model_dtype: torch.dtype | None = None
    try:
        first_param = next(model.parameters())  # type: ignore[attr-defined]
        if isinstance(first_param.device, torch.device):
            model_device = first_param.device
        if isinstance(first_param.dtype, torch.dtype):
            model_dtype = first_param.dtype
    except (StopIteration, AttributeError, TypeError):
        # mock model (테스트) 또는 parameters() 미지원 — get_device() fallback
        model_device = get_device()
        model_dtype = None

    input_tensor: torch.Tensor = transform(rgb_image)  # type: ignore[operator]
    input_tensor = input_tensor.unsqueeze(0).to(model_device)
    if model_dtype is not None:
        input_tensor = input_tensor.to(dtype=model_dtype)

    with torch.inference_mode():
        preds = model(input_tensor)  # type: ignore[operator]

    # RMBG-1.4 forward 반환 형태:
    #   1. 단일 Tensor (transformers wrapper)
    #   2. list[Tensor] (단일 d list)
    #   3. (d_list, s_list) — briarmbg.py original. d_list[0]이 fused 최종 mask.
    # 어떤 형태든 첫 번째 Tensor에 도달할 때까지 unwrap 한다.
    pred: Any = preds
    while isinstance(pred, list | tuple):
        if len(pred) == 0:
            raise RuntimeError("RMBG model returned empty output")
        pred = pred[0]

    # (1, 1, H, W) → (H, W)
    mask_tensor = pred.squeeze().float().cpu()
    mask_tensor = torch.sigmoid(mask_tensor) if mask_tensor.min() < 0 else mask_tensor
    mask_tensor = mask_tensor.clamp(0.0, 1.0)

    # 원본 해상도로 리사이즈
    mask_pil = Image.fromarray((mask_tensor.numpy() * 255).astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    mask: np.ndarray = np.array(mask_pil, dtype=np.float32) / 255.0
    return mask


def remove_background(
    image: Image.Image,
    model_name: str = "briaai/RMBG-1.4",
    *,
    force_remove: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    """배경을 분리해 RGBA 이미지와 (H, W) float32 mask를 반환한다.

    입력이 RGBA이고 투명 픽셀이 하나라도 있으면(= alpha 채널이 전부 1.0이 아니면)
    모델 호출 없이 기존 alpha 채널을 mask로 그대로 사용한다.
    force_remove=True 이거나 alpha가 전부 불투명이면 RMBG 모델로 분리한다.

    Args:
        image: RGB 또는 RGBA PIL Image.
        model_name: HuggingFace 모델 식별자 (RMBG-1.4 기본).
        force_remove: True면 입력 alpha 채널 무시하고 강제로 모델 호출.

    Returns:
        (foreground_rgba, mask). mask는 [0, 1] float32, shape (H, W).

    Raises:
        ImportError: transformers 또는 torchvision 미설치 시.
        OSError: 모델 다운로드/로딩 실패 시.
    """
    # 빠른 경로: RGBA 입력이고 투명 영역이 존재하는 경우
    if image.mode == "RGBA" and not force_remove:
        existing_mask = _alpha_to_mask(image)
        if not _is_fully_opaque(existing_mask):
            logger.debug("RGBA input with existing alpha — skipping model inference.")
            rgba = image.convert("RGBA")
            return rgba, existing_mask

    # 느린 경로: 모델로 배경 분리
    rgb_image = image.convert("RGB")
    model, transform = _load_rmbg_model(model_name)

    logger.info("Running RMBG inference (model=%s)", model_name)
    mask = _run_rmbg_inference(rgb_image, model, transform)

    # mask를 alpha 채널로 적용해 RGBA 생성
    alpha_channel = (mask * 255).astype(np.uint8)
    rgb_arr = np.array(rgb_image, dtype=np.uint8)
    h, w = rgb_arr.shape[:2]
    rgba_arr = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_arr[:, :, :3] = rgb_arr
    rgba_arr[:, :, 3] = alpha_channel

    foreground_rgba = Image.fromarray(rgba_arr, mode="RGBA")
    logger.debug("bg_removal complete: output shape=%s", foreground_rgba.size)
    return foreground_rgba, mask
