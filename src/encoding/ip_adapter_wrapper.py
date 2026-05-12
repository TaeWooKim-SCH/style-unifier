"""IP-Adapter CLIP 이미지 임베딩 추출 모듈.

Reference 이미지의 CLIP 이미지 임베딩을 추출·집계한다.
IP-Adapter weight의 pipeline 주입은 ``StyleUnificationPipeline.load_ip_adapter()`` 의
책임이므로 이 모듈은 encoding 로직만 담당한다.

이 분리 덕분에:
- StyleAligned 응용(그룹 D5)에서 embedding 단계를 명시적으로 제어할 수 있다.
- 동일 reference에 대한 반복 인코딩 비용을 인스턴스 캐시로 절감한다.
- 단위 테스트가 모델 없이도 mock으로 검증 가능하다.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as functional
from PIL import Image

from src.utils.device import get_device, get_dtype
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

logger = get_logger(__name__)

# IP-Adapter Plus (ViT-H) image encoder 의 HuggingFace subfolder
_DEFAULT_ENCODER_REPO = "h94/IP-Adapter"
_DEFAULT_ENCODER_SUBFOLDER = "models/image_encoder"

# cosine similarity softmax temperature for "weighted" strategy
_WEIGHTED_TEMPERATURE = 0.1

# valid aggregation strategies
_VALID_STRATEGIES = frozenset({"mean", "weighted", "first"})


def _image_cache_key(image: Image.Image) -> str:
    """PIL Image를 SHA-256 hex digest 캐시 키로 변환한다.

    동일 PIL 객체가 아닌 동일 픽셀 내용 기준으로 캐시를 히트시키기 위해
    tobytes()를 직렬화해 해싱한다. 비용은 있지만 그룹 D5의 batch 안에서
    같은 reference를 반복 인코딩하는 문제를 정확하게 회피한다.

    Args:
        image: PIL Image.

    Returns:
        40자 SHA-256 hex digest 문자열 (이미지 모드 + 크기 + 픽셀).
    """
    header = f"{image.mode}_{image.size[0]}_{image.size[1]}_".encode()
    digest = hashlib.sha256(header + image.tobytes()).hexdigest()
    return digest


class StyleEncoder:
    """Reference 이미지의 CLIP 이미지 임베딩을 추출·집계하는 유틸리티.

    IP-Adapter의 weight 주입은 ``StyleUnificationPipeline.load_ip_adapter()`` 의
    책임이다. 이 클래스는 그 보조 역할로서 다음을 담당한다:

    - 단일/다중 reference에서 CLIP image embedding 추출
    - Multi-reference 집계 전략 (mean / weighted / first)
    - 동일 reference 픽셀 내용에 대한 embedding 캐싱
      (그룹 D5에서 반복 호출 시 비용 절감)

    모델은 첫 ``encode_*`` 호출 시 lazy-load된다. ``__init__`` 에서 모델을
    로드하지 않으므로 모듈 import 또는 인스턴스 생성 자체는 비용이 없다.

    Attributes:
        encoder_repo: CLIPVisionModelWithProjection HuggingFace 저장소.
        encoder_subfolder: 저장소 내 image_encoder 서브폴더.
        device: 임베딩 계산 device.
        dtype: 임베딩 연산 dtype.
        image_encoder: lazy-load된 CLIPVisionModelWithProjection 인스턴스.
            첫 encode_* 호출 전에는 None.
        image_processor: lazy-load된 CLIPImageProcessor 인스턴스.
            첫 encode_* 호출 전에는 None.

    Example:
        >>> encoder = StyleEncoder()
        >>> embedding = encoder.encode_single(ref_image)  # (1, 257, 1280)
        >>> multi_emb = encoder.encode_multi([ref1, ref2], strategy="mean")
    """

    def __init__(
        self,
        encoder_repo: str = _DEFAULT_ENCODER_REPO,
        encoder_subfolder: str = _DEFAULT_ENCODER_SUBFOLDER,
        device: torch.device | None = None,
    ) -> None:
        """StyleEncoder 초기화. 모델 로딩은 첫 encode_* 호출 시로 지연된다.

        Args:
            encoder_repo: CLIPVisionModelWithProjection을 포함한 HuggingFace 저장소.
                기본값은 ``"h94/IP-Adapter"``.
            encoder_subfolder: 저장소 내 image_encoder 경로.
                기본값은 ``"models/image_encoder"``.
            device: 추론에 사용할 device. None이면 ``get_device()`` 로 자동 선택.
        """
        self.encoder_repo = encoder_repo
        self.encoder_subfolder = encoder_subfolder
        self.device: torch.device = device if device is not None else get_device()
        self.dtype: torch.dtype = get_dtype(self.device)

        # lazy-load 대상 — 첫 encode_* 호출 전까지 None
        self.image_encoder: CLIPVisionModelWithProjection | None = None
        self.image_processor: CLIPImageProcessor | None = None

        # 인스턴스별 embedding 캐시: sha256_hex → (1, seq_len, dim) Tensor (CPU)
        self._cache: dict[str, torch.Tensor] = {}

        logger.info(
            "StyleEncoder initialized (repo=%s, subfolder=%s, device=%s, dtype=%s)",
            encoder_repo,
            encoder_subfolder,
            self.device,
            self.dtype,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """image_encoder와 image_processor가 로드되어 있지 않으면 로드한다."""
        if self.image_encoder is not None:
            return

        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "transformers is required for StyleEncoder. Install it with: pip install transformers"
            ) from exc

        logger.info(
            "Loading CLIPVisionModelWithProjection: %s/%s",
            self.encoder_repo,
            self.encoder_subfolder,
        )

        raw_encoder: Any = CLIPVisionModelWithProjection.from_pretrained(
            self.encoder_repo,
            subfolder=self.encoder_subfolder,
        )
        moved_encoder: Any = raw_encoder.to(device=self.device, dtype=self.dtype)
        moved_encoder.eval()
        self.image_encoder = cast("CLIPVisionModelWithProjection", moved_encoder)

        raw_processor: Any = CLIPImageProcessor.from_pretrained(
            self.encoder_repo,
            subfolder=self.encoder_subfolder,
        )
        self.image_processor = cast("CLIPImageProcessor", raw_processor)

        logger.info("CLIPVisionModelWithProjection loaded successfully.")

    def _encode_raw(self, image: Image.Image) -> torch.Tensor:
        """단일 이미지를 (1, seq_len, dim) embedding으로 인코딩한다.

        캐시를 확인하고, miss 시에만 모델을 호출한다.

        Args:
            image: PIL Image (RGBA, RGB 모두 허용).

        Returns:
            (1, seq_len, dim) float32 Tensor (CPU에 저장).
        """
        cache_key = _image_cache_key(image)

        if cache_key in self._cache:
            logger.debug("Cache hit for image (key=%s...)", cache_key[:8])
            return self._cache[cache_key]

        self._ensure_loaded()
        if self.image_encoder is None or self.image_processor is None:
            raise RuntimeError(
                "StyleEncoder is not fully loaded after _ensure_loaded(). "
                "This indicates a logic bug in the loader."
            )

        # CLIPImageProcessor는 RGB 입력만 처리하므로 변환
        rgb_image = image.convert("RGB")

        pixel_values = self.image_processor(
            images=rgb_image,
            return_tensors="pt",
        ).pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        with torch.inference_mode():
            outputs = self.image_encoder(pixel_values=pixel_values)

        # IP-Adapter Plus는 last_hidden_state 전체 sequence를 사용한다.
        # image_embeds (CLS 단일 벡터)는 일반 IP-Adapter에서만 사용.
        embedding: torch.Tensor = outputs.last_hidden_state  # (1, seq_len, dim)
        embedding = embedding.float().cpu()

        self._cache[cache_key] = embedding
        logger.debug(
            "Encoded image (key=%s..., shape=%s)",
            cache_key[:8],
            tuple(embedding.shape),
        )
        return embedding

    def _aggregate_weighted(
        self,
        embeddings: list[torch.Tensor],
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """target_embedding과의 cosine 유사도를 가중치로 embedding을 집계한다.

        Args:
            embeddings: 각각 (1, seq_len, dim) CPU float32 Tensor 리스트.
            target_embedding: (1, seq_len, dim) CPU float32 Tensor.

        Returns:
            (1, seq_len, dim) float32 Tensor.
        """
        # 시퀀스 평균으로 sentence-level vector 생성: (1, dim)
        target_vec = target_embedding.mean(dim=1)  # (1, dim)

        similarities: list[torch.Tensor] = []
        for emb in embeddings:
            ref_vec = emb.mean(dim=1)  # (1, dim)
            sim = functional.cosine_similarity(ref_vec, target_vec, dim=-1)  # (1,)
            similarities.append(sim)

        # (N,) → softmax weights
        sim_tensor = torch.cat(similarities, dim=0)  # (N,)
        weights = torch.softmax(sim_tensor / _WEIGHTED_TEMPERATURE, dim=0)  # (N,)

        logger.debug(
            "Weighted strategy — cosine sims: %s, weights: %s",
            sim_tensor.tolist(),
            weights.tolist(),
        )

        # 가중 합산: each weight * (1, seq_len, dim)
        stacked = torch.stack([emb.squeeze(0) for emb in embeddings], dim=0)  # (N, seq_len, dim)
        weighted_sum = (weights[:, None, None] * stacked).sum(
            dim=0, keepdim=True
        )  # (1, seq_len, dim)
        return weighted_sum

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_single(self, reference: Image.Image) -> torch.Tensor:
        """단일 reference를 (1, seq_len, dim) embedding으로 인코딩한다.

        IP-Adapter Plus (ViT-H) 기준으로 seq_len=257, dim=1280.
        캐시가 있으면 모델 호출 없이 반환한다.

        Args:
            reference: PIL Image. RGBA, RGB 모두 허용.

        Returns:
            (1, seq_len, dim) float32 Tensor (detached, CPU).

        Raises:
            ImportError: transformers가 설치되어 있지 않을 때.
            OSError: 모델 다운로드 또는 로딩에 실패했을 때.
        """
        embedding = self._encode_raw(reference)
        logger.debug("encode_single complete: shape=%s", tuple(embedding.shape))
        return embedding

    def encode_multi(
        self,
        references: list[Image.Image],
        *,
        strategy: str = "mean",
        target_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """다중 reference를 하나의 집계 embedding으로 변환한다.

        Args:
            references: PIL Image 리스트 (N >= 1).
            strategy: 집계 전략.

                - ``"mean"``: uniform 평균. 기본값이자 안전한 baseline.
                - ``"weighted"``: target_embedding과의 cosine 유사도 softmax 가중.
                  target_embedding이 None이면 "mean"으로 자동 fallback.
                - ``"first"``: references[0]만 인코딩하고 반환. 단일 reference
                  baseline 시나리오에서 오버헤드 없이 사용.

            target_embedding: "weighted" 전략에서 가중치 기준으로 쓸 embedding.
                (1, seq_len, dim) 또는 (seq_len, dim) float32 Tensor.
                None이면 "weighted"가 "mean"으로 fallback한다.

        Returns:
            (1, seq_len, dim) float32 Tensor (detached, CPU).

        Raises:
            ValueError: references가 비어 있을 때.
            ValueError: strategy가 지원되지 않는 값일 때.
            ImportError: transformers가 설치되어 있지 않을 때.
            OSError: 모델 다운로드 또는 로딩에 실패했을 때.
        """
        if not references:
            raise ValueError("references must not be empty.")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}, got {strategy!r}"
            )

        if strategy == "first":
            result = self._encode_raw(references[0])
            logger.debug("encode_multi (first): shape=%s", tuple(result.shape))
            return result

        embeddings = [self._encode_raw(ref) for ref in references]

        if len(embeddings) == 1:
            result = embeddings[0]
            logger.debug("encode_multi single-ref shortcut: shape=%s", tuple(result.shape))
            return result

        if strategy == "mean":
            stacked = torch.stack([emb.squeeze(0) for emb in embeddings], dim=0)
            result = stacked.mean(dim=0, keepdim=True)
            logger.debug(
                "encode_multi (mean, N=%d): shape=%s", len(embeddings), tuple(result.shape)
            )
            return result

        # strategy == "weighted"
        if target_embedding is None:
            logger.warning(
                "encode_multi strategy='weighted' requires target_embedding. Falling back to 'mean'."
            )
            stacked = torch.stack([emb.squeeze(0) for emb in embeddings], dim=0)
            result = stacked.mean(dim=0, keepdim=True)
            return result

        # target_embedding을 (1, seq_len, dim)으로 정규화
        tgt = target_embedding.float().cpu()
        if tgt.ndim == 2:
            tgt = tgt.unsqueeze(0)

        result = self._aggregate_weighted(embeddings, tgt)
        logger.debug(
            "encode_multi (weighted, N=%d): shape=%s", len(embeddings), tuple(result.shape)
        )
        return result

    def clear_cache(self) -> None:
        """Embedding 캐시를 비운다.

        같은 reference PIL 객체를 새 세션에서 재인코딩해야 하거나
        메모리를 확보해야 할 때 명시적으로 호출한다.
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info("StyleEncoder cache cleared (%d entries removed).", count)
