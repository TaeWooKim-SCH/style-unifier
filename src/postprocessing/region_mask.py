"""Region masking 모듈 (D3).

사용자 마스크 또는 자동 마스크를 이용해 변환 영역을 공간적으로 제한한다.
'얼굴 보존, 옷만 변환'과 같은 selective stylization을 지원한다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
_VALID_AUTO_MODES = frozenset({"face_preserve", "focal", "none"})

# focal 마스크에서 중앙 영역이 차지하는 비율 (각 축 기준)
_FOCAL_RATIO = 0.8


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------


def apply_region_mask(
    source: Image.Image,
    transformed: Image.Image,
    mask: npt.NDArray[np.float32],
    *,
    feather_px: int = 4,
) -> Image.Image:
    """mask가 1인 영역만 transformed를 사용하고 0인 영역은 source를 유지한다.

    경계는 ``feather_px`` 만큼 Gaussian blur로 부드럽게 처리한다.
    alpha 채널은 source와 transformed 각각의 alpha를 동일한 mask 비율로
    블렌딩해 자연스러운 합성을 보장한다.

    Args:
        source: 원본 PIL Image (RGB 또는 RGBA).
        transformed: 변환된 PIL Image. ``source`` 와 같은 크기여야 한다.
        mask: ``(H, W)`` float32 in [0, 1]. 1 = 변환 영역, 0 = 원본 보존.
            배열 shape은 ``(source.height, source.width)`` 와 일치해야 한다.
        feather_px: Gaussian filter sigma (px). 0이면 feathering 없음.

    Returns:
        합성된 RGBA PIL Image. source가 RGB면 alpha=255로 처리한다.

    Raises:
        ValueError: ``source.size != transformed.size``, mask shape 불일치,
            또는 ``feather_px`` 가 음수일 때.

    Example:
        >>> src = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        >>> tgt = Image.new("RGBA", (64, 64), (0, 0, 255, 255))
        >>> mask = np.ones((64, 64), dtype=np.float32)
        >>> result = apply_region_mask(src, tgt, mask, feather_px=0)
        >>> result.mode
        'RGBA'
    """
    if feather_px < 0:
        raise ValueError(f"feather_px must be >= 0, got {feather_px}")

    _validate_size_match(source, transformed)
    mask = _prepare_mask(mask, source, feather_px)

    src_arr = _to_rgba_array(source)
    tgt_arr = _to_rgba_array(transformed)

    blended = _blend_arrays(src_arr, tgt_arr, mask)

    result = Image.fromarray(blended, mode="RGBA")
    logger.debug(
        "apply_region_mask complete: size=%s, feather_px=%d",
        result.size,
        feather_px,
    )
    return result


def auto_mask(
    source: Image.Image,
    *,
    mode: str = "face_preserve",
) -> npt.NDArray[np.float32]:
    """프리셋 자동 마스크를 생성한다.

    지원 모드:
      - ``"face_preserve"``: 1차 구현은 full mask(전체 변환) 반환 + WARNING.
        face detection은 그룹 E 이후 구현 예정.
      - ``"focal"``: 이미지 중앙 80% 사각형 영역만 1.0. 캐릭터·아이템 중앙 정렬 가정.
      - ``"none"``: 전체 1.0 (= region mask 비활성화와 동등).

    Args:
        source: 원본 PIL Image. 크기 정보만 사용한다.
        mode: 위 3개 중 하나. 기본값은 ``"face_preserve"``.

    Returns:
        ``(H, W)`` float32 in [0, 1].

    Raises:
        ValueError: 알 수 없는 모드일 때.

    Example:
        >>> img = Image.new("RGBA", (128, 128))
        >>> mask = auto_mask(img, mode="none")
        >>> mask.shape
        (128, 128)
        >>> float(mask.min()), float(mask.max())
        (1.0, 1.0)
    """
    if mode not in _VALID_AUTO_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_AUTO_MODES)}, got {mode!r}")

    w, h = source.size

    if mode == "none":
        return np.ones((h, w), dtype=np.float32)

    if mode == "face_preserve":
        logger.warning(
            "auto_mask face_preserve mode: face detection not implemented;"
            + " falling back to full mask. Set explicit mask for face preservation."
        )
        return np.ones((h, w), dtype=np.float32)

    # mode == "focal"
    return _make_focal_mask(h, w, ratio=_FOCAL_RATIO)


# ---------------------------------------------------------------------------
# Private 헬퍼
# ---------------------------------------------------------------------------


def _validate_size_match(source: Image.Image, transformed: Image.Image) -> None:
    """source와 transformed의 크기가 일치하는지 검증한다.

    Args:
        source: 원본 이미지.
        transformed: 변환된 이미지.

    Raises:
        ValueError: 크기가 다를 때.
    """
    if source.size != transformed.size:
        raise ValueError(
            f"source.size {source.size} != transformed.size {transformed.size}."
            + " Resize one image before calling apply_region_mask()."
        )


def _prepare_mask(
    mask: npt.NDArray[np.float32],
    reference: Image.Image,
    feather_px: int,
) -> npt.NDArray[np.float32]:
    """Mask shape 검증, dtype 변환, clip, feathering을 순서대로 적용한다.

    Args:
        mask: 입력 마스크 배열.
        reference: 크기 기준 이미지.
        feather_px: Gaussian filter sigma.

    Returns:
        처리 완료된 ``(H, W)`` float32 마스크.

    Raises:
        ValueError: mask가 2-D가 아니거나, 크기가 reference와 불일치할 때.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got ndim={mask.ndim}")

    ref_w, ref_h = reference.size
    mask_h, mask_w = mask.shape
    if mask_h != ref_h or mask_w != ref_w:
        raise ValueError(
            f"mask shape ({mask_h}, {mask_w}) does not match image size (H={ref_h}, W={ref_w})."
        )

    prepared = np.clip(mask.astype(np.float32), 0.0, 1.0)

    if feather_px > 0:
        from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

        prepared = np.asarray(
            gaussian_filter(prepared, sigma=feather_px),
            dtype=np.float32,
        )

    return prepared


def _to_rgba_array(image: Image.Image) -> npt.NDArray[np.uint8]:
    """PIL Image를 RGBA uint8 배열로 변환한다.

    RGB 이미지는 alpha=255를 추가한다.

    Args:
        image: 변환할 PIL Image.

    Returns:
        ``(H, W, 4)`` uint8 RGBA 배열.
    """
    if image.mode == "RGBA":
        return np.array(image, dtype=np.uint8)

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=-1)


def _blend_arrays(
    src: npt.NDArray[np.uint8],
    tgt: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    """source와 transformed 배열을 mask 비율로 블렌딩한다.

    RGB와 alpha 채널 모두 동일한 mask로 블렌딩한다.

    Args:
        src: ``(H, W, 4)`` uint8 source RGBA.
        tgt: ``(H, W, 4)`` uint8 transformed RGBA.
        mask: ``(H, W)`` float32 in [0, 1]. 1 = tgt 사용.

    Returns:
        ``(H, W, 4)`` uint8 블렌딩 결과.
    """
    src_f = src.astype(np.float32)
    tgt_f = tgt.astype(np.float32)

    weight = mask[..., None]  # (H, W, 1) 브로드캐스트용
    blended = tgt_f * weight + src_f * (1.0 - weight)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _make_focal_mask(h: int, w: int, *, ratio: float) -> npt.NDArray[np.float32]:
    """이미지 중앙 영역만 1.0인 사각형 마스크를 생성한다.

    Args:
        h: 이미지 높이.
        w: 이미지 너비.
        ratio: 각 축에서 중앙 영역이 차지하는 비율 (0~1).

    Returns:
        ``(H, W)`` float32 마스크. 중앙 영역 1.0, 주변 0.0.
    """
    mask = np.zeros((h, w), dtype=np.float32)

    margin_h = int(h * (1.0 - ratio) / 2)
    margin_w = int(w * (1.0 - ratio) / 2)

    top = margin_h
    bottom = h - margin_h
    left = margin_w
    right = w - margin_w

    mask[top:bottom, left:right] = 1.0

    logger.debug(
        "_make_focal_mask: h=%d, w=%d, ratio=%.2f, roi=[%d:%d, %d:%d]",
        h,
        w,
        ratio,
        top,
        bottom,
        left,
        right,
    )
    return mask
