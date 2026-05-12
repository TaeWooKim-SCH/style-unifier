"""알파 채널 복원 모듈.

생성된 RGB 이미지에 원본 source의 mask를 다시 적용해 RGBA로 복원한다.
경계는 Gaussian feathering으로 부드럽게 처리한다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def restore_alpha(
    generated: Image.Image,
    original_mask: npt.NDArray[np.float32],
    *,
    feather_px: int = 1,
) -> Image.Image:
    """생성된 RGB 이미지에 원본 mask를 적용해 RGBA로 복원한다.

    Args:
        generated: 생성된 PIL Image. RGB 또는 RGBA(alpha는 무시됨).
        original_mask: (H, W) float32 in [0, 1]. ``remove_background()`` 반환의 두 번째 값.
        feather_px: gaussian filter sigma (px). 0이면 feathering 없음.

    Returns:
        RGBA PIL Image. 크기는 ``generated``와 동일.

    Raises:
        ValueError: ``original_mask`` shape이 ``generated`` 크기와 불일치하면.
        ValueError: ``feather_px`` 가 음수일 때.
    """
    if feather_px < 0:
        raise ValueError(f"feather_px must be >= 0, got {feather_px}")

    gen_w, gen_h = generated.size
    _validate_mask_shape(original_mask, expected_h=gen_h, expected_w=gen_w)

    rgb_arr = np.array(generated.convert("RGB"), dtype=np.uint8)

    mask = np.clip(original_mask, 0.0, 1.0)
    mask = _apply_feathering(mask, feather_px)

    alpha_channel = (mask * 255).astype(np.uint8)

    rgba_arr = np.concatenate(
        [rgb_arr, alpha_channel[..., None]],
        axis=-1,
    )
    result = Image.fromarray(rgba_arr, mode="RGBA")

    logger.debug(
        "restore_alpha complete: size=%s, feather_px=%d",
        result.size,
        feather_px,
    )
    return result


def _validate_mask_shape(
    mask: npt.NDArray[np.float32],
    *,
    expected_h: int,
    expected_w: int,
) -> None:
    """Mask shape이 (expected_h, expected_w)인지 검증한다.

    Args:
        mask: 검증할 numpy 배열.
        expected_h: 기대하는 높이.
        expected_w: 기대하는 너비.

    Raises:
        ValueError: mask ndim이 2가 아니거나, shape이 불일치하면.
    """
    if mask.ndim != 2:
        raise ValueError(f"original_mask must be 2-D (H, W), got ndim={mask.ndim}")
    mask_h, mask_w = mask.shape
    if mask_h != expected_h or mask_w != expected_w:
        msg = (
            f"original_mask shape ({mask_h}, {mask_w}) does not match generated "
            f"image size ({expected_w}, {expected_h}). "
            f"Resize the mask before calling restore_alpha()."
        )
        raise ValueError(msg)


def _apply_feathering(
    mask: npt.NDArray[np.float32],
    feather_px: int,
) -> npt.NDArray[np.float32]:
    """mask에 Gaussian feathering을 적용한다.

    Args:
        mask: (H, W) float32 in [0, 1].
        feather_px: gaussian sigma. 0이면 원본 그대로 반환.

    Returns:
        feathering이 적용된 (H, W) float32 mask.
    """
    if feather_px == 0:
        return mask

    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

    feathered = np.asarray(
        gaussian_filter(mask.astype(np.float32), sigma=feather_px),
        dtype=np.float32,
    )
    return feathered
