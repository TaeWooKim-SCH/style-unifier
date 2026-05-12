"""팔레트 양자화 모듈.

생성 결과 색상을 reference 팔레트 방향으로 soft-snap한다.
완전 양자화가 아닌 보간 방식이므로 색상 품질 저하가 최소화된다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def quantize_to_palette(
    image: Image.Image,
    palette: npt.NDArray[np.uint8],
    *,
    strength: float = 0.7,
) -> Image.Image:
    """이미지 색상을 reference 팔레트 방향으로 보간한다.

    각 픽셀에서 가장 가까운 팔레트 색을 찾아 ``strength`` 만큼 그 방향으로 이동.
    ``strength=0`` 이면 원본 유지, ``strength=1`` 이면 완전 양자화.

    Args:
        image: 입력 PIL Image. RGBA 권장 (alpha 채널은 보존됨).
        palette: ``(k, 3)`` uint8 RGB. ``extract_palette()`` 반환값을 사용.
        strength: [0, 1] float. 보간 강도.

    Returns:
        같은 모드의 PIL Image. 색상이 팔레트 방향으로 이동된 결과.

    Raises:
        ValueError: ``strength`` 가 [0, 1] 밖이거나, ``palette`` shape이 (k, 3)이 아닐 때.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")

    _validate_palette_shape(palette)

    input_mode = image.mode
    has_alpha = input_mode == "RGBA"

    rgb_arr, alpha_channel = _split_rgba(image)

    blended_rgb = _blend_toward_palette(
        rgb_arr=rgb_arr,
        palette=palette,
        strength=strength,
    )

    result = _merge_rgba(blended_rgb, alpha_channel, has_alpha=has_alpha)

    logger.debug(
        "quantize_to_palette complete: mode=%s, strength=%.2f, palette_k=%d",
        result.mode,
        strength,
        len(palette),
    )
    return result


def _validate_palette_shape(palette: npt.NDArray[np.uint8]) -> None:
    """palette가 (k, 3) 2-D 배열인지 검증한다.

    Args:
        palette: 검증할 numpy 배열.

    Raises:
        ValueError: ndim이 2가 아니거나 두 번째 차원이 3이 아닐 때.
    """
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape (k, 3), got shape={palette.shape}")


def _split_rgba(
    image: Image.Image,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8] | None]:
    """이미지를 RGB float32 배열과 alpha 채널로 분리한다.

    Args:
        image: 입력 PIL Image. RGBA 또는 RGB.

    Returns:
        (rgb_float32, alpha_or_None). alpha는 RGBA 입력일 때만 반환.
    """
    if image.mode == "RGBA":
        arr = np.array(image, dtype=np.uint8)
        rgb: npt.NDArray[np.float32] = arr[..., :3].astype(np.float32)
        alpha: npt.NDArray[np.uint8] = arr[..., 3]
        return rgb, alpha

    rgb = np.array(image.convert("RGB"), dtype=np.float32)
    return rgb, None


def _blend_toward_palette(
    rgb_arr: npt.NDArray[np.float32],
    palette: npt.NDArray[np.uint8],
    strength: float,
) -> npt.NDArray[np.uint8]:
    """각 픽셀을 가장 가까운 팔레트 색 방향으로 보간한다.

    Args:
        rgb_arr: (H, W, 3) float32 RGB.
        palette: (k, 3) uint8 팔레트.
        strength: [0, 1] 보간 강도.

    Returns:
        (H, W, 3) uint8 보간된 RGB.
    """
    palette_f = palette.astype(np.float32)

    # (H, W, 1, 3) - (1, 1, k, 3) → (H, W, k, 3)
    diff = rgb_arr[..., None, :] - palette_f[None, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)  # (H, W, k)

    nearest_idx = distances.argmin(axis=-1)  # (H, W)
    nearest_color = palette_f[nearest_idx]  # (H, W, 3)

    blended = rgb_arr * (1.0 - strength) + nearest_color * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _merge_rgba(
    rgb: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.uint8] | None,
    *,
    has_alpha: bool,
) -> Image.Image:
    """보간된 RGB와 원본 alpha를 합쳐 PIL Image로 반환한다.

    Args:
        rgb: (H, W, 3) uint8 RGB 배열.
        alpha: (H, W) uint8 alpha 배열. has_alpha=False면 None.
        has_alpha: True면 RGBA 출력, False면 RGB 출력.

    Returns:
        PIL Image (RGBA 또는 RGB).
    """
    if has_alpha and alpha is not None:
        rgba_arr = np.concatenate([rgb, alpha[..., None]], axis=-1)
        return Image.fromarray(rgba_arr, mode="RGBA")

    return Image.fromarray(rgb, mode="RGB")
