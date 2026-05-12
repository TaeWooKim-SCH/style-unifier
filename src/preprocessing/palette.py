"""팔레트 추출 모듈.

이미지에서 k-means 클러스터링으로 주요 색상을 추출한다.
결과는 후처리 단계의 팔레트 양자화 및 배치 일관성 측정에 사용된다.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_palette(
    image: Image.Image,
    k: int = 12,
    *,
    ignore_alpha: bool = True,
    random_state: int = 42,
) -> npt.NDArray[np.uint8]:
    """k-means로 (k, 3) uint8 RGB palette를 추출한다.

    픽셀 수가 k 미만이면 0으로 채운 (k, 3) 배열을 반환한다.
    결정론적 동작을 위해 random_state를 항상 사용한다.

    Args:
        image: 입력 PIL Image. RGB, RGBA 모두 허용.
        k: 클러스터 수. 추출할 대표 색상의 개수.
        ignore_alpha: True면 alpha < 128인 픽셀을 클러스터링에서 제외한다.
            RGB 입력이면 alpha=255로 가정해 모든 픽셀을 포함한다.
        random_state: KMeans 결정론적 동작용 시드.

    Returns:
        (k, 3) uint8 ndarray. 각 행이 대표 색상의 RGB 값.

    Raises:
        ValueError: k < 1일 때.
        ImportError: scikit-learn이 설치되어 있지 않을 때.

    Example:
        >>> img = Image.open("char.png")
        >>> palette = extract_palette(img, k=12)
        >>> palette.shape
        (12, 3)
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    try:
        from sklearn.cluster import KMeans  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for palette extraction."
            + " Install it with: pip install scikit-learn"
        ) from exc

    # RGBA로 통일 (RGB 입력은 alpha=255로 패딩)
    rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)

    if ignore_alpha:
        alpha_mask = rgba_arr[:, :, 3] >= 128
        pixels = rgba_arr[alpha_mask][:, :3]
    else:
        pixels = rgba_arr.reshape(-1, 4)[:, :3]

    fallback: npt.NDArray[np.uint8] = np.zeros((k, 3), dtype=np.uint8)

    if len(pixels) < k:
        logger.warning("Pixel count (%d) < k (%d): returning zero-padded palette.", len(pixels), k)
        return fallback

    logger.debug("Running KMeans palette extraction (k=%d, pixels=%d)", k, len(pixels))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)  # type: ignore[call-arg]
    kmeans.fit(pixels.astype(np.float32))

    palette: npt.NDArray[np.uint8] = kmeans.cluster_centers_.astype(np.uint8)
    logger.debug("Palette extracted: shape=%s", palette.shape)
    return palette
