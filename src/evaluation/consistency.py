"""배치 일관성 메트릭 모듈 (C2).

N개 출력 이미지 집합에서 style-unifier의 차별화 핵심 메트릭을 제공한다:
- sigma_palette: 팔레트 색 분포 표준편차 (LAB 공간)
- sigma_linewidth: 평균 라인 두께 표준편차
- sigma_shading: 밝기 분포 KL divergence

N=1 edge case는 모두 0.0을 반환한다 (분산 없음).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.preprocessing.palette import extract_palette
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _rgb_uint8_to_lab(rgb_palette: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """(k, 3) uint8 RGB palette를 (k, 3) float32 LAB로 변환한다."""
    from skimage.color import rgb2lab  # type: ignore[import-untyped]

    rgb_f32 = rgb_palette.astype(np.float32) / 255.0  # [0, 1]
    rgb_hw3 = rgb_f32.reshape(-1, 1, 3)  # (k, 1, 3)
    lab_hw3 = rgb2lab(rgb_hw3)  # (k, 1, 3)
    return lab_hw3.reshape(-1, 3).astype(np.float32)  # (k, 3)


def _image_to_lab_l_channel(image: Image.Image) -> npt.NDArray[np.float32]:
    """RGBA 이미지에서 alpha>0 픽셀의 LAB L* 채널 배열을 반환한다.

    반환되는 배열은 L* 값의 1D float32 배열 (범위 0 ~ 100).
    alpha>0 픽셀이 없으면 빈 배열을 반환한다.
    """
    from skimage.color import rgb2lab  # type: ignore[import-untyped]

    rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)  # (H, W, 4)
    alpha_mask = rgba_arr[:, :, 3] > 0  # (H, W) bool

    if not alpha_mask.any():
        logger.warning("No non-transparent pixels found; L* channel is empty.")
        return np.array([], dtype=np.float32)

    rgb_pixels = rgba_arr[:, :, :3][alpha_mask]  # (N, 3) uint8
    rgb_f32 = rgb_pixels.astype(np.float32) / 255.0  # (N, 3) float [0,1]

    # rgb2lab: (H, W, 3) 또는 (N, 1, 3) 형태 필요
    rgb_hw3 = rgb_f32.reshape(-1, 1, 3)  # (N, 1, 3)
    lab_hw3 = rgb2lab(rgb_hw3)  # (N, 1, 3)
    lab_arr = lab_hw3.reshape(-1, 3)  # (N, 3)

    return lab_arr[:, 0].astype(np.float32)  # L* channel (N,)


def _compute_canny_edge_mask(
    image: Image.Image,
    canny_low: int,
    canny_high: int,
) -> npt.NDArray[np.bool_]:
    """이미지에서 Canny edge binary mask를 반환한다.

    skimage.feature.canny 사용. 입력이 RGBA면 alpha>0 영역만 사용한다.
    """
    from skimage.feature import canny  # type: ignore[import-untyped]

    rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    alpha_mask = rgba_arr[:, :, 3] > 0

    # Grayscale 변환
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0  # (H, W)

    # alpha=0 영역은 0으로 마스킹해 edge 오검출 방지
    gray[~alpha_mask] = 0.0

    edge_mask = canny(
        gray,
        low_threshold=canny_low / 255.0,
        high_threshold=canny_high / 255.0,
    )
    return edge_mask.astype(np.bool_)


def _estimate_mean_line_thickness(edge_mask: npt.NDArray[np.bool_]) -> float:
    """Binary edge mask에서 평균 라인 두께를 추정한다.

    근사 방법: edge 영역의 distance transform 평균값 * 2.
    두꺼운 라인 → 큰 값, 가는 라인 → 작은 값. 결정론적, 빠름.

    edge pixel이 없으면 0.0을 반환한다.
    """
    from scipy.ndimage import distance_transform_edt  # type: ignore[import-untyped]

    if not edge_mask.any():
        return 0.0

    # edge 픽셀에서 edt를 직접 계산해야 한다. ``distance_transform_edt`` 는
    # 입력이 True/1인 셀에서 가장 가까운 False/0 셀까지의 거리를 반환하므로,
    # ``edge_mask`` 를 그대로 전달하면 edge 픽셀 위치에서 "가장 가까운 비-edge
    # 픽셀까지의 거리" = 라인 반두께 근사가 된다. ``~edge_mask`` 를 전달하면
    # edge 위치의 값이 항상 0이 되어 의미 있는 두께를 얻을 수 없다.
    dist: npt.NDArray[np.float64] = np.asarray(distance_transform_edt(edge_mask))
    edge_dist_values = dist[edge_mask]
    mean_half_thickness = float(np.mean(edge_dist_values))

    # 전체 두께 = 반두께 * 2
    return mean_half_thickness * 2.0


def _normalized_histogram(
    values: npt.NDArray[np.float32],
    n_bins: int,
    val_min: float = 0.0,
    val_max: float = 100.0,
    eps: float = 1e-8,
) -> npt.NDArray[np.float32]:
    """1D 값 배열을 정규화된 히스토그램(확률 분포)으로 변환한다.

    eps 스무딩으로 0-bin을 방지한다.
    """
    hist, _ = np.histogram(values, bins=n_bins, range=(val_min, val_max))
    hist_f = hist.astype(np.float32) + eps
    return hist_f / hist_f.sum()


def _symmetric_kl(
    p: npt.NDArray[np.float32],
    q: npt.NDArray[np.float32],
) -> float:
    """두 확률 분포 사이의 대칭 KL divergence (JSD 분자) 를 반환한다.

    symmetric KL = (KL(p||q) + KL(q||p)) / 2
    """
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return (kl_pq + kl_qp) / 2.0


# ---------------------------------------------------------------------------
# 공개 메트릭 함수
# ---------------------------------------------------------------------------


def sigma_palette(
    outputs: list[Image.Image],
    k: int = 12,
    random_state: int = 42,
) -> float:
    """N개 출력의 palette 색 분포 표준편차 (LAB 공간).

    각 출력에서 k-color palette를 추출(RGB) → LAB 변환 → 출력당 평균 LAB 색 계산.
    N개 평균 LAB 벡터의 채널별 표준편차를 L2 norm으로 스칼라화한다.

    낮을수록 N개 출력이 동일한 palette 가족에 속한다.

    Args:
        outputs: 같은 reference·다른 source로 변환된 N개 RGBA 이미지 리스트.
        k: 각 출력에서 추출할 palette 색 수. 기본값 12.
        random_state: KMeans seed. 기본값 42.

    Returns:
        scalar float. 0이면 모든 출력이 정확히 같은 평균 LAB 색.
        값이 클수록 출력 간 색 분포가 흩어짐.

    Raises:
        ValueError: outputs가 비어있을 때.
    """
    if not outputs:
        raise ValueError("outputs must be a non-empty list of images.")

    if len(outputs) == 1:
        logger.debug("sigma_palette: N=1, returning 0.0")
        return 0.0

    # 각 출력의 평균 LAB 색 계산 → (N, 3)
    mean_lab_per_output: list[npt.NDArray[np.float32]] = []
    for idx, img in enumerate(outputs):
        palette_rgb = extract_palette(img, k=k, random_state=random_state)  # (k, 3) uint8
        palette_lab = _rgb_uint8_to_lab(palette_rgb)  # (k, 3) float32
        mean_lab = palette_lab.mean(axis=0)  # (3,)
        mean_lab_per_output.append(mean_lab)
        logger.debug("Output %d: mean LAB = %s", idx, mean_lab)

    mean_lab_matrix = np.stack(mean_lab_per_output, axis=0)  # (N, 3)

    # 채널별 std → L2 norm
    per_channel_std = mean_lab_matrix.std(axis=0)  # (3,) std across N outputs
    result = float(np.linalg.norm(per_channel_std))

    logger.info("sigma_palette: N=%d, result=%.4f", len(outputs), result)
    return result


def sigma_linewidth(
    outputs: list[Image.Image],
    canny_low: int = 100,
    canny_high: int = 200,
) -> float:
    """N개 출력의 평균 라인 두께 표준편차 (px).

    각 출력에서 Canny edge를 추출하고 distance transform 기반으로
    평균 라인 두께를 추정한다. N개 추정값의 표준편차를 반환한다.

    낮을수록 N개 출력의 라인 두께가 일관된다.

    Args:
        outputs: 같은 reference·다른 source로 변환된 N개 이미지 리스트.
        canny_low: Canny low threshold (0-255 스케일). 기본값 100.
        canny_high: Canny high threshold (0-255 스케일). 기본값 200.

    Returns:
        scalar float. 라인 두께 추정값의 표준편차 (단위: px 근사).

    Raises:
        ValueError: outputs가 비어있을 때.
    """
    if not outputs:
        raise ValueError("outputs must be a non-empty list of images.")

    if len(outputs) == 1:
        logger.debug("sigma_linewidth: N=1, returning 0.0")
        return 0.0

    thicknesses: list[float] = []
    for idx, img in enumerate(outputs):
        edge_mask = _compute_canny_edge_mask(img, canny_low, canny_high)
        thickness = _estimate_mean_line_thickness(edge_mask)
        thicknesses.append(thickness)
        logger.debug("Output %d: mean_thickness=%.4f px", idx, thickness)

    result = float(np.std(thicknesses))
    logger.info("sigma_linewidth: N=%d, result=%.4f", len(outputs), result)
    return result


def sigma_shading(
    outputs: list[Image.Image],
    n_bins: int = 32,
) -> float:
    """N개 출력의 LAB L* 밝기 분포 평균 대칭 KL divergence.

    각 출력에서 alpha>0 픽셀의 L* 채널 히스토그램을 계산한다.
    모든 pairwise 대칭 KL divergence의 평균을 반환한다.

    낮을수록 N개 출력의 shading 분포가 일관된다.

    Args:
        outputs: 같은 reference·다른 source로 변환된 N개 이미지 리스트.
        n_bins: L* 히스토그램 bin 수. 기본값 32.

    Returns:
        scalar float. pairwise 대칭 KL divergence 평균. 0이면 완전 동일 분포.

    Raises:
        ValueError: outputs가 비어있을 때.
    """
    if not outputs:
        raise ValueError("outputs must be a non-empty list of images.")

    if len(outputs) == 1:
        logger.debug("sigma_shading: N=1, returning 0.0")
        return 0.0

    # 각 출력의 L* 히스토그램 계산
    histograms: list[npt.NDArray[np.float32]] = []
    for idx, img in enumerate(outputs):
        l_values = _image_to_lab_l_channel(img)  # (M,) float [0, 100]
        if len(l_values) == 0:
            logger.warning("Output %d has no non-transparent pixels; using uniform hist.", idx)
            hist = np.ones(n_bins, dtype=np.float32) / n_bins
        else:
            hist = _normalized_histogram(l_values, n_bins=n_bins, val_min=0.0, val_max=100.0)
        histograms.append(hist)
        logger.debug("Output %d: L* hist computed (n_bins=%d)", idx, n_bins)

    # 모든 pairwise 대칭 KL divergence 계산
    n = len(histograms)
    kl_values: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            sym_kl = _symmetric_kl(histograms[i], histograms[j])
            kl_values.append(sym_kl)

    result = float(np.mean(kl_values))
    logger.info("sigma_shading: N=%d, pairs=%d, result=%.4f", n, len(kl_values), result)
    return result
