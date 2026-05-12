"""§5.5.1 — 후처리 기반 배치 일관성 강제.

ADR-010의 안전한 baseline. mechanism level 차별화는 §5.5.2 (StyleAligned 응용)이며
이 모듈은 그 fallback이다. 본 모듈만으로 GPT Image 등 API-only 모델 대비 우위가
보장되지는 않지만, sigma_palette/sigma_linewidth/sigma_shading 메트릭에서 측정 가능한 감소를 제공한다.

`extract_style_statistics(reference)` → `enforce_consistency(outputs, stats)` 순서로 사용.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from src.postprocessing.palette_quantize import quantize_to_palette
from src.preprocessing.lineart import extract_lineart
from src.preprocessing.palette import extract_palette
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.encoding.ip_adapter_wrapper import StyleEncoder

logger = get_logger(__name__)

# 형태학 연산 최대 반복 횟수 (linewidth 보정이 너무 강해지지 않도록 제한)
_MAX_MORPH_ITERATIONS = 5


@dataclass(frozen=True)
class StyleStatistics:
    """Reference 1장에서 추출한 통계. 모든 출력에 동일하게 적용.

    Attributes:
        palette: (k, 3) uint8 RGB 색 중심.
        palette_weights: (k,) float32, 각 색의 픽셀 비율 (합=1).
        mean_linewidth: float, reference 라인 두께 평균 (px).
        linewidth_distribution: (n_bins,) float32, 라인 두께 히스토그램.
        shading_histogram: (n_bins,) float32, LAB L* 채널 히스토그램 (정규화됨).
        ip_adapter_embedding: torch.Tensor | None, reference의 CLIP image embedding (캐싱용).
            None이면 임베딩 캐싱 미적용 (즉 pipeline.transform_batch에서는 매번 계산).
    """

    palette: npt.NDArray[np.uint8]
    palette_weights: npt.NDArray[np.float32]
    mean_linewidth: float
    linewidth_distribution: npt.NDArray[np.float32]
    shading_histogram: npt.NDArray[np.float32]
    ip_adapter_embedding: torch.Tensor | None


def _compute_linewidth_stats(
    image: Image.Image,
    n_bins: int,
) -> tuple[float, npt.NDArray[np.float32]]:
    """Canny edge map에서 라인 두께 평균과 히스토그램을 계산한다.

    distance transform을 이용해 각 edge 픽셀에서 가장 가까운 비-edge 픽셀까지의
    거리를 구하고, 그 2배를 두께 근사값으로 사용한다.

    Args:
        image: 입력 PIL Image (RGB 또는 RGBA).
        n_bins: 히스토그램 bin 수.

    Returns:
        (mean_linewidth, linewidth_distribution) 튜플.
        edge 픽셀이 없으면 (0.0, zeros) 반환.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError(
            "scipy is required for linewidth computation. Install it with: pip install scipy"
        ) from exc

    lineart_img = extract_lineart(image, detector="canny")
    # canny 출력: 흰 선 / 검은 배경 (RGB). edge=흰색(255).
    gray_arr = np.array(lineart_img.convert("L"), dtype=np.uint8)
    edge_mask = gray_arr > 128  # True = edge 픽셀

    if not edge_mask.any():
        logger.warning("No edges detected; linewidth stats will be zero.")
        zeros: npt.NDArray[np.float32] = np.zeros(n_bins, dtype=np.float32)
        return 0.0, zeros

    # edge 마스크에서 직접 edt 계산 — distance_transform_edt는 True 셀에서
    # 가장 가까운 False 셀까지의 거리를 반환하므로 edge_mask 자체를 전달해야
    # edge 픽셀 위치에서 "가장 가까운 비-edge 픽셀까지의 거리" = 라인 반두께를
    # 얻을 수 있다. ~edge_mask 를 전달하면 edge 위치 값이 항상 0이 되어 두께
    # 측정이 무력화된다.
    distance_map: npt.NDArray[np.float64] = distance_transform_edt(edge_mask)  # type: ignore[assignment]

    # edge 픽셀 위치에서의 반두께 * 2 = 전체 두께
    thickness_values: npt.NDArray[np.float64] = distance_map[edge_mask] * 2.0  # type: ignore[index]

    mean_lw = float(np.mean(thickness_values))

    # 두께 히스토그램 (range: 0 ~ 합리적 최대값)
    max_thickness = max(float(np.percentile(thickness_values, 99)), 1.0)
    counts, _ = np.histogram(thickness_values, bins=n_bins, range=(0.0, max_thickness))
    total = counts.sum()
    distribution: npt.NDArray[np.float32] = (
        counts.astype(np.float32) / total if total > 0 else counts.astype(np.float32)
    )

    logger.debug("Linewidth stats: mean=%.2f, bins=%d", mean_lw, n_bins)
    return mean_lw, distribution


def _compute_shading_histogram(
    image: Image.Image,
    n_bins: int,
) -> npt.NDArray[np.float32]:
    """Alpha > 0인 영역의 LAB L* 채널 히스토그램을 계산한다.

    Args:
        image: 입력 PIL Image (RGB 또는 RGBA).
        n_bins: 히스토그램 bin 수.

    Returns:
        (n_bins,) float32 정규화된 히스토그램.
    """
    try:
        from skimage.color import rgb2lab
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for shading histogram. Install it with: pip install scikit-image"
        ) from exc

    rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    alpha_mask = rgba_arr[:, :, 3] > 0

    rgb_f32 = rgba_arr[:, :, :3].astype(np.float32) / 255.0
    lab_arr = rgb2lab(rgb_f32)  # (H, W, 3), L in [0, 100]
    l_channel = lab_arr[:, :, 0]  # (H, W)

    l_values = l_channel[alpha_mask]

    if len(l_values) == 0:
        logger.warning("No visible pixels (alpha > 0) found; shading histogram will be zero.")
        return np.zeros(n_bins, dtype=np.float32)

    counts, _ = np.histogram(l_values, bins=n_bins, range=(0.0, 100.0))
    total = counts.sum()
    histogram: npt.NDArray[np.float32] = (
        counts.astype(np.float32) / total if total > 0 else counts.astype(np.float32)
    )

    logger.debug(
        "Shading histogram computed: bins=%d, l_mean=%.2f", n_bins, float(np.mean(l_values))
    )
    return histogram


def _compute_palette_weights(
    image: Image.Image,
    palette: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """각 팔레트 색의 픽셀 비율을 계산한다.

    Alpha < 128인 픽셀은 제외하고, 나머지 픽셀을 가장 가까운 팔레트 색에 할당한다.

    Args:
        image: 입력 PIL Image (RGB 또는 RGBA).
        palette: (k, 3) uint8 RGB 팔레트.

    Returns:
        (k,) float32, 합=1로 정규화된 비율. 유효 픽셀 없으면 uniform.
    """
    rgba_arr = np.array(image.convert("RGBA"), dtype=np.uint8)
    alpha_mask = rgba_arr[:, :, 3] >= 128
    pixels = rgba_arr[alpha_mask][:, :3].astype(np.float32)

    k = len(palette)
    if len(pixels) == 0:
        logger.warning("No opaque pixels found; returning uniform palette weights.")
        return np.full(k, 1.0 / k, dtype=np.float32)

    palette_f = palette.astype(np.float32)
    # (N, 1, 3) - (1, k, 3) → (N, k, 3) → (N, k) distances
    diff = pixels[:, None, :] - palette_f[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)  # (N, k)
    nearest_idx = distances.argmin(axis=-1)  # (N,)

    counts = np.bincount(nearest_idx, minlength=k).astype(np.float32)
    total = counts.sum()
    weights: npt.NDArray[np.float32] = counts / total if total > 0 else counts

    logger.debug("Palette weights computed: k=%d", k)
    return weights


def extract_style_statistics(
    reference: Image.Image,
    *,
    palette_k: int = 12,
    n_bins: int = 32,
    cache_embedding: bool = False,
    style_encoder: StyleEncoder | None = None,
) -> StyleStatistics:
    """Reference 1장에서 일관성 강제용 통계를 추출한다.

    Args:
        reference: 참조 PIL Image (RGB 또는 RGBA).
        palette_k: 추출할 palette 색 수.
        n_bins: 히스토그램 bin 수 (linewidth, shading 공용).
        cache_embedding: True면 IP-Adapter용 image embedding을 미리 계산해 캐싱.
            ``transform_batch`` 에서 N번 재계산을 피하는 용도. **기본값은 False**
            — True 설정 시 CLIPVisionModelWithProjection 모델이 즉시 로드되므로
            CLAUDE.md 절대 원칙(윈도우/macOS에서 모델 추론 금지)을 위반할 수 있다.
            Linux RTX 4070 환경에서만 True로 설정할 것.
        style_encoder: 외부에서 만든 StyleEncoder 인스턴스. None이고
            ``cache_embedding=True`` 면 내부에서 생성.

    Returns:
        ``StyleStatistics`` 인스턴스.

    Raises:
        ValueError: ``palette_k`` 가 1 미만이거나 ``n_bins`` 가 1 미만일 때.
    """
    if palette_k < 1:
        raise ValueError(f"palette_k must be >= 1, got {palette_k}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    logger.info(
        "Extracting style statistics (palette_k=%d, n_bins=%d, cache_embedding=%s)",
        palette_k,
        n_bins,
        cache_embedding,
    )

    palette = extract_palette(reference, k=palette_k)
    palette_weights = _compute_palette_weights(reference, palette)
    mean_linewidth, linewidth_distribution = _compute_linewidth_stats(reference, n_bins)
    shading_histogram = _compute_shading_histogram(reference, n_bins)

    ip_adapter_embedding: torch.Tensor | None = None
    if cache_embedding:
        if style_encoder is None:
            from src.encoding.ip_adapter_wrapper import StyleEncoder

            style_encoder = StyleEncoder()
        logger.info("Computing IP-Adapter embedding for reference (model load will occur).")
        ip_adapter_embedding = style_encoder.encode_single(reference)

    stats = StyleStatistics(
        palette=palette,
        palette_weights=palette_weights,
        mean_linewidth=mean_linewidth,
        linewidth_distribution=linewidth_distribution,
        shading_histogram=shading_histogram,
        ip_adapter_embedding=ip_adapter_embedding,
    )

    logger.info(
        "StyleStatistics extracted: palette_k=%d, mean_lw=%.2f, embedding=%s",
        palette_k,
        mean_linewidth,
        "cached" if ip_adapter_embedding is not None else "None",
    )
    return stats


def _apply_linewidth_correction(
    image: Image.Image,
    target_mean_lw: float,
    strength: float,
) -> Image.Image:
    """형태학 연산으로 라인 두께를 target에 근사하게 보정한다.

    정밀도보다 결정론·단조성이 중요한 보수적 구현이다.
    iterations는 최대 _MAX_MORPH_ITERATIONS로 제한해 과도한 변형을 막는다.

    **한계**: 모든 라인을 균일하게 팽창/침식하므로 세밀한 두께 분포는 제어 불가.
    linewidth_strength=0.5 이하의 보수적 값을 권장한다.

    Args:
        image: RGBA PIL Image.
        target_mean_lw: 목표 라인 두께 평균 (px).
        strength: [0, 1] 보정 강도.

    Returns:
        RGBA PIL Image (alpha 채널 보존).
    """
    try:
        from scipy.ndimage import binary_dilation, binary_erosion
    except ImportError as exc:
        raise ImportError(
            "scipy is required for linewidth correction. Install it with: pip install scipy"
        ) from exc

    current_mean_lw, _ = _compute_linewidth_stats(image, n_bins=16)

    diff = target_mean_lw - current_mean_lw
    raw_iterations = round(abs(diff) * strength)
    iterations = min(raw_iterations, _MAX_MORPH_ITERATIONS)

    if iterations == 0:
        logger.debug("Linewidth correction: no adjustment needed (diff=%.2f, iterations=0)", diff)
        return image

    logger.debug(
        "Linewidth correction: diff=%.2f, iterations=%d, direction=%s",
        diff,
        iterations,
        "dilate" if diff > 0 else "erode",
    )

    rgba_arr = np.array(image, dtype=np.uint8)
    rgb = rgba_arr[:, :, :3]
    alpha = rgba_arr[:, :, 3]

    # 라인 마스크: 밝기 기준으로 선 영역 추정 (어두운 픽셀 = 선)
    gray = np.mean(rgb, axis=-1)
    line_mask = gray < 128

    if diff > 0:
        corrected_mask = binary_dilation(line_mask, iterations=iterations)
    else:
        corrected_mask = binary_erosion(line_mask, iterations=iterations)

    # 보정 마스크를 RGB에 적용: 선 영역은 검정, 비선 영역은 흰색 유지
    corrected: npt.NDArray[np.bool_] = np.asarray(corrected_mask, dtype=np.bool_)
    result_rgb = rgb.copy()
    result_rgb[corrected & ~line_mask] = 0  # 새로 추가된 선 픽셀
    result_rgb[~corrected & line_mask] = 255  # 제거된 선 픽셀

    rgba_result = np.concatenate([result_rgb, alpha[:, :, None]], axis=-1)
    return Image.fromarray(rgba_result, mode="RGBA")


def _apply_shading_correction(
    image: Image.Image,
    target_histogram: npt.NDArray[np.float32],
    strength: float,
) -> Image.Image:
    """LAB L* 채널 히스토그램 매칭으로 명암 분포를 보정한다.

    skimage.exposure.match_histograms로 완전 매칭 후 strength로 원본과 blend한다.

    Args:
        image: RGBA PIL Image.
        target_histogram: (n_bins,) float32 목표 shading 히스토그램. 현재는
            참조용으로 전달되며, match_histograms가 내부적으로 CDF를 사용한다.
        strength: [0, 1] 보정 강도. 0이면 원본, 1이면 완전 히스토그램 매칭.

    Returns:
        RGBA PIL Image (alpha 채널 보존).
    """
    try:
        from skimage.color import lab2rgb, rgb2lab
        from skimage.exposure import match_histograms
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for shading correction. Install it with: pip install scikit-image"
        ) from exc

    rgba_arr = np.array(image, dtype=np.uint8)
    rgb = rgba_arr[:, :, :3]
    alpha = rgba_arr[:, :, 3]

    rgb_f32 = rgb.astype(np.float32) / 255.0
    lab_arr = rgb2lab(rgb_f32)  # (H, W, 3), L in [0, 100]

    # 목표 히스토그램에서 샘플 L* 값 생성 (match_histograms용)
    n_bins = len(target_histogram)
    bin_edges = np.linspace(0.0, 100.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total_pixels = rgb.shape[0] * rgb.shape[1]
    target_counts = (target_histogram * total_pixels).astype(int)
    # 누적합 부족분 보정
    deficit = total_pixels - target_counts.sum()
    if deficit > 0:
        target_counts[target_counts.argmax()] += deficit

    reference_l = np.repeat(bin_centers, target_counts).astype(np.float32)

    l_channel = lab_arr[:, :, 0].astype(np.float32)
    matched_l = match_histograms(l_channel, reference_l).astype(np.float32)

    blended_l = l_channel * (1.0 - strength) + matched_l * strength
    blended_l = np.clip(blended_l, 0.0, 100.0)

    lab_result = lab_arr.copy()
    lab_result[:, :, 0] = blended_l

    rgb_result_f32 = np.clip(lab2rgb(lab_result), 0.0, 1.0)
    rgb_result = (rgb_result_f32 * 255.0).astype(np.uint8)

    rgba_result = np.concatenate([rgb_result, alpha[:, :, None]], axis=-1)
    return Image.fromarray(rgba_result, mode="RGBA")


def enforce_consistency(
    generated_images: list[Image.Image],
    style_stats: StyleStatistics,
    *,
    palette_strength: float = 0.7,
    linewidth_strength: float = 0.5,
    shading_strength: float = 0.5,
) -> list[Image.Image]:
    """N개 생성 이미지에 reference 통계를 강제 적용한다.

    각 단계는 strength=0이면 no-op. 단계 순서는: palette → linewidth → shading.

    Args:
        generated_images: N개 RGBA PIL Image (변환 직후, 후처리 전).
        style_stats: ``extract_style_statistics()`` 반환값.
        palette_strength: palette quantize 강도 [0, 1].
        linewidth_strength: linewidth 보정 강도 [0, 1]. 0이면 보정 없음.
            **주의**: 형태학 연산 기반이므로 0.5 이하 보수적 값 권장.
            정밀 두께 제어는 보장되지 않는다.
        shading_strength: shading 히스토그램 매칭 강도 [0, 1].

    Returns:
        N개 RGBA PIL Image (원본 alpha 보존).

    Raises:
        ValueError: strength 인자가 [0, 1] 밖이거나 ``generated_images`` 가 빈 리스트.
    """
    if not generated_images:
        raise ValueError("generated_images must not be empty.")
    if not 0.0 <= palette_strength <= 1.0:
        raise ValueError(f"palette_strength must be in [0, 1], got {palette_strength}")
    if not 0.0 <= linewidth_strength <= 1.0:
        raise ValueError(f"linewidth_strength must be in [0, 1], got {linewidth_strength}")
    if not 0.0 <= shading_strength <= 1.0:
        raise ValueError(f"shading_strength must be in [0, 1], got {shading_strength}")

    logger.info(
        "enforce_consistency: N=%d, palette=%.2f, linewidth=%.2f, shading=%.2f",
        len(generated_images),
        palette_strength,
        linewidth_strength,
        shading_strength,
    )

    results: list[Image.Image] = []

    for idx, img in enumerate(generated_images):
        # RGBA 보장
        current = img if img.mode == "RGBA" else img.convert("RGBA")

        # 1. palette enforcement
        if palette_strength > 0.0:
            current = quantize_to_palette(current, style_stats.palette, strength=palette_strength)
            logger.debug("[%d/%d] palette applied", idx + 1, len(generated_images))

        # 2. linewidth enforcement
        if linewidth_strength > 0.0:
            current = _apply_linewidth_correction(
                current, style_stats.mean_linewidth, linewidth_strength
            )
            logger.debug("[%d/%d] linewidth applied", idx + 1, len(generated_images))

        # 3. shading enforcement
        if shading_strength > 0.0:
            current = _apply_shading_correction(
                current, style_stats.shading_histogram, shading_strength
            )
            logger.debug("[%d/%d] shading applied", idx + 1, len(generated_images))

        results.append(current)

    logger.info("enforce_consistency complete: %d images processed.", len(results))
    return results
