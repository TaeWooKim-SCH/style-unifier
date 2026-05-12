"""batch_consistency.py 단위 테스트 (§5.5.1).

extract_style_statistics는 cache_embedding=False 경로에서만 테스트.
모델 로딩 없이 전처리 + 통계 계산 로직만 검증한다.
extract_lineart는 mock으로 우회한다.
skimage 의존 테스트는 macOS 환경에서 skip.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.encoding.batch_consistency import (
    StyleStatistics,
    enforce_consistency,
    extract_style_statistics,
)

# skimage 사용 가능 여부 확인 (macOS: lzma 미설치 시 ImportError)
try:
    import skimage  # noqa: F401

    _SKIMAGE_AVAILABLE = True
except (ImportError, Exception):
    _SKIMAGE_AVAILABLE = False

skip_if_no_skimage = pytest.mark.skipif(
    not _SKIMAGE_AVAILABLE,
    reason="skimage not available (lzma dependency missing on macOS)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_rgba() -> Image.Image:
    """64x64 빨간 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [200, 50, 50, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def another_rgba() -> Image.Image:
    """64x64 파란 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [50, 50, 200, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def mock_lineart_image() -> Image.Image:
    """Canny edge 출력을 대체할 흰 선 / 검은 배경 이미지."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[20:40, 20:40] = 255  # 일부 흰 픽셀
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# StyleStatistics dataclass
# ---------------------------------------------------------------------------


def test_style_statistics_is_frozen():
    """StyleStatistics 인스턴스가 frozen dataclass인가 (불변성 검증)."""
    stats = StyleStatistics(
        palette=np.zeros((4, 3), dtype=np.uint8),
        palette_weights=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        mean_linewidth=1.0,
        linewidth_distribution=np.zeros(8, dtype=np.float32),
        shading_histogram=np.zeros(8, dtype=np.float32),
        ip_adapter_embedding=None,
    )
    with pytest.raises((AttributeError, TypeError)):
        stats.mean_linewidth = 99.0  # type: ignore[misc]


def test_style_statistics_ip_adapter_embedding_none():
    """ip_adapter_embedding이 None으로 설정될 수 있는가."""
    stats = StyleStatistics(
        palette=np.zeros((4, 3), dtype=np.uint8),
        palette_weights=np.ones(4, dtype=np.float32) / 4,
        mean_linewidth=0.0,
        linewidth_distribution=np.zeros(8, dtype=np.float32),
        shading_histogram=np.zeros(8, dtype=np.float32),
        ip_adapter_embedding=None,
    )
    assert stats.ip_adapter_embedding is None


def test_style_statistics_ip_adapter_embedding_tensor():
    """ip_adapter_embedding이 torch.Tensor를 수용하는가."""
    emb = torch.zeros(1, 257, 1280)
    stats = StyleStatistics(
        palette=np.zeros((4, 3), dtype=np.uint8),
        palette_weights=np.ones(4, dtype=np.float32) / 4,
        mean_linewidth=0.0,
        linewidth_distribution=np.zeros(8, dtype=np.float32),
        shading_histogram=np.zeros(8, dtype=np.float32),
        ip_adapter_embedding=emb,
    )
    assert stats.ip_adapter_embedding is emb


# ---------------------------------------------------------------------------
# extract_style_statistics — cache_embedding=False 경로
# ---------------------------------------------------------------------------


def test_extract_style_statistics_palette_k_lt1_raises(small_rgba):
    """palette_k < 1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="palette_k"):
        extract_style_statistics(small_rgba, palette_k=0, cache_embedding=False)


def test_extract_style_statistics_n_bins_lt1_raises(small_rgba):
    """n_bins < 1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="n_bins"):
        extract_style_statistics(small_rgba, n_bins=0, cache_embedding=False)


@skip_if_no_skimage
def test_extract_style_statistics_returns_style_statistics(small_rgba, mock_lineart_image):
    """cache_embedding=False 경로에서 StyleStatistics를 반환하는가."""
    with patch("src.encoding.batch_consistency.extract_lineart", return_value=mock_lineart_image):
        stats = extract_style_statistics(small_rgba, palette_k=4, n_bins=8, cache_embedding=False)
    assert isinstance(stats, StyleStatistics)


@skip_if_no_skimage
def test_extract_style_statistics_palette_shape(small_rgba, mock_lineart_image):
    """반환된 palette의 shape가 (palette_k, 3)인가."""
    with patch("src.encoding.batch_consistency.extract_lineart", return_value=mock_lineart_image):
        stats = extract_style_statistics(small_rgba, palette_k=4, n_bins=8, cache_embedding=False)
    assert stats.palette.shape == (4, 3)


@skip_if_no_skimage
def test_extract_style_statistics_embedding_is_none_when_not_cached(small_rgba, mock_lineart_image):
    """cache_embedding=False이면 ip_adapter_embedding이 None인가."""
    with patch("src.encoding.batch_consistency.extract_lineart", return_value=mock_lineart_image):
        stats = extract_style_statistics(small_rgba, cache_embedding=False)
    assert stats.ip_adapter_embedding is None


@skip_if_no_skimage
def test_extract_style_statistics_shading_histogram_shape(small_rgba, mock_lineart_image):
    """shading_histogram의 shape가 (n_bins,)인가."""
    with patch("src.encoding.batch_consistency.extract_lineart", return_value=mock_lineart_image):
        stats = extract_style_statistics(small_rgba, n_bins=16, cache_embedding=False)
    assert stats.shading_histogram.shape == (16,)


# ---------------------------------------------------------------------------
# enforce_consistency — 인자 검증 + no-op 검증
# ---------------------------------------------------------------------------


def test_enforce_consistency_empty_raises():
    """generated_images가 빈 리스트이면 ValueError가 발생하는가."""
    dummy_stats = StyleStatistics(
        palette=np.zeros((4, 3), dtype=np.uint8),
        palette_weights=np.ones(4, dtype=np.float32) / 4,
        mean_linewidth=0.0,
        linewidth_distribution=np.zeros(8, dtype=np.float32),
        shading_histogram=np.zeros(8, dtype=np.float32),
        ip_adapter_embedding=None,
    )
    with pytest.raises(ValueError, match="empty"):
        enforce_consistency([], dummy_stats)


def _make_dummy_stats(k: int = 4, n_bins: int = 8) -> StyleStatistics:
    """테스트용 더미 StyleStatistics (skimage 미사용)."""
    return StyleStatistics(
        palette=np.zeros((k, 3), dtype=np.uint8),
        palette_weights=np.ones(k, dtype=np.float32) / k,
        mean_linewidth=0.0,
        linewidth_distribution=np.zeros(n_bins, dtype=np.float32),
        shading_histogram=np.zeros(n_bins, dtype=np.float32),
        ip_adapter_embedding=None,
    )


def test_enforce_consistency_invalid_palette_strength_raises(small_rgba):
    """palette_strength > 1이면 ValueError가 발생하는가."""
    stats = _make_dummy_stats()
    with pytest.raises(ValueError, match="palette_strength"):
        enforce_consistency([small_rgba], stats, palette_strength=1.5)


def test_enforce_consistency_invalid_linewidth_strength_raises(small_rgba):
    """linewidth_strength < 0이면 ValueError가 발생하는가."""
    stats = _make_dummy_stats()
    with pytest.raises(ValueError, match="linewidth_strength"):
        enforce_consistency([small_rgba], stats, linewidth_strength=-0.1)


def test_enforce_consistency_all_strength_zero_is_noop(small_rgba):
    """모든 strength=0이면 출력이 입력과 동일한가 (no-op)."""
    stats = _make_dummy_stats()
    results = enforce_consistency(
        [small_rgba],
        stats,
        palette_strength=0.0,
        linewidth_strength=0.0,
        shading_strength=0.0,
    )

    assert len(results) == 1
    result_arr = np.array(results[0].convert("RGBA"))
    source_arr = np.array(small_rgba.convert("RGBA"))
    np.testing.assert_array_equal(result_arr, source_arr)


def test_enforce_consistency_returns_rgba_list(small_rgba, another_rgba):
    """N개 입력에 대해 N개 RGBA 이미지를 반환하는가."""
    stats = _make_dummy_stats()
    results = enforce_consistency(
        [small_rgba, another_rgba],
        stats,
        palette_strength=0.0,
        linewidth_strength=0.0,
        shading_strength=0.0,
    )

    assert len(results) == 2
    for img in results:
        assert img.mode == "RGBA"
