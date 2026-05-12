"""region_mask.py 단위 테스트 (D3).

모델 로딩 없음. 순수 numpy/PIL 로직을 검증한다.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.postprocessing.region_mask import apply_region_mask, auto_mask

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def src_64() -> Image.Image:
    """64x64 빨간 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [255, 0, 0, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def tgt_64() -> Image.Image:
    """64x64 파란 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [0, 0, 255, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def all_ones_mask() -> np.ndarray:
    """64x64 전체 1.0 마스크."""
    return np.ones((64, 64), dtype=np.float32)


@pytest.fixture
def all_zeros_mask() -> np.ndarray:
    """64x64 전체 0.0 마스크."""
    return np.zeros((64, 64), dtype=np.float32)


@pytest.fixture
def half_mask() -> np.ndarray:
    """64x64 전체 0.5 마스크."""
    return np.full((64, 64), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# apply_region_mask — 기본 검증
# ---------------------------------------------------------------------------


def test_apply_region_mask_size_mismatch_raises(src_64, tgt_64):
    """source와 transformed 크기 불일치 시 ValueError가 발생하는가."""
    tgt_32 = Image.new("RGBA", (32, 32), (0, 0, 255, 255))
    mask = np.ones((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="size"):
        apply_region_mask(src_64, tgt_32, mask)


def test_apply_region_mask_shape_mismatch_raises(src_64, tgt_64):
    """mask shape이 이미지 크기와 다를 때 ValueError가 발생하는가."""
    mask_bad = np.ones((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        apply_region_mask(src_64, tgt_64, mask_bad)


def test_apply_region_mask_negative_feather_raises(src_64, tgt_64, all_ones_mask):
    """feather_px < 0이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="feather_px"):
        apply_region_mask(src_64, tgt_64, all_ones_mask, feather_px=-1)


def test_apply_region_mask_returns_rgba(src_64, tgt_64, all_ones_mask):
    """반환 이미지가 RGBA 모드인가."""
    result = apply_region_mask(src_64, tgt_64, all_ones_mask, feather_px=0)
    assert result.mode == "RGBA"


def test_apply_region_mask_all_ones_equals_transformed(src_64, tgt_64, all_ones_mask):
    """mask=all 1.0이면 출력이 transformed와 동일한가."""
    result = apply_region_mask(src_64, tgt_64, all_ones_mask, feather_px=0)
    result_arr = np.array(result)
    tgt_arr = np.array(tgt_64)
    np.testing.assert_array_equal(result_arr, tgt_arr)


def test_apply_region_mask_all_zeros_equals_source(src_64, tgt_64, all_zeros_mask):
    """mask=all 0.0이면 출력이 source와 동일한가."""
    result = apply_region_mask(src_64, tgt_64, all_zeros_mask, feather_px=0)
    result_arr = np.array(result)
    src_arr = np.array(src_64.convert("RGBA"))
    np.testing.assert_array_equal(result_arr, src_arr)


def test_apply_region_mask_half_mask_midpoint(src_64, tgt_64, half_mask):
    """mask=0.5이면 출력 픽셀이 source와 transformed의 중간값인가."""
    result = apply_region_mask(src_64, tgt_64, half_mask, feather_px=0)
    result_arr = np.array(result).astype(np.float32)
    src_arr = np.array(src_64.convert("RGBA")).astype(np.float32)
    tgt_arr = np.array(tgt_64.convert("RGBA")).astype(np.float32)
    expected = (src_arr * 0.5 + tgt_arr * 0.5).astype(np.uint8)
    np.testing.assert_array_almost_equal(result_arr, expected.astype(np.float32), decimal=0)


def test_apply_region_mask_feather_zero_no_gaussian(src_64, tgt_64, all_ones_mask):
    """feather_px=0이면 gaussian_filter가 호출되지 않는가."""
    with patch("scipy.ndimage.gaussian_filter") as mock_gf:
        apply_region_mask(src_64, tgt_64, all_ones_mask, feather_px=0)
    mock_gf.assert_not_called()


# ---------------------------------------------------------------------------
# auto_mask — 모드별 검증
# ---------------------------------------------------------------------------


def test_auto_mask_none_returns_all_ones(src_64):
    """mode='none'이 전체 1.0 마스크를 반환하는가."""
    mask = auto_mask(src_64, mode="none")
    assert mask.shape == (64, 64)
    assert float(mask.min()) == pytest.approx(1.0)
    assert float(mask.max()) == pytest.approx(1.0)


def test_auto_mask_face_preserve_returns_all_ones(src_64):
    """mode='face_preserve'가 (현재) 전체 1.0 마스크를 반환하는가."""
    mask = auto_mask(src_64, mode="face_preserve")
    assert float(mask.min()) == pytest.approx(1.0)
    assert float(mask.max()) == pytest.approx(1.0)


def test_auto_mask_focal_center_is_one(src_64):
    """mode='focal'에서 중앙 영역이 1.0인가."""
    mask = auto_mask(src_64, mode="focal")
    # 중앙 픽셀 (32, 32)는 1.0이어야 함
    assert mask[32, 32] == pytest.approx(1.0)


def test_auto_mask_focal_corner_is_zero(src_64):
    """mode='focal'에서 모서리 픽셀이 0.0인가."""
    mask = auto_mask(src_64, mode="focal")
    assert mask[0, 0] == pytest.approx(0.0)


def test_auto_mask_focal_shape_correct(src_64):
    """mode='focal' 마스크 shape가 (H, W)인가."""
    mask = auto_mask(src_64, mode="focal")
    assert mask.shape == (64, 64)


def test_auto_mask_unknown_mode_raises(src_64):
    """알 수 없는 mode에서 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="mode"):
        auto_mask(src_64, mode="invalid_mode")


def test_auto_mask_face_preserve_returns_float32(src_64):
    """반환 마스크의 dtype이 float32인가."""
    mask = auto_mask(src_64, mode="face_preserve")
    assert mask.dtype == np.float32
