"""quantize_to_palette() 단위 테스트.

모델 로딩 없이 순수 numpy/PIL 로직만 검증한다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.postprocessing.palette_quantize import quantize_to_palette

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_color_palette() -> np.ndarray:
    """두 가지 색상(빨강, 파랑)으로 구성된 (2, 3) uint8 팔레트."""
    return np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)


@pytest.fixture
def rgb_image_32() -> Image.Image:
    """32x32 단색(중간 회색) RGB 이미지."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image_32() -> Image.Image:
    """32x32 RGBA 이미지. alpha 채널은 고정값 200."""
    arr = np.full((32, 32, 4), 128, dtype=np.uint8)
    arr[:, :, 3] = 200
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def two_region_rgb() -> Image.Image:
    """왼쪽: 순수 빨강, 오른쪽: 순수 파랑인 32x32 RGB 이미지."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :16, :] = [255, 0, 0]  # 빨강
    arr[:, 16:, :] = [0, 0, 255]  # 파랑
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# strength=0: 원본 보존
# ---------------------------------------------------------------------------


def test_quantize_strength_zero_preserves_rgb(rgb_image_32, two_color_palette):
    """strength=0이면 출력 RGB가 입력 RGB와 동일한가."""
    result = quantize_to_palette(rgb_image_32, two_color_palette, strength=0.0)
    in_arr = np.array(rgb_image_32)
    out_arr = np.array(result)
    assert np.array_equal(in_arr, out_arr)


# ---------------------------------------------------------------------------
# strength=1: 완전 양자화 — 각 픽셀이 팔레트 색상 중 하나와 정확히 일치
# ---------------------------------------------------------------------------


def test_quantize_strength_one_each_pixel_matches_palette(two_region_rgb, two_color_palette):
    """strength=1이면 각 픽셀 RGB가 팔레트 색상 중 하나와 정확히 일치하는가."""
    result = quantize_to_palette(two_region_rgb, two_color_palette, strength=1.0)
    out_arr = np.array(result)[:, :, :3]  # RGB만
    palette_f = two_color_palette.tolist()

    for row in out_arr:
        for pixel in row:
            pixel_list = pixel.tolist()
            assert pixel_list in palette_f, f"Pixel {pixel_list} not in palette {palette_f}"


# ---------------------------------------------------------------------------
# strength 범위 검증 → ValueError
# ---------------------------------------------------------------------------


def test_quantize_raises_on_strength_below_zero(rgb_image_32, two_color_palette):
    """strength=-0.1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="strength must be in"):
        quantize_to_palette(rgb_image_32, two_color_palette, strength=-0.1)


def test_quantize_raises_on_strength_above_one(rgb_image_32, two_color_palette):
    """strength=1.1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="strength must be in"):
        quantize_to_palette(rgb_image_32, two_color_palette, strength=1.1)


# ---------------------------------------------------------------------------
# palette shape 검증 → ValueError
# ---------------------------------------------------------------------------


def test_quantize_raises_on_palette_shape_k4(rgb_image_32):
    """palette shape이 (k, 4)면 ValueError가 발생하는가."""
    bad_palette = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="palette must have shape"):
        quantize_to_palette(rgb_image_32, bad_palette)


def test_quantize_raises_on_palette_1d(rgb_image_32):
    """palette가 1D 배열이면 ValueError가 발생하는가."""
    bad_palette = np.zeros((12,), dtype=np.uint8)
    with pytest.raises(ValueError, match="palette must have shape"):
        quantize_to_palette(rgb_image_32, bad_palette)


# ---------------------------------------------------------------------------
# RGBA alpha 보존
# ---------------------------------------------------------------------------


def test_quantize_rgba_preserves_alpha_channel(rgba_image_32, two_color_palette):
    """RGBA 입력의 alpha 채널이 출력에서 그대로 보존되는가."""
    result = quantize_to_palette(rgba_image_32, two_color_palette, strength=0.5)
    assert result.mode == "RGBA"

    in_alpha = np.array(rgba_image_32)[:, :, 3]
    out_alpha = np.array(result)[:, :, 3]
    assert np.array_equal(in_alpha, out_alpha)


def test_quantize_rgba_output_mode_is_rgba(rgba_image_32, two_color_palette):
    """RGBA 입력의 출력 mode가 RGBA인가."""
    result = quantize_to_palette(rgba_image_32, two_color_palette, strength=0.5)
    assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# RGB 입력 → RGB 출력 (mode 유지)
# ---------------------------------------------------------------------------


def test_quantize_rgb_input_output_mode_is_rgb(rgb_image_32, two_color_palette):
    """RGB 입력의 출력 mode가 RGB인가."""
    result = quantize_to_palette(rgb_image_32, two_color_palette, strength=0.5)
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# 보간 방향성 — 중간값은 원본과 팔레트 사이에 있어야 함
# ---------------------------------------------------------------------------


def test_quantize_strength_half_blends_toward_palette(rgb_image_32, two_color_palette):
    """strength=0.5이면 출력 픽셀이 원본과 가장 가까운 팔레트 색 사이에 있는가.

    입력: 균일 회색(128,128,128). 팔레트: 빨강(255,0,0), 파랑(0,0,255).
    회색에서 빨강/파랑 모두 거리가 비슷하므로 어느 쪽으로든 이동되어야 한다.
    출력이 순수 회색(128,128,128)이 아니어야 한다.
    """
    result = quantize_to_palette(rgb_image_32, two_color_palette, strength=0.5)
    out_arr = np.array(result)[:, :, :3]
    # 적어도 한 채널이 128이 아니어야 blending이 적용된 것
    assert not np.all(out_arr == 128)
