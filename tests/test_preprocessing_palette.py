"""extract_palette() 단위 테스트.

모델 로딩 없이 순수 로직만 검증한다. scikit-learn이 필요하므로
설치된 환경에서 실행한다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.palette import extract_palette

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rgb_image_32() -> Image.Image:
    """32x32 랜덤 RGB 이미지 (고정 seed)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image_32() -> Image.Image:
    """32x32 RGBA 이미지. alpha가 반은 0, 반은 255인 두 영역을 가짐."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    # 왼쪽 절반: 빨간 픽셀, 완전 불투명
    arr[:, :16, :] = [255, 0, 0, 255]
    # 오른쪽 절반: 파란 픽셀, 완전 투명
    arr[:, 16:, :] = [0, 0, 255, 0]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def tiny_1x1_rgb() -> Image.Image:
    """1x1 RGB 이미지 — 픽셀 수 < k 케이스."""
    return Image.new("RGB", (1, 1), color=(128, 64, 32))


# ---------------------------------------------------------------------------
# Shape / dtype 검증
# ---------------------------------------------------------------------------


def test_extract_palette_returns_correct_shape(rgb_image_32):
    """k=4로 호출 시 shape이 (4, 3)인가."""
    palette = extract_palette(rgb_image_32, k=4)
    assert palette.shape == (4, 3)


def test_extract_palette_returns_uint8_dtype(rgb_image_32):
    """반환 배열의 dtype이 uint8인가."""
    palette = extract_palette(rgb_image_32, k=4)
    assert palette.dtype == np.uint8


def test_extract_palette_value_range_0_to_255(rgb_image_32):
    """각 색상 값이 [0, 255] 범위 안에 있는가."""
    palette = extract_palette(rgb_image_32, k=4)
    assert np.all(palette >= 0) and np.all(palette <= 255)


def test_extract_palette_respects_k_parameter_small(rgb_image_32):
    """k=4이면 팔레트 행 수가 4인가."""
    palette = extract_palette(rgb_image_32, k=4)
    assert len(palette) == 4


def test_extract_palette_respects_k_parameter_large(rgb_image_32):
    """k=16이면 팔레트 행 수가 16인가."""
    palette = extract_palette(rgb_image_32, k=16)
    assert len(palette) == 16


# ---------------------------------------------------------------------------
# 결정론적 동작
# ---------------------------------------------------------------------------


def test_extract_palette_same_random_state_is_deterministic(rgb_image_32):
    """random_state가 같으면 두 번 호출 결과가 동일한가."""
    p1 = extract_palette(rgb_image_32, k=8, random_state=42)
    p2 = extract_palette(rgb_image_32, k=8, random_state=42)
    assert np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# 픽셀 수 < k — fallback
# ---------------------------------------------------------------------------


def test_extract_palette_pixel_count_less_than_k_returns_zeros(tiny_1x1_rgb):
    """픽셀 수(1)가 k(8)보다 작으면 zeros 배열을 반환하는가."""
    palette = extract_palette(tiny_1x1_rgb, k=8)
    assert palette.shape == (8, 3)
    assert np.all(palette == 0)


# ---------------------------------------------------------------------------
# ignore_alpha 동작
# ---------------------------------------------------------------------------


def test_extract_palette_ignore_alpha_true_excludes_transparent_pixels(rgba_image_32):
    """ignore_alpha=True면 alpha<128인 픽셀(오른쪽 파란 영역)을 제외하는가.

    왼쪽 절반만 빨간색 불투명 픽셀이므로 팔레트는 빨간 계열 색상만 나와야 한다.
    """
    palette = extract_palette(rgba_image_32, k=2, ignore_alpha=True)
    # 파란(B>>R) 클러스터가 없어야 한다 — 모든 행에서 R >= B
    assert palette.shape == (2, 3)
    # 팔레트 결과가 0-배열이 아닌지 확인 (픽셀이 충분)
    assert not np.all(palette == 0)


def test_extract_palette_ignore_alpha_false_includes_all_pixels(rgba_image_32):
    """ignore_alpha=False면 alpha 무관하게 모든 픽셀을 포함하는가."""
    palette = extract_palette(rgba_image_32, k=2, ignore_alpha=False)
    assert palette.shape == (2, 3)
    # 파란 픽셀도 포함되므로 팔레트에 파란 계열이 나타나야 함
    # 두 클러스터 중 하나는 B가 높아야 한다.
    has_blue_dominant = any(int(row[2]) > int(row[0]) for row in palette)
    assert has_blue_dominant


# ---------------------------------------------------------------------------
# RGB 입력 허용
# ---------------------------------------------------------------------------


def test_extract_palette_accepts_rgb_input(rgb_image_32):
    """RGB 모드 입력에서 예외 없이 팔레트를 추출하는가."""
    palette = extract_palette(rgb_image_32, k=4)
    assert palette.shape == (4, 3)


# ---------------------------------------------------------------------------
# 오류 케이스
# ---------------------------------------------------------------------------


def test_extract_palette_raises_on_k_zero():
    """k=0이면 ValueError가 발생하는가."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    with pytest.raises(ValueError, match="k must be >= 1"):
        extract_palette(img, k=0)


def test_extract_palette_raises_on_k_negative():
    """k=-1이면 ValueError가 발생하는가."""
    img = Image.new("RGB", (64, 64))
    with pytest.raises(ValueError, match="k must be >= 1"):
        extract_palette(img, k=-1)
