"""배치 일관성 메트릭 단위 테스트 (C2).

모델 로딩 없음. skimage/scipy는 설치 확인 후 진행.
macOS 환경에서 skimage가 lzma 의존성 오류로 로드 불가능할 수 있어
개별 테스트에 skip 데코레이터를 적용한다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.evaluation.consistency import sigma_linewidth, sigma_palette, sigma_shading

# skimage 사용 가능 여부 선행 확인 (macOS: lzma 미설치 시 ImportError 발생)
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
def solid_red_rgba() -> Image.Image:
    """64x64 단색 빨간 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [255, 0, 0, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def solid_blue_rgba() -> Image.Image:
    """64x64 단색 파란 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [0, 0, 255, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def solid_green_rgba() -> Image.Image:
    """64x64 단색 초록 RGBA 이미지."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [0, 255, 0, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def transparent_rgba() -> Image.Image:
    """64x64 완전 투명 RGBA 이미지 (alpha=0)."""
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


# ---------------------------------------------------------------------------
# sigma_palette — N=1 edge case
# ---------------------------------------------------------------------------


def test_sigma_palette_n1_returns_zero(solid_red_rgba):
    """N=1 이미지 리스트에서 sigma_palette 반환값이 0.0인가."""
    result = sigma_palette([solid_red_rgba])
    assert result == 0.0


def test_sigma_palette_empty_raises():
    """빈 리스트 입력 시 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="non-empty"):
        sigma_palette([])


@skip_if_no_skimage
def test_sigma_palette_identical_images_is_zero(solid_red_rgba):
    """동일 이미지 두 장의 sigma_palette가 0.0인가."""
    result = sigma_palette([solid_red_rgba, solid_red_rgba])
    assert result == pytest.approx(0.0, abs=1e-6)


@skip_if_no_skimage
def test_sigma_palette_deterministic(solid_red_rgba, solid_blue_rgba):
    """같은 입력에서 두 번 호출 결과가 동일한가."""
    imgs = [solid_red_rgba, solid_blue_rgba]
    r1 = sigma_palette(imgs)
    r2 = sigma_palette(imgs)
    assert r1 == r2


@skip_if_no_skimage
def test_sigma_palette_different_colors_larger_than_same(solid_red_rgba, solid_blue_rgba):
    """다른 색 이미지 쌍이 같은 색 이미지 쌍보다 sigma_palette가 큰가 (단조성)."""
    same_pair = [solid_red_rgba, solid_red_rgba]
    diff_pair = [solid_red_rgba, solid_blue_rgba]
    assert sigma_palette(diff_pair) > sigma_palette(same_pair)


@skip_if_no_skimage
def test_sigma_palette_returns_float(solid_red_rgba, solid_blue_rgba):
    """반환값이 float 타입인가."""
    result = sigma_palette([solid_red_rgba, solid_blue_rgba])
    assert isinstance(result, float)


@skip_if_no_skimage
def test_sigma_palette_nonnegative(solid_red_rgba, solid_blue_rgba, solid_green_rgba):
    """sigma_palette 결과가 항상 >= 0인가."""
    result = sigma_palette([solid_red_rgba, solid_blue_rgba, solid_green_rgba])
    assert result >= 0.0


# ---------------------------------------------------------------------------
# sigma_linewidth — N=1 edge case + error
# ---------------------------------------------------------------------------


def test_sigma_linewidth_n1_returns_zero(solid_red_rgba):
    """N=1 이미지 리스트에서 sigma_linewidth 반환값이 0.0인가."""
    result = sigma_linewidth([solid_red_rgba])
    assert result == 0.0


def test_sigma_linewidth_empty_raises():
    """빈 리스트 입력 시 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="non-empty"):
        sigma_linewidth([])


@skip_if_no_skimage
def test_sigma_linewidth_identical_images_is_zero(solid_red_rgba):
    """동일 이미지 두 장의 sigma_linewidth가 0.0인가."""
    result = sigma_linewidth([solid_red_rgba, solid_red_rgba])
    assert result == pytest.approx(0.0, abs=1e-6)


@skip_if_no_skimage
def test_sigma_linewidth_deterministic(solid_red_rgba, solid_blue_rgba):
    """같은 입력에서 두 번 호출 결과가 동일한가."""
    imgs = [solid_red_rgba, solid_blue_rgba]
    r1 = sigma_linewidth(imgs)
    r2 = sigma_linewidth(imgs)
    assert r1 == r2


@skip_if_no_skimage
def test_sigma_linewidth_returns_float(solid_red_rgba, solid_blue_rgba):
    """반환값이 float 타입인가."""
    result = sigma_linewidth([solid_red_rgba, solid_blue_rgba])
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# sigma_shading — 특수 케이스
# ---------------------------------------------------------------------------


def test_sigma_shading_n1_returns_zero(solid_red_rgba):
    """N=1 이미지 리스트에서 sigma_shading 반환값이 0.0인가."""
    result = sigma_shading([solid_red_rgba])
    assert result == 0.0


def test_sigma_shading_empty_raises():
    """빈 리스트 입력 시 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="non-empty"):
        sigma_shading([])


@skip_if_no_skimage
def test_sigma_shading_transparent_image_no_crash(transparent_rgba, solid_red_rgba):
    """alpha=0 완전 투명 이미지가 포함된 경우 예외 없이 처리되는가."""
    result = sigma_shading([transparent_rgba, solid_red_rgba])
    assert isinstance(result, float)
    assert result >= 0.0


@skip_if_no_skimage
def test_sigma_shading_identical_images_is_zero(solid_red_rgba):
    """동일 이미지 두 장의 sigma_shading이 0.0인가."""
    result = sigma_shading([solid_red_rgba, solid_red_rgba])
    assert result == pytest.approx(0.0, abs=1e-4)


@skip_if_no_skimage
def test_sigma_shading_returns_float(solid_red_rgba, solid_blue_rgba):
    """반환값이 float 타입인가."""
    result = sigma_shading([solid_red_rgba, solid_blue_rgba])
    assert isinstance(result, float)


@skip_if_no_skimage
def test_sigma_shading_nonnegative(solid_red_rgba, solid_blue_rgba):
    """sigma_shading 결과가 항상 >= 0인가."""
    result = sigma_shading([solid_red_rgba, solid_blue_rgba])
    assert result >= 0.0
