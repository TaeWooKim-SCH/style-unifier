"""restore_alpha() 단위 테스트.

scipy.ndimage.gaussian_filter 호출 여부를 mock으로 검증하고,
shape mismatch 에러와 feather_px=0 분기를 확인한다.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.postprocessing.alpha_restore import restore_alpha

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rgb_image_32() -> Image.Image:
    """32x32 RGB 이미지."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def rgba_image_32() -> Image.Image:
    """32x32 RGBA 이미지."""
    arr = np.full((32, 32, 4), 200, dtype=np.uint8)
    arr[:, :, 3] = 180
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def mask_32() -> np.ndarray:
    """32x32 float32 mask, 값 0.5 균일."""
    return np.full((32, 32), 0.5, dtype=np.float32)


@pytest.fixture
def mask_16() -> np.ndarray:
    """16x16 float32 mask (size mismatch 테스트용)."""
    return np.full((16, 16), 0.8, dtype=np.float32)


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------


def test_restore_alpha_returns_rgba_image(rgb_image_32, mask_32):
    """정상 입력 시 반환값이 RGBA PIL Image인가."""
    result = restore_alpha(rgb_image_32, mask_32, feather_px=0)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"


def test_restore_alpha_output_size_matches_input(rgb_image_32, mask_32):
    """출력 이미지 크기가 입력 이미지 크기와 동일한가."""
    result = restore_alpha(rgb_image_32, mask_32, feather_px=0)
    assert result.size == rgb_image_32.size


def test_restore_alpha_alpha_channel_reflects_mask_when_no_feather(rgb_image_32, mask_32):
    """feather_px=0일 때 alpha 채널이 (mask * 255)와 근사하게 일치하는가."""
    result = restore_alpha(rgb_image_32, mask_32, feather_px=0)
    alpha_arr = np.array(result)[:, :, 3]
    expected_alpha = (mask_32 * 255).astype(np.uint8)
    # float → uint8 변환 오차 범위 1 이내
    assert np.allclose(alpha_arr.astype(float), expected_alpha.astype(float), atol=1.0)


def test_restore_alpha_rgba_input_alpha_is_replaced_by_mask(rgba_image_32, mask_32):
    """RGBA 입력의 원본 alpha는 무시되고 mask로 재구성되는가."""
    original_alpha_val = 180  # rgba_image_32의 alpha
    result = restore_alpha(rgba_image_32, mask_32, feather_px=0)
    result_alpha = np.array(result)[:, :, 3]
    # 원본 alpha(180)가 아닌 mask 기반 alpha(127~128)여야 함
    expected = (mask_32 * 255).astype(np.uint8)
    assert np.allclose(result_alpha.astype(float), expected.astype(float), atol=1.0)
    # 원본 alpha와 같지 않아야 함
    assert not np.all(result_alpha == original_alpha_val)


# ---------------------------------------------------------------------------
# shape mismatch → ValueError
# ---------------------------------------------------------------------------


def test_restore_alpha_raises_on_mask_shape_mismatch(rgb_image_32, mask_16):
    """mask가 (16,16)이고 이미지가 32x32면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="original_mask shape"):
        restore_alpha(rgb_image_32, mask_16, feather_px=0)


def test_restore_alpha_raises_on_3d_mask(rgb_image_32):
    """mask가 3D 배열이면 ValueError가 발생하는가."""
    bad_mask = np.ones((32, 32, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        restore_alpha(rgb_image_32, bad_mask, feather_px=0)


# ---------------------------------------------------------------------------
# feather_px=0 → gaussian_filter 호출 안 됨
# ---------------------------------------------------------------------------


def test_restore_alpha_no_feathering_when_feather_px_zero(rgb_image_32, mask_32):
    """feather_px=0이면 scipy.ndimage.gaussian_filter가 호출되지 않는가.

    gaussian_filter는 _apply_feathering 내부에서 lazy import되므로
    scipy.ndimage 모듈을 패치 대상으로 사용한다.
    """
    with patch("scipy.ndimage.gaussian_filter") as mock_filter:
        restore_alpha(rgb_image_32, mask_32, feather_px=0)
        mock_filter.assert_not_called()


# ---------------------------------------------------------------------------
# feather_px=1 → gaussian_filter 호출됨
# ---------------------------------------------------------------------------


def test_restore_alpha_feathering_calls_gaussian_filter_when_feather_px_positive(
    rgb_image_32, mask_32
):
    """feather_px=1이면 scipy.ndimage.gaussian_filter가 호출되는가."""
    sentinel = np.full((32, 32), 0.5, dtype=np.float32)

    with patch("scipy.ndimage.gaussian_filter", return_value=sentinel) as mock_filter:
        restore_alpha(rgb_image_32, mask_32, feather_px=1)
        mock_filter.assert_called_once()


def test_restore_alpha_gaussian_filter_called_with_correct_sigma(rgb_image_32, mask_32):
    """gaussian_filter가 sigma=feather_px 인자로 호출되는가."""
    sentinel = np.full((32, 32), 0.5, dtype=np.float32)

    with patch("scipy.ndimage.gaussian_filter", return_value=sentinel) as mock_filter:
        restore_alpha(rgb_image_32, mask_32, feather_px=3)

    called_kwargs = mock_filter.call_args.kwargs
    called_args = mock_filter.call_args.args
    # sigma는 args[1] 또는 kwargs["sigma"]로 전달될 수 있음
    sigma_value = called_kwargs.get("sigma") or (called_args[1] if len(called_args) > 1 else None)
    assert sigma_value == 3


# ---------------------------------------------------------------------------
# feather_px 음수 → ValueError
# ---------------------------------------------------------------------------


def test_restore_alpha_raises_on_negative_feather_px(rgb_image_32, mask_32):
    """feather_px=-1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="feather_px must be >= 0"):
        restore_alpha(rgb_image_32, mask_32, feather_px=-1)
