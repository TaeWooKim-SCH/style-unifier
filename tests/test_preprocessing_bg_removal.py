"""remove_background() 단위 테스트.

모델 로딩을 mock으로 우회하고, fast path / slow path / force_remove 분기를 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.bg_removal import _RMBG_CACHE, remove_background

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_rmbg_cache():
    """각 테스트 전후로 _RMBG_CACHE를 비워 캐시 누수를 방지한다."""
    _RMBG_CACHE.clear()
    yield
    _RMBG_CACHE.clear()


@pytest.fixture
def rgba_with_transparency() -> Image.Image:
    """일부 픽셀이 alpha=0인 32x32 RGBA 이미지 — fast path를 유발."""
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:16, :, :] = [200, 100, 50, 255]  # 위쪽 절반: 불투명
    arr[16:, :, :] = [0, 0, 0, 0]  # 아래쪽 절반: 완전 투명
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def rgba_fully_opaque() -> Image.Image:
    """alpha가 모두 255인 32x32 RGBA 이미지 — 모델 경로를 유발."""
    arr = np.full((32, 32, 4), 255, dtype=np.uint8)
    arr[:, :, :3] = [180, 120, 60]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def rgb_image() -> Image.Image:
    """32x32 RGB 이미지 — alpha 채널 없음, 항상 모델 경로."""
    arr = np.full((32, 32, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _make_mock_model_and_transform(h: int = 32, w: int = 32):
    """모델과 transform을 흉내내는 mock 쌍을 반환한다.

    모델 호출 시 (H, W) 형태의 all-ones float32 텐서를 반환해
    mask 값이 [0,1]에 들어오도록 보장한다.
    """
    import torch

    mock_transform = MagicMock()
    # transform(image) → (1, 3, H, W) float 텐서 흉내
    dummy_tensor = torch.zeros(1, 3, h, w)
    mock_transform.return_value = dummy_tensor

    mock_model = MagicMock()
    # model(input_tensor) → (1, 1, H, W) 형태로 반환
    dummy_pred = torch.ones(1, 1, h, w)
    mock_model.return_value = dummy_pred

    return mock_model, mock_transform


# ---------------------------------------------------------------------------
# Fast path 테스트 (투명 픽셀 있는 RGBA)
# ---------------------------------------------------------------------------


def test_remove_background_fast_path_skips_model_when_transparent(rgba_with_transparency):
    """투명 픽셀이 있는 RGBA 입력이면 _load_rmbg_model을 호출하지 않는가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model") as mock_load:
        foreground, mask = remove_background(rgba_with_transparency)
        mock_load.assert_not_called()

    assert foreground is not None
    assert mask is not None


def test_remove_background_fast_path_returns_rgba_image(rgba_with_transparency):
    """Fast path 반환 foreground가 RGBA PIL Image인가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        foreground, _ = remove_background(rgba_with_transparency)

    assert isinstance(foreground, Image.Image)
    assert foreground.mode == "RGBA"


def test_remove_background_fast_path_returns_correct_size(rgba_with_transparency):
    """Fast path 반환 foreground의 크기가 입력과 동일한가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        foreground, _ = remove_background(rgba_with_transparency)

    assert foreground.size == rgba_with_transparency.size


def test_remove_background_fast_path_mask_shape(rgba_with_transparency):
    """Fast path 반환 mask의 shape이 (H, W)인가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        _, mask = remove_background(rgba_with_transparency)

    h, w = rgba_with_transparency.size[1], rgba_with_transparency.size[0]
    assert mask.shape == (h, w)


def test_remove_background_fast_path_mask_dtype(rgba_with_transparency):
    """Fast path 반환 mask의 dtype이 float32인가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        _, mask = remove_background(rgba_with_transparency)

    assert mask.dtype == np.float32


def test_remove_background_fast_path_mask_range(rgba_with_transparency):
    """Fast path 반환 mask의 값이 [0, 1] 범위인가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        _, mask = remove_background(rgba_with_transparency)

    assert mask.min() >= 0.0
    assert mask.max() <= 1.0


def test_remove_background_fast_path_does_not_touch_cache(rgba_with_transparency):
    """Fast path에서 _RMBG_CACHE가 건드려지지 않는가."""
    with patch("src.preprocessing.bg_removal._load_rmbg_model"):
        remove_background(rgba_with_transparency)

    assert len(_RMBG_CACHE) == 0


# ---------------------------------------------------------------------------
# Slow path 테스트 (완전 불투명 RGBA → 모델 호출)
# ---------------------------------------------------------------------------


def test_remove_background_fully_opaque_rgba_calls_model(rgba_fully_opaque):
    """alpha가 전부 255인 RGBA 입력이면 모델이 호출되는가."""
    mock_model, mock_transform = _make_mock_model_and_transform(32, 32)

    with patch(
        "src.preprocessing.bg_removal._load_rmbg_model",
        return_value=(mock_model, mock_transform),
    ):
        foreground, mask = remove_background(rgba_fully_opaque)
        mock_model.assert_called_once()

    assert foreground.mode == "RGBA"
    assert mask.shape == (32, 32)


def test_remove_background_rgb_input_calls_model(rgb_image):
    """RGB 입력이면 모델이 호출되는가."""
    mock_model, mock_transform = _make_mock_model_and_transform(32, 32)

    with patch(
        "src.preprocessing.bg_removal._load_rmbg_model",
        return_value=(mock_model, mock_transform),
    ):
        foreground, mask = remove_background(rgb_image)
        mock_model.assert_called_once()

    assert isinstance(foreground, Image.Image)
    assert foreground.mode == "RGBA"


def test_remove_background_rgb_input_mask_shape(rgb_image):
    """RGB 입력의 반환 mask shape이 (H, W)인가."""
    mock_model, mock_transform = _make_mock_model_and_transform(32, 32)

    with patch(
        "src.preprocessing.bg_removal._load_rmbg_model",
        return_value=(mock_model, mock_transform),
    ):
        _, mask = remove_background(rgb_image)

    assert mask.shape == (32, 32)


def test_remove_background_rgb_input_mask_dtype(rgb_image):
    """RGB 입력의 반환 mask dtype이 float32인가."""
    mock_model, mock_transform = _make_mock_model_and_transform(32, 32)

    with patch(
        "src.preprocessing.bg_removal._load_rmbg_model",
        return_value=(mock_model, mock_transform),
    ):
        _, mask = remove_background(rgb_image)

    assert mask.dtype == np.float32


# ---------------------------------------------------------------------------
# force_remove=True 테스트
# ---------------------------------------------------------------------------


def test_remove_background_force_remove_true_calls_model_even_with_transparent_rgba(
    rgba_with_transparency,
):
    """force_remove=True면 투명 픽셀 있는 RGBA라도 모델을 호출하는가."""
    mock_model, mock_transform = _make_mock_model_and_transform(32, 32)

    with patch(
        "src.preprocessing.bg_removal._load_rmbg_model",
        return_value=(mock_model, mock_transform),
    ):
        foreground, mask = remove_background(rgba_with_transparency, force_remove=True)
        mock_model.assert_called_once()

    assert foreground.mode == "RGBA"
    assert mask.dtype == np.float32
