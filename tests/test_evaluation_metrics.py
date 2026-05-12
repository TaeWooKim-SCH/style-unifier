"""정량 메트릭 단위 테스트 (C1).

모델 로딩은 모두 mock으로 우회한다.
palette_distance / gram_matrix_distance 내 비모델 로직은 직접 테스트한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation.metrics import (
    _cosine_similarity,
    _gram_matrix,
    _mean_nearest_distance,
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
def solid_red_rgba() -> Image.Image:
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [255, 0, 0, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def solid_blue_rgba() -> Image.Image:
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :] = [0, 0, 255, 255]
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def sample_rgb() -> Image.Image:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# palette_distance — 모델 미사용 (extract_palette + skimage 의존)
# ---------------------------------------------------------------------------


@skip_if_no_skimage
def test_palette_distance_identical_images_near_zero(solid_red_rgba):
    """동일 이미지 쌍의 palette_distance가 0에 가까운가."""
    from src.evaluation.metrics import palette_distance

    dist = palette_distance(solid_red_rgba, solid_red_rgba, k=4)
    assert dist == pytest.approx(0.0, abs=1.0)


@skip_if_no_skimage
def test_palette_distance_different_colors_positive(solid_red_rgba, solid_blue_rgba):
    """다른 색 이미지 쌍의 palette_distance > 동일 이미지 쌍인가."""
    from src.evaluation.metrics import palette_distance

    same = palette_distance(solid_red_rgba, solid_red_rgba, k=4)
    diff = palette_distance(solid_red_rgba, solid_blue_rgba, k=4)
    assert diff > same


@skip_if_no_skimage
def test_palette_distance_returns_float(solid_red_rgba, solid_blue_rgba):
    """반환값이 float 타입인가."""
    from src.evaluation.metrics import palette_distance

    result = palette_distance(solid_red_rgba, solid_blue_rgba, k=4)
    assert isinstance(result, float)


@skip_if_no_skimage
def test_palette_distance_nonnegative(solid_red_rgba, solid_blue_rgba):
    """palette_distance가 항상 >= 0인가."""
    from src.evaluation.metrics import palette_distance

    result = palette_distance(solid_red_rgba, solid_blue_rgba, k=4)
    assert result >= 0.0


@skip_if_no_skimage
def test_palette_distance_deterministic(solid_red_rgba, solid_blue_rgba):
    """같은 입력에서 두 번 호출 결과가 동일한가."""
    from src.evaluation.metrics import palette_distance

    r1 = palette_distance(solid_red_rgba, solid_blue_rgba, k=4, random_state=42)
    r2 = palette_distance(solid_red_rgba, solid_blue_rgba, k=4, random_state=42)
    assert r1 == r2


# ---------------------------------------------------------------------------
# _mean_nearest_distance — 순수 numpy 함수 직접 테스트
# ---------------------------------------------------------------------------


def test_mean_nearest_distance_same_sets_is_zero():
    """동일한 query/target에서 _mean_nearest_distance가 0.0인가."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    result = _mean_nearest_distance(pts, pts)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_mean_nearest_distance_returns_float():
    """반환값이 float 타입인가."""
    query = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = _mean_nearest_distance(query, target)
    assert isinstance(result, float)


def test_mean_nearest_distance_correct_value():
    """단일 점 쌍에서 유클리드 거리가 올바른가."""
    query = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)  # 거리 = 5.0
    result = _mean_nearest_distance(query, target)
    assert result == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# _cosine_similarity — 순수 torch 함수 직접 테스트
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one():
    """동일 벡터의 cosine similarity가 1.0인가."""
    v = torch.tensor([1.0, 2.0, 3.0])
    result = _cosine_similarity(v, v)
    assert result == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    """직교 벡터의 cosine similarity가 0에 가까운가."""
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    result = _cosine_similarity(a, b)
    assert result == pytest.approx(0.0, abs=1e-5)


def test_cosine_similarity_returns_float():
    """반환값이 float 타입인가."""
    v = torch.tensor([1.0, 0.0])
    result = _cosine_similarity(v, v)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _gram_matrix — 순수 torch 함수 직접 테스트
# ---------------------------------------------------------------------------


def test_gram_matrix_shape():
    """(1, C, H, W) 입력에서 Gram matrix shape이 (1, C, C)인가."""
    feat = torch.randn(1, 8, 4, 4)
    gram = _gram_matrix(feat)
    assert gram.shape == (1, 8, 8)


def test_gram_matrix_symmetric():
    """Gram matrix가 대칭 행렬인가."""
    feat = torch.randn(1, 4, 8, 8)
    gram = _gram_matrix(feat)
    diff = gram[0] - gram[0].t()
    assert diff.abs().max().item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# clip_style_similarity — 모델 mock
# ---------------------------------------------------------------------------


def test_clip_style_similarity_mock(solid_red_rgba, solid_blue_rgba):
    """CLIP 모델을 mock해서 clip_style_similarity가 float을 반환하는가."""
    import src.evaluation.metrics as m

    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.ones(1, 768)
    mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))

    with patch.dict(m._CLIP_CACHE, {("ViT-L-14", "openai"): (mock_model, mock_preprocess)}):
        from src.evaluation.metrics import clip_style_similarity

        result = clip_style_similarity(solid_red_rgba, solid_blue_rgba)

    assert isinstance(result, float)
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# lpips_structure — 사이즈 불일치 ValueError + mock
# ---------------------------------------------------------------------------


def test_lpips_structure_size_mismatch_raises():
    """source와 generated 크기가 다를 때 ValueError가 발생하는가."""
    from src.evaluation.metrics import lpips_structure

    src = Image.new("RGB", (64, 64))
    gen = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="same size"):
        lpips_structure(src, gen)


def test_lpips_structure_mock(solid_red_rgba):
    """LPIPS 모델을 mock해서 lpips_structure가 float을 반환하는가."""
    pytest.importorskip("torchvision", reason="torchvision not installed")

    import src.evaluation.metrics as m

    mock_loss_fn = MagicMock()
    mock_loss_fn.return_value = torch.tensor(0.25)

    with (
        patch.dict(m._LPIPS_CACHE, {"vgg": mock_loss_fn}),
        patch("torchvision.transforms.functional.to_tensor", return_value=torch.zeros(3, 64, 64)),
    ):
        from src.evaluation.metrics import lpips_structure

        result = lpips_structure(solid_red_rgba, solid_red_rgba)

    assert isinstance(result, float)
