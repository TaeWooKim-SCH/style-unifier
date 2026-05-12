"""StyleEncoder 단위 테스트.

transformers의 CLIPVisionModelWithProjection / CLIPImageProcessor를 mock으로
우회하고, lazy loading, cache hit, encode_multi 전략, clear_cache를 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.encoding.ip_adapter_wrapper import StyleEncoder, _image_cache_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 16
DIM = 32


def _make_mock_clip_model_and_processor(seq_len: int = SEQ_LEN, dim: int = DIM):
    """CLIPVisionModelWithProjection과 CLIPImageProcessor mock을 반환한다.

    processor(images=..., return_tensors="pt").pixel_values → (1, 3, 224, 224)
    model(pixel_values=...).last_hidden_state → (1, seq_len, dim)
    """
    mock_processor = MagicMock()
    dummy_pixel_values = torch.zeros(1, 3, 224, 224)
    mock_processor_output = MagicMock()
    mock_processor_output.pixel_values = dummy_pixel_values
    mock_processor.return_value = mock_processor_output

    mock_model = MagicMock()
    mock_model_output = MagicMock()
    dummy_embedding = torch.ones(1, seq_len, dim)
    mock_model_output.last_hidden_state = dummy_embedding
    mock_model.return_value = mock_model_output

    # from_pretrained mock
    mock_clip_class = MagicMock()
    mock_clip_class.from_pretrained.return_value = mock_model

    mock_processor_class = MagicMock()
    mock_processor_class.from_pretrained.return_value = mock_processor

    return mock_clip_class, mock_processor_class, mock_model, mock_processor


@pytest.fixture
def encoder_with_mocked_model():
    """CLIPVisionModelWithProjection, CLIPImageProcessor를 mock한 StyleEncoder."""
    mock_clip_class, mock_processor_class, mock_model, mock_processor = (
        _make_mock_clip_model_and_processor()
    )
    with patch.dict(
        "sys.modules",
        {
            "transformers": MagicMock(
                CLIPVisionModelWithProjection=mock_clip_class,
                CLIPImageProcessor=mock_processor_class,
            )
        },
    ):
        encoder = StyleEncoder(device=torch.device("cpu"))
        # _ensure_loaded가 transformers를 import하도록 강제로 호출
        with (
            patch(
                "src.encoding.ip_adapter_wrapper.CLIPVisionModelWithProjection",
                mock_clip_class,
                create=True,
            ),
            patch(
                "src.encoding.ip_adapter_wrapper.CLIPImageProcessor",
                mock_processor_class,
                create=True,
            ),
        ):
            # image_encoder와 image_processor를 직접 주입
            encoder.image_encoder = mock_model
            encoder.image_processor = mock_processor

    return encoder, mock_model, mock_processor


@pytest.fixture
def sample_image() -> Image.Image:
    """64x64 RGB 테스트 이미지."""
    return Image.new("RGB", (64, 64), color=(100, 150, 200))


@pytest.fixture
def another_image() -> Image.Image:
    """64x64 다른 색상 RGB 테스트 이미지."""
    return Image.new("RGB", (64, 64), color=(200, 50, 80))


# ---------------------------------------------------------------------------
# __init__ lazy 가드 — 인스턴스 생성 시 모델 미로드
# ---------------------------------------------------------------------------


def test_style_encoder_init_does_not_load_model():
    """StyleEncoder() 생성 시 from_pretrained가 호출되지 않는가."""
    with (
        patch("src.encoding.ip_adapter_wrapper.get_device", return_value=torch.device("cpu")),
        patch("src.encoding.ip_adapter_wrapper.get_dtype", return_value=torch.float32),
        patch("transformers.CLIPVisionModelWithProjection.from_pretrained") as mock_from_pretrained,
    ):
        encoder = StyleEncoder()
        mock_from_pretrained.assert_not_called()

    assert encoder.image_encoder is None
    assert encoder.image_processor is None


def test_style_encoder_init_pipe_is_none():
    """StyleEncoder() 생성 직후 image_encoder가 None인가."""
    encoder = StyleEncoder(device=torch.device("cpu"))
    assert encoder.image_encoder is None


def test_style_encoder_init_loaded_is_none():
    """StyleEncoder() 생성 직후 image_processor가 None인가."""
    encoder = StyleEncoder(device=torch.device("cpu"))
    assert encoder.image_processor is None


# ---------------------------------------------------------------------------
# _image_cache_key — 동일 픽셀 이미지는 같은 키
# ---------------------------------------------------------------------------


def test_image_cache_key_same_content_same_key(sample_image):
    """픽셀 내용이 같은 두 PIL 이미지 → 동일 캐시 키."""
    img1 = Image.new("RGB", (64, 64), color=(100, 150, 200))
    img2 = Image.new("RGB", (64, 64), color=(100, 150, 200))
    assert _image_cache_key(img1) == _image_cache_key(img2)


def test_image_cache_key_different_content_different_key(sample_image, another_image):
    """픽셀 내용이 다른 두 이미지 → 다른 캐시 키."""
    assert _image_cache_key(sample_image) != _image_cache_key(another_image)


def test_image_cache_key_returns_string(sample_image):
    """_image_cache_key 반환값이 문자열인가."""
    key = _image_cache_key(sample_image)
    assert isinstance(key, str)


# ---------------------------------------------------------------------------
# encode_single — cache miss 시 모델 1번 호출, cache hit 시 0번
# ---------------------------------------------------------------------------


def test_encode_single_returns_tensor(encoder_with_mocked_model, sample_image):
    """encode_single() 반환값이 torch.Tensor인가."""
    encoder, _, _ = encoder_with_mocked_model
    result = encoder.encode_single(sample_image)
    assert isinstance(result, torch.Tensor)


def test_encode_single_returns_correct_shape(encoder_with_mocked_model, sample_image):
    """encode_single() 반환 shape이 (1, seq_len, dim)인가."""
    encoder, _, _ = encoder_with_mocked_model
    result = encoder.encode_single(sample_image)
    assert result.ndim == 3
    assert result.shape[0] == 1


def test_encode_single_cache_hit_same_image_model_called_once(
    encoder_with_mocked_model, sample_image
):
    """같은 이미지로 encode_single()을 두 번 호출하면 모델이 1번만 호출되는가."""
    encoder, mock_model, _ = encoder_with_mocked_model
    encoder.encode_single(sample_image)
    encoder.encode_single(sample_image)
    # 캐시 hit이면 model.__call__은 1번만
    assert mock_model.call_count == 1


def test_encode_single_cache_hit_same_pixels_model_called_once(encoder_with_mocked_model):
    """픽셀이 같은 두 별개 PIL 이미지 → 모델 1번만 호출 (sha256 캐시 기반)."""
    encoder, mock_model, _ = encoder_with_mocked_model
    img_a = Image.new("RGB", (64, 64), color=(10, 20, 30))
    img_b = Image.new("RGB", (64, 64), color=(10, 20, 30))
    encoder.encode_single(img_a)
    encoder.encode_single(img_b)
    assert mock_model.call_count == 1


def test_encode_single_different_images_model_called_twice(
    encoder_with_mocked_model, sample_image, another_image
):
    """다른 이미지로 encode_single()을 각각 호출하면 모델이 2번 호출되는가."""
    encoder, mock_model, _ = encoder_with_mocked_model
    encoder.encode_single(sample_image)
    encoder.encode_single(another_image)
    assert mock_model.call_count == 2


# ---------------------------------------------------------------------------
# encode_multi — strategy 검증
# ---------------------------------------------------------------------------


def test_encode_multi_mean_returns_tensor(encoder_with_mocked_model, sample_image, another_image):
    """encode_multi(strategy='mean') 반환값이 torch.Tensor인가."""
    encoder, _, _ = encoder_with_mocked_model
    result = encoder.encode_multi([sample_image, another_image], strategy="mean")
    assert isinstance(result, torch.Tensor)


def test_encode_multi_mean_shape_matches_encode_single(
    encoder_with_mocked_model, sample_image, another_image
):
    """encode_multi(strategy='mean') shape이 encode_single과 동일한가."""
    encoder, _, _ = encoder_with_mocked_model
    single = encoder.encode_single(sample_image)
    encoder.clear_cache()
    multi = encoder.encode_multi([sample_image, another_image], strategy="mean")
    assert multi.shape == single.shape


def test_encode_multi_first_equals_encode_single_of_first_image(
    encoder_with_mocked_model, sample_image, another_image
):
    """encode_multi(strategy='first') 결과가 references[0]의 encode_single과 같은가."""
    encoder, _, _ = encoder_with_mocked_model
    first_result = encoder.encode_multi([sample_image, another_image], strategy="first")
    encoder.clear_cache()
    single_result = encoder.encode_single(sample_image)
    # 캐시 동작 덕분에 동일한 텐서 객체
    assert torch.equal(first_result, single_result)


def test_encode_multi_weighted_no_target_falls_back_to_mean(
    encoder_with_mocked_model, sample_image, another_image
):
    """strategy='weighted' + target_embedding=None이면 mean과 동일한 shape을 반환하는가."""
    encoder, _, _ = encoder_with_mocked_model
    weighted = encoder.encode_multi(
        [sample_image, another_image], strategy="weighted", target_embedding=None
    )
    encoder.clear_cache()
    mean = encoder.encode_multi([sample_image, another_image], strategy="mean")
    assert weighted.shape == mean.shape


def test_encode_multi_raises_on_empty_references(encoder_with_mocked_model):
    """references=[]이면 ValueError가 발생하는가."""
    encoder, _, _ = encoder_with_mocked_model
    with pytest.raises(ValueError, match="references must not be empty"):
        encoder.encode_multi([])


def test_encode_multi_raises_on_invalid_strategy(encoder_with_mocked_model, sample_image):
    """지원하지 않는 strategy이면 ValueError가 발생하는가."""
    encoder, _, _ = encoder_with_mocked_model
    with pytest.raises(ValueError, match="strategy must be one of"):
        encoder.encode_multi([sample_image], strategy="invalid_strategy")


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_forces_model_recall_on_next_encode(encoder_with_mocked_model, sample_image):
    """clear_cache() 후 같은 이미지 재호출 시 모델이 다시 호출되는가."""
    encoder, mock_model, _ = encoder_with_mocked_model
    encoder.encode_single(sample_image)
    assert mock_model.call_count == 1

    encoder.clear_cache()

    encoder.encode_single(sample_image)
    assert mock_model.call_count == 2


def test_clear_cache_empties_internal_cache(encoder_with_mocked_model, sample_image):
    """clear_cache() 후 _cache가 비어 있는가."""
    encoder, _, _ = encoder_with_mocked_model
    encoder.encode_single(sample_image)
    assert len(encoder._cache) > 0

    encoder.clear_cache()
    assert len(encoder._cache) == 0
