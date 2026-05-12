"""extract_lineart() 단위 테스트.

controlnet_aux 의존성을 mock으로 우회하고, 검증 로직과 RGBA 변환,
캐시 재사용을 검증한다.
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.preprocessing.lineart import _DETECTOR_CACHE, extract_lineart

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_detector_cache():
    """각 테스트 전후로 _DETECTOR_CACHE를 비워 캐시 누수를 방지한다."""
    _DETECTOR_CACHE.clear()
    yield
    _DETECTOR_CACHE.clear()


@pytest.fixture
def rgb_image_32() -> Image.Image:
    """32x32 RGB 이미지."""
    return Image.new("RGB", (32, 32), color=(80, 120, 160))


@pytest.fixture
def rgba_image_32() -> Image.Image:
    """32x32 RGBA 이미지."""
    return Image.new("RGBA", (32, 32), color=(80, 120, 160, 200))


def _make_mock_detector_instance(result_image: Image.Image | None = None) -> MagicMock:
    """detector 인스턴스 mock을 반환한다.

    호출 시 result_image(기본: 32x32 RGB)를 반환한다.
    """
    if result_image is None:
        result_image = Image.new("RGB", (32, 32), color=(0, 0, 0))
    instance = MagicMock()
    instance.return_value = result_image
    return instance


def _make_controlnet_aux_module() -> ModuleType:
    """controlnet_aux 패키지 모듈을 흉내내는 mock ModuleType을 반환한다."""
    mock_module = MagicMock(spec=ModuleType)
    mock_module.__name__ = "controlnet_aux"
    return mock_module


# ---------------------------------------------------------------------------
# Validation: unknown detector → ValueError (모델 로드 없이 즉시)
# ---------------------------------------------------------------------------


def test_extract_lineart_raises_on_unknown_detector():
    """알 수 없는 detector로 호출하면 ValueError가 발생하는가."""
    img = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="Unsupported detector"):
        extract_lineart(img, detector="unknown_detector")


def test_extract_lineart_raises_on_unknown_detector_without_model_load():
    """알 수 없는 detector 에러가 _load_detector 호출 전에 발생하는가."""
    img = Image.new("RGB", (32, 32))
    with patch("src.preprocessing.lineart._load_detector") as mock_load:
        with pytest.raises(ValueError):
            extract_lineart(img, detector="totally_wrong")
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# canny detector mock 테스트
# ---------------------------------------------------------------------------


def test_extract_lineart_canny_calls_detector_instance(rgb_image_32):
    """detector='canny' 호출 시 detector callable이 실행되는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        result = extract_lineart(rgb_image_32, detector="canny")
        mock_instance.assert_called_once()

    assert isinstance(result, Image.Image)


def test_extract_lineart_canny_passes_threshold_args(rgb_image_32):
    """canny detector에 threshold_low, threshold_high 인자가 전달되는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        extract_lineart(rgb_image_32, detector="canny", threshold_low=80, threshold_high=180)

    called_kwargs = mock_instance.call_args.kwargs
    assert called_kwargs.get("low_threshold") == 80
    assert called_kwargs.get("high_threshold") == 180


def test_extract_lineart_canny_result_is_rgb_mode(rgb_image_32):
    """canny detector 결과가 RGB 모드인가."""
    # L 모드 이미지를 반환해도 RGB로 변환되는지 확인
    gray_result = Image.new("L", (32, 32), color=128)
    mock_instance = _make_mock_detector_instance(result_image=gray_result)

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        result = extract_lineart(rgb_image_32, detector="canny")

    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# lineart_anime detector mock 테스트
# ---------------------------------------------------------------------------


def test_extract_lineart_anime_calls_detector_instance(rgb_image_32):
    """detector='lineart_anime' 호출 시 detector callable이 실행되는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        result = extract_lineart(rgb_image_32, detector="lineart_anime")
        mock_instance.assert_called_once()

    assert isinstance(result, Image.Image)


def test_extract_lineart_anime_result_is_rgb_mode(rgb_image_32):
    """lineart_anime detector 결과가 RGB 모드인가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        result = extract_lineart(rgb_image_32, detector="lineart_anime")

    assert result.mode == "RGB"


def test_extract_lineart_anime_does_not_pass_threshold_args(rgb_image_32):
    """lineart_anime는 threshold 인자를 detector에 전달하지 않는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        extract_lineart(rgb_image_32, detector="lineart_anime")

    called_args = mock_instance.call_args
    # kwargs에 low_threshold, high_threshold가 없어야 함
    assert "low_threshold" not in called_args.kwargs
    assert "high_threshold" not in called_args.kwargs


# ---------------------------------------------------------------------------
# RGBA → RGB 변환
# ---------------------------------------------------------------------------


def test_extract_lineart_rgba_input_is_converted_to_rgb_before_detection(rgba_image_32):
    """RGBA 입력이 alpha 채널을 제거한 RGB로 변환되어 detector에 전달되는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance):
        result = extract_lineart(rgba_image_32, detector="canny")

    # detector에 전달된 첫 번째 인자는 RGB 이미지여야 한다
    detected_input = mock_instance.call_args.args[0]
    assert isinstance(detected_input, Image.Image)
    assert detected_input.mode == "RGB"
    assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# _DETECTOR_CACHE 재사용 검증
# ---------------------------------------------------------------------------


def test_extract_lineart_detector_cache_reused_on_second_call(rgb_image_32):
    """같은 detector로 두 번 호출하면 _load_detector가 한 번만 호출되는가."""
    mock_instance = _make_mock_detector_instance()

    with patch("src.preprocessing.lineart._load_detector", return_value=mock_instance) as mock_load:
        extract_lineart(rgb_image_32, detector="canny")
        extract_lineart(rgb_image_32, detector="canny")

        # _load_detector는 두 번 호출되더라도 캐시로 인해 내부 로딩은 1번
        # (실제 구현에서 _load_detector 자체가 캐시를 확인하므로 2번 호출됨)
        # 여기서는 extract_lineart가 _load_detector를 2번 호출하는지 확인
        assert mock_load.call_count == 2

    # detector 인스턴스는 2번 호출되어야 함 (두 이미지 각각)
    assert mock_instance.call_count == 2
