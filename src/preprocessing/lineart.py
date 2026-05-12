"""라인아트 추출 모듈.

ControlNet 입력으로 사용할 구조 조건 신호(엣지 맵)를 생성한다.
lineart_anime, lineart_realistic, canny 세 가지 detector를 지원한다.
"""

from __future__ import annotations

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

# detector 인스턴스 캐시: detector 이름 → 인스턴스
_DETECTOR_CACHE: dict[str, object] = {}

_SUPPORTED_DETECTORS = frozenset({"lineart_anime", "lineart_realistic", "canny"})


def _get_controlnet_aux() -> object:
    """controlnet_aux를 lazy-import해 반환한다.

    Raises:
        ImportError: controlnet_aux가 설치되어 있지 않을 때.
    """
    try:
        import controlnet_aux

        return controlnet_aux
    except ImportError as exc:
        raise ImportError(
            "controlnet_aux is required for lineart extraction."
            + " Install it with: pip install controlnet-aux"
        ) from exc


def _load_detector(detector: str) -> object:
    """Detector 인스턴스를 lazy-load해 캐시에 저장한다.

    Args:
        detector: "lineart_anime" | "lineart_realistic" | "canny".

    Returns:
        초기화된 detector 인스턴스.

    Raises:
        ImportError: controlnet_aux 미설치 시.
        ValueError: 지원하지 않는 detector 이름일 때.
    """
    if detector in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[detector]

    aux = _get_controlnet_aux()

    if detector == "lineart_anime":
        from controlnet_aux import LineartAnimeDetector  # type: ignore[import-untyped]

        logger.info("Loading LineartAnimeDetector")
        instance = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    elif detector == "lineart_realistic":
        from controlnet_aux import LineartDetector  # type: ignore[import-untyped]

        logger.info("Loading LineartDetector (realistic)")
        instance = LineartDetector.from_pretrained("lllyasviel/Annotators")
    elif detector == "canny":
        from controlnet_aux import CannyDetector  # type: ignore[import-untyped]

        logger.info("Initializing CannyDetector (no model download required)")
        instance = CannyDetector()
    else:
        raise ValueError(
            f"Unsupported detector: '{detector}'. Choose one of {sorted(_SUPPORTED_DETECTORS)}."
        )

    _ = aux  # suppress unused import warning
    _DETECTOR_CACHE[detector] = instance
    logger.debug("Detector cached: %s", detector)
    return instance


def extract_lineart(
    image: Image.Image,
    detector: str = "lineart_anime",
    *,
    threshold_low: int = 100,
    threshold_high: int = 200,
) -> Image.Image:
    """라인아트 맵을 RGB 이미지로 반환한다 (ControlNet 입력 형식).

    벡터/일러스트 스타일에 적합한 구조 조건 신호를 생성한다.
    lineart_anime와 lineart_realistic은 모델 로딩이 필요하며, 첫 호출 시에만
    다운로드된다. 이후 호출은 캐시된 인스턴스를 재사용한다.

    Args:
        image: 입력 PIL Image (RGB 또는 RGBA). RGBA면 alpha 무시 후 RGB 변환.
        detector: 사용할 검출기. "lineart_anime" | "lineart_realistic" | "canny".
        threshold_low: canny 사용 시 low threshold. lineart 검출기에서는 무시됨.
        threshold_high: canny 사용 시 high threshold. lineart 검출기에서는 무시됨.

    Returns:
        라인아트 PIL Image. mode="RGB". controlnet_aux 출력 형식 그대로.

    Raises:
        ValueError: detector가 지원하지 않는 값일 때.
        ImportError: controlnet_aux 미설치 시.

    Example:
        >>> img = Image.open("char.png")
        >>> lineart = extract_lineart(img, detector="lineart_anime")
        >>> lineart.save("char_lineart.png")
    """
    if detector not in _SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unsupported detector: '{detector}'. Choose one of {sorted(_SUPPORTED_DETECTORS)}."
        )

    # RGBA 입력은 RGB로 변환 (lineart 검출은 alpha 불필요)
    rgb_image = image.convert("RGB")

    detector_instance = _load_detector(detector)

    logger.info("Extracting lineart (detector=%s, size=%s)", detector, rgb_image.size)

    if detector == "canny":
        result = detector_instance(  # type: ignore[operator]
            rgb_image,
            low_threshold=threshold_low,
            high_threshold=threshold_high,
        )
    else:
        result = detector_instance(rgb_image)  # type: ignore[operator]

    # controlnet_aux는 PIL Image를 반환하지만 mode가 다를 수 있음 → RGB로 통일
    if not isinstance(result, Image.Image):
        result = Image.fromarray(result)

    if result.mode != "RGB":
        result = result.convert("RGB")

    logger.debug("Lineart extracted: mode=%s, size=%s", result.mode, result.size)
    return result
