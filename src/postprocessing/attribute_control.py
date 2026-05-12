"""속성 단위 스타일 제어 모듈 (D2).

사용자가 'palette만', 'lineart만' 등 속성 단위로 변환 범위를 제어할 수 있도록
ControlNet/IP-Adapter scale과 후처리 단계 on/off를 라우팅한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 유효 모드 집합
# ---------------------------------------------------------------------------
_VALID_MODES = frozenset({"palette", "lineart", "shading", "full", "selective"})


# ---------------------------------------------------------------------------
# 공개 타입
# ---------------------------------------------------------------------------


class StyleAttribute(str, Enum):
    """Selective stylization 대상 속성."""

    PALETTE = "palette"
    LINEART = "lineart"
    SHADING = "shading"


@dataclass(frozen=True)
class AttributeScales:
    """``route_scales()`` 반환 — pipeline.transform()에 전달할 scale 묶음.

    Attributes:
        ip_adapter_scale: IP-Adapter scale (0~1).
        controlnet_lineart_scale: lineart ControlNet scale (0~1.5).
        controlnet_depth_scale: depth ControlNet scale (0~1.5).
        palette_quantize_strength: 후처리 palette quantize 강도 (0~1).
        shading_match_strength: 후처리 shading 히스토그램 매칭 강도 (0~1).
        lineart_postprocess: lineart 후처리 단계 on/off.
    """

    ip_adapter_scale: float
    controlnet_lineart_scale: float
    controlnet_depth_scale: float
    palette_quantize_strength: float
    shading_match_strength: float
    lineart_postprocess: bool


# ---------------------------------------------------------------------------
# 프리셋 테이블 (4개 고정 모드)
# ---------------------------------------------------------------------------
_PRESETS: dict[str, AttributeScales] = {
    "palette": AttributeScales(
        ip_adapter_scale=0.3,
        controlnet_lineart_scale=1.2,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=1.0,
        shading_match_strength=0.0,
        lineart_postprocess=False,
    ),
    "lineart": AttributeScales(
        ip_adapter_scale=0.4,
        controlnet_lineart_scale=0.6,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=0.0,
        shading_match_strength=0.0,
        lineart_postprocess=True,
    ),
    "shading": AttributeScales(
        ip_adapter_scale=0.5,
        controlnet_lineart_scale=1.2,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=0.0,
        shading_match_strength=1.0,
        lineart_postprocess=False,
    ),
    "full": AttributeScales(
        ip_adapter_scale=0.7,
        controlnet_lineart_scale=0.9,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=0.7,
        shading_match_strength=0.5,
        lineart_postprocess=True,
    ),
}


# ---------------------------------------------------------------------------
# 공개 함수
# ---------------------------------------------------------------------------


def route_scales(
    attributes_on: dict[StyleAttribute, float],
    *,
    mode: str = "selective",
) -> AttributeScales:
    """속성별 강도를 ControlNet/IP-Adapter scale + 후처리 토글로 라우팅한다.

    다섯 가지 모드를 지원한다:
      - ``"palette"``: palette만 적용. 나머지 속성은 0.
      - ``"lineart"``: lineart만 적용. 나머지 속성은 0.
      - ``"shading"``: shading만 적용. 나머지 속성은 0.
      - ``"full"``: 모든 속성 적용 (= 현재 baseline 동작).
      - ``"selective"``: ``attributes_on`` 인자대로 동적 라우팅.

    Args:
        attributes_on: 각 속성의 강도 (0~1). 누락된 키는 0(off)으로 간주.
            ``mode != "selective"`` 인 경우 이 인자는 무시된다.
        mode: 위 5개 중 하나. 기본값은 ``"selective"``.

    Returns:
        ``AttributeScales`` 인스턴스. ``StyleUnificationPipeline.transform()``
        의 scale 인자로 매핑 가능한 형식.

    Raises:
        ValueError: ``mode`` 가 유효하지 않거나, 강도 값이 [0, 1] 밖일 때.

    Example:
        >>> scales = route_scales({StyleAttribute.PALETTE: 1.0}, mode="palette")
        >>> scales.ip_adapter_scale
        0.3
        >>> scales.palette_quantize_strength
        1.0
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")

    if mode != "selective":
        result = _PRESETS[mode]
        logger.debug("route_scales: preset mode=%r -> %r", mode, result)
        return result

    # selective 모드: attributes_on 검증 후 동적 계산
    _validate_attributes(attributes_on)

    result = _compute_selective(attributes_on)
    logger.debug("route_scales: selective mode, attributes_on=%r -> %r", attributes_on, result)
    return result


def attribute_from_string(name: str) -> StyleAttribute:
    """문자열을 ``StyleAttribute``로 변환한다.

    Gradio UI 등에서 텍스트→enum 변환이 필요할 때 사용한다.

    Args:
        name: 속성 이름 문자열. 대소문자 구분 없음.

    Returns:
        대응하는 ``StyleAttribute`` 값.

    Raises:
        ValueError: 알 수 없는 속성 이름일 때.

    Example:
        >>> attribute_from_string("palette")
        <StyleAttribute.PALETTE: 'palette'>
        >>> attribute_from_string("LINEART")
        <StyleAttribute.LINEART: 'lineart'>
    """
    normalized = name.strip().lower()
    try:
        return StyleAttribute(normalized)
    except ValueError as err:
        valid = [a.value for a in StyleAttribute]
        raise ValueError(f"Unknown attribute {name!r}. Valid values: {valid}") from err


# ---------------------------------------------------------------------------
# Private 헬퍼
# ---------------------------------------------------------------------------


def _validate_attributes(attributes_on: dict[StyleAttribute, float]) -> None:
    """Selective 모드 입력 검증.

    Args:
        attributes_on: 검증할 속성→강도 딕셔너리.

    Raises:
        ValueError: 강도 값이 [0, 1] 밖일 때.
    """
    for attr, strength in attributes_on.items():
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"Strength for {attr!r} must be in [0, 1], got {strength}")


def _compute_selective(
    attributes_on: dict[StyleAttribute, float],
) -> AttributeScales:
    """Selective 모드 동적 라우팅 계산.

    강도 합이 0이면 full 프리셋으로 fallback한다.

    Mapping 규칙:
      - PALETTE 강도 → ``palette_quantize_strength`` 직접 매핑.
      - LINEART 강도 → ``controlnet_lineart_scale`` = 0.6 + 0.6 * 강도.
      - SHADING 강도 → ``shading_match_strength`` 직접 매핑.
      - IP-Adapter scale은 활성 속성의 최대 강도 x 0.7로 결정.
      - ``lineart_postprocess`` = LINEART 강도 > 0.

    Args:
        attributes_on: 검증 완료된 속성→강도 딕셔너리.

    Returns:
        계산된 ``AttributeScales``.
    """
    palette_strength = attributes_on.get(StyleAttribute.PALETTE, 0.0)
    lineart_strength = attributes_on.get(StyleAttribute.LINEART, 0.0)
    shading_strength = attributes_on.get(StyleAttribute.SHADING, 0.0)

    total = palette_strength + lineart_strength + shading_strength
    if total == 0.0:
        logger.warning(
            "route_scales(selective): all strengths are 0; falling back to 'full' preset."
        )
        return _PRESETS["full"]

    max_strength = max(palette_strength, lineart_strength, shading_strength)

    return AttributeScales(
        ip_adapter_scale=max_strength * 0.7,
        controlnet_lineart_scale=0.6 + 0.6 * lineart_strength,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=palette_strength,
        shading_match_strength=shading_strength,
        lineart_postprocess=lineart_strength > 0.0,
    )
