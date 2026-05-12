"""attribute_control.py 단위 테스트 (D2).

모델 로딩 없음. 순수 로직(프리셋 라우팅, enum 변환, 검증)만 검증한다.
"""

from __future__ import annotations

import pytest

from src.postprocessing.attribute_control import (
    AttributeScales,
    StyleAttribute,
    attribute_from_string,
    route_scales,
)

# ---------------------------------------------------------------------------
# route_scales — 프리셋 모드
# ---------------------------------------------------------------------------


def test_route_scales_palette_mode_returns_attribute_scales():
    """mode='palette'가 AttributeScales 인스턴스를 반환하는가."""
    result = route_scales({}, mode="palette")
    assert isinstance(result, AttributeScales)


def test_route_scales_palette_mode_palette_quantize_strength():
    """mode='palette' 프리셋의 palette_quantize_strength가 1.0인가."""
    result = route_scales({}, mode="palette")
    assert result.palette_quantize_strength == 1.0


def test_route_scales_palette_mode_ip_adapter_scale():
    """mode='palette' 프리셋의 ip_adapter_scale이 0.3인가."""
    result = route_scales({}, mode="palette")
    assert result.ip_adapter_scale == pytest.approx(0.3)


def test_route_scales_lineart_mode_lineart_postprocess_true():
    """mode='lineart' 프리셋의 lineart_postprocess가 True인가."""
    result = route_scales({}, mode="lineart")
    assert result.lineart_postprocess is True


def test_route_scales_shading_mode_shading_match_strength():
    """mode='shading' 프리셋의 shading_match_strength가 1.0인가."""
    result = route_scales({}, mode="shading")
    assert result.shading_match_strength == pytest.approx(1.0)


def test_route_scales_full_mode_all_nonzero():
    """mode='full' 프리셋의 핵심 scale이 모두 0이 아닌가."""
    result = route_scales({}, mode="full")
    assert result.ip_adapter_scale > 0.0
    assert result.controlnet_lineart_scale > 0.0
    assert result.palette_quantize_strength > 0.0


def test_route_scales_unknown_mode_raises():
    """유효하지 않은 mode에서 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="mode"):
        route_scales({}, mode="unknown_mode")


# ---------------------------------------------------------------------------
# route_scales — selective 모드
# ---------------------------------------------------------------------------


def test_route_scales_selective_mode_returns_attribute_scales():
    """mode='selective'가 AttributeScales를 반환하는가."""
    result = route_scales({StyleAttribute.PALETTE: 0.5}, mode="selective")
    assert isinstance(result, AttributeScales)


def test_route_scales_selective_all_zero_falls_back_to_full():
    """selective 모드에서 모든 강도가 0이면 full 프리셋으로 fallback하는가."""
    full_preset = route_scales({}, mode="full")
    selective_result = route_scales({}, mode="selective")
    assert selective_result.ip_adapter_scale == full_preset.ip_adapter_scale


def test_route_scales_selective_strength_out_of_range_raises():
    """강도 값이 1.1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="[Ss]trength"):
        route_scales({StyleAttribute.PALETTE: 1.1}, mode="selective")


def test_route_scales_selective_negative_strength_raises():
    """강도 값이 음수이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="[Ss]trength"):
        route_scales({StyleAttribute.LINEART: -0.1}, mode="selective")


def test_route_scales_selective_palette_maps_correctly():
    """selective 모드에서 PALETTE=1.0이 palette_quantize_strength=1.0인가."""
    result = route_scales({StyleAttribute.PALETTE: 1.0}, mode="selective")
    assert result.palette_quantize_strength == pytest.approx(1.0)


def test_route_scales_selective_lineart_postprocess_enabled():
    """selective 모드에서 LINEART > 0이면 lineart_postprocess=True인가."""
    result = route_scales({StyleAttribute.LINEART: 0.5}, mode="selective")
    assert result.lineart_postprocess is True


def test_route_scales_selective_lineart_zero_postprocess_disabled():
    """selective 모드에서 LINEART=0이면 lineart_postprocess=False인가."""
    result = route_scales(
        {StyleAttribute.PALETTE: 1.0, StyleAttribute.LINEART: 0.0}, mode="selective"
    )
    assert result.lineart_postprocess is False


# ---------------------------------------------------------------------------
# attribute_from_string
# ---------------------------------------------------------------------------


def test_attribute_from_string_palette():
    """'palette' 문자열이 StyleAttribute.PALETTE로 변환되는가."""
    result = attribute_from_string("palette")
    assert result is StyleAttribute.PALETTE


def test_attribute_from_string_lineart():
    """'lineart' 문자열이 StyleAttribute.LINEART로 변환되는가."""
    result = attribute_from_string("lineart")
    assert result is StyleAttribute.LINEART


def test_attribute_from_string_shading():
    """'shading' 문자열이 StyleAttribute.SHADING으로 변환되는가."""
    result = attribute_from_string("shading")
    assert result is StyleAttribute.SHADING


def test_attribute_from_string_case_insensitive():
    """대소문자를 무시하고 변환하는가."""
    result = attribute_from_string("PALETTE")
    assert result is StyleAttribute.PALETTE


def test_attribute_from_string_unknown_raises():
    """알 수 없는 속성 이름에서 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="Unknown attribute"):
        attribute_from_string("unknown_attr")
