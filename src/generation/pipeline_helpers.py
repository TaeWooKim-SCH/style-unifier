"""StyleUnificationPipeline 변환 헬퍼 자유 함수 모음.

``pipeline.py`` 클래스 메서드 중 ``self`` 의존성이 없는 순수 로직을 분리한 모듈.
``pipeline.py`` 에서 직접 import 하여 사용한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.postprocessing.attribute_control import AttributeScales
from src.postprocessing.region_mask import apply_region_mask
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ControlNet 인자 빌드
# ---------------------------------------------------------------------------


def build_controlnet_args(
    config: dict,
    lineart_img: Image.Image,
    scales: AttributeScales | None = None,
) -> tuple[list[Image.Image], list[float]]:
    """Enabled ControlNet의 control_image·scale 리스트를 빌드한다.

    Args:
        config: ``configs/default.yaml`` 구조의 설정 dict.
        lineart_img: 전처리에서 추출한 lineart RGB 이미지.
        scales: None이면 config 값, 아니면 controlnet_lineart_scale 오버라이드.

    Returns:
        ``(control_images, conditioning_scales)``.
    """
    enabled_cns = [c for c in config["model"]["controlnet"] if c.get("enabled", True)]

    control_images: list[Image.Image] = []
    cn_scales: list[float] = []

    for cn_cfg in enabled_cns:
        control_images.append(lineart_img)
        if scales is not None:
            cn_scales.append(float(scales.controlnet_lineart_scale))
        else:
            cn_scales.append(float(cn_cfg["scale"]))

    return control_images, cn_scales


def pack_controlnet_args(
    config: dict,
    lineart_img: Image.Image,
    effective_scales: AttributeScales | None,
) -> tuple[Any, Any]:
    """ControlNet 인자를 단일 값 또는 리스트로 패킹한다.

    Args:
        config: ``configs/default.yaml`` 구조의 설정 dict.
        lineart_img: 전처리에서 추출한 lineart RGB 이미지.
        effective_scales: None이면 config 스케일 사용.

    Returns:
        ``(control_arg, scale_arg)``. 단일 ControlNet이면 스칼라, 복수면 리스트.
    """
    control_images, conditioning_scales = build_controlnet_args(
        config, lineart_img, scales=effective_scales
    )
    control_arg: Any = control_images[0] if len(control_images) == 1 else control_images
    scale_arg: Any = (
        conditioning_scales[0] if len(conditioning_scales) == 1 else conditioning_scales
    )
    return control_arg, scale_arg


# ---------------------------------------------------------------------------
# 샘플링 설정 읽기
# ---------------------------------------------------------------------------


def read_sampling_cfg(
    config: dict,
    effective_scales: AttributeScales | None,
) -> tuple[int, float, float, float | None]:
    """Sampling 설정에서 추론 파라미터를 읽어 반환한다.

    Args:
        config: ``configs/default.yaml`` 구조의 설정 dict.
        effective_scales: None이면 ip_adapter_scale을 None으로 반환.

    Returns:
        ``(num_steps, guidance_scale, strength, ip_adapter_scale)``.
    """
    sampling_cfg = config["sampling"]
    num_steps: int = int(sampling_cfg["steps"])
    guidance_scale: float = float(sampling_cfg["cfg_scale"])
    strength: float = float(sampling_cfg["denoising_strength"])
    ip_adapter_scale: float | None = (
        float(effective_scales.ip_adapter_scale) if effective_scales is not None else None
    )
    return num_steps, guidance_scale, strength, ip_adapter_scale


# ---------------------------------------------------------------------------
# Effective scales 결정
# ---------------------------------------------------------------------------


def resolve_effective_scales(
    scales: AttributeScales | None,
    attribute_mode: str | None,
) -> AttributeScales | None:
    """Scales 또는 attribute_mode로부터 effective AttributeScales를 결정한다.

    Args:
        scales: 직접 전달된 AttributeScales. None이면 attribute_mode 참조.
        attribute_mode: 속성 모드 문자열. scales가 None일 때만 사용.

    Returns:
        결정된 AttributeScales. 둘 다 None이면 None (config 경로 사용).
    """
    if scales is not None:
        return scales
    if attribute_mode is not None:
        from src.postprocessing.attribute_control import route_scales

        resolved = route_scales({}, mode=attribute_mode)
        logger.debug("attribute_mode=%r -> scales=%r", attribute_mode, resolved)
        return resolved
    return None


# ---------------------------------------------------------------------------
# Region mask 적용
# ---------------------------------------------------------------------------


def apply_region_mask_if_needed(
    source: Image.Image,
    result_rgba: Image.Image,
    region_mask: npt.NDArray[np.float32] | None,
) -> Image.Image:
    """region_mask가 있으면 source와 result를 블렌딩해 반환한다.

    Args:
        source: 원본 에셋 PIL Image. 마스크 0 영역에 사용.
        result_rgba: 변환 결과 RGBA Image. 마스크 1 영역에 사용.
        region_mask: (H, W) float32 [0, 1] 마스크. None이면 result_rgba 그대로 반환.

    Returns:
        region_mask 적용 후 RGBA Image (또는 원본 result_rgba).

    Raises:
        ValueError: region_mask shape이 result_rgba 크기와 맞지 않을 때.
    """
    if region_mask is None:
        return result_rgba

    source_rgba = source if source.mode == "RGBA" else source.convert("RGBA")

    mask_h, mask_w = region_mask.shape
    res_w, res_h = result_rgba.size
    if (mask_h, mask_w) != (res_h, res_w):
        # 자동 리사이즈 — 사용자가 그린 mask는 canvas 크기를 따르고, result는 8-multiple로
        # 보정된 크기라 거의 항상 불일치한다. Bilinear interpolation으로 mask 자체를 결과
        # 해상도에 맞춘다. (binary mask가 아닌 soft mask 가정)
        logger.info(
            "Auto-resizing region_mask from (%d, %d) to (%d, %d)",
            mask_h,
            mask_w,
            res_h,
            res_w,
        )
        mask_uint8 = (np.clip(region_mask, 0.0, 1.0) * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8, mode="L")
        mask_pil = mask_pil.resize((res_w, res_h), Image.Resampling.BILINEAR)
        region_mask = np.array(mask_pil, dtype=np.float32) / 255.0

    if source_rgba.size != result_rgba.size:
        source_rgba = source_rgba.resize(result_rgba.size, Image.Resampling.BILINEAR)

    blended = apply_region_mask(source=source_rgba, transformed=result_rgba, mask=region_mask)
    logger.debug("region_mask applied: result size=%s", blended.size)
    return blended
