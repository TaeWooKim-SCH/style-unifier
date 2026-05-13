"""StyleUnificationPipeline + load_config() 단위 테스트.

모델 로딩을 mock으로 우회하고, lazy 가드와 config 검증 로직을 검증한다.
C2: attribute_mode 라우팅 검증.
C3: region_mask 통합 검증.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.generation.pipeline import StyleUnificationPipeline, load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"

_MINIMAL_CONFIG = {
    "model": {
        "base_checkpoint": "fake/checkpoint",
        "ip_adapter": {
            "repo": "fake/ip-adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip.safetensors",
            "scale": 0.7,
        },
        "controlnet": [],
    },
    "sampling": {"steps": 2, "cfg_scale": 1.0, "denoising_strength": 0.5},
    "preprocessing": {},
    "postprocessing": {},
}


@pytest.fixture
def minimal_config() -> dict:
    """필수 키 모두 포함한 최소 설정 dict."""
    return dict(_MINIMAL_CONFIG)


# ---------------------------------------------------------------------------
# load_config() 검증
# ---------------------------------------------------------------------------


def test_load_config_returns_dict():
    """load_config() 반환값이 dict인가."""
    cfg = load_config(_DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, dict)


def test_load_config_required_keys_present():
    """필수 키 4개(model, sampling, preprocessing, postprocessing)가 모두 있는가."""
    cfg = load_config(_DEFAULT_CONFIG_PATH)
    for key in ("model", "sampling", "preprocessing", "postprocessing"):
        assert key in cfg, f"Required key '{key}' missing from config"


def test_load_config_raises_file_not_found_on_missing_file(tmp_path):
    """존재하지 않는 파일에 대해 FileNotFoundError가 발생하는가."""
    nonexistent = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(nonexistent)


def test_load_config_raises_key_error_on_missing_required_key(tmp_path):
    """필수 키가 빠진 YAML 파일에 대해 KeyError가 발생하는가."""
    incomplete_yaml = tmp_path / "incomplete.yaml"
    # 'postprocessing' 키 누락
    incomplete_yaml.write_text(
        "model:\n  base_checkpoint: fake\nsampling:\n  steps: 2\npreprocessing: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="postprocessing"):
        load_config(incomplete_yaml)


def test_load_config_raises_key_error_when_all_required_keys_missing(tmp_path):
    """완전히 비어 있는 YAML에 대해 KeyError가 발생하는가."""
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("{}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(empty_yaml)


# ---------------------------------------------------------------------------
# StyleUnificationPipeline.__init__ lazy 가드
# ---------------------------------------------------------------------------


def test_pipeline_init_does_not_load_models(minimal_config):
    """StyleUnificationPipeline(cfg) 생성 시 diffusers from_pretrained가 호출되지 않는가."""
    with patch(
        "diffusers.StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained"
    ) as mock_from_pretrained:
        pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
        mock_from_pretrained.assert_not_called()

    assert pipeline._loaded is False


def test_pipeline_init_pipe_is_none(minimal_config):
    """StyleUnificationPipeline(cfg) 생성 직후 pipe가 None인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert pipeline.pipe is None


def test_pipeline_init_loaded_is_false(minimal_config):
    """StyleUnificationPipeline(cfg) 생성 직후 _loaded가 False인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert pipeline._loaded is False


# ---------------------------------------------------------------------------
# device / dtype 속성
# ---------------------------------------------------------------------------


def test_pipeline_device_is_torch_device(minimal_config):
    """pipeline.device가 torch.device 인스턴스인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert isinstance(pipeline.device, torch.device)


def test_pipeline_dtype_is_torch_dtype(minimal_config):
    """pipeline.dtype이 torch.dtype 인스턴스인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert isinstance(pipeline.dtype, torch.dtype)


def test_pipeline_cpu_device_dtype_is_float32(minimal_config):
    """CPU device로 초기화하면 dtype이 float32인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert pipeline.dtype == torch.float32


# ---------------------------------------------------------------------------
# config 보존
# ---------------------------------------------------------------------------


def test_pipeline_stores_config_reference(minimal_config):
    """pipeline.config가 전달된 config dict와 동일한 객체인가."""
    pipeline = StyleUnificationPipeline(minimal_config, device=torch.device("cpu"))
    assert pipeline.config is minimal_config


# ---------------------------------------------------------------------------
# Helpers — mock pipeline with loaded pipe
# ---------------------------------------------------------------------------


def _make_rgba_image(w: int = 64, h: int = 64, color: tuple = (0, 0, 255, 255)) -> Image.Image:
    """단색 RGBA 이미지를 반환한다."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr, mode="RGBA")


def _make_loaded_pipeline(config: dict) -> StyleUnificationPipeline:
    """_loaded=True이고 pipe가 mock인 파이프라인. _load_pipeline()을 우회한다."""
    pipeline = StyleUnificationPipeline(config, device=torch.device("cpu"))
    pipeline._loaded = True

    mock_pipe_output = MagicMock()
    mock_pipe_output.images = [_make_rgba_image(color=(0, 0, 255, 255)).convert("RGB")]

    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_pipe_output
    pipeline.pipe = mock_pipe

    return pipeline


def _patch_preprocess(pipeline: StyleUnificationPipeline, lineart_size: tuple = (64, 64)) -> None:
    """_preprocess_source를 mock으로 교체해 모델 로딩 없이 전처리를 우회한다."""
    foreground = _make_rgba_image(color=(255, 0, 0, 255))
    lineart = Image.new("RGB", lineart_size, (0, 0, 0))
    mask = np.ones((64, 64), dtype=np.float32)
    pipeline._preprocess_source = MagicMock(return_value=(foreground, lineart, mask))


# ---------------------------------------------------------------------------
# C2: attribute_mode 라우팅
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attribute_mode,expected_cn_scale",
    [
        ("palette", 1.2),
        ("lineart", 0.6),
        ("shading", 1.2),
        ("full", 0.9),
    ],
)
def test_transform_attribute_mode_controlnet_scale(
    minimal_config, attribute_mode, expected_cn_scale
):
    """attribute_mode별로 controlnet_conditioning_scale이 preset 값과 일치하는가."""
    config = {
        **_MINIMAL_CONFIG,
        "model": {
            **_MINIMAL_CONFIG["model"],
            "controlnet": [{"enabled": True, "scale": 0.9}],
        },
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.palette_quantize.quantize_to_palette") as mock_quantize,
        patch("src.preprocessing.palette.extract_palette") as mock_palette,
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        mock_alpha.return_value = _make_rgba_image()
        mock_quantize.side_effect = lambda img, palette, strength: img
        mock_palette.return_value = np.zeros((12, 3), dtype=np.uint8)

        pipeline.transform(source, reference, attribute_mode=attribute_mode)

    call_kwargs = pipeline.pipe.call_args.kwargs
    assert call_kwargs["controlnet_conditioning_scale"] == pytest.approx(expected_cn_scale)


def test_transform_attribute_mode_selective_uses_route_scales(minimal_config):
    """attribute_mode='selective'이면 route_scales가 호출되는가."""
    config = {
        **_MINIMAL_CONFIG,
        "model": {
            **_MINIMAL_CONFIG["model"],
            "controlnet": [{"enabled": True, "scale": 0.9}],
        },
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.attribute_control.route_scales") as mock_route,
        patch("src.postprocessing.palette_quantize.quantize_to_palette") as mock_quantize,
        patch("src.preprocessing.palette.extract_palette") as mock_palette,
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        from src.postprocessing.attribute_control import AttributeScales

        fake_scales = AttributeScales(
            ip_adapter_scale=0.5,
            controlnet_lineart_scale=0.8,
            controlnet_depth_scale=0.5,
            palette_quantize_strength=0.5,
            shading_match_strength=0.0,
            lineart_postprocess=False,
        )
        mock_route.return_value = fake_scales
        mock_alpha.return_value = _make_rgba_image()
        mock_quantize.side_effect = lambda img, palette, strength: img
        mock_palette.return_value = np.zeros((12, 3), dtype=np.uint8)

        pipeline.transform(source, reference, attribute_mode="selective")

    mock_route.assert_called_once()
    assert mock_route.call_args.kwargs["mode"] == "selective"


def test_transform_scales_direct_skips_route_scales(minimal_config):
    """scales를 직접 전달하면 route_scales가 호출되지 않는가."""
    from src.postprocessing.attribute_control import AttributeScales

    config = {
        **_MINIMAL_CONFIG,
        "model": {
            **_MINIMAL_CONFIG["model"],
            "controlnet": [{"enabled": True, "scale": 0.9}],
        },
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))
    direct_scales = AttributeScales(
        ip_adapter_scale=0.5,
        controlnet_lineart_scale=0.75,
        controlnet_depth_scale=0.5,
        palette_quantize_strength=0.5,
        shading_match_strength=0.0,
        lineart_postprocess=False,
    )

    with (
        patch("src.postprocessing.attribute_control.route_scales") as mock_route,
        patch("src.postprocessing.palette_quantize.quantize_to_palette") as mock_quantize,
        patch("src.preprocessing.palette.extract_palette") as mock_palette,
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        mock_alpha.return_value = _make_rgba_image()
        mock_quantize.side_effect = lambda img, palette, strength: img
        mock_palette.return_value = np.zeros((12, 3), dtype=np.uint8)

        pipeline.transform(source, reference, scales=direct_scales)

    mock_route.assert_not_called()
    call_kwargs = pipeline.pipe.call_args.kwargs
    assert call_kwargs["controlnet_conditioning_scale"] == pytest.approx(0.75)


def test_transform_no_scales_no_attribute_mode_uses_config_default(minimal_config):
    """scales=None, attribute_mode=None이면 config 기본 scale이 사용되는가 (하위 호환)."""
    config = {
        **_MINIMAL_CONFIG,
        "model": {
            **_MINIMAL_CONFIG["model"],
            "controlnet": [{"enabled": True, "scale": 1.1}],
        },
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
        patch("src.postprocessing.palette_quantize.quantize_to_palette") as mock_quantize,
        patch("src.preprocessing.palette.extract_palette") as mock_palette,
    ):
        mock_alpha.return_value = _make_rgba_image()
        mock_quantize.side_effect = lambda img, palette, strength: img
        mock_palette.return_value = np.zeros((12, 3), dtype=np.uint8)

        pipeline.transform(source, reference)

    call_kwargs = pipeline.pipe.call_args.kwargs
    assert call_kwargs["controlnet_conditioning_scale"] == pytest.approx(1.1)


def test_transform_attribute_mode_palette_ip_adapter_scale(minimal_config):
    """attribute_mode='palette' → ip_adapter_scale이 preset 값(0.3)과 일치하는가."""
    config = {
        **_MINIMAL_CONFIG,
        "model": {
            **_MINIMAL_CONFIG["model"],
            "controlnet": [],
        },
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
        patch("src.postprocessing.palette_quantize.quantize_to_palette") as mock_quantize,
        patch("src.preprocessing.palette.extract_palette") as mock_palette,
    ):
        mock_alpha.return_value = _make_rgba_image()
        mock_quantize.side_effect = lambda img, palette, strength: img
        mock_palette.return_value = np.zeros((12, 3), dtype=np.uint8)

        pipeline.transform(source, reference, attribute_mode="palette")

    call_kwargs = pipeline.pipe.call_args.kwargs
    assert call_kwargs["ip_adapter_scale"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# C3: region_mask 통합
# ---------------------------------------------------------------------------


def test_transform_region_mask_left_half_one_right_half_zero():
    """mask 좌반 1.0 / 우반 0.0 → 좌반이 transformed 색, 우반이 source 색인가."""
    config = {
        **_MINIMAL_CONFIG,
        "postprocessing": {"palette_quantize": False},
    }
    pipeline = _make_loaded_pipeline(config)

    # source: 전체 빨강 RGBA
    source = _make_rgba_image(w=64, h=64, color=(255, 0, 0, 255))
    # mock 파이프 출력: 전체 파랑 RGB
    blue_rgb = _make_rgba_image(w=64, h=64, color=(0, 0, 255, 255)).convert("RGB")
    pipeline.pipe.return_value.images = [blue_rgb]

    # 좌반 1.0, 우반 0.0 mask
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[:, :32] = 1.0  # 좌반

    reference = _make_rgba_image(color=(128, 128, 128, 255))
    foreground_rgba = _make_rgba_image(w=64, h=64, color=(255, 0, 0, 255))
    lineart = Image.new("RGB", (64, 64), (0, 0, 0))
    fg_mask = np.ones((64, 64), dtype=np.float32)
    pipeline._preprocess_source = MagicMock(return_value=(foreground_rgba, lineart, fg_mask))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        # restore_alpha 출력: 전체 파랑 RGBA
        blue_rgba = _make_rgba_image(w=64, h=64, color=(0, 0, 255, 255))
        mock_alpha.return_value = blue_rgba

        result = pipeline.transform(source, reference, region_mask=mask)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"

    result_arr = np.array(result)
    # 우반 (0인 영역): source 빨강 이어야 함
    right_mean_r = result_arr[:, 32:, 0].mean()
    right_mean_b = result_arr[:, 32:, 2].mean()
    assert right_mean_r > right_mean_b, "우반은 source(빨강)가 우세해야 함"

    # 좌반 (1인 영역): transformed 파랑 이어야 함
    left_mean_r = result_arr[:, :32, 0].mean()
    left_mean_b = result_arr[:, :32, 2].mean()
    assert left_mean_b > left_mean_r, "좌반은 transformed(파랑)가 우세해야 함"


def test_transform_region_mask_shape_mismatch_auto_resizes():
    """mask shape이 result와 다르면 자동 리사이즈되어 변환이 성공하는가.

    이전엔 ValueError를 던졌으나, Gradio ImageEditor canvas와 source 8-multiple
    보정 후 result 해상도가 거의 항상 불일치하므로 자동 리사이즈로 동작 변경.
    """
    config = {
        **_MINIMAL_CONFIG,
        "postprocessing": {"palette_quantize": False},
    }
    pipeline = _make_loaded_pipeline(config)

    source = _make_rgba_image(w=64, h=64, color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))
    wrong_mask = np.ones((32, 32), dtype=np.float32)  # 크기 불일치 — auto-resize 대상

    blue_rgb = _make_rgba_image(w=64, h=64, color=(0, 0, 255, 255)).convert("RGB")
    pipeline.pipe.return_value.images = [blue_rgb]

    foreground_rgba = _make_rgba_image(w=64, h=64, color=(255, 0, 0, 255))
    lineart = Image.new("RGB", (64, 64), (0, 0, 0))
    fg_mask = np.ones((64, 64), dtype=np.float32)
    pipeline._preprocess_source = MagicMock(return_value=(foreground_rgba, lineart, fg_mask))

    with patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha:
        mock_alpha.return_value = _make_rgba_image(w=64, h=64, color=(0, 0, 255, 255))

        result = pipeline.transform(source, reference, region_mask=wrong_mask)

    # auto-resize 후 변환이 성공해야 한다
    assert result.size == (64, 64)
    assert result.mode == "RGBA"


def test_transform_region_mask_none_skips_apply_region_mask(minimal_config):
    """region_mask=None이면 apply_region_mask가 호출되지 않는가."""
    config = {
        **_MINIMAL_CONFIG,
        "postprocessing": {"palette_quantize": False},
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
        patch("src.postprocessing.region_mask.apply_region_mask") as mock_apply_rm,
    ):
        mock_alpha.return_value = _make_rgba_image()

        pipeline.transform(source, reference, region_mask=None)

    mock_apply_rm.assert_not_called()


def test_transform_result_is_rgba(minimal_config):
    """transform() 결과가 RGBA PIL Image인가 (region_mask 유무 무관)."""
    config = {
        **_MINIMAL_CONFIG,
        "postprocessing": {"palette_quantize": False},
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        mock_alpha.return_value = _make_rgba_image()

        result = pipeline.transform(source, reference)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"


def test_transform_signature_backward_compat(minimal_config):
    """기존 transform(source, reference, seed=42) 시그니처가 그대로 동작하는가."""
    config = {
        **_MINIMAL_CONFIG,
        "postprocessing": {"palette_quantize": False},
    }
    pipeline = _make_loaded_pipeline(config)
    _patch_preprocess(pipeline)

    source = _make_rgba_image(color=(255, 0, 0, 255))
    reference = _make_rgba_image(color=(128, 128, 128, 255))

    with (
        patch("src.postprocessing.alpha_restore.restore_alpha") as mock_alpha,
    ):
        mock_alpha.return_value = _make_rgba_image()

        result = pipeline.transform(source, reference, seed=42)

    assert isinstance(result, Image.Image)
