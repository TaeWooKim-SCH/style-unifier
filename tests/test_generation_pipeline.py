"""StyleUnificationPipeline + load_config() 단위 테스트.

모델 로딩을 mock으로 우회하고, lazy 가드와 config 검증 로직을 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.generation.pipeline import StyleUnificationPipeline, load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path("/Users/atrocom/Desktop/style-unifier/configs/default.yaml")

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
