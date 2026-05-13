"""build_app() 단위 테스트.

gradio는 이미 설치되어 있으므로 실제 gradio를 사용해 build_app()을 호출하고,
모델 로딩이 트리거되지 않음을 검증한다.
C5: Batch Tab 컴포넌트 존재 + handler 검증.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# build_app() 기본 동작
# ---------------------------------------------------------------------------


def test_build_app_returns_gradio_blocks_instance():
    """build_app() 반환값이 gr.Blocks 인스턴스인가."""
    import gradio as gr
    from app.gradio_app import build_app

    result = build_app()
    assert isinstance(result, gr.Blocks)


def test_build_app_does_not_trigger_pipeline_init():
    """build_app() 호출 동안 StyleUnificationPipeline.__init__이 호출되지 않는가."""
    from app.gradio_app import build_app

    from src.generation.pipeline import StyleUnificationPipeline

    with patch.object(
        StyleUnificationPipeline,
        "__init__",
        return_value=None,
    ) as mock_init:
        build_app()
        mock_init.assert_not_called()


def test_build_app_does_not_call_load_config():
    """build_app() 호출 동안 load_config가 호출되지 않는가."""
    import app.gradio_app as gradio_app_module
    from app.gradio_app import build_app

    with patch.object(gradio_app_module, "load_config") as mock_load_config:
        build_app()
        mock_load_config.assert_not_called()


# ---------------------------------------------------------------------------
# C5: Batch Tab 컴포넌트 존재 검증
# ---------------------------------------------------------------------------


def _collect_components(blocks, target_type):
    """gr.Blocks 트리를 DFS 순회해 target_type 컴포넌트 리스트를 반환한다."""

    found = []

    def _walk(obj):
        if isinstance(obj, target_type):
            found.append(obj)
        # gr.Blocks / Tabs / Tab / Row / Column / Accordion 등은 children을 가짐
        for child in getattr(obj, "children", []) or []:
            _walk(child)

    _walk(blocks)
    return found


def test_build_app_has_single_and_batch_tabs():
    """build_app() 결과에 'Single' Tab과 'Batch' Tab이 모두 존재하는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    tabs = _collect_components(app, gr.Tab)
    labels = [getattr(t, "label", None) for t in tabs]
    assert "Single" in labels
    assert "Batch" in labels


def test_batch_tab_has_files_components():
    """Batch Tab에 gr.Files 컴포넌트가 최소 2개(sources + additional refs) 있는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    files_comps = _collect_components(app, gr.Files)
    assert len(files_comps) >= 2


def test_batch_tab_has_gallery():
    """Batch Tab에 gr.Gallery 컴포넌트가 존재하는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    galleries = _collect_components(app, gr.Gallery)
    assert len(galleries) >= 1


def test_batch_tab_has_accordion():
    """Batch Tab에 Experimental Accordion이 존재하는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    accordions = _collect_components(app, gr.Accordion)
    assert len(accordions) >= 1


def test_batch_tab_has_checkboxes():
    """Batch Tab에 gr.Checkbox 컴포넌트가 최소 2개(consistency + shared_attention) 있는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    checkboxes = _collect_components(app, gr.Checkbox)
    assert len(checkboxes) >= 2


def test_batch_tab_has_batch_transform_button():
    """Batch Tab에 'Transform Batch' 버튼이 존재하는가."""
    import gradio as gr
    from app.gradio_app import build_app

    app = build_app()
    buttons = _collect_components(app, gr.Button)
    labels = [getattr(b, "value", None) for b in buttons]
    assert any("Batch" in str(lbl) for lbl in labels if lbl)


# ---------------------------------------------------------------------------
# C5: transform_batch_handler 입력 검증 (gr.Error)
# ---------------------------------------------------------------------------


def test_transform_batch_handler_empty_sources_raises_gr_error(tmp_path):
    """sources_files가 빈 리스트이면 gr.Error가 발생하는가."""
    import gradio as gr
    from app.gradio_batch_tab import transform_batch_handler

    reference = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
    config_file = tmp_path / "test.yaml"
    config_file.write_text(
        "model:\n  base_checkpoint: x\n  ip_adapter: {repo: x, subfolder: x, weight_name: x, scale: 0.7}\n  controlnet: []\nsampling:\n  steps: 2\n  cfg_scale: 1.0\n  denoising_strength: 0.5\npreprocessing: {}\npostprocessing: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(gr.Error):
        transform_batch_handler(
            sources_files=[],
            reference_in=reference,
            references_files=None,
            enforce_consistency=True,
            attribute_mode="full",
            seed_batch=42,
            prompt_batch="",
            config_path=str(config_file),
            shared_attention=False,
            share_layers_text="",
        )


def test_transform_batch_handler_none_sources_raises_gr_error(tmp_path):
    """sources_files가 None이면 gr.Error가 발생하는가."""
    import gradio as gr
    from app.gradio_batch_tab import transform_batch_handler

    reference = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
    config_file = tmp_path / "test.yaml"
    config_file.write_text(
        "model:\n  base_checkpoint: x\n  ip_adapter: {repo: x, subfolder: x, weight_name: x, scale: 0.7}\n  controlnet: []\nsampling:\n  steps: 2\n  cfg_scale: 1.0\n  denoising_strength: 0.5\npreprocessing: {}\npostprocessing: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(gr.Error):
        transform_batch_handler(
            sources_files=None,
            reference_in=reference,
            references_files=None,
            enforce_consistency=True,
            attribute_mode="full",
            seed_batch=42,
            prompt_batch="",
            config_path=str(config_file),
            shared_attention=False,
            share_layers_text="",
        )


def test_transform_batch_handler_none_reference_raises_gr_error(tmp_path):
    """reference_in이 None이면 gr.Error가 발생하는가."""
    import gradio as gr
    from app.gradio_batch_tab import transform_batch_handler

    config_file = tmp_path / "test.yaml"
    config_file.write_text(
        "model:\n  base_checkpoint: x\n  ip_adapter: {repo: x, subfolder: x, weight_name: x, scale: 0.7}\n  controlnet: []\nsampling:\n  steps: 2\n  cfg_scale: 1.0\n  denoising_strength: 0.5\npreprocessing: {}\npostprocessing: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(gr.Error):
        transform_batch_handler(
            sources_files=["some_file.png"],
            reference_in=None,
            references_files=None,
            enforce_consistency=True,
            attribute_mode="full",
            seed_batch=42,
            prompt_batch="",
            config_path=str(config_file),
            shared_attention=False,
            share_layers_text="",
        )


# ---------------------------------------------------------------------------
# C5: _extract_region_mask_from_editor 헬퍼 검증
# ---------------------------------------------------------------------------


def test_extract_region_mask_from_editor_none_returns_none():
    """None 입력 → None 반환."""
    from app.gradio_app import _extract_region_mask_from_editor

    result = _extract_region_mask_from_editor(None)
    assert result is None


def test_extract_region_mask_from_editor_empty_layers_returns_none():
    """{'layers': []} → None 반환."""
    from app.gradio_app import _extract_region_mask_from_editor

    result = _extract_region_mask_from_editor({"layers": []})
    assert result is None


def test_extract_region_mask_from_editor_missing_layers_returns_none():
    """layers 키가 없는 dict → None 반환."""
    from app.gradio_app import _extract_region_mask_from_editor

    result = _extract_region_mask_from_editor({"background": None})
    assert result is None


def test_extract_region_mask_from_editor_rgba_layer_returns_float32_mask():
    """RGBA ndarray layer → (H, W) float32 [0, 1] 반환."""
    from app.gradio_app import _extract_region_mask_from_editor

    # (64, 64, 4) RGBA — 우반 alpha=255, 좌반 alpha=0
    layer = np.zeros((64, 64, 4), dtype=np.uint8)
    layer[:, 32:, 3] = 255  # 우반 alpha = 255 → mask = 1.0
    layer[:, :32, 3] = 0    # 좌반 alpha = 0   → mask = 0.0

    result = _extract_region_mask_from_editor({"layers": [layer]})

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (64, 64)
    assert result[:, 32:].mean() == pytest.approx(1.0)
    assert result[:, :32].mean() == pytest.approx(0.0)


def test_extract_region_mask_from_editor_alpha_normalized_0_to_1():
    """alpha=128인 픽셀이 0.5로 정규화되는가."""
    from app.gradio_app import _extract_region_mask_from_editor

    layer = np.zeros((8, 8, 4), dtype=np.uint8)
    layer[:, :, 3] = 128  # 전체 alpha = 128

    result = _extract_region_mask_from_editor({"layers": [layer]})

    assert result is not None
    assert result.mean() == pytest.approx(128 / 255.0, abs=1e-3)
