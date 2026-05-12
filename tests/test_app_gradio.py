"""build_app() 단위 테스트.

gradio는 이미 설치되어 있으므로 실제 gradio를 사용해 build_app()을 호출하고,
모델 로딩이 트리거되지 않음을 검증한다.
"""

from __future__ import annotations

from unittest.mock import patch

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
