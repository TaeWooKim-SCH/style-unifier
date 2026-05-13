"""Gradio UI: Single / Batch 탭 레이아웃으로 스타일 변환 기능을 노출한다.

Single Tab: 단일 source → 단일 output (D2 속성 제어, D3 region mask 포함).
Batch Tab: 다중 source → 다중 output (D1 일관성, D4 다중 ref, D5 shared attention).

모델 로딩은 첫 Transform 클릭 시 지연(lazy)된다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _patch_gradio_client_bool_schema() -> None:
    """gradio_client 1.3.0의 boolean schema 처리 버그 우회.

    JSON Schema 표준상 ``additionalProperties`` 는 dict 또는 bool(True/False)일 수 있으나,
    gradio_client 1.3.0의 ``_json_schema_to_python_type``은 bool을 dict로 가정하고
    ``get_type(True)``를 호출 → ``"const" in True`` 에서 ``TypeError``.
    gradio_client 1.4+에서 수정됐으나 gradio 4.44.1이 1.3.0을 정확히 핀하므로 런타임 패치한다.
    """
    import gradio_client.utils as _gcu

    _original = _gcu._json_schema_to_python_type

    def _patched(schema, defs):  # type: ignore[no-untyped-def]
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _original(schema, defs)

    _gcu._json_schema_to_python_type = _patched


_patch_gradio_client_bool_schema()

import gradio as gr
import numpy as np
from PIL import Image

from src.generation import StyleUnificationPipeline, load_config
from src.utils import get_logger

logger = get_logger(__name__)

# config 경로를 키로 파이프라인 인스턴스를 캐시한다.
_pipeline_cache: dict[str, StyleUnificationPipeline] = {}


def _get_or_create_pipeline(config_path: str) -> StyleUnificationPipeline:
    """Config 경로별 파이프라인 인스턴스를 캐시해 반환한다.

    캐시에 없으면 새 인스턴스를 생성해 저장한다.
    모델 로딩 자체는 StyleUnificationPipeline의 lazy 전략에 따라
    첫 transform() 호출 시에 발생한다.

    Args:
        config_path: YAML 설정 파일 경로 문자열.

    Returns:
        캐시된 또는 새로 생성된 StyleUnificationPipeline 인스턴스.

    Raises:
        FileNotFoundError: config_path가 존재하지 않을 때.
        KeyError: 필수 config 키가 없을 때.
    """
    if config_path not in _pipeline_cache:
        logger.info("Creating new pipeline for config: %s", config_path)
        cfg = load_config(Path(config_path))
        _pipeline_cache[config_path] = StyleUnificationPipeline(cfg)
    return _pipeline_cache[config_path]


def _evict_pipeline(config_path: str) -> None:
    """캐시에서 파이프라인을 제거하고 모델을 메모리에서 해제한다.

    Args:
        config_path: 제거할 파이프라인의 config 경로 문자열.
    """
    if config_path in _pipeline_cache:
        logger.info("Evicting pipeline for config: %s", config_path)
        _pipeline_cache[config_path].unload()
        del _pipeline_cache[config_path]


def _extract_region_mask_from_editor(
    region_mask_edit: dict | None,
) -> np.ndarray | None:
    """gr.ImageEditor 반환값에서 region mask를 추출한다.

    ``region_mask_edit`` 은 Gradio ImageEditor가 반환하는 dict:
    ``{"background": ..., "layers": [ndarray, ...], "composite": ...}``.
    layers[0]의 alpha 채널(4번째 채널)을 [0, 1] float32로 정규화한다.

    Args:
        region_mask_edit: ImageEditor 반환 dict 또는 None.

    Returns:
        (H, W) float32 [0, 1] mask 또는 None (layers 비어있거나 None 입력).
    """
    if region_mask_edit is None:
        return None
    layers = region_mask_edit.get("layers")
    if not layers:
        return None
    layer0: np.ndarray = layers[0]
    if layer0 is None or layer0.size == 0:
        return None
    # layers[0]은 (H, W, 4) RGBA ndarray.
    # CRITICAL: Gradio ImageEditor는 사용자가 brush를 안 칠려도 layers에 빈 RGBA layer
    # (alpha 전부 0)을 반환한다. 그 빈 layer를 valid mask로 처리하면 apply_region_mask가
    # mask=0 영역에 source를 유지 → 전체 결과가 source와 동일해진다.
    # alpha의 합이 0이면(= 아무 stroke 없음) 의도 없는 것으로 보고 None 반환.
    if layer0.ndim == 3 and layer0.shape[2] >= 4:
        alpha = layer0[..., 3].astype(np.float32) / 255.0
        if float(alpha.sum()) == 0.0:
            return None
        return alpha
    # RGBA 아닌 경우 — 전체 1 mask 반환 (user가 그렸다고 가정)
    return np.ones(layer0.shape[:2], dtype=np.float32)


def transform_handler(
    source: Image.Image | None,
    reference: Image.Image | None,
    seed: float,
    prompt: str,
    config_path: str,
    attribute_mode: str,
    region_mask_edit: dict | None,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> Image.Image:
    """Single Tab Transform 버튼 클릭 시 호출되는 Gradio 핸들러.

    Args:
        source: 변환할 원본 게임 에셋 PIL Image.
        reference: 스타일 reference PIL Image.
        seed: 재현성 seed. 음수이면 랜덤(None) 으로 처리.
        prompt: 프롬프트 오버라이드 문자열. 빈 문자열이면 config 기본값 사용.
        config_path: YAML 설정 파일 경로 문자열.
        attribute_mode: D2 속성 모드. "full"이면 None으로 처리.
        region_mask_edit: D3 ImageEditor 반환 dict. layers 없으면 None으로 처리.
        progress: Gradio Progress 객체.

    Returns:
        RGBA PIL Image.

    Raises:
        gr.Error: 이미지 미제공, 파일 미발견, 변환 실패 등.
    """
    if source is None or reference is None:
        raise gr.Error("source와 reference 이미지 모두 필요합니다.")

    progress(0.05, desc="설정 로드 중...")
    config_path = config_path.strip()
    if not config_path:
        raise gr.Error("config 경로가 비어 있습니다.")

    if not Path(config_path).exists():
        raise gr.Error(f"config 파일을 찾을 수 없습니다: {config_path}")

    try:
        progress(0.1, desc="파이프라인 준비 중 (처음 실행 시 모델 로드가 시작됩니다)...")
        pipe = _get_or_create_pipeline(config_path)
    except FileNotFoundError as exc:
        logger.error("Config not found: %s", exc)
        raise gr.Error(f"설정 파일을 찾을 수 없습니다: {exc}") from exc
    except KeyError as exc:
        logger.error("Config key missing: %s", exc)
        raise gr.Error(f"설정 파일에 필수 키가 없습니다: {exc}") from exc

    progress(0.3, desc="변환 중 (모델 로드 포함 수십 초~수 분 소요될 수 있습니다)...")
    seed_arg: int | None = None if int(seed) < 0 else int(seed)
    prompt_arg: str | None = prompt.strip() or None
    attr_mode: str | None = attribute_mode if attribute_mode != "full" else None
    region_mask = _extract_region_mask_from_editor(region_mask_edit)

    logger.debug(
        "transform_handler: attribute_mode=%r, region_mask=%s",
        attr_mode,
        "set" if region_mask is not None else "None",
    )

    try:
        result = pipe.transform(
            source,
            reference,
            seed=seed_arg,
            prompt=prompt_arg,
            attribute_mode=attr_mode,
            region_mask=region_mask,
        )
    except OSError as exc:
        logger.error("Model load/IO error during transform: %s", exc)
        raise gr.Error(f"모델 로드 또는 IO 오류: {exc}") from exc
    except RuntimeError as exc:
        logger.error("Runtime error during transform: %s", exc)
        raise gr.Error(f"변환 중 런타임 오류: {exc}") from exc

    progress(1.0, desc="완료")
    if isinstance(result, dict):
        return result["result"]  # type: ignore[return-value]
    return result  # type: ignore[return-value]


def on_config_change(new_config: str, old_config: str) -> str:
    """Config 경로 변경 시 이전 파이프라인을 메모리에서 해제한다.

    Args:
        new_config: 새 config 경로 문자열.
        old_config: 이전 config 경로 문자열.

    Returns:
        new_config 그대로 (Gradio State 업데이트용).
    """
    old_config = old_config.strip()
    if old_config and old_config != new_config.strip():
        _evict_pipeline(old_config)
    return new_config.strip()


def build_app(default_config: str = "configs/default.yaml") -> gr.Blocks:
    """Gradio Blocks UI 인스턴스를 빌드해 반환한다.

    이 함수는 모델을 로드하지 않는다. 모델 로딩은 첫 Transform 클릭 시 발생.

    Args:
        default_config: 초기 config 경로. argparse --config 값이 전달된다.

    Returns:
        구성된 gr.Blocks 앱 인스턴스.
    """
    from app.gradio_batch_tab import build_batch_tab

    with gr.Blocks(title="Style Unifier") as app:
        gr.Markdown("# Style Unifier — Reference 기반 스타일 변환")
        gr.Markdown(
            "> 처음 **Transform** 클릭 시 모델 로드(수십 초~수 분)가 발생합니다. "
            + "같은 config로 재호출 시에는 캐시된 파이프라인을 재사용해 빠릅니다."
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Single Tab (기존 UI + D2/D3 옵션)
            # ------------------------------------------------------------------
            with gr.Tab("Single"):
                gr.Markdown("### Single Transform")

                # 이전 config를 추적하기 위한 hidden state
                prev_config_state = gr.State(value=default_config)

                with gr.Row():
                    source_in = gr.Image(
                        label="Source (원본 에셋)",
                        type="pil",
                        image_mode="RGBA",
                    )
                    reference_in = gr.Image(
                        label="Reference (스타일 기준)",
                        type="pil",
                        image_mode="RGBA",
                    )
                    output_img = gr.Image(
                        label="Output (변환 결과, RGBA)",
                        type="pil",
                        image_mode="RGBA",
                        interactive=False,
                    )

                with gr.Row():
                    seed_in = gr.Number(
                        label="Seed (-1: 랜덤)",
                        value=-1,
                        precision=0,
                    )
                    prompt_in = gr.Textbox(
                        label="Prompt override (빈 값: config 기본 프롬프트 사용)",
                        value="",
                        placeholder="예: high quality game asset, clean vector illustration",
                    )

                with gr.Row():
                    config_in = gr.Textbox(
                        label="Config path",
                        value=default_config,
                        placeholder="configs/default.yaml",
                    )

                # D2: 속성 모드 선택
                with gr.Row():
                    attribute_mode_radio = gr.Radio(
                        choices=["full", "palette", "lineart", "shading", "selective"],
                        value="full",
                        label="Attribute Mode (D2)",
                    )

                # D3: Region Mask 에디터
                with gr.Row():
                    region_mask_editor = gr.ImageEditor(
                        label="Region Mask (D3, optional — 빨간 영역만 변환, 나머지는 원본 보존)",
                        type="numpy",
                        brush=gr.Brush(
                            default_size=24,
                            colors=["#FF0000"],
                            color_mode="fixed",
                        ),
                    )

                run_btn = gr.Button("Transform", variant="primary")

                # config 변경 시 이전 파이프라인 해제
                config_in.change(
                    fn=on_config_change,
                    inputs=[config_in, prev_config_state],
                    outputs=[prev_config_state],
                )

                run_btn.click(
                    fn=transform_handler,
                    inputs=[
                        source_in,
                        reference_in,
                        seed_in,
                        prompt_in,
                        config_in,
                        attribute_mode_radio,
                        region_mask_editor,
                    ],
                    outputs=[output_img],
                )

            # ------------------------------------------------------------------
            # Batch Tab (D1 / D4 / D5)
            # ------------------------------------------------------------------
            with gr.Tab("Batch"):
                build_batch_tab(default_config=default_config)

    return app


def main() -> None:
    """CLI 진입점. 인자를 파싱하고 Gradio 앱을 실행한다."""
    parser = argparse.ArgumentParser(
        description="Style Unifier Gradio UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="초기 config YAML 파일 경로",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Gradio 공개 링크 생성 (share=True)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="서버 포트",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="서버 호스트",
    )

    args = parser.parse_args()

    logger.info(
        "Launching Style Unifier UI — config=%s, host=%s, port=%d, share=%s",
        args.config,
        args.host,
        args.port,
        args.share,
    )

    demo = build_app(default_config=args.config)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
