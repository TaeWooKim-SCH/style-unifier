"""Gradio 최소 UI: source/reference 입력 → 변환 결과 출력.

3패널 레이아웃(source · reference · output)과 seed/prompt/config 컨트롤을
제공한다. 모델 로딩은 첫 Transform 클릭 시 지연(lazy)된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
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


def transform_handler(
    source: Image.Image | None,
    reference: Image.Image | None,
    seed: float,
    prompt: str,
    config_path: str,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> Image.Image:
    """Transform 버튼 클릭 시 호출되는 Gradio 핸들러.

    Args:
        source: 변환할 원본 게임 에셋 PIL Image.
        reference: 스타일 reference PIL Image.
        seed: 재현성 seed. 음수이면 랜덤(None) 으로 처리.
        prompt: 프롬프트 오버라이드 문자열. 빈 문자열이면 config 기본값 사용.
        config_path: YAML 설정 파일 경로 문자열.
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

    try:
        result = pipe.transform(source, reference, seed=seed_arg, prompt=prompt_arg)
    except OSError as exc:
        logger.error("Model load/IO error during transform: %s", exc)
        raise gr.Error(f"모델 로드 또는 IO 오류: {exc}") from exc
    except RuntimeError as exc:
        logger.error("Runtime error during transform: %s", exc)
        raise gr.Error(f"변환 중 런타임 오류: {exc}") from exc

    progress(1.0, desc="완료")
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
    with gr.Blocks(title="Style Unifier") as app:
        gr.Markdown("# Style Unifier — 단일 reference 스타일 변환")
        gr.Markdown(
            "> 처음 **Transform** 클릭 시 모델 로드(수십 초~수 분)가 발생합니다. "
            + "같은 config로 재호출 시에는 캐시된 파이프라인을 재사용해 빠릅니다."
        )

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

        run_btn = gr.Button("Transform", variant="primary")

        # config 변경 시 이전 파이프라인 해제
        config_in.change(
            fn=on_config_change,
            inputs=[config_in, prev_config_state],
            outputs=[prev_config_state],
        )

        run_btn.click(
            fn=transform_handler,
            inputs=[source_in, reference_in, seed_in, prompt_in, config_in],
            outputs=[output_img],
        )

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
    )


if __name__ == "__main__":
    main()
