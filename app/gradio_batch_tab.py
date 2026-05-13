"""Batch Tab UI 빌더 및 핸들러 (D1 / D4 / D5 기능 노출).

``build_batch_tab()`` 이 Batch Tab 컴포넌트를 생성하고,
``transform_batch_handler()`` 가 Transform Batch 버튼 클릭 시 호출된다.

이 모듈은 ``app.gradio_app`` 에서 import해 ``gr.Tabs`` 안에 배치된다.
직접 실행하지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_share_layers(text: str) -> list[int] | None:
    """share_layers 텍스트 입력을 파싱해 정수 리스트 또는 None으로 반환한다.

    Args:
        text: 쉼표 구분 인덱스 문자열 (예: "0,1,2"). 빈 문자열이면 None 반환.

    Returns:
        정수 리스트 또는 None (빈 입력).

    Raises:
        ValueError: 숫자가 아닌 값이 포함된 경우.
    """
    text = text.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_pil_from_file(file_obj: object) -> Image.Image:
    """Gradio Files 컴포넌트가 반환한 파일 객체에서 PIL Image를 로드한다.

    Gradio 4.31의 Files 컴포넌트는 ``NamedString`` (str 서브클래스)을 반환하므로
    ``str()`` 으로 경로를 추출한다.

    Args:
        file_obj: Gradio Files 반환값 원소 (NamedString 또는 str).

    Returns:
        RGBA PIL Image.
    """
    path = Path(str(file_obj))
    return Image.open(path).convert("RGBA")


def _load_sources_from_files(sources_files: list) -> list[Image.Image]:
    """Gradio Files 컴포넌트 반환값에서 source PIL Image 리스트를 로드한다.

    Args:
        sources_files: Gradio Files 반환 리스트 (NamedString 또는 str 원소).

    Returns:
        RGBA PIL Image 리스트.

    Raises:
        gr.Error: 개별 파일 로드 실패 시.
    """
    sources: list[Image.Image] = []
    for fobj in sources_files:
        try:
            sources.append(_load_pil_from_file(fobj))
        except OSError as exc:
            raise gr.Error(f"Source 파일 로드 실패: {exc}") from exc
    return sources


def _load_references_from_files(
    references_files: list | None,
) -> list[Image.Image] | None:
    """Gradio Files 컴포넌트 반환값에서 reference PIL Image 리스트를 로드한다.

    Args:
        references_files: Gradio Files 반환 리스트 또는 None.

    Returns:
        RGBA PIL Image 리스트 또는 None (입력이 None이거나 빈 리스트인 경우).

    Raises:
        gr.Error: 개별 파일 로드 실패 시.
    """
    if not references_files:
        return None
    references: list[Image.Image] = []
    for fobj in references_files:
        try:
            references.append(_load_pil_from_file(fobj))
        except OSError as exc:
            raise gr.Error(f"Reference 파일 로드 실패: {exc}") from exc
    return references


def transform_batch_handler(
    sources_files: list | None,
    reference_in: Image.Image | None,
    references_files: list | None,
    enforce_consistency: bool,
    attribute_mode: str,
    seed_batch: float,
    prompt_batch: str,
    config_path: str,
    shared_attention: bool,
    share_layers_text: str,
    progress: gr.Progress = gr.Progress(),  # noqa: B008
) -> list[tuple[Image.Image, str]]:
    """Transform Batch 버튼 클릭 시 호출되는 Gradio 핸들러.

    Args:
        sources_files: Gradio Files 컴포넌트 반환 리스트 (파일 경로 목록).
        reference_in: 단일 reference PIL Image.
        references_files: 다중 reference 파일 리스트 (D4, optional).
        enforce_consistency: D1 배치 일관성 후처리 여부.
        attribute_mode: D2 속성 모드 문자열.
        seed_batch: 배치 기준 seed. 음수이면 None.
        prompt_batch: 프롬프트 오버라이드. 빈 문자열이면 None.
        config_path: YAML 설정 파일 경로.
        shared_attention: D5 shared attention 사용 여부 (ADR-010).
        share_layers_text: 공유 layer 인덱스 쉼표 구분 문자열.
        progress: Gradio Progress 객체.

    Returns:
        Gallery가 표시할 ``(PIL Image, 캡션)`` 튜플 리스트.

    Raises:
        gr.Error: 입력 미제공, 파일 미발견, 변환 실패 등.
    """
    if not sources_files:
        raise gr.Error("Source 이미지를 하나 이상 업로드해주세요.")
    if reference_in is None:
        raise gr.Error("Reference 이미지를 제공해주세요.")

    config_path = config_path.strip()
    if not config_path:
        raise gr.Error("config 경로가 비어 있습니다.")
    if not Path(config_path).exists():
        raise gr.Error(f"config 파일을 찾을 수 없습니다: {config_path}")

    progress(0.05, desc="설정 로드 중...")

    from app.gradio_app import _get_or_create_pipeline

    try:
        pipe = _get_or_create_pipeline(config_path)
    except FileNotFoundError as exc:
        logger.error("Config not found: %s", exc)
        raise gr.Error(f"설정 파일을 찾을 수 없습니다: {exc}") from exc
    except KeyError as exc:
        logger.error("Config key missing: %s", exc)
        raise gr.Error(f"설정 파일에 필수 키가 없습니다: {exc}") from exc

    progress(0.1, desc="Source 이미지 로드 중...")
    sources = _load_sources_from_files(sources_files)
    references = _load_references_from_files(references_files)

    try:
        share_layers = _parse_share_layers(share_layers_text)
    except ValueError as exc:
        raise gr.Error(f"share_layers 파싱 실패: {exc}") from exc

    seed_int: int | None = None if int(seed_batch) < 0 else int(seed_batch)
    if prompt_batch.strip():
        # transform_batch는 현재 prompt 파라미터를 미지원 — Linux 실측 후 보강 예정.
        logger.warning("transform_batch_handler: prompt_batch는 현재 미지원. 무시됩니다.")
    attr_mode: str | None = attribute_mode if attribute_mode != "full" else None

    progress(0.2, desc=f"배치 변환 시작 ({len(sources)}장)...")
    logger.info(
        "transform_batch_handler: N=%d, consistency=%s, shared_attn=%s, seed=%s, mode=%s",
        len(sources),
        enforce_consistency,
        shared_attention,
        seed_int,
        attr_mode,
    )

    try:
        results: list[Image.Image] = pipe.transform_batch(
            sources,
            reference_in,
            references=references or None,
            enforce_consistency=enforce_consistency,
            use_shared_attention=shared_attention,
            scales=None,
            attribute_mode=attr_mode,
            seed=seed_int,
            share_layers=share_layers,
        )
    except ValueError as exc:
        logger.error("Validation error during transform_batch: %s", exc)
        raise gr.Error(f"입력 검증 실패: {exc}") from exc
    except OSError as exc:
        logger.error("Model load/IO error during transform_batch: %s", exc)
        raise gr.Error(f"모델 로드 또는 IO 오류: {exc}") from exc
    except RuntimeError as exc:
        logger.error("Runtime error during transform_batch: %s", exc)
        raise gr.Error(f"배치 변환 중 런타임 오류: {exc}") from exc

    progress(1.0, desc="완료")
    return [(img, f"output_{i}") for i, img in enumerate(results)]


def build_batch_tab(default_config: str = "configs/default.yaml") -> None:
    """Batch Tab 컴포넌트를 빌드한다.

    이 함수는 ``gr.Tab("Batch")`` 컨텍스트 내부에서 호출되어야 한다.
    컴포넌트를 생성하고 버튼에 이벤트 핸들러를 연결한다.

    Args:
        default_config: 초기 config 경로.
    """
    gr.Markdown("### Batch Transform (D1 / D4 / D5)")
    gr.Markdown("> 여러 Source 에셋을 한 번에 변환합니다. D1·D4·D5 기능을 토글합니다.")

    with gr.Row():
        sources_files = gr.Files(
            label="Source Images (multiple)",
            file_count="multiple",
            file_types=["image"],
        )
        reference_in_batch = gr.Image(
            label="Reference (single)",
            type="pil",
            image_mode="RGBA",
        )

    with gr.Row():
        references_files = gr.Files(
            label="Additional References (D4 multi-ref, optional)",
            file_count="multiple",
            file_types=["image"],
        )
        gr.Markdown("Multi-ref strategy: v1에서 'mean' 고정. 추후 weighted/first 지원 예정.")

    with gr.Row():
        enforce_consistency_check = gr.Checkbox(
            value=True,
            label="Enforce Batch Consistency (D1 post-process)",
        )
        attribute_mode_batch = gr.Radio(
            choices=["full", "palette", "lineart", "shading", "selective"],
            value="full",
            label="Attribute Mode (D2)",
        )

    with gr.Row():
        seed_batch = gr.Number(
            value=42,
            label="Seed (base; each source uses seed+i; -1 for random)",
            precision=0,
        )
        prompt_batch = gr.Textbox(
            label="Prompt override (optional)",
            value="",
            placeholder="예: high quality game asset, clean vector illustration",
        )

    with gr.Row():
        config_path_batch = gr.Textbox(
            value=default_config,
            label="Config path",
            placeholder="configs/default.yaml",
        )

    with gr.Accordion("Experimental (D5 — ADR-010)", open=False):
        gr.Markdown(
            "> **경고**: D5 Shared Attention은 검증 단계 시도입니다 (ADR-010). RTX 4070 전용."
        )
        shared_attention_check = gr.Checkbox(
            value=False,
            label="Use Shared Attention (D5, RTX 4070 only, ADR-010)",
        )
        share_layers_text = gr.Textbox(
            value="",
            label="Share layers (comma-separated indices, empty = all)",
            placeholder="예: 0,1,2,3",
        )

    batch_button = gr.Button("Transform Batch", variant="primary")
    outputs_gallery = gr.Gallery(
        label="Outputs",
        columns=4,
        # height=None: Gradio 4.31에서 "auto"는 미지원 — None으로 자동 결정
        height=None,
    )

    batch_button.click(
        fn=transform_batch_handler,
        inputs=[
            sources_files,
            reference_in_batch,
            references_files,
            enforce_consistency_check,
            attribute_mode_batch,
            seed_batch,
            prompt_batch,
            config_path_batch,
            shared_attention_check,
            share_layers_text,
        ],
        outputs=[outputs_gallery],
    )
