"""배치 스타일 변환 흐름 모듈 (D1 / D4 / D5 통합 진입점).

``StyleUnificationPipeline.transform_batch()`` 가 이 모듈의
``run_batch_transform()`` 에 위임한다.

흐름 개요:
1. (D4) multi-reference가 있으면 ``aggregate_references`` 로 임베딩 집계.
   v1에서는 임베딩 주입이 NotImplementedError — references[0]을 reference로 사용한다.
2. (D5) ``use_shared_attention=True`` 면 ``_shared_attention_scope`` 컨텍스트로
   shared attention을 적용한 뒤 batched denoise 시도.
   v1에서는 batched denoise 경로가 NotImplementedError — fallback으로 sequential 처리.
   실제 D5 denoise는 Linux RTX 4070 실측 후 보강 예정.
3. (D1) ``enforce_consistency=True`` 면 ``extract_style_statistics`` +
   ``enforce_consistency`` 로 배치 일관성 후처리를 적용한다.

WARNING (D5): 검증 단계 architectural approach — ADR-010.
첫 시도에 작동 보장 없음. D1 fallback이 항상 보장됨.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.generation.pipeline import StyleUnificationPipeline
    from src.postprocessing.attribute_control import AttributeScales

logger = get_logger(__name__)

# D5 batched denoise는 Linux 실측 단계에서 구현. 이 상수로 의도를 명시.
_D5_BATCHED_DENOISE_IMPLEMENTED = False


# ---------------------------------------------------------------------------
# Private context manager — shared attention scope (D5)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _shared_attention_scope(
    pipe_obj: Any,
    *,
    enabled: bool,
    share_layers: list[int] | None,
    max_batch_size: int,
) -> Generator[None, None, None]:
    """Shared attention apply/remove를 try/finally로 보장하는 컨텍스트 매니저.

    WARNING: 검증 단계 — ADR-010. ``enabled=False`` 면 즉시 yield하고 no-op.

    ``apply_shared_attention`` 이 ``pipe_obj.pipe.unet`` 을 수정하므로,
    예외 발생 시 ``remove_shared_attention`` 이 finally에서 호출되어
    다음 호출에 SharedKVAttnProcessor가 잔존하는 사고를 방지한다.

    Args:
        pipe_obj: ``StyleUnificationPipeline`` 인스턴스.
            ``pipe_obj.pipe`` 가 로드된 상태여야 한다.
        enabled: True면 shared attention 적용. False면 no-op.
        share_layers: 공유할 self-attention layer 인덱스. None이면 전체.
        max_batch_size: VRAM 상한 (RTX 4070 기준 4 권장).

    Yields:
        None. 컨텍스트 블록 실행 후 finally에서 processor를 복원한다.

    Raises:
        AttributeError: ``pipe_obj.pipe`` 가 None일 때 (모델 미로딩 상태).
        RuntimeError: ``apply_shared_attention`` 실패 시 (diffusers 버전 불일치 등).
    """
    if not enabled:
        yield
        return

    from src.encoding.shared_attention import apply_shared_attention, remove_shared_attention

    inner_pipe = pipe_obj.pipe
    if inner_pipe is None:
        raise AttributeError(
            "_shared_attention_scope: pipe_obj.pipe is None. "
            "Call pipe_obj._load_pipeline() before entering shared attention scope."
        )

    logger.info(
        "_shared_attention_scope: applying shared attention (share_layers=%s, max_batch_size=%d)",
        share_layers,
        max_batch_size,
    )
    apply_shared_attention(inner_pipe, share_layers=share_layers, max_batch_size=max_batch_size)
    try:
        yield
    finally:
        remove_shared_attention(inner_pipe)
        logger.info("_shared_attention_scope: shared attention removed (finally).")


# ---------------------------------------------------------------------------
# Private helper — multi-ref 임베딩 주입 경고
# ---------------------------------------------------------------------------


def _resolve_effective_reference(
    reference: Image.Image,
    references: list[Image.Image] | None,
) -> Image.Image:
    """D4 multi-reference 정책에 따라 실제 사용할 단일 reference를 결정한다.

    v1에서는 임베딩 직접 주입이 미구현이므로 다음 규칙을 따른다:
    - ``references`` 가 None이거나 빈 리스트면 ``reference`` 사용.
    - ``references`` 가 1개면 ``references[0]`` 사용.
    - ``references`` 가 2개 이상이면 ``aggregate_references`` 를 로깅 전용으로 호출하고
      ``references[0]`` 으로 fallback한다.

    Args:
        reference: 기본 단일 reference 이미지.
        references: 다중 reference 리스트 (D4). None이면 ``reference`` 사용.

    Returns:
        실제 변환에 사용할 단일 reference PIL Image.
    """
    if references is None or len(references) == 0:
        return reference
    if len(references) == 1:
        return references[0]

    # 2개 이상: aggregate_references 로깅만, fallback to references[0]
    _warn_multi_ref_embed_not_implemented(len(references))
    logger.info(
        "_resolve_effective_reference: D4 aggregate_references 호출 (참조만, 주입 미구현): "
        "strategy=mean, N=%d",
        len(references),
    )
    try:
        from src.encoding.multi_ref import aggregate_references

        _embed = aggregate_references(references, strategy="mean")
        logger.debug(
            "D4 aggregate_references 완료: embed shape=%s (주입 미구현 — 무시됨)",
            list(_embed.shape),
        )
    except NotImplementedError:
        logger.warning("aggregate_references NotImplementedError — 무시하고 계속.")
    return references[0]


def _validate_shared_attention_if_needed(
    enabled: bool,
    effective_reference: Image.Image,
    sources: list[Image.Image],
    max_batch_size: int,
) -> None:
    """use_shared_attention=True 일 때 배치 크기 검증을 수행한다.

    Args:
        enabled: False면 즉시 반환 (no-op).
        effective_reference: D4에서 결정된 단일 reference 이미지.
        sources: 변환할 원본 에셋 PIL Image 리스트.
        max_batch_size: VRAM 상한 (RTX 4070 기준 4 권장).

    Raises:
        ValueError: ``validate_batch_for_shared_attention`` 이 N > max_batch_size 감지 시.
    """
    if not enabled:
        return
    from src.encoding.shared_attention import validate_batch_for_shared_attention

    validate_batch_for_shared_attention(
        [effective_reference, *sources], max_batch_size=max_batch_size
    )


def _warn_multi_ref_embed_not_implemented(n_refs: int) -> None:
    """v1에서 임베딩 직접 주입 경로가 미구현임을 로깅하고 fallback을 기록한다.

    Args:
        n_refs: references 개수.
    """
    logger.warning(
        "transform_batch: %d references 제공됐으나 임베딩 주입 경로는 v1 미구현 "
        "(NotImplementedError — Linux 실측 후 보강 예정). "
        "references[0]을 단일 reference로 사용합니다.",
        n_refs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_batch_transform(
    pipe_obj: StyleUnificationPipeline,
    sources: list[Image.Image],
    reference: Image.Image,
    *,
    references: list[Image.Image] | None = None,
    enforce_consistency: bool = True,
    use_shared_attention: bool = False,
    scales: AttributeScales | None = None,
    attribute_mode: str | None = None,
    region_mask: npt.NDArray[np.float32] | None = None,
    seed: int | None = None,
    share_layers: list[int] | None = None,
    max_batch_size: int = 4,
) -> list[Image.Image]:
    """Sources 배치를 reference 스타일로 변환한다.

    D1(후처리 일관성) · D4(multi-ref 집계) · D5(shared attention) 통합 진입점.

    배치 흐름:
      1. D4: ``references`` 가 2개 이상이면 ``aggregate_references`` 로 임베딩 집계.
         v1에서는 임베딩 직접 주입 미구현 — ``references[0]`` 을 단일 reference로 사용.
      2. D5: ``use_shared_attention=True`` 면 ``_shared_attention_scope`` 로 UNet processor를
         교체. v1에서는 batched denoise 경로 미구현 — sequential 경로로 fallback.
      3. sequential 경로: sources 각각에 ``pipe_obj.transform()`` 호출.
         ``seed`` 가 있으면 각 이미지마다 ``seed + i`` 로 결정론적 재현성 보장.
      4. D1: ``enforce_consistency=True`` 면 ``extract_style_statistics`` +
         ``enforce_consistency`` 로 배치 일관성 후처리.

    Args:
        pipe_obj: 초기화된 ``StyleUnificationPipeline`` 인스턴스.
        sources: 변환할 원본 에셋 PIL Image 리스트 (N >= 1).
        reference: 스타일 reference PIL Image. D4 ``references`` 가 없으면 이것만 사용.
        references: 다중 reference 리스트 (D4). None이면 ``reference`` 단일 사용.
            2개 이상이면 ``aggregate_references`` 호출 — v1 임베딩 주입은 미구현.
        enforce_consistency: True면 배치 출력에 D1 일관성 후처리 적용 (기본 True).
            ``extract_style_statistics`` + ``enforce_consistency`` 순으로 실행.
        use_shared_attention: True면 D5 shared attention 시도 (기본 False, ADR-010).
            WARNING: 첫 시도 동작 보장 없음. v1 batched denoise 미구현이라
            현재는 sequential 경로로 fallback. ``_shared_attention_scope`` try/finally는 유효.
        scales: 속성 단위 scale 묶음 (D2). 각 ``transform()`` 호출에 전달.
        attribute_mode: 속성 모드 문자열 (D2). ``scales`` 가 None일 때 ``transform()`` 내부
            에서 ``route_scales`` 로 변환.
        region_mask: (H, W) float32 사용자 마스크 (D3). 각 ``transform()`` 호출에 전달.
        seed: 재현성 seed. 각 source에 ``seed + i`` 로 적용 (None이면 미적용).
        share_layers: D5 shared attention 적용 layer 인덱스 리스트. None이면 전체.
        max_batch_size: D5 VRAM 상한 (RTX 4070 기준 기본값 4).

    Returns:
        N개 RGBA PIL Image 리스트. sources와 같은 순서.

    Raises:
        ValueError: ``sources`` 가 빈 리스트일 때.
        AttributeError: ``pipe_obj.pipe`` 가 None일 때 (``use_shared_attention=True`` 경우).

    Example:
        >>> outputs = run_batch_transform(
        ...     pipe,
        ...     sources=[src1, src2, src3],
        ...     reference=ref,
        ...     enforce_consistency=True,
        ...     use_shared_attention=False,
        ...     seed=42,
        ... )
    """
    if not sources:
        raise ValueError("sources must not be empty.")

    n = len(sources)
    logger.info(
        "run_batch_transform: N=%d, consistency=%s, shared_attn=%s, seed=%s",
        n,
        enforce_consistency,
        use_shared_attention,
        seed,
    )

    # D4: 실제 사용할 단일 reference 결정 (v1: 임베딩 주입 미구현)
    effective_reference = _resolve_effective_reference(reference, references)

    # D5: validate batch size if shared attention is requested
    _validate_shared_attention_if_needed(
        use_shared_attention, effective_reference, sources, max_batch_size
    )

    with _shared_attention_scope(
        pipe_obj,
        enabled=use_shared_attention,
        share_layers=share_layers,
        max_batch_size=max_batch_size,
    ):
        if use_shared_attention and _D5_BATCHED_DENOISE_IMPLEMENTED:
            # 미래 구현 자리 — Linux 실측 후 채울 batched denoise 경로
            raise NotImplementedError(
                "D5 batched denoise not yet implemented. "
                "Set use_shared_attention=False or wait for Linux validation phase."
            )
        if use_shared_attention:
            logger.warning(
                "D5 batched denoise 미구현 (v1 골격 단계). "
                "shared attention scope 활성화 but sequential 경로로 fallback. "
                "Linux 실측 후 batched denoise로 교체 예정 (ADR-010)."
            )

        outputs = _run_sequential(
            pipe_obj=pipe_obj,
            sources=sources,
            reference=effective_reference,
            scales=scales,
            attribute_mode=attribute_mode,
            region_mask=region_mask,
            seed=seed,
        )

    if enforce_consistency:
        outputs = _apply_consistency(outputs, reference=effective_reference)

    logger.info("run_batch_transform: done, returning %d outputs.", len(outputs))
    return outputs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _run_sequential(
    pipe_obj: StyleUnificationPipeline,
    sources: list[Image.Image],
    reference: Image.Image,
    *,
    scales: AttributeScales | None,
    attribute_mode: str | None,
    region_mask: npt.NDArray[np.float32] | None,
    seed: int | None,
) -> list[Image.Image]:
    """Sources를 하나씩 순차적으로 transform 한다.

    Args:
        pipe_obj: 초기화된 ``StyleUnificationPipeline`` 인스턴스.
        sources: 변환할 원본 에셋 PIL Image 리스트.
        reference: 스타일 reference PIL Image.
        scales: 속성 단위 scale 묶음. None이면 config 값 사용.
        attribute_mode: 속성 모드 문자열. scales가 None일 때 참조.
        region_mask: (H, W) float32 사용자 마스크. None이면 미적용.
        seed: 재현성 seed. 각 source에 seed + i 적용. None이면 미적용.

    Returns:
        N개 RGBA PIL Image 리스트.
    """
    results: list[Image.Image] = []
    for i, src in enumerate(sources):
        per_seed: int | None = (seed + i) if seed is not None else None
        logger.debug("Sequential transform [%d/%d] seed=%s", i + 1, len(sources), per_seed)

        transform_kwargs: dict[str, Any] = {
            "seed": per_seed,
            "scales": scales,
            "attribute_mode": attribute_mode,
            "region_mask": region_mask,
        }
        result = pipe_obj.transform(src, reference, **transform_kwargs)
        if isinstance(result, dict):
            result = result["result"]
        results.append(result)

    return results


def _apply_consistency(
    outputs: list[Image.Image],
    reference: Image.Image,
) -> list[Image.Image]:
    """D1 배치 일관성 후처리를 적용한다.

    ``extract_style_statistics`` 로 reference에서 통계를 추출한 뒤
    ``enforce_consistency`` 로 outputs 전체에 적용한다.

    Args:
        outputs: N개 RGBA PIL Image 리스트 (transform 직후).
        reference: 스타일 reference PIL Image. 통계 추출 기준.

    Returns:
        일관성 후처리된 N개 RGBA PIL Image 리스트.
    """
    from src.encoding.batch_consistency import enforce_consistency, extract_style_statistics

    logger.info("D1 enforce_consistency: extracting style statistics from reference.")
    stats = extract_style_statistics(reference, cache_embedding=False)

    logger.info("D1 enforce_consistency: applying to %d outputs.", len(outputs))
    consistent_outputs = enforce_consistency(outputs, stats)

    return consistent_outputs
