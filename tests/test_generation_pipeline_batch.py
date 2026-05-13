"""pipeline_batch.py (run_batch_transform) 단위 테스트 (D1 / D4 / D5).

모델 로딩 없이 mock pipe_obj를 주입해 흐름 검증만 수행한다.

patch 전략:
  - extract_style_statistics, enforce_consistency → 소스 모듈 경로 패치
    (pipeline_batch는 _apply_consistency 내부에서 lazy import 사용)
  - validate_batch_for_shared_attention → 소스 모듈 경로 패치
    (pipeline_batch는 run_batch_transform 내부에서 lazy import 사용)
  - apply_shared_attention, remove_shared_attention → _shared_attention_scope
    내부 lazy import 이므로 소스 모듈 경로 패치
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import src.generation.pipeline_batch as pipeline_batch_module
from src.generation.pipeline_batch import _shared_attention_scope, run_batch_transform

# ---------------------------------------------------------------------------
# Patch 경로 상수
# ---------------------------------------------------------------------------
_EXTRACT_STATS = "src.encoding.batch_consistency.extract_style_statistics"
_ENFORCE_CONSISTENCY = "src.encoding.batch_consistency.enforce_consistency"
_VALIDATE_BATCH = "src.encoding.shared_attention.validate_batch_for_shared_attention"
_APPLY_SHARED = "src.encoding.shared_attention.apply_shared_attention"
_REMOVE_SHARED = "src.encoding.shared_attention.remove_shared_attention"
_AGGREGATE_REFS = "src.encoding.multi_ref.aggregate_references"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgba(w: int = 64, h: int = 64, color: tuple = (0, 255, 0, 255)) -> Image.Image:
    """단색 RGBA 이미지를 반환한다."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :] = color
    return Image.fromarray(arr, mode="RGBA")


def _make_mock_pipe_obj(transform_return: Image.Image | None = None) -> MagicMock:
    """StyleUnificationPipeline mock. transform()이 단색 RGBA를 반환한다."""
    green = transform_return or _make_rgba(color=(0, 255, 0, 255))
    pipe_obj = MagicMock()
    pipe_obj.transform.return_value = green
    pipe_obj.pipe = MagicMock()  # _shared_attention_scope에서 pipe_obj.pipe 접근
    return pipe_obj


def _make_sources(n: int = 3) -> list[Image.Image]:
    """n개 고유 색 RGBA 이미지 리스트."""
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (128, 128, 0, 255)]
    return [_make_rgba(color=colors[i % len(colors)]) for i in range(n)]


def _make_dummy_stats():
    """StyleStatistics 더미. enforce_consistency mock 반환용."""
    return MagicMock()


# ---------------------------------------------------------------------------
# C1: transform_batch 시그니처 + flow
# ---------------------------------------------------------------------------


def test_run_batch_transform_calls_transform_n_times():
    """run_batch_transform이 pipe_obj.transform을 len(sources)번 호출하는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(3)
    reference = _make_rgba(color=(128, 128, 128, 255))

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, seed=42, enforce_consistency=True)

    assert pipe_obj.transform.call_count == 3


def test_run_batch_transform_passes_seed_plus_i():
    """각 source 호출마다 seed+i가 전달되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(3)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, seed=42, enforce_consistency=True)

    call_seeds = [c.kwargs["seed"] for c in pipe_obj.transform.call_args_list]
    assert call_seeds == [42, 43, 44]


def test_run_batch_transform_passes_reference_to_each_transform():
    """각 transform() 호출의 두 번째 위치 인자가 effective_reference인가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(3)
    reference = _make_rgba(color=(100, 100, 100, 255))

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, seed=10, enforce_consistency=True)

    for c in pipe_obj.transform.call_args_list:
        assert c.args[1] is reference


def test_run_batch_transform_passes_source_i_as_first_arg():
    """각 transform() 호출의 첫 번째 위치 인자가 sources[i]인가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(3)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, seed=0, enforce_consistency=True)

    for i, c in enumerate(pipe_obj.transform.call_args_list):
        assert c.args[0] is sources[i]


def test_run_batch_transform_enforce_consistency_true_calls_extract_stats():
    """enforce_consistency=True → extract_style_statistics가 1회 호출되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()) as mock_stats,
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, enforce_consistency=True)

    mock_stats.assert_called_once()


def test_run_batch_transform_enforce_consistency_true_calls_enforce_consistency():
    """enforce_consistency=True → enforce_consistency가 1회 호출되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs) as mock_enforce,
    ):
        run_batch_transform(pipe_obj, sources, reference, enforce_consistency=True)

    mock_enforce.assert_called_once()


def test_run_batch_transform_enforce_consistency_false_skips_extract_stats():
    """enforce_consistency=False → extract_style_statistics 미호출."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS) as mock_stats,
        patch(_ENFORCE_CONSISTENCY) as mock_enforce,
    ):
        run_batch_transform(pipe_obj, sources, reference, enforce_consistency=False)

    mock_stats.assert_not_called()
    mock_enforce.assert_not_called()


def test_run_batch_transform_empty_sources_raises():
    """sources가 빈 리스트이면 ValueError가 발생하는가."""
    pipe_obj = _make_mock_pipe_obj()
    reference = _make_rgba()

    with pytest.raises(ValueError, match="empty"):
        run_batch_transform(pipe_obj, [], reference)


def test_run_batch_transform_returns_n_outputs():
    """run_batch_transform 반환값 길이가 len(sources)와 같은가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(4)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        results = run_batch_transform(pipe_obj, sources, reference, enforce_consistency=True)

    assert len(results) == 4


def test_run_batch_transform_no_seed_passes_none():
    """seed=None이면 각 transform()에 seed=None이 전달되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(pipe_obj, sources, reference, seed=None, enforce_consistency=True)

    call_seeds = [c.kwargs["seed"] for c in pipe_obj.transform.call_args_list]
    assert call_seeds == [None, None]


# ---------------------------------------------------------------------------
# C1: references 다중 전달 — multi-ref WARNING + fallback
# ---------------------------------------------------------------------------


def test_run_batch_transform_multi_references_logs_warning(caplog):
    """references=[ref1, ref2] 전달 시 WARNING 로그가 발생하는가."""
    import logging

    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba(color=(1, 1, 1, 255))
    ref1 = _make_rgba(color=(200, 200, 200, 255))
    ref2 = _make_rgba(color=(100, 100, 100, 255))

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
        patch(_AGGREGATE_REFS, return_value=MagicMock(shape=[1, 257, 1280])),
        caplog.at_level(logging.WARNING),
    ):
        run_batch_transform(
            pipe_obj, sources, reference, references=[ref1, ref2], enforce_consistency=True
        )

    # 경고 메시지에 references 개수나 fallback 관련 단어가 포함돼야 함
    assert len(caplog.records) > 0
    messages = " ".join(r.message for r in caplog.records)
    assert any(kw in messages for kw in ["2", "references", "fallback", "미구현"])


def test_run_batch_transform_multi_references_uses_references_0_as_reference():
    """references=[ref1, ref2] 전달 시 references[0]이 effective_reference로 사용되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(1)
    reference = _make_rgba(color=(1, 1, 1, 255))
    ref1 = _make_rgba(color=(200, 200, 200, 255))
    ref2 = _make_rgba(color=(100, 100, 100, 255))

    with (
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
        patch(_AGGREGATE_REFS, return_value=MagicMock(shape=[1, 257, 1280])),
    ):
        run_batch_transform(
            pipe_obj, sources, reference, references=[ref1, ref2], enforce_consistency=True
        )

    # effective_reference = references[0] = ref1 이어야 함
    second_arg = pipe_obj.transform.call_args.args[1]
    assert second_arg is ref1


# ---------------------------------------------------------------------------
# C4: shared_attention 토글 — validate 호출 검증
# ---------------------------------------------------------------------------


def test_run_batch_transform_shared_attention_true_calls_validate():
    """use_shared_attention=True → validate_batch_for_shared_attention이 1회 호출되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_VALIDATE_BATCH) as mock_validate,
        patch(_APPLY_SHARED),
        patch(_REMOVE_SHARED),
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(
            pipe_obj, sources, reference, use_shared_attention=True, enforce_consistency=True
        )

    mock_validate.assert_called_once()


def test_run_batch_transform_shared_attention_true_validate_receives_reference_plus_sources():
    """validate_batch_for_shared_attention의 첫 번째 인자가 [reference, *sources]인가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba(color=(99, 99, 99, 255))

    with (
        patch(_VALIDATE_BATCH) as mock_validate,
        patch(_APPLY_SHARED),
        patch(_REMOVE_SHARED),
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(
            pipe_obj, sources, reference, use_shared_attention=True, enforce_consistency=True
        )

    batch_arg = mock_validate.call_args.args[0]
    assert batch_arg[0] is reference
    assert batch_arg[1] is sources[0]
    assert batch_arg[2] is sources[1]


def test_run_batch_transform_shared_attention_false_skips_validate():
    """use_shared_attention=False → validate_batch_for_shared_attention 미호출."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(2)
    reference = _make_rgba()

    with (
        patch(_VALIDATE_BATCH) as mock_validate,
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(
            pipe_obj, sources, reference, use_shared_attention=False, enforce_consistency=True
        )

    mock_validate.assert_not_called()


def test_run_batch_transform_share_layers_passed_to_scope():
    """share_layers=[10, 15] 전달 시 _shared_attention_scope에 그대로 전달되는가."""
    pipe_obj = _make_mock_pipe_obj()
    sources = _make_sources(1)
    reference = _make_rgba()

    captured: dict = {}

    @contextlib.contextmanager
    def capturing_scope(pipe_obj_arg, *, enabled, share_layers, max_batch_size):
        captured["enabled"] = enabled
        captured["share_layers"] = share_layers
        captured["max_batch_size"] = max_batch_size
        yield

    with (
        patch.object(pipeline_batch_module, "_shared_attention_scope", capturing_scope),
        patch(_VALIDATE_BATCH),
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
    ):
        run_batch_transform(
            pipe_obj,
            sources,
            reference,
            use_shared_attention=True,
            share_layers=[10, 15],
            max_batch_size=3,
            enforce_consistency=True,
        )

    assert captured["enabled"] is True
    assert captured["share_layers"] == [10, 15]
    assert captured["max_batch_size"] == 3


# ---------------------------------------------------------------------------
# C4: _shared_attention_scope — apply/remove try/finally 보장
# ---------------------------------------------------------------------------


def test_shared_attention_scope_apply_and_remove_called_on_success():
    """정상 실행 시 apply 1회 → remove 1회(finally) 호출되는가."""
    pipe_obj = _make_mock_pipe_obj()
    pipe_obj.pipe = MagicMock()

    mock_apply = MagicMock()
    mock_remove = MagicMock()

    with (
        patch(_APPLY_SHARED, mock_apply),
        patch(_REMOVE_SHARED, mock_remove),
        _shared_attention_scope(pipe_obj, enabled=True, share_layers=None, max_batch_size=4),
    ):
        pass

    mock_apply.assert_called_once()
    mock_remove.assert_called_once()


def test_shared_attention_scope_remove_called_in_finally_on_error():
    """transform() 예외 시 remove_shared_attention이 finally에서 호출되는가."""
    pipe_obj = _make_mock_pipe_obj()
    pipe_obj.pipe = MagicMock()

    mock_apply = MagicMock()
    mock_remove = MagicMock()

    with (
        patch(_APPLY_SHARED, mock_apply),
        patch(_REMOVE_SHARED, mock_remove),
    ):
        ctx = _shared_attention_scope(pipe_obj, enabled=True, share_layers=None, max_batch_size=4)
        try:
            with ctx:
                raise RuntimeError("injected error")
        except RuntimeError:
            pass

    mock_apply.assert_called_once()
    mock_remove.assert_called_once()


def test_shared_attention_scope_enabled_false_is_noop():
    """enabled=False이면 apply/remove가 호출되지 않는가."""
    pipe_obj = _make_mock_pipe_obj()

    mock_apply = MagicMock()
    mock_remove = MagicMock()

    with (
        patch(_APPLY_SHARED, mock_apply),
        patch(_REMOVE_SHARED, mock_remove),
        _shared_attention_scope(pipe_obj, enabled=False, share_layers=None, max_batch_size=4),
    ):
        pass

    mock_apply.assert_not_called()
    mock_remove.assert_not_called()


def test_run_batch_transform_transform_error_propagates_but_remove_called():
    """pipe_obj.transform에서 RuntimeError → 예외 전파되고 remove는 finally에서 호출."""
    pipe_obj = _make_mock_pipe_obj()
    pipe_obj.transform.side_effect = RuntimeError("injected transform error")
    sources = _make_sources(1)
    reference = _make_rgba()

    mock_apply = MagicMock()
    mock_remove = MagicMock()

    with (
        patch(_APPLY_SHARED, mock_apply),
        patch(_REMOVE_SHARED, mock_remove),
        patch(_VALIDATE_BATCH),
        patch(_EXTRACT_STATS, return_value=_make_dummy_stats()),
        patch(_ENFORCE_CONSISTENCY, side_effect=lambda outputs, stats: outputs),
        pytest.raises(RuntimeError, match="injected transform error"),
    ):
        run_batch_transform(
            pipe_obj,
            sources,
            reference,
            use_shared_attention=True,
            enforce_consistency=True,
        )

    mock_remove.assert_called_once()
