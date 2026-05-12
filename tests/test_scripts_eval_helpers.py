"""scripts/_eval_helpers.py 단위 테스트.

chunk_by_batch_size, aggregate_metric_values는 순수 함수로 직접 테스트.
_load_eval_pairs는 run_eval.py에서 import해서 테스트한다 (tmp_path 활용).
"""

from __future__ import annotations

import math

import pytest
from PIL import Image
from scripts._eval_helpers import aggregate_metric_values, chunk_by_batch_size
from scripts.run_eval import _compute_sigma_metrics, _group_by_reference

# ---------------------------------------------------------------------------
# chunk_by_batch_size
# ---------------------------------------------------------------------------


def test_chunk_by_batch_size_basic():
    """5개 항목을 batch_size=2로 나누면 [[1,2],[3,4],[5]]인가."""
    pairs = [{"id": i} for i in range(1, 6)]
    result = chunk_by_batch_size(pairs, 2)
    assert len(result) == 3
    assert len(result[0]) == 2
    assert len(result[1]) == 2
    assert len(result[2]) == 1


def test_chunk_by_batch_size_empty():
    """빈 리스트에서 빈 리스트를 반환하는가."""
    result = chunk_by_batch_size([], 5)
    assert result == []


def test_chunk_by_batch_size_exact_division():
    """6개 항목을 batch_size=3으로 나누면 2 청크인가."""
    pairs = [{"id": i} for i in range(6)]
    result = chunk_by_batch_size(pairs, 3)
    assert len(result) == 2
    assert all(len(c) == 3 for c in result)


def test_chunk_by_batch_size_single_batch():
    """batch_size >= len(pairs)이면 청크가 1개인가."""
    pairs = [{"id": i} for i in range(3)]
    result = chunk_by_batch_size(pairs, 10)
    assert len(result) == 1
    assert len(result[0]) == 3


def test_chunk_by_batch_size_invalid_batch_size_raises():
    """batch_size < 1이면 ValueError가 발생하는가."""
    with pytest.raises(ValueError, match="batch_size|consistency_batch_size"):
        chunk_by_batch_size([{"id": 0}], 0)


def test_chunk_by_batch_size_preserves_order():
    """청크가 원본 순서를 보존하는가."""
    pairs = [{"id": i} for i in range(4)]
    result = chunk_by_batch_size(pairs, 2)
    flattened = [item for chunk in result for item in chunk]
    assert flattened == pairs


# ---------------------------------------------------------------------------
# aggregate_metric_values
# ---------------------------------------------------------------------------


def test_aggregate_metric_values_mean_correct():
    """[1.0, 2.0, 3.0]의 mean이 2.0인가."""
    result = aggregate_metric_values([1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(2.0, abs=1e-6)


def test_aggregate_metric_values_std_correct():
    """[1.0, 2.0, 3.0]의 std가 올바른가 (population std)."""
    result = aggregate_metric_values([1.0, 2.0, 3.0])
    expected_std = math.sqrt(((1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 2) ** 2) / 3)
    assert result["std"] == pytest.approx(expected_std, abs=1e-4)


def test_aggregate_metric_values_excludes_none():
    """None 값은 집계에서 제외되는가."""
    result = aggregate_metric_values([1.0, None, 3.0])
    assert result["mean"] == pytest.approx(2.0, abs=1e-6)


def test_aggregate_metric_values_excludes_nan():
    """NaN 값은 집계에서 제외되는가."""
    result = aggregate_metric_values([1.0, float("nan"), 3.0])
    assert result["mean"] == pytest.approx(2.0, abs=1e-6)


def test_aggregate_metric_values_all_none_returns_none():
    """모든 값이 None이면 mean이 None인가."""
    result = aggregate_metric_values([None, None])
    assert result["mean"] is None
    assert result["std"] is None


def test_aggregate_metric_values_all_nan_returns_none():
    """모든 값이 NaN이면 mean이 None인가."""
    result = aggregate_metric_values([float("nan"), float("nan")])
    assert result["mean"] is None


def test_aggregate_metric_values_single_value_std_zero():
    """단일 유효 값의 std가 0.0인가."""
    result = aggregate_metric_values([5.0])
    assert result["std"] == pytest.approx(0.0, abs=1e-6)


def test_aggregate_metric_values_values_key_preserved():
    """반환 dict의 'values' 키에 원본 시퀀스가 보존되는가."""
    values = [1.0, 2.0, None]
    result = aggregate_metric_values(values)
    assert result["values"] == values


# ---------------------------------------------------------------------------
# _load_eval_pairs — run_eval.py에서 직접 import
# ---------------------------------------------------------------------------


def test_load_eval_pairs_nonexistent_dir_raises(tmp_path):
    """존재하지 않는 eval_dir에서 FileNotFoundError가 발생하는가."""
    from scripts.run_eval import _load_eval_pairs

    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        _load_eval_pairs(nonexistent, n_samples=10)


def test_load_eval_pairs_empty_dir_returns_empty(tmp_path):
    """빈 eval_dir에서 빈 리스트를 반환하는가."""
    from scripts.run_eval import _load_eval_pairs

    result = _load_eval_pairs(tmp_path, n_samples=10)
    assert result == []


def test_load_eval_pairs_single_pair_found(tmp_path):
    """source_001.png + reference_001.png 쌍에서 1개 dict를 반환하는가."""
    # 테스트용 작은 PNG 파일 생성
    src_path = tmp_path / "source_001.png"
    ref_path = tmp_path / "reference_001.png"
    img = Image.new("RGB", (4, 4), (0, 0, 0))
    img.save(src_path)
    img.save(ref_path)

    from scripts.run_eval import _load_eval_pairs

    result = _load_eval_pairs(tmp_path, n_samples=10)

    assert len(result) == 1
    assert result[0]["pair_id"] == 1
    assert result[0]["source"] == src_path
    assert result[0]["reference"] == ref_path


def test_load_eval_pairs_n_samples_limits_results(tmp_path):
    """n_samples=1이면 페어가 여러 개여도 1개만 반환하는가."""
    for i in range(1, 4):
        idx = f"{i:03d}"
        img = Image.new("RGB", (4, 4))
        img.save(tmp_path / f"source_{idx}.png")
        img.save(tmp_path / f"reference_{idx}.png")

    from scripts.run_eval import _load_eval_pairs

    result = _load_eval_pairs(tmp_path, n_samples=1)
    assert len(result) == 1


def test_load_eval_pairs_missing_reference_skips(tmp_path):
    """source는 있고 reference가 없는 쌍은 제외되는가."""
    img = Image.new("RGB", (4, 4))
    img.save(tmp_path / "source_001.png")
    # reference_001.png 없음

    from scripts.run_eval import _load_eval_pairs

    result = _load_eval_pairs(tmp_path, n_samples=10)
    assert result == []


# ---------------------------------------------------------------------------
# _group_by_reference — 이슈 C3: reference 기반 그룹화
# ---------------------------------------------------------------------------


def _make_success_pair(ref: str, output: str) -> dict:
    """테스트용 성공 페어 dict를 반환한다."""
    return {"reference": ref, "output": output}


def test_group_by_reference_groups_same_ref():
    """같은 reference를 공유하는 페어가 하나의 그룹으로 묶이는가."""
    img = Image.new("RGBA", (4, 4))
    pairs = [
        _make_success_pair("ref_A.png", "out_1.png"),
        _make_success_pair("ref_A.png", "out_2.png"),
        _make_success_pair("ref_B.png", "out_3.png"),
    ]
    imgs: list[object] = [img, img, img]
    groups = _group_by_reference(pairs, imgs)

    assert "ref_A.png" in groups
    assert "ref_B.png" in groups
    assert len(groups["ref_A.png"]) == 2
    assert len(groups["ref_B.png"]) == 1


def test_group_by_reference_all_different_refs():
    """모든 reference가 다른 경우 각 그룹 크기가 1인가."""
    img = Image.new("RGBA", (4, 4))
    pairs = [
        _make_success_pair("ref_1.png", "out_1.png"),
        _make_success_pair("ref_2.png", "out_2.png"),
        _make_success_pair("ref_3.png", "out_3.png"),
    ]
    imgs: list[object] = [img, img, img]
    groups = _group_by_reference(pairs, imgs)

    assert len(groups) == 3
    assert all(len(v) == 1 for v in groups.values())


def test_group_by_reference_empty_input():
    """빈 입력에서 빈 dict를 반환하는가."""
    groups = _group_by_reference([], [])
    assert groups == {}


# ---------------------------------------------------------------------------
# _compute_sigma_metrics — 이슈 C3: reference 기반 그룹화 통합 검증
# ---------------------------------------------------------------------------


def _make_per_sample_with_refs(refs: list[str]) -> tuple[list[dict], list[object]]:
    """reference 목록으로 성공 페어 리스트와 출력 이미지 리스트를 만든다."""
    img = Image.new("RGBA", (4, 4))
    per_sample = [
        {"reference": ref, "output": f"out_{i}.png", "metrics": {}} for i, ref in enumerate(refs)
    ]
    output_imgs: list[object] = [img] * len(refs)
    return per_sample, output_imgs


def test_compute_sigma_metrics_all_different_refs_returns_nan():
    """모든 reference가 다른 1:1 평가셋에서 sigma 값이 NaN인가."""
    per_sample, output_imgs = _make_per_sample_with_refs(["ref_1.png", "ref_2.png", "ref_3.png"])
    result = _compute_sigma_metrics(per_sample, output_imgs, consistency_batch_size=2)

    # 모든 그룹이 크기 1이므로 max_group_size < 2 → NaN 반환
    sigma_p = result["sigma_palette"]
    assert isinstance(sigma_p, dict)
    # mean이 None이거나 NaN이어야 함 (aggregate_metric_values([nan]) 결과)
    mean_val = sigma_p.get("mean")
    assert mean_val is None or (isinstance(mean_val, float) and math.isnan(mean_val))


def test_compute_sigma_metrics_same_ref_group_below_k_skipped():
    """같은 reference 그룹이 있지만 K 미만이면 skip되어 mean이 None인가."""
    # 그룹 A: 1개 (K=2 미만 → skip), 그룹 B: 1개 (skip)
    per_sample, output_imgs = _make_per_sample_with_refs(["ref_A.png", "ref_B.png"])
    result = _compute_sigma_metrics(per_sample, output_imgs, consistency_batch_size=2)

    # max_group_size == 1 이므로 NaN 경로로 진입
    sigma_p = result["sigma_palette"]
    assert isinstance(sigma_p, dict)
    mean_val = sigma_p.get("mean")
    assert mean_val is None or (isinstance(mean_val, float) and math.isnan(mean_val))


def test_compute_sigma_metrics_empty_per_sample():
    """성공 페어가 없으면 mean이 None인가."""
    result = _compute_sigma_metrics([], [], consistency_batch_size=2)

    sigma_p = result["sigma_palette"]
    assert isinstance(sigma_p, dict)
    assert sigma_p.get("mean") is None
