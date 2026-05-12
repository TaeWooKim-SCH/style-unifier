"""평가 헬퍼 모듈 (run_eval.py 전용 내부 모듈).

메트릭 계산, 통계 집계, 결과 저장 함수를 모아 run_eval.py의 줄 수를 500 이내로 유지한다.
이 모듈을 직접 실행하거나 외부에서 import하지 말 것.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_CONSISTENCY_BATCH_SIZE = 5

_METRIC_INTERPRETATION: dict[str, str] = {
    "clip_style_similarity": "높을수록 reference와 유사",
    "lpips_structure": "낮을수록 원본 구조 보존",
    "dino_identity": "높을수록 객체 identity 유지",
    "palette_distance": "낮을수록 reference 팔레트 근접",
    "gram_matrix_distance": "낮을수록 스타일 통계 유사",
    "sigma_palette": "낮을수록 출력들 색 분포 일치",
    "sigma_linewidth": "낮을수록 라인 두께 일치",
    "sigma_shading": "낮을수록 shading 분포 일치",
}


# ---------------------------------------------------------------------------
# IQA 메트릭 계산
# ---------------------------------------------------------------------------


def compute_iqa_metrics(
    source_img: object,
    reference_img: object,
    output_img: object,
    skip: frozenset[str],
) -> dict[str, float | None]:
    """단일 페어에 대해 IQA 메트릭 5개를 계산한다.

    ImportError 발생 시 해당 메트릭은 NaN으로 처리하고 WARNING을 남긴다.

    Args:
        source_img: 원본 PIL Image.
        reference_img: reference PIL Image.
        output_img: 변환 결과 PIL Image.
        skip: 생략할 메트릭 키 집합 (``"clip"``, ``"dino"``, ``"lpips"``, ``"gram"``).

    Returns:
        메트릭 이름 → float | None dict. 생략된 메트릭은 None.
    """
    from PIL import Image

    src: Image.Image = source_img  # type: ignore[assignment]
    ref: Image.Image = reference_img  # type: ignore[assignment]
    out: Image.Image = output_img  # type: ignore[assignment]

    src = src.convert("RGBA")
    ref = ref.convert("RGBA")
    out = out.convert("RGBA")

    results: dict[str, float | None] = {}

    if "clip" in skip:
        results["clip_style_similarity"] = None
    else:
        try:
            from src.evaluation.metrics import clip_style_similarity

            results["clip_style_similarity"] = clip_style_similarity(out, ref)
        except ImportError as exc:
            logger.warning("clip_style_similarity skipped (ImportError): %s", exc)
            results["clip_style_similarity"] = float("nan")

    if "lpips" in skip:
        results["lpips_structure"] = None
    else:
        try:
            from src.evaluation.metrics import lpips_structure

            out_resized = out.resize(src.size, resample=0)
            results["lpips_structure"] = lpips_structure(src, out_resized)
        except ImportError as exc:
            logger.warning("lpips_structure skipped (ImportError): %s", exc)
            results["lpips_structure"] = float("nan")
        except ValueError as exc:
            logger.warning("lpips_structure ValueError: %s", exc)
            results["lpips_structure"] = float("nan")

    if "dino" in skip:
        results["dino_identity"] = None
    else:
        try:
            from src.evaluation.metrics import dino_identity

            results["dino_identity"] = dino_identity(src, out)
        except ImportError as exc:
            logger.warning("dino_identity skipped (ImportError): %s", exc)
            results["dino_identity"] = float("nan")

    try:
        from src.evaluation.metrics import palette_distance

        results["palette_distance"] = palette_distance(ref, out)
    except ImportError as exc:
        logger.warning("palette_distance skipped (ImportError): %s", exc)
        results["palette_distance"] = float("nan")

    if "gram" in skip:
        results["gram_matrix_distance"] = None
    else:
        try:
            from src.evaluation.metrics import gram_matrix_distance

            results["gram_matrix_distance"] = gram_matrix_distance(ref, out)
        except ImportError as exc:
            logger.warning("gram_matrix_distance skipped (ImportError): %s", exc)
            results["gram_matrix_distance"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# 일관성 메트릭 계산
# ---------------------------------------------------------------------------


def chunk_by_batch_size(
    pairs: list[dict],
    batch_size: int,
) -> list[list[dict]]:
    """pairs를 batch_size 크기의 청크 리스트로 나눈다.

    Args:
        pairs: 페어 dict 리스트.
        batch_size: 각 청크 크기. 1 이상이어야 한다.

    Returns:
        청크 리스트. 마지막 청크는 batch_size보다 작을 수 있다.

    Raises:
        ValueError: batch_size < 1.
    """
    if batch_size < 1:
        raise ValueError(f"consistency_batch_size must be >= 1, got {batch_size}")

    return [pairs[start : start + batch_size] for start in range(0, len(pairs), batch_size)]


def compute_consistency_metrics(
    output_imgs: Sequence[object],
) -> dict[str, float]:
    """출력 이미지 그룹에 대해 일관성 메트릭 3개를 계산한다.

    ImportError 발생 시 해당 메트릭은 NaN으로 처리하고 WARNING을 남긴다.

    Args:
        output_imgs: 같은 그룹(batch)의 변환 결과 PIL Image 시퀀스.

    Returns:
        ``{"sigma_palette": float, "sigma_linewidth": float, "sigma_shading": float}``.
        NaN은 라이브러리 부재를 의미한다.
    """
    from PIL import Image

    imgs: list[Image.Image] = [img.convert("RGBA") for img in output_imgs]  # type: ignore[union-attr]
    results: dict[str, float] = {}

    try:
        from src.evaluation.consistency import sigma_palette

        results["sigma_palette"] = sigma_palette(imgs)
    except ImportError as exc:
        logger.warning("sigma_palette skipped (ImportError): %s", exc)
        results["sigma_palette"] = float("nan")

    try:
        from src.evaluation.consistency import sigma_linewidth

        results["sigma_linewidth"] = sigma_linewidth(imgs)
    except ImportError as exc:
        logger.warning("sigma_linewidth skipped (ImportError): %s", exc)
        results["sigma_linewidth"] = float("nan")

    try:
        from src.evaluation.consistency import sigma_shading

        results["sigma_shading"] = sigma_shading(imgs)
    except ImportError as exc:
        logger.warning("sigma_shading skipped (ImportError): %s", exc)
        results["sigma_shading"] = float("nan")

    return results


# ---------------------------------------------------------------------------
# 일관성 그룹화 (reference 기반)
# ---------------------------------------------------------------------------


def group_by_reference(
    success_pairs: list[dict],
    success_imgs: Sequence[object],
) -> dict[str, list[object]]:
    """성공 페어를 reference 경로 기준으로 묶어 {ref_path_str: [img, ...]} dict 반환.

    Args:
        success_pairs: output이 None이 아닌 페어 dict 리스트.
        success_imgs: success_pairs와 1:1 대응하는 출력 PIL Image 시퀀스.

    Returns:
        reference 경로 문자열 → 출력 이미지 리스트 dict.
    """
    groups: dict[str, list[object]] = {}
    for pair, img in zip(success_pairs, success_imgs, strict=True):
        ref_key = str(pair["reference"])
        groups.setdefault(ref_key, []).append(img)
    return groups


def compute_sigma_metrics(
    per_sample: list[dict],
    output_imgs: Sequence[object],
    consistency_batch_size: int,
) -> dict[str, object]:
    """per_sample + output_imgs를 reference 기준으로 묶어 sigma 메트릭을 집계한다.

    같은 reference를 공유하는 출력 이미지 그룹 안에서만 sigma_palette / sigma_linewidth /
    sigma_shading을 계산한다. 그룹 내 이미지 수가 K (``consistency_batch_size``) 미만이면
    skip하고 warning을 남긴다. 모든 reference가 서로 다른 경우(1:1 평가셋)에는
    측정 불가로 NaN을 반환한다.

    Args:
        per_sample: 변환 결과 dict 리스트. 각 원소의 ``"reference"`` 키에 Path 또는 str.
        output_imgs: 변환 성공한 PIL Image 시퀀스 (per_sample 성공 항목과 순서 대응).
        consistency_batch_size: 그룹 내 최소 이미지 수 K. K 미만 그룹은 skip.

    Returns:
        ``{"sigma_palette": ..., "sigma_linewidth": ..., "sigma_shading": ...}`` aggregate dict.
        모든 그룹이 skip되면 각 값의 mean은 None.
    """
    success_pairs = [p for p in per_sample if p.get("output") is not None]

    groups = group_by_reference(success_pairs, list(output_imgs))

    if not groups:
        logger.warning("compute_sigma_metrics: 성공 페어가 없습니다.")
        empty: list[float | None] = []
        return {
            "sigma_palette": aggregate_metric_values(empty),
            "sigma_linewidth": aggregate_metric_values(empty),
            "sigma_shading": aggregate_metric_values(empty),
        }

    max_group_size = max(len(imgs) for imgs in groups.values())
    if max_group_size < 2:
        logger.warning(
            "compute_sigma_metrics: 모든 reference 그룹 크기 < 2. "
            "1:1 페어링 평가셋에서는 sigma_* 측정 불가 - NaN 반환. "
            "배치 평가셋(같은 reference로 N개 출력)을 사용하세요."
        )
        nan_val = float("nan")
        return {
            "sigma_palette": aggregate_metric_values([nan_val]),
            "sigma_linewidth": aggregate_metric_values([nan_val]),
            "sigma_shading": aggregate_metric_values([nan_val]),
        }

    sigma_p: list[float | None] = []
    sigma_l: list[float | None] = []
    sigma_s: list[float | None] = []

    for ref_key, group_imgs in groups.items():
        if len(group_imgs) < consistency_batch_size:
            logger.warning(
                "reference 그룹 '%s': 이미지 %d개 < K=%d - sigma_* skip.",
                ref_key,
                len(group_imgs),
                consistency_batch_size,
            )
            continue

        c = compute_consistency_metrics(group_imgs)
        sigma_p.append(c["sigma_palette"])
        sigma_l.append(c["sigma_linewidth"])
        sigma_s.append(c["sigma_shading"])
        logger.debug("reference group '%s' consistency: %s", ref_key, c)

    if not sigma_p:
        logger.warning(
            "compute_sigma_metrics: 유효한 reference 그룹이 없습니다 (모두 K=%d 미만). "
            "sigma_* = None 반환.",
            consistency_batch_size,
        )

    return {
        "sigma_palette": aggregate_metric_values(sigma_p),
        "sigma_linewidth": aggregate_metric_values(sigma_l),
        "sigma_shading": aggregate_metric_values(sigma_s),
    }


# ---------------------------------------------------------------------------
# 통계 집계
# ---------------------------------------------------------------------------


def aggregate_metric_values(values: Sequence[float | None]) -> dict[str, object]:
    """float | None 시퀀스에서 mean, std, values를 집계한다.

    None 및 NaN은 집계에서 제외한다.

    Args:
        values: 메트릭 값 시퀀스.

    Returns:
        ``{"mean": float | None, "std": float | None, "values": list}``.
    """
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return {"mean": None, "std": None, "values": list(values)}

    mean_val = sum(valid) / len(valid)
    std_val = (
        math.sqrt(sum((x - mean_val) ** 2 for x in valid) / len(valid)) if len(valid) > 1 else 0.0
    )

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "values": list(values),
    }


# ---------------------------------------------------------------------------
# 결과 저장
# ---------------------------------------------------------------------------


def save_results(exp_dir: Path, results: dict) -> Path:
    """results dict를 exp_dir/results.json으로 저장한다.

    Args:
        exp_dir: 실험 디렉토리.
        results: 저장할 결과 딕셔너리.

    Returns:
        저장된 results.json 경로.
    """
    results_path = exp_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("results.json saved: %s", results_path)
    return results_path


def save_summary(exp_dir: Path, results: dict) -> Path:
    """results dict를 사람이 읽기 쉬운 summary.md로 저장한다.

    Args:
        exp_dir: 실험 디렉토리.
        results: save_results()와 동일한 results dict.

    Returns:
        저장된 summary.md 경로.
    """
    exp_info = results["experiment"]
    iqa = results["iqa_overall"]
    consistency = results["consistency"]
    k = exp_info.get("consistency_batch_size", _DEFAULT_CONSISTENCY_BATCH_SIZE)

    def _fmt(val: object) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float) and math.isnan(val):
            return "NaN"
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    lines: list[str] = [
        f"# 실험 결과: {exp_dir.name}",
        "",
        f"- 시스템: {exp_info['system']}",
        f"- 페어 수: {exp_info['n_samples']}",
        f"- Seed: {exp_info['seed']}",
        f"- Config: {exp_info['config_path']}",
        "",
        "## IQA 메트릭 (각 페어 독립)",
        "",
        "| 메트릭 | 평균 | std | 해석 |",
        "|---|---|---|---|",
    ]

    for label, key in [
        ("CLIP Style Similarity", "clip_style_similarity"),
        ("LPIPS Structure", "lpips_structure"),
        ("DINOv2 Identity", "dino_identity"),
        ("Palette Distance (EMD)", "palette_distance"),
        ("Gram Matrix Distance", "gram_matrix_distance"),
    ]:
        stat = iqa.get(key, {})
        lines.append(
            f"| {label} | {_fmt(stat.get('mean'))} | {_fmt(stat.get('std'))} | {_METRIC_INTERPRETATION.get(key, '')} |"
        )

    lines.extend(
        [
            "",
            f"## 일관성 메트릭 (그룹 단위, K={k})",
            "",
            "| 메트릭 | 평균 | std | 해석 |",
            "|---|---|---|---|",
        ]
    )

    for label, key in [
        ("sigma_palette", "sigma_palette"),
        ("sigma_linewidth", "sigma_linewidth"),
        ("sigma_shading", "sigma_shading"),
    ]:
        stat = consistency.get(key, {})
        lines.append(
            f"| {label} | {_fmt(stat.get('mean'))} | {_fmt(stat.get('std'))} | {_METRIC_INTERPRETATION.get(key, '')} |"
        )

    lines.extend(
        [
            "",
            "## 일관성 메트릭 주의 사항",
            "",
            "sigma_palette / sigma_linewidth / sigma_shading은 **같은 reference를 공유하는 N개 출력**의",
            "분포 분산을 측정한다. 본 평가셋의 페어 구성에 따라:",
            "",
            "- 같은 reference 그룹이 존재 (배치 평가셋): 본래 의미대로 측정됨",
            f"- 한 reference 그룹에 최소 K={k}개 이상 있어야 계산됨 (미달 그룹은 skip)",
            "- 모든 페어가 서로 다른 reference (1:1 평가셋): 측정 불가 → NaN 반환",
        ]
    )

    summary_path = exp_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("summary.md saved: %s", summary_path)
    return summary_path
