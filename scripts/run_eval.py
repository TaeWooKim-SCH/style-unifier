"""평가 자동화 스크립트 (C5).

config 한 줄 인자로 평가셋 전체에 시스템 또는 baseline 변환을 적용하고,
9개 메트릭을 자동 측정해서 experiments/NNN_name/results.json + summary.md로 저장한다.

사용 예시::

    python scripts/run_eval.py --config configs/default.yaml --dry-run
    python scripts/run_eval.py --config configs/default.yaml --system gpt_image --n-samples 10
    python scripts/run_eval.py --config configs/default.yaml --skip-models clip,dino
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._eval_helpers import (
    aggregate_metric_values,
    compute_iqa_metrics,
    compute_sigma_metrics,
    group_by_reference,
    save_results,
    save_summary,
)

from src.utils.experiment import init_experiment
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_DEFAULT_EVAL_DIR = Path("data/eval")
_DEFAULT_EXPERIMENTS_ROOT = Path("experiments")
_DEFAULT_SEED = 42
_DEFAULT_N_SAMPLES = 30
_DEFAULT_CONSISTENCY_BATCH_SIZE = 5
_SKIPPABLE_METRICS = frozenset({"clip", "dino", "lpips", "gram"})


# ---------------------------------------------------------------------------
# 헬퍼: 평가셋 로딩
# ---------------------------------------------------------------------------


def _load_eval_pairs(eval_dir: Path, n_samples: int) -> list[dict]:
    """평가 디렉토리에서 source/reference 페어 리스트를 만든다.

    명명 규약: ``source_001.png`` + ``reference_001.png`` 쌍.
    인덱스가 일치하는 파일끼리 페어로 묶는다.

    Args:
        eval_dir: 평가셋 디렉토리.
        n_samples: 최대 페어 수. 0이면 전체 사용.

    Returns:
        페어 dict 리스트. 각 원소는 ``{"pair_id": int, "source": Path, "reference": Path}``.

    Raises:
        FileNotFoundError: eval_dir가 존재하지 않을 때.
    """
    if not eval_dir.exists():
        raise FileNotFoundError(
            f"Eval directory not found: {eval_dir}. Run scripts/collect_data.py first."
        )

    pairs: list[dict] = []
    for src_path in sorted(eval_dir.glob("source_*.png")):
        suffix = src_path.stem[len("source_") :]
        ref_path = eval_dir / f"reference_{suffix}.png"
        if ref_path.exists():
            pairs.append({"pair_id": int(suffix), "source": src_path, "reference": ref_path})
        else:
            logger.warning("Reference not found for %s — skipping pair.", src_path.name)

    return pairs[:n_samples] if n_samples > 0 else pairs


# ---------------------------------------------------------------------------
# 헬퍼: 시스템 인스턴스화
# ---------------------------------------------------------------------------


def _instantiate_system(system_name: str, cfg: dict, exp_dir: Path) -> object:
    """system_name에 따라 변환 시스템 인스턴스를 반환한다.

    Args:
        system_name: ``"style_unifier"`` | ``"gpt_image"`` | ``"ipadapter_naive"``.
        cfg: load_config() 반환 딕셔너리.
        exp_dir: 현재 실험 디렉토리 (GPT cache 경로에 사용).

    Returns:
        ``transform(source, reference) -> Image.Image`` 를 가진 객체.

    Raises:
        ValueError: 알 수 없는 system_name일 때.
    """
    if system_name == "style_unifier":
        from src.generation.pipeline import StyleUnificationPipeline

        return StyleUnificationPipeline(cfg)

    if system_name == "gpt_image":
        from src.baselines.gpt_image import GPTImageBaseline

        return GPTImageBaseline(cache_dir=exp_dir / "cache_gpt")

    if system_name == "ipadapter_naive":
        from src.baselines.ipadapter_naive import IPAdapterNaiveBaseline

        return IPAdapterNaiveBaseline()

    raise ValueError(
        f"Unknown system: '{system_name}'. Valid options: style_unifier, gpt_image, ipadapter_naive, auto."
    )


# ---------------------------------------------------------------------------
# 메인 로직
# ---------------------------------------------------------------------------


def _run_transform_loop(
    pairs: list[dict],
    system: object,
    samples_dir: Path,
    skip_models: frozenset[str],
) -> tuple[list[dict], Sequence[object]]:
    """페어 목록 전체를 변환하고 메트릭을 계산해 결과를 반환한다.

    Args:
        pairs: 페어 dict 리스트.
        system: transform(source, ref) 메서드를 가진 시스템 객체.
        samples_dir: 출력 이미지 저장 디렉토리.
        skip_models: 생략할 메트릭 키 집합.

    Returns:
        (per_sample 리스트, 변환 성공 output_imgs 리스트).
    """
    from PIL import Image

    per_sample: list[dict] = []
    output_imgs: list[Image.Image] = []

    for idx, pair in enumerate(pairs):
        pair_id: int = pair["pair_id"]
        src_path: Path = pair["source"]
        ref_path: Path = pair["reference"]

        logger.info(
            "[%d/%d] Transforming pair %03d: %s", idx + 1, len(pairs), pair_id, src_path.name
        )

        source_img = Image.open(src_path)
        reference_img = Image.open(ref_path)

        try:
            raw_output = system.transform(source_img, reference_img)  # type: ignore[union-attr]
        except (RuntimeError, OSError, ValueError) as exc:
            logger.error("Transform failed for pair %03d: %s", pair_id, exc)
            per_sample.append(
                {
                    "pair_id": pair_id,
                    "source": str(src_path),
                    "reference": str(ref_path),
                    "output": None,
                    "metrics": {},
                    "error": str(exc),
                }
            )
            continue

        # RGBA 보장 (CLAUDE.md 원칙 §4)
        if hasattr(raw_output, "mode") and raw_output.mode != "RGBA":
            raw_output = raw_output.convert("RGBA")

        output_img: Image.Image = raw_output  # type: ignore[assignment]
        output_path = samples_dir / f"output_{pair_id:03d}.png"
        output_img.save(output_path, format="PNG")
        output_imgs.append(output_img)

        metrics = compute_iqa_metrics(source_img, reference_img, output_img, skip_models)
        logger.debug("pair %03d metrics: %s", pair_id, metrics)

        per_sample.append(
            {
                "pair_id": pair_id,
                "source": str(src_path),
                "reference": str(ref_path),
                "output": str(output_path),
                "metrics": metrics,
            }
        )

    return per_sample, output_imgs


# Module-level aliases for test backward compatibility.
# Logic lives in _eval_helpers to keep run_eval.py under 500 lines.
_group_by_reference = group_by_reference
_compute_sigma_metrics = compute_sigma_metrics


def run_eval(
    config_path: Path,
    system_name: str,
    eval_dir: Path,
    exp_name: str,
    seed: int,
    n_samples: int,
    consistency_batch_size: int,
    skip_models: frozenset[str],
    dry_run: bool,
    experiments_root: Path,
) -> int:
    """평가 파이프라인 전체를 실행한다.

    Args:
        config_path: configs/*.yaml 경로.
        system_name: ``"style_unifier"`` | ``"gpt_image"`` | ``"ipadapter_naive"`` | ``"auto"``.
        eval_dir: 평가셋 디렉토리.
        exp_name: 실험 이름 (실험 디렉토리 이름에 사용).
        seed: 재현성 seed.
        n_samples: 최대 페어 수. 0이면 전체.
        consistency_batch_size: sigma_* 메트릭 계산 그룹 크기.
        skip_models: 계산 생략할 메트릭 키 집합.
        dry_run: True면 변환·메트릭 없이 평가셋 스캔만 실행.
        experiments_root: 실험 루트 디렉토리.

    Returns:
        0이면 정상 종료. 1이면 오류.
    """
    from src.generation.pipeline import load_config

    logger.info("=== run_eval start ===")
    logger.info("system=%s | config=%s | dry_run=%s", system_name, config_path, dry_run)
    logger.info("eval_dir=%s | n_samples=%s | seed=%d", eval_dir, n_samples or "all", seed)

    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Config load failed: %s", exc)
        return 1

    if system_name == "auto":
        system_name = cfg.get("system", "style_unifier")
        logger.info("auto mode resolved to system: %s", system_name)

    exp_dir = init_experiment(name=exp_name, config=cfg, seed=seed, root=experiments_root)
    logger.info("Experiment dir: %s", exp_dir)

    try:
        pairs = _load_eval_pairs(eval_dir, n_samples=n_samples)
    except FileNotFoundError as exc:
        logger.error("No eval set found: %s. Run scripts/collect_data.py first.", exc)
        return 1

    if len(pairs) == 0:
        logger.error(
            "No pairs found in %s. Expected source_NNN.png / reference_NNN.png. Run scripts/collect_data.py first.",
            eval_dir,
        )
        return 1

    logger.info("Found %d eval pairs (n_samples limit=%s)", len(pairs), n_samples or "all")

    if dry_run:
        logger.info("[dry-run] Scanning only — no transform or metrics.")
        for pair in pairs:
            logger.info(
                "[dry-run] pair %03d: %s | %s",
                pair["pair_id"],
                pair["source"].name,
                pair["reference"].name,
            )
        logger.info("[dry-run] Scan complete. %d pairs would be processed.", len(pairs))
        return 0

    try:
        system = _instantiate_system(system_name, cfg, exp_dir)
    except ValueError as exc:
        logger.error("System instantiation failed: %s", exc)
        return 1
    except OSError as exc:
        logger.error("System init error (check API key / model path): %s", exc)
        return 1

    samples_dir = exp_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    per_sample, output_imgs = _run_transform_loop(pairs, system, samples_dir, skip_models)

    consistency_overall = _compute_sigma_metrics(per_sample, output_imgs, consistency_batch_size)

    iqa_keys = [
        "clip_style_similarity",
        "lpips_structure",
        "dino_identity",
        "palette_distance",
        "gram_matrix_distance",
    ]
    iqa_overall: dict[str, object] = {
        key: aggregate_metric_values([p["metrics"].get(key) for p in per_sample if "metrics" in p])
        for key in iqa_keys
    }

    results: dict = {
        "experiment": {
            "name": exp_dir.name,
            "system": system_name,
            "seed": seed,
            "n_samples": len(pairs),
            "consistency_batch_size": consistency_batch_size,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
        },
        "iqa_overall": iqa_overall,
        "consistency": consistency_overall,
        "per_sample": per_sample,
    }

    save_results(exp_dir, results)
    save_summary(exp_dir, results)

    logger.info("Eval done: results saved to %s", exp_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="style-unifier 평가 자동화 스크립트 (C5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python scripts/run_eval.py --config configs/default.yaml --dry-run\n"
            "  python scripts/run_eval.py --config configs/default.yaml --system gpt_image --n-samples 10\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        metavar="PATH",
        help="configs/*.yaml 경로 (기본: configs/default.yaml)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="style_unifier",
        choices=["style_unifier", "gpt_image", "ipadapter_naive", "auto"],
        help="평가 대상 시스템 (기본: style_unifier)",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=_DEFAULT_EVAL_DIR,
        metavar="PATH",
        dest="eval_set",
        help="평가셋 디렉토리 (기본: data/eval/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="TEXT",
        help="실험 이름. 미지정 시 system 이름 기반 자동 생성",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="재현성 seed (기본: 42)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=_DEFAULT_N_SAMPLES,
        metavar="INT",
        dest="n_samples",
        help="평가 페어 개수 상한. 0이면 eval-set 전체 사용 (기본: 30)",
    )
    parser.add_argument(
        "--consistency-batch-size",
        type=int,
        default=_DEFAULT_CONSISTENCY_BATCH_SIZE,
        metavar="INT",
        dest="consistency_batch_size",
        help="sigma_* 메트릭 계산용 batch group 크기 (기본: 5)",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default="",
        metavar="LIST",
        dest="skip_models",
        help="계산 생략할 메트릭 (콤마 구분, 예: clip,dino,lpips,gram)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        dest="dry_run",
        help="실제 변환·메트릭 없이 평가셋 스캔만 수행 (sanity check)",
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    _raw_skip = _args.skip_models.strip()
    _skip: frozenset[str] = frozenset()
    if _raw_skip:
        _parsed = frozenset(s.strip().lower() for s in _raw_skip.split(",") if s.strip())
        _invalid = _parsed - _SKIPPABLE_METRICS
        if _invalid:
            logger.warning(
                "Unknown skip-models values (will be ignored): %s. Valid values: %s",
                _invalid,
                sorted(_SKIPPABLE_METRICS),
            )
        _skip = _parsed & _SKIPPABLE_METRICS

    _exp_name: str = _args.name if _args.name else f"eval_{_args.system}"

    _exit_code = run_eval(
        config_path=_args.config,
        system_name=_args.system,
        eval_dir=_args.eval_set,
        exp_name=_exp_name,
        seed=_args.seed,
        n_samples=_args.n_samples,
        consistency_batch_size=_args.consistency_batch_size,
        skip_models=_skip,
        dry_run=_args.dry_run,
        experiments_root=_DEFAULT_EXPERIMENTS_ROOT,
    )
    sys.exit(_exit_code)
