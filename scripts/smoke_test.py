"""그룹 A6 — 인프라 스모크 테스트 스크립트.

Python/PyTorch 환경, src.utils 모듈, diffusers/transformers import,
실험 디렉토리 생성까지 단계별로 검증한다.
--full 플래그로 SDXL 모델 실제 로딩까지 시도할 수 있다 (Linux GPU 전용).
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 검증 단계
# ---------------------------------------------------------------------------


def _fmt_step(idx: int, total: int, label: str) -> str:
    return f"[{idx}/{total}] {label}"


def check_pytorch_env() -> tuple[bool, str]:
    """단계 1: Python / PyTorch 환경 확인.

    Returns:
        (success, detail_message) 튜플.
    """
    try:
        import torch

        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        cuda_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"

        detail = f"torch {torch_ver}, cuda={'available' if cuda_ok else 'unavailable'}"
        if cuda_ok:
            detail += f" ({cuda_name})"

        # Linux에서 CUDA가 없으면 실질적 FAIL (GPU 기대 환경)
        if platform.system() == "Linux" and not cuda_ok:
            logger.warning("Linux에서 CUDA를 감지하지 못했습니다. GPU 드라이버를 확인하세요.")
            return False, f"CUDA not available on Linux — {detail}"

        return True, detail
    except ImportError as exc:
        return False, f"torch import 실패: {exc}"


def check_device_util() -> tuple[bool, str]:
    """단계 2: src.utils.get_device / get_dtype 동작 확인.

    Returns:
        (success, detail_message) 튜플.
    """
    try:
        from src.utils import get_device, get_dtype

        device = get_device()
        dtype = get_dtype(device)
        return True, f"{device} / dtype={dtype}"
    except ImportError as exc:
        return False, f"import 실패: {exc}"
    except RuntimeError as exc:
        return False, f"RuntimeError: {exc}"


def check_repro_util() -> tuple[bool, str]:
    """단계 3: set_seed 재현성 확인.

    동일 seed로 두 번 호출한 결과가 element-wise로 같은지 검증한다.

    Returns:
        (success, detail_message) 튜플.
    """
    try:
        import torch

        from src.utils import set_seed

        set_seed(42)
        a = torch.randn(3)
        set_seed(42)
        b = torch.randn(3)

        if torch.allclose(a, b):
            return True, f"seed=42 재현 확인 ({a.tolist()})"
        return False, f"결과 불일치: {a.tolist()} vs {b.tolist()}"
    except ImportError as exc:
        return False, f"import 실패: {exc}"
    except RuntimeError as exc:
        return False, f"RuntimeError: {exc}"


def check_diffusers_import() -> tuple[bool, str]:
    """단계 4: diffusers / transformers 버전 확인.

    Returns:
        (success, detail_message) 튜플.
    """
    msgs: list[str] = []
    all_ok = True

    try:
        import diffusers

        msgs.append(f"diffusers {diffusers.__version__}")
    except ImportError as exc:
        msgs.append(f"diffusers 실패: {exc}")
        all_ok = False

    try:
        import transformers

        msgs.append(f"transformers {transformers.__version__}")
    except ImportError as exc:
        msgs.append(f"transformers 실패: {exc}")
        all_ok = False

    return all_ok, ", ".join(msgs)


def check_sdxl_load() -> tuple[bool, str]:
    """단계 5: SDXL 파이프라인 실제 로딩 확인.

    ``enable_model_cpu_offload()`` 로 VRAM 절약. Linux + GPU 전용 검증.

    Returns:
        (success, detail_message) 튜플.
    """
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline  # pyright: ignore[reportPrivateImportUsage]

        from src.utils import get_device, get_dtype

        device = get_device()
        dtype = get_dtype(device)

        logger.info("SDXL 로딩 시작 (이 단계는 수십 초 소요될 수 있습니다)")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant="fp16" if device.type == "cuda" else None,
        )
        pipe.enable_model_cpu_offload()

        logger.info("SDXL 로딩 완료, 메모리 해제 중")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True, f"SDXL 로딩 성공 (device={device}, dtype={dtype})"
    except ImportError as exc:
        return False, f"import 실패: {exc}"
    except OSError as exc:
        return False, f"모델 파일 없음 (먼저 download_models.py 실행): {exc}"
    except RuntimeError as exc:
        return False, f"RuntimeError (OOM 가능성): {exc}"


def check_experiment_dir() -> tuple[bool, str]:
    """단계 6: init_experiment 정상 동작 확인.

    임시 디렉토리에서 실험 디렉토리 생성을 검증하고 정리한다.

    Returns:
        (success, detail_message) 튜플.
    """
    try:
        import shutil

        from src.utils import init_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir) / "experiments"
            exp_dir = init_experiment("smoke_test", seed=42, root=tmp_root)

            env_json = exp_dir / "env.json"
            samples_dir = exp_dir / "samples"

            if not env_json.exists():
                return False, "env.json 미생성"
            if not samples_dir.is_dir():
                return False, "samples/ 디렉토리 미생성"

            shutil.rmtree(tmpdir, ignore_errors=True)

        return True, f"실험 디렉토리 생성 및 env.json 확인: {exp_dir.name}"
    except ImportError as exc:
        return False, f"import 실패: {exc}"
    except OSError as exc:
        return False, f"OSError: {exc}"
    except ValueError as exc:
        return False, f"ValueError: {exc}"


# ---------------------------------------------------------------------------
# 실행 조율
# ---------------------------------------------------------------------------


def run_smoke_test(full: bool) -> int:
    """스모크 테스트 전체 단계를 순서대로 실행한다.

    Args:
        full: ``True`` 이면 SDXL 실제 로딩(단계 5)도 실행.

    Returns:
        실패 단계 수. 0이면 전체 통과.
    """
    total = 6

    steps = [
        ("Python/PyTorch env", check_pytorch_env),
        ("Device util (get_device / get_dtype)", check_device_util),
        ("Repro util (set_seed 재현성)", check_repro_util),
        ("diffusers / transformers import", check_diffusers_import),
        ("SDXL pipeline load", None),  # 특수 처리
        ("Experiment dir creation", check_experiment_dir),
    ]

    fail_count = 0

    for idx, (label, checker) in enumerate(steps, start=1):
        prefix = _fmt_step(idx, total, label)

        # 단계 5: SDXL 로딩
        if checker is None:
            if not full:
                logger.info("%s... SKIPPED (--full 플래그 없음)", prefix)
                continue
            checker = check_sdxl_load

        success, detail = checker()

        if success:
            logger.info("%s... PASS (%s)", prefix, detail)
        else:
            logger.error("%s... FAIL (%s)", prefix, detail)
            fail_count += 1

    logger.info("")  # 빈 줄 구분

    if fail_count == 0:
        logger.info("All checks passed")
    else:
        logger.error("%d check(s) failed", fail_count)

    return fail_count


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="style-unifier 인프라 스모크 테스트 (그룹 A6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python scripts/smoke_test.py                  # 빠른 환경 검증\n"
            "  python scripts/smoke_test.py --full           # SDXL 실제 로딩 포함\n"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="SDXL 파이프라인 실제 로딩까지 시도 (Linux GPU 전용, 느림)",
    )
    group.add_argument(
        "--skip-model-load",
        action="store_true",
        default=False,
        help="단계 5(SDXL 로딩) 건너뜀. 기본 동작과 동일 (명시적 사용 시)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # --skip-model-load는 기본 동작과 동일 (full=False)
    full_mode = args.full and not args.skip_model_load

    fail_count = run_smoke_test(full=full_mode)
    sys.exit(1 if fail_count > 0 else 0)
