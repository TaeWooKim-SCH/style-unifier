"""실험 디렉토리 자동 생성 및 환경 기록 유틸리티.

experiments/NNN_name/ 형식의 디렉토리를 자동으로 할당하고,
재현에 필요한 환경 정보(git hash, python 버전, 설정 등)를 기록한다.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.utils.logging import get_logger
from src.utils.repro import set_seed

logger = get_logger(__name__)

_ID_WIDTH = 3  # 001, 002, ...


def _next_experiment_id(root: Path) -> int:
    """Root 디렉토리에서 다음 실험 ID를 반환한다.

    기존 디렉토리의 최대 ID + 1을 반환한다. 삭제된 ID를 재사용하지 않으므로
    과거 참조(보고서·노트의 "실험 002")가 다른 실험을 가리키게 되는 것을 막는다.
    """
    if not root.exists():
        return 1

    used_ids: set[int] = set()
    for child in root.iterdir():
        if child.is_dir():
            prefix = child.name.split("_")[0]
            if prefix.isdigit() and len(prefix) == _ID_WIDTH:
                used_ids.add(int(prefix))

    return max(used_ids, default=0) + 1


def make_experiment_dir(name: str, root: Path = Path("experiments")) -> Path:
    """다음 사용 가능한 ID를 할당하고 실험 디렉토리를 생성한다.

    ``{root}/{NNN}_{name}/`` 형식으로 생성하며, 하위에 ``samples/`` 도 함께 만든다.
    ID 충돌은 기존 디렉토리 목록을 스캔하여 방지한다.

    Args:
        name: 실험 이름. 짧고 식별 가능한 snake_case 권장 (예: ``ipadapter_sweep``).
        root: 실험 루트 디렉토리. 기본값은 프로젝트 루트의 ``experiments/``.

    Returns:
        생성된 실험 디렉토리의 ``Path``.

    Raises:
        ValueError: name이 빈 문자열인 경우.

    Example:
        >>> exp_dir = make_experiment_dir("ipadapter_sweep")
        >>> # experiments/001_ipadapter_sweep/ 이 생성됨
    """
    if not name:
        raise ValueError("Experiment name must not be empty.")

    exp_id = _next_experiment_id(root)
    dir_name = f"{exp_id:0{_ID_WIDTH}d}_{name}"
    exp_dir = root / dir_name

    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "samples").mkdir()

    logger.info("Experiment directory created: %s", exp_dir)
    return exp_dir


def _get_git_hash() -> str:
    """현재 HEAD의 git commit hash를 반환한다. 실패 시 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _is_git_dirty() -> bool:
    """워킹 트리에 uncommitted 변경이 있으면 True를 반환한다."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            check=False,
        )
        return result.returncode != 0
    except FileNotFoundError:
        return False


def record_environment(exp_dir: Path, config: dict | None = None) -> Path:
    """실험 환경 정보를 exp_dir/env.json에 기록한다.

    config가 주어지면 exp_dir/config.yaml 에도 저장한다.

    Args:
        exp_dir: 대상 실험 디렉토리. ``make_experiment_dir()`` 반환값을 사용할 것.
        config: 실험에 사용된 설정 딕셔너리. ``None`` 이면 config.yaml 생략.

    Returns:
        작성된 ``env.json`` 파일의 ``Path``.

    Example:
        >>> exp_dir = make_experiment_dir("baseline")
        >>> env_path = record_environment(exp_dir, config={"seed": 42})
    """
    torch_version: str = "unknown"
    torch_cuda_version: str = "unknown"
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_version = torch.version.cuda or "N/A"
    except ImportError:
        pass

    env: dict[str, object] = {
        "git_hash": _get_git_hash(),
        "git_dirty": _is_git_dirty(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda_version,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    env_path = exp_dir / "env.json"
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Environment recorded: %s", env_path)

    if config is not None:
        config_path = exp_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Config saved: %s", config_path)

    return env_path


def init_experiment(
    name: str,
    config: dict | None = None,
    seed: int | None = None,
    root: Path = Path("experiments"),
) -> Path:
    """실험 디렉토리 생성, 환경 기록, seed 설정을 한 번에 수행하는 편의 함수.

    내부적으로 ``make_experiment_dir`` → ``record_environment`` → ``set_seed``
    순서로 호출한다.

    Args:
        name: 실험 이름 (예: ``ipadapter_sweep``).
        config: 실험 설정 딕셔너리. ``None`` 이면 config.yaml 생략.
        seed: 재현성 시드. ``None`` 이면 seed 설정 생략.
        root: 실험 루트 디렉토리. 기본값은 ``experiments/``.

    Returns:
        생성된 실험 디렉토리의 ``Path``.

    Example:
        >>> exp_dir = init_experiment("palette_sweep", config=cfg, seed=42)
        >>> # experiments/001_palette_sweep/ 생성 + env.json + config.yaml + seed=42
    """
    exp_dir = make_experiment_dir(name, root=root)
    record_environment(exp_dir, config=config)

    if seed is not None:
        set_seed(seed)

    return exp_dir
