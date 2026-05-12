"""make_experiment_dir(), record_environment(), init_experiment() 단위 테스트."""

from __future__ import annotations

import json

import pytest
import torch

from src.utils.experiment import init_experiment, make_experiment_dir, record_environment


def test_make_experiment_dir_creates_directory(tmp_path):
    """make_experiment_dir() 호출 결과 디렉토리가 실제로 생성되는가."""
    exp_dir = make_experiment_dir("test_exp", root=tmp_path)
    assert exp_dir.exists()
    assert exp_dir.is_dir()


def test_make_experiment_dir_first_call_is_001(tmp_path):
    """빈 root에서 첫 호출이 001_{name} 형식으로 생성되는가."""
    exp_dir = make_experiment_dir("test_exp", root=tmp_path)
    assert exp_dir.name == "001_test_exp"


def test_make_experiment_dir_second_call_is_002(tmp_path):
    """두 번째 호출이 002_{name} 형식으로 생성되는가."""
    make_experiment_dir("first_exp", root=tmp_path)
    exp_dir = make_experiment_dir("second_exp", root=tmp_path)
    assert exp_dir.name == "002_second_exp"


def test_make_experiment_dir_creates_samples_subdir(tmp_path):
    """samples/ 하위 디렉토리가 함께 생성되는가."""
    exp_dir = make_experiment_dir("test_exp", root=tmp_path)
    assert (exp_dir / "samples").exists()
    assert (exp_dir / "samples").is_dir()


def test_make_experiment_dir_skips_used_ids(tmp_path):
    """001, 002, 003이 모두 있을 때 다음 호출은 004를 사용하는가."""
    (tmp_path / "001_foo").mkdir()
    (tmp_path / "002_bar").mkdir()
    (tmp_path / "003_baz").mkdir()

    exp_dir = make_experiment_dir("next_exp", root=tmp_path)
    assert exp_dir.name == "004_next_exp"


def test_make_experiment_dir_uses_max_plus_one_when_gap_exists(tmp_path):
    """001, 003이 있고 002가 없어도 다음 ID는 004 (max+1, 삭제된 ID 재사용 금지)."""
    (tmp_path / "001_foo").mkdir()
    (tmp_path / "003_baz").mkdir()

    exp_dir = make_experiment_dir("gap_exp", root=tmp_path)
    # ID 재사용 시 과거 참조가 다른 실험을 가리키게 되므로 max+1 = 004.
    assert exp_dir.name == "004_gap_exp"


def test_make_experiment_dir_raises_on_empty_name(tmp_path):
    """name이 빈 문자열이면 ValueError를 발생시키는가."""
    with pytest.raises(ValueError, match="must not be empty"):
        make_experiment_dir("", root=tmp_path)


def test_record_environment_creates_env_json(tmp_path):
    """record_environment() 호출 후 env.json 파일이 생성되는가."""
    exp_dir = make_experiment_dir("env_test", root=tmp_path)
    record_environment(exp_dir)
    assert (exp_dir / "env.json").exists()


def test_record_environment_env_json_is_valid_json(tmp_path):
    """생성된 env.json이 유효한 JSON으로 파싱되는가."""
    exp_dir = make_experiment_dir("env_json_valid", root=tmp_path)
    record_environment(exp_dir)

    content = (exp_dir / "env.json").read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


def test_record_environment_env_json_has_required_keys(tmp_path):
    """env.json에 python_version, platform, timestamp_utc 키가 존재하는가."""
    exp_dir = make_experiment_dir("env_keys", root=tmp_path)
    record_environment(exp_dir)

    parsed = json.loads((exp_dir / "env.json").read_text(encoding="utf-8"))
    assert "python_version" in parsed
    assert "platform" in parsed
    assert "timestamp_utc" in parsed


def test_record_environment_with_config_creates_config_yaml(tmp_path):
    """config 인자 전달 시 config.yaml이 생성되는가."""
    exp_dir = make_experiment_dir("config_yaml_test", root=tmp_path)
    record_environment(exp_dir, config={"seed": 42, "steps": 10})
    assert (exp_dir / "config.yaml").exists()


def test_record_environment_without_config_no_config_yaml(tmp_path):
    """config=None이면 config.yaml이 생성되지 않는가."""
    exp_dir = make_experiment_dir("no_config_yaml", root=tmp_path)
    record_environment(exp_dir, config=None)
    assert not (exp_dir / "config.yaml").exists()


def test_record_environment_returns_env_json_path(tmp_path):
    """record_environment() 반환값이 env.json의 Path인가."""
    exp_dir = make_experiment_dir("returns_env_path", root=tmp_path)
    env_path = record_environment(exp_dir)
    assert env_path == exp_dir / "env.json"


def test_init_experiment_creates_env_json(tmp_path):
    """init_experiment() 호출 후 env.json이 존재하는가."""
    exp_dir = init_experiment("seeded", seed=42, root=tmp_path)
    assert (exp_dir / "env.json").exists()


def test_init_experiment_seed_is_applied(tmp_path):
    """init_experiment(seed=42) 후 torch 랜덤 상태가 seed 42로 초기화되는가."""
    from src.utils.repro import set_seed

    init_experiment("seeded_a", seed=42, root=tmp_path)
    val_after_init = torch.rand(1).item()

    # 동일 seed를 직접 설정 후 값 비교
    set_seed(42)
    val_direct = torch.rand(1).item()

    assert val_after_init == pytest.approx(val_direct)


def test_init_experiment_no_seed_does_not_raise(tmp_path):
    """seed=None으로 호출해도 예외가 발생하지 않는가."""
    exp_dir = init_experiment("no_seed_exp", seed=None, root=tmp_path)
    assert exp_dir.exists()


def test_init_experiment_returns_exp_dir_path(tmp_path):
    """init_experiment() 반환값이 생성된 실험 디렉토리 Path인가."""
    exp_dir = init_experiment("returns_path", seed=1, root=tmp_path)
    assert exp_dir.is_dir()
    assert exp_dir.parent == tmp_path
