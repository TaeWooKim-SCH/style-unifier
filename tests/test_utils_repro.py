"""set_seed(), make_generator(), set_deterministic() 단위 테스트."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.repro import make_generator, set_deterministic, set_seed


def test_set_seed_torch_is_deterministic():
    """set_seed(42) 후 torch.rand(3)가 두 번 동일한 결과를 내는가."""
    set_seed(42)
    first = torch.rand(3).tolist()

    set_seed(42)
    second = torch.rand(3).tolist()

    assert first == second


def test_set_seed_numpy_is_deterministic():
    """set_seed(42) 후 numpy.random.rand(3)가 두 번 동일한 결과를 내는가."""
    set_seed(42)
    first = np.random.rand(3).tolist()

    set_seed(42)
    second = np.random.rand(3).tolist()

    assert first == second


def test_set_seed_random_is_deterministic():
    """set_seed(42) 후 random.random()이 두 번 동일한 결과를 내는가."""
    set_seed(42)
    first = random.random()

    set_seed(42)
    second = random.random()

    assert first == second


def test_make_generator_returns_torch_generator():
    """make_generator() 반환값이 torch.Generator 인스턴스인가."""
    gen = make_generator(42, device=torch.device("cpu"))
    assert isinstance(gen, torch.Generator)


def test_make_generator_same_seed_same_sequence():
    """동일 seed로 만든 두 generator가 같은 sequence를 생성하는가."""
    gen1 = make_generator(42, device=torch.device("cpu"))
    gen2 = make_generator(42, device=torch.device("cpu"))

    out1 = torch.randn(3, generator=gen1)
    out2 = torch.randn(3, generator=gen2)

    assert torch.equal(out1, out2)


def test_make_generator_different_seeds_differ():
    """다른 seed로 만든 generator는 다른 sequence를 생성하는가."""
    gen1 = make_generator(42, device=torch.device("cpu"))
    gen2 = make_generator(99, device=torch.device("cpu"))

    out1 = torch.randn(3, generator=gen1)
    out2 = torch.randn(3, generator=gen2)

    assert not torch.equal(out1, out2)


def test_set_deterministic_does_not_raise():
    """set_deterministic() 호출이 예외 없이 완료되는가."""
    set_deterministic(True)
    set_deterministic(False)


def test_set_deterministic_sets_cublas_env_when_unset(monkeypatch):
    """CUBLAS_WORKSPACE_CONFIG 미설정 상태에서 set_deterministic(True) 후 환경변수가 설정되는가."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    set_deterministic(True)

    assert "CUBLAS_WORKSPACE_CONFIG" in os.environ
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_set_deterministic_does_not_overwrite_existing_cublas_env(monkeypatch):
    """CUBLAS_WORKSPACE_CONFIG가 이미 설정된 경우 덮어쓰지 않는가."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":custom:value")

    set_deterministic(True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":custom:value"
