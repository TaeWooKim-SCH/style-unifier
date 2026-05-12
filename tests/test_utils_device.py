"""get_device(), get_dtype() 단위 테스트."""

from __future__ import annotations

import torch

from src.utils.device import get_device, get_dtype


def test_get_device_returns_torch_device():
    """get_device() 반환값이 torch.device 인스턴스인가."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_dtype_cuda_returns_float16():
    """CUDA device에 대해 torch.float16을 반환하는가."""
    assert get_dtype(torch.device("cuda")) == torch.float16


def test_get_dtype_mps_returns_float16():
    """MPS device에 대해 torch.float16을 반환하는가."""
    assert get_dtype(torch.device("mps")) == torch.float16


def test_get_dtype_cpu_returns_float32():
    """CPU device에 대해 torch.float32를 반환하는가."""
    assert get_dtype(torch.device("cpu")) == torch.float32


def test_get_device_selects_cuda_when_available(monkeypatch):
    """CUDA가 available하면 cuda device를 반환하는가."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "MockGPU")

    device = get_device()
    assert device.type == "cuda"


def test_get_device_selects_mps_when_cuda_unavailable(monkeypatch):
    """CUDA가 없고 MPS가 available하면 mps device를 반환하는가."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = get_device()
    assert device.type == "mps"


def test_get_device_selects_cpu_when_no_gpu(monkeypatch):
    """CUDA, MPS 모두 없으면 cpu device를 반환하는가."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = get_device()
    assert device.type == "cpu"
