"""Device 및 dtype 자동 선택 유틸리티.

실행 환경에 따라 최적 device(CUDA > MPS > CPU)와 dtype을 반환한다.
코드 어디에서도 device를 하드코딩하지 말고 이 모듈을 사용할 것.
"""

from __future__ import annotations

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    """사용 가능한 최적 device를 반환한다.

    우선순위: CUDA > MPS > CPU.
    선택된 device는 INFO 레벨로 로깅된다.

    Returns:
        선택된 ``torch.device`` 인스턴스.

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device selected: cuda (%s)", torch.cuda.get_device_name(0))
        return device

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device selected: mps (Apple Silicon)")
        return device

    device = torch.device("cpu")
    logger.info("Device selected: cpu (no GPU acceleration available)")
    return device


def get_dtype(device: torch.device) -> torch.dtype:
    """device에 맞는 기본 float dtype을 반환한다.

    CUDA와 MPS는 fp16을 지원하므로 메모리 효율을 위해 float16을 반환한다.
    CPU는 float16 연산이 비효율적이므로 float32를 반환한다.

    Args:
        device: 대상 device. ``get_device()`` 반환값을 사용할 것.

    Returns:
        ``torch.float16`` (CUDA/MPS) 또는 ``torch.float32`` (CPU).

    Example:
        >>> device = get_device()
        >>> dtype = get_dtype(device)
        >>> model = model.to(device=device, dtype=dtype)
    """
    if device.type in ("cuda", "mps"):
        return torch.float16
    return torch.float32
