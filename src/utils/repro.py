"""재현성 헬퍼 유틸리티.

실험 결과의 bit-exact 재현을 위해 모든 랜덤 소스에 동일한 seed를 설정한다.
모든 실험 코드는 반드시 set_seed()를 먼저 호출해야 한다.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.utils.device import get_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """모든 랜덤 소스에 seed를 설정한다.

    ``random``, ``numpy``, ``torch``, CUDA 모두 동일한 seed로 초기화한다.
    이 함수를 호출한 이후의 연산은 동일 seed에서 동일 결과를 보장한다.

    Args:
        seed: 사용할 시드 값. 0 이상의 정수를 권장.

    Example:
        >>> set_seed(42)
        >>> # 이후 모든 랜덤 연산이 결정론적으로 동작
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seed set to %d", seed)


def make_generator(seed: int, device: torch.device | None = None) -> torch.Generator:
    """seed가 설정된 torch.Generator를 반환한다.

    diffusers 파이프라인의 ``generator`` 인자에 전달하여 이미지 생성 재현성을
    보장할 때 사용한다.

    Args:
        seed: 생성기에 설정할 시드 값.
        device: Generator를 생성할 device. ``None`` 이면 ``get_device()`` 결과를 사용.

    Returns:
        시드가 설정된 ``torch.Generator`` 인스턴스.

    Example:
        >>> generator = make_generator(seed=42)
        >>> image = pipe(..., generator=generator).images[0]
    """
    if device is None:
        device = get_device()
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.debug("Generator created: seed=%d, device=%s", seed, device)
    return generator


def set_deterministic(deterministic: bool = True) -> None:
    """PyTorch의 결정론적 알고리즘 모드를 토글한다.

    ``True`` 로 설정하면 ``torch.use_deterministic_algorithms(True)`` 와
    ``cudnn.deterministic = True`` 가 활성화된다. 일부 연산(ex. UpSample)이
    결정론적 구현을 제공하지 않아 RuntimeError를 던질 수 있으므로,
    CUBLAS_WORKSPACE_CONFIG 환경변수도 함께 설정한다.

    주의: 결정론적 모드는 성능을 저하시킬 수 있다. 벤치마크/실험 최종 실행에만
    사용하고, 빠른 탐색 단계에서는 비활성화(``False``)를 권장.

    부작용: ``deterministic=True`` 호출이 ``CUBLAS_WORKSPACE_CONFIG`` 환경변수를
    프로세스 전역에 설정한다. 이후 ``deterministic=False`` 로 되돌려도 환경변수는
    해제되지 않으며, 프로세스 재시작 없이는 원복할 수 없다. 동일 프로세스에서
    CUBLAS를 사용하는 다른 코드의 메모리·성능 특성에 영향을 줄 수 있다.

    Args:
        deterministic: ``True`` 면 결정론적 모드 활성화, ``False`` 면 비활성화.

    Example:
        >>> set_deterministic(True)
        >>> # 이후 연산이 완전 결정론적으로 동작 (단, 속도 저하)
        >>> set_deterministic(False)
        >>> # 성능 우선 모드로 복귀
    """
    torch.backends.cudnn.deterministic = deterministic  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = not deterministic  # type: ignore[attr-defined]

    if deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        # CUBLAS_WORKSPACE_CONFIG 없으면 일부 CUDA 연산이 RuntimeError를 던짐
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.debug("CUBLAS_WORKSPACE_CONFIG set to :4096:8")

    try:
        torch.use_deterministic_algorithms(deterministic)
    except RuntimeError as exc:
        msg = (
            "torch.use_deterministic_algorithms(%s) raised RuntimeError: %s."
            " Some ops may not have deterministic implementations."
        )
        logger.warning(msg, deterministic, exc)

    logger.info("Deterministic mode: %s", deterministic)
