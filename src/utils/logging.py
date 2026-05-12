"""프로젝트 공통 로거 팩토리.

모든 모듈에서 get_logger(__name__)로 일관된 포맷의 로거를 사용한다.
환경변수 STYLE_UNIFIER_LOG_LEVEL로 레벨 override 가능.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_ENV_LEVEL_KEY = "STYLE_UNIFIER_LOG_LEVEL"


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 반환한다.

    중복 핸들러 추가를 방지하는 가드가 포함되어 있어 반복 호출에도 안전하다.
    환경변수 ``STYLE_UNIFIER_LOG_LEVEL`` 이 설정되어 있으면 해당 레벨로 override한다.

    Args:
        name: 로거 이름. 각 모듈에서 ``__name__`` 을 전달하는 것을 권장.

    Returns:
        설정이 완료된 ``logging.Logger`` 인스턴스.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Pipeline initialized")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        level_name = os.environ.get(_ENV_LEVEL_KEY, "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

    return logger
