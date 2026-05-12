"""get_logger() 단위 테스트."""

from __future__ import annotations

import logging

from src.utils.logging import get_logger


def test_get_logger_returns_logger_instance():
    """get_logger() 반환값이 logging.Logger 인스턴스인가."""
    logger = get_logger("test.returns_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_same_name_no_duplicate_handlers():
    """같은 이름으로 두 번 호출해도 핸들러가 1개만 추가되는가."""
    name = "test.no_dup_handlers"
    # 기존 상태 초기화
    existing = logging.getLogger(name)
    existing.handlers.clear()

    get_logger(name)
    get_logger(name)

    logger = logging.getLogger(name)
    assert len(logger.handlers) == 1


def test_get_logger_env_level_debug_applied(monkeypatch):
    """STYLE_UNIFIER_LOG_LEVEL=DEBUG 시 logger level이 DEBUG로 적용되는가."""
    monkeypatch.setenv("STYLE_UNIFIER_LOG_LEVEL", "DEBUG")
    # 이미 핸들러가 등록된 logger와 충돌하지 않도록 새 이름 사용
    logger = get_logger("test.env_level_debug_unique_xyz")
    assert logger.level == logging.DEBUG


def test_get_logger_default_level_is_info():
    """환경변수 미설정 시 기본 level이 INFO인가."""
    import os

    # 환경변수가 없는 상태에서 새 이름으로 logger 생성
    original = os.environ.pop("STYLE_UNIFIER_LOG_LEVEL", None)
    try:
        logger = get_logger("test.default_level_info_unique_abc")
        assert logger.level == logging.INFO
    finally:
        if original is not None:
            os.environ["STYLE_UNIFIER_LOG_LEVEL"] = original


def test_get_logger_has_stream_handler():
    """반환된 logger에 StreamHandler가 포함되어 있는가."""
    name = "test.has_stream_handler"
    existing = logging.getLogger(name)
    existing.handlers.clear()

    logger = get_logger(name)
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types
