"""Run-scoped file handlers keep concurrent runs' logs separate."""

from __future__ import annotations

import asyncio
import logging

from renewal.logging_setup import attach_file_log, detach_file_log, run_log_scope


async def test_scoped_handlers_are_isolated(tmp_path):
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    handler_a = attach_file_log(tmp_path / "a.log", "DEBUG", run_key="a")
    handler_b = attach_file_log(tmp_path / "b.log", "DEBUG", run_key="b")
    log = logging.getLogger("renewal.test.scope")

    async def emit(key: str, message: str) -> None:
        with run_log_scope(key):
            log.warning(message)

    try:
        await asyncio.gather(emit("a", "message-for-a"), emit("b", "message-for-b"))
    finally:
        detach_file_log(handler_a)
        detach_file_log(handler_b)
        root.setLevel(previous_level)

    text_a = (tmp_path / "a.log").read_text()
    text_b = (tmp_path / "b.log").read_text()
    assert "message-for-a" in text_a
    assert "message-for-b" not in text_a
    assert "message-for-b" in text_b
    assert "message-for-a" not in text_b


async def test_unscoped_records_are_captured(tmp_path):
    """A handler still works when no scope is active (direct callers/tests)."""
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.DEBUG)
    handler = attach_file_log(tmp_path / "plain.log", "DEBUG", run_key="a")
    log = logging.getLogger("renewal.test.unscoped")
    try:
        log.warning("unscoped-message")
    finally:
        detach_file_log(handler)
        root.setLevel(previous_level)

    assert "unscoped-message" in (tmp_path / "plain.log").read_text()
