"""Project logger shim.

Uses loguru when installed; otherwise falls back to stdlib logging
while keeping a compatible subset of the loguru API used in this repo.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


class _StdLoggerAdapter:
    """Minimal loguru-like adapter for environments without loguru."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("cad_pipeline")
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(message, *args, **kwargs)

    # loguru convenience alias
    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message, *args, **kwargs)

    # loguru API compatibility (best-effort)
    def add(self, sink: Any = None, level: str | int = "INFO", **kwargs: Any) -> int:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        self._logger.setLevel(level)
        return 1

    def remove(self, handler_id: int | None = None) -> None:
        if handler_id is None:
            self._logger.handlers.clear()
            return
        if self._logger.handlers:
            self._logger.handlers.clear()


try:
    from loguru import logger as _loguru_logger  # type: ignore

    logger = _loguru_logger
except Exception:  # pragma: no cover
    logger = _StdLoggerAdapter()
