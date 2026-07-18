"""Centralized logging configuration using loguru.

Simple, elegant logging setup for WorldReasoner.
All modules can just import logger and use it.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


def is_finance_cli_invocation(argv: Sequence[str]) -> bool:
    """Return whether argv is the installed ``wr finance`` command."""
    if not argv or Path(argv[0]).name.casefold() not in {"wr", "wr.exe"}:
        return False
    for argument in argv[1:]:
        if argument == "--":
            return False
        if argument.startswith("-"):
            continue
        return argument == "finance"
    return False


def _shorten_name(record: Record) -> bool:
    """Filter to shorten module names in logs."""
    record["extra"]["short_name"] = (record["name"] or "").split(".")[-1]
    return True


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
    enable_file: bool = True,
) -> None:
    """Configure loguru logger with sensible defaults.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file. If None, uses default from constants
        rotation: When to rotate log file
        retention: How long to keep old logs
    """
    # Use default log file pattern if not specified
    if log_file is None:
        from src.config import LOGS_DIR

        log_file = str(LOGS_DIR / "worldreasoner_{time}.log")

    # Remove default handler
    _ = logger.remove()

    # Add console handler with nice formatting
    _ = logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[short_name]:<20}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        filter=_shorten_name,
    )

    if enable_file:
        # Add file handler with rotation only when file logging is enabled.
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        _ = logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging initialized at level {level}")


# Preserve legacy non-finance startup logging while keeping finance imports quiet.
if is_finance_cli_invocation(sys.argv):
    logger.remove()
else:
    setup_logging()


__all__ = ["is_finance_cli_invocation", "logger", "setup_logging"]
