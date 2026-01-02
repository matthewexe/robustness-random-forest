"""
Logging module for the robustness package.

This module provides a configured logger with stdout output.
"""

import logging
import sys
from typing import Optional


# Classic logging configuration (INFO level)
CLASSIC_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CLASSIC_LOG_LEVEL = logging.INFO

# Detailed logging configuration (DEBUG level with function, line number, etc.)
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
DETAILED_LOG_LEVEL = logging.DEBUG


def _configure_stdout_handler(
    logger: logging.Logger,
    level: int,
    log_format: str
) -> None:
    """
    Configure a stdout handler for the logger.

    Args:
        logger: The logger instance to configure
        level: The logging level
        log_format: The format string for log messages
    """
    # Check if stdout handler already exists to avoid duplicates
    has_stdout_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers
    )
    if not has_stdout_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with classic stdout configuration.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with a stdout handler using classic format.

    Classic format: INFO level with timestamp, name, level, and message.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance with classic format

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    _configure_stdout_handler(logger, CLASSIC_LOG_LEVEL, CLASSIC_LOG_FORMAT)
    return logger


def get_detailed_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with detailed stdout configuration.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with a stdout handler using detailed format.

    Detailed format: DEBUG level with additional info like filename,
    line number, and function name for debugging.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance with detailed format

    Example:
        >>> logger = get_detailed_logger("my_module")
        >>> logger.debug("Detailed debugging information")
    """
    logger = logging.getLogger(name)
    _configure_stdout_handler(logger, DETAILED_LOG_LEVEL, DETAILED_LOG_FORMAT)
    return logger


__all__ = [
    "get_logger",
    "get_detailed_logger",
    "CLASSIC_LOG_FORMAT",
    "CLASSIC_LOG_LEVEL",
    "DETAILED_LOG_FORMAT",
    "DETAILED_LOG_LEVEL",
]
