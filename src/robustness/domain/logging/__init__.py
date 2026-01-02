"""
Logging module for the robustness package.

This module provides a configured logger with stdout output.
"""

import logging
import sys
from typing import Optional


# Default logging configuration
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO


def _configure_stdout_handler(
    logger: logging.Logger,
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT
) -> None:
    """
    Configure a stdout handler for the logger.

    Args:
        logger: The logger instance to configure
        level: The logging level (default: INFO)
        log_format: The format string for log messages
    """
    # Check if handler already exists to avoid duplicates
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)


def get_logger(
    name: Optional[str] = None,
    level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT
) -> logging.Logger:
    """
    Get a logger with stdout configuration.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with a stdout handler.

    Args:
        name: The name of the logger. If None, returns the root logger.
        level: The logging level (default: INFO)
        log_format: The format string for log messages

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    _configure_stdout_handler(logger, level, log_format)
    return logger


__all__ = ["get_logger", "DEFAULT_LOG_FORMAT", "DEFAULT_LOG_LEVEL"]
