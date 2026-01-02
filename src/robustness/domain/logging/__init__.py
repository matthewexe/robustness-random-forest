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


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with both classic and detailed stdout handlers.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with two stdout handlers:
    1. Classic handler: INFO level with timestamp, name, level, and message
    2. Detailed handler: DEBUG level with additional info like filename,
       line number, and function name for debugging

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance with both classic and detailed handlers

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing started")  # Output in both formats
        >>> logger.debug("Debug info")  # Only in detailed format (DEBUG level)
    """
    logger = logging.getLogger(name)
    
    # Check if handlers already exist to avoid duplicates
    has_classic_handler = False
    has_detailed_handler = False
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            formatter_str = handler.formatter._fmt if handler.formatter else ""
            if CLASSIC_LOG_FORMAT in formatter_str:
                has_classic_handler = True
            elif DETAILED_LOG_FORMAT in formatter_str:
                has_detailed_handler = True
    
    # Add classic handler if it doesn't exist
    if not has_classic_handler:
        classic_handler = logging.StreamHandler(sys.stdout)
        classic_handler.setLevel(CLASSIC_LOG_LEVEL)
        classic_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
        classic_handler.setFormatter(classic_formatter)
        logger.addHandler(classic_handler)
    
    # Add detailed handler if it doesn't exist
    if not has_detailed_handler:
        detailed_handler = logging.StreamHandler(sys.stdout)
        detailed_handler.setLevel(DETAILED_LOG_LEVEL)
        detailed_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
        detailed_handler.setFormatter(detailed_formatter)
        logger.addHandler(detailed_handler)
    
    # Set logger level to DEBUG to allow all messages through
    # (handlers will filter based on their own levels)
    logger.setLevel(logging.DEBUG)
    
    return logger


__all__ = [
    "get_logger",
    "CLASSIC_LOG_FORMAT",
    "CLASSIC_LOG_LEVEL",
    "DETAILED_LOG_FORMAT",
    "DETAILED_LOG_LEVEL",
]
