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


class LevelBasedFormatter(logging.Formatter):
    """
    Custom formatter that uses different formats based on log level.
    
    - DEBUG level: detailed format with filename, line number, and function name
    - INFO and above: classic format with basic information
    """
    
    def __init__(self):
        super().__init__()
        self.classic_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
        self.detailed_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
    
    def format(self, record):
        """Format the log record using appropriate formatter based on level."""
        if record.levelno == logging.DEBUG:
            return self.detailed_formatter.format(record)
        else:
            return self.classic_formatter.format(record)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with level-based format configuration.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with a stdout handler that uses different
    formats based on the log level:
    - DEBUG level: detailed format with filename, line number, and function name
    - INFO and above: classic format with timestamp, name, level, and message

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance with level-based formatting

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing started")  # Classic format
        >>> logger.debug("Debug info")  # Detailed format
    """
    logger = logging.getLogger(name)
    
    # Check if our custom handler already exists to avoid duplicates
    has_custom_handler = False
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            # Use custom attribute to identify our handler
            if hasattr(handler, '_is_level_based_handler'):
                has_custom_handler = True
                break
    
    # Add handler if it doesn't exist
    if not has_custom_handler:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)  # Allow all levels through
        handler.setFormatter(LevelBasedFormatter())
        handler._is_level_based_handler = True  # Custom attribute for identification
        logger.addHandler(handler)
    
    # Set logger level to DEBUG to allow all messages through
    logger.setLevel(logging.DEBUG)
    
    return logger


__all__ = [
    "get_logger",
    "CLASSIC_LOG_FORMAT",
    "CLASSIC_LOG_LEVEL",
    "DETAILED_LOG_FORMAT",
    "DETAILED_LOG_LEVEL",
]
