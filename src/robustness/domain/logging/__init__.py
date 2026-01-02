"""
Logging module for the robustness package.

This module provides a configured logger with dual output:
- Classic format to stdout
- Detailed format to JSON file
"""

import json
import logging
import os
import sys
from typing import Optional


# Classic logging configuration (INFO level)
CLASSIC_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
CLASSIC_LOG_LEVEL = logging.INFO

# Detailed logging configuration (DEBUG level)
DETAILED_LOG_LEVEL = logging.DEBUG

# Default log file path
DEFAULT_LOG_FILE = "logs/robustness.json"


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.
    """
    
    def format(self, record):
        """Format the log record as JSON."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "filename": record.filename,
            "lineno": record.lineno,
            "funcName": record.funcName,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def get_logger(name: Optional[str] = None, log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    """
    Get a logger with dual handler configuration.

    This is a wrapper around logging.getLogger that automatically
    configures the logger with two handlers:
    1. Classic handler: stdout with INFO level and classic format
    2. Detailed handler: JSON file with DEBUG level and detailed format

    Args:
        name: The name of the logger. If None, returns the root logger.
        log_file: Path to the JSON log file (default: logs/robustness.json)

    Returns:
        A configured logger instance with dual handlers

    Example:
        >>> logger = get_logger("my_module")
        >>> logger.info("Processing started")  # Goes to stdout
        >>> logger.debug("Debug info")  # Goes to JSON file only
    """
    logger = logging.getLogger(name)
    
    # Check if handlers already exist to avoid duplicates
    has_stdout_handler = False
    has_file_handler = False
    
    for handler in logger.handlers:
        if hasattr(handler, '_handler_type'):
            if handler._handler_type == 'classic_stdout':
                has_stdout_handler = True
            elif handler._handler_type == 'detailed_json':
                has_file_handler = True
    
    # Add classic stdout handler if it doesn't exist
    if not has_stdout_handler:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(CLASSIC_LOG_LEVEL)  # INFO and above
        stdout_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
        stdout_handler.setFormatter(stdout_formatter)
        stdout_handler._handler_type = 'classic_stdout'
        logger.addHandler(stdout_handler)
    
    # Add detailed JSON file handler if it doesn't exist
    if not has_file_handler:
        # Create parent directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(DETAILED_LOG_LEVEL)  # DEBUG and above
        file_handler.setFormatter(JSONFormatter())
        file_handler._handler_type = 'detailed_json'
        logger.addHandler(file_handler)
    
    # Set logger level to DEBUG to allow all messages through
    # (handlers will filter based on their own levels)
    logger.setLevel(logging.DEBUG)
    
    return logger


__all__ = [
    "get_logger",
    "CLASSIC_LOG_FORMAT",
    "CLASSIC_LOG_LEVEL",
    "DETAILED_LOG_LEVEL",
    "JSONFormatter",
]
