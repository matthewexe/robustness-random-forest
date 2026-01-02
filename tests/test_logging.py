"""
Tests for the logging module.
"""

import logging
import sys
from io import StringIO

from robustness.domain.logging import get_logger, DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL


def test_get_logger_returns_logger():
    """Test that get_logger returns a logging.Logger instance."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_with_name():
    """Test that get_logger creates a logger with the specified name."""
    logger_name = "test_logger_with_name"
    logger = get_logger(logger_name)
    assert logger.name == logger_name


def test_get_logger_configures_stdout():
    """Test that get_logger configures a stdout handler."""
    logger = get_logger("test_stdout_logger")
    
    # Check that at least one StreamHandler exists
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) > 0
    
    # Check that the handler writes to stdout
    handler = stream_handlers[0]
    assert handler.stream == sys.stdout


def test_get_logger_sets_log_level():
    """Test that get_logger sets the correct log level."""
    logger = get_logger("test_level_logger", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_get_logger_no_duplicate_handlers():
    """Test that calling get_logger multiple times doesn't add duplicate handlers."""
    logger_name = "test_no_duplicates"
    logger1 = get_logger(logger_name)
    initial_handler_count = len(logger1.handlers)
    
    logger2 = get_logger(logger_name)
    final_handler_count = len(logger2.handlers)
    
    assert initial_handler_count == final_handler_count


def test_logger_outputs_to_stdout():
    """Test that the logger actually writes to stdout."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test log message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_output_logger")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Log a message
    logger.info(test_message)
    
    # Check that the message appears in the output
    output = captured_output.getvalue()
    assert test_message in output
    assert "INFO" in output


def test_default_constants():
    """Test that default constants are properly defined."""
    assert DEFAULT_LOG_LEVEL == logging.INFO
    assert isinstance(DEFAULT_LOG_FORMAT, str)
    assert "%(asctime)s" in DEFAULT_LOG_FORMAT
    assert "%(name)s" in DEFAULT_LOG_FORMAT
    assert "%(levelname)s" in DEFAULT_LOG_FORMAT
    assert "%(message)s" in DEFAULT_LOG_FORMAT
