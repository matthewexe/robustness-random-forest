"""
Tests for the logging module.
"""

import logging
import sys
from io import StringIO

from robustness.domain.logging import (
    get_logger,
    CLASSIC_LOG_FORMAT,
    CLASSIC_LOG_LEVEL,
    DETAILED_LOG_FORMAT,
    DETAILED_LOG_LEVEL,
)


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


def test_get_logger_classic_mode():
    """Test that classic mode uses INFO level."""
    logger = get_logger("test_classic_logger", mode="classic")
    assert logger.level == logging.INFO


def test_get_logger_detailed_mode():
    """Test that detailed mode uses DEBUG level."""
    logger = get_logger("test_detailed_logger", mode="detailed")
    assert logger.level == logging.DEBUG


def test_get_logger_default_mode_is_classic():
    """Test that default mode is classic (INFO level)."""
    logger = get_logger("test_default_logger")
    assert logger.level == logging.INFO


def test_get_logger_no_duplicate_handlers():
    """Test that calling get_logger multiple times doesn't add duplicate handlers."""
    logger_name = "test_no_duplicates"
    logger1 = get_logger(logger_name)
    initial_handler_count = len(logger1.handlers)
    
    logger2 = get_logger(logger_name)
    final_handler_count = len(logger2.handlers)
    
    assert initial_handler_count == final_handler_count


def test_logger_classic_output_format():
    """Test that the classic logger uses the correct format."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test classic log message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_classic_output_logger")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # Log a message
    logger.info(test_message)
    
    # Check that the message appears in the output with expected format
    output = captured_output.getvalue()
    assert test_message in output
    assert "INFO" in output
    assert "test_classic_output_logger" in output
    # Classic format should NOT include filename, lineno, or funcName
    assert "::" not in output  # filename:lineno separator
    assert "test_logging.py" not in output


def test_logger_detailed_output_format():
    """Test that the detailed logger includes function name and line number."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test detailed log message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_detailed_output_logger")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(DETAILED_LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Log a message
    logger.debug(test_message)
    
    # Check that the message appears in the output with detailed format
    output = captured_output.getvalue()
    assert test_message in output
    assert "DEBUG" in output
    assert "test_detailed_output_logger" in output
    # Detailed format should include filename, lineno, and funcName
    assert "test_logging.py" in output
    assert "test_logger_detailed_output_format()" in output or "test_logger_detailed_output_format" in output


def test_constant_values():
    """Test that constants are properly defined."""
    assert CLASSIC_LOG_LEVEL == logging.INFO
    assert DETAILED_LOG_LEVEL == logging.DEBUG
    assert isinstance(CLASSIC_LOG_FORMAT, str)
    assert isinstance(DETAILED_LOG_FORMAT, str)
    
    # Classic format should have basic fields
    assert "%(asctime)s" in CLASSIC_LOG_FORMAT
    assert "%(name)s" in CLASSIC_LOG_FORMAT
    assert "%(levelname)s" in CLASSIC_LOG_FORMAT
    assert "%(message)s" in CLASSIC_LOG_FORMAT
    
    # Detailed format should have additional debugging fields
    assert "%(asctime)s" in DETAILED_LOG_FORMAT
    assert "%(name)s" in DETAILED_LOG_FORMAT
    assert "%(levelname)s" in DETAILED_LOG_FORMAT
    assert "%(message)s" in DETAILED_LOG_FORMAT
    assert "%(filename)s" in DETAILED_LOG_FORMAT
    assert "%(lineno)d" in DETAILED_LOG_FORMAT
    assert "%(funcName)s" in DETAILED_LOG_FORMAT
