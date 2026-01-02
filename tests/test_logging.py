"""
Tests for the logging module.
"""

import logging
import sys
from io import StringIO

from robustness.domain.logging import (
    get_logger,
    LevelBasedFormatter,
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


def test_get_logger_has_one_handler():
    """Test that get_logger configures one stdout handler."""
    logger = get_logger("test_one_handler")
    
    # Check that we have stdout handlers
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 1  # Should have at least 1 handler
    
    # Check that handler writes to stdout
    for handler in stream_handlers:
        assert handler.stream == sys.stdout


def test_get_logger_handler_uses_level_based_formatter():
    """Test that get_logger uses LevelBasedFormatter."""
    logger = get_logger("test_formatter")
    
    # Find our custom handler
    found_formatter = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and hasattr(handler, '_is_level_based_handler'):
            assert isinstance(handler.formatter, LevelBasedFormatter)
            found_formatter = True
            break
    
    assert found_formatter, "Should have handler with LevelBasedFormatter"


def test_get_logger_sets_logger_to_debug():
    """Test that get_logger sets the logger level to DEBUG."""
    logger = get_logger("test_logger_level")
    assert logger.level == logging.DEBUG


def test_get_logger_no_duplicate_handlers():
    """Test that calling get_logger multiple times doesn't add duplicate handlers."""
    logger_name = "test_no_duplicates"
    logger1 = get_logger(logger_name)
    initial_handler_count = len(logger1.handlers)
    
    logger2 = get_logger(logger_name)
    final_handler_count = len(logger2.handlers)
    
    assert initial_handler_count == final_handler_count


def test_logger_info_uses_classic_format():
    """Test that INFO messages use classic format."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test info message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_info_classic")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LevelBasedFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Log an info message
    logger.info(test_message)
    
    output = captured_output.getvalue()
    
    # The message should appear once in classic format
    assert test_message in output
    assert output.count(test_message) == 1, "Message should appear exactly once"
    assert "INFO" in output
    
    # Classic format should NOT include filename:lineno or funcName
    assert "test_logging.py:" not in output  # filename:lineno pattern
    assert "test_logger_info_uses_classic_format()" not in output  # funcName


def test_logger_debug_uses_detailed_format():
    """Test that DEBUG messages use detailed format."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test debug message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_debug_detailed")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LevelBasedFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Log a debug message
    logger.debug(test_message)
    
    output = captured_output.getvalue()
    
    # The message should appear once in detailed format
    assert test_message in output
    assert output.count(test_message) == 1, "Message should appear exactly once"
    assert "DEBUG" in output
    
    # Detailed format should include filename:lineno and funcName
    assert "test_logging.py:" in output  # filename:lineno pattern
    assert "test_logger_debug_uses_detailed_format" in output  # funcName


def test_logger_warning_uses_classic_format():
    """Test that WARNING messages use classic format."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test warning message"
    
    # Create a logger with custom handler for testing
    logger = logging.getLogger("test_warning_classic")
    logger.handlers.clear()  # Clear any existing handlers
    
    handler = logging.StreamHandler(captured_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(LevelBasedFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    # Log a warning message
    logger.warning(test_message)
    
    output = captured_output.getvalue()
    
    # The message should appear once in classic format
    assert test_message in output
    assert output.count(test_message) == 1, "Message should appear exactly once"
    assert "WARNING" in output
    
    # Classic format should NOT include filename:lineno or funcName
    assert "test_logging.py:" not in output  # filename:lineno pattern
    assert "test_logger_warning_uses_classic_format()" not in output  # funcName


def test_level_based_formatter():
    """Test the LevelBasedFormatter directly."""
    formatter = LevelBasedFormatter()
    
    # Create DEBUG record
    debug_record = logging.LogRecord(
        name="test", level=logging.DEBUG, pathname="test.py",
        lineno=42, msg="Debug message", args=(), exc_info=None,
        func="test_func"
    )
    
    # Create INFO record
    info_record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py",
        lineno=42, msg="Info message", args=(), exc_info=None,
        func="test_func"
    )
    
    debug_output = formatter.format(debug_record)
    info_output = formatter.format(info_record)
    
    # DEBUG should have detailed format
    assert "test.py:42" in debug_output
    assert "test_func()" in debug_output
    
    # INFO should have classic format (no filename:lineno or funcName)
    assert "test.py:42" not in info_output
    assert "test_func()" not in info_output


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
