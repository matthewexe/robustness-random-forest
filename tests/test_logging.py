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


def test_get_logger_has_two_handlers():
    """Test that get_logger configures two stdout handlers."""
    logger = get_logger("test_two_handlers")
    
    # Check that we have stdout handlers
    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(stream_handlers) >= 2  # Should have at least 2 handlers
    
    # Check that handlers write to stdout
    for handler in stream_handlers:
        assert handler.stream == sys.stdout


def test_get_logger_has_classic_and_detailed_handlers():
    """Test that get_logger has both classic and detailed format handlers."""
    logger = get_logger("test_both_formats")
    
    # Find handlers by their format
    classic_handlers = []
    detailed_handlers = []
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.formatter:
            format_str = handler.formatter._fmt
            if CLASSIC_LOG_FORMAT in format_str:
                classic_handlers.append(handler)
            elif DETAILED_LOG_FORMAT in format_str:
                detailed_handlers.append(handler)
    
    assert len(classic_handlers) >= 1, "Should have at least one classic handler"
    assert len(detailed_handlers) >= 1, "Should have at least one detailed handler"


def test_get_logger_classic_handler_uses_info_level():
    """Test that the classic handler uses INFO level."""
    logger = get_logger("test_classic_level")
    
    # Find classic handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.formatter:
            format_str = handler.formatter._fmt
            if CLASSIC_LOG_FORMAT in format_str:
                assert handler.level == logging.INFO


def test_get_logger_detailed_handler_uses_debug_level():
    """Test that the detailed handler uses DEBUG level."""
    logger = get_logger("test_detailed_level")
    
    # Find detailed handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.formatter:
            format_str = handler.formatter._fmt
            if DETAILED_LOG_FORMAT in format_str:
                assert handler.level == logging.DEBUG


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


def test_logger_outputs_both_formats():
    """Test that the logger outputs in both formats for INFO level."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test info message"
    
    # Create a logger and replace handlers with our capture
    logger = logging.getLogger("test_both_outputs")
    logger.handlers.clear()  # Clear any existing handlers
    
    # Add classic handler
    classic_handler = logging.StreamHandler(captured_output)
    classic_handler.setLevel(logging.INFO)
    classic_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
    classic_handler.setFormatter(classic_formatter)
    logger.addHandler(classic_handler)
    
    # Add detailed handler
    detailed_handler = logging.StreamHandler(captured_output)
    detailed_handler.setLevel(logging.DEBUG)
    detailed_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
    detailed_handler.setFormatter(detailed_formatter)
    logger.addHandler(detailed_handler)
    
    logger.setLevel(logging.DEBUG)
    
    # Log an info message
    logger.info(test_message)
    
    output = captured_output.getvalue()
    
    # The message should appear twice - once in classic format, once in detailed
    # Count occurrences of the test message
    message_count = output.count(test_message)
    assert message_count == 2, f"Expected message to appear twice, but appeared {message_count} times"
    
    # Check that both formats are present
    assert "INFO" in output
    assert "test_both_outputs" in output


def test_logger_debug_only_in_detailed():
    """Test that DEBUG messages only appear in detailed format."""
    # Capture stdout
    captured_output = StringIO()
    test_message = "Test debug message"
    
    # Create a logger and replace handlers with our capture
    logger = logging.getLogger("test_debug_detailed")
    logger.handlers.clear()  # Clear any existing handlers
    
    # Add classic handler (INFO level - should NOT show DEBUG)
    classic_handler = logging.StreamHandler(captured_output)
    classic_handler.setLevel(logging.INFO)
    classic_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
    classic_handler.setFormatter(classic_formatter)
    logger.addHandler(classic_handler)
    
    # Add detailed handler (DEBUG level - should show DEBUG)
    detailed_handler = logging.StreamHandler(captured_output)
    detailed_handler.setLevel(logging.DEBUG)
    detailed_formatter = logging.Formatter(DETAILED_LOG_FORMAT)
    detailed_handler.setFormatter(detailed_formatter)
    logger.addHandler(detailed_handler)
    
    logger.setLevel(logging.DEBUG)
    
    # Log a debug message
    logger.debug(test_message)
    
    output = captured_output.getvalue()
    
    # The message should appear once - only in detailed format
    message_count = output.count(test_message)
    assert message_count == 1, f"Expected message to appear once (detailed only), but appeared {message_count} times"
    
    # Check that detailed format markers are present
    assert "DEBUG" in output
    assert "test_logging.py:" in output  # filename:lineno pattern from detailed format


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
