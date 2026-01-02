"""
Tests for the logging module.
"""

import json
import logging
import os
import sys
import tempfile
from io import StringIO

from robustness.domain.logging import (
    get_logger,
    JSONFormatter,
    CLASSIC_LOG_FORMAT,
    CLASSIC_LOG_LEVEL,
    DETAILED_LOG_LEVEL,
)


def test_get_logger_returns_logger():
    """Test that get_logger returns a logging.Logger instance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_logger", log_file=log_file)
        assert isinstance(logger, logging.Logger)


def test_get_logger_with_name():
    """Test that get_logger creates a logger with the specified name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger_name = "test_logger_with_name"
        logger = get_logger(logger_name, log_file=log_file)
        assert logger.name == logger_name


def test_get_logger_has_two_handlers():
    """Test that get_logger configures two handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_two_handlers", log_file=log_file)
        
        # Should have at least 2 handlers
        assert len(logger.handlers) >= 2


def test_get_logger_has_stdout_and_file_handlers():
    """Test that get_logger has both stdout and file handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_both_handlers", log_file=log_file)
        
        # Find handlers by their custom type attribute
        stdout_handlers = []
        file_handlers = []
        
        for handler in logger.handlers:
            if hasattr(handler, '_handler_type'):
                if handler._handler_type == 'classic_stdout':
                    stdout_handlers.append(handler)
                elif handler._handler_type == 'detailed_json':
                    file_handlers.append(handler)
        
        assert len(stdout_handlers) >= 1, "Should have at least one stdout handler"
        assert len(file_handlers) >= 1, "Should have at least one file handler"


def test_get_logger_stdout_handler_uses_info_level():
    """Test that the stdout handler uses INFO level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_stdout_level", log_file=log_file)
        
        # Find stdout handler
        for handler in logger.handlers:
            if hasattr(handler, '_handler_type') and handler._handler_type == 'classic_stdout':
                assert handler.level == logging.INFO


def test_get_logger_file_handler_uses_debug_level():
    """Test that the file handler uses DEBUG level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_file_level", log_file=log_file)
        
        # Find file handler
        for handler in logger.handlers:
            if hasattr(handler, '_handler_type') and handler._handler_type == 'detailed_json':
                assert handler.level == logging.DEBUG


def test_get_logger_sets_logger_to_debug():
    """Test that get_logger sets the logger level to DEBUG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger = get_logger("test_logger_level", log_file=log_file)
        assert logger.level == logging.DEBUG


def test_get_logger_no_duplicate_handlers():
    """Test that calling get_logger multiple times doesn't add duplicate handlers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        logger_name = "test_no_duplicates"
        logger1 = get_logger(logger_name, log_file=log_file)
        initial_handler_count = len(logger1.handlers)
        
        logger2 = get_logger(logger_name, log_file=log_file)
        final_handler_count = len(logger2.handlers)
        
        assert initial_handler_count == final_handler_count


def test_logger_info_goes_to_stdout():
    """Test that INFO messages go to stdout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        captured_output = StringIO()
        test_message = "Test info message"
        
        # Create a logger and replace stdout handler
        logger = logging.getLogger("test_info_stdout")
        logger.handlers.clear()
        
        stdout_handler = logging.StreamHandler(captured_output)
        stdout_handler.setLevel(logging.INFO)
        stdout_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
        stdout_handler.setFormatter(stdout_formatter)
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.DEBUG)
        
        # Log an info message
        logger.info(test_message)
        
        output = captured_output.getvalue()
        assert test_message in output
        assert "INFO" in output


def test_logger_debug_not_in_stdout():
    """Test that DEBUG messages don't go to stdout (only INFO and above)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        captured_output = StringIO()
        test_message = "Test debug message"
        
        # Create a logger with INFO level stdout handler
        logger = logging.getLogger("test_debug_not_stdout")
        logger.handlers.clear()
        
        stdout_handler = logging.StreamHandler(captured_output)
        stdout_handler.setLevel(logging.INFO)  # INFO and above only
        stdout_formatter = logging.Formatter(CLASSIC_LOG_FORMAT)
        stdout_handler.setFormatter(stdout_formatter)
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.DEBUG)
        
        # Log a debug message
        logger.debug(test_message)
        
        output = captured_output.getvalue()
        assert test_message not in output  # Should NOT appear in stdout


def test_logger_debug_goes_to_file():
    """Test that DEBUG messages go to the JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.json")
        test_message = "Test debug message"
        
        logger = get_logger("test_debug_file", log_file=log_file)
        logger.debug(test_message)
        
        # Read the log file
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            content = f.read()
            assert test_message in content
            assert "DEBUG" in content


def test_json_formatter():
    """Test the JSONFormatter directly."""
    formatter = JSONFormatter()
    
    # Create a log record
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py",
        lineno=42, msg="Test message", args=(), exc_info=None,
        func="test_func"
    )
    
    output = formatter.format(record)
    
    # Parse JSON
    log_data = json.loads(output)
    
    assert log_data["name"] == "test"
    assert log_data["level"] == "INFO"
    assert log_data["filename"] == "test.py"
    assert log_data["lineno"] == 42
    assert log_data["funcName"] == "test_func"
    assert log_data["message"] == "Test message"


def test_logger_creates_log_directory():
    """Test that get_logger creates the log directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "subdir", "test.json")
        assert not os.path.exists(os.path.dirname(log_file))
        
        logger = get_logger("test_create_dir", log_file=log_file)
        logger.info("Test message")
        
        assert os.path.exists(os.path.dirname(log_file))


def test_constant_values():
    """Test that constants are properly defined."""
    assert CLASSIC_LOG_LEVEL == logging.INFO
    assert DETAILED_LOG_LEVEL == logging.DEBUG
    assert isinstance(CLASSIC_LOG_FORMAT, str)
    
    # Classic format should have basic fields
    assert "%(asctime)s" in CLASSIC_LOG_FORMAT
    assert "%(name)s" in CLASSIC_LOG_FORMAT
    assert "%(levelname)s" in CLASSIC_LOG_FORMAT
    assert "%(message)s" in CLASSIC_LOG_FORMAT
