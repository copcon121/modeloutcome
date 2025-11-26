"""
Logging Utilities
Centralized logging setup for the feature engine
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    format_string: str = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Optional custom format string

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamped_logfile(base_dir: str, prefix: str) -> str:
    """
    Generate a timestamped log file path

    Args:
        base_dir: Base directory for logs
        prefix: Log file prefix

    Returns:
        Full path to log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{prefix}_{timestamp}.log"
    return str(Path(base_dir) / filename)


# Default logger for feature engine
feature_engine_logger = setup_logger(
    'feature_engine',
    level=logging.INFO
)
