"""
Module for configuring application logging.

Provides functions to set up logging handlers, formatters, and file rotation.

Version: 1.0
Author: Xiao-Fei Zhang

Example:
    >>> import logging
    >>> import logging_config

    >>> logger = logging.getLogger(__name__)

    >>> logger.debug("Debug message")
    >>> logger.info("Info message")
    >>> logger.warning("Warning message")
    >>> logger.error("Error message")
    >>> logger.critical("Critical message")
"""

import logging
import logging.handlers
import os
from utils.find_project_root import find_project_root


def get_username():
    """
    Retrieves the current username.

    Returns:
        str: Username of the current user.
    """
    return os.getlogin()


def get_log_file_path(logs_dir):
    """
    Constructs the log file path based on the username.

    Args:
        logs_dir (str): Directory path for logs.

    Returns:
        str: Log file path with username.
    """
    username = get_username()
    return os.path.join(logs_dir, f"{username}_app.log")


def configure_logging():
    """
    Configures logging settings, including handlers, formatters, and file rotation.

    Returns:
        None
    """
    root_dir = find_project_root()
    logs_dir = os.path.join(root_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file_path = get_log_file_path(logs_dir)

    # Set up log file rotation: max 10MB per file, up to 5 backup files
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Create a console handler with a specific log level
    console_handler = logging.StreamHandler()
