"""
production/logger.py
--------------------
Centralized logging for Patra ML backend.
Supports console logging and optional file logging with configurable levels.
"""

import logging
from production.config_loader import load_config

config = load_config()


def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with console and optional file handlers.
    
    Args:
        name: Logger name, usually __name__ from the calling module.

    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        # Avoid adding multiple handlers if logger is reused
        return logger

    # Logging level
    level_str = config.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                                     datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Optional file handler
    log_file = config.get("logging", {}).get("file")
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(ch_formatter)
        logger.addHandler(fh)

    return logger
