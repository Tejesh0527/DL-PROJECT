"""
utils/logger_setup.py
======================
Logging configuration for RATFA.
"""

import logging
import os
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: Optional[str] = "ratfa.log",
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """Configure root logger with console and optional file handler.

    Parameters
    ----------
    level : str
        Log level string: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_dir : str
        Directory for the log file.
    log_file : str, optional
        Log file name.  Pass None to disable file logging.
    fmt : str
        Log message format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Avoid duplicate handlers on re-import
    root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

