"""Logging configuration.

Basic, filterable console logging plus a per-component error log file:
- ``LOG_LEVEL`` (default ``INFO``) sets console verbosity.
- ``ERROR_LOG_FILE`` (default ``logs/<component>-error.log``) captures ERROR+
  records to a rotating file — the send-failure / write-failure log.

Line format: ``<ts> <LEVEL> [<component>] <logger>: <message>`` — greppable by
level, component, and logger name.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

_configured = False


def configure_logging(component: str) -> None:
    """Idempotently configure root logging for a service component."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    console_level = getattr(logging, level_name, logging.INFO)
    error_log_file = os.getenv("ERROR_LOG_FILE", f"logs/{component}-error.log")

    fmt = logging.Formatter(
        f"%(asctime)s %(levelname)-8s [{component}] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    root = logging.getLogger()
    # Keep the root threshold low enough that ERROR always reaches the file
    # handler, even if the console is set to a higher level.
    root.setLevel(min(console_level, logging.ERROR))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    os.makedirs(os.path.dirname(error_log_file) or ".", exist_ok=True)
    error_file = RotatingFileHandler(
        error_log_file, maxBytes=5_000_000, backupCount=3
    )
    error_file.setLevel(logging.ERROR)
    error_file.setFormatter(fmt)
    root.addHandler(error_file)

    _configured = True
    logging.getLogger(component).info(
        "Logging configured (level=%s, error_log=%s)", level_name, error_log_file
    )
