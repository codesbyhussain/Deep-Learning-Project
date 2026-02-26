"""Configure root logger with optional file output."""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_root_logger(
    log_file_path: Optional[str | Path] = None,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> None:
    """Configure the root logger. If log_file_path is set, also log to that file."""
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called multiple times
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    if log_file_path is not None:
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)
