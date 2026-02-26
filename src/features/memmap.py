"""Disk-backed feature storage via numpy memmap (float32)."""

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def create_memmap(path: Union[str, Path], shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.memmap:
    """Create a new memmap file of given shape; overwrites if exists. Caller must flush when done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(str(path), mode="w+", dtype=dtype, shape=shape)
    logger.debug("Created memmap %s shape %s", path, shape)
    return mm


def write_array_to_memmap(path: Union[str, Path], array: np.ndarray, dtype: np.dtype = np.float32) -> None:
    """Write a full array to a memmap file and flush."""
    path = Path(path)
    arr = np.asarray(array, dtype=dtype)
    mm = create_memmap(path, arr.shape, dtype=dtype)
    mm[:] = arr[:]
    mm.flush()
    del mm
    logger.info("Wrote array shape %s to memmap %s", arr.shape, path)


def open_memmap_read(
    path: Union[str, Path],
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float32,
) -> np.memmap:
    """Open an existing memmap for reading. Shape must match file size."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memmap not found: {path}")
    mm = np.memmap(str(path), mode="r", dtype=dtype, shape=shape)
    return mm
