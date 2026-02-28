"""Data I/O for Chapman-Shaoxing ECG (WFDBRecords version).
Fast parallel loader + caching + detailed progress logging.

Call:
    X, y = load_raw_dataset()

First run: slow (parses ~45k files) with progress logs.
Later runs: instant (loads cached file).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import wfdb
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.paths import get_raw_data_dir, get_processed_data_dir, ensure_dir

logger = logging.getLogger(__name__)


# ---- Target SNOMED codes (4-class subset) ----
# Order: 3 conditions we detect + Sinus Rhythm (normal / "none")
TARGET_CODES = [
    "164889003",  # AF
    "427172004",  # GSVT
    "426177001",  # SB
    "426783006",  # SR
]

# Human-readable names; same order as TARGET_CODES (0=AF, 1=SVT, 2=SB, 3=SR)
CLASS_NAMES = [
    "AF",              # Atrial Fibrillation
    "SVT",             # Supraventricular Tachycardia
    "Sinus Brady",     # Sinus Bradycardia
    "Sinus Rhythm",    # normal / none of the above
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _extract_dx_codes(comments) -> set[str]:
    if not comments:
        return set()

    for c in comments:
        if "Dx:" in c:
            dx = c.split("Dx:")[-1].strip()
            return {p.strip() for p in dx.split(",") if p.strip()}

    return set()


def _load_one_record(hea_path_str: str):
    hea_path = Path(hea_path_str)
    record_base = str(hea_path.with_suffix(""))

    try:
        signal, meta = wfdb.rdsamp(record_base)
    except Exception:
        return None

    # (T, 12) → (12, T)
    signal = signal.T.astype(np.float32, copy=False)

    codes = _extract_dx_codes(meta.get("comments", []))
    y = np.array([1 if c in codes else 0 for c in TARGET_CODES], dtype=np.int8)

    return signal, y


# ---------------------------------------------------------------------
# MAIN FUNCTION (use this everywhere)
# ---------------------------------------------------------------------

def load_raw_dataset(
    cache_name="chapman_wfdb_Xy.npz",
    max_workers=None,
    force_reload=False,
):

    t_global_start = time.time()

    processed_dir = ensure_dir(get_processed_data_dir())
    cache_path = processed_dir / cache_name
    # -------------------------------------------------
    # Optional: force deletion of cached dataset
    # -------------------------------------------------
    if force_reload and cache_path.exists():
        logger.warning("force_reload=True — removing cached dataset: %s", cache_path)
        try:
            cache_path.unlink()
        except PermissionError:
            logger.error(
                "Could not delete cache (file in use). "
                "Restart kernel or close processes."
            )
            raise
            

    # ---------- FAST PATH ----------
    if cache_path.exists():
        t0 = time.time()
        data = np.load(cache_path)
        X, y = data["X"], data["y"]

        logger.info(
            "Loaded cached dataset %s | X=%s y=%s | %.2fs",
            cache_path, X.shape, y.shape, time.time() - t0
        )

        return X, y

    # ---------- LOAD RAW FILES ----------
    wfdb_dir = get_raw_data_dir() / "chapman" / "WFDBRecords"

    if not wfdb_dir.exists():
        raise FileNotFoundError(f"WFDBRecords not found: {wfdb_dir}")

    hea_paths = sorted(wfdb_dir.rglob("*.hea"))
    total = len(hea_paths)

    if total == 0:
        raise FileNotFoundError("No .hea files found")

    logger.info("Stage 1/4 — Loading raw WFDB files (%d records)", total)

    t0 = time.time()

    X_list = []
    y_list = []

    completed = 0
    last_percent = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_one_record, str(p)) for p in hea_paths]

        for fut in as_completed(futures):
            res = fut.result()
            completed += 1

            if res is not None:
                x, y = res
                X_list.append(x)
                y_list.append(y)

            percent = int(100 * completed / total)

            # ---- Report every 10% ----
            if percent >= last_percent + 10 or completed == total:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else float("inf")

                logger.info(
                    "Loading progress %d/%d (%d%%) | %.0f rec/s | ETA %.0fs | kept %d",
                    completed,
                    total,
                    percent,
                    rate,
                    eta,
                    len(X_list),
                )

                last_percent = percent

    logger.info(
        "Stage 1 complete — loaded %d valid records in %.2fs",
        len(X_list), time.time() - t0
    )

    if not X_list:
        raise RuntimeError("No records loaded")

    # ---------- STACK ARRAYS ----------
    logger.info("Stage 2/4 — Stacking arrays (memory allocation)")
    t1 = time.time()

    lengths = np.array([x.shape[1] for x in X_list])
    vals, counts = np.unique(lengths, return_counts=True)
    modal_T = int(vals[np.argmax(counts)])

    keep = [i for i, x in enumerate(X_list) if x.shape[1] == modal_T]

    X = np.stack([X_list[i] for i in keep]).astype(np.float32, copy=False)
    y = np.stack([y_list[i] for i in keep]).astype(np.int8, copy=False)

    logger.info(
        "Stage 2 complete — X=%s y=%s | %.2fs",
        X.shape, y.shape, time.time() - t1
    )

    # ---------- SAVE CACHE ----------
    logger.info("Stage 3/4 — Saving cache (compression may be slow)")
    t2 = time.time()

    np.savez_compressed(cache_path, X=X, y=y)

    logger.info(
        "Stage 3 complete — saved to %s | %.2fs",
        cache_path, time.time() - t2
    )

    logger.info(
        "All stages complete — total time %.2fs",
        time.time() - t_global_start
    )

    return X, y


# ---------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------

def save_labels(y: np.ndarray, path: str | None = None) -> None:
    from src.utils.paths import get_labels_dir

    out_dir = ensure_dir(get_labels_dir())
    path = out_dir / "labels.npy" if path is None else Path(path)

    np.save(path, y)
    logger.info("Saved labels to %s", path)


def load_labels(path: str | None = None) -> np.ndarray:
    from src.utils.paths import get_labels_dir

    path = get_labels_dir() / "labels.npy" if path is None else Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    return np.load(path)