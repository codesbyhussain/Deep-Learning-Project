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
from collections import Counter
from pathlib import Path
from typing import Tuple

import numpy as np
import wfdb
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.utils.paths import get_raw_data_dir, get_processed_data_dir, ensure_dir

logger = logging.getLogger(__name__)


# ---- 11 rhythms merged into 4 groups (SB, AFIB, GSVT, SR) ----
# Each group is positive if any of its SNOMED codes appear in the record Dx.
# SB: sinus bradycardia only. AFIB: AF + AFL. GSVT: SVT, AT, AVNRT, AVRT, wandering atrial pacemaker. SR: sinus rhythm + sinus irregularity.
CODE_GROUPS = [
    ["164889003", "164890007"],   # AFIB: Atrial Fibrillation, Atrial Flutter
    ["426761007", "713422000", "233896004", "233897008", "195101003", "427172004"],  # GSVT: SVT, AT, AVNRT, AVRT, wandering, general SVT
    ["426177001"],                # SB: Sinus Bradycardia
    ["426783006", "427393009"],   # SR: Sinus Rhythm, Sinus Irregularity
]

# Human-readable names; same order as CODE_GROUPS (0=AF, 1=SVT, 2=SB, 3=SR)
CLASS_NAMES = [
    "AF",              # AFIB: atrial fibrillation + atrial flutter
    "SVT",             # GSVT: supraventricular tachycardia and related
    "Sinus Brady",     # Sinus Bradycardia
    "Sinus Rhythm",    # sinus rhythm + sinus irregularity
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


def _extract_dx_codes_from_hea_text(hea_text: str) -> set[str]:
    """Parse Dx line from raw .hea file text (no WFDB dependency)."""
    for line in hea_text.splitlines():
        if "Dx:" in line:
            dx = line.split("Dx:")[-1].strip()
            return {p.strip() for p in dx.split(",") if p.strip()}
    return set()


def get_all_unique_dx_codes(counts: bool = False):
    """Scan all .hea files and return unique SNOMED Dx codes (and optionally counts).
    Does not load signal data. Returns set of codes, or if counts=True, (set, Counter)."""
    wfdb_dir = get_raw_data_dir() / "chapman" / "WFDBRecords"
    if not wfdb_dir.exists():
        raise FileNotFoundError(f"WFDBRecords not found: {wfdb_dir}")
    hea_paths = sorted(wfdb_dir.rglob("*.hea"))
    all_codes: set[str] = set()
    code_counter: Counter = Counter()
    records_with_dx = 0
    records_without_dx = 0
    for p in hea_paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            records_without_dx += 1
            continue
        codes = _extract_dx_codes_from_hea_text(text)
        if codes:
            records_with_dx += 1
            for c in codes:
                all_codes.add(c)
                code_counter[c] += 1
        else:
            records_without_dx += 1
    if counts:
        return all_codes, code_counter, records_with_dx, records_without_dx
    return all_codes


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
    y = np.array([1 if any(c in codes for c in group) else 0 for group in CODE_GROUPS], dtype=np.int8)

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