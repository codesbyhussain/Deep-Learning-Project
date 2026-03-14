"""Evaluation: weighted F1, confusion matrix, per-class metrics; save artifacts."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "multiclass",
    labels: Optional[List[int]] = None,
) -> dict:
    """Compute weighted F1 and confusion matrix. For multiclass, y are class indices."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return {"weighted_f1": float(f1), "confusion_matrix": cm.tolist()}


def compute_relaxed_accuracy(
    y_multilabel: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
) -> float:
    """
    For multi-label ground truth: count prediction correct if it matches *any* true positive class.
    y_multilabel: (n_samples, n_classes) binary; y_pred: (n_samples,) predicted class indices.
    Returns fraction of samples where pred is in the set of true positive labels.
    """
    y_multilabel = np.asarray(y_multilabel)
    y_pred = np.asarray(y_pred).ravel()
    if num_classes is None:
        num_classes = y_multilabel.shape[1]
    n = len(y_pred)
    if n == 0:
        return 0.0
    correct = np.zeros(n, dtype=bool)
    for i in range(n):
        true_classes = np.where(y_multilabel[i, :num_classes] > 0)[0]
        correct[i] = y_pred[i] in true_classes if len(true_classes) > 0 else False
    return float(np.mean(correct))


def compute_relaxed_per_class_metrics(
    y_multilabel: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    target_names: Optional[List[str]] = None,
) -> tuple:
    """
    Per-class precision, recall, F1 under relaxed rule: prediction is "correct for class c"
    if model predicted c and c is one of the true labels for that sample.
    Returns (per_class_list, weighted_f1_relaxed).
    """
    y_multilabel = np.asarray(y_multilabel)
    y_pred = np.asarray(y_pred).ravel()
    if num_classes is None:
        num_classes = y_multilabel.shape[1]
    if target_names is None:
        target_names = [str(c) for c in range(num_classes)]
    n = len(y_pred)
    per_class = []
    for c in range(num_classes):
        true_c = (y_multilabel[:, c] > 0) if y_multilabel.ndim == 2 else (y_multilabel == c)
        pred_c = y_pred == c
        tp = int(np.logical_and(pred_c, true_c).sum())
        fp = int(np.logical_and(pred_c, ~true_c).sum())
        fn = int(np.logical_and(true_c, ~pred_c).sum())
        support = int(true_c.sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class.append({
            "class": target_names[c] if c < len(target_names) else str(c),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support,
        })
    # Weighted F1 (by support)
    total_support = sum(p["support"] for p in per_class)
    if total_support > 0:
        weighted_f1 = sum(p["f1"] * p["support"] for p in per_class) / total_support
    else:
        weighted_f1 = 0.0
    return per_class, float(weighted_f1)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None,
) -> List[dict]:
    """Return per-class precision, recall, f1, support. labels: class indices; target_names: optional display names."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    names = target_names or [str(i) for i in labels]
    return [
        {
            "class": name,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(names, prec, rec, f1, sup)
    ]


def save_metrics(metrics: dict, path: Union[str, Path]) -> Path:
    """Save metrics dict to JSON (confusion_matrix as list)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    out["confusion_matrix"] = metrics.get("confusion_matrix", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved metrics to %s", path)
    return path


def save_confusion_matrix(
    cm: np.ndarray,
    path_npy: Union[str, Path],
    path_png: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Save confusion matrix as .npy and optionally as .png figure."""
    path_npy = Path(path_npy)
    path_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_npy, cm)
    logger.info("Saved confusion matrix to %s", path_npy)
    if path_png is not None:
        path_png = Path(path_png)
        path_png.parent.mkdir(parents=True, exist_ok=True)
        n = cm.shape[0]
        names = class_names or [str(i) for i in range(n)]
        fig, ax = plt.subplots(figsize=(max(4, n * 0.6), max(4, n * 0.5)))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(path_png, dpi=150)
        plt.close()
        logger.info("Saved confusion matrix figure to %s", path_png)
