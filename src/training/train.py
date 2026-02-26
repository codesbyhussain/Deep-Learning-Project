"""CLI-driven training pipeline: features -> reduction -> classifier."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data.io import load_raw_dataset, load_labels, save_labels
from src.data.splits import create_splits, load_splits, save_splits
from src.features.memmap import create_memmap, open_memmap_read, write_array_to_memmap
from src.features.multirocket import (
    create_multirocket_transformer,
    fit_multirocket,
    load_multirocket_transformer,
    save_multirocket_transformer,
    transform_multirocket_batched,
)
from src.features.pooling import structured_pool
from src.features.scaling import fit_scaler, load_scaler, save_scaler, transform_with_scaler
from src.models.autoencoder import load_encoder_for_inference, save_autoencoder, train_autoencoder
from src.models.ft_transformer import FTTransformer
from src.models.mlp import MLPClassifier
from src.training.callbacks import EarlyStopping
from src.training.evaluation import compute_metrics, save_confusion_matrix, save_metrics
from src.utils.config import get_nested, load_config
from src.utils.logging import configure_root_logger
from src.utils.paths import (
    ensure_dir,
    get_experiment_dir,
    get_experiment_output_dir,
    get_labels_dir,
    get_models_dir,
    get_multirocket_features_dir,
    get_reduced_dir,
    get_splits_dir,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = get_nested(cfg, "train.device", "auto")
    if dev == "auto" or dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _resolve_condition_from_config_path(config_path: Path) -> str:
    """e.g. experiments/A1_autoencoder_mlp/config.yaml -> A1_autoencoder_mlp"""
    return config_path.parent.name


def run(
    config_path: str | Path,
) -> None:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    condition = _resolve_condition_from_config_path(config_path)
    exp_dir = ensure_dir(get_experiment_dir(condition))
    log_dir = ensure_dir(get_experiment_output_dir(condition, logs=True))
    ckpt_dir = ensure_dir(get_experiment_output_dir(condition, checkpoints=True))
    configure_root_logger(log_file_path=log_dir / "train.log")

    seed = get_nested(cfg, "multirocket.seed", 0)
    set_global_seed(seed)
    logger.info("Condition: %s | Seed: %s", condition, seed)
    try:
        import numpy as np
        import torch
        logger.info("numpy %s | torch %s", np.__version__, torch.__version__)
    except Exception:
        pass

    # --- Data ---
    X_raw, y = load_raw_dataset()
    save_labels(y, get_labels_dir() / "labels.npy")
    task_type = get_nested(cfg, "task_type", "multiclass")
    split_seed = get_nested(cfg, "splits.seed", seed)
    test_size = get_nested(cfg, "splits.test_size", 0.2)
    val_size = get_nested(cfg, "splits.val_size", 0.1)
    splits_path = get_splits_dir() / "splits.npz"
    if splits_path.exists():
        idx_train, idx_val, idx_test = load_splits(splits_path)
        logger.info("Loaded existing splits: train %s val %s test %s", len(idx_train), len(idx_val), len(idx_test))
    else:
        idx_train, idx_val, idx_test = create_splits(y, task_type=task_type, test_size=test_size, val_size=val_size, seed=split_seed)
        save_splits(idx_train, idx_val, idx_test, splits_path)

    X_train_raw = X_raw[idx_train]
    X_val_raw = X_raw[idx_val] if len(idx_val) > 0 else None
    X_test_raw = X_raw[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]
    if task_type == "multiclass":
        num_classes = int(np.max(y)) + 1
    else:
        num_classes = y.shape[1]

    # --- MultiROCKET ---
    num_kernels = get_nested(cfg, "multirocket.num_kernels", 2048)
    mr_dir = get_multirocket_features_dir()
    mr_transformer_path = get_models_dir() / "multirocket" / "transformer.joblib"
    if mr_transformer_path.exists():
        transformer = load_multirocket_transformer(mr_transformer_path)
        logger.info("Loaded existing MultiROCKET transformer")
    else:
        transformer = create_multirocket_transformer(num_kernels=num_kernels, seed=seed)
        fit_multirocket(transformer, X_train_raw)
        ensure_dir(mr_transformer_path.parent)
        save_multirocket_transformer(transformer, mr_transformer_path)

    batch_size_transform = get_nested(cfg, "multirocket.batch_size", 1024)
    F_train = transform_multirocket_batched(transformer, X_train_raw, batch_size=batch_size_transform)
    F_test = transform_multirocket_batched(transformer, X_test_raw, batch_size=batch_size_transform)
    n_features = F_train.shape[1]
    logger.info("MultiROCKET features: train %s test %s n_features %s", F_train.shape, F_test.shape, n_features)

    # Memmap persistence for full-dataset features if desired (optional: we keep in memory for pipeline simplicity; can write to memmap here)
    feat_train_path = mr_dir / "train.dat"
    feat_test_path = mr_dir / "test.dat"
    write_array_to_memmap(feat_train_path, F_train)
    write_array_to_memmap(feat_test_path, F_test)

    # --- Scaling ---
    scaling = get_nested(cfg, "scaling.type", "standard")
    scaler_path = get_models_dir() / "scalers" / "scaler.joblib"
    if scaler_path.exists():
        scaler = load_scaler(scaler_path)
    else:
        scaler = fit_scaler(F_train)
        ensure_dir(scaler_path.parent)
        save_scaler(scaler, scaler_path)
    F_train = np.asarray(transform_with_scaler(scaler, F_train), dtype=np.float32)
    F_test = np.asarray(transform_with_scaler(scaler, F_test), dtype=np.float32)

    # --- Reduction ---
    reduction = get_nested(cfg, "reduction.type", "autoencoder")
    if reduction == "autoencoder":
        latent_dim = get_nested(cfg, "reduction.latent_dim", 256)
        ae_hidden = get_nested(cfg, "reduction.autoencoder_hidden_dims", [512, 256])
        ae_path = get_models_dir() / "autoencoders" / "encoder.pt"
        red_train_path = get_reduced_dir() / "autoencoder" / "train.npy"
        red_test_path = get_reduced_dir() / "autoencoder" / "test.npy"
        ensure_dir(red_train_path.parent)
        device = _get_device(cfg)
        if ae_path.exists():
            encoder = load_encoder_for_inference(ae_path, n_features, ae_hidden, latent_dim, device=device)
            encoder = encoder.to(device)
            encoder.eval()
            with torch.no_grad():
                t_train = torch.from_numpy(F_train).to(device)
                t_test = torch.from_numpy(F_test).to(device)
                Z_train = encoder(t_train).cpu().numpy()
                Z_test = encoder(t_test).cpu().numpy()
            logger.info("Loaded encoder; reduced shapes %s %s", Z_train.shape, Z_test.shape)
        else:
            t_train = torch.from_numpy(F_train).float().to(device)
            encoder, _ = train_autoencoder(
                t_train,
                input_dim=n_features,
                hidden_dims=ae_hidden,
                latent_dim=latent_dim,
                epochs=get_nested(cfg, "reduction.ae_epochs", 50),
                batch_size=get_nested(cfg, "train.batch_size", 256),
                lr=get_nested(cfg, "train.lr", 1e-3),
                device=device,
            )
            ensure_dir(ae_path.parent)
            save_autoencoder(encoder, ae_path)
            with torch.no_grad():
                Z_train = encoder(t_train).cpu().numpy()
                Z_test = encoder(torch.from_numpy(F_test).float().to(device)).cpu().numpy()
        np.save(red_train_path, Z_train.astype(np.float32))
        np.save(red_test_path, Z_test.astype(np.float32))
        X_train_clf = Z_train
        X_test_clf = Z_test
        input_dim_clf = latent_dim
    else:
        n_origins = get_nested(cfg, "reduction.pooling.n_origins", 2)
        n_stats = get_nested(cfg, "reduction.pooling.n_stats", 4)
        pool = get_nested(cfg, "reduction.pooling.pool", "mean")
        X_train_clf = structured_pool(F_train, n_origins=n_origins, n_stats=n_stats, pool=pool)
        X_test_clf = structured_pool(F_test, n_origins=n_origins, n_stats=n_stats, pool=pool)
        red_train_path = get_reduced_dir() / "pooled" / "train.npy"
        red_test_path = get_reduced_dir() / "pooled" / "test.npy"
        ensure_dir(red_train_path.parent)
        np.save(red_train_path, X_train_clf.astype(np.float32))
        np.save(red_test_path, X_test_clf.astype(np.float32))
        input_dim_clf = X_train_clf.shape[1]
        logger.info("Pooled features: train %s test %s dim %s", X_train_clf.shape, X_test_clf.shape, input_dim_clf)

    # --- Classifier ---
    classifier_type = get_nested(cfg, "classifier.type", "mlp")
    epochs = get_nested(cfg, "train.epochs", 20)
    batch_size = get_nested(cfg, "train.batch_size", 256)
    lr = get_nested(cfg, "train.lr", 1e-3)
    weight_decay = get_nested(cfg, "train.weight_decay", 1e-4)
    device = _get_device(cfg)
    use_early_stopping = get_nested(cfg, "train.early_stopping", False)
    early_patience = get_nested(cfg, "train.early_stopping_patience", 5)

    if classifier_type == "mlp":
        model = MLPClassifier(
            input_dim=input_dim_clf,
            num_classes=num_classes,
            hidden_dims=get_nested(cfg, "classifier.mlp.hidden_dims", [512, 256]),
            dropout=get_nested(cfg, "classifier.mlp.dropout", 0.1),
        ).to(device)
        ckpt_path = ckpt_dir / "mlp.pt"
    else:
        model = FTTransformer(
            n_features=input_dim_clf,
            num_classes=num_classes,
            d_token=get_nested(cfg, "classifier.ft_transformer.d_token", 128),
            n_heads=get_nested(cfg, "classifier.ft_transformer.n_heads", 8),
            n_layers=get_nested(cfg, "classifier.ft_transformer.n_layers", 3),
            dropout=get_nested(cfg, "classifier.ft_transformer.dropout", 0.1),
        ).to(device)
        ckpt_path = ckpt_dir / "ft_transformer.pt"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(X_train_clf)
    best_f1 = 0.0
    early_stop = EarlyStopping(patience=early_patience, mode="max") if use_early_stopping else None
    X_val_clf = None
    y_val = None
    if len(idx_val) > 0:
        if reduction == "autoencoder":
            t_val = torch.from_numpy(
                transform_with_scaler(scaler, transform_multirocket_batched(transformer, X_val_raw, batch_size=batch_size_transform))
            ).float()
            with torch.no_grad():
                X_val_clf = encoder(t_val.to(device)).cpu().numpy()
        else:
            F_val = transform_multirocket_batched(transformer, X_val_raw, batch_size=batch_size_transform)
            F_val = np.asarray(transform_with_scaler(scaler, F_val), dtype=np.float32)
            X_val_clf = structured_pool(F_val, n_origins=get_nested(cfg, "reduction.pooling.n_origins", 2), n_stats=get_nested(cfg, "reduction.pooling.n_stats", 4), pool=get_nested(cfg, "reduction.pooling.pool", "mean"))
        y_val = y[idx_val]

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        n_b = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bx = torch.from_numpy(X_train_clf[idx]).float().to(device)
            by = torch.from_numpy(y_train[idx]).long().to(device)
            if task_type == "multilabel":
                by = by.float()
                logits = model(bx)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, by)
            else:
                logits = model(bx)
                loss = torch.nn.functional.cross_entropy(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            logger.info("Epoch %d loss %.4f", ep + 1, total_loss / max(n_b, 1))

        if X_val_clf is not None and y_val is not None and use_early_stopping and (ep + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X_val_clf).float().to(device))
                if task_type == "multiclass":
                    pred = logits.argmax(dim=1).cpu().numpy()
                else:
                    pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            from sklearn.metrics import f1_score
            f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), ckpt_path)
            if early_stop and early_stop.step(f1):
                break
    if not use_early_stopping or best_f1 == 0:
        torch.save(model.state_dict(), ckpt_path)

    # --- Evaluate on test ---
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_clf).float().to(device))
        if task_type == "multiclass":
            y_pred = logits.argmax(dim=1).cpu().numpy()
        else:
            y_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()

    class_names = get_nested(cfg, "class_names", None)
    labels_list = list(range(num_classes))
    metrics = compute_metrics(y_test, y_pred, task=task_type, labels=labels_list)
    save_metrics(metrics, exp_dir / "metrics.json")
    cm = np.array(metrics["confusion_matrix"])
    save_confusion_matrix(
        cm,
        exp_dir / "confusion_matrix.npy",
        path_png=exp_dir / "confusion_matrix.png",
        class_names=class_names,
    )
    logger.info("Test weighted F1: %.4f", metrics["weighted_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ablation: features -> reduction -> classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML (e.g. experiments/A1_autoencoder_mlp/config.yaml)")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
