# evaluator.py

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import torch
from kan.wrapper import KANWrapper



_VAR_EPS = 1e-8
_STD_EPS = 1e-6
_CLIP_Z = 8.0


@dataclass
class PCAPipeline:
    keep_mask: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    components_: np.ndarray


def _fit_pca(X_train: np.ndarray, var_ratio: float = 0.95, random_state: int = 42) -> PCAPipeline:
    X = np.asarray(X_train, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    var = X.var(axis=0)
    keep = var > _VAR_EPS
    if not np.any(keep):
        return PCAPipeline(keep_mask=np.zeros(X.shape[1], dtype=bool), mean_=np.array([]), scale_=np.array([]), components_=np.zeros((0, 0)))
    Xk = X[:, keep]
    mean = Xk.mean(axis=0)
    std = Xk.std(axis=0)
    std = np.maximum(std, _STD_EPS)
    Z = (Xk - mean) / std
    Z = np.clip(Z, -_CLIP_Z, _CLIP_Z)
    pca = PCA(n_components=var_ratio, svd_solver="full", random_state=random_state)
    pca.fit(Z)
    return PCAPipeline(keep_mask=keep, mean_=mean, scale_=std, components_=pca.components_.copy())


def _transform_pca(pipe: Optional[PCAPipeline], X: Optional[np.ndarray]) -> np.ndarray:
    if pipe is None or X is None:
        return np.zeros((X.shape[0], 0), dtype=np.float32) if X is not None else np.zeros((0, 0), dtype=np.float32)
    X = np.asarray(X, dtype=np.float64)
    if pipe.keep_mask.size == 0 or not np.any(pipe.keep_mask):
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    Xk = X[:, pipe.keep_mask]
    Z = (Xk - pipe.mean_) / pipe.scale_
    Z = np.clip(Z, -_CLIP_Z, _CLIP_Z)
    Xt = Z @ pipe.components_.T
    return np.asarray(Xt, dtype=np.float32)


def _metrics_binary(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= thr).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float("nan")
    brier = float(np.mean((y_prob - y_true) ** 2))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = float((y_pred == y_true).mean())
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    except Exception:
        sens, spec = float("nan"), float("nan")
    return {
        "AUC": auc,
        "AUPRC": auprc,
        "Brier": brier,
        "ACC": acc,
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "Sensitivity": sens,
        "Specificity": spec,
    }


class KANEvaluator:
    def __init__(
        self,
        arch_overrides: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        random_state: int = 42,
        pca_var_ratio: float = 0.95,
        output_dir: str = "kan_run",
    ):
        self.arch_overrides = arch_overrides or {}
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.random_state = int(random_state)
        self.pca_var_ratio = float(pca_var_ratio)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.of_pipe: Optional[PCAPipeline] = None
        self.fr_pipe: Optional[PCAPipeline] = None
        self.wrapper: Optional[KANWrapper] = None

    def fit_eval(
        self,
        X_only_fixed: np.ndarray,
        X_fixed_and_random: Optional[np.ndarray],
        y: np.ndarray,
        y_lags: Optional[np.ndarray] = None,
        dt_lags: Optional[np.ndarray] = None,
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 50,
        threshold: float = 0.5,
        pos_weight: Optional[float] = None,
        verbose: bool = True,
        save_prefix: str = "model",
        pid_idx: Optional[np.ndarray] = None,
        group_by_pid: bool = True,   # <-- NEW
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.random_state)
        X_of = np.asarray(X_only_fixed, dtype=np.float32)
        X_fr = None if X_fixed_and_random is None else np.asarray(X_fixed_and_random, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        y_lags = None if y_lags is None else np.asarray(y_lags, dtype=np.float32)
        dt_lags = None if dt_lags is None else np.asarray(dt_lags, dtype=np.float32)
        N = y.shape[0]

        if group_by_pid and (pid_idx is not None):
            pid_idx = np.asarray(pid_idx).ravel()
            uniq = np.unique(pid_idx)
            if uniq.size >= 2:
                perm = rng.permutation(uniq.size)
                cut = max(1, int(np.floor(0.8 * uniq.size)))
                tr_pids = set(uniq[perm[:cut]].tolist())
                mask_tr = np.isin(pid_idx, list(tr_pids))
                tr_idx = np.where(mask_tr)[0]
                te_idx = np.where(~mask_tr)[0]
                if te_idx.size == 0:
                    idx = rng.permutation(N)
                    split = int(np.floor(N * 0.8))
                    tr_idx, te_idx = idx[:split], idx[split:]
            else:
                idx = rng.permutation(N)
                split = int(np.floor(N * 0.8))
                tr_idx, te_idx = idx[:split], idx[split:]
        else:
            idx = rng.permutation(N)
            split = int(np.floor(N * 0.8))
            tr_idx, te_idx = idx[:split], idx[split:]

        self.of_pipe = _fit_pca(X_of[tr_idx], var_ratio=self.pca_var_ratio, random_state=self.random_state)
        self.fr_pipe = _fit_pca(X_fr[tr_idx], var_ratio=self.pca_var_ratio, random_state=self.random_state) if (X_fr is not None and X_fr.shape[1] > 0) else None

        def TX(idxs):
            of = _transform_pca(self.of_pipe, X_of[idxs])
            fr = _transform_pca(self.fr_pipe, X_fr[idxs]) if self.fr_pipe is not None else np.zeros((len(idxs), 0), dtype=np.float32)
            X_fix = np.concatenate([of, fr], axis=1) if fr.shape[1] > 0 else of
            return X_fix, of, fr

        Xf_tr, TC_tr, Zr_tr = TX(tr_idx)
        Xf_te, TC_te, Zr_te = TX(te_idx)

        self.wrapper = KANWrapper(arch_overrides=self.arch_overrides, device=self.device)
        hist = self.wrapper.fit(
            X_fix=Xf_tr,
            TC=TC_tr,
            Zrand=Zr_tr,
            y=y[tr_idx],
            y_lags=None if y_lags is None else y_lags[tr_idx],
            dt_lags=None if dt_lags is None else dt_lags[tr_idx],
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            threshold=threshold,
            pos_weight=pos_weight,
            verbose=verbose,
        )

        probs = self.wrapper.predict_proba(
            X_fix=Xf_te,
            TC=TC_te,
            Zrand=Zr_te,
            y_lags=None if y_lags is None else y_lags[te_idx],
            dt_lags=None if dt_lags is None else dt_lags[te_idx],
        ).ravel()
        preds = (probs >= float(threshold)).astype(int)
        y_true = y[te_idx].ravel()

        metrics = _metrics_binary(y_true, probs, thr=float(threshold))
        if verbose:
            print({k: round(v, 4) if isinstance(v, float) and np.isfinite(v) else v for k, v in metrics.items()})

        pred_df = pd.DataFrame({"y_true": y_true, "y_prob": probs, "y_pred": preds})
        preds_path = os.path.join(self.output_dir, f"{save_prefix}_predictions.csv")
        pred_df.to_csv(preds_path, index=False)

        model_path = os.path.join(self.output_dir, f"{save_prefix}.pt")
        self.wrapper.save(model_path)

        pca_path = os.path.join(self.output_dir, f"{save_prefix}_pca.pkl")
        with open(pca_path, "wb") as f:
            pickle.dump({"of_pipe": self.of_pipe, "fr_pipe": self.fr_pipe}, f)

        return {
            "metrics": metrics,
            "history": hist,
            "predictions": pred_df,
            "predictions_path": preds_path,
            "model_path": model_path,
            "pca_path": pca_path,
            "test_index": te_idx,
        }
