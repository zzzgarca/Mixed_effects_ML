# wrapper.py

from typing import Any, Dict, Optional
import os
import numpy as np
import torch

import torch.nn.functional as F
from kan.architecture import KANAdditiveMixed, KANDefaults


class KANWrapper:
    def __init__(self, arch_overrides: Optional[Dict[str, Any]] = None, device: Optional[torch.device] = None):
        self.defaults = KANDefaults(**(arch_overrides or {}))
        self.model: Optional[KANAdditiveMixed] = None
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.trained_dims = None
        self.threshold: float = 0.5

    def build(self, d_fix: int, d_tc: int, d_zrand: int):
        self.model = KANAdditiveMixed.from_dims(y_dim=1, d_fix=d_fix, d_tc=d_tc, d_zrand=d_zrand, **self.defaults.__dict__)
        self.model.to(self.device)
        self.trained_dims = dict(d_fix=d_fix, d_tc=d_tc, d_zrand=d_zrand)

    def fit(
        self,
        X_fix: np.ndarray,
        TC: Optional[np.ndarray],
        Zrand: Optional[np.ndarray],
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
    ) -> Dict[str, list]:
        X_fix = np.asarray(X_fix, dtype=np.float32)
        TC = None if TC is None else np.asarray(TC, dtype=np.float32)
        Zrand = None if Zrand is None else np.asarray(Zrand, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        y_lags = None if y_lags is None else np.asarray(y_lags, dtype=np.float32)
        dt_lags = None if dt_lags is None else np.asarray(dt_lags, dtype=np.float32)

        if self.model is None:
            d_fix = X_fix.shape[1]
            d_tc = 0 if TC is None else TC.shape[1]
            d_zr = 0 if Zrand is None else Zrand.shape[1]
            self.build(d_fix=d_fix, d_tc=d_tc, d_zrand=d_zr)

        self.threshold = float(threshold)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        history = {"loss": []}
        N = X_fix.shape[0]
        idx = np.arange(N)

        if pos_weight is None:
            p = float(y.mean()) if N > 0 else 0.0
            if 0.0 < p < 1.0:
                pos_weight = float((1.0 - p) / max(p, 1e-8))
            else:
                pos_weight = 1.0
        pw = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)

        toT = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device) if a is not None else None

        for ep in range(1, epochs + 1):
            np.random.shuffle(idx)
            total, seen = 0.0, 0
            for s in range(0, N, batch_size):
                sel = idx[s : s + batch_size]
                Xf_b = toT(X_fix[sel])
                TC_b = None if TC is None else toT(TC[sel])
                Zr_b = None if Zrand is None else toT(Zrand[sel])
                y_b = toT(y[sel])
                yl_b = None if y_lags is None else toT(y_lags[sel])
                dt_b = None if dt_lags is None else toT(dt_lags[sel])

                opt.zero_grad(set_to_none=True)
                logits, _ = self.model(Xf_b, TC_b, Zr_b, yl_b, dt_b)
                loss = F.binary_cross_entropy_with_logits(logits, y_b, pos_weight=pw)
                loss.backward()
                opt.step()

                bs = logits.size(0)
                total += float(loss.detach().cpu()) * bs
                seen += bs

            epoch_loss = total / max(1, seen)
            history["loss"].append(epoch_loss)
            if verbose:
                print(f"epoch {ep:03d} | loss {epoch_loss:.6f}")

        return history

    @torch.no_grad()
    def predict_proba(
        self,
        X_fix: np.ndarray,
        TC: Optional[np.ndarray],
        Zrand: Optional[np.ndarray],
        y_lags: Optional[np.ndarray] = None,
        dt_lags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        assert self.model is not None
        X_fix = np.asarray(X_fix, dtype=np.float32)
        TC = None if TC is None else np.asarray(TC, dtype=np.float32)
        Zrand = None if Zrand is None else np.asarray(Zrand, dtype=np.float32)
        y_lags = None if y_lags is None else np.asarray(y_lags, dtype=np.float32)
        dt_lags = None if dt_lags is None else np.asarray(dt_lags, dtype=np.float32)
        toT = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device) if a is not None else None
        logits, _ = self.model(
            toT(X_fix),
            None if TC is None else toT(TC),
            None if Zrand is None else toT(Zrand),
            None if y_lags is None else toT(y_lags),
            None if dt_lags is None else toT(dt_lags),
        )
        return torch.sigmoid(logits).cpu().numpy()

    @torch.no_grad()
    def predict(
        self,
        X_fix: np.ndarray,
        TC: Optional[np.ndarray],
        Zrand: Optional[np.ndarray],
        y_lags: Optional[np.ndarray] = None,
        dt_lags: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        thr = float(self.threshold if threshold is None else threshold)
        p = self.predict_proba(X_fix, TC, Zrand, y_lags, dt_lags)
        return (p >= thr).astype(np.int32)

    def save(self, path: str):
        assert self.model is not None and self.trained_dims is not None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "arch_defaults": self.defaults.__dict__,
            "trained_dims": self.trained_dims,
            "threshold": float(self.threshold),
            "model_state": self.model.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "KANWrapper":
        state = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        obj = cls(arch_overrides=state["arch_defaults"], device=device)
        dims = state["trained_dims"]
        obj.build(d_fix=int(dims["d_fix"]), d_tc=int(dims["d_tc"]), d_zrand=int(dims["d_zrand"]))
        obj.model.load_state_dict(state["model_state"])
        obj.threshold = float(state.get("threshold", 0.5))
        return obj
