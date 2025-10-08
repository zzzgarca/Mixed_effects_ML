# predictor.py

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

from kan.wrapper import KANWrapper



@dataclass
class PCAPipeline:
    keep_mask: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    components_: np.ndarray


def _transform_pca(pipe: Optional[PCAPipeline], X: Optional[np.ndarray]) -> np.ndarray:
    if pipe is None or X is None:
        return np.zeros((X.shape[0], 0), dtype=np.float32) if X is not None else np.zeros((0, 0), dtype=np.float32)
    X = np.asarray(X, dtype=np.float64)
    if pipe.keep_mask.size == 0 or not np.any(pipe.keep_mask):
        return np.zeros((X.shape[0], 0), dtype=np.float32)
    Xk = X[:, pipe.keep_mask]
    Z = (Xk - pipe.mean_) / pipe.scale_
    Z = np.clip(Z, -8.0, 8.0)
    Xt = Z @ pipe.components_.T
    return np.asarray(Xt, dtype=np.float32)


class KANPredictor:
    def __init__(
        self,
        wrapper: KANWrapper,
        of_pipe: Optional[PCAPipeline],
        fr_pipe: Optional[PCAPipeline],
        device: Optional[torch.device] = None,
    ):
        self.wrapper = wrapper
        self.of_pipe = of_pipe
        self.fr_pipe = fr_pipe
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    @classmethod
    def load(cls, model_path: str, pca_path: str, device: Optional[torch.device] = None) -> "KANPredictor":
        wrapper = KANWrapper.load(model_path, device=device)
        with open(pca_path, "rb") as f:
            pca_state = pickle.load(f)
        of_pipe = pca_state.get("of_pipe", None)
        fr_pipe = pca_state.get("fr_pipe", None)
        return cls(wrapper=wrapper, of_pipe=of_pipe, fr_pipe=fr_pipe, device=device)

    def transform(self, X_only_fixed: np.ndarray, X_fixed_and_random: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        of = _transform_pca(self.of_pipe, X_only_fixed)
        fr = _transform_pca(self.fr_pipe, X_fixed_and_random) if (X_fixed_and_random is not None and self.fr_pipe is not None) else np.zeros((of.shape[0], 0), dtype=np.float32)
        X_fix = np.concatenate([of, fr], axis=1) if fr.shape[1] > 0 else of
        return X_fix, of, fr

    def predict_proba_arrays(
        self,
        X_only_fixed: np.ndarray,
        X_fixed_and_random: Optional[np.ndarray],
        y_lags: Optional[np.ndarray] = None,
        dt_lags: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X_fix, TC, Zr = self.transform(X_only_fixed, X_fixed_and_random)
        return self.wrapper.predict_proba(X_fix=X_fix, TC=TC, Zrand=Zr, y_lags=y_lags, dt_lags=dt_lags).ravel()

    def predict_arrays(
        self,
        X_only_fixed: np.ndarray,
        X_fixed_and_random: Optional[np.ndarray],
        y_lags: Optional[np.ndarray] = None,
        dt_lags: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        X_fix, TC, Zr = self.transform(X_only_fixed, X_fixed_and_random)
        return self.wrapper.predict(X_fix=X_fix, TC=TC, Zrand=Zr, y_lags=y_lags, dt_lags=dt_lags, threshold=threshold).ravel()

    def predict_from_pack(
        self,
        pack: Dict[str, Any],
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
        id_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        X_of = pack["X_only_fixed"]
        X_fr = pack.get("X_fixed_and_random", None)
        y_lags = pack.get("y_lags", None)
        dt_lags = pack.get("dt_lags", None)

        probs = self.predict_proba_arrays(X_of, X_fr, y_lags=y_lags, dt_lags=dt_lags)
        preds = (probs >= float(self.wrapper.threshold if threshold is None else threshold)).astype(int)

        df = pd.DataFrame({"y_prob": probs, "y_pred": preds})
        if id_series is not None:
            df.insert(0, "id", id_series.reset_index(drop=True).astype(str))

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            df.to_csv(save_path, index=False)

        return {"proba": probs, "pred": preds, "pred_df": df, "save_path": save_path}
