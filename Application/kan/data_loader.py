# data_loader.py

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd




class KANDataLoader:
    def __init__(self, df_features: pd.DataFrame, df_flags: pd.DataFrame):
        self.df = df_features.copy()
        self.flags = df_flags.copy()
        if self.flags.index.name is None:
            self.flags.index.name = "column"
        self.flags = self.flags.reindex(self.df.columns, fill_value=False)

    def _cols(self, key: str) -> List[str]:
        if key not in self.flags.columns:
            return []
        mask = self.flags[key].astype(bool)
        return [c for c, m in zip(self.flags.index.tolist(), mask.tolist()) if m and c in self.df.columns]

    def _to_float_block(self, cols: List[str]) -> np.ndarray:
        if len(cols) == 0:
            return np.zeros((len(self.df), 0), dtype=np.float32)
        X = self.df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return X

    def _y_vector(self, cols: List[str]) -> np.ndarray:
        if len(cols) == 0:
            raise ValueError("no target column flagged under 'y'")
        y = self.df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if y.ndim == 2 and y.shape[1] > 1:
            y = y[:, [0]]
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return y

    def _lags_block(self, cols: List[str]) -> Optional[np.ndarray]:
        if len(cols) == 0:
            return None
        L = self.df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return L

    def _pid_time(self, pid_cols: List[str], time_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict[Any, int]]:
        if len(pid_cols) == 0:
            pid_vals = np.arange(len(self.df), dtype=np.int64)
        else:
            s = self.df[pid_cols[0]]
            vals, idx = np.unique(s.astype(str).to_numpy(), return_inverse=True)
            pid_vals = idx.astype(np.int64)
        if len(time_cols) == 0:
            t = np.arange(len(self.df))
        else:
            t = pd.to_numeric(self.df[time_cols[0]], errors="coerce").fillna(0).to_numpy()
        mapping = {}
        if len(pid_cols) > 0:
            s = self.df[pid_cols[0]].astype(str).to_numpy()
            u = np.unique(s)
            mapping = {k: int(i) for i, k in enumerate(u)}
        return pid_vals, t, mapping

    def prepare_train(self) -> Dict[str, Any]:
        cols_of = self._cols("only_fixed")
        cols_fr = self._cols("fixed_and_random")
        cols_y = self._cols("y")
        cols_y_lags = self._cols("y_lags")
        cols_dt_lags = self._cols("dt_lags")
        cols_pid = self._cols("pid")
        cols_time = self._cols("time")

        X_only_fixed = self._to_float_block(cols_of)
        X_fixed_and_random = self._to_float_block(cols_fr)
        y = self._y_vector(cols_y)
        y_lags = self._lags_block(cols_y_lags)
        dt_lags = self._lags_block(cols_dt_lags)
        pid_idx, time_index, pid_map = self._pid_time(cols_pid, cols_time)

        return {
            "X_only_fixed": X_only_fixed,
            "X_fixed_and_random": X_fixed_and_random if X_fixed_and_random.shape[1] > 0 else None,
            "y": y,
            "y_lags": y_lags,
            "dt_lags": dt_lags,
            "pid_idx": pid_idx,
            "time_index": time_index,
            "column_groups": {
                "only_fixed": cols_of,
                "fixed_and_random": cols_fr,
                "y": cols_y,
                "y_lags": cols_y_lags,
                "dt_lags": cols_dt_lags,
                "pid": cols_pid,
                "time": cols_time,
            },
            "pid_map": pid_map,
        }

    def prepare_predict(self) -> Dict[str, Any]:
        cols_of = self._cols("only_fixed")
        cols_fr = self._cols("fixed_and_random")
        cols_y_lags = self._cols("y_lags")
        cols_dt_lags = self._cols("dt_lags")
        cols_pid = self._cols("pid")
        cols_time = self._cols("time")

        X_only_fixed = self._to_float_block(cols_of)
        X_fixed_and_random = self._to_float_block(cols_fr)
        y_lags = self._lags_block(cols_y_lags)
        dt_lags = self._lags_block(cols_dt_lags)
        pid_idx, time_index, pid_map = self._pid_time(cols_pid, cols_time)

        return {
            "X_only_fixed": X_only_fixed,
            "X_fixed_and_random": X_fixed_and_random if X_fixed_and_random.shape[1] > 0 else None,
            "y_lags": y_lags,
            "dt_lags": dt_lags,
            "pid_idx": pid_idx,
            "time_index": time_index,
            "column_groups": {
                "only_fixed": cols_of,
                "fixed_and_random": cols_fr,
                "y_lags": cols_y_lags,
                "dt_lags": cols_dt_lags,
                "pid": cols_pid,
                "time": cols_time,
            },
            "pid_map": pid_map,
        }
