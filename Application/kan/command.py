# command.py

from typing import Optional, Dict, Any
import os
import pandas as pd

from kan.data_loader import KANDataLoader
from kan.evaluator   import KANEvaluator
from kan.predictor   import KANPredictor


class KANCommand:
    def __init__(self):
        pass

    def train(
        self,
        df_features: pd.DataFrame,
        df_flags: pd.DataFrame,
        *,
        arch_overrides: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        pca_var_ratio: float = 0.95,
        output_dir: str = "kan_run",
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 256,
        epochs: int = 50,
        threshold: float = 0.5,
        pos_weight: Optional[float] = None,
        verbose: bool = True,
        save_prefix: str = "kan_model",
        group_by_pid: bool = True,   # <-- NEW
    ) -> Dict[str, Any]:
        loader = KANDataLoader(df_features, df_flags)
        pack = loader.prepare_train()
        ev = KANEvaluator(
            arch_overrides=arch_overrides,
            random_state=random_state,
            pca_var_ratio=pca_var_ratio,
            output_dir=output_dir,
        )
        out = ev.fit_eval(
            X_only_fixed=pack["X_only_fixed"],
            X_fixed_and_random=pack["X_fixed_and_random"],
            y=pack["y"],
            y_lags=pack["y_lags"],
            dt_lags=pack["dt_lags"],
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            threshold=threshold,
            pos_weight=pos_weight,
            verbose=verbose,
            save_prefix=save_prefix,
            pid_idx=pack.get("pid_idx", None),   
            group_by_pid=group_by_pid,          
        )
        return out

    
    def predict(
        self,
        df_features: pd.DataFrame,
        df_flags: pd.DataFrame,
        *,
        model_path: str,
        pca_path: str,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
        output_dir: str = "./output_data",
    ) -> Dict[str, Any]:
        loader = KANDataLoader(df_features, df_flags)
        pack = loader.prepare_predict()

        if save_path is None:
            base = os.path.splitext(os.path.basename(model_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{base}_predictions.csv")
        else:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        pred = KANPredictor.load(model_path=model_path, pca_path=pca_path)

        mask = df_flags.reindex(df_features.columns, fill_value=False)
        id_series = None
        if "pid" in mask.columns:
            pid_cols = [c for c, v in mask["pid"].items() if bool(v)]
            if pid_cols and pid_cols[0] in df_features.columns:
                id_series = df_features[pid_cols[0]]

        out = pred.predict_from_pack(pack, threshold=threshold, save_path=save_path, id_series=id_series)
        return out


    def run(
        self,
        task: str,
        df_features: pd.DataFrame,
        df_flags: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        t = task.lower().strip()
        if t in ("train", "fit"):
            return self.train(df_features, df_flags, **kwargs)
        elif t in ("predict", "inference", "infer"):
            return self.predict(df_features, df_flags, **kwargs)
        else:
            raise ValueError("task must be 'train' or 'predict'")
