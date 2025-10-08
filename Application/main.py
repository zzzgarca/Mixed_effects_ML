# main.py

import argparse
import os
import sys
import pandas as pd

from kan.command import KANCommand



def _load_csvs(features_path: str, flags_path: str):
    if not features_path or not os.path.exists(features_path):
        raise FileNotFoundError("features CSV not found")
    if not flags_path or not os.path.exists(flags_path):
        raise FileNotFoundError("flags CSV not found")
    df_features = pd.read_csv(features_path)
    df_flags = pd.read_csv(flags_path, index_col=0)
    return df_features, df_flags


def main():
    p = argparse.ArgumentParser(prog="kan_app")
    sub = p.add_subparsers(dest="task")

    p.add_argument("--ui", action="store_true")

    pt = sub.add_parser("train")
    pt.add_argument("--features", required=True)
    pt.add_argument("--flags", required=True)
    pt.add_argument("--output_dir", default="./models")
    pt.add_argument("--save_prefix", default="kan_model")
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--batch_size", type=int, default=256)
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--threshold", type=float, default=0.5)
    pt.add_argument("--random_state", type=int, default=42)
    pt.add_argument("--pca_var_ratio", type=float, default=0.95)
    pt.add_argument("--quiet", action="store_true")

    pp = sub.add_parser("predict")
    pp.add_argument("--features", required=True)
    pp.add_argument("--flags", required=True)
    pp.add_argument("--model_path", required=True)
    pp.add_argument("--pca_path", required=True)
    pp.add_argument("--threshold", type=float, default=None)
    pp.add_argument("--output_dir", default="./output_data")
    pp.add_argument("--save_path", default=None)

    args = p.parse_args()

    if args.ui:
        from ui.app import KANUI
        app = KANUI()
        app.root.mainloop()
        return

    if args.task is None:
        p.print_help()
        sys.exit(1)

    cmd = KANCommand()

    if args.task == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        df_features, df_flags = _load_csvs(args.features, args.flags)
        out = cmd.train(
            df_features=df_features,
            df_flags=df_flags,
            arch_overrides=None,
            random_state=args.random_state,
            pca_var_ratio=args.pca_var_ratio,
            output_dir=args.output_dir,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            threshold=args.threshold,
            pos_weight=None,
            verbose=not args.quiet,
            save_prefix=args.save_prefix,
        )
        metrics = out.get("metrics", {})
        print("metrics:")
        for k in ["AUC", "AUPRC", "Brier", "ACC", "F1", "Precision", "Recall", "Sensitivity", "Specificity"]:
            if k in metrics:
                v = metrics[k]
                print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        print("model_path:", os.path.abspath(out.get("model_path", "")))
        print("pca_path:", os.path.abspath(out.get("pca_path", "")))
        print("predictions_path:", os.path.abspath(out.get("predictions_path", "")))

        return

    if args.task == "predict":
        df_features, df_flags = _load_csvs(args.features, args.flags)
        if args.save_path is None:
            base = os.path.splitext(os.path.basename(args.model_path))[0]
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, f"{base}_predictions.csv")
        else:
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            save_path = args.save_path
        out = cmd.predict(
            df_features=df_features,
            df_flags=df_flags,
            model_path=args.model_path,
            pca_path=args.pca_path,
            threshold=args.threshold,
            save_path=save_path,
        )
        print("predictions_path:", out.get("save_path"))
        return


if __name__ == "__main__":
    main()
