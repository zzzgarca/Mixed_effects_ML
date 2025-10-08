# -*- coding: utf-8 -*-

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd

try:
    from kan.command import KANCommand
except Exception:
    # Stub pentru simulare UI, ca aplicația să pornească și fără pachetul KAN instalat
    class KANCommand:
        def train(self, **kwargs):
            return {
                "metrics": {"AUC": 0.0},
                "predictions_path": os.path.join(kwargs.get("output_dir", "."), "predictions.csv"),
                "predictions": pd.DataFrame({"y_pred": []}),
                "model_path": os.path.join(kwargs.get("output_dir", "."), "model.pt"),
                "pca_path": os.path.join(kwargs.get("output_dir", "."), "pca.pkl"),
            }


class KANUI:
    def __init__(self, root=None):
        self.root = root or tk.Tk()
        self.root.title("KAN pentru serii de timp")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)
        self.command = KANCommand()

        # Căi fișiere (menținem variabilele pentru compatibilitate, dar nu afișăm meniurile eliminate)
        self.features_path = tk.StringVar(value="")
        self.flags_path = tk.StringVar(value="")  # ascuns în UI

        # Hiperparametri / setări
        self.lr = tk.StringVar(value="0.0003")
        self.weight_decay = tk.StringVar(value="0.0")
        self.batch_size = tk.StringVar(value="256")
        self.epochs = tk.StringVar(value="50")
        self.threshold = tk.StringVar(value="0.5")  # păstrat intern; caseta din UI a fost ștearsă
        self.random_state = tk.StringVar(value="42")
        self.save_prefix = tk.StringVar(value="kan_model")

        # NOU: câmpuri cerute
        self.noduri_kan = tk.StringVar(value="64")
        self.putere_kan = tk.StringVar(value="2")

        # Predict
        self.model_path = tk.StringVar(value="")
        self.pca_path = tk.StringVar(value="")  # ascuns în UI
        self.predict_threshold = tk.StringVar(value="")  # păstrat intern; caseta din UI a fost ștearsă

        # Bife (fără funcționalitate deocamdată)
        self.multi_task = tk.BooleanVar(value=False)
        self.hierarchical = tk.BooleanVar(value=False)

        self.models_dir = os.path.abspath("./models")
        self.outputs_dir = os.path.abspath("./output_data")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        self._build_layout()

    def _build_layout(self):
        pad = {"padx": 8, "pady": 6}

        # ===== Date =====
        frm_load = ttk.LabelFrame(self.root, text="Date")
        frm_load.grid(row=0, column=0, sticky="nsew", **pad)
        frm_load.columnconfigure(1, weight=1)

        ttk.Label(frm_load, text="Date CSV").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_load, textvariable=self.features_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(frm_load, text="Răsfoiește…", command=self._pick_features).grid(row=0, column=2)

        # ===== Antrenare =====
        frm_train = ttk.LabelFrame(self.root, text="Antrenare")
        frm_train.grid(row=1, column=0, sticky="nsew", **pad)
        # Vom organiza câmpurile "una după alta" (fără spații goale), 4 perechi pe rând
        max_pairs_per_row = 4
        for c in range(max_pairs_per_row * 2):
            frm_train.columnconfigure(c, weight=1)

        pairs = [
            ("lr", self.lr),
            ("weight_decay", self.weight_decay),
            ("batch_size", self.batch_size),
            ("epoci", self.epochs),  # traducere
            ("random_state", self.random_state),
            ("noduri_kan", self.noduri_kan),
            ("putere_kan", self.putere_kan),
            ("prefix salvare", self.save_prefix),
        ]

        for i, (label_text, var) in enumerate(pairs):
            r = i // max_pairs_per_row
            c = (i % max_pairs_per_row) * 2
            ttk.Label(frm_train, text=label_text).grid(row=r, column=c, sticky="w")
            ttk.Entry(frm_train, width=12, textvariable=var).grid(row=r, column=c + 1, sticky="ew")

        # Rândul următor pentru bife și buton
        last_row = (len(pairs) - 1) // max_pairs_per_row + 1
        ttk.Checkbutton(frm_train, text="Multi-sarcină", variable=self.multi_task).grid(row=last_row, column=0, sticky="w")
        ttk.Checkbutton(frm_train, text="Ierarhic", variable=self.hierarchical).grid(row=last_row, column=2, sticky="w")
        ttk.Button(frm_train, text="Pornește antrenarea", command=self._start_training).grid(row=last_row, column=max_pairs_per_row * 2 - 1, sticky="e")

        # ===== Prezicere =====
        frm_predict = ttk.LabelFrame(self.root, text="Predicție")
        frm_predict.grid(row=2, column=0, sticky="nsew", **pad)
        for c in range(4):
            frm_predict.columnconfigure(c, weight=1)

        ttk.Label(frm_predict, text="Model .pt").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm_predict, textvariable=self.model_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(frm_predict, text="Predicție", command=self._start_predict).grid(row=0, column=2, sticky="e")
        ttk.Button(frm_predict, text="Vizualizeza rezultatele").grid(row=0, column=3, sticky="e")

        # ===== Ieșire =====
        frm_out = ttk.LabelFrame(self.root, text="Ieșire")
        frm_out.grid(row=3, column=0, sticky="nsew", **pad)
        frm_out.columnconfigure(0, weight=1)
        frm_out.rowconfigure(0, weight=1)

        self.txt = tk.Text(frm_out, height=22)
        self.txt.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(frm_out, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)

        self._log(f"Director modele: {self.models_dir}")
        self._log(f"Director ieșiri: {self.outputs_dir}")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)  # rândul cu cadrul Ieșire

    # ===== File pickers =====
    def _pick_features(self):
        p = filedialog.askopenfilename(title="Selectează Date CSV", filetypes=[("Fișiere CSV", "*.csv"), ("Toate fișierele", "*.*")])
        if p:
            self.features_path.set(p)

    def _pick_model(self):
        p = filedialog.askopenfilename(title="Selectează Model .pt", initialdir=self.models_dir, filetypes=[("PyTorch", "*.pt"), ("Toate fișierele", "*.*")])
        if p:
            self.model_path.set(p)

    # ===== Data loading =====
    def _load_dfs(self):
        fpath = self.features_path.get().strip()
        if not fpath or not os.path.exists(fpath):
            raise FileNotFoundError("Fișierul de date (CSV) nu este selectat sau nu a fost găsit")
        df_features = pd.read_csv(fpath)

        # Steaguri (flags) opționale în această simulare: dacă lipsesc, generăm un cadru gol compatibil
        gpath = self.flags_path.get().strip()
        if gpath and os.path.exists(gpath):
            df_flags = pd.read_csv(gpath, index_col=0)
        else:
            df_flags = pd.DataFrame(index=df_features.columns)
        return df_features, df_flags

    # ===== Actions =====
    def _start_training(self):
        try:
            df_features, df_flags = self._load_dfs()
            lr = float(self.lr.get())
            weight_decay = float(self.weight_decay.get())
            batch_size = int(self.batch_size.get())
            epochs = int(self.epochs.get())
            threshold = float(self.threshold.get()) if self.threshold.get() else 0.5  # intern
            random_state = int(self.random_state.get())

            # pca_var_ratio a fost șters din UI; folosim o valoare implicită pentru compatibilitate
            pca_var_ratio_default = 0.95
            save_prefix = self.save_prefix.get().strip() or "kan_model"

            # Notăm, dar nu folosim încă, noile câmpuri
            _ = self.noduri_kan.get(), self.putere_kan.get()

            out = self.command.train(
                df_features=df_features,
                df_flags=df_flags,
                arch_overrides=None,
                random_state=random_state,
                pca_var_ratio=pca_var_ratio_default,
                output_dir=self.models_dir,
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                epochs=epochs,
                threshold=threshold,
                pos_weight=None,
                verbose=True,
                save_prefix=save_prefix,
            )

            metrics = out.get("metrics", {})
            self._log("Antrenarea s-a încheiat.")
            if metrics:
                self._log("Metrice:")
                for k in ["AUC", "AUPRC", "Brier", "ACC", "F1", "Precision", "Recall", "Sensitivity", "Specificity"]:
                    if k in metrics:
                        v = metrics[k]
                        try:
                            self._log(f"  {k}: {float(v):.4f}")
                        except Exception:
                            self._log(f"  {k}: {v}")

            src_pred_path = out.get("predictions_path", "predictions.csv")
            base = os.path.basename(src_pred_path)
            dst_pred_path = os.path.join(self.outputs_dir, base)
            try:
                if "predictions" in out:
                    out["predictions"].to_csv(dst_pred_path, index=False)
            except Exception:
                dst_pred_path = src_pred_path

            model_path = out.get("model_path", "model.pt")
            pca_path = out.get("pca_path", "pca.pkl")
            self._log(f"Model salvat: {model_path}")
            self._log(f"PCA salvat: {pca_path}")
            self._log(f"CSV predicții: {dst_pred_path}")

        except Exception as e:
            messagebox.showerror("Eroare la antrenare", str(e))

    def _start_predict(self):
        try:
            df_features, df_flags = self._load_dfs()
            model_path = self.model_path.get().strip()
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError("Fișierul model .pt nu este selectat sau nu a fost găsit")

            # PCA a fost scos din UI. Încercăm fără pca_path; dacă biblioteca necesită PCA, afișăm mesaj prietenos
            pca_path = self.pca_path.get().strip() or None

            from kan.data_loader import KANDataLoader
            loader = KANDataLoader(df_features, df_flags)
            pack = loader.prepare_predict()

            from kan.predictor import KANPredictor
            if pca_path:
                predictor = KANPredictor.load(model_path=model_path, pca_path=pca_path)
            else:
                predictor = KANPredictor.load(model_path=model_path)

            # Fără caseta de prag în UI -> folosim None
            thr = None

            fname = os.path.splitext(os.path.basename(model_path))[0] + "_predictions.csv"
            save_path = os.path.join(self.outputs_dir, fname)

            mask = df_flags.reindex(df_features.columns, fill_value=False) if not df_flags.empty else pd.DataFrame(index=df_features.columns)
            id_series = None
            if not df_flags.empty and "pid" in mask.columns:
                pid_cols = [c for c, v in mask["pid"].items() if bool(v)]
                if pid_cols and pid_cols[0] in df_features.columns:
                    id_series = df_features[pid_cols[0]]

            out = predictor.predict_from_pack(pack, threshold=thr, save_path=save_path, id_series=id_series)

            self._log("Prezicerea s-a încheiat.")
            self._log(f"CSV predicții: {out.get('save_path', save_path)}")

        except Exception as e:
            # mesaj simplu, fără formatare specială
            messagebox.showerror("Eroare la prezicere", str(e))

    def _log(self, msg: str):
        self.txt.insert("end", msg + "")
        self.txt.see("end")


if __name__ == "__main__":
    app = KANUI()
    app.root.mainloop()
