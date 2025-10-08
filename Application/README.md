# KAN App — Train & Predict

App to train a  Mixed-Effects inspired KAN-based classifier and run predictions.

---

## 1) Setup

```bash
pip install -U pip
pip install -r requirements.txt

```

 ## 2) Run the UI
python main.py --ui


## 3) Expected column names to flags which feature gets into which component

"only_fixed"      : 0 or more columns
"fixed_and_random": 0 or more columns
"y"               : exactly 1 column for TRAINING; 0 columns for PREDICTION
"y_lag"           : 0 or more columns (L); if present, "dt_lag" must have the same L
"dt_lag"          : 0 or more columns; count must equal "y_lag"
"pid"             : 0 or 1 column (1 enables participant-grouped split)
"time"            : 0 or 1 column
