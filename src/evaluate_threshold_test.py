
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

DATA_FILE = "DDoS-Friday-no-metadata.parquet"
MODEL_FILE = ROOT / "models" / "random_forest_threshold_tuned.joblib"
THRESHOLDS = [0.50, 0.15]


def load_test_sample(path):
    df = pd.read_parquet(path)

    label_col = next(
        c for c in df.columns if c.strip().lower() == "label"
    )
    labels = df[label_col].astype(str).str.strip()
    binary = np.where(labels.str.lower().eq("benign"), 0, 1)

    X = df.drop(columns=[label_col])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X["Label"] = binary

    parts = []
    for target in [0, 1]:
        group = X[X["Label"] == target]
        if len(group) > 5000:
            group = group.sample(n=5000, random_state=42)
        parts.append(group)

    test = pd.concat(parts, ignore_index=True)
    y = test.pop("Label").to_numpy(dtype=int)
    return test, y


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = bundle["features"]
    medians = pd.Series(bundle["medians"])
    chosen_threshold = float(bundle["threshold"])

    path = RAW_DIR / DATA_FILE
    print(f"Loading independent test file: {DATA_FILE}")
    X, y = load_test_sample(path)

    X = X.reindex(columns=features)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(medians.reindex(features))

    probabilities = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probabilities)

    rows = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y, pred),
            "Precision_Attack": precision_score(
                y, pred, zero_division=0
            ),
            "Recall_Attack": recall_score(
                y, pred, zero_division=0
            ),
            "F1_Attack": f1_score(
                y, pred, zero_division=0
            ),
            "ROC_AUC": auc,
            "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    results = pd.DataFrame(rows)
    output = REPORTS_DIR / "threshold_test_ddos.csv"
    results.to_csv(output, index=False)

    print(f"\nTest rows: {len(y)}")
    print(f"Selected validation threshold: {chosen_threshold:.2f}")
    print("ROC-AUC (threshold-independent):", round(auc, 4))
    print("\nIndependent DDoS test results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {output}")
    print(
        "\nReminder: this test file was not used to train the model "
        "or select the threshold."
    )


if __name__ == "__main__":
    main()
