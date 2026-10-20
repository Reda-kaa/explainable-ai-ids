
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HOLDOUT_FILE = "DDoS-Friday-no-metadata.parquet"
MAX_PER_CLASS_PER_FILE = 5000
RANDOM_STATE = 42
MAX_FPR = 0.05


def load_sample(path):
    print(f"Loading {path.name}...")
    df = pd.read_parquet(path)

    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"),
        None,
    )
    if label_col is None:
        raise ValueError(f"Label column missing: {path.name}")

    labels = df[label_col].astype(str).str.strip()
    y = np.where(labels.str.lower().eq("benign"), 0, 1)

    X = df.drop(columns=[label_col])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    parts = []
    for target in [0, 1]:
        indices = np.flatnonzero(y == target)
        if len(indices) > MAX_PER_CLASS_PER_FILE:
            rng = np.random.RandomState(RANDOM_STATE)
            indices = rng.choice(
                indices,
                MAX_PER_CLASS_PER_FILE,
                replace=False,
            )
        if len(indices):
            part = X.iloc[indices].copy()
            part["Label"] = y[indices]
            parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    print(f"  Sampled: {len(result)} rows")
    return result


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision_Attack": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall_Attack": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Attack": f1_score(
            y_true, predictions, zero_division=0
        ),
        "F2_Attack": fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        ),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in RAW_DIR.glob("*.parquet")
        if p.name != HOLDOUT_FILE
    )
    if not paths:
        raise FileNotFoundError(f"No training files in {RAW_DIR}")

    frames = [load_sample(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)

    X = data.drop(columns=["Label"])
    y = data["Label"].astype(int)

    # Validation is taken only from the non-DDoS training files.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit preprocessing on training rows only.
    medians = X_train.median()
    usable_features = medians[medians.notna()].index.tolist()
    medians = medians[usable_features]

    X_train = X_train[usable_features].fillna(medians)
    X_val = X_val.reindex(columns=usable_features).fillna(medians)

    print(f"\nTraining rows:   {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Features:        {len(usable_features)}")
    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    # These validation probabilities select the operating threshold.
    val_probabilities = model.predict_proba(X_val)[:, 1]
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    results = pd.DataFrame([
        metrics_at_threshold(y_val, val_probabilities, t)
        for t in thresholds
    ])

    results_path = REPORTS_DIR / "threshold_validation.csv"
    results.to_csv(results_path, index=False)

    eligible = results[results["FPR"] <= MAX_FPR]
    if eligible.empty:
        # Conservative fallback: choose the lowest validation FPR.
        chosen = results.sort_values(
            ["FPR", "Threshold"],
            ascending=[True, False],
        ).iloc[0]
        selection_note = (
            "No tested threshold met the 5% FPR target; "
            "selected the lowest-FPR candidate."
        )
    else:
        # Prioritize recall while keeping validation FPR <= 5%.
        chosen = eligible.sort_values(
            ["Recall_Attack", "Threshold"],
            ascending=[False, False],
        ).iloc[0]
        selection_note = (
            "Selected the highest validation attack recall "
            "among thresholds with FPR <= 5%."
        )

    threshold = float(chosen["Threshold"])

    bundle = {
        "model": model,
        "features": usable_features,
        "medians": medians.to_dict(),
        "labels": {0: "Benign", 1: "Attack"},
        "threshold": threshold,
        "threshold_selection": selection_note,
        "validation_max_fpr": MAX_FPR,
        "test_file": HOLDOUT_FILE,
    }

    model_path = MODELS_DIR / "random_forest_threshold_tuned.joblib"
    joblib.dump(bundle, model_path)

    print("\nValidation results by threshold:")
    print(results[
        [
            "Threshold",
            "Precision_Attack",
            "Recall_Attack",
            "F1_Attack",
            "FPR",
            "FP",
            "FN",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSelected threshold:", threshold)
    print(selection_note)
    print(f"Validation table saved: {results_path}")
    print(f"Model bundle saved: {model_path}")
    print(
        "\nDDoS-Friday was not used for training, "
        "validation, or threshold selection."
    )


if __name__ == "__main__":
    main()
