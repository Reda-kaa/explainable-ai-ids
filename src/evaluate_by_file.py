
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

HOLDOUT_FILE = "DDoS-Friday-no-metadata.parquet"
MAX_PER_CLASS_PER_FILE = 5000
RANDOM_STATE = 42


def load_sample(path):
    print(f"Reading: {path.name}")
    df = pd.read_parquet(path)

    label_col = next(
        (col for col in df.columns if col.strip().lower() == "label"),
        None,
    )
    if label_col is None:
        raise ValueError(f"No Label column found in {path.name}")

    labels = df[label_col].astype(str).str.strip()
    binary_labels = np.where(
        labels.str.lower().eq("benign"), "Benign", "Attack"
    )

    df = df.drop(columns=[label_col])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df["Label"] = binary_labels

    parts = []
    for label in ["Benign", "Attack"]:
        group = df[df["Label"] == label]
        if len(group) > MAX_PER_CLASS_PER_FILE:
            group = group.sample(
                n=MAX_PER_CLASS_PER_FILE,
                random_state=RANDOM_STATE,
            )
        if not group.empty:
            parts.append(group)

    if not parts:
        raise ValueError(f"No usable rows found in {path.name}")

    result = pd.concat(parts, ignore_index=True)
    print(f"  Sampled rows: {len(result)}")
    print(f"  Classes: {result['Label'].value_counts().to_dict()}")
    return result


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    holdout_path = RAW_DIR / HOLDOUT_FILE
    if not holdout_path.exists():
        raise FileNotFoundError(f"File not found: {holdout_path}")

    train_files = sorted(
        p for p in RAW_DIR.glob("*.parquet")
        if p.name != HOLDOUT_FILE
    )
    if not train_files:
        raise FileNotFoundError(f"No training Parquet files in {RAW_DIR}")

    train_parts = [load_sample(path) for path in train_files]
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = load_sample(holdout_path)

    # Use training feature columns only.
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"].map({"Benign": 0, "Attack": 1})

    X_test = test_df.drop(columns=["Label"]).reindex(
        columns=X_train.columns
    )
    y_test = test_df["Label"].map({"Benign": 0, "Attack": 1})

    # Learn missing-value replacements from training data only.
    medians = X_train.median().replace([np.inf, -np.inf], np.nan)
    usable_columns = medians[medians.notna()].index.tolist()

    X_train = X_train[usable_columns].fillna(medians[usable_columns])
    X_test = X_test[usable_columns].fillna(medians[usable_columns])

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Training and test data must each contain both classes.")

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows:  {len(X_test)}")
    print(f"Features:      {len(usable_columns)}")
    print("\nTraining a separate Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    output = (
        "HELD-OUT FILE EVALUATION: DDoS-Friday\n"
        "Training data: all other Parquet files\n"
        "Test data: DDoS-Friday only\n\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Features: {len(usable_columns)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"ROC-AUC: {auc:.4f}\n\n"
        "Classification report:\n"
        f"{report}\n"
        "Confusion matrix (rows=actual, columns=predicted):\n"
        f"{matrix}\n"
    )

    model_path = MODELS_DIR / "random_forest_holdout_ddos.joblib"
    report_path = REPORTS_DIR / "holdout_ddos_evaluation.txt"

    joblib.dump(
        {
            "model": model,
            "features": usable_columns,
            "medians": medians[usable_columns].to_dict(),
            "labels": {0: "Benign", 1: "Attack"},
            "test_file": HOLDOUT_FILE,
        },
        model_path,
    )
    report_path.write_text(output, encoding="utf-8")

    print("\n" + output)
    print(f"Model saved to:  {model_path}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
