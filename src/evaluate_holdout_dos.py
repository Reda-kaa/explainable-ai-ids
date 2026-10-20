from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

TEST_FILE = "DoS-Wednesday-no-metadata.parquet"
SAMPLE_PER_CLASS = 5000
THRESHOLD = 0.50

def load_sample(path, per_class=5000):
    df = pd.read_parquet(path)
    label_col = next(c for c in df.columns if c.strip().lower() == "label")

    labels = df[label_col].astype(str).str.strip().str.lower()
    y = (~labels.eq("benign")).astype(int)

    X = df.drop(columns=[label_col])
    X.columns = X.columns.astype(str).str.strip()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Deterministic balanced sample for each class
    parts = []
    for target in [0, 1]:
        idx = np.flatnonzero(y.to_numpy() == target)
        if len(idx) > per_class:
            rng = np.random.RandomState(42)
            idx = rng.choice(idx, size=per_class, replace=False)
        parts.append(idx)

    selected = np.concatenate(parts)
    return X.iloc[selected].reset_index(drop=True), y.iloc[selected].to_numpy()

def main():
    print("Preparing training data...")
    train_parts = []

    for path in sorted(RAW_DIR.glob("*.parquet")):
        if path.name == TEST_FILE:
            continue

        print(f"Reading: {path.name}")
        X_part, y_part = load_sample(path, SAMPLE_PER_CLASS)
        train_parts.append((X_part, y_part))

    # Use only features shared by all training files and the test file.
    test_path = RAW_DIR / TEST_FILE
    X_test, y_test = load_sample(test_path, SAMPLE_PER_CLASS)

    common_features = list(X_test.columns)
    for X_part, _ in train_parts:
        common_features = [
            c for c in common_features if c in X_part.columns
        ]

    if not common_features:
        raise RuntimeError("No common features found.")

    X_train = pd.concat(
        [X[common_features] for X, _ in train_parts],
        ignore_index=True
    )
    y_train = np.concatenate([y for _, y in train_parts])
    X_test = X_test[common_features]

    # Remove features that are entirely missing in training.
    usable = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, usable]
    X_test = X_test.loc[:, usable]
    features = list(X_train.columns)

    # Learn imputation values from training data only.
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print(f"\nTraining rows: {len(y_train):,}")
    print(f"Test rows: {len(y_test):,}")
    print(f"Features: {len(features)}")
    print(f"Test file: {TEST_FILE}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=2,
    )
    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nThreshold: {THRESHOLD:.2f}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Attack precision: {tp / (tp + fp) if tp + fp else 0:.4f}")
    print(f"Attack recall: {tp / (tp + fn) if tp + fn else 0:.4f}")
    print(f"FPR: {fp / (fp + tn) if fp + tn else 0:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        digits=4,
        zero_division=0
    ))

    # Diagnostic: zero backward packets in each actual/predicted group.
    backward = "Total Backward Packets"
    if backward in features:
        backward_values = pd.Series(
            X_test[:, features.index(backward)]
        ).to_numpy()
        zero_mask = backward_values == 0

        print("\nDetection rate by actual class and backward-packet zeros:")
        for actual in [0, 1]:
            for zero in [False, True]:
                mask = (y_test == actual) & (zero_mask == zero)
                count = int(mask.sum())
                if count:
                    detected = int((predictions[mask] == 1).sum())
                    print(
                        f"Actual={'Attack' if actual else 'Benign'}, "
                        f"BackwardZero={zero}: n={count}, "
                        f"predicted_attack={detected}, "
                        f"rate={detected/count:.4f}"
                    )

    report_path = REPORTS / "holdout_dos_wednesday.txt"
    report_path.write_text(
        f"Test file: {TEST_FILE}\n"
        f"Threshold: {THRESHOLD}\n"
        f"Training rows: {len(y_train)}\n"
        f"Test rows: {len(y_test)}\n"
        f"Features: {len(features)}\n"
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n"
        f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}\n"
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n",
        encoding="utf-8"
    )

    model_path = MODELS / "random_forest_holdout_dos_wednesday.joblib"
    joblib.dump({
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": THRESHOLD,
        "test_file": TEST_FILE,
    }, model_path)

    print(f"\nReport saved: {report_path}")
    print(f"Model saved: {model_path}")
    print("\nNote: DoS-Wednesday was excluded from training.")

if __name__ == "__main__":
    main()
