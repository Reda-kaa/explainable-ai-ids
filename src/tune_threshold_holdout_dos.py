from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_FILE = "DoS-Wednesday-no-metadata.parquet"
MAX_PER_CLASS_PER_FILE = 5000
RANDOM_STATE = 42
FPR_LIMIT = 0.05

LABEL_CANDIDATES = ["Label", "label"]
NON_FEATURE_COLUMNS = {"Label", "label", "Attack", "attack"}


def find_label_column(df):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("Impossible de trouver la colonne Label.")


def load_sample(path, max_per_class=MAX_PER_CLASS_PER_FILE):
    df = pd.read_parquet(path)
    label_col = find_label_column(df)

    labels = df[label_col].astype(str).str.strip()
    y = (labels.str.lower() != "benign").astype("int8")

    df = df.drop(columns=[label_col])
    df = df.replace([np.inf, -np.inf], np.nan)

    # CIC-IDS2017 features should be numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.select_dtypes(include=[np.number])

    sampled_parts = []
    for cls in (0, 1):
        indices = np.flatnonzero(y.to_numpy() == cls)
        if len(indices) > max_per_class:
            rng = np.random.default_rng(RANDOM_STATE + cls)
            indices = rng.choice(indices, size=max_per_class, replace=False)

        if len(indices):
            part = df.iloc[indices].copy()
            part["__target__"] = y.iloc[indices].to_numpy()
            sampled_parts.append(part)

    if not sampled_parts:
        raise ValueError(f"Aucune donnée exploitable dans {path.name}")

    result = pd.concat(sampled_parts, ignore_index=True)
    return result


def metrics_at_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision_attack": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall_attack": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1_attack": f1_score(y_true, predictions, zero_division=0),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    train_files = [
        p for p in parquet_files if p.name != HOLDOUT_FILE
    ]
    holdout_path = RAW_DIR / HOLDOUT_FILE

    if not holdout_path.exists():
        raise FileNotFoundError(f"Fichier test absent : {holdout_path}")
    if not train_files:
        raise FileNotFoundError(f"Aucun fichier d'entraînement dans {RAW_DIR}")

    print("=== 1. Chargement des fichiers d'entraînement ===")
    training_parts = []

    for path in train_files:
        part = load_sample(path)
        print(
            f"{path.name}: {len(part):,} lignes, "
            f"Benign={(part['__target__'] == 0).sum():,}, "
            f"Attack={(part['__target__'] == 1).sum():,}"
        )
        training_parts.append(part)

    training_df = pd.concat(training_parts, ignore_index=True)
    y_all = training_df.pop("__target__").astype("int8")

    # Use only features common to all training rows (columns were aligned by concat).
    X_all = training_df.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.dropna(axis=1, how="all")
    feature_names = X_all.columns.tolist()

    print(f"\nTotal entraînement : {len(X_all):,} lignes")
    print(f"Nombre de features : {len(feature_names)}")

    print("\n=== 2. Séparation entraînement / validation ===")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    imputer_val = SimpleImputer(strategy="median")
    X_train_imp = imputer_val.fit_transform(X_train)
    X_val_imp = imputer_val.transform(X_val)

    validation_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    validation_model.fit(X_train_imp, y_train)

    attack_index = list(validation_model.classes_).index(1)
    val_probabilities = validation_model.predict_proba(X_val_imp)[:, attack_index]

    auc = roc_auc_score(y_val, val_probabilities)
    print(f"Validation ROC-AUC : {auc:.4f}")

    print("\n=== 3. Sélection du threshold sur validation uniquement ===")
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = [
        metrics_at_threshold(y_val, val_probabilities, t)
        for t in thresholds
    ]
    validation_results = pd.DataFrame(rows)
    validation_csv = REPORTS_DIR / "threshold_validation_holdout_dos.csv"
    validation_results.to_csv(validation_csv, index=False)

    eligible = validation_results[
        validation_results["fpr"] <= FPR_LIMIT
    ]

    if eligible.empty:
        # Conservative fallback: choose the threshold with the lowest validation FPR.
        chosen_row = validation_results.sort_values(
            ["fpr", "threshold"], ascending=[True, False]
        ).iloc[0]
        print("Aucun threshold ne respecte la limite FPR ; fallback conservateur.")
    else:
        # Maximize recall, then prefer lower FPR, then higher threshold.
        chosen_row = eligible.sort_values(
            ["recall_attack", "fpr", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]

    chosen_threshold = float(chosen_row["threshold"])
    print(f"Limite FPR validation : {FPR_LIMIT:.1%}")
    print(f"Threshold sélectionné : {chosen_threshold:.2f}")
    print(f"Validation recall     : {chosen_row['recall_attack']:.4f}")
    print(f"Validation FPR         : {chosen_row['fpr']:.4f}")
    print(f"Rapport validation     : {validation_csv}")

    print("\n=== 4. Réentraînement sur tous les fichiers hors holdout ===")
    final_imputer = SimpleImputer(strategy="median")
    X_all_imp = final_imputer.fit_transform(X_all)

    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )
    final_model.fit(X_all_imp, y_all)

    bundle_path = MODELS_DIR / "random_forest_threshold_tuned_dos_wednesday.joblib"
    joblib.dump(
        {
            "model": final_model,
            "imputer": final_imputer,
            "features": feature_names,
            "threshold": chosen_threshold,
            "excluded_holdout_file": HOLDOUT_FILE,
        },
        bundle_path,
    )
    print(f"Modèle sauvegardé : {bundle_path}")

    print("\n=== 5. Évaluation finale sur DoS-Wednesday (jamais utilisé avant) ===")
    holdout_df = load_sample(holdout_path)
    y_test = holdout_df.pop("__target__").astype("int8")

    # Align test features to the training feature set.
    X_test = holdout_df.reindex(columns=feature_names)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test_imp = final_imputer.transform(X_test)

    test_attack_index = list(final_model.classes_).index(1)
    test_probabilities = final_model.predict_proba(X_test_imp)[:, test_attack_index]

    result_rows = []
    for threshold in sorted(set([0.50, chosen_threshold])):
        result = metrics_at_threshold(y_test, test_probabilities, threshold)
        result["test_roc_auc"] = roc_auc_score(y_test, test_probabilities)
        result["test_rows"] = len(y_test)
        result_rows.append(result)

    test_results = pd.DataFrame(result_rows)
    test_csv = REPORTS_DIR / "threshold_test_dos_wednesday.csv"
    test_results.to_csv(test_csv, index=False)

    print(f"Test ROC-AUC (indépendant) : {roc_auc_score(y_test, test_probabilities):.4f}")
    print("\nRésultats test :")
    print(test_results.to_string(index=False))
    print(f"\nRapport test : {test_csv}")
    print("\nTerminé. Le fichier DoS-Wednesday n'a servi ni à choisir le threshold ni à entraîner le modèle.")


if __name__ == "__main__":
    main()
