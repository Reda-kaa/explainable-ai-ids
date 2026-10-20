from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

DATA_PATH = Path("data/processed/ids_binary_sample.parquet")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement du dataset...")
df = pd.read_parquet(DATA_PATH)

y = df["Label"].map({"Benign": 0, "Attack": 1})

if y.isna().any():
    raise ValueError("Labels inattendus dans le dataset.")

X = df.drop(columns=["Label"]).copy()

print("Conversion des features en valeurs numeriques...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

# Supprimer les colonnes entièrement vides
X = X.dropna(axis=1, how="all")

# Remplacer les valeurs manquantes par la médiane
medians = X.median()
X = X.fillna(medians).astype("float32")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Des valeurs non finies restent dans les features.")

print("Dimensions :", X.shape)
print("\nRépartition des classes :")
print(y.value_counts().rename(index={0: "Benign", 1: "Attack"}))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEntrainement du modele Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2
)

model.fit(X_train, y_train)

print("\nEvaluation sur le jeu de test...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"F1-score Attack : {f1:.4f}")
print(f"ROC-AUC : {auc:.4f}")

print("\nClassification report :")
report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Attack"],
    digits=4
)
print(report)

print("Matrice de confusion (lignes = réel, colonnes = prédit) :")
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
print(matrix)

model_path = MODEL_DIR / "random_forest_ids.joblib"
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "medians": medians.to_dict(),
        "labels": {0: "Benign", 1: "Attack"}
    },
    model_path
)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.to_csv(REPORT_DIR / "feature_importance.csv", header=["importance"])

with open(REPORT_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1-score Attack: {f1:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(matrix))

print(f"\nModele sauvegarde : {model_path}")
print("Rapports sauvegardes dans reports/")
