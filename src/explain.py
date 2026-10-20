from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/ids_binary_sample.parquet")
MODEL_PATH = Path("models/random_forest_ids.joblib")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement du modele et des donnees...")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
medians = pd.Series(bundle["medians"])

df = pd.read_parquet(DATA_PATH)

X = df.reindex(columns=features).sample(
    n=min(100, len(df)),
    random_state=42
).copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(medians.reindex(X.columns))
X = X.astype("float32")

print("Nombre d'observations expliquees :", len(X))
print("Calcul des valeurs SHAP...")

explainer = shap.TreeExplainer(model)
values = explainer.shap_values(X)

# Extraire les valeurs SHAP associees a la classe Attack (1)
if isinstance(values, list):
    attack_values = values[1]
else:
    attack_values = np.asarray(values)
    if attack_values.ndim == 3:
        attack_values = attack_values[:, :, 1]

importance = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(attack_values).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

csv_path = REPORT_DIR / "shap_global_importance.csv"
importance.to_csv(csv_path, index=False)

print("\nTop 15 des features les plus influentes :")
print(importance.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))
top = importance.head(15).sort_values("MeanAbsSHAP")

plt.barh(top["Feature"], top["MeanAbsSHAP"])
plt.xlabel("Mean absolute SHAP value")
plt.title("Top 15 - Global SHAP Feature Importance")
plt.tight_layout()

plot_path = REPORT_DIR / "shap_global_importance.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nFichiers crees :")
print(csv_path)
print(plot_path)
