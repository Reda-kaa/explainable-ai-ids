import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)

root = Path(".")
bundle_path = root / "models" / "random_forest_threshold_tuned_dos_wednesday.joblib"
test_path = root / "data" / "raw" / "DoS-Wednesday-no-metadata.parquet"

bundle = joblib.load(bundle_path)
model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["features"]

df = pd.read_parquet(test_path)
label_col = "Label" if "Label" in df.columns else "label"
y_text = df[label_col].astype(str).str.strip()
y_all = (y_text.str.lower() != "benign").astype("int8")

# Reproduce the same class sampling used by the evaluation script.
parts = []
for cls in (0, 1):
    idx = np.flatnonzero(y_all.to_numpy() == cls)
    if len(idx) > 5000:
        rng = np.random.default_rng(42 + cls)
        idx = rng.choice(idx, size=5000, replace=False)
    parts.extend(idx.tolist())

df = df.iloc[parts].copy()
y = y_all.iloc[parts].to_numpy()

X = df.drop(columns=[label_col]).replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.reindex(columns=features)

X_imp = imputer.transform(X)
attack_index = list(model.classes_).index(1)
probs = model.predict_proba(X_imp)[:, attack_index]

print("Independent DoS-Wednesday test")
print("ROC-AUC:", round(roc_auc_score(y, probs), 4))

for threshold in (0.10, 0.22, 0.50):
    pred = (probs >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y, pred):.4f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y, pred, zero_division=0):.4f}")
    print(f"FPR:       {fp / (fp + tn):.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
