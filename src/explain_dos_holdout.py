from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]

data_path = ROOT / "data" / "raw" / "DoS-Wednesday-no-metadata.parquet"
model_path = ROOT / "models" / "random_forest_threshold_tuned_dos_wednesday.joblib"
output_path = ROOT / "reports" / "dos_missed_attacks_shap.csv"

THRESHOLD = 0.10
MAX_EXPLAIN = 20

print("Loading saved model...")
bundle = joblib.load(model_path)
model = bundle["model"]
imputer = bundle["imputer"]
features = bundle["features"]

print("Loading DoS-Wednesday test data...")
df = pd.read_parquet(data_path)

label_col = next(
    c for c in df.columns if c.strip().lower() == "label"
)
labels = df[label_col].astype(str).str.strip()
y_all = (labels.str.lower() != "benign").astype("int8")

# Reproduce the deterministic class sampling used in the evaluation.
parts = []
for target in [0, 1]:
    indices = np.flatnonzero(y_all.to_numpy() == target)
    if len(indices) > 5000:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, size=5000, replace=False)
    parts.extend(indices.tolist())

test_df = df.iloc[parts].copy().reset_index(drop=True)
y = (
    test_df[label_col].astype(str).str.strip().str.lower() != "benign"
).astype("int8").to_numpy()

X = test_df.drop(columns=[label_col])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.reindex(columns=features)

X_imp_array = imputer.transform(X)
X_imp = pd.DataFrame(X_imp_array, columns=features)

attack_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X_imp)[:, attack_index]

# False negatives for the selected operational threshold.
false_negatives = np.flatnonzero((y == 1) & (probabilities < THRESHOLD))

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Total test rows: {len(y)}")
print(f"Missed attacks: {len(false_negatives)}")

if len(false_negatives) == 0:
    print("No missed attacks found.")
    raise SystemExit(0)

# Explain the 20 attacks with the lowest predicted attack probabilities.
selected = false_negatives[
    np.argsort(probabilities[false_negatives])[:MAX_EXPLAIN]
]
X_explain = X_imp.iloc[selected]

print(f"Explaining {len(selected)} missed attacks with SHAP...")
explainer = shap.TreeExplainer(model)
raw_values = explainer.shap_values(X_explain)
values = np.asarray(raw_values)

# SHAP may return (samples, features, classes) for binary classifiers.
if values.ndim == 3:
    if values.shape[1] == len(features) and values.shape[2] == 2:
        values = values[:, :, attack_index]
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

if values.shape != X_explain.shape:
    raise ValueError(
        f"SHAP shape {values.shape} does not match input {X_explain.shape}"
    )

importance = np.abs(values).mean(axis=0)
signed_effect = values.mean(axis=0)
top_indices = np.argsort(importance)[::-1][:15]

global_results = pd.DataFrame({
    "Feature": np.array(features)[top_indices],
    "MeanAbsSHAP": importance[top_indices],
    "MeanSignedSHAP": signed_effect[top_indices],
})
global_results.to_csv(output_path, index=False)

# Also save case-level probabilities and the top feature contributions.
case_rows = []
for row_number, original_index in enumerate(selected):
    row = {
        "TestRowIndex": int(original_index),
        "Actual": "Attack",
        "PredictedAttackProbability": float(probabilities[original_index]),
        "Threshold": THRESHOLD,
    }
    local_order = np.argsort(np.abs(values[row_number]))[::-1][:5]
    for rank, feature_index in enumerate(local_order, start=1):
        row[f"TopFeature{rank}"] = features[feature_index]
        row[f"SHAP{rank}"] = float(values[row_number, feature_index])
    case_rows.append(row)

case_path = ROOT / "reports" / "dos_missed_attacks_shap_cases.csv"
pd.DataFrame(case_rows).to_csv(case_path, index=False)

print("\nTop features influencing the selected missed attacks:")
print(global_results.to_string(index=False))
print(f"\nSaved feature report: {output_path}")
print(f"Saved case-level report: {case_path}")
