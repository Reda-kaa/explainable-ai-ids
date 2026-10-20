
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]

data_path = (
    ROOT / "data" / "raw" /
    "DDoS-Friday-no-metadata.parquet"
)
model_path = ROOT / "models" / "random_forest_holdout_ddos.joblib"
output_path = ROOT / "reports" / "ddos_missed_attacks_shap.csv"

bundle = joblib.load(model_path)
model = bundle["model"]
features = bundle["features"]
medians = pd.Series(bundle["medians"])

print("Loading DDoS test data...")
df = pd.read_parquet(data_path)

label_col = next(
    c for c in df.columns if c.strip().lower() == "label"
)
labels = df[label_col].astype(str).str.strip()
y = (labels.str.lower() != "benign").astype(int)

# Reproduce the same deterministic test sampling.
parts = []
for target in [0, 1]:
    group = df.loc[y == target]
    if len(group) > 5000:
        group = group.sample(n=5000, random_state=42)
    parts.append(group)

test_df = pd.concat(parts).reset_index(drop=True)
test_labels = (
    test_df[label_col].astype(str).str.strip().str.lower()
    != "benign"
).astype(int)

X = test_df.drop(columns=[label_col])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.reindex(columns=features)
X = X.fillna(medians.reindex(features))

predictions = model.predict(X)
false_negatives = np.flatnonzero(
    (test_labels.to_numpy() == 1) & (predictions == 0)
)

if len(false_negatives) == 0:
    print("No missed attacks found.")
    raise SystemExit(0)

# Explain up to 20 missed attacks to limit RAM usage.
selected = false_negatives[:20]
X_explain = X.iloc[selected]

print(f"Missed attacks found: {len(false_negatives)}")
print(f"Explaining {len(selected)} missed attacks with SHAP...")

explainer = shap.TreeExplainer(model)

values = explainer.shap_values(X_explain)
print("Model classes:", model.classes_)
print("SHAP expected_value:", explainer.expected_value)
print("Raw SHAP shape:", np.asarray(values).shape)
print("First sample, first 5 features, both classes:")
print(np.asarray(values)[0, :5, :])
print("SHAP sum for sample 0, class 0:",
      np.asarray(values)[0, :, 0].sum())
print("SHAP sum for sample 0, class 1:",
      np.asarray(values)[0, :, 1].sum())
print("Model probabilities for sample 0:",
      model.predict_proba(X_explain.iloc[[0]])[0])

print("SHAP output type:", type(values))

if isinstance(values, list):
    print("SHAP class output shapes:", [np.asarray(v).shape for v in values])
    values = values[1]
else:
    values = np.asarray(values)
    print("SHAP output shape:", values.shape)


if values.ndim == 3:
    # Shape is (samples, features, classes).
    if values.shape[1] == len(features) and values.shape[2] == 2:
        values = values[:, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")


if values.shape != X_explain.shape:
    raise ValueError(
        f"SHAP shape {values.shape} does not match input {X_explain.shape}"
    )

importance = np.abs(values).mean(axis=0)
signed_effect = values.mean(axis=0)


if isinstance(values, list):
    values = values[1]
values = np.asarray(values)

# Handle SHAP outputs that include a class dimension.
if values.ndim == 3:
    values = values[:, :, 1]

importance = np.abs(values).mean(axis=0)
signed_effect = values.mean(axis=0)

top_indices = np.argsort(importance)[::-1][:15]

global_results = pd.DataFrame({
    "Feature": np.array(features)[top_indices],
    "MeanAbsSHAP": importance[top_indices],
    "MeanSignedSHAP": signed_effect[top_indices],
})

global_results.to_csv(output_path, index=False)

print("\nTop features influencing these missed attacks:")
print(global_results.to_string(index=False))
print(f"\nSaved report: {output_path}")

top_indices = np.argsort(importance)[::-1][:15]

global_results = pd.DataFrame({
    "Feature": np.array(features)[top_indices],
    "MeanAbsSHAP": importance[top_indices],
})

global_results.to_csv(output_path, index=False)

print("\nTop features influencing these missed attacks:")
print(global_results.to_string(index=False))
print(f"\nSaved report: {output_path}")
