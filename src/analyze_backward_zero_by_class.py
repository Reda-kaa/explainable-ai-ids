from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "DDoS-Friday-no-metadata.parquet"
MODEL = ROOT / "models" / "random_forest_threshold_tuned.joblib"

bundle = joblib.load(MODEL)
model = bundle["model"]
features = bundle["features"]
medians = pd.Series(bundle["medians"])

df = pd.read_parquet(RAW)
label_col = next(c for c in df.columns if c.strip().lower() == "label")
labels = df[label_col].astype(str).str.strip().str.lower()
df["_target"] = (~labels.eq("benign")).astype(int)

# Reproduce the same independent test sample
parts = []
for target in [0, 1]:
    group = df[df["_target"] == target].copy()
    if len(group) > 5000:
        group = group.sample(n=5000, random_state=42)
    parts.append(group)

test = pd.concat(parts, ignore_index=True)
y = test.pop("_target").to_numpy(dtype=int)
test = test.drop(columns=[label_col])

X = test.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.reindex(columns=features)
X = X.fillna(medians.reindex(features))

prob = model.predict_proba(X)[:, 1]
pred = (prob >= 0.50).astype(int)

backward_col = "Total Backward Packets"
if backward_col not in X.columns:
    raise ValueError(f"Missing feature: {backward_col}")

zero_backward = X[backward_col].eq(0).to_numpy()
result = pd.DataFrame({
    "Actual": np.where(y == 1, "DDoS", "Benign"),
    "BackwardPacketsZero": zero_backward,
    "Predicted": np.where(pred == 1, "Attack", "Benign"),
    "AttackProbability": prob,
})

print("\nCounts and detection rates by actual class and backward packets:")
summary = result.groupby(
    ["Actual", "BackwardPacketsZero"], observed=True
).agg(
    Flows=("Predicted", "size"),
    MeanAttackProbability=("AttackProbability", "mean"),
    MedianAttackProbability=("AttackProbability", "median"),
    PredictedAttackPercent=("Predicted", lambda s: (s == "Attack").mean() * 100),
).reset_index()

print(summary.to_string(
    index=False, float_format=lambda v: f"{v:.4f}"
))

out = ROOT / "reports" / "ddos_backward_zero_by_class.csv"
summary.to_csv(out, index=False)
print(f"\nSaved: {out}")
