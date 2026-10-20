from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "DDoS-Friday-no-metadata.parquet"
MODEL = ROOT / "models" / "random_forest_threshold_tuned.joblib"
OUT = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL)
model = bundle["model"]
features = bundle["features"]
medians = pd.Series(bundle["medians"])

df = pd.read_parquet(RAW)
label_col = next(c for c in df.columns if c.strip().lower() == "label")
labels = df[label_col].astype(str).str.strip().str.lower()
df["_target"] = (~labels.eq("benign")).astype(int)

# Same deterministic sample as the previous test
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

fn = (y == 1) & (pred == 0)
tp = (y == 1) & (pred == 1)

print(f"Missed attacks (FN): {fn.sum()}")
print(f"Detected attacks (TP): {tp.sum()}")

print("\nAttack probability distribution for missed attacks:")
print(pd.Series(prob[fn]).describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
).to_string())

selected = [
    "Total Backward Packets",
    "Fwd Packet Length Min",
    "Fwd Packet Length Max",
    "Bwd Packets Length Total",
    "ACK Flag Count",
    "Bwd Header Length",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Duration",
    "Flow Packets/s",
    "Flow Bytes/s",
]

rows = []
for feature in selected:
    if feature not in X.columns:
        continue

    row = {"Feature": feature}
    for name, mask in [("FN", fn), ("TP", tp)]:
        values = X.loc[mask, feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        row[f"{name}_Zero_Percent"] = (
            (values == 0).mean() * 100 if len(values) else np.nan
        )
        row[f"{name}_Q25"] = values.quantile(0.25) if len(values) else np.nan
        row[f"{name}_Median"] = values.median() if len(values) else np.nan
        row[f"{name}_Q75"] = values.quantile(0.75) if len(values) else np.nan

    rows.append(row)

result = pd.DataFrame(rows)
path = OUT / "ddos_zero_value_analysis.csv"
result.to_csv(path, index=False)

print("\nZero percentages and quantiles (FN vs TP):")
print(result.to_string(
    index=False, float_format=lambda v: f"{v:.3f}"
))
print(f"\nSaved report: {path}")
