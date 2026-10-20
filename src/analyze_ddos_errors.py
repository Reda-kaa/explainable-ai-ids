from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "DDoS-Friday-no-metadata.parquet"
MODEL_FILE = ROOT / "models" / "random_forest_threshold_tuned.joblib"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50
SAMPLE_PER_CLASS = 5000

print("Loading model and DDoS test data...")
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]
medians = pd.Series(bundle["medians"])

# Read the independent test file
df = pd.read_parquet(RAW_FILE)
label_col = next(c for c in df.columns if c.strip().lower() == "label")

labels = df[label_col].astype(str).str.strip()
y_all = (~labels.str.lower().eq("benign")).astype(int)

# Reproduce the same deterministic test sampling
parts = []
for target in [0, 1]:
    group = df.loc[y_all == target].copy()
    if len(group) > SAMPLE_PER_CLASS:
        group = group.sample(n=SAMPLE_PER_CLASS, random_state=42)
    parts.append(group)

test = pd.concat(parts, ignore_index=True)
y = (~test.pop(label_col).astype(str).str.strip().str.lower().eq("benign")).astype(int).to_numpy()

X = test.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
X = X.reindex(columns=features)
X = X.fillna(medians.reindex(features))

probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

print(f"\nThreshold: {THRESHOLD}")
print(f"True Positives (detected attacks): {tp}")
print(f"False Negatives (missed attacks): {fn}")
print(f"False Positives (false alarms): {fp}")
print(f"True Negatives: {tn}")

# Compare missed attacks with correctly detected attacks only
fn_mask = (y == 1) & (predictions == 0)
tp_mask = (y == 1) & (predictions == 1)

X_fn = X.loc[fn_mask]
X_tp = X.loc[tp_mask]

print(f"\nRows used for comparison: FN={len(X_fn)}, TP={len(X_tp)}")

if X_fn.empty or X_tp.empty:
    raise RuntimeError("One comparison group is empty; cannot compare features.")

rows = []
for feature in features:
    missed = X_fn[feature].replace([np.inf, -np.inf], np.nan).dropna()
    detected = X_tp[feature].replace([np.inf, -np.inf], np.nan).dropna()

    if missed.empty or detected.empty:
        continue

    fn_median = missed.median()
    tp_median = detected.median()
    combined = pd.concat([missed, detected], ignore_index=True)
    q25, q75 = combined.quantile([0.25, 0.75])
    iqr = q75 - q25

    # Robust difference, scaled by the combined interquartile range
    scaled_diff = (
        (fn_median - tp_median) / iqr
        if pd.notna(iqr) and iqr > 0
        else np.nan
    )

    rows.append({
        "Feature": feature,
        "FN_Median": fn_median,
        "TP_Median": tp_median,
        "Median_Difference_FN_minus_TP": fn_median - tp_median,
        "Combined_IQR": iqr,
        "Scaled_Median_Difference": scaled_diff,
        "Abs_Scaled_Difference": abs(scaled_diff)
            if pd.notna(scaled_diff) else np.nan,
    })

comparison = pd.DataFrame(rows)
comparison = comparison.sort_values(
    "Abs_Scaled_Difference", ascending=False, na_position="last"
)

output = REPORTS / "ddos_fn_vs_tp_features.csv"
comparison.to_csv(output, index=False)

print("\nTop 20 features with the largest robust median differences:")
print(comparison.head(20).to_string(
    index=False, float_format=lambda value: f"{value:.5g}"
))
print(f"\nFull report saved to: {output}")
print("\nNote: this is diagnostic analysis of the held-out test sample.")
