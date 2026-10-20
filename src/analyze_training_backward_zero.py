from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

FEATURE = "Total Backward Packets"
EXCLUDED = "DDoS-Friday-no-metadata.parquet"

rows = []

for path in sorted(RAW_DIR.glob("*.parquet")):
    if path.name == EXCLUDED:
        continue

    try:
        columns = pd.read_parquet(path).columns
        label_col = next(
            c for c in columns if c.strip().lower() == "label"
        )
        feature_col = next(
            c for c in columns
            if c.strip().lower() == FEATURE.lower()
        )

        # Read only the label and the feature being investigated
        df = pd.read_parquet(path, columns=[label_col, feature_col])
        df[label_col] = df[label_col].astype(str).str.strip()
        df[feature_col] = pd.to_numeric(
            df[feature_col], errors="coerce"
        )

        df["_class"] = df[label_col].str.lower().eq("benign").map(
            {True: "Benign", False: "Attack"}
        )

        for class_name, group in df.groupby("_class"):
            values = group[feature_col]
            valid = values.notna()

            rows.append({
                "File": path.name,
                "Class": class_name,
                "Total": len(group),
                "ValidValues": int(valid.sum()),
                "ZeroCount": int((values[valid] == 0).sum()),
                "ZeroPercent": (
                    (values[valid] == 0).mean() * 100
                    if valid.any() else float("nan")
                ),
                "Median": values.median(),
                "MissingPercent": values.isna().mean() * 100,
            })

    except Exception as exc:
        print(f"[ERROR] {path.name}: {exc}")

result = pd.DataFrame(rows)
output = REPORTS / "training_backward_zero_distribution.csv"
result.to_csv(output, index=False)

print("\nBackward-packet zero distribution in training files:")
print(result.to_string(
    index=False, float_format=lambda x: f"{x:.3f}"
))
print(f"\nSaved report: {output}")
print("\nDDoS-Friday was intentionally excluded.")
