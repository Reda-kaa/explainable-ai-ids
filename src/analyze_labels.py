from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
results = []

for path in sorted(RAW_DIR.glob("*.parquet")):
    try:
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema.names

        label_col = next(
            (c for c in columns if c.strip().lower() == "label"),
            None
        )

        if label_col is None:
            print(f"[SKIP] No label column: {path.name}")
            continue

        # Read only labels to keep RAM usage low
        labels = pd.read_parquet(path, columns=[label_col])[label_col]
        labels = labels.astype("string").str.strip()

        counts = labels.value_counts(dropna=False)

        print(f"\n{'=' * 65}")
        print(f"File: {path.name}")
        print(f"Total rows: {len(labels):,}")
        print("Labels:")

        for label, count in counts.items():
            print(f"  {label}: {count:,} ({count / len(labels) * 100:.2f}%)")

            results.append({
                "File": path.name,
                "Label": str(label),
                "Count": int(count),
                "Percentage": round(count / len(labels) * 100, 4)
            })

    except Exception as exc:
        print(f"[ERROR] {path.name}: {exc}")

output = REPORTS_DIR / "attack_label_distribution.csv"
pd.DataFrame(results).to_csv(output, index=False)

print(f"\nReport saved to: {output}")
