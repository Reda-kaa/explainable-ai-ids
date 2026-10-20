from pathlib import Path
import joblib
import numpy as np
import pandas as pd

root = Path.cwd()
bundle = joblib.load(root / "models" / "random_forest_holdout_ddos.joblib")
df = pd.read_parquet(root / "data" / "raw" / "DDoS-Friday-no-metadata.parquet")
pred = pd.read_csv(root / "reports" / "ddos_flow_duration_predictions.csv")

benign = df[df["Label"] == "Benign"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
attacks = df[df["Label"] == "DDoS"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
test = pd.concat([benign, attacks]).reset_index(drop=True)

if len(test) != len(pred):
    raise ValueError("Le nombre de lignes ne correspond pas au CSV.")

if not np.array_equal(test["Label"].astype(str), pred["Label"].astype(str)):
    raise ValueError("Les labels ne correspondent pas.")

feature = "Init Bwd Win Bytes"
raw_values = test[feature]
imputed_values = raw_values.replace([np.inf, -np.inf], np.nan)
imputed_values = imputed_values.fillna(bundle["medians"][feature])

test["Probability"] = pred["Probability"]
test["PredictedAttack"] = (
    pred["PredictedAttack"].astype(str).str.lower().isin(["true", "1", "yes"])
)
test["RawFeature"] = raw_values
test["ImputedFeature"] = imputed_values

for group_name, mask in {
    "FN_DDoS": (test["Label"] == "DDoS") & (~test["PredictedAttack"]),
    "TP_DDoS": (test["Label"] == "DDoS") & (test["PredictedAttack"]),
}.items():
    subset = test.loc[mask]
    print(f"\n=== {group_name} ===")
    print("Nombre de flux :", len(subset))
    print("Valeurs manquantes originales :", int(subset["RawFeature"].isna().sum()))
    print("Valeurs originales :")
    print(subset["RawFeature"].value_counts(dropna=False).head(10).to_string())
    print("Valeurs apres imputation :")
    print(subset["ImputedFeature"].value_counts(dropna=False).head(10).to_string())

print("\nMedian utilise pour l'imputation :", bundle["medians"][feature])
print("\nVerification terminee.")
