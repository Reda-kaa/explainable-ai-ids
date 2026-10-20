from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS_PER_FILE = 5000
RANDOM_STATE = 42

files = sorted(RAW_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError("Aucun fichier Parquet trouvé dans data/raw")

samples = []

for file in files:
    print(f"\nLecture : {file.name}")
    df = pd.read_parquet(file)

    if "Label" not in df.columns:
        print("Fichier ignoré : colonne Label absente.")
        continue

    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)

    df["Label"] = np.where(df["Label"].str.casefold() == "benign",
                           "Benign", "Attack")

    benign = df[df["Label"] == "Benign"]
    attacks = df[df["Label"] == "Attack"]

    n_benign = min(SAMPLES_PER_CLASS_PER_FILE, len(benign))
    n_attacks = min(SAMPLES_PER_CLASS_PER_FILE, len(attacks))

    if n_benign:
        samples.append(
            benign.sample(n=n_benign, random_state=RANDOM_STATE)
        )

    if n_attacks:
        samples.append(
            attacks.sample(n=n_attacks, random_state=RANDOM_STATE)
        )

    print(f"Benign sélectionnés : {n_benign}")
    print(f"Attack sélectionnés : {n_attacks}")

    del df, benign, attacks

result = pd.concat(samples, ignore_index=True)
result = result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

output_file = OUT_DIR / "ids_binary_sample.parquet"
result.to_parquet(output_file, index=False)

print("\n--- Résultat final ---")
print("Dimensions :", result.shape)
print("\nRépartition des classes :")
print(result["Label"].value_counts())
print("\nFichier créé :", output_file)
