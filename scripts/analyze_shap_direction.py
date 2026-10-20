from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "random_forest_holdout_ddos.joblib"
DATA_PATH = ROOT / "data" / "raw" / "DDoS-Friday-no-metadata.parquet"
PRED_PATH = ROOT / "reports" / "ddos_flow_duration_predictions.csv"
OUT_DIR = ROOT / "reports" / "shap_direction"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Chargement du modele, des donnees et des predictions...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
medians = bundle["medians"]

df = pd.read_parquet(DATA_PATH)
pred = pd.read_csv(PRED_PATH)

# Reconstituer exactement l'echantillon utilise lors de l'evaluation
benign = df[df["Label"] == "Benign"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
attacks = df[df["Label"] == "DDoS"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
test = pd.concat([benign, attacks]).reset_index(drop=True)

if len(test) != len(pred):
    raise ValueError(
        f"Taille incoherente: test={len(test)}, predictions={len(pred)}"
    )

if not np.array_equal(
    test["Label"].astype(str).to_numpy(),
    pred["Label"].astype(str).to_numpy()
):
    raise ValueError("Les labels ne correspondent pas au rapport de predictions.")

X = test[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(pd.Series(medians).reindex(features))
X = X.astype(float)

test["Probability"] = pred["Probability"].to_numpy()
test["PredictedAttack"] = pred["PredictedAttack"].astype(bool).to_numpy()
test["Init Bwd Win Bytes"] = X["Init Bwd Win Bytes"].to_numpy()

# Selectionner un FN et un TP representatifs parmi les groupes connus
fn = test[
    (test["Label"] == "DDoS")
    & (~test["PredictedAttack"])
    & (test["Init Bwd Win Bytes"] == -1)
].copy()

tp = test[
    (test["Label"] == "DDoS")
    & (test["PredictedAttack"])
    & (test["Init Bwd Win Bytes"] > 0)
].copy()

if fn.empty or tp.empty:
    raise ValueError("Impossible de trouver un FN et un TP correspondants.")

# Choisir des cas proches des probabilites medianes de chaque groupe
fn_case = fn.iloc[(fn["Probability"] - fn["Probability"].median()).abs().argmin()]
tp_case = tp.iloc[(tp["Probability"] - tp["Probability"].median()).abs().argmin()]

cases = pd.DataFrame([
    {"Case": "FN_missed_attack", "Index": int(fn_case.name)},
    {"Case": "TP_detected_attack", "Index": int(tp_case.name)},
])

case_indices = cases["Index"].tolist()
X_cases = X.loc[case_indices].copy()
X_cases.index = cases["Case"]

# Background mixte, sans inclure les deux cas expliques
rng = np.random.RandomState(42)
available = np.setdiff1d(np.arange(len(X)), np.array(case_indices))
bg_indices = rng.choice(available, size=min(100, len(available)), replace=False)
background = X.iloc[bg_indices]

print("Calcul des SHAP values signees...")
explainer = shap.TreeExplainer(
    model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional",
)

raw = explainer.shap_values(X_cases)

# Normaliser les formats renvoyes par differentes versions de SHAP
if isinstance(raw, list):
    # RandomForest binaire: choisir la classe 1 = Attack
    values = np.asarray(raw[1])
else:
    values = np.asarray(raw)
    if values.ndim == 3:
        # Selon la version, axe classe dernier ou premier
        if values.shape[1] == len(features):
            values = values[:, :, 1]
        elif values.shape[2] == len(features):
            values = values[1, :, :]

if values.shape != (2, len(features)):
    raise ValueError(
        f"Format SHAP inattendu: {values.shape}; "
        f"attendu (2, {len(features)})"
    )

expected = np.asarray(explainer.expected_value)
base = float(expected[1] if expected.ndim > 0 else expected)

probs = model.predict_proba(X_cases)[:, 1]
reconstructed = base + values.sum(axis=1)

summary = pd.DataFrame({
    "Case": cases["Case"],
    "DatasetIndex": case_indices,
    "TrueLabel": ["DDoS", "DDoS"],
    "PredictedProbabilityAttack": probs,
    "SHAP_Base_Attack": base,
    "Base_plus_SHAP": reconstructed,
    "AdditivityError": np.abs(probs - reconstructed),
    "Init Bwd Win Bytes": X_cases["Init Bwd Win Bytes"].to_numpy(),
})
summary.to_csv(OUT_DIR / "selected_cases.csv", index=False)

# Contributions signees par feature et par cas
rows = []
for i, case_name in enumerate(cases["Case"]):
    for j, feature in enumerate(features):
        rows.append({
            "Case": case_name,
            "Feature": feature,
            "FeatureValue": X_cases.iloc[i, j],
            "SHAP": values[i, j],
            "AbsoluteSHAP": abs(values[i, j]),
            "Direction": (
                "Augmente P(Attack)" if values[i, j] > 0
                else "Diminue P(Attack)" if values[i, j] < 0
                else "Neutre"
            ),
        })

details = pd.DataFrame(rows)
details.to_csv(OUT_DIR / "signed_shap_all_features.csv", index=False)

for i, case_name in enumerate(cases["Case"]):
    case_details = details[details["Case"] == case_name].copy()
    case_details["Abs"] = case_details["SHAP"].abs()
    case_details = case_details.sort_values("Abs", ascending=False)

    print(f"\n=== {case_name} ===")
    print(f"Probabilite Attack: {probs[i]:.6f}")
    print(f"Init Bwd Win Bytes: {X_cases.iloc[i]['Init Bwd Win Bytes']}")
    print("\nTop contributions signees:")
    print(
        case_details[["Feature", "FeatureValue", "SHAP", "Direction"]]
        .head(12).to_string(index=False)
    )

    # Waterfall plot avec valeurs signees
    explanation = shap.Explanation(
        values=values[i],
        base_values=base,
        data=X_cases.iloc[i].to_numpy(),
        feature_names=features,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"SHAP - {case_name}")
    plt.tight_layout()
    safe_name = case_name.lower()
    plt.savefig(OUT_DIR / f"{safe_name}_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close()

print("\n=== VERIFICATION ADDITIVITE ===")
print(summary.to_string(index=False))
print("Erreur maximale:", summary["AdditivityError"].max())

print("\nFichiers sauvegardes dans:", OUT_DIR)
print("- selected_cases.csv")
print("- signed_shap_all_features.csv")
print("- fn_missed_attack_waterfall.png")
print("- tp_detected_attack_waterfall.png")
print("\nANALYSE TERMINEE")
