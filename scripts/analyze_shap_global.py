from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path.cwd()
MODEL_PATH = ROOT / "models" / "random_forest_holdout_ddos.joblib"
DATA_PATH = ROOT / "data" / "raw" / "DDoS-Friday-no-metadata.parquet"
PRED_PATH = ROOT / "reports" / "ddos_flow_duration_predictions.csv"
OUT_DIR = ROOT / "reports" / "shap_global"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_GROUP = 300
BACKGROUND_SIZE = 100
SEED = 42

print("Chargement du modele et des donnees...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]
medians = bundle["medians"]

df = pd.read_parquet(DATA_PATH)
pred = pd.read_csv(PRED_PATH)

# Reconstituer exactement l'echantillon de l'evaluation initiale
benign = df[df["Label"] == "Benign"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
attacks = df[df["Label"] == "DDoS"].sample(
    n=5000, random_state=np.random.RandomState(42)
)
test = pd.concat([benign, attacks]).reset_index(drop=True)

if len(test) != len(pred):
    raise ValueError(
        f"Nombre de lignes incoherent: test={len(test)}, predictions={len(pred)}"
    )

if not np.array_equal(
    test["Label"].astype(str).to_numpy(),
    pred["Label"].astype(str).to_numpy()
):
    raise ValueError("Les labels du dataset ne correspondent pas au rapport CSV.")

X = test[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(pd.Series(medians).reindex(features))
X = X.astype(float)

test["Probability"] = pred["Probability"].to_numpy()
test["PredictedAttack"] = (
    pred["PredictedAttack"].astype(str).str.strip().str.lower()
    .isin(["true", "1", "yes"])
)
test["Init Bwd Win Bytes"] = X["Init Bwd Win Bytes"].to_numpy()

fn = test[
    (test["Label"] == "DDoS") & (~test["PredictedAttack"])
].copy()

tp = test[
    (test["Label"] == "DDoS") & (test["PredictedAttack"])
].copy()

print(f"Total FN dans l'evaluation : {len(fn)}")
print(f"Total TP dans l'evaluation : {len(tp)}")

# Echantillonner separement chaque groupe pour une comparaison equitable
n_fn = min(N_PER_GROUP, len(fn))
n_tp = min(N_PER_GROUP, len(tp))

fn_sample = fn.sample(n=n_fn, random_state=SEED)
tp_sample = tp.sample(n=n_tp, random_state=SEED)

cases = pd.concat([
    fn_sample.assign(CaseGroup="FN"),
    tp_sample.assign(CaseGroup="TP"),
]).copy()

X_cases = X.loc[cases.index].copy()
X_cases.index = cases["CaseGroup"].to_numpy() + "_" + cases.index.astype(str)

# Background mixte Benign + DDoS, sans les cas expliques
rng = np.random.RandomState(SEED)
excluded = cases.index.to_numpy()
available = np.setdiff1d(np.arange(len(X)), excluded)
bg_idx = rng.choice(
    available, size=min(BACKGROUND_SIZE, len(available)), replace=False
)
background = X.iloc[bg_idx]

print(f"Echantillon SHAP : {len(fn_sample)} FN + {len(tp_sample)} TP")
print(f"Background de reference : {len(background)} flux")
print("Calcul SHAP en cours; cela peut prendre quelques minutes...")

explainer = shap.TreeExplainer(
    model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional",
)

raw = explainer.shap_values(X_cases)

# SHAP binaire: extraire la classe 1 = Attack
if isinstance(raw, list):
    values = np.asarray(raw[1])
else:
    values = np.asarray(raw)
    if values.ndim == 3:
        if values.shape[1] == len(features):
            values = values[:, :, 1]
        elif values.shape[2] == len(features):
            values = values[:, 1, :]

if values.shape != (len(cases), len(features)):
    raise ValueError(
        f"Format SHAP inattendu: {values.shape}; "
        f"attendu {(len(cases), len(features))}"
    )

# Exporter toutes les contributions signees
long_rows = []
for i in range(len(cases)):
    group = cases.iloc[i]["CaseGroup"]
    idx = cases.index[i]
    for j, feature in enumerate(features):
        sv = float(values[i, j])
        long_rows.append({
            "DatasetIndex": int(idx),
            "CaseGroup": group,
            "Feature": feature,
            "FeatureValue": float(X.loc[idx, feature]),
            "SHAP": sv,
            "AbsoluteSHAP": abs(sv),
        })

long_df = pd.DataFrame(long_rows)
long_df.to_csv(OUT_DIR / "shap_all_sampled_cases.csv", index=False)

# Statistiques par groupe et feature
summary = long_df.groupby(["CaseGroup", "Feature"]).agg(
    MeanSignedSHAP=("SHAP", "mean"),
    MeanAbsoluteSHAP=("AbsoluteSHAP", "mean"),
    MedianAbsoluteSHAP=("AbsoluteSHAP", "median"),
    PctNegative=("SHAP", lambda s: 100 * (s < 0).mean()),
    PctPositive=("SHAP", lambda s: 100 * (s > 0).mean()),
    MeanFeatureValue=("FeatureValue", "mean"),
).reset_index()

summary.to_csv(OUT_DIR / "shap_global_feature_summary.csv", index=False)

# Comparaison directe FN vs TP pour chaque feature
wide = summary.pivot(index="Feature", columns="CaseGroup")
wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
wide = wide.reset_index()

wide["AbsSHAP_Difference_TP_minus_FN"] = (
    wide["MeanAbsoluteSHAP_TP"] - wide["MeanAbsoluteSHAP_FN"]
)
wide["SignedSHAP_Difference_TP_minus_FN"] = (
    wide["MeanSignedSHAP_TP"] - wide["MeanSignedSHAP_FN"]
)
wide["NegativeRate_Difference_TP_minus_FN"] = (
    wide["PctNegative_TP"] - wide["PctNegative_FN"]
)
wide.to_csv(OUT_DIR / "shap_FN_TP_comparison.csv", index=False)

print("\n=== TOP 15 FN: FEATURES QUI DIMINUENT P(ATTACK) ===")
fn_neg = summary[
    (summary["CaseGroup"] == "FN") & (summary["MeanSignedSHAP"] < 0)
].sort_values("MeanSignedSHAP")
print(
    fn_neg[[
        "Feature", "MeanSignedSHAP", "MeanAbsoluteSHAP",
        "PctNegative", "MeanFeatureValue"
    ]].head(15).to_string(index=False)
)

print("\n=== TOP 15 TP: FEATURES QUI AUGMENTENT P(ATTACK) ===")
tp_pos = summary[
    (summary["CaseGroup"] == "TP") & (summary["MeanSignedSHAP"] > 0)
].sort_values("MeanSignedSHAP", ascending=False)
print(
    tp_pos[[
        "Feature", "MeanSignedSHAP", "MeanAbsoluteSHAP",
        "PctPositive", "MeanFeatureValue"
    ]].head(15).to_string(index=False)
)

print("\n=== INIT BWD WIN BYTES ===")
print(
    summary[summary["Feature"] == "Init Bwd Win Bytes"][
        ["CaseGroup", "MeanFeatureValue", "MeanSignedSHAP",
         "MeanAbsoluteSHAP", "PctNegative", "PctPositive"]
    ].to_string(index=False)
)

print("\n=== PLUS GRANDES DIFFERENCES D'IMPORTANCE FN / TP ===")
comparison = wide.sort_values(
    "AbsSHAP_Difference_TP_minus_FN", ascending=False
)
cols = [
    "Feature", "MeanAbsoluteSHAP_FN", "MeanAbsoluteSHAP_TP",
    "AbsSHAP_Difference_TP_minus_FN",
    "MeanSignedSHAP_FN", "MeanSignedSHAP_TP"
]
print(comparison[cols].head(15).to_string(index=False))

print("\nFichiers sauvegardes dans:", OUT_DIR)
print("- shap_all_sampled_cases.csv")
print("- shap_global_feature_summary.csv")
print("- shap_FN_TP_comparison.csv")
print("\nANALYSE GLOBALE TERMINEE")
