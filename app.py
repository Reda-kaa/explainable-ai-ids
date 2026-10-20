from pathlib import Path
from io import BytesIO

import shap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_ids.joblib"

st.set_page_config(
    page_title="AI-IDS | Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

@st.cache_resource
def load_bundle():
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Le fichier modèle n'a pas le format attendu.")
    if "features" not in bundle:
        raise ValueError("La liste des features est absente du modèle.")
    return bundle

def read_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if filename.endswith(".csv"):
        return pd.read_csv(BytesIO(content), low_memory=False)
    if filename.endswith(".parquet"):
        return pd.read_parquet(BytesIO(content))

    raise ValueError("Format non supporté. Utilisez CSV ou Parquet.")

def prepare_features(data, bundle):
    features = list(bundle["features"])
    missing = [col for col in features if col not in data.columns]

    if missing:
        preview = ", ".join(missing[:15])
        if len(missing) > 15:
            preview += f", ... (+{len(missing) - 15} autres)"
        raise ValueError(
            f"{len(missing)} features requises sont absentes : {preview}"
        )

    X = data[features].copy()

    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    medians = bundle.get("medians", {})
    if hasattr(medians, "to_dict"):
        medians = medians.to_dict()

    for col in features:
        median = medians.get(col, 0) if isinstance(medians, dict) else 0
        try:
            median = float(median)
            if not np.isfinite(median):
                median = 0.0
        except (TypeError, ValueError):
            median = 0.0

        X[col] = X[col].fillna(median)

    return X

st.title("🛡️ AI-IDS")
st.subheader("Explainable Network Intrusion Detection System")
st.write(
    "Analysez des flux réseau et identifiez les activités potentiellement "
    "malveillantes grâce à un modèle Random Forest."
)

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Impossible de charger le modèle : {exc}")
    st.stop()

model = bundle["model"]
features = list(bundle["features"])
labels = bundle.get("labels", {0: "Benign", 1: "Attack"})

st.sidebar.header("Configuration")
st.sidebar.success("Random Forest chargé")
st.sidebar.caption(f"{len(features)} features attendues")

threshold = st.sidebar.slider(
    "Seuil de détection",
    min_value=0.01,
    max_value=0.99,
    value=0.50,
    step=0.01,
    help="Une probabilité supérieure ou égale au seuil est classée Attack.",
)

uploaded_file = st.file_uploader(
    "Importer un fichier de flux réseau",
    type=["csv", "parquet"],
    help="Le fichier doit contenir les colonnes attendues par le modèle.",
)

if uploaded_file is None:
    st.info("Importez un fichier CSV ou Parquet pour lancer la détection.")
    st.markdown(
        """
        **Fonctionnement**
        1. Import du fichier réseau.
        2. Vérification et préparation des features.
        3. Prédiction avec Random Forest.
        4. Résultats téléchargeables au format CSV.
        """
    )
    st.stop()

try:
    data = read_uploaded_file(uploaded_file)
except Exception as exc:
    st.error(f"Erreur de lecture du fichier : {exc}")
    st.stop()

if data.empty:
    st.warning("Le fichier ne contient aucune ligne.")
    st.stop()

st.write(f"Fichier importé : **{uploaded_file.name}**")
st.write(f"Nombre de flows : **{len(data):,}**")

try:
    X = prepare_features(data, bundle)
    classes = list(model.classes_)
    attack_class = 1 if 1 in classes else None

    if attack_class is None:
        st.error("La classe Attack (1) est absente du modèle.")
        st.stop()

    attack_index = classes.index(attack_class)
    probabilities = model.predict_proba(X)[:, attack_index]
    predicted = (probabilities >= threshold).astype(int)

except Exception as exc:
    st.error(f"Erreur pendant la préparation ou la prédiction : {exc}")
    st.stop()

benign_label = labels.get(0, "Benign")
attack_label = labels.get(1, "Attack")

results = data.copy()
results["Attack_Probability"] = probabilities
results["Predicted_Class"] = predicted
results["Prediction"] = np.where(
    predicted == 1, str(attack_label), str(benign_label)
)

n_attack = int(predicted.sum())
n_benign = len(predicted) - n_attack
attack_rate = 100 * n_attack / len(predicted)

st.divider()
st.header("Résultats de détection")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total flows", f"{len(results):,}")
c2.metric("Attacks détectées", f"{n_attack:,}")
c3.metric("Benign détectés", f"{n_benign:,}")
c4.metric("Part classée Attack", f"{attack_rate:.2f}%")

left, right = st.columns(2)

with left:
    st.subheader("Répartition des prédictions")
    distribution = pd.DataFrame(
        {"Nombre de flows": [n_benign, n_attack]},
        index=[str(benign_label), str(attack_label)],
    )
    st.bar_chart(distribution)

with right:
    st.subheader("Probabilités d'attaque")
    fig, ax = plt.subplots()
    ax.hist(probabilities, bins=30)
    ax.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}",
    )
    ax.set_xlabel("Probabilité d'attaque")
    ax.set_ylabel("Nombre de flows")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Flows détectés comme Attack")
attacks = results[results["Predicted_Class"] == 1].sort_values(
    "Attack_Probability", ascending=False
)

if attacks.empty:
    st.success("Aucun flow classé Attack avec ce seuil.")
else:
    st.dataframe(attacks.head(100), use_container_width=True)

st.caption("L'aperçu est limité à 100 flows pour garder une interface fluide.")

st.subheader("Télécharger les résultats")
csv_bytes = results.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="Télécharger toutes les prédictions (CSV)",
    data=csv_bytes,
    file_name="ai_ids_predictions.csv",
    mime="text/csv",
)

st.divider()
st.header("🔎 Explainability — SHAP")
st.write(
    "Comprenez les facteurs qui influencent les prédictions du modèle. "
    "Les valeurs SHAP expliquent la contribution de chaque feature "
    "à la sortie du modèle."
)

try:
    with st.spinner("Calcul des explications SHAP..."):
        # Limiter l'échantillon pour garder une interface réactive.
        sample_size = min(200, len(X))
        X_sample = X.iloc[:sample_size].copy()

        explainer = shap.TreeExplainer(model)
        raw_shap = explainer.shap_values(X_sample)

        # Compatibilité avec les différentes versions de SHAP.
        classes = list(model.classes_)
        attack_index = classes.index(1)

        if isinstance(raw_shap, list):
            attack_shap = np.asarray(raw_shap[attack_index])
        else:
            raw_shap = np.asarray(raw_shap)

            if raw_shap.ndim == 3:
                if raw_shap.shape == (
                    len(X_sample), len(features), len(classes)
                ):
                    attack_shap = raw_shap[:, :, attack_index]
                elif raw_shap.shape[0] == len(classes):
                    attack_shap = raw_shap[attack_index]
                else:
                    raise ValueError(
                        f"Format SHAP inattendu : {raw_shap.shape}"
                    )
            elif raw_shap.ndim == 2:
                attack_shap = raw_shap
            else:
                raise ValueError(
                    f"Format SHAP inattendu : {raw_shap.shape}"
                )

        # Explication globale : moyenne des contributions absolues.
        importance = pd.DataFrame({
            "Feature": features,
            "Importance SHAP": np.abs(attack_shap).mean(axis=0),
        }).sort_values("Importance SHAP", ascending=False)

    st.subheader("1. Vue globale — Features les plus importantes")
    st.caption(
        "Calculée sur un échantillon de "
        f"{sample_size} flows. L'importance mesure l'amplitude moyenne "
        "de la contribution, pas sa direction."
    )

    st.bar_chart(
        importance.head(15).set_index("Feature")["Importance SHAP"],
        horizontal=True,
    )

    st.subheader("2. Explication locale — Un flow spécifique")

    row_idx = st.number_input(
        "Index du flow à expliquer",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1,
    )
    row_idx = int(row_idx)

    # Recalculer SHAP uniquement pour le flow sélectionné.
    X_one = X.iloc[[row_idx]]
    raw_one = explainer.shap_values(X_one)

    if isinstance(raw_one, list):
        one_values = np.asarray(raw_one[attack_index])[0]
    else:
        raw_one = np.asarray(raw_one)
        if raw_one.ndim == 3:
            if raw_one.shape[0] == 1:
                one_values = raw_one[0, :, attack_index]
            elif raw_one.shape[0] == len(classes):
                one_values = raw_one[attack_index, 0, :]
            else:
                raise ValueError(
                    f"Format SHAP local inattendu : {raw_one.shape}"
                )
        else:
            one_values = raw_one[0]

    expected = np.asarray(explainer.expected_value)
    base_value = (
        float(expected[attack_index])
        if expected.ndim > 0
        else float(expected)
    )

    explanation = shap.Explanation(
        values=one_values,
        base_values=base_value,
        data=X_one.iloc[0].to_numpy(),
        feature_names=features,
    )

    probability = float(probabilities[row_idx])
    st.metric(
        "Probabilité d'attaque",
        f"{probability:.2%}",
        delta="Attack" if predicted[row_idx] == 1 else "Benign",
        delta_color="inverse",
    )

    st.caption(
        "Le graphique explique la sortie du modèle pour la classe Attack. "
        "Une contribution SHAP positive pousse la sortie vers Attack ; "
        "une contribution négative la pousse dans le sens opposé."
    )

    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False,
    )
    st.pyplot(plt.gcf())
    plt.close("all")

except Exception as exc:
    st.warning(
        "Les explications SHAP n'ont pas pu être générées. "
        "La détection principale reste disponible."
    )
    st.exception(exc)

if "Label" in data.columns:
    st.divider()
    st.header("Évaluation sur les labels fournis")

    true_labels = (
        data["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["ddos", "dos", "attack", "1", "true"])
        .astype(int)
        .to_numpy()
    )

    y_pred = predicted
    tn, fp, fn, tp = confusion_matrix(
        true_labels, y_pred, labels=[0, 1]
    ).ravel()

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{accuracy_score(true_labels, y_pred):.4f}")
    b.metric(
        "Recall Attack",
        f"{recall_score(true_labels, y_pred, zero_division=0):.4f}",
    )
    c.metric(
        "Precision Attack",
        f"{precision_score(true_labels, y_pred, zero_division=0):.4f}",
    )
    d.metric("False Positives", f"{fp:,}")

    st.write("Confusion matrix — lignes : vérité, colonnes : prédiction")
    st.dataframe(
        pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Benign réel", "Attack réel"],
            columns=["Prédit Benign", "Prédit Attack"],
        ),
        use_container_width=True,
    )

st.divider()
st.caption(
    "AI-IDS — Prototype de détection. Les prédictions sont des estimations "
    "du modèle et nécessitent une validation avant tout usage opérationnel."
)
