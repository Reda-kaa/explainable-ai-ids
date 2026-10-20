from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

root = Path.cwd()
pred_path = root / "reports" / "ddos_flow_duration_predictions.csv"
out_path = root / "reports" / "ddos_threshold_comparison.csv"

df = pd.read_csv(pred_path)

y_true = (df["Label"].astype(str) == "DDoS").astype(int).to_numpy()
prob = df["Probability"].astype(float).to_numpy()

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01]
results = []

for threshold in thresholds:
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "TP": int(tp),
        "FN": int(fn),
        "TN": int(tn),
        "FP": int(fp),
        "Recall_DDoS": tp / (tp + fn) if tp + fn else 0,
        "False_Positive_Rate": fp / (fp + tn) if fp + tn else 0,
        "Precision_Attack": tp / (tp + fp) if tp + fp else 0,
    })

result = pd.DataFrame(results)
result.to_csv(out_path, index=False)

print("\n=== COMPARAISON DES THRESHOLDS ===")
print(result.to_string(
    index=False,
    formatters={
        "Recall_DDoS": "{:.4f}".format,
        "False_Positive_Rate": "{:.4f}".format,
        "Precision_Attack": "{:.4f}".format,
    }
))
print("\nRapport sauvegarde :", out_path)
