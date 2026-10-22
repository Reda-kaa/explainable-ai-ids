# AI-IDS — Explainable Network Intrusion Detection System

An explainable machine-learning prototype for network intrusion detection.

## Overview

AI-IDS uses a Random Forest classifier to analyse network-flow features and
identify potentially malicious traffic. Its Streamlit dashboard provides
predictions, probability scores, downloadable results and SHAP explanations.

## Main Features

- Network-flow analysis from CSV and Parquet files
- Input validation and numerical preprocessing
- Benign vs Attack predictions
- Configurable prediction threshold
- Detection statistics and probability distribution
- Downloadable prediction results in CSV format
- Global SHAP feature importance
- Local SHAP waterfall explanations for individual flows
- Optional evaluation when ground-truth labels are provided

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Random Forest
- SHAP
- Streamlit
- Matplotlib
- Joblib

## Project Structure

```text
ai-ids/
├── app.py
├── requirements.txt
├── .gitignore
├── models/
├── src/
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
└── reports/
```

## Installation

Python 3.11 is recommended.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Dashboard

From the project root:

```powershell
python -m streamlit run app.py
```

Upload a CSV or Parquet file containing the features expected by the
saved model. The application validates the required columns before prediction.

## SHAP Explainability

The dashboard provides:

- Global feature importance based on mean absolute SHAP values
- Local waterfall explanations for a selected flow

SHAP explanations describe how features contribute to model output.
They do not establish causality or guarantee that a detected flow is malicious.

## Model and Data

The dashboard expects the model bundle at:

`models/random_forest_ids.joblib`

The model must be available locally when running the application.
Datasets are excluded from version control by default.

Use only datasets you are authorised to analyse. Check the dataset licence
and remove sensitive information before publishing files or reports.

## Limitations

- Predictions depend on the quality and distribution of the training data.
- Performance may change on traffic from different environments.
- Results should be validated before any operational security use.
- The application is a research prototype, not a replacement for a
  production security monitoring system.

## Future Improvements

- Automated testing and continuous integration
- Improved dataset and label validation
- More detailed evaluation reports
- Deployment documentation and reproducible experiments

## License

No license has been specified yet. Add an appropriate license before
allowing others to reuse or redistribute this project.


---

## Dashboard Preview

The dashboard displays network traffic analysis, attack predictions, detection probabilities, and evaluation metrics.

![AI-IDS Detection Dashboard](assets/dashboard.png)

## Explainable AI with SHAP

SHAP provides global feature importance and local explanations to help understand why the model classifies a network flow as an attack.

![SHAP Explainability](assets/shap-explanation.png)


## Model Evaluation & Generalization

The project evaluates the Random Forest IDS using a general evaluation set and separate held-out traffic experiments. Results from different models and evaluation procedures are reported separately.

### General Evaluation - Original Model

| Metric | Score |
|---|---:|
| Accuracy | 99.27% |
| Attack F1-score | 98.91% |
| ROC-AUC | 99.96% |

Evaluation set: 12,115 samples.

These results describe the original model's evaluation and should not be interpreted as its expected performance on every unseen dataset.

### Held-Out Dataset Evaluation - Baseline Experiments

| Test dataset | Accuracy | Attack Recall | Attack F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| DDoS-Friday | 81.22% | 62.66% | 77.00% | 89.98% |
| DoS-Wednesday | 51.53% | 3.12% | Not reported | 97.19% |

These baseline experiments show that performance can vary substantially across traffic datasets. In particular, the original DoS-Wednesday experiment achieved very low attack recall at the default 0.50 decision threshold.

### DoS-Wednesday - Threshold-Tuned Experiment

A separate Random Forest model was trained using the non-holdout traffic files. Median imputation was fitted on the training data, and the decision threshold was selected on a validation split with a target false positive rate (FPR) of at most 5%.

The selected threshold was **0.10**, with a validation recall of **99.84%** and a validation FPR of **4.36%**. The DoS-Wednesday file was excluded from training and threshold selection for this experiment.

#### Final held-out test results

| Threshold | Accuracy | Attack Precision | Attack Recall | Attack F1-score | FPR |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 91.69% | 94.49% | 88.54% | 91.42% | 5.16% |
| 0.50 | 51.61% | 97.63% | 3.30% | 6.38% | 0.08% |

Test set: 10,000 samples, including 5,000 benign and 5,000 attack flows. ROC-AUC: **96.61%**.

At threshold 0.10, the model detected 4,427 of 5,000 attack flows, with 258 false positives. At threshold 0.50, it detected 165 attack flows, with 4 false positives.

The threshold-tuned results demonstrate the trade-off between attack detection and false alarms. The 5% FPR target was met on validation, but the test FPR was 5.16%, showing that the target does not necessarily generalize to unseen traffic.

**Methodology note:** The baseline and threshold-tuned DoS-Wednesday results come from separate model experiments and should not be treated as a direct comparison of threshold changes alone. The held-out file was excluded from training and threshold selection for this experiment, although it had been evaluated separately in an earlier baseline experiment.

### Explainability

SHAP is used to investigate global feature importance and individual predictions. Global explanations summarize the influence of features across a sample, while local waterfall plots show how features contribute to a specific prediction.

SHAP explanations describe model behavior; they do not establish causality or guarantee that a flow is malicious.
