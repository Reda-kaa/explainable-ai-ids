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

## Dashboard Preview

![AI-IDS Detection Dashboard](assets/dashboard.png)

## Explainable AI with SHAP

![SHAP Local Explanation](assets/shap-explanation.png)

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
