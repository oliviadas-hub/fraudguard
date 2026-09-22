# 🛡️ FraudGuard: Real-Time Transaction Fraud Detection

An end-to-end machine learning project: it trains and compares fraud models, explains each prediction, scores transactions in bulk, and estimates business impact, all in a live web app.

**Live demo:** _add your Streamlit link here_

## Problem
Fraud is ~1% of transactions, so accuracy is misleading (a model that always says "legit" is 99% accurate). This project handles the extreme class imbalance and tunes the decision threshold to balance caught fraud against false alarms.

## Features
- **Model comparison:** Logistic Regression vs Random Forest vs Gradient Boosting, ranked by PR-AUC
- **3D fraud explorer:** rotate and zoom 3,000 transactions in 3D (distance x hour x amount), coloured by fraud probability, with your own transaction shown as a gold marker
- **Live risk gauge:** speedometer-style gauge with green / yellow / red zones
- **Explainable predictions:** each transaction shows which factors pushed it toward fraud (what-if analysis)
- **Batch scoring:** upload a CSV, score every row, download the results
- **Model insights:** feature importance and precision-recall curve
- **Business impact:** simulated fraud value caught vs. manual-review cost
- **Adjustable threshold:** trade off recall against false alarms live

## Approach
- **Data:** 60,000 transactions (synthetic, ~1.2% fraud) with amount, hour, merchant category, distance from home, 24h velocity, foreign and card-present flags
- **Imbalance:** balanced sample weights + stratified train/test split
- **Evaluation:** ROC-AUC, PR-AUC, precision, recall, F1-optimized threshold
- **Interpretability:** permutation feature importance (global) + what-if explanations (per transaction)

## Run it
```bash
pip install -r requirements.txt
python generate_data.py   # creates transactions.csv (or use a Kaggle dataset)
python train.py           # trains 3 models, saves the best one + reports
streamlit run app.py
```

## Project structure
| File | Purpose |
|---|---|
| `generate_data.py` | Creates the synthetic dataset |
| `train.py` | Trains and compares models, saves model and reports |
| `fraud_utils.py` | Per-transaction explanation logic |
| `app.py` | Streamlit web app (5 tabs, dark colourful theme) |
| `charts.py` | Interactive Plotly charts (3D scatter, gauge, donut, bars) |
| `.streamlit/config.toml` | Dark theme colours |
| `sample_batch.csv` | Sample file for the batch-scoring tab |

## Deploy free
1. Push this folder to a public GitHub repo (include `model.joblib` and the `.csv` files)
2. Go to share.streamlit.io, connect the repo, select `app.py`, and deploy
3. Paste the live link at the top of this README and on your resume

## Ideas to extend
- Replace the synthetic data with the Kaggle Credit Card Fraud dataset
- Add SHAP explanations, a FastAPI endpoint, and a Dockerfile
