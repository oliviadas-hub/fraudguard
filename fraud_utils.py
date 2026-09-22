"""Helper functions: explain a single prediction without any extra libraries."""
import pandas as pd
from pandas.api.types import is_numeric_dtype

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "hour": "Time of day",
    "category": "Merchant category",
    "distance_from_home_km": "Distance from home",
    "txns_last_24h": "Number of recent transactions",
    "is_foreign": "Foreign transaction",
    "card_present": "Card present",
}


def make_reference(df: pd.DataFrame) -> dict:
    """'Typical legit transaction' values used as the neutral baseline."""
    legit = df[df["is_fraud"] == 0].drop(columns="is_fraud")
    ref = {}
    for col in legit.columns:
        ref[col] = legit[col].median() if is_numeric_dtype(legit[col]) else legit[col].mode()[0]
    return ref


def explain_transaction(model, row: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Simple what-if explanation (a lightweight cousin of SHAP).

    For every feature we measure two things and average them:
      1. Remove it: swap the feature for a typical legit value and see how much
         the fraud probability DROPS.
      2. Add it alone: start from a typical legit transaction, insert only this
         feature's real value and see how much the probability RISES.
    Positive impact = the feature pushes the transaction toward "fraud".
    """
    base = float(model.predict_proba(row)[0, 1])
    neutral_all = row.copy()
    for col in row.columns:
        neutral_all[col] = reference[col]
    p_neutral = float(model.predict_proba(neutral_all)[0, 1])

    rows = []
    for col in row.columns:
        removed = row.copy()
        removed[col] = reference[col]
        drop = base - float(model.predict_proba(removed)[0, 1])

        added = neutral_all.copy()
        added[col] = row[col].iloc[0]
        rise = float(model.predict_proba(added)[0, 1]) - p_neutral

        rows.append({"feature": FEATURE_LABELS.get(col, col), "impact": (drop + rise) / 2})
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)
