"""FraudGuard: Streamlit app for fraud scoring, explanation and model insights."""
import joblib
import pandas as pd
import streamlit as st

from fraud_utils import explain_transaction

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

FEATURES = ["amount", "hour", "category", "distance_from_home_km",
            "txns_last_24h", "is_foreign", "card_present"]
CATEGORIES = ["grocery", "fuel", "restaurant", "electronics",
              "travel", "online_retail", "jewelry", "atm"]


@st.cache_resource
def load():
    return joblib.load("model.joblib")


bundle = load()
model, default_thr = bundle["model"], bundle["threshold"]
metrics, impact, reference = bundle["metrics"], bundle["impact"], bundle["reference"]

st.title("🛡️ FraudGuard")
st.caption(f"Real-time transaction fraud detection · Model: {bundle['model_name']}")

with st.sidebar:
    st.header("Model performance")
    st.metric("ROC-AUC", metrics["ROC-AUC"])
    st.metric("PR-AUC", metrics["PR-AUC"])
    st.metric("Recall (fraud caught)", f"{metrics['Recall']:.0%}")
    st.metric("Precision", f"{metrics['Precision']:.0%}")
    threshold = st.slider("Decision threshold", 0.05, 0.99, float(default_thr), 0.01,
                          help="Lower = catch more fraud but more false alarms")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Check a transaction", "📁 Batch scoring", "📊 Model insights", "💰 Business impact"]
)

# ---------------- Tab 1: single transaction + explanation ----------------
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount ($)", 0.0, 50000.0, 120.0)
    hour = c2.slider("Hour of day", 0, 23, 14)
    category = c1.selectbox("Merchant category", CATEGORIES)
    distance = c2.number_input("Distance from home (km)", 0.0, 20000.0, 5.0)
    txns = c1.number_input("Transactions in last 24h", 0, 50, 1)
    foreign = c2.checkbox("Foreign transaction")
    card_present = c2.checkbox("Card present", value=True)

    if st.button("Check transaction", type="primary"):
        row = pd.DataFrame([{
            "amount": amount, "hour": hour, "category": category,
            "distance_from_home_km": distance, "txns_last_24h": txns,
            "is_foreign": int(foreign), "card_present": int(card_present),
        }])
        p = float(model.predict_proba(row)[0, 1])
        st.progress(min(p, 1.0), text=f"Fraud probability: {p:.1%}")
        if p >= threshold:
            st.error("🚨 High risk: block or send for manual review")
        elif p >= threshold / 2:
            st.warning("⚠️ Medium risk: request extra verification (OTP)")
        else:
            st.success("✅ Low risk: approve")

        st.subheader("Why this score?")
        expl = explain_transaction(model, row, reference)
        flags = expl[expl["impact"] > 0.02]
        if flags.empty:
            st.write("No single factor stands out as suspicious.")
        else:
            st.write("Factors pushing this transaction toward fraud:")
            st.bar_chart(flags.set_index("feature")["impact"])

# ---------------- Tab 2: batch scoring ----------------
with tab2:
    st.write("Upload a CSV of transactions to score them all at once. "
             f"Required columns: `{', '.join(FEATURES)}`")
    try:
        with open("sample_batch.csv", "rb") as f:
            st.download_button("Download a sample file to try", f, "sample_batch.csv", "text/csv")
    except FileNotFoundError:
        pass
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            scored = data.copy()
            scored["fraud_probability"] = model.predict_proba(data[FEATURES])[:, 1].round(4)
            scored["flagged"] = scored["fraud_probability"] >= threshold
            a, b, c = st.columns(3)
            a.metric("Transactions", f"{len(scored):,}")
            b.metric("Flagged as fraud", f"{int(scored['flagged'].sum()):,}")
            c.metric("Flag rate", f"{scored['flagged'].mean():.1%}")
            st.dataframe(scored.sort_values("fraud_probability", ascending=False),
                         use_container_width=True)
            st.download_button("Download results", scored.to_csv(index=False),
                               "scored_transactions.csv", "text/csv")

# ---------------- Tab 3: model insights ----------------
with tab3:
    st.subheader("Model comparison")
    st.caption("Three models trained on the same data, ranked by PR-AUC (best metric for rare fraud).")
    st.dataframe(pd.read_csv("model_comparison.csv"), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("What drives predictions")
        imp = pd.read_csv("feature_importance.csv").set_index("feature")
        st.bar_chart(imp["importance"])
    with right:
        st.subheader("Precision-recall curve")
        st.line_chart(pd.read_csv("pr_curve.csv").set_index("recall")["precision"])

# ---------------- Tab 4: business impact ----------------
with tab4:
    st.subheader("Simulated business impact (test set)")
    st.caption("Assumes a missed fraud costs the full transaction amount and each "
               "manual review costs $5. Based on synthetic data.")
    a, b, c = st.columns(3)
    a.metric("Fraud value caught", f"${impact['fraud_caught_value']:,.0f}",
             f"{impact['pct_of_fraud_value_caught']}% of all fraud")
    b.metric("Fraud value missed", f"${impact['fraud_missed_value']:,.0f}")
    c.metric("Net saving after review costs", f"${impact['net_saving']:,.0f}")
