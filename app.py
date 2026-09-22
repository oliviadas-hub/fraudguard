"""FraudGuard: Streamlit app for fraud scoring, explanation, 3D exploration and insights."""
import joblib
import pandas as pd
import streamlit as st

import charts
from fraud_utils import explain_transaction

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

FEATURES = ["amount", "hour", "category", "distance_from_home_km",
            "txns_last_24h", "is_foreign", "card_present"]
CATEGORIES = ["grocery", "fuel", "restaurant", "electronics",
              "travel", "online_retail", "jewelry", "atm"]

# ---- Colourful look: gradient title + glowing metric cards ----
st.markdown("""
<style>
.hero {font-size: 3rem; font-weight: 800; line-height: 1.1;
       background: linear-gradient(90deg, #00E5A8, #4DA3FF, #B388FF, #FF80AB);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
[data-testid="stMetric"] {background: linear-gradient(135deg, #151B33, #1E2747);
       border: 1px solid #2B3A6B; border-radius: 14px; padding: 14px;}
button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    return joblib.load("model.joblib")


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


bundle = load()
model, default_thr = bundle["model"], bundle["threshold"]
metrics, impact, reference = bundle["metrics"], bundle["impact"], bundle["reference"]
viz_sample = load_csv("viz_sample.csv")

st.markdown('<div class="hero">🛡️ FraudGuard</div>', unsafe_allow_html=True)
st.caption(f"Real-time transaction fraud detection · Model: {bundle['model_name']}")

with st.sidebar:
    st.header("Model performance")
    st.metric("ROC-AUC", metrics["ROC-AUC"])
    st.metric("PR-AUC", metrics["PR-AUC"])
    st.metric("Recall (fraud caught)", f"{metrics['Recall']:.0%}")
    st.metric("Precision", f"{metrics['Precision']:.0%}")
    threshold = st.slider("Decision threshold", 0.05, 0.99, float(default_thr), 0.01,
                          help="Lower = catch more fraud but more false alarms")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Check a transaction", "🌐 3D Explorer", "📁 Batch scoring",
     "📊 Model insights", "💰 Business impact"]
)

# ---------------- Tab 1: single transaction, gauge, explanation, 3D position ----------------
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
        st.session_state["last_row"] = row
        p = float(model.predict_proba(row)[0, 1])

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(charts.risk_gauge(p, threshold), key="gauge_tab1")
            if p >= threshold:
                st.error("🚨 High risk: block or send for manual review")
            elif p >= threshold / 2:
                st.warning("⚠️ Medium risk: request extra verification (OTP)")
            else:
                st.success("✅ Low risk: approve")
        with right:
            expl = explain_transaction(model, row, reference)
            st.plotly_chart(charts.explanation_bars(expl), key="explain_tab1")

        st.subheader("Where your transaction sits among 3,000 others")
        st.caption("Gold diamond = your transaction. Drag to rotate, scroll to zoom.")
        st.plotly_chart(charts.scatter_3d(viz_sample, highlight=row), key="scatter3d_tab1")

# ---------------- Tab 2: 3D explorer ----------------
with tab2:
    st.subheader("3D fraud explorer")
    st.caption("3,000 test transactions (all real fraud cases kept, so fraud looks more common "
               "here than the true ~1%). Fraudulent behaviour clusters far from home, at odd "
               "hours, with large amounts.")
    a, b = st.columns([1, 2])
    color_by = a.radio("Colour points by", ["Fraud probability", "Actual label"])
    cats = b.multiselect("Merchant categories", CATEGORIES, default=CATEGORIES)
    shown = viz_sample[viz_sample["category"].isin(cats)]
    hl = st.session_state.get("last_row")
    st.plotly_chart(charts.scatter_3d(shown, highlight=hl, color_by=color_by), key="scatter3d_tab2")
    if hl is not None:
        st.caption("Your last checked transaction is shown as a gold diamond.")

# ---------------- Tab 3: batch scoring ----------------
with tab3:
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
            st.dataframe(scored.sort_values("fraud_probability", ascending=False))
            st.download_button("Download results", scored.to_csv(index=False),
                               "scored_transactions.csv", "text/csv")

# ---------------- Tab 4: model insights ----------------
with tab4:
    comp = load_csv("model_comparison.csv")
    st.plotly_chart(charts.model_comparison_bars(comp), key="model_comp_tab4")
    st.dataframe(comp, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.importance_bars(load_csv("feature_importance.csv")), key="importance_tab4")
    with right:
        st.plotly_chart(charts.pr_curve(load_csv("pr_curve.csv")), key="prcurve_tab4")

# ---------------- Tab 5: business impact ----------------
with tab5:
    st.subheader("Simulated business impact (test set)")
    st.caption("Assumes a missed fraud costs the full transaction amount and each "
               "manual review costs $5. Based on synthetic data.")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Fraud value caught", f"${impact['fraud_caught_value']:,.0f}",
                  f"{impact['pct_of_fraud_value_caught']}% of all fraud")
        st.metric("Fraud value missed", f"${impact['fraud_missed_value']:,.0f}")
        st.metric("Net saving after review costs", f"${impact['net_saving']:,.0f}")
    with right:
        st.plotly_chart(charts.impact_donut(impact), key="donut_tab5")