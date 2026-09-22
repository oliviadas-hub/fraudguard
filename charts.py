"""All Plotly charts for FraudGuard (dark theme, colourful, interactive)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

GREEN, YELLOW, RED = "#00E5A8", "#FFC107", "#FF4B6E"
BLUE, PURPLE, PINK = "#4DA3FF", "#B388FF", "#FF80AB"
TRANSPARENT = "rgba(0,0,0,0)"


def _style(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def risk_gauge(prob: float, threshold: float):
    """Speedometer-style fraud risk gauge (green / yellow / red zones)."""
    low, high = threshold / 2 * 100, threshold * 100
    color = GREEN if prob * 100 < low else (YELLOW if prob * 100 < high else RED)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 46}},
        title={"text": "Fraud risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, low], "color": "rgba(0,229,168,0.25)"},
                {"range": [low, high], "color": "rgba(255,193,7,0.25)"},
                {"range": [high, 100], "color": "rgba(255,75,110,0.25)"},
            ],
            "threshold": {"line": {"color": "white", "width": 4},
                          "thickness": 0.85, "value": high},
        },
    ))
    return _style(fig, 300)


def explanation_bars(expl: pd.DataFrame):
    """Red bars push toward fraud, green bars push toward legit."""
    d = expl[expl["impact"].abs() > 0.005].sort_values("impact")
    fig = go.Figure(go.Bar(
        x=d["impact"], y=d["feature"], orientation="h",
        marker_color=[RED if v > 0 else GREEN for v in d["impact"]],
    ))
    fig.update_layout(title="Why this score? (red = pushes toward fraud)")
    return _style(fig, 320)


def scatter_3d(sample: pd.DataFrame, highlight: pd.DataFrame = None,
               color_by: str = "Fraud probability"):
    """3D map of transactions: distance x hour x amount. Optional gold star = your transaction."""
    def logx(v):
        return np.log10(1 + np.asarray(v, dtype=float))

    hover = ("Amount: $" + sample["amount"].round(2).astype(str)
             + "<br>Distance: " + sample["distance_from_home_km"].round(1).astype(str) + " km"
             + "<br>Hour: " + sample["hour"].astype(str)
             + "<br>Category: " + sample["category"].astype(str)
             + "<br>Fraud prob: " + (sample["fraud_probability"] * 100).round(1).astype(str) + "%")

    if color_by == "Actual label":
        marker = dict(size=3.5, opacity=0.85,
                      color=sample["actual_fraud"].map({0: BLUE, 1: RED}).tolist())
    else:
        marker = dict(size=3.5, opacity=0.85, color=sample["fraud_probability"].tolist(),
                      colorscale="Turbo", cmin=0, cmax=1,
                      colorbar=dict(title=dict(text="Fraud prob"), len=0.7))

    fig = go.Figure(go.Scatter3d(
        x=logx(sample["distance_from_home_km"]), y=sample["hour"],
        z=logx(sample["amount"]), mode="markers", marker=marker,
        text=hover, hoverinfo="text", name="Transactions",
    ))
    if highlight is not None:
        fig.add_trace(go.Scatter3d(
            x=logx(highlight["distance_from_home_km"]), y=highlight["hour"],
            z=logx(highlight["amount"]), mode="markers", name="Your transaction",
            marker=dict(size=11, color="#FFD700", symbol="diamond",
                        line=dict(color="white", width=2)),
        ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=dict(text="Distance from home (log)")),
            yaxis=dict(title=dict(text="Hour of day")),
            zaxis=dict(title=dict(text="Amount (log)")),
            bgcolor=TRANSPARENT,
        ),
        legend=dict(orientation="h", y=1.02),
    )
    return _style(fig, 560)


def model_comparison_bars(comp: pd.DataFrame):
    fig = go.Figure()
    for metric, color in [("ROC-AUC", BLUE), ("PR-AUC", PURPLE), ("Precision", PINK), ("Recall", GREEN)]:
        fig.add_trace(go.Bar(name=metric, x=comp["Model"], y=comp[metric], marker_color=color))
    fig.update_layout(barmode="group", title="Model comparison",
                      yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.12))
    return _style(fig, 400)


def importance_bars(imp: pd.DataFrame):
    d = imp.sort_values("importance")
    fig = go.Figure(go.Bar(
        x=d["importance"], y=d["feature"], orientation="h",
        marker=dict(color=d["importance"], colorscale="Plasma"),
    ))
    fig.update_layout(title="What drives predictions")
    return _style(fig, 380)


def pr_curve(pr: pd.DataFrame):
    fig = go.Figure(go.Scatter(
        x=pr["recall"], y=pr["precision"], mode="lines", fill="tozeroy",
        line=dict(color=GREEN, width=3), fillcolor="rgba(0,229,168,0.18)",
    ))
    fig.update_layout(title="Precision-recall curve",
                      xaxis=dict(title=dict(text="Recall (fraud caught)")),
                      yaxis=dict(title=dict(text="Precision")))
    return _style(fig, 380)


def impact_donut(impact: dict):
    fig = go.Figure(go.Pie(
        labels=["Fraud caught", "Fraud missed"],
        values=[impact["fraud_caught_value"], impact["fraud_missed_value"]],
        hole=0.62, marker=dict(colors=[GREEN, RED]), textinfo="percent",
    ))
    fig.update_layout(title="Share of fraud value caught (simulated)")
    return _style(fig, 380)
