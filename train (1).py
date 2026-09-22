def run_training():
    """Train several models, compare them, pick the best, and save everything the app needs."""
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (average_precision_score, confusion_matrix,
                                 precision_recall_curve, roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight

    from fraud_utils import make_reference

    # ---------- Business assumptions (simulated, change as needed) ----------
    REVIEW_COST = 5.0       # cost of manually reviewing one flagged transaction ($)
    # Each missed fraud costs its full transaction amount.

    df = pd.read_csv("transactions.csv")
    X, y = df.drop(columns="is_fraud"), df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    weights = compute_sample_weight("balanced", y_train)

    CAT, NUM = ["category"], [c for c in X.columns if c != "category"]


    def build(clf, scale=False):
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler() if scale else "passthrough", NUM),
        ])
        return Pipeline([("pre", pre), ("clf", clf)])


    candidates = {
        "Logistic Regression": build(LogisticRegression(max_iter=1000), scale=True),
        "Random Forest": build(RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)),
        "Gradient Boosting": build(HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=42)),
    }


    def best_threshold(y_true, proba):
        prec, rec, thr = precision_recall_curve(y_true, proba)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        return float(thr[f1[:-1].argmax()])


    results, fitted, probas = [], {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train, clf__sample_weight=weights)
        proba = model.predict_proba(X_test)[:, 1]
        thr = best_threshold(y_test, proba)
        pred = (proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        results.append({
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, proba), 4),
            "PR-AUC": round(average_precision_score(y_test, proba), 4),
            "Precision": round(tp / max(tp + fp, 1), 3),
            "Recall": round(tp / max(tp + fn, 1), 3),
            "Threshold": round(thr, 3),
        })
        fitted[name], probas[name] = model, proba
        print(f"{name:20s} PR-AUC={results[-1]['PR-AUC']}  ROC-AUC={results[-1]['ROC-AUC']}")

    comparison = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    comparison.to_csv("model_comparison.csv", index=False)

    # ---------- Pick the best model by PR-AUC ----------
    best_name = comparison.iloc[0]["Model"]
    model, proba = fitted[best_name], probas[best_name]
    threshold = best_threshold(y_test, proba)
    pred = (proba >= threshold).astype(int)
    print(f"\nBest model: {best_name} (threshold {threshold:.3f})")
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    # ---------- Simulated business impact on the test set ----------
    amounts = X_test["amount"].to_numpy()
    is_f = y_test.to_numpy() == 1
    caught = float(amounts[(pred == 1) & is_f].sum())
    missed = float(amounts[(pred == 0) & is_f].sum())
    review_cost = float(((pred == 1).sum()) * REVIEW_COST)
    total_fraud = caught + missed
    net_saving = caught - review_cost
    impact = {
        "total_fraud_value": round(total_fraud, 2),
        "fraud_caught_value": round(caught, 2),
        "fraud_missed_value": round(missed, 2),
        "review_cost": round(review_cost, 2),
        "net_saving": round(net_saving, 2),
        "pct_of_fraud_value_caught": round(100 * caught / total_fraud, 1),
    }
    print("Business impact (simulated):", json.dumps(impact, indent=2))

    # ---------- Feature importance (permutation) ----------
    imp = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                 n_repeats=3, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean}
                              ).sort_values("importance", ascending=False)
    importance.to_csv("feature_importance.csv", index=False)

    # ---------- Precision-recall curve data ----------
    prec, rec, _ = precision_recall_curve(y_test, proba)
    step = max(len(prec) // 300, 1)
    pd.DataFrame({"recall": rec[::step], "precision": prec[::step]}).to_csv("pr_curve.csv", index=False)

    # ---------- Sample of scored transactions for the 3D explorer ----------
    viz = X_test.copy()
    viz["fraud_probability"] = proba.round(4)
    viz["actual_fraud"] = y_test.values
    frauds = viz[viz["actual_fraud"] == 1]                       # keep every fraud so it is visible
    legit = viz[viz["actual_fraud"] == 0].sample(2850, random_state=42)
    pd.concat([frauds, legit]).sample(frac=1, random_state=1).to_csv("viz_sample.csv", index=False)

    # ---------- Save everything the app needs ----------
    metrics = comparison.iloc[0].to_dict()
    joblib.dump({
        "model": model, "threshold": threshold, "model_name": best_name,
        "metrics": metrics, "impact": impact, "reference": make_reference(df),
    }, "model.joblib")

    # Small sample file for testing the app's batch upload feature
    X_test.head(300).assign(actual_fraud=y_test.head(300).values).to_csv("sample_batch.csv", index=False)
    print("\nSaved model.joblib, model_comparison.csv, feature_importance.csv, pr_curve.csv, viz_sample.csv, sample_batch.csv")


if __name__ == '__main__':
    run_training()
