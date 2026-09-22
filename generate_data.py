"""Generate a realistic synthetic transactions dataset (~1% fraud).

Swap this out for the Kaggle "Credit Card Fraud Detection" dataset or
IEEE-CIS Fraud dataset to make the project even stronger.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60_000
FRAUD_RATE = 0.012

is_fraud = rng.random(N) < FRAUD_RATE
# Hard cases: subtle fraud that behaves like normal spending, and odd-but-legit purchases
subtle = is_fraud & (rng.random(N) < 0.15)
odd_legit = (~is_fraud) & (rng.random(N) < 0.008)
looks_fraud = (is_fraud & ~subtle) | odd_legit
categories = ["grocery", "fuel", "restaurant", "electronics", "travel", "online_retail", "jewelry", "atm"]

df = pd.DataFrame({
    "amount": np.where(
        looks_fraud,
        rng.lognormal(mean=5.6, sigma=1.0, size=N),   # fraud: larger amounts
        rng.lognormal(mean=3.6, sigma=1.0, size=N),
    ).round(2),
    "hour": np.where(
        looks_fraud,
        rng.choice([0, 1, 2, 3, 4, 22, 23, 12, 15], size=N),  # fraud: odd hours
        rng.integers(7, 23, size=N),
    ),
    "category": np.where(
        looks_fraud,
        rng.choice(["electronics", "jewelry", "online_retail", "travel", "atm"], size=N),
        rng.choice(categories, size=N, p=[.28, .14, .2, .07, .05, .16, .01, .09]),
    ),
    "distance_from_home_km": np.where(
        looks_fraud, rng.exponential(180, N), rng.exponential(12, N)
    ).round(1),
    "txns_last_24h": np.where(
        looks_fraud, rng.poisson(6, N), rng.poisson(1.5, N)
    ),
    "is_foreign": np.where(looks_fraud, rng.random(N) < 0.4, rng.random(N) < 0.04).astype(int),
    "card_present": np.where(looks_fraud, rng.random(N) < 0.25, rng.random(N) < 0.72).astype(int),
    "is_fraud": is_fraud.astype(int),
})

df.to_csv("transactions.csv", index=False)
print(f"Saved {len(df):,} rows | fraud rate: {df.is_fraud.mean():.2%}")
