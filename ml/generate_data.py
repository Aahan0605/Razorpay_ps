"""Synthetic payment-transaction generator with realistic fraud.

Emits raw transaction events only -- no aggregates. Velocity/behavioural
features are derived causally in features.py so training and inference share
one code path.

Design goals (these are what make the eval honest):
  - fraud and legit distributions OVERLAP; no feature separates them cleanly
  - legit-but-suspicious traffic exists (travel, payday bursts, new phones)
    so false positives are real, not decorative
  - a slice of fraud is deliberately unlearnable -> recall has a real ceiling

Usage: python ml/generate_data.py
"""

import numpy as np
import pandas as pd

SEED = 7
N_CUSTOMERS = 8_000
DAYS = 180
FRAUD_RATE = 0.007  # ~0.7%, realistic for payments

START = pd.Timestamp("2025-01-01")
COUNTRIES = ["IN", "US", "AE", "SG", "GB", "NG", "RU"]
METHODS = ["upi", "card", "netbanking", "wallet"]
CATEGORIES = [
    "groceries", "food_delivery", "electronics", "travel", "gaming",
    "utilities", "fashion", "crypto", "gift_cards", "subscription",
]
# typical spend per category (lognormal mu in INR-ish log space)
CAT_MU = {
    "groceries": 6.4, "food_delivery": 5.9, "electronics": 9.2, "travel": 9.6,
    "gaming": 6.2, "utilities": 7.3, "fashion": 7.6, "crypto": 9.4,
    "gift_cards": 8.1, "subscription": 5.7,
}


def _customers(rng):
    """Stable per-customer profiles. Fraud later deviates from these."""
    home = rng.choice(COUNTRIES, N_CUSTOMERS, p=[0.86, 0.04, 0.03, 0.03, 0.02, 0.01, 0.01])
    return pd.DataFrame({
        "customer_id": [f"cust_{i:05d}" for i in range(N_CUSTOMERS)],
        "home_country": home,
        "spend_scale": rng.normal(0, 0.55, N_CUSTOMERS),      # shifts their lognormal
        "rate": rng.gamma(2.0, 4.0, N_CUSTOMERS) + 1,          # txns over the window
        "device_id": [f"dev_{i:05d}" for i in range(N_CUSTOMERS)],
        # account age: mix of long-tenured and recent signups
        "signup_offset": np.where(
            rng.random(N_CUSTOMERS) < 0.75,
            -rng.uniform(30, 1500, N_CUSTOMERS),
            rng.uniform(0, DAYS, N_CUSTOMERS),
        ),
        "email_domain": rng.choice(
            ["free", "corporate", "disposable"], N_CUSTOMERS, p=[0.78, 0.20, 0.02]
        ),
    })


def _amount(rng, cats, scales):
    mu = np.array([CAT_MU[c] for c in cats]) + scales
    return np.round(np.exp(rng.normal(mu, 0.55)), 2)


def _legit(rng, cust):
    """Baseline traffic: customers behaving like themselves."""
    n_per = rng.poisson(cust["rate"].to_numpy())
    idx = np.repeat(np.arange(N_CUSTOMERS), n_per)
    n = len(idx)
    c = cust.iloc[idx].reset_index(drop=True)

    # hour-of-day: bimodal (lunch + evening), wrapped into [0,24)
    hour = np.where(
        rng.random(n) < 0.35, rng.normal(13, 2.0, n), rng.normal(20, 2.5, n)
    ) % 24
    ts = START + pd.to_timedelta(rng.uniform(0, DAYS, n), "D").floor("D") \
        + pd.to_timedelta(hour, "h")

    cats = rng.choice(CATEGORIES, n, p=[.20, .19, .07, .05, .07, .12, .13, .02, .03, .12])
    df = pd.DataFrame({
        "timestamp": ts,
        "customer_id": c["customer_id"],
        "amount": _amount(rng, cats, c["spend_scale"].to_numpy()),
        "payment_method": rng.choice(METHODS, n, p=[0.52, 0.28, 0.12, 0.08]),
        "merchant_category": cats,
        "merchant_id": [f"mch_{i:04d}" for i in rng.integers(0, 2500, n)],
        "device_id": c["device_id"],
        "ip_country": c["home_country"],
        "billing_country": c["home_country"],
        "email_domain": c["email_domain"],
        "signup_offset": c["signup_offset"],
        "failed_attempts": rng.binomial(2, 0.04, n),
        "is_fraud": 0,
    })

    # --- legit-but-suspicious: these SHOULD trip a naive rule engine ---
    # 1. travellers: abroad IP, own card, elevated spend
    trav = rng.random(n) < 0.02
    df.loc[trav, "ip_country"] = rng.choice(["US", "AE", "SG", "GB"], trav.sum())
    df.loc[trav, "amount"] *= rng.uniform(1.5, 4.0, trav.sum())
    # 2. phone upgrade: brand-new device, otherwise normal
    upg = rng.random(n) < 0.015
    df.loc[upg, "device_id"] = [f"dev_new_{i}" for i in rng.integers(0, 99999, upg.sum())]
    # 3. genuine big-ticket purchases
    big = rng.random(n) < 0.01
    df.loc[big, "amount"] *= rng.uniform(4, 12, big.sum())
    return df


def _fraud(rng, cust, n_target):
    """Four archetypes. Amounts/timing overlap legit traffic on purpose."""
    rows = []
    victims = cust.sample(n_target, random_state=SEED, replace=True).reset_index(drop=True)
    # archetype mix: card testing bursts are the most common in payments
    kind = rng.choice(["card_testing", "ato", "stolen_card", "slow_drain"],
                      n_target, p=[0.34, 0.28, 0.26, 0.12])

    for i, v in victims.iterrows():
        k = kind[i]
        t0 = START + pd.Timedelta(days=float(rng.uniform(0, DAYS)))
        base = dict(customer_id=v.customer_id, billing_country=v.home_country,
                    email_domain=v.email_domain, signup_offset=v.signup_offset,
                    is_fraud=1)

        if k == "card_testing":
            # rapid low-value probes across many merchants, new device, declines
            for j in range(rng.integers(3, 9)):
                rows.append({**base,
                    "timestamp": t0 + pd.Timedelta(minutes=float(j * rng.uniform(0.5, 4))),
                    "amount": round(float(rng.uniform(5, 120)), 2),
                    "payment_method": "card",
                    "merchant_category": rng.choice(["gaming", "subscription", "gift_cards"]),
                    "merchant_id": f"mch_{rng.integers(0, 2500):04d}",
                    "device_id": f"dev_bot_{rng.integers(0, 999)}",
                    "ip_country": rng.choice(["RU", "NG", "US"]),
                    "failed_attempts": int(rng.integers(1, 5))})

        elif k == "ato":
            # known customer, new device + new geo, drains upward
            dev = f"dev_ato_{rng.integers(0, 9999)}"
            ipc = rng.choice(["RU", "NG", "AE", "US"])
            for j in range(rng.integers(2, 5)):
                rows.append({**base,
                    "timestamp": t0 + pd.Timedelta(hours=float(j * rng.uniform(0.2, 2))),
                    "amount": round(float(np.exp(rng.normal(8.6 + 0.4 * j, 0.5))), 2),
                    "payment_method": rng.choice(["card", "netbanking", "wallet"]),
                    "merchant_category": rng.choice(["electronics", "crypto", "gift_cards", "travel"]),
                    "merchant_id": f"mch_{rng.integers(0, 2500):04d}",
                    "device_id": dev, "ip_country": ipc,
                    "failed_attempts": int(rng.integers(0, 3))})

        else:  # stolen_card and slow_drain
            hard = k == "slow_drain"
            for j in range(rng.integers(1, 4)):
                rows.append({**base,
                    "timestamp": t0 + pd.Timedelta(hours=float(j * rng.uniform(1, 20))),
                    # slow_drain sits inside the legit amount band -> genuinely hard
                    "amount": round(float(np.exp(rng.normal(6.6 if hard else 9.1, 0.6))), 2),
                    "payment_method": rng.choice(METHODS),
                    "merchant_category": rng.choice(CATEGORIES),
                    "merchant_id": f"mch_{rng.integers(0, 2500):04d}",
                    "device_id": v.device_id if hard else f"dev_stl_{rng.integers(0, 9999)}",
                    # slow_drain keeps the victim's own geo: no easy tell
                    "ip_country": v.home_country if hard else rng.choice(["US", "RU", "NG", "GB"]),
                    "failed_attempts": int(rng.integers(0, 2))})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    cust = _customers(rng)

    legit = _legit(rng, cust)
    n_fraud_txn = int(len(legit) * FRAUD_RATE / (1 - FRAUD_RATE))
    fraud = _fraud(rng, cust, max(1, n_fraud_txn // 3))  # each episode -> several txns

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # account age at txn time; clip so pre-signup drift can't go negative
    df["customer_age_days"] = (
        (df["timestamp"] - START).dt.total_seconds() / 86400 - df["signup_offset"]
    ).clip(lower=0).round(2)
    df = df.drop(columns=["signup_offset"])
    df.insert(0, "txn_id", [f"txn_{i:07d}" for i in range(len(df))])

    # label noise: 8% of fraud is scrubbed of every tell. Unlearnable by design,
    # so reported recall reflects a real ceiling rather than a leaked one.
    f_idx = df.index[df.is_fraud == 1]
    for i in rng.choice(f_idx, int(0.08 * len(f_idx)), replace=False):
        row = df.loc[i]
        home = cust.loc[cust.customer_id == row.customer_id, "home_country"]
        df.loc[i, ["ip_country", "device_id", "failed_attempts"]] = [
            home.iloc[0] if len(home) else "IN",
            f"dev_{rng.integers(0, N_CUSTOMERS):05d}", 0]
        df.loc[i, "amount"] = round(float(np.exp(rng.normal(6.5, 0.5))), 2)

    import pathlib
    out = pathlib.Path(__file__).parent.parent / "data"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "transactions.csv", index=False)

    print(f"{len(df):,} transactions  |  {df.is_fraud.mean():.2%} fraud "
          f"({df.is_fraud.sum():,})  |  {df.timestamp.min().date()} -> {df.timestamp.max().date()}")
    return df


if __name__ == "__main__":
    main()
