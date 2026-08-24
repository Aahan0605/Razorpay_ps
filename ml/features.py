"""Causal feature engineering. Shared verbatim by training and inference.

Every behavioural aggregate for row i is computed from rows strictly BEFORE i
(pandas rolling with closed="left"). A whole-dataset groupby would embed the
future into each row and inflate the eval -- this is the single most common
leak in fraud pipelines, so it gets one code path and no shortcuts.

At inference the backend appends the new transaction to that customer's recent
history and calls compute_features() on the same shape of frame, taking the
last row. Identical code, so train/serve skew cannot creep in.
"""

import numpy as np
import pandas as pd

# Fixed category vocabularies: the model's encoding must not depend on which
# values happen to appear in a given batch.
CATS = {
    "payment_method": ["upi", "card", "netbanking", "wallet"],
    "merchant_category": ["groceries", "food_delivery", "electronics", "travel",
                          "gaming", "utilities", "fashion", "crypto",
                          "gift_cards", "subscription"],
    "email_domain": ["free", "corporate", "disposable"],
    "ip_country": ["IN", "US", "AE", "SG", "GB", "NG", "RU"],
}

FEATURES = [
    "log_amount", "hour", "day_of_week", "is_night",
    "payment_method", "merchant_category", "email_domain", "ip_country",
    "ip_billing_mismatch", "customer_age_days", "is_new_account",
    "failed_attempts", "txns_1h", "txns_24h", "distinct_merchants_24h",
    "amount_vs_trailing_mean", "is_new_device", "devices_seen_prior",
    "mins_since_prev",
]
CAT_FEATURES = [f for f in FEATURES if f in CATS]

_NEVER_SEEN = 10_000.0  # minutes stand-in for "no prior transaction"


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw transaction events -> model-ready features. Input need not be sorted."""
    d = df.sort_values("timestamp").copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])

    # --- stateless ---
    d["log_amount"] = np.log1p(d["amount"])
    d["hour"] = d["timestamp"].dt.hour
    d["day_of_week"] = d["timestamp"].dt.dayofweek
    d["is_night"] = d["hour"].between(0, 5).astype(int)
    d["ip_billing_mismatch"] = (d["ip_country"] != d["billing_country"]).astype(int)
    d["is_new_account"] = (d["customer_age_days"] < 7).astype(int)

    # --- causal, per customer ---
    # NB: groupby(...).rolling(...) returns rows in GROUP order, not original
    # row order, so assigning its .to_numpy() back onto a time-sorted frame
    # silently scrambles features across customers. Each group is therefore
    # computed against its own row ids and reassembled by those ids.
    d = d.reset_index(drop=True)
    d["_i"] = np.arange(len(d))
    d["_mch"] = pd.factorize(d["merchant_id"])[0].astype(float)

    def _causal(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.set_index("timestamp")
        # closed="left" excludes the current row -> strictly prior history only
        prior = sub["amount"].rolling("24h", closed="left")
        trailing_mean = sub["amount"].shift().expanding().mean()
        return pd.DataFrame({
            "txns_1h": sub["amount"].rolling("1h", closed="left").count().to_numpy(),
            "txns_24h": prior.count().to_numpy(),
            "distinct_merchants_24h": sub["_mch"].rolling("24h", closed="left")
                .apply(lambda a: len(np.unique(a)), raw=True).to_numpy(),
            "amount_vs_trailing_mean": (
                sub["amount"] / trailing_mean.where(trailing_mean > 0)).to_numpy(),
        }, index=sub["_i"].to_numpy())

    agg = pd.concat([_causal(sub) for _, sub in d.groupby("customer_id", sort=False)])
    d = d.join(agg.sort_index())
    # first sighting of this (customer, device) pair
    d["is_new_device"] = (d.groupby(["customer_id", "device_id"]).cumcount() == 0).astype(int)
    d["devices_seen_prior"] = (
        d.groupby("customer_id")["device_id"]
        .transform(lambda s: (~s.duplicated()).cumsum().shift().fillna(0))
    )
    d["mins_since_prev"] = (
        d.groupby("customer_id")["timestamp"].diff().dt.total_seconds() / 60
    ).fillna(_NEVER_SEEN)

    # no history yet -> neutral values, never a peek forward
    d["txns_1h"] = d["txns_1h"].fillna(0)
    d["txns_24h"] = d["txns_24h"].fillna(0)
    d["distinct_merchants_24h"] = d["distinct_merchants_24h"].fillna(0)
    d["amount_vs_trailing_mean"] = d["amount_vs_trailing_mean"].fillna(1.0)

    for col, vocab in CATS.items():
        d[col] = pd.Categorical(d[col], categories=vocab)

    keep = FEATURES + [c for c in ("txn_id", "is_fraud", "timestamp", "customer_id",
                                   "amount", "merchant_id") if c in d.columns]
    return d[keep]
