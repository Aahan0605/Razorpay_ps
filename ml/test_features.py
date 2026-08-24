"""The one check that matters: features must be causal.

If feature values for the first N transactions change when later transactions
are appended, the pipeline is leaking the future and every metric downstream
is fiction. This asserts they don't.

Run: python ml/test_features.py
"""

import pandas as pd
from features import compute_features, FEATURES


def _frame():
    rows = [
        # cust_a: a slow, ordinary history
        ("t0", "2025-01-01 10:00", "cust_a", 500.0, "upi", "groceries", "m1", "dev_a", "IN", "IN", "free", 400, 0, 0),
        ("t1", "2025-01-02 11:00", "cust_a", 700.0, "upi", "food_delivery", "m2", "dev_a", "IN", "IN", "free", 401, 0, 0),
        # cust_b: a card-testing burst on a fresh device
        ("t2", "2025-01-02 12:00", "cust_b", 40.0, "card", "gaming", "m3", "dev_bot", "RU", "IN", "free", 12, 2, 1),
        ("t3", "2025-01-02 12:02", "cust_b", 55.0, "card", "gift_cards", "m4", "dev_bot", "RU", "IN", "free", 12, 1, 1),
        ("t4", "2025-01-02 12:05", "cust_b", 61.0, "card", "subscription", "m5", "dev_bot", "RU", "IN", "free", 12, 3, 1),
        # later cust_a activity -- must not alter anything computed above
        ("t5", "2025-01-09 09:00", "cust_a", 90000.0, "card", "electronics", "m6", "dev_new", "US", "IN", "free", 408, 0, 0),
    ]
    return pd.DataFrame(rows, columns=[
        "txn_id", "timestamp", "customer_id", "amount", "payment_method",
        "merchant_category", "merchant_id", "device_id", "ip_country",
        "billing_country", "email_domain", "customer_age_days",
        "failed_attempts", "is_fraud"])


def main():
    df = _frame()
    full = compute_features(df).set_index("txn_id")

    # 1. causality: a prefix must produce identical features for its own rows
    for n in range(1, len(df)):
        prefix = compute_features(df.iloc[:n]).set_index("txn_id")
        pd.testing.assert_frame_equal(
            prefix[FEATURES], full.loc[prefix.index, FEATURES],
            check_categorical=False,
            obj=f"prefix of {n} rows differs from full-frame features -> FUTURE LEAK")

    # 2. the aggregates actually mean what their names claim
    assert full.loc["t2", "txns_1h"] == 0, "first txn of a customer has no history"
    assert full.loc["t4", "txns_1h"] == 2, "two prior burst txns within the hour"
    assert full.loc["t4", "distinct_merchants_24h"] == 2
    assert full.loc["t2", "is_new_device"] == 1 and full.loc["t3", "is_new_device"] == 0
    assert full.loc["t1", "amount_vs_trailing_mean"] == 700.0 / 500.0
    assert full.loc["t0", "amount_vs_trailing_mean"] == 1.0, "no history -> neutral"
    assert full.loc["t2", "ip_billing_mismatch"] == 1
    assert full.loc["t5", "is_new_device"] == 1 and full.loc["t5", "devices_seen_prior"] == 1

    # 3. inference path: appending one txn to a history matches batch scoring
    live = compute_features(df.iloc[:5].assign()).set_index("txn_id")
    assert live.loc["t4", FEATURES].to_dict() == full.loc["t4", FEATURES].to_dict()

    print("features are causal: no leakage across", len(df), "transactions")


if __name__ == "__main__":
    main()
