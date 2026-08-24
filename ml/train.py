"""Train + honestly evaluate the fraud classifier.

Split is TEMPORAL (70/10/20 by time), never random: fraud tactics drift, and a
random split lets the model learn from transactions that hadn't happened yet.
Threshold is tuned on validation. The test set is scored exactly once, at the
end, with the threshold already fixed.

Run: python ml/generate_data.py && python ml/train.py
"""

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from features import CAT_FEATURES, FEATURES, compute_features

ROOT = pathlib.Path(__file__).parent.parent
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10


def temporal_split(d):
    """Cut on time, not on rows drawn at random."""
    d = d.sort_values("timestamp")
    t = d["timestamp"]
    t_train = t.quantile(TRAIN_FRAC)
    t_val = t.quantile(TRAIN_FRAC + VAL_FRAC)
    return (d[t <= t_train], d[(t > t_train) & (t <= t_val)], d[t > t_val])


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {"threshold": round(float(thr), 4), "precision": round(float(prec), 4),
            "recall": round(float(rec), 4), "f1": round(float(f1), 4)}


def main():
    raw = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["timestamp"])
    feats = compute_features(raw)

    train, val, test = temporal_split(feats)
    print(f"train {len(train):,} ({train.is_fraud.mean():.2%} fraud)  "
          f"val {len(val):,} ({val.is_fraud.mean():.2%})  "
          f"test {len(test):,} ({test.is_fraud.mean():.2%})")
    assert train.timestamp.max() <= val.timestamp.min(), "splits overlap in time"
    assert val.timestamp.max() <= test.timestamp.min(), "splits overlap in time"

    model = HistGradientBoostingClassifier(
        categorical_features=CAT_FEATURES,
        class_weight="balanced",     # 0.7% positives; without this it predicts all-legit
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=7,
    ).fit(train[FEATURES], train.is_fraud)

    # --- threshold chosen on VALIDATION only ---
    p_val = model.predict_proba(val[FEATURES])[:, 1]
    grid = np.unique(np.quantile(p_val, np.linspace(0.90, 0.9999, 400)))
    best = max((scores(val.is_fraud, p_val, t) for t in grid), key=lambda s: s["f1"])
    print(f"\nchosen on val -> {best}")

    # --- test set: scored once, threshold already frozen ---
    p_test = model.predict_proba(test[FEATURES])[:, 1]
    final = scores(test.is_fraud, p_test, best["threshold"])
    tn, fp, fn, tp = confusion_matrix(
        test.is_fraud, (p_test >= best["threshold"]).astype(int)).ravel()

    # the tradeoff judges should actually see, not one flattering number
    curve = [scores(test.is_fraud, p_test, t)
             for t in np.quantile(p_test, [0.90, 0.95, 0.98, 0.99, 0.995, 0.999])]

    metrics = {
        "test": final,
        "pr_auc": round(float(average_precision_score(test.is_fraud, p_test)), 4),
        "roc_auc": round(float(roc_auc_score(test.is_fraud, p_test)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "counts": {"train": len(train), "val": len(val), "test": len(test),
                   "test_fraud": int(test.is_fraud.sum())},
        "threshold_curve": curve,
        "base_rate": round(float(test.is_fraud.mean()), 5),
    }

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump({"model": model, "threshold": best["threshold"],
                 "features": FEATURES}, out / "fraud_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nTEST (held out, scored once)")
    print(f"  precision {final['precision']:.3f}   recall {final['recall']:.3f}   "
          f"f1 {final['f1']:.3f}   PR-AUC {metrics['pr_auc']:.3f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn:,}")
    print(f"  -> {fp} false positives against {tn + fp:,} legit txns "
          f"({fp / max(tn + fp, 1):.3%} of good traffic held)")
    print("\n  threshold tradeoff:")
    for c in curve:
        print(f"    thr {c['threshold']:.4f}  P {c['precision']:.3f}  "
              f"R {c['recall']:.3f}  F1 {c['f1']:.3f}")


if __name__ == "__main__":
    main()
