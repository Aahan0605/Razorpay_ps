"""FastAPI service: model scores every transaction, Claude explains the flagged ones.

Two endpoints back the dashboard:
  GET  /transactions      the scored feed (held-out test set, replayed)
  POST /verify/{txn_id}   Claude's plain-language explanation + bounded action

The verifier is deliberately constrained: it picks from three actions and cannot
invent a fourth, and the action band is derived from the model score rather than
left to the LLM's discretion. The LLM explains and recommends inside that band --
it does not get to override the classifier.

Run: uvicorn backend.main:app --reload   (from the repo root)
"""

import json
import os
import pathlib
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from features import FEATURES, compute_features  # noqa: E402
from train import temporal_split  # noqa: E402

# minimal .env loader -- avoids a python-dotenv dependency for three lines of work
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

MODEL = "claude-opus-5"
ACTIONS = ["auto_block", "hold_for_review", "auto_clear"]

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {
            "type": "string",
            "description": "2-3 sentences, plain language, for a risk analyst who "
                           "cannot see the model internals. Name the specific signals.",
        },
        "recommended_action": {"type": "string", "enum": ACTIONS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "key_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The 2-4 features that drove the score, in human terms.",
        },
    },
    "required": ["explanation", "recommended_action", "confidence", "key_signals"],
    "additionalProperties": False,
}

SYSTEM = """You are a fraud-risk analyst reviewing transactions that an ML classifier \
has already scored. Your job is to explain the score in plain language and recommend one \
bounded action.

You are given the transaction, its behavioural features, and the model's fraud probability.
Explain WHY this looks risky (or doesn't) by naming the specific signals. Do not restate the
probability as if it were an explanation.

Action bands -- stay inside the band you are given:
- score >= 0.85 -> auto_block
- 0.30 <= score < 0.85 -> hold_for_review
- score < 0.30 -> auto_clear

You may recommend a MORE cautious action than the band allows (hold instead of clear) if a
signal genuinely warrants it, but never a less cautious one. Legitimate travel, a new phone,
and one-off large purchases are common false positives -- say so when the evidence fits them
better than fraud."""

app = FastAPI(title="Fraud Risk Detector + LLM Verifier")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_bundle = joblib.load(ROOT / "models" / "fraud_model.joblib")
_model, _threshold = _bundle["model"], _bundle["threshold"]

# Replay the held-out test set as the live feed: these are transactions the model
# has genuinely never seen, so the demo shows real performance, not memorisation.
_raw = pd.read_csv(ROOT / "data" / "transactions.csv", parse_dates=["timestamp"])
_feats = compute_features(_raw)
_, _, _test = temporal_split(_feats)
_test = _test.copy()
_test["score"] = _model.predict_proba(_test[FEATURES])[:, 1]
_test = _test.sort_values("timestamp", ascending=False).reset_index(drop=True)

_raw_by_id = _raw.set_index("txn_id")
_verdicts: dict[str, dict] = {}


def _band(score: float) -> str:
    return "auto_block" if score >= 0.85 else "hold_for_review" if score >= 0.30 else "auto_clear"


def _row(r) -> dict:
    return {
        "txn_id": r.txn_id,
        "timestamp": r.timestamp.isoformat(),
        "customer_id": r.customer_id,
        "amount": round(float(r.amount), 2),
        "payment_method": str(r.payment_method),
        "merchant_category": str(r.merchant_category),
        "merchant_id": r.merchant_id,
        "score": round(float(r.score), 4),
        "flagged": bool(r.score >= _threshold),
        "band": _band(float(r.score)),
        "is_fraud": int(r.is_fraud),  # ground truth -- demo only, never shown to the verifier
        "verdict": _verdicts.get(r.txn_id),
    }


@app.get("/transactions")
def transactions(limit: int = 60, flagged_only: bool = False):
    d = _test[_test.score >= _threshold] if flagged_only else _test
    return {
        "threshold": round(float(_threshold), 4),
        "total": len(_test),
        "flagged_total": int((_test.score >= _threshold).sum()),
        "transactions": [_row(r) for r in d.head(limit).itertuples()],
    }


@app.get("/metrics")
def metrics():
    return json.loads((ROOT / "models" / "metrics.json").read_text())


@app.post("/verify/{txn_id}")
def verify(txn_id: str):
    if txn_id in _verdicts:
        return _verdicts[txn_id]

    match = _test[_test.txn_id == txn_id]
    if match.empty:
        raise HTTPException(404, f"{txn_id} not in the scored feed")
    row, score = match.iloc[0], float(match.iloc[0].score)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set -- copy .env.example to .env")

    raw = _raw_by_id.loc[txn_id]
    payload = {
        "transaction": {
            "amount": float(raw.amount), "payment_method": raw.payment_method,
            "merchant_category": raw.merchant_category, "ip_country": raw.ip_country,
            "billing_country": raw.billing_country, "email_domain": raw.email_domain,
            "failed_attempts": int(raw.failed_attempts),
            "customer_age_days": float(raw.customer_age_days),
        },
        "behavioural_features": {f: _fmt(row[f]) for f in FEATURES},
        "model_fraud_probability": round(score, 4),
        "allowed_action_for_this_band": _band(score),
    }

    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM,
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
    )
    if resp.stop_reason == "refusal":
        raise HTTPException(502, "verifier declined this request")

    verdict = json.loads(next(b.text for b in resp.content if b.type == "text"))
    verdict["model"] = MODEL
    _verdicts[txn_id] = verdict
    return verdict


def _fmt(v):
    """Categoricals serialise as-is; floats get rounded so the prompt stays readable."""
    return round(float(v), 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
