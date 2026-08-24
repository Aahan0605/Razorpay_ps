# Fraud Risk Detector + LLM Verifier

**Razorpay AI Builder Buildathon — Track 02: AI Risk Manager**

A gradient-boosted classifier scores every payment transaction for fraud risk. Every
flagged transaction is then passed to Claude, which explains the flag in plain language
and recommends one **bounded** action: `auto_block`, `hold_for_review`, or `auto_clear`.

The point of the second stage is that a fraud score is not an explanation. A risk analyst
staring at `0.94` cannot tell a card-testing burst from a customer who bought a laptop
while on holiday — and the difference decides whether you block a good customer. The
verifier turns the model's signals into something a human can act on or overrule.

---

## The problem

Fraud detection has an asymmetric cost function. A missed fraud costs the chargeback; a
false positive costs a legitimate customer, who often does not come back. At a 0.76%
fraud base rate, a model that flags 1% of traffic is touching thousands of good payments
a day. So the interesting question is not "what's the F1" — it's **what happens to the
transactions in the middle**, where the model is unsure.

This project routes that middle band to an LLM reviewer instead of forcing a binary call.

## Architecture

```mermaid
flowchart LR
    A[Transaction events<br/>synthetic generator] --> B[Causal feature pipeline<br/>ml/features.py]
    B --> C[HistGradientBoosting<br/>classifier]
    C --> D{Fraud score}
    D -->|>= 0.85| E[auto_block]
    D -->|0.30 - 0.85| F[hold_for_review]
    D -->|< 0.30| G[auto_clear]
    E --> H[Claude verifier<br/>explain + recommend]
    F --> H
    G --> H
    H --> I[FastAPI<br/>/transactions /verify /metrics]
    I --> J[React dashboard]

    style C fill:#0C2651,color:#fff
    style H fill:#0D94FB,color:#fff
    style J fill:#F7F9FC,color:#0C2651
```

The verifier is deliberately fenced in. It receives the transaction, its behavioural
features, the model's probability, and **the action band that score falls into**. It may
escalate caution (recommend `hold_for_review` where the band allows `auto_clear`) but it
cannot recommend a less cautious action than the classifier's score permits. The LLM
explains and advises; it does not get to overrule the model downward.

| Path | What |
|---|---|
| `ml/generate_data.py` | Synthetic payment generator — 4 fraud archetypes + legit-suspicious traffic |
| `ml/features.py` | Causal feature engineering, shared by training and inference |
| `ml/test_features.py` | Leak test: asserts prefix-computed features equal full-frame features |
| `ml/train.py` | Temporal split, training, honest evaluation |
| `backend/main.py` | FastAPI: scored feed + Claude verifier |
| `frontend/` | React + Tailwind dashboard |

## Data

**A purpose-built synthetic generator, not IEEE-CIS or the ULB credit-card set.**

Both of those public datasets ship their signal as PCA components (`V1…V339`) — the
meaning was destroyed before you could download it. A classifier trains on them fine, but
the LLM verifier would be asked to explain `V257 = 3.2`, which is not explainable. Since
explanation is the headline feature here, anonymised features break the product by
construction. A generator also keeps the domain right (UPI / card / netbanking / wallet)
and makes the repo reproducible from one command with no Kaggle auth.

72,452 transactions over 180 days, 8,000 customers, **0.76% fraud**.

Four fraud archetypes, all with deliberate overlap into legitimate behaviour:

| Archetype | Signature |
|---|---|
| Card testing | Rapid low-value probes across many merchants, fresh device, declines |
| Account takeover | Known customer, new device + new geography, escalating amounts |
| Stolen card | New-ish account, high value, country mismatch |
| Slow drain | Victim's own device and geography, ordinary amounts — intentionally hard |

Roughly 30% of fraud episodes are "careful" — the fraudster proxies through the victim's
own country and device — so geography and device mismatch are useful signals but never
giveaways. The generator also emits **legitimate traffic engineered to look like fraud**:
travellers, phone upgrades, genuine big-ticket purchases, and a combined
new-device-plus-abroad-plus-large-amount pattern that is indistinguishable from account
takeover on the available features. Those are the false positives, and they are supposed
to be there — without them the review queue is empty and precision is fiction.

Finally, 8% of fraud is scrubbed of every tell, so recall has a real ceiling.

### Features

Stateless: log amount, hour, day of week, night flag, payment method, merchant category,
email-domain class, IP country, IP/billing mismatch, account age, new-account flag,
failed attempts.

Behavioural (per customer, computed causally): transactions in the last 1h and 24h,
distinct merchants in 24h, amount vs. the customer's trailing mean, first-sighting-of-device
flag, distinct devices seen before, minutes since previous transaction.

### Split strategy and leakage

**Temporal split — 70% train / 10% validation / 20% test, cut on time, not on rows.** Fraud
tactics drift, and a random split lets the model learn from transactions that had not
happened yet. Returning customers do appear on both sides, which mirrors production:
you score tomorrow's traffic with today's model.

**The real leakage risk was never the split — it was the aggregates.** Velocity features
computed with a whole-dataset `groupby` embed the future into every row and are the
standard way fraud pipelines inflate their own scores. Every behavioural feature here is
computed from strictly prior transactions (`rolling(..., closed="left")`), through the same
code path at training and inference time.

`ml/test_features.py` enforces this: it computes features on every prefix of a frame and
asserts they equal the full-frame values for those rows. If a later transaction can change
an earlier row's features, the test fails. **It caught a real bug** —
`groupby(...).rolling(...)` returns rows in *group* order, not original row order, so
assigning the result back onto a time-sorted frame was silently scrambling velocity
features across customers.

The threshold is tuned on validation. The test set is scored exactly once, at the end.

## Model card

`HistGradientBoostingClassifier` (scikit-learn), 300 iterations, learning rate 0.06,
`class_weight="balanced"`, native categorical support. Chosen over XGBoost because sklearn
handles the categoricals natively and adds no dependency; the gap on tabular data this
size is not meaningful.

**Held-out test set: 14,491 transactions, 110 fraudulent (0.76% base rate).**
Threshold 0.463, selected on validation.

| Metric | Value |
|---|---|
| Precision | **0.769** |
| Recall | **0.727** |
| F1 | **0.748** |
| PR-AUC | **0.789** |
| ROC-AUC | 0.976 |

Confusion matrix:

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | 14,357 | 24 |
| **Actually fraud** | 30 | 80 |

### The false-positive tradeoff, stated plainly

24 false positives against 14,381 legitimate transactions — **0.17% of good traffic held**.
30 frauds missed. Neither number is free, and the threshold is the dial between them:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.0003 | 0.070 | 0.927 | 0.131 |
| 0.0096 | 0.317 | 0.836 | 0.460 |
| 0.077 | 0.579 | 0.764 | 0.659 |
| **0.463** (chosen) | **0.769** | **0.727** | **0.748** |
| 0.950 | 0.959 | 0.636 | 0.765 |
| 1.000 | 1.000 | 0.136 | 0.240 |

Catching 93% of fraud means holding roughly one in fourteen good payments. That is the
whole argument for the `hold_for_review` band: the middle of this curve is where an LLM
reviewer earns its cost, because a binary threshold has to be wrong in one direction.

Note that **ROC-AUC (0.976) flatters the model** relative to PR-AUC (0.789). At a 0.76%
base rate, ROC-AUC is the wrong headline metric — PR-AUC is the honest one.

### Honest limitations

1. **The data is synthetic.** The model partly learns patterns I planted, so these numbers
   are optimistic versus real payment traffic. The mitigations above (overlapping
   distributions, careful fraudsters, unlearnable fraud, look-alike legitimate traffic)
   make it a fair test of the *pipeline*, not a prediction of production performance.
2. **Validation F1 was 0.857, test F1 is 0.748.** That drop is genuine temporal drift and
   is exactly what a time-based split is supposed to expose. A random split would have
   hidden it.
3. **The verifier is not evaluated.** Explanation quality is assessed by reading it, not
   by a metric. A proper eval would need labelled analyst judgements.
4. **No calibration.** Scores are usable for ranking and banding; they are not calibrated
   probabilities.
5. **Feature attribution is implicit.** The verifier reasons over feature values rather
   than SHAP values, so its "key signals" are plausible rather than provably the model's
   actual drivers. SHAP would close that gap.

## Running it

Requires Python 3.11+ and Node 18+.

```bash
pip install -r requirements.txt
```

Generate the data, verify causality, train, and print the evaluation:

```bash
python ml/generate_data.py && python ml/test_features.py && python ml/train.py
```

Add your Anthropic key — without it the API and dashboard still run, but `/verify`
returns a clear 503 instead of an explanation:

```bash
cp .env.example .env
```

Backend:

```bash
python -m uvicorn backend.main:app --port 8000
```

Frontend, in a second terminal:

```bash
npm install --prefix frontend && npm run dev --prefix frontend
```

Open http://localhost:5173.

### API

| Endpoint | Purpose |
|---|---|
| `GET /transactions?limit=&flagged_only=` | Scored feed, replayed from the held-out test set |
| `POST /verify/{txn_id}` | Claude explanation + bounded action (cached per transaction) |
| `GET /metrics` | Evaluation results behind the dashboard model card |

The feed is the **test set**, not a random sample — every score shown is a genuine
out-of-sample prediction.

## Notes

`data/` and `models/*.joblib` are gitignored and regenerate from the commands above;
`models/metrics.json` is versioned so evaluation results stay reviewable in the diff.
Real keys stay in `.env`, which is gitignored — `.env.example` is the template.
