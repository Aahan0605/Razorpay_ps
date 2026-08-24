# Fraud Risk Detector + LLM Verifier

**Razorpay AI Builder Buildathon — Track 02: AI Risk Manager**

A gradient-boosted classifier scores every payment transaction for fraud risk. Every
flagged transaction is then passed to an LLM (Gemini 3.6 Flash), which explains the flag
in plain language and recommends one **bounded** action: `auto_block`, `hold_for_review`,
or `auto_clear`.

A fraud score is not an explanation — a risk analyst staring at `0.94` cannot tell a
card-testing burst from a customer who bought a laptop on holiday, and that difference
decides whether you block a good customer. The verifier turns the model's signals into
something a human can act on or overrule.

## The problem

Fraud detection has an asymmetric cost function: a missed fraud costs the chargeback, a
false positive costs a customer who often doesn't come back. At a 0.76% fraud base rate,
flagging even 1% of traffic touches thousands of good payments a day. The interesting
question isn't "what's the F1" — it's **what happens to the transactions in the middle**,
where the model is unsure. This project routes that middle band to an LLM reviewer
instead of forcing a binary call.

## Architecture

```mermaid
flowchart TB
    subgraph BUILT["Built in this repo"]
        A["Transaction events"] --> B["Causal feature pipeline<br/>device / location / velocity / history"]
        B --> C["Gradient-boosted classifier"]
        C --> D{"Threshold + action bands"}
        D -->|"score >= 0.85"| E["auto_block"]
        D -->|"0.30 to 0.85"| F["hold_for_review"]
        D -->|"score < 0.30"| G["auto_clear"]
        E --> H["LLM verifier<br/>explain + bounded action"]
        F --> H
        G --> H
        H --> I["FastAPI"]
        I --> J["Dashboard<br/>feed + false-positive KPI"]
        J -.->|"FP rate feeds threshold review"| D
    end

    subgraph OUT["Consumes the signal - deliberately out of scope"]
        K["AFA / step-up engine<br/>RBI 2025 Section 8 decision"]
        L["Network view<br/>cross-merchant ring detection"]
    end

    D -.->|"risk signal"| K
    C -.->|"per-customer blind spot"| L

    style C fill:#0C2651,color:#fff
    style H fill:#0D94FB,color:#fff
    style K stroke-dasharray: 4 4
    style L stroke-dasharray: 4 4
```

**The verifier is fenced in — in code, not just in the prompt.** It gets the transaction,
its behavioural features, the model's probability, and the action band that score falls
into. It may escalate caution but a recommendation *below* the band floor is overridden
server-side by `clamp_action()` in `backend/main.py` (tagged `band_clamped: true`), and
`backend/test_verify.py` asserts the floor holds. A system prompt is a request, not a
guarantee — the one direction that costs real money is a model talking you down from a
block, so that direction is enforced in code.

**This system scores risk; it does not authenticate.** The dashed boxes in the diagram —
an AFA/step-up engine, cross-merchant network detection — are what would consume this
signal, not things built here. See "Why this matters" and "Honest limitations" below.

| Path | What |
|---|---|
| `ml/generate_data.py` | Synthetic payment generator — 4 fraud archetypes + legit-suspicious traffic |
| `ml/features.py` | Causal feature engineering, shared by training and inference |
| `ml/test_features.py` | Leak test: asserts prefix-computed features equal full-frame features |
| `ml/train.py` | Temporal split, training, honest evaluation |
| `backend/main.py` | FastAPI: scored feed + LLM verifier (Gemini) |
| `frontend/` | React + Tailwind dashboard |

## Why this matters for Razorpay

**1. Risk-based authentication now has a deadline — and it names these exact features.**
The [RBI Authentication Mechanisms Directions, 2025](https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12898&Mode=0)
(circular RBI/2025-26/79) take effect 1 April 2026 and require two authentication
factors. It regulates *authentication*, not fraud scoring — but Section 8 explicitly
permits risk-based evaluation against "transaction location, user behaviour patterns,
device attributes, historical transaction profile" — almost exactly this feature set
(`ip_country`, `txns_24h`, `is_new_device`, `amount_vs_trailing_mean`, ...). A step-up
decision is made *at authorization time*, with only the past available, which is why the
causal-features fix (below) isn't academic: a velocity feature computed with a
whole-dataset `groupby` isn't just optimistic, it's unimplementable at that point in
time. This system supplies the risk signal a compliant AFA engine would consume — it is
not itself an AFA implementation.

**2. UPI is a push rail, so there is no chargeback to fall back on.** UPI carried
[83.4% of India's digital payment volume in FY25](https://www.business-standard.com/finance/news/upi-s-contribution-to-payments-ecosystem-volume-grows-to-83-4-in-fy25-125052900871_1.html),
rising to [85.5% in H2 2025](https://www.ibef.org/news/upi-accounted-for-85-5-of-digital-transaction-volume-in-h2-2025-rbi-report).
It's a *push* mechanism — no chargeback rail, recovery means freezing the beneficiary
account before cash-out, a race measured in minutes. Post-hoc detection is worth far
less than pre-authorization scoring on this rail, which is why `auto_block` exists as a
distinct action instead of routing everything to human review — review latency is only
affordable on the middle band.

**3. False-positive rate is a business KPI, not an accuracy footnote.** Razorpay's own
[Vulcan launch](https://www.dqindia.com/news/razorpay-vulcan-8x-fraud-detection-baseline-12398348)
framing — 5× more fraud caught **without increasing alert volume** — is the
false-positive constraint stated as the headline metric. That's why the dashboard puts
**good traffic held (0.17%)** next to precision/recall, and why the first trained model
(P 0.986 / R 0.953, 2 false positives in 14,000) was rejected rather than shipped: it
was a data artifact that would have left `hold_for_review` empty and decorative.

**4. Cross-merchant rings are the real gap, and this design cannot see them.** Every
velocity feature here is per customer, at one merchant — a ring spreading thin across
200 merchants looks like 200 unremarkable first-time transactions. This is precisely the
gap [Vulcan](https://press.aboutamazon.com/aws-international/2026/8/razorpay-launches-vulcan-indias-first-ai-payments-foundation-model-fueled-by-nvidia-and-aws-re-architecting-payments-for-a-350-bn-e-comm-future-by-2030)
is built to close, trained across the whole network rather than one merchant's data.
These are complementary layers, not competitors: this repo builds the merchant-side
explainable layer and is explicit about not attempting network-view detection.

## Data

**A purpose-built synthetic generator, not IEEE-CIS or the ULB credit-card set.** Both
public datasets ship signal as PCA components (`V1…V339`) — meaning destroyed before
download. A classifier trains on that fine, but the verifier would be asked to explain
`V257 = 3.2`, which isn't explainable. A generator keeps the domain right (UPI / card /
netbanking / wallet) and the repo reproducible with no Kaggle auth.

72,452 transactions, 180 days, 8,000 customers, **0.76% fraud**, four archetypes with
deliberate overlap into legitimate behaviour:

| Archetype | Signature |
|---|---|
| Card testing | Rapid low-value probes across many merchants, fresh device, declines |
| Account takeover | Known customer, new device + new geography, escalating amounts |
| Stolen card | New-ish account, high value, country mismatch |
| Slow drain | Victim's own device and geography, ordinary amounts — intentionally hard |

~30% of fraud episodes are "careful" (proxied through the victim's own device/country),
so geography and device mismatch are useful but never giveaways. The generator also
emits **legitimate traffic engineered to look like fraud** — travellers, phone upgrades,
big-ticket purchases, and a new-device-plus-abroad-plus-large-amount pattern
indistinguishable from takeover on the available features. Those are the false
positives, and they're supposed to be there — without them the review queue is empty
and precision is fiction. 8% of fraud is scrubbed of every tell, so recall has a real
ceiling.

**Features** — stateless: amount, hour, day, payment method, merchant category,
IP/billing mismatch, account age. Behavioural (per customer, computed causally):
transactions in the last 1h/24h, distinct merchants in 24h, amount vs. trailing mean,
new-device flag, minutes since previous transaction.

**Split & leakage.** Temporal 70/10/20 split, cut on time — fraud tactics drift, and a
random split would let the model learn from transactions that hadn't happened yet. The
real leakage risk was never the split, though — it was the aggregates. A velocity
feature computed with a whole-dataset `groupby` embeds the future into every row, which
is the standard way fraud pipelines inflate their own scores. Every behavioural feature
here is computed from strictly prior transactions (`rolling(..., closed="left")`)
through the same code path at training and inference.

`ml/test_features.py` enforces this by computing features on every prefix of a frame and
asserting they equal the full-frame values. **It caught a real bug:**
`groupby(...).rolling(...)` returns rows in *group* order, not original row order, so
assigning back onto a time-sorted frame was silently scrambling velocity features across
customers. Threshold is tuned on validation; the test set is scored exactly once.

## Model card

`HistGradientBoostingClassifier`, 300 iterations, `class_weight="balanced"`, native
categorical support.

**Held-out test set: 14,491 transactions, 110 fraudulent.** Threshold 0.463 (selected on
validation).

| Metric | Value |
|---|---|
| Precision | **0.769** |
| Recall | **0.727** |
| F1 | **0.748** |
| PR-AUC | **0.789** |
| ROC-AUC | 0.976 |

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actually legit** | 14,357 | 24 |
| **Actually fraud** | 30 | 80 |

**The false-positive tradeoff, stated plainly:** 24 false positives against 14,381
legitimate transactions (**0.17% of good traffic held**), 30 frauds missed. The
threshold is the dial between them:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.0003 | 0.070 | 0.927 | 0.131 |
| 0.0096 | 0.317 | 0.836 | 0.460 |
| 0.077 | 0.579 | 0.764 | 0.659 |
| **0.463** (chosen) | **0.769** | **0.727** | **0.748** |
| 0.950 | 0.959 | 0.636 | 0.765 |
| 1.000 | 1.000 | 0.136 | 0.240 |

Catching 93% of fraud means holding roughly one in fourteen good payments — the whole
argument for the `hold_for_review` band, since a binary threshold has to be wrong in one
direction. Note **ROC-AUC (0.976) flatters the model** relative to PR-AUC (0.789); at a
0.76% base rate, PR-AUC is the honest headline number.

### Honest limitations

Ordered by production impact, not by ease of admission.

1. **Synthetic data.** The model partly learns patterns I planted, so these numbers are
   optimistic vs. real traffic — the overlap/careful-fraudster/look-alike mitigations
   above make it a fair test of the *pipeline*, not a production performance prediction.
2. **No cross-merchant view** — every velocity feature is scoped to one customer at one
   merchant, so a thin-spread ring is invisible by construction. Architectural boundary,
   not a bug; closing it needs network-level data this repo doesn't have.
3. **No beneficiary-side signals**, which is where UPI fraud actually lives (mule-account
   age, inflow velocity, freeze-window race). The generator models the payer side only —
   the UPI framing above is about design priorities, not UPI-specific detection.
4. **Not an AFA implementation.** Produces a risk score only; no authentication, no
   compliance claim under RBI/2025-26/79.
5. **Validation F1 was 0.857, test F1 is 0.748** — genuine temporal drift, exactly what a
   time-based split is supposed to expose.
6. **Threshold is F1-optimal, not cost-optimal.** F1 implicitly weights a false positive
   and a missed fraud equally; they aren't equal on a no-chargeback rail. A production
   threshold should minimise expected cost using real per-outcome figures — everything
   needed is in `models/metrics.json` (regenerate via `ml/train.py`), it just needs the
   business's actual numbers.
7. **Verifier is unevaluated** — explanation quality is assessed by reading it, not by a
   metric.
8. **Feature attribution is implicit**, not SHAP — the verifier's "key signals" are
   plausible reasoning over feature values, not provably the model's actual drivers.
9. **No calibration** — scores rank and band correctly but aren't calibrated
   probabilities; the 0.30/0.85 edges are operating points, not likelihood statements.

## Running it

Requires Python 3.11+ and Node 18+.

```bash
pip install -r requirements.txt
python ml/generate_data.py && python ml/test_features.py && python ml/train.py
```

Add your Gemini key — free tier, no card required, at
[aistudio.google.com](https://aistudio.google.com). Without it the API and dashboard
still run, but `/verify` returns a clear 503:

```bash
cp .env.example .env   # paste your key into GEMINI_API_KEY=
```

```bash
python -m uvicorn backend.main:app --port 8000        # backend
npm install --prefix frontend && npm run dev --prefix frontend   # frontend, second terminal
```

Open http://localhost:5173.

### API

| Endpoint | Purpose |
|---|---|
| `GET /transactions?limit=&flagged_only=` | Scored feed, replayed from the held-out test set |
| `POST /verify/{txn_id}` | LLM explanation + bounded action (cached per transaction) |
| `GET /metrics` | Evaluation results behind the dashboard model card |

The feed is the **test set**, not a random sample — every score is a genuine
out-of-sample prediction.

## Notes

`data/` and `models/` are gitignored and regenerate from the commands above. Real keys
stay in `.env`, which is gitignored — `.env.example` is the template.
