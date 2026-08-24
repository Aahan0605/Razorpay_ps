# Fraud Risk Detector + LLM Verifier

Razorpay AI Builder Buildathon — Track 02: AI Risk Manager

An ML classifier scores payment transactions for fraud risk. Every flagged
transaction is then passed to an LLM verifier (Claude) which explains the flag in
plain language and recommends a bounded action: `auto_block`, `hold_for_review`,
or `auto_clear`.

**Status:** in progress. See commit history.

## Structure

| Path | What |
|---|---|
| `ml/` | dataset generation, feature engineering, model training + eval |
| `backend/` | FastAPI service: `/score` (model) + `/verify` (LLM) |
| `frontend/` | React + Tailwind dashboard |

## Setup

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
```
