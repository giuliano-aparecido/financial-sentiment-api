# Project Overview

## What this is

A FastAPI backend for a financial-news sentiment/reasoning demo: takes a
free-text (or `$TICKER`-cashtagged) query, fetches live news for the
ticker, and runs a fine-tuned LLM's chain-of-thought reasoning against it
via a Hugging Face Inference endpoint. Serves
[`financial-sentiment-web`](https://github.com/giuliano-aparecido/financial-sentiment-web).
The fine-tuning pipeline for the model itself lives in
[`financial-sentiment-model`](https://github.com/giuliano-aparecido/financial-sentiment-model).

Swiss market research scans (volatility indicator, crash rebound,
same-day big-loss alert) used to live here too - extracted to a private
sibling repo, [`financial-research-api`](https://github.com/giuliano-aparecido/financial-research-api),
so this repo is only the `$TICKER` AI-reasoning feature.

## Architecture

```
main.py                  App creation, middleware, lifespan (shared httpx
                          client, seeds the yfinance session), router
                          mounting - deliberately thin
app/
  config.py               All env-var reads in one place
  models.py                Pydantic request models
  deps.py                  API-key auth dependency
  limiter.py               Global slowapi rate limit
  services/
    ticker.py               $CASHTAG extraction (falls back to a
                             capitalized-word heuristic if no cashtag)
    parsing.py               Brace-balanced JSON extraction from raw LLM
                             output (handles trailing model chatter,
                             markdown fences, special tokens)
    news.py                  Google News RSS fetch, with a timeout
                             (feedparser's own url-fetching has none)
    news_classifier.py       Gemini-based relevance/importance screen for
                             which headlines actually reach the model
    fundamentals.py          yfinance current-price/P-E/dividend/52wk
                             fetch + "Current Market Data" block renderer
    valuation.py             Scenario-weighted 2-stage DCF valuation
                             (deterministic math, never LLM-generated) +
                             block renderer
    earnings.py               yfinance last-quarter revenue/EPS/next-
                             earnings-date fetch + block renderer
    inference.py              HF inference call + response parsing; also
                             holds the mutable in-memory per-model
                             inference URLs (see below)
    yf_session.py             Seeds/hot-swaps yfinance's crumb/cookie
                             singleton (see below)
  routers/
    health.py, analyze.py, admin.py   Thin HTTP layer only
```

The routers are deliberately thin. All business logic — news filtering,
DCF math, inference-URL resolution — lives in `app/services/` and is
unit-tested there.

## Key design decisions

- **Multiple models behind one API, with no model names in code.**
  `/api/analyze?model=<name>` picks which fine-tuned model answers;
  `DEFAULT_MODEL` (default `llama`) is used when the param is absent, and
  a name that isn't registered is a 400. Models are registered two ways,
  neither needing a code change:
  - **Env var convention at startup:** every `<NAME>_INFERENCE_URL` env
    var registers model `<name>` (lowercased) — `KIM_INFERENCE_URL` makes
    `?model=kim` work. `DEFAULT_MODEL` must have one, or the app refuses
    to start.
  - **At runtime:** `POST /api/update-inference-url` with a `model` that
    isn't registered yet registers it (see next bullet).
  `GET /health` lists what's registered. All models share the same auth
  (`HF_TOKEN` as bearer), host allowlist, and rate limit.
- **The per-model inference URLs are mutable, in-memory globals**,
  updatable via `POST /api/update-inference-url` (`{"url": ..., "model":
  ...}`, `model` defaulting to `DEFAULT_MODEL`) — this exists so a
  Colab-hosted model (behind an ngrok tunnel that gets a new URL on every
  restart) can push its current URL without a redeploy. Deliberately not
  made more robust/persistent: this app runs a single process/instance
  and doesn't need to scale, so the added complexity of a shared store
  wouldn't earn its keep.
- **The update-inference-url host allowlist** (`ALLOWED_INFERENCE_HOST_SUFFIXES`)
  matches on an exact host or a `.`-bounded suffix, not a bare
  `str.endswith()` — a plain `endswith` would accept a registered
  look-alike domain like `evil-huggingface.co`, and since the resolved
  URL receives the HF bearer token on every subsequent request, that bug
  would have been a token-exfiltration path, not just a bad-config one.
- **Valuation is always deterministic, code-only math — never
  LLM-generated.** `valuation.py` replaced an earlier Graham Number
  implementation (`sqrt(22.5 x EPS x book value/share)`) that was
  confirmed live to be badly broken for asset-light, buyback-heavy
  companies (Apple showed "725% overvalued" purely because its book
  value/share is tiny). The current model instead classifies each
  company into one of four valuation bases — FCF (asset-heavy
  industrials), EPS/Net Income (tech/growth/platform), Dividends (mature
  cash-cow/REIT-style payers), or Revenue (early-stage unprofitable
  growth) — since a single metric can't meaningfully value both a bank
  and a pre-profit growth company. The classified metric is projected
  across a 2-stage (years 1-5, years 6-10), 3-scenario (Normal/Best/Worst
  probability-weighted) growth model with an exit-multiple terminal
  value, discounted at a flat rate. Missing/non-positive inputs for the
  classified basis render "Not applicable" (a real, expected outcome); a
  failed fundamentals fetch renders "Data unavailable." — the two are
  deliberately distinct strings so the model can learn to tell "no
  defined value" from "couldn't fetch anything." Known limitation:
  extreme-multiple growth stocks (e.g. Tesla, trailing P/E ~300) can
  still show implausibly high "overvalued" percentages even after this
  change, since a disciplined, non-circular growth assumption can't fully
  bridge a razor-thin trailing EPS — see `app/services/valuation.py`'s
  module docstring for the two earlier, rejected attempts at fixing this
  and why they made it worse.
- **The yfinance crumb is seeded at startup, and can be hot-swapped
  without a restart.** `app/services/yf_session.py` works around Render's
  outbound IP being blocked at Yahoo's crumb-fetch endpoint by seeding a
  crumb/cookie pair captured from elsewhere; `POST /api/update-yf-crumb`
  lets `scripts/refresh_yf_crumb.py` push a fresh one in without a
  redeploy when the seeded one goes stale. `financial-research-api` has
  its own independent copy of this same mechanism for its own yfinance
  calls - each process needs its own seeded session.
- **The rate limiter keys on a constant, not client IP.** Every real
  request arrives via the Next.js frontend's single proxy IP, so per-IP
  keying already bucketed all legitimate traffic together — and since
  Render forwards whatever `X-Forwarded-For` a direct caller sends,
  per-IP keying was also spoofable (a fresh header value = a fresh
  bucket). A single global bucket matches this app's actual traffic shape
  and can't be rotated around.

## Auth

`app/deps.py::verify_api_key` gates every route that spends inference
quota or Yahoo Finance request budget (`/api/analyze`, `/api/update-*`),
via `secrets.compare_digest` against `API_KEY` (header `X-API-Key`).
There's no per-user auth — this is a single-tenant demo API — the key
just distinguishes "the frontend" from anyone else hitting the URL
directly.

## Rate limiting

A global `slowapi` limiter (`app/limiter.py`), keyed on a constant string
rather than client IP (see "Key design decisions" above for why). Applies
to every route uniformly rather than per-route decorators, matching the
single-caller (the frontend proxy) traffic shape this API actually sees.
