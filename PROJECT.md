# Project Overview

## What this is

A FastAPI backend for a financial-news sentiment/reasoning demo: takes a
free-text (or `$TICKER`-cashtagged) query, fetches live news for the
ticker, and runs a fine-tuned LLM's chain-of-thought reasoning against it
via a Hugging Face Inference endpoint. Serves
[`financial-sentiment-web`](https://github.com/GiulianoAparecido/financial-sentiment-web).
The fine-tuning pipeline for the model itself lives in
[`financial-sentiment-model`](https://github.com/GiulianoAparecido/financial-sentiment-model).

Two largely-independent feature areas share this app: the `$TICKER`
analysis endpoint (the core "AI reasoning" feature) and a set of Swiss
market research scans (volatility indicator, crash rebound, same-day
big-loss alert) that run against yfinance and persist results to Postgres.

## Architecture

```
main.py                  App creation, middleware, lifespan (shared httpx
                          client), router mounting - deliberately thin
app/
  config.py               All env-var reads in one place
  models.py                Pydantic request models
  deps.py                  API-key auth dependency
  limiter.py               Global slowapi rate limit
  db/                      SQLAlchemy engine/session + models for the
                            persisted research scans
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
    swiss_universe.py         The scanned Swiss-listed ticker universe
    swiss_today_screener.py, swiss_crash_rebound.py,
    swiss_volatility_indicator.py   The three Swiss research scans' logic
    research_job.py           Single-flight guarded pipeline that runs a
                             scan and persists it
    scan_persistence.py       Postgres read/write for the last-persisted
                             scan of each type
    scheduler.py              On-demand scan trigger (name is historical -
                             no longer a real scheduler, see below)
    alerts.py                 Renders the big-loss alert email from a
                             scan result
  routers/
    health.py, analyze.py, admin.py, research.py   Thin HTTP layer only
scripts/send_big_loss_alert.py   Run by the GitHub Actions cron - starts
                             a scan via the API, polls for the rendered
                             email, sends it over SMTP itself (see below)
```

The routers are deliberately thin. All business logic — news filtering,
DCF math, inference-URL resolution, the Swiss scan pipelines — lives in
`app/services/` and is unit-tested there.

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
- **The rebound and volatility-indicator research tables are on-demand,
  served from the last persisted run.** `GET /api/research/volatility/
  {rebound,indicator}` reads the latest saved scan from Postgres (Neon)
  — one indexed query, no Yahoo call — so the page always shows whatever
  data exists, with its "Last updated" time. A scan only runs when the
  user clicks Refresh (`POST …/start`), through the single-flight guarded
  pipeline in `app/services/scheduler.py` (the name is historical - no
  longer a real APScheduler cron; that was removed so the Yahoo Finance
  request budget goes to the twice-daily big-loss alert instead). The
  batched discovery (`RESEARCH_SCAN_NUM_BATCHES`,
  `RESEARCH_SCAN_BATCH_DELAY_SECONDS`) is kept for the on-demand runs:
  the user waits a few extra minutes behind a disabled button, but the
  rate-limit safety is worth more than the wait.
- **The big-loss email alert is triggered externally, by a GitHub
  Actions cron, and sent from the Actions runner, not from the API**
  (`.github/workflows/big-loss-alert.yml` →
  `scripts/send_big_loss_alert.py`, which `POST`s
  `/api/research/volatility/today/alert` to start the scan, polls `GET
  …/today/alert?started_at=` until `app/services/alerts.py` returns the
  rendered email, and sends it over SMTP itself), twice a day at 12:00
  and 16:00 Europe/Zurich. An in-process cron can't do this: on Render's
  free tier the process is asleep between requests, and a cron inside a
  sleeping process never fires. The runner sends the email itself because
  Render's free tier blocks outbound SMTP (ports 25/465/587) — the API's
  own `smtplib` send failed in production before this change. An email
  goes out only when the scan finds at least one match; a failed or
  timed-out scan, a failed send, or a missing secret fails the workflow
  run (GitHub's failed-run notification email is the error signal); an
  empty scan passes quietly. GitHub also disables `schedule` workflows in
  a repo with no commits for 60 days (re-enable from the Actions tab).
- **The rate limiter keys on a constant, not client IP.** Every real
  request arrives via the Next.js frontend's single proxy IP, so per-IP
  keying already bucketed all legitimate traffic together — and since
  Render forwards whatever `X-Forwarded-For` a direct caller sends,
  per-IP keying was also spoofable (a fresh header value = a fresh
  bucket). A single global bucket matches this app's actual traffic shape
  and can't be rotated around.

## Auth

`app/deps.py::verify_api_key` gates every route that spends inference
quota or Yahoo Finance request budget (`/api/analyze`, all of
`/api/research/*`), via `secrets.compare_digest` against `API_KEY`
(header `X-API-Key`). There's no per-user auth — this is a single-tenant
demo API — the key just distinguishes "the frontend" from anyone else
hitting the URL directly.

## Rate limiting

A global `slowapi` limiter (`app/limiter.py`), keyed on a constant string
rather than client IP (see "Key design decisions" above for why). Applies
to every route uniformly rather than per-route decorators, matching the
single-caller (the frontend proxy) traffic shape this API actually sees.
