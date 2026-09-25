# Multi-Model Financial RAG Reasoning Engine — API

A FastAPI backend for a financial-news sentiment/reasoning demo: takes a
free-text (or `$TICKER`-cashtagged) query, fetches live news for the
ticker, and runs a fine-tuned LLM's chain-of-thought reasoning against it
via a Hugging Face Inference endpoint. Serves
[`financial-sentiment-web`](https://github.com/GiulianoAparecido/financial-sentiment-web).

Built as a portfolio/curriculum project — hardened for the practice of
doing it properly, not because it needs to scale or handle real traffic.

## Stack

- **FastAPI** + **uvicorn**
- **httpx** — shared `AsyncClient` (started in the app's lifespan, not
  recreated per request) for both the HF inference call and the news fetch
- **feedparser** — Google News RSS, no API key needed
- **yfinance** — market fundamentals/earnings (keyless; see `app/services/
  fundamentals.py`/`earnings.py`/`valuation.py`)
- **slowapi** — global rate limit
- **SQLAlchemy 2.0** + **Alembic** + **psycopg3** — Postgres (Neon)
  persistence for the two on-demand research scans (`app/db/`,
  `app/services/scan_persistence.py`) — same pattern as
  `portfolio-manager-backend`'s DB layer, adapted where noted (see that
  module's own comments)
- **pytest** — run locally before opening a PR (no CI configured currently, see Deployment)

## Architecture

```
main.py                       App creation, middleware, lifespan (shared
                              httpx client), router mounting - deliberately
                              thin
app/
  config.py                   All env-var reads in one place
  models.py                   Pydantic request models
  deps.py                     API-key auth dependency
  services/
    ticker.py                 $CASHTAG extraction (falls back to a
                              capitalized-word heuristic if no cashtag)
    parsing.py                 Brace-balanced JSON extraction from raw
                              LLM output (handles trailing model chatter,
                              markdown fences, special tokens)
    news.py                    Google News RSS fetch, with a timeout
                              (feedparser's own url-fetching has none)
    fundamentals.py            yfinance current-price/P-E/dividend/52wk
                              fetch + "Current Market Data" block renderer
    valuation.py               Scenario-weighted 2-stage DCF valuation
                              (deterministic math, never LLM-generated)
                              + block renderer
    earnings.py                yfinance last-quarter revenue/EPS/next-
                              earnings-date fetch + block renderer
    inference.py                HF inference call + response parsing;
                              also holds the mutable in-memory
                              per-model inference URLs (see below)
  routers/
    health.py, analyze.py, admin.py   Thin HTTP layer only
```

## Key design decisions

- **Multiple models behind one API, with no model names in code.**
  `/api/analyze?model=<name>` picks which fine-tuned model answers;
  `DEFAULT_MODEL` (default `llama`) is used when the param is absent,
  and a name that isn't registered is a 400. Models are registered two
  ways, neither needing a code change:
  - **Env var convention at startup:** every `<NAME>_INFERENCE_URL`
    env var registers model `<name>` (lowercased) — `KIM_INFERENCE_URL`
    makes `?model=kim` work. `DEFAULT_MODEL` must have one, or the app
    refuses to start.
  - **At runtime:** `POST /api/update-inference-url` with a `model`
    that isn't registered yet registers it (see next bullet).
  `GET /health` lists what's registered. All models share the same auth
  (`HF_TOKEN` as bearer), host allowlist, and rate limit.
- **The per-model inference URLs are mutable, in-memory globals**,
  updatable via `POST /api/update-inference-url` (`{"url": ...,
  "model": ...}`, `model` defaulting to `DEFAULT_MODEL`) — this exists
  so a Colab-hosted model (behind an ngrok tunnel that gets a new URL on
  every restart) can push its current URL without a redeploy.
  Deliberately not made more robust/persistent: this app runs a single
  process/instance and doesn't need to scale, so the added complexity
  of a shared store wouldn't earn its keep.
- **The update-inference-url host allowlist** (`ALLOWED_INFERENCE_HOST_SUFFIXES`)
  matches on an exact host or a `.`-bounded suffix, not a bare
  `str.endswith()` — a plain `endswith` would accept a registered
  look-alike domain like `evil-huggingface.co`, and since the resolved
  URL receives the HF bearer token on every subsequent request, that
  bug would have been a token-exfiltration path, not just a
  bad-config one.
- **Valuation is always deterministic, code-only math — never LLM-generated.**
  `valuation.py` replaced an earlier Graham Number implementation
  (`sqrt(22.5 x EPS x book value/share)`) that was confirmed live to be
  badly broken for asset-light, buyback-heavy companies (Apple showed
  "725% overvalued" purely because its book value/share is tiny — Graham
  Number treats book value as a proxy for a company's worth, which fails
  hard when most of the value is intangible). The current model instead
  classifies each company into one of four valuation bases — FCF
  (asset-heavy industrials), EPS/Net Income (tech/growth/platform),
  Dividends (mature cash-cow/REIT-style payers), or Revenue (early-stage
  unprofitable growth) — since a single metric can't meaningfully value
  both a bank and a pre-profit growth company. The classified metric is
  projected across a 2-stage (years 1-5, years 6-10), 3-scenario
  (Normal/Best/Worst probability-weighted) growth model with an
  exit-multiple terminal value, discounted at a flat rate. All computed in
  Python from `fundamentals.py`'s yfinance data, not asked of the model.
  Missing/non-positive inputs for the classified basis render "Not
  applicable" (a real, expected outcome); a failed fundamentals fetch
  renders "Data unavailable." — the two are deliberately distinct strings
  (see the tests) so the eventual model can learn to tell "no defined
  value" from "couldn't fetch anything." Known limitation: extreme-multiple
  growth stocks (e.g. Tesla, trailing P/E ~300) can still show implausibly
  high "overvalued" percentages even after this change, since a
  disciplined, non-circular growth assumption can't fully bridge a
  razor-thin trailing EPS — see `app/services/valuation.py`'s module
  docstring for the two earlier, rejected attempts at fixing this and why
  they made it worse.
- **`/api/analyze` sends the full v4 prompt (market data + valuation +
  earnings + news), but no v4 model is live yet.** The default model's
  inference URL still points at whatever is currently deployed via
  `/api/update-inference-url` — until `financial-sentiment-model`'s
  v4 model is trained, evaluated, and that URL is repointed, this app is
  serving the new prompt shape to an OLDER model that was never trained on
  it. `analyze_with_hf` parses the new `answer` field with `.get`, not
  indexing, specifically so an older model's response (which has no
  `answer` key at all) degrades to an omitted field rather than a 500 —
  but the model's actual JSON output quality against the new prompt is
  untested until the cutover happens. Do not treat this as "the analyst
  pipeline is live" until that model swap is confirmed.
- **The rebound and volatility-indicator research tables are on-demand,
  served from the last persisted run.** `GET /api/research/volatility/
  {rebound,indicator}` reads the latest saved scan from Neon Postgres
  (`app/services/scan_persistence.py`) — one indexed query, no Yahoo
  call — so the page always shows whatever data exists, with its "Last
  updated" time. A scan only runs when the user clicks Refresh (`POST
  …/start`), through the single-flight guarded pipeline in
  `app/services/scheduler.py` (the name is historical). The daily/
  monthly APScheduler cron and its startup catch-up that used to drive
  these two were removed on 2026-09-17: the Yahoo Finance request budget
  is the scarce resource, and it now goes to the twice-daily big-loss
  alert (next bullet) — the old startup catch-up would have launched a
  full ~150-ticker rebound scan at the exact moment the alert woke the
  process. The batched discovery (`RESEARCH_SCAN_NUM_BATCHES`,
  `RESEARCH_SCAN_BATCH_DELAY_SECONDS`) is kept for the on-demand runs:
  the user waits a few extra minutes behind a disabled button, but the
  rate-limit safety is worth more than the wait.
- **The big-loss email alert is triggered externally, by cron-job.org
  dispatching a GitHub Actions workflow** (cron-job.org → GitHub
  `workflow_dispatch` API → `.github/workflows/big-loss-alert.yml` →
  `POST /api/research/volatility/today/alert` → `app/services/alerts.py`),
  twice a day at 12:00 and 16:00 Europe/Zurich (the evening slot moved
  from 17:30 to 16:00 on 2026-09-18 — the owner wants it landed by 17:00
  at the latest). An in-process cron
  can't do this: on Render's free tier the process is asleep between
  requests, and a cron inside a sleeping process never fires. GitHub's
  own `schedule:` trigger was used until 2026-09-25 and dropped: it fired
  4-5 hours late every day in this repo (the 12:00 slot landed ~17:00,
  the 16:00 slot ~20:20), which no gating can fix. cron-job.org fires
  on the minute, handles DST itself (job timezone Europe/Zurich), and
  only has to make a fast API call — the workflow keeps the curl retries
  that ride out Render's cold start. Setup: a fine-grained GitHub token
  scoped to this repo only with **Actions: Read and write**, and two
  cron-job.org jobs (Mon-Fri, 12:00 and 16:00, timezone Europe/Zurich)
  doing `POST https://api.github.com/repos/GiulianoAparecido/financial-sentiment-api/actions/workflows/big-loss-alert.yml/dispatches`
  with headers `Authorization: Bearer <token>`, `Accept:
  application/vnd.github+json` and body `{"ref":"master"}` (expects
  HTTP 204). Turn on both jobs' failure notifications in cron-job.org:
  an expired token fails the dispatch before any Actions run exists,
  and cron-job.org auto-disables a job after repeated failures.
  An email goes out
  only when the scan finds at least one match (each ticker linked to
  its Yahoo Finance quote page, like the web table); an empty or failed
  scan is logged on the API side, not mailed. That means a quiet inbox
  is ambiguous — check cron-job.org's execution history, the workflow's
  run history in the Actions tab, and the Render logs if you doubt it
  fired. A failed `curl` only shows
  up as GitHub's failed-run notification email. The
  workflow needs two repository secrets, `RAG_API_KEY` and `RAG_API_URL`
  (the same ones `scripts/refresh_yf_crumb.py` uses — same server, same
  single API key); the API needs `SMTP_*` and `ALERT_EMAIL_*` (see
  `.env.example`).
- **The rate limiter keys on a constant, not client IP.** Every real
  request arrives via the Next.js frontend's single proxy IP, so per-IP
  keying already bucketed all legitimate traffic together — and since
  Render forwards whatever `X-Forwarded-For` a direct caller sends,
  per-IP keying was also spoofable (a fresh header value = a fresh
  bucket). A single global bucket matches this app's actual traffic
  shape and can't be rotated around.

## Local development

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --reload
```

`.env`/environment needs at minimum `API_KEY` and `HF_TOKEN` — see
`app/config.py` for the full list (model selection, allowed inference
host suffixes, etc.). `.env` is loaded automatically (`python-dotenv`) if
present — copy `.env.example` to start.

The persisted research scans additionally need `DATABASE_CONNECTION_STRING`
(a Neon Postgres connection string) and a migration:

```bash
python -m alembic upgrade head
```

Tests never touch the real database or trigger a real scan — see
`tests/test_scan_persistence.py`'s in-memory-SQLite fixture.

## Tests

```bash
pytest
```

## Deployment

Render, via the `Dockerfile` (pinned base image digest, non-root user,
healthcheck against `/health`). No CI is configured — the only workflow in
`.github/workflows/` is the big-loss alert trigger, not a test runner — run `pytest` locally
before opening a PR.

## Contributing

No dedicated `CONTRIBUTING.md` yet, but the fleet-wide default (see the
`AGENTS.md` one directory up, outside this repo, alongside its sibling
repos) applies: **branch + PR, never push directly to `main`/`master`.**
