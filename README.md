# Multi-Model Financial RAG Reasoning Engine — API

*An experimental project exploring agentic coding workflows with Claude Code.*

A FastAPI backend for a financial-news sentiment/reasoning demo: takes a
free-text (or `$TICKER`-cashtagged) query, fetches live news for the
ticker, and runs a fine-tuned LLM's chain-of-thought reasoning against it
via a Hugging Face Inference endpoint. Serves
[`financial-sentiment-web`](https://github.com/GiulianoAparecido/financial-sentiment-web).

Hardened for correctness and security even though it's a single-instance
app that doesn't need to scale.

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

- **Multiple models behind one API, no model names hardcoded.**
  `/api/analyze?model=<name>` picks the model; every `<NAME>_INFERENCE_URL`
  env var registers one at startup, and `POST /api/update-inference-url`
  can add or repoint one at runtime (useful for a Colab-hosted model behind
  an ngrok tunnel that gets a new URL on every restart).
- **The inference-URL host allowlist matches on exact host or a
  `.`-bounded suffix, not `str.endswith()`** — a bare `endswith` would
  accept a look-alike domain like `evil-huggingface.co`, and the resolved
  URL receives the HF bearer token on every request after that.
- **Valuation is deterministic, code-only math, never LLM-generated.**
  Replaced an earlier Graham Number implementation that was badly broken
  for asset-light, buyback-heavy companies (it showed Apple as "725%
  overvalued" purely because its book value/share is tiny). Each company
  is classified into one of four valuation bases (FCF, EPS, Dividends, or
  Revenue) and projected through a 2-stage, 3-scenario growth model —
  since a single metric can't value both a bank and a pre-profit
  growth company.
- **The rebound/volatility-indicator research tables are on-demand, not
  scheduled.** They read the last persisted scan from Postgres; a scan
  only runs when the user clicks Refresh. The old daily/monthly cron was
  removed so the Yahoo Finance request budget goes to the twice-daily
  big-loss alert instead.
- **The big-loss email alert is triggered externally by a GitHub Actions
  cron, and sent from the Actions runner, not the API.** An in-process
  cron can't do it — Render's free tier sleeps the process between
  requests — and the runner has to send the email itself because Render's
  free tier also blocks outbound SMTP. The workflow starts the scan via
  the API, polls for the rendered email, then sends it over SMTP directly.
- **The rate limiter keys on a constant, not client IP.** All real traffic
  arrives via the frontend's single proxy IP, so per-IP keying already
  bucketed everything together — and was spoofable via `X-Forwarded-For`
  besides.

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
`.github/workflows/` is the big-loss alert cron, not a test runner — run `pytest` locally
before opening a PR.

## Contributing

No dedicated `CONTRIBUTING.md` yet, but the fleet-wide default (see the
`AGENTS.md` one directory up, outside this repo, alongside its sibling
repos) applies: **branch + PR, never push directly to `main`/`master`.**

## License

Dual-licensed under either of

- [MIT license](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
