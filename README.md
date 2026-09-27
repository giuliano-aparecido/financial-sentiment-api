# Multi-Model Financial RAG Reasoning Engine — API

*An experimental project exploring agentic coding workflows with Claude Code.*
*Also an experiment in LoRA fine-tuning and self-hosting a small LLM.*

A FastAPI backend for a financial-news sentiment/reasoning demo: takes a
free-text (or `$TICKER`-cashtagged) query, fetches live news for the
ticker, and runs a fine-tuned LLM's chain-of-thought reasoning against it
via a Hugging Face Inference endpoint. Serves
[`financial-sentiment-web`](https://github.com/GiulianoAparecido/financial-sentiment-web).

Hardened for correctness and security even though it's a single-instance
app that doesn't need to scale.

## Stack

- **FastAPI** — the API itself
- **SQLAlchemy 2.0** + **Postgres** (Neon) — persistence for the two
  on-demand research scans
- **pytest** — run locally before opening a PR (no CI configured currently, see Deployment)

Talks to a Hugging Face Inference endpoint for the fine-tuned model,
Google News RSS for live headlines, and yfinance for market
fundamentals/earnings — all keyless except the HF token.

## Architecture

`app/services/` holds the business logic (news fetching, DCF valuation
math, LLM inference call + response parsing); `app/routers/` is a thin
HTTP layer with no logic of its own; `app/db/` plus
`services/scan_persistence.py` handle Postgres persistence for the two
on-demand research scans. `main.py` wires it together (middleware, the
shared httpx client's lifespan, router mounting) and stays thin by design.

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
