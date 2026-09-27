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

## Documentation

- [`PROJECT.md`](PROJECT.md) — architecture, key design decisions, auth,
  rate limiting

## Stack

- **FastAPI** — the API itself
- **SQLAlchemy 2.0** + **Postgres** (Neon) — persistence for the two
  on-demand research scans
- **pytest** — run locally before opening a PR (no CI configured currently, see Deployment)

Talks to a Hugging Face Inference endpoint for the fine-tuned model,
Google News RSS for live headlines, and yfinance for market
fundamentals/earnings — all keyless except the HF token.

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
