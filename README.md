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
  persistence for the two scheduled research scans (`app/db/`,
  `app/services/scan_persistence.py`) — same pattern as
  `portfolio-manager-backend`'s DB layer, adapted where noted (see that
  module's own comments)
- **APScheduler** — in-process cron for those same two scans (`app/
  services/scheduler.py`) — no external trigger (GitHub Actions, etc.)
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
                              HF_INFERENCE_URL (see below)
  routers/
    health.py, analyze.py, admin.py   Thin HTTP layer only
```

## Key design decisions

- **`HF_INFERENCE_URL` is a mutable, in-memory global**, updatable via
  `POST /api/update-inference-url` — this exists so a Colab-hosted model
  (behind an ngrok tunnel that gets a new URL on every restart) can push
  its current URL without a redeploy. Deliberately not made more
  robust/persistent: this app runs a single process/instance and doesn't
  need to scale, so the added complexity of a shared store wouldn't earn
  its keep.
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
  earnings + news), but no v4 model is live yet.** `HF_INFERENCE_URL`
  still points at whatever model is currently deployed via
  `/api/update-inference-url` — until `financial-sentiment-model`'s
  v4 model is trained, evaluated, and that URL is repointed, this app is
  serving the new prompt shape to an OLDER model that was never trained on
  it. `analyze_with_hf` parses the new `answer` field with `.get`, not
  indexing, specifically so an older model's response (which has no
  `answer` key at all) degrades to an omitted field rather than a 500 —
  but the model's actual JSON output quality against the new prompt is
  untested until the cutover happens. Do not treat this as "the analyst
  pipeline is live" until that model swap is confirmed.
- **The rebound and volatility-indicator research tables are scanned on a
  schedule (daily / monthly), not live on button click.** Added at the
  user's explicit request to cut Yahoo Finance call volume, since neither
  table's underlying data changes meaningfully more often than that. An
  in-process APScheduler cron (`app/services/scheduler.py`) runs each
  scan in the early morning, batches the ~150-ticker universe discovery
  into several groups with a delay between them (on top of, not instead
  of, `swiss_universe.filter_domestic`'s existing per-ticker pacing —
  affordable for an unattended overnight job in a way it isn't for a
  live one), and persists the result to Neon Postgres
  (`app/services/scan_persistence.py`). `GET /api/research/volatility/
  {rebound,indicator}` just reads the latest saved run — no scan runs on
  request anymore for these two. Deliberately NOT an external trigger
  (GitHub Actions cron, Render Cron Jobs): this repo already tried a
  GitHub Actions cron for a different purpose (keeping Render's free
  tier warm) and it proved unreliable in practice (see git history:
  `ci/remove-keep-alive-workflow` — a `*/10` schedule silently went 24+
  minutes without firing). A catch-up check runs on every app startup
  (`scheduler._is_rebound_stale`/`_is_indicator_stale`) so a missed
  scheduled firing (a redeploy landing exactly on the hour) just runs a
  bit late instead of being silently skipped. The "today" big-loss
  screener is untouched — it stays fully live/on-demand, since intraday
  data has no meaningful cache window.
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

The scheduled-scan feature additionally needs `DATABASE_CONNECTION_STRING`
(a Neon Postgres connection string) and a migration:

```bash
python -m alembic upgrade head
```

Tests never touch the real database or trigger a real scan — see
`conftest.py`'s `RESEARCH_SCHEDULER_DISABLED` guard and `tests/
test_scan_persistence.py`'s in-memory-SQLite fixture.

## Tests

```bash
pytest
```

## Deployment

Render, via the `Dockerfile` (pinned base image digest, non-root user,
healthcheck against `/health`). No CI is currently configured (the
`.github/workflows/` directory was removed) — run `pytest` locally
before opening a PR.

## Contributing

No dedicated `CONTRIBUTING.md` yet, but the fleet-wide default (see the
`AGENTS.md` one directory up, outside this repo, alongside its sibling
repos) applies: **branch + PR, never push directly to `main`/`master`.**
