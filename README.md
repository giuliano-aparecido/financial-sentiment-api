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
- **pytest** — 53 tests, run via GitHub Actions on every push/PR

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
host suffixes, etc.).

## Tests

```bash
pytest
```

## Deployment

Render, via the `Dockerfile` (pinned base image digest, non-root user,
healthcheck against `/health`). CI (`.github/workflows/test.yml`) runs
the test suite on every push/PR to `master`.

## Contributing

No dedicated `CONTRIBUTING.md` yet, but the fleet-wide default (see the
`CLAUDE.md` one directory up, outside this repo, alongside its sibling
repos) applies: **branch + PR, never push directly to `main`/`master`.**
