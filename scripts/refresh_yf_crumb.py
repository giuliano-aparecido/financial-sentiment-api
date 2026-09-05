"""Automates the Yahoo crumb-refresh workaround documented in
app/services/swiss_universe.py's seed_yf_session_from_env()/reseed_
yf_session(). Run from a machine whose outbound IP ISN'T blocked by
Yahoo (originally scheduled daily via .github/workflows/
refresh-yf-crumb.yml on a GitHub Actions runner, not Render, where this
same fetch is what's actually blocked - that workflow was removed along
with the rest of .github/workflows/, so this now has to be run manually
until/unless a replacement schedule is set up) - fetches a fresh
crumb+cookie pair, then hot-swaps it into the running Render service via
POST /api/update-yf-crumb.

Why this has to run somewhere other than Render: the whole reason a
seeded crumb is needed is that Render's own outbound IP is blocked
specifically at Yahoo's crumb-fetch endpoint (see swiss_universe.py's
module-level comment) - a refresh attempted FROM Render would just hit
the same block. This script is deliberately meant to run on different
infrastructure. GitHub Actions runners aren't guaranteed to stay
unblocked forever either (they're well-known cloud IP ranges too) - if
this starts failing with the same YFRateLimitError this workaround
exists to route around, that's the signal this approach itself needs to
change (a different unblocked runner, a proxy, etc.), not a bug in this
script.

Revised 2026-08-20 (previously pushed YF_SEED_CRUMB/YF_SEED_COOKIES to
Render's env vars via Render's API and explicitly triggered a redeploy
on every refresh - at the user's explicit request, "I dont want to
always trigger and redeploy the app", switched to calling the running
process directly instead): POSTs to /api/update-yf-crumb, which hot-
swaps the process-wide yfinance crumb/cookie singleton with no restart
at all - see app/routers/admin.py's own docstring for that endpoint.
YF_SEED_CRUMB/YF_SEED_COOKIES still exist as Render env vars (unchanged
by this script now) purely as the cold-start fallback for whenever the
process DOES restart for an unrelated reason (a real code deploy) - this
script never kept them in sync even when it ran on a schedule, so update
them by hand in Render's dashboard occasionally if that fallback staying
reasonably fresh matters to you. With no scheduled run at all now, the
hot-swapped crumb this script pushes via /api/update-yf-crumb will
itself go stale between manual runs - see swiss_universe.py's own
comment for what that looks like when it happens.

Required environment variables (previously set as GitHub Actions repo
secrets for the now-removed scheduled workflow; set them in your own
shell/CI when running this manually):
    RAG_API_URL   - the deployed API's base URL (same value financial-
                    sentiment-web's proxy routes use)
    RAG_API_KEY   - the API key /api/update-yf-crumb is gated behind
                    (same one used for every other authenticated route)

Usage:
    pip install yfinance requests
    export RAG_API_URL=...
    export RAG_API_KEY=...
    python scripts/refresh_yf_crumb.py
"""

import os
import sys

import requests
import yfinance as yf
from yfinance.data import YfData


def fetch_fresh_crumb_and_cookies() -> tuple[str, dict]:
    # Any real yfinance call forces a crumb fetch if one isn't cached yet -
    # NESN.SW (Nestle) chosen arbitrarily, just needs to be a valid,
    # actively-traded ticker so the call succeeds cleanly.
    yf.Ticker("NESN.SW").info
    data = YfData()
    if not data._crumb:
        raise RuntimeError(
            "yfinance didn't obtain a crumb - this runner's outbound IP may "
            "itself be blocked by Yahoo now (see this script's own docstring)."
        )
    return data._crumb, dict(data._session.cookies)


def push_crumb_to_running_service(api_url: str, api_key: str, crumb: str, cookies: dict) -> None:
    response = requests.post(
        f"{api_url.rstrip('/')}/api/update-yf-crumb",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        json={"crumb": crumb, "cookies": cookies},
        timeout=30,
    )
    response.raise_for_status()


def main() -> None:
    api_url = os.environ.get("RAG_API_URL")
    api_key = os.environ.get("RAG_API_KEY")
    if not api_url or not api_key:
        sys.exit("RAG_API_URL and RAG_API_KEY must both be set.")

    print("Fetching a fresh crumb+cookie pair from this runner...")
    crumb, cookies = fetch_fresh_crumb_and_cookies()
    print(f"Got crumb (len={len(crumb)}) and {len(cookies)} cookies.")

    print(f"Hot-swapping the running service's crumb via POST {api_url}/api/update-yf-crumb ...")
    push_crumb_to_running_service(api_url, api_key, crumb, cookies)

    print("Done. No redeploy needed - the running process picked up the fresh crumb immediately.")


if __name__ == "__main__":
    main()
