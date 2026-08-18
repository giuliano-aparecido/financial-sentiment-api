"""
Automates the Yahoo crumb-refresh workaround documented in
app/services/swiss_universe.py's _seed_yf_session_from_env(). Run on a
schedule from a machine whose outbound IP ISN'T blocked by Yahoo (see
.github/workflows/refresh-yf-crumb.yml - GitHub Actions runners, not
Render, where this same fetch is what's actually blocked) - fetches a
fresh crumb+cookie pair, then pushes it into the target Render service's
YF_SEED_CRUMB/YF_SEED_COOKIES env vars via Render's API and triggers a
redeploy so the new process picks them up.

Why this has to run somewhere other than Render: the whole reason
YF_SEED_CRUMB/YF_SEED_COOKIES exist is that Render's own outbound IP is
blocked specifically at Yahoo's crumb-fetch endpoint (see that module's
docstring) - a refresh attempted FROM Render would just hit the same
block. This script is deliberately meant to run on different
infrastructure. GitHub Actions runners aren't guaranteed to stay unblocked
forever either (they're well-known cloud IP ranges too) - if this starts
failing with the same YFRateLimitError this workaround exists to route
around, that's the signal this approach itself needs to change (a
different unblocked runner, a proxy, etc.), not a bug in this script.

Uses Render's per-key env var endpoint (PUT /v1/services/{id}/env-vars/
{key}), not the bulk env-vars endpoint - the bulk one REPLACES the
service's entire env var set, which would silently wipe every other
credential (HF_TOKEN, GEMINI_API_KEY, etc.) - a real, easy-to-make mistake
this deliberately avoids.

Required environment variables (set as GitHub Actions repo secrets - see
the companion workflow file):
    RENDER_API_KEY     - a Render API key (Account Settings -> API Keys)
    RENDER_SERVICE_ID  - the target service's ID (Render dashboard URL,
                          or `curl -H "Authorization: Bearer $RENDER_API_KEY"
                          https://api.render.com/v1/services`)

Usage:
    pip install yfinance requests
    export RENDER_API_KEY=...
    export RENDER_SERVICE_ID=...
    python scripts/refresh_yf_crumb.py
"""

import json
import os
import sys

import requests
import yfinance as yf
from yfinance.data import YfData

RENDER_API_BASE = "https://api.render.com/v1"


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


def set_render_env_var(api_key: str, service_id: str, key: str, value: str) -> None:
    response = requests.put(
        f"{RENDER_API_BASE}/services/{service_id}/env-vars/{key}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"value": value},
        timeout=30,
    )
    response.raise_for_status()


def trigger_render_deploy(api_key: str, service_id: str) -> None:
    # Explicit, rather than relying on Render's env-var-update-triggers-a-
    # deploy behavior implicitly - redeploying this service is documented
    # as safe (see swiss_universe.py's _seed_yf_session_from_env comment
    # and the memory this session left about it): the seeded env vars
    # persist across restarts, so a redeploy no longer risks losing the
    # only working crumb the way it did before that mechanism existed.
    response = requests.post(
        f"{RENDER_API_BASE}/services/{service_id}/deploys",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={},
        timeout=30,
    )
    response.raise_for_status()


def main() -> None:
    api_key = os.environ.get("RENDER_API_KEY")
    service_id = os.environ.get("RENDER_SERVICE_ID")
    if not api_key or not service_id:
        sys.exit("RENDER_API_KEY and RENDER_SERVICE_ID must both be set.")

    print("Fetching a fresh crumb+cookie pair from this runner...")
    crumb, cookies = fetch_fresh_crumb_and_cookies()
    print(f"Got crumb (len={len(crumb)}) and {len(cookies)} cookies.")

    print("Pushing YF_SEED_CRUMB to Render...")
    set_render_env_var(api_key, service_id, "YF_SEED_CRUMB", crumb)
    print("Pushing YF_SEED_COOKIES to Render...")
    set_render_env_var(api_key, service_id, "YF_SEED_COOKIES", json.dumps(cookies))

    print("Triggering a redeploy so the new process picks up the refreshed values...")
    trigger_render_deploy(api_key, service_id)

    print("Done. The service will restart with the fresh crumb/cookies shortly.")


if __name__ == "__main__":
    main()
