"""
Seeds/hot-swaps yfinance's process-wide crumb+cookie jar (YfData is a true
singleton - yfinance.data.YfData, one per process, metaclass=SingletonMeta)
so app/services/fundamentals.py's and earnings.py's yfinance calls survive
Yahoo blocking this server's outbound IP at the crumb-fetch endpoint
(yfinance.data.YfData._get_crumb_csrf -> query2.finance.yahoo.com/v1/
test/getcrumb, which raises YFRateLimitError). A fresh process has no
cached crumb (never persisted to disk, and yfinance never re-fetches once
it has one - no expiry check in _get_crumb_csrf), so every request on a
freshly-restarted process would re-attempt that same blocked fetch and
fail immediately, even though the actual data endpoints (quoteSummary,
etc.) were never even reached to know if THEY'D have worked.

Seeding a crumb+cookie pair captured from a DIFFERENT, unblocked network
lets this process skip the blocked fetch entirely. NOT guaranteed to work:
Yahoo may bind a crumb/cookie pair to the IP that requested it, in which
case using it from a different IP fails too. If the exact same
YFRateLimitError keeps happening after setting these, that's the answer:
IP-bound, and this doesn't help.

Same mechanism as `financial-research-api`'s copy of this module (used
there for the Swiss market-scan yfinance calls instead) - each process
has its own independent yfinance singleton, so each app needs its own
seeded crumb; this isn't shared state between the two repos, just shared
logic, deliberately duplicated (same "ported, not imported" convention
used elsewhere in this fleet).

Refresh via scripts/refresh_yf_crumb.py (run manually whenever the seeded
crumb/cookies go stale - see that script's own docstring), which calls
reseed_yf_session() below via the /api/update-yf-crumb admin endpoint to
hot-swap the running process's crumb with no restart needed.
"""

import json
import logging
import os

from yfinance.data import YfData

logger = logging.getLogger(__name__)


def seed_yf_session_from_env():
    seed_crumb = os.environ.get("YF_SEED_CRUMB")
    seed_cookies_json = os.environ.get("YF_SEED_COOKIES")
    if not seed_crumb or not seed_cookies_json:
        return
    data = YfData()
    if data._crumb:
        return  # already seeded or already fetched its own this process - don't clobber either
    for name, value in json.loads(seed_cookies_json).items():
        data._session.cookies.set(name, value)
    data._crumb = seed_crumb
    logger.info("Seeded yfinance crumb/cookies from YF_SEED_CRUMB/YF_SEED_COOKIES (Yahoo crumb-fetch workaround)")


def reseed_yf_session(crumb: str, cookies: dict) -> None:
    """Hot-swaps the running process's yfinance crumb/cookie jar with a
    freshly captured pair - called from app/routers/admin.py's /api/
    update-yf-crumb endpoint, itself called by scripts/refresh_yf_crumb.py
    (see that script's own docstring). Unlike seed_yf_session_from_env
    above, this ALWAYS overwrites (no "already seeded, don't clobber"
    guard) - the whole point is refreshing an already-seeded, now-stale
    crumb without restarting the process at all."""
    data = YfData()
    data._session.cookies.clear()
    for name, value in cookies.items():
        data._session.cookies.set(name, value)
    data._crumb = crumb
    logger.info("Hot-reseeded yfinance crumb/cookies via reseed_yf_session (no restart)")
