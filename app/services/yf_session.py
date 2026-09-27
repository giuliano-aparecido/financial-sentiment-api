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
    freshly captured pair. Unlike seed_yf_session_from_env above, this
    ALWAYS overwrites (no "already seeded, don't clobber" guard) - the
    whole point is refreshing an already-seeded, now-stale crumb without
    restarting the process at all."""
    data = YfData()
    data._session.cookies.clear()
    for name, value in cookies.items():
        data._session.cookies.set(name, value)
    data._crumb = crumb
    logger.info("Hot-reseeded yfinance crumb/cookies via reseed_yf_session (no restart)")
