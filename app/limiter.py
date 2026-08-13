from fastapi import Request
from slowapi import Limiter


def rate_limit_key(request: Request) -> str:
    # Not client IP: every real request arrives via the Next.js proxy on a
    # single shared Vercel egress IP, so per-IP keying already bucketed all
    # legitimate traffic together - and since Render forwards whatever
    # X-Forwarded-For a direct caller sends, per-IP keying was also
    # attacker-spoofable (a fresh header value = a fresh bucket). A single
    # global bucket matches this app's actual traffic shape and can't be
    # rotated around.
    return "global"


# Factored out of main.py (which used to define this inline) so router
# modules can import `limiter` for per-route @limiter.limit(...) overrides
# (see app/routers/research.py) without a circular import back to main.py.
#
# default_limits=["10/minute"] applies to every route that doesn't
# override it - sized for this app's normal per-request cost (one HF
# inference call + a handful of yfinance calls), not for
# research.py's volatility scan, which makes ~150+ yfinance calls per run
# and gets its own much stricter override.
limiter = Limiter(key_func=rate_limit_key, default_limits=["10/minute"])
