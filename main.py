import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.routers import admin, analyze, health
from app.services import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Shared client so every /api/analyze request reuses one connection pool
    # instead of paying TCP/TLS setup on each call.
    await inference.start_client()
    try:
        yield
    finally:
        await inference.stop_client()


app = FastAPI(title="Multi-Model Financial RAG Reasoning Engine", lifespan=lifespan)


def _rate_limit_key(request: Request) -> str:
    # Not client IP: every real request arrives via the Next.js proxy on a
    # single shared Vercel egress IP, so per-IP keying already bucketed all
    # legitimate traffic together - and since Render forwards whatever
    # X-Forwarded-For a direct caller sends, per-IP keying was also
    # attacker-spoofable (a fresh header value = a fresh bucket). A single
    # global bucket matches this app's actual traffic shape and can't be
    # rotated around.
    return "global"


# Applies to every route via default_limits, no per-route decorators needed.
# The real cost here is per-request HF inference + yfinance calls, so the
# limit is deliberately tight - this endpoint is not meant for bursts.
limiter = Limiter(key_func=_rate_limit_key, default_limits=["10/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(admin.router)
