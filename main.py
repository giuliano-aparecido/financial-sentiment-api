import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.limiter import limiter
from app.routers import admin, analyze, health
from app.services import inference, yf_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Shared client so every /api/analyze request reuses one connection pool
    # instead of paying TCP/TLS setup on each call.
    await inference.start_client()
    # Seeds the process-wide yfinance crumb/cookie singleton before any
    # request-path code makes its own unseeded fetch - see
    # app/services/yf_session.py's docstring for why. Wrapped defensively:
    # this reads externally-set env vars (YF_SEED_COOKIES is JSON) and must
    # never be able to take down the whole app's startup just because that
    # value is malformed or missing; degrading to "no seed this boot" is far
    # better than the entire service failing to come up.
    try:
        yf_session.seed_yf_session_from_env()
    except Exception:
        logging.getLogger(__name__).exception("Failed to seed yfinance crumb/cookies at startup - continuing without a seed")
    try:
        yield
    finally:
        await inference.stop_client()


app = FastAPI(title="Multi-Model Financial RAG Reasoning Engine", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(admin.router)
