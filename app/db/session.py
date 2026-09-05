from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import DATABASE_CONNECTION_STRING


def _normalized_url(raw: str) -> str:
    """Neon (like most Postgres providers) hands out a plain postgresql://
    URL, which SQLAlchemy defaults to opening with psycopg2 - not what's
    installed here (see requirements.txt: psycopg[binary], psycopg3, chosen
    because psycopg2-binary ships no prebuilt wheel for this Python
    version). Rewriting the scheme to postgresql+psycopg:// picks psycopg3
    instead, without requiring the value stored in .env/Render to be
    hand-edited to match.
    """
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://"):]
    return raw


# Lazy, not created at import time: most of this app's test suite never
# touches the DB layer at all and shouldn't need DATABASE_CONNECTION_STRING
# set just to import a module that happens to import this one.
_engine = None
_SessionLocal: sessionmaker | None = None


def _session_factory() -> sessionmaker:
    global _engine, _SessionLocal
    if _SessionLocal is None:
        if not DATABASE_CONNECTION_STRING:
            raise RuntimeError(
                "DATABASE_CONNECTION_STRING is not set - required for the "
                "research-scan persistence layer (app/db, app/services/scan_persistence.py)."
            )
        _engine = create_engine(_normalized_url(DATABASE_CONNECTION_STRING), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _SessionLocal


def get_session() -> Session:
    """One-off session for non-request code (the scheduler's background
    jobs) - caller is responsible for closing it (context manager or
    try/finally)."""
    return _session_factory()()
