import json

import app.services.yf_session as yf_session_module


class _FakeCookies(dict):
    def set(self, name, value):
        self[name] = value


class _FakeSession:
    def __init__(self):
        self.cookies = _FakeCookies()


class _FakeYfData:
    def __init__(self, crumb=None):
        self._crumb = crumb
        self._session = _FakeSession()


def test_seed_yf_session_from_env_noop_when_unset(monkeypatch):
    monkeypatch.delenv("YF_SEED_CRUMB", raising=False)
    monkeypatch.delenv("YF_SEED_COOKIES", raising=False)
    # Asserts YfData() is never even constructed - not just "did nothing to
    # it" - so the normal (no env vars set) path has zero overhead/side
    # effects on the real yfinance singleton.
    monkeypatch.setattr(yf_session_module, "YfData", lambda: (_ for _ in ()).throw(
        AssertionError("YfData() should not be called when env vars are unset")))
    yf_session_module.seed_yf_session_from_env()


def test_seed_yf_session_from_env_seeds_crumb_and_cookies(monkeypatch):
    monkeypatch.setenv("YF_SEED_CRUMB", "test-crumb-123")
    monkeypatch.setenv("YF_SEED_COOKIES", json.dumps({"A1": "abc", "A3": "def"}))
    fake = _FakeYfData()
    monkeypatch.setattr(yf_session_module, "YfData", lambda: fake)

    yf_session_module.seed_yf_session_from_env()

    assert fake._crumb == "test-crumb-123"
    assert dict(fake._session.cookies) == {"A1": "abc", "A3": "def"}


def test_seed_yf_session_from_env_does_not_clobber_existing_crumb(monkeypatch):
    # A process that already fetched (or already seeded) its own crumb this
    # run shouldn't have it overwritten - guards against a real fetch that
    # succeeded moments ago being clobbered by a now-stale seed value.
    monkeypatch.setenv("YF_SEED_CRUMB", "seed-crumb")
    monkeypatch.setenv("YF_SEED_COOKIES", json.dumps({"A1": "abc"}))
    fake = _FakeYfData(crumb="already-have-one")
    monkeypatch.setattr(yf_session_module, "YfData", lambda: fake)

    yf_session_module.seed_yf_session_from_env()

    assert fake._crumb == "already-have-one"
    assert dict(fake._session.cookies) == {}


def test_reseed_yf_session_overwrites_existing_crumb(monkeypatch):
    # Unlike seed_yf_session_from_env, reseed_yf_session's whole point is
    # replacing an already-seeded, now-stale crumb - it must NOT skip just
    # because one is already present.
    fake = _FakeYfData(crumb="old-stale-crumb")
    fake._session.cookies.set("OLD", "cookie")
    monkeypatch.setattr(yf_session_module, "YfData", lambda: fake)

    yf_session_module.reseed_yf_session("fresh-crumb", {"A1": "abc"})

    assert fake._crumb == "fresh-crumb"
    assert dict(fake._session.cookies) == {"A1": "abc"}


def test_reseed_yf_session_clears_stale_cookies_not_just_adds(monkeypatch):
    # A hot reseed replaces the WHOLE cookie jar - a stale cookie from the
    # old session lingering alongside the new ones could confuse Yahoo's
    # own session validation.
    fake = _FakeYfData(crumb="old-crumb")
    fake._session.cookies.set("STALE", "leftover")
    monkeypatch.setattr(yf_session_module, "YfData", lambda: fake)

    yf_session_module.reseed_yf_session("fresh-crumb", {"NEW": "cookie"})

    assert "STALE" not in fake._session.cookies
    assert dict(fake._session.cookies) == {"NEW": "cookie"}
