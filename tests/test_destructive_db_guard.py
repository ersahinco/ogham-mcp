"""Unit tests for the _destructive_db_safe() guard in tests/conftest.py.

Guard default-denies destructive fixtures (pg_fresh_db, pg_test_profile)
against non-scratch DBs. These tests verify the logic directly so we
don't need to actually point a fixture at a prod DB to exercise the
refusal path.
"""

from __future__ import annotations


def _call_guard(monkeypatch, *, allow=None, url=None):
    """Invoke the guard with a specific env configuration."""
    import tests.conftest as ct

    if allow is None:
        monkeypatch.delenv("OGHAM_TEST_ALLOW_DESTRUCTIVE", raising=False)
    else:
        monkeypatch.setenv("OGHAM_TEST_ALLOW_DESTRUCTIVE", allow)

    if url is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", url)

    return ct._destructive_db_safe()


def test_guard_allows_explicit_env_opt_in(monkeypatch):
    allowed, _ = _call_guard(monkeypatch, allow="1", url="postgresql://x@prod-db:5432/app")
    assert allowed


def test_guard_allows_scratch_in_url(monkeypatch):
    allowed, _ = _call_guard(
        monkeypatch,
        url="postgresql://scratch:pw@localhost:5433/scratch",
    )
    assert allowed


def test_guard_denies_prod_looking_url(monkeypatch):
    allowed, reason = _call_guard(monkeypatch, url="postgresql://x@db.supabase.co:5432/app")
    assert not allowed
    assert "refusing" in reason.lower()


def test_guard_denies_empty_env(monkeypatch):
    allowed, reason = _call_guard(monkeypatch)
    assert not allowed
    assert "refusing" in reason.lower()


def test_guard_env_opt_in_accepts_various_truthy(monkeypatch):
    for value in ("1", "true", "TRUE", "yes"):
        allowed, _ = _call_guard(monkeypatch, allow=value, url="postgresql://x@prod:5432/app")
        assert allowed, f"{value!r} should be truthy"


def test_guard_env_opt_in_rejects_falsy(monkeypatch):
    for value in ("0", "false", "no", ""):
        allowed, _ = _call_guard(monkeypatch, allow=value, url="postgresql://x@prod:5432/app")
        assert not allowed, f"{value!r} should not be truthy"
