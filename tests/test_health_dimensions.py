"""Unit tests for the v0.13 8-dimension health readout.

Each ``compute_<dim>`` function returns a ``DimensionResult`` carrying
score (0.0-10.0), zone, and a human-readable detail string. The CLI and
dashboard surfaces consume the same values.

These tests mock the DB / pool / search interfaces so they run without
a Postgres instance. The integration test for the e2e probe lives below
under the ``postgres_integration`` marker.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# ----- zone helper --------------------------------------------------------


def test_zone_green_at_threshold():
    from ogham.health_dimensions import zone

    assert zone(8.0) == "GREEN"
    assert zone(9.99) == "GREEN"
    assert zone(10.0) == "GREEN"


def test_zone_amber_in_band():
    from ogham.health_dimensions import zone

    assert zone(5.0) == "AMBER"
    assert zone(6.4) == "AMBER"
    assert zone(7.999) == "AMBER"


def test_zone_red_below_5():
    from ogham.health_dimensions import zone

    assert zone(0.0) == "RED"
    assert zone(4.999) == "RED"


# ----- 1. DB freshness ----------------------------------------------------


class _FakeBackend:
    """Stand-in for the active backend; tests inject the rows it returns."""

    def __init__(self, scalar=None, rows=None, raises=None):
        self._scalar = scalar
        self._rows = rows or []
        self._raises = raises
        self.calls = []

    def _execute(self, sql, params=None, fetch="all"):
        self.calls.append((sql, params, fetch))
        if self._raises is not None:
            raise self._raises
        if fetch == "scalar":
            return self._scalar
        if fetch == "one":
            return self._rows[0] if self._rows else None
        return self._rows


def _patch_backend(backend):
    return patch("ogham.health_dimensions.get_backend", return_value=backend)


def test_db_freshness_green_when_recent_write():
    from ogham.health_dimensions import compute_db_freshness

    recent = datetime.now(timezone.utc) - timedelta(hours=4)
    backend = _FakeBackend(scalar=recent)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="test")
    assert result.score >= 8.0
    assert result.zone == "GREEN"
    assert "h" in result.detail.lower() or "ago" in result.detail.lower()


def test_db_freshness_amber_when_24_to_72h():
    from ogham.health_dimensions import compute_db_freshness

    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    backend = _FakeBackend(scalar=stale)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="test")
    assert 5.0 <= result.score < 8.0
    assert result.zone == "AMBER"


def test_db_freshness_red_when_over_72h():
    from ogham.health_dimensions import compute_db_freshness

    very_stale = datetime.now(timezone.utc) - timedelta(days=30)
    backend = _FakeBackend(scalar=very_stale)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="test")
    assert result.score < 5.0
    assert result.zone == "RED"


def test_db_freshness_handles_empty_profile():
    from ogham.health_dimensions import compute_db_freshness

    backend = _FakeBackend(scalar=None)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="empty")
    # No writes ever -- treat as red ("dead writer")
    assert result.score < 5.0
    assert result.zone == "RED"


# ----- Supabase backend fallback paths (v0.13 regression fix) ------------


class _FakeSupabaseBackend:
    """Backend without ``_execute`` (matches SupabaseBackend's PostgREST surface).

    DB freshness routes through ``list_recent_memories``; corpus size routes
    through ``get_memory_stats``. Wiki coverage / Profile health degrade to
    GREEN N/A since they need raw SQL we don't expose via PostgREST yet.
    """

    def __init__(self, recent=None, total=0):
        self._recent = recent
        self._total = total

    def list_recent_memories(self, profile, limit=10, source=None, tags=None):
        if self._recent is None:
            return []
        return [{"created_at": self._recent}]

    def get_memory_stats(self, profile):
        return {"profile": profile, "total": self._total, "sources": {}, "top_tags": []}


def test_db_freshness_supabase_routes_through_list_recent_memories():
    from ogham.health_dimensions import compute_db_freshness

    recent = datetime.now(timezone.utc) - timedelta(hours=2)
    backend = _FakeSupabaseBackend(recent=recent)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="work")
    assert result.score >= 8.0
    assert result.zone == "GREEN"


def test_db_freshness_supabase_handles_iso_string_created_at():
    """PostgREST returns created_at as ISO-8601 strings; tolerate the trailing Z."""
    from ogham.health_dimensions import compute_db_freshness

    iso = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat().replace("+00:00", "Z")
    backend = _FakeSupabaseBackend(recent=iso)
    with _patch_backend(backend):
        result = compute_db_freshness(profile="work")
    assert result.score >= 8.0
    assert result.zone == "GREEN"


def test_corpus_size_supabase_routes_through_get_memory_stats():
    from ogham.health_dimensions import compute_corpus_size

    backend = _FakeSupabaseBackend(total=1158)
    with _patch_backend(backend):
        result = compute_corpus_size(profile="work")
    assert result.score == 10.0
    assert result.zone == "GREEN"
    assert "1,158" in result.detail


def test_wiki_coverage_supabase_returns_honest_n_a():
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeSupabaseBackend()
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="work")
    assert result.zone == "GREEN"
    assert "Supabase" in result.detail
    assert "N/A" in result.detail


def test_profile_health_supabase_returns_honest_n_a():
    from ogham.health_dimensions import compute_profile_health

    backend = _FakeSupabaseBackend()
    with _patch_backend(backend):
        result = compute_profile_health(profile="work")
    assert result.zone == "GREEN"
    assert "Supabase" in result.detail
    assert "N/A" in result.detail


# ----- 2. Schema integrity -----------------------------------------------


def test_schema_integrity_green_when_all_pass():
    from ogham.health_dimensions import compute_schema_integrity

    with patch(
        "ogham.health_dimensions._run_schema_integrity_checks",
        return_value=(True, "all migrations applied, parity OK"),
    ):
        result = compute_schema_integrity()
    assert result.score == 10.0
    assert result.zone == "GREEN"


def test_schema_integrity_red_when_check_fails():
    from ogham.health_dimensions import compute_schema_integrity

    with patch(
        "ogham.health_dimensions._run_schema_integrity_checks",
        return_value=(False, "missing migration 028"),
    ):
        result = compute_schema_integrity()
    assert result.score == 0.0
    assert result.zone == "RED"


# ----- 3. Hybrid search latency -----------------------------------------


def test_hybrid_search_latency_green_when_p50_below_50ms():
    from ogham.health_dimensions import compute_hybrid_search_latency

    # Five fast samples (<10 ms each)
    with patch(
        "ogham.health_dimensions._sample_hybrid_search_latencies",
        return_value=[0.005, 0.006, 0.007, 0.008, 0.009],
    ):
        result = compute_hybrid_search_latency(profile="test")
    assert result.score >= 8.0
    assert result.zone == "GREEN"


def test_hybrid_search_latency_amber_in_50_200ms_band():
    from ogham.health_dimensions import compute_hybrid_search_latency

    with patch(
        "ogham.health_dimensions._sample_hybrid_search_latencies",
        return_value=[0.07, 0.08, 0.09, 0.10, 0.12],
    ):
        result = compute_hybrid_search_latency(profile="test")
    assert 5.0 <= result.score < 8.0
    assert result.zone == "AMBER"


def test_hybrid_search_latency_red_above_200ms():
    from ogham.health_dimensions import compute_hybrid_search_latency

    with patch(
        "ogham.health_dimensions._sample_hybrid_search_latencies",
        return_value=[0.40, 0.45, 0.50, 0.55, 0.60],
    ):
        result = compute_hybrid_search_latency(profile="test")
    assert result.score < 5.0
    assert result.zone == "RED"


# ----- 4. Corpus size ----------------------------------------------------


def test_corpus_size_green_when_over_100():
    from ogham.health_dimensions import compute_corpus_size

    backend = _FakeBackend(scalar=6633)
    with _patch_backend(backend):
        result = compute_corpus_size(profile="test")
    assert result.score == 10.0
    assert result.zone == "GREEN"


def test_corpus_size_amber_when_10_to_99():
    from ogham.health_dimensions import compute_corpus_size

    backend = _FakeBackend(scalar=42)
    with _patch_backend(backend):
        result = compute_corpus_size(profile="test")
    assert 5.0 <= result.score < 8.0
    assert result.zone == "AMBER"


def test_corpus_size_red_when_below_10():
    from ogham.health_dimensions import compute_corpus_size

    backend = _FakeBackend(scalar=3)
    with _patch_backend(backend):
        result = compute_corpus_size(profile="test")
    assert result.score < 5.0
    assert result.zone == "RED"


# ----- 5. Wiki coverage --------------------------------------------------


def test_wiki_coverage_green_when_all_fresh():
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeBackend(rows=[{"fresh": 25, "total": 25}])
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="test")
    assert result.score == 10.0
    assert result.zone == "GREEN"


def test_wiki_coverage_amber_when_partly_stale():
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeBackend(rows=[{"fresh": 18, "total": 25}])  # 72%
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="test")
    # Per spec: any stale exists => AMBER, even if score might otherwise round to 10
    assert result.zone == "AMBER"
    assert result.score < 8.0  # 72% => 7.2


def test_wiki_coverage_red_when_mostly_stale():
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeBackend(rows=[{"fresh": 2, "total": 25}])  # 8%
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="test")
    assert result.score < 5.0
    assert result.zone == "RED"


def test_wiki_coverage_handles_empty_table():
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeBackend(rows=[{"fresh": 0, "total": 0}])
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="test")
    # No summaries -- N/A, treat as full score (not-applicable, no problem)
    assert result.score == 10.0
    assert "N/A" in result.detail or "no summaries" in result.detail.lower()


def test_wiki_coverage_handles_missing_table():
    """Pre-028 DBs have no topic_summaries table. Should report N/A, not crash."""
    from ogham.health_dimensions import compute_wiki_coverage

    backend = _FakeBackend(raises=Exception('relation "topic_summaries" does not exist'))
    with _patch_backend(backend):
        result = compute_wiki_coverage(profile="test")
    assert result.score == 10.0
    assert "N/A" in result.detail or "not available" in result.detail.lower()


# ----- 6. Profile health (avg tags + orphan %) ---------------------------


def test_profile_health_green_when_well_tagged_and_linked():
    from ogham.health_dimensions import compute_profile_health

    # 3.2 avg tags, 4% orphans
    backend = _FakeBackend(rows=[{"avg_tags": 3.2, "total": 100, "orphans": 4}])
    with _patch_backend(backend):
        result = compute_profile_health(profile="test")
    assert result.score >= 8.0
    assert result.zone == "GREEN"


def test_profile_health_amber_when_few_tags():
    from ogham.health_dimensions import compute_profile_health

    backend = _FakeBackend(rows=[{"avg_tags": 1.2, "total": 100, "orphans": 8}])
    with _patch_backend(backend):
        result = compute_profile_health(profile="test")
    assert 5.0 <= result.score < 8.0
    assert result.zone == "AMBER"


def test_profile_health_red_when_many_orphans():
    from ogham.health_dimensions import compute_profile_health

    backend = _FakeBackend(rows=[{"avg_tags": 0.5, "total": 100, "orphans": 80}])
    with _patch_backend(backend):
        result = compute_profile_health(profile="test")
    assert result.score < 5.0
    assert result.zone == "RED"


def test_profile_health_handles_empty():
    from ogham.health_dimensions import compute_profile_health

    backend = _FakeBackend(rows=[{"avg_tags": 0.0, "total": 0, "orphans": 0}])
    with _patch_backend(backend):
        result = compute_profile_health(profile="empty")
    # No memories => can't measure; report N/A neutral
    assert result.score == 10.0
    assert "N/A" in result.detail or "no memories" in result.detail.lower()


# ----- 7. Concurrency (pool stats) ---------------------------------------


def test_concurrency_green_when_pool_idle():
    from ogham.health_dimensions import compute_concurrency

    fake_pool_stats = {
        "pool_size": 5,
        "pool_available": 4,  # 1 busy => 20%
        "requests_waiting": 0,
        "usage_ms": 8.0,
    }
    backend = MagicMock()
    backend._pool = MagicMock()
    backend._pool.get_stats = MagicMock(return_value=fake_pool_stats)
    with patch("ogham.health_dimensions.get_backend", return_value=backend):
        result = compute_concurrency()
    assert result.score >= 8.0
    assert result.zone == "GREEN"


def test_concurrency_red_when_pool_saturated():
    from ogham.health_dimensions import compute_concurrency

    fake_pool_stats = {
        "pool_size": 5,
        "pool_available": 0,  # 100% busy
        "requests_waiting": 4,
        "usage_ms": 1500.0,  # very high wait
    }
    backend = MagicMock()
    backend._pool = MagicMock()
    backend._pool.get_stats = MagicMock(return_value=fake_pool_stats)
    with patch("ogham.health_dimensions.get_backend", return_value=backend):
        result = compute_concurrency()
    assert result.score < 5.0
    assert result.zone == "RED"


def test_concurrency_na_for_supabase_backend():
    """Supabase backend has no _pool attribute. Should report N/A, not crash."""
    from ogham.health_dimensions import compute_concurrency

    backend = MagicMock(spec=["_get_client"])  # no _pool
    with patch("ogham.health_dimensions.get_backend", return_value=backend):
        result = compute_concurrency()
    assert result.score == 10.0
    assert "N/A" in result.detail or "not available" in result.detail.lower()


# ----- 8. E2E probe ------------------------------------------------------


def test_e2e_probe_green_when_round_trip_succeeds():
    from ogham.health_dimensions import compute_e2e_probe

    with (
        patch("ogham.health_dimensions._run_e2e_probe") as run_probe,
    ):
        run_probe.return_value = (True, 84.0, None)
        result = compute_e2e_probe(profile="test")
    assert result.score == 10.0
    assert result.zone == "GREEN"


def test_e2e_probe_red_when_step_fails():
    from ogham.health_dimensions import compute_e2e_probe

    with (
        patch("ogham.health_dimensions._run_e2e_probe") as run_probe,
    ):
        run_probe.return_value = (False, 0.0, "search returned 0 results")
        result = compute_e2e_probe(profile="test")
    assert result.score == 0.0
    assert result.zone == "RED"


# ----- compose_health (top-level wrapper) --------------------------------


def test_compose_health_returns_eight_dimensions():
    """compose_health stitches all eight ``compute_*`` results together."""
    from ogham.health_dimensions import DimensionResult, compose_health

    fake = DimensionResult(name="x", score=10.0, zone="GREEN", detail="ok")
    with (
        patch("ogham.health_dimensions.compute_db_freshness", return_value=fake),
        patch("ogham.health_dimensions.compute_schema_integrity", return_value=fake),
        patch("ogham.health_dimensions.compute_hybrid_search_latency", return_value=fake),
        patch("ogham.health_dimensions.compute_corpus_size", return_value=fake),
        patch("ogham.health_dimensions.compute_wiki_coverage", return_value=fake),
        patch("ogham.health_dimensions.compute_profile_health", return_value=fake),
        patch("ogham.health_dimensions.compute_concurrency", return_value=fake),
        patch("ogham.health_dimensions.compute_e2e_probe", return_value=fake),
    ):
        results = compose_health(profile="test")
    assert len(results) == 8
    names = [r.name for r in results]
    expected = [
        "DB freshness",
        "Schema integrity",
        "Hybrid search",
        "Corpus size",
        "Wiki coverage",
        "Profile health",
        "Concurrency",
        "E2E probe",
    ]
    assert names == expected


def test_compose_health_handles_dimension_errors_gracefully():
    """One dim raising should not abort the whole report."""
    from ogham.health_dimensions import compose_health

    with (
        patch(
            "ogham.health_dimensions.compute_db_freshness",
            side_effect=RuntimeError("DB down"),
        ),
        patch("ogham.health_dimensions.compute_schema_integrity"),
        patch("ogham.health_dimensions.compute_hybrid_search_latency"),
        patch("ogham.health_dimensions.compute_corpus_size"),
        patch("ogham.health_dimensions.compute_wiki_coverage"),
        patch("ogham.health_dimensions.compute_profile_health"),
        patch("ogham.health_dimensions.compute_concurrency"),
        patch("ogham.health_dimensions.compute_e2e_probe"),
    ):
        results = compose_health(profile="test")
    assert len(results) == 8
    # Failed dim is reported as RED with the error in detail
    failed = results[0]
    assert failed.zone == "RED"
    assert failed.score == 0.0
    assert "DB down" in failed.detail or "error" in failed.detail.lower()


# ----- CLI integration ---------------------------------------------------


def test_cli_health_renders_eight_dim_table(monkeypatch):
    """`ogham health` prints eight rows + an Overall line."""
    from typer.testing import CliRunner

    monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.cli import app
    from ogham.health_dimensions import DimensionResult

    fake_results = [
        DimensionResult(name="DB freshness", score=10.0, zone="GREEN", detail="last write 4h"),
        DimensionResult(name="Schema integrity", score=10.0, zone="GREEN", detail="parity OK"),
        DimensionResult(name="Hybrid search", score=9.2, zone="GREEN", detail="p50 18ms"),
        DimensionResult(name="Corpus size", score=10.0, zone="GREEN", detail="6633 memories"),
        DimensionResult(name="Wiki coverage", score=7.2, zone="AMBER", detail="72% fresh"),
        DimensionResult(name="Profile health", score=9.1, zone="GREEN", detail="3.2 tags"),
        DimensionResult(name="Concurrency", score=10.0, zone="GREEN", detail="pool 12% busy"),
        DimensionResult(name="E2E probe", score=10.0, zone="GREEN", detail="84ms ok"),
    ]
    runner = CliRunner()
    with patch("ogham.health_dimensions.compose_health", return_value=fake_results):
        result = runner.invoke(app, ["health"])
    assert result.exit_code == 0, result.output
    # All eight rows rendered
    assert "DB freshness" in result.output
    assert "Wiki coverage" in result.output
    assert "E2E probe" in result.output
    # Overall summary printed
    assert "Overall" in result.output


def test_cli_health_json_mode_emits_list(monkeypatch):
    from typer.testing import CliRunner

    monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.cli import app
    from ogham.health_dimensions import DimensionResult

    fake_results = [
        DimensionResult(name=f"Dim {i}", score=10.0, zone="GREEN", detail="ok") for i in range(8)
    ]
    runner = CliRunner()
    with patch("ogham.health_dimensions.compose_health", return_value=fake_results):
        result = runner.invoke(app, ["health", "--json"])
    assert result.exit_code == 0, result.output
    import json as _json

    parsed = _json.loads(result.output)
    assert isinstance(parsed, dict)
    assert "dimensions" in parsed
    assert len(parsed["dimensions"]) == 8
    assert all("name" in d and "score" in d and "zone" in d for d in parsed["dimensions"])


# ----- Integration: real e2e probe round-trip ----------------------------


@pytest.mark.postgres_integration
def test_e2e_probe_round_trip_against_scratch_db():
    """Live test: store -> hybrid_search -> delete must succeed end-to-end.

    Runs only when ``DATABASE_BACKEND=postgres`` and ``OGHAM_TEST_ALLOW_DESTRUCTIVE=1``
    are set. Uses a unique tag so it can clean up on any prior failure.
    """
    import os

    if os.environ.get("DATABASE_BACKEND") != "postgres":
        pytest.skip("e2e probe needs Postgres backend")
    if os.environ.get("OGHAM_TEST_ALLOW_DESTRUCTIVE") != "1":
        pytest.skip("e2e probe needs OGHAM_TEST_ALLOW_DESTRUCTIVE=1")

    from ogham.health_dimensions import compute_e2e_probe

    result = compute_e2e_probe(profile="health-probe-integration")
    assert result.score == 10.0, f"Probe failed: {result.detail}"
    assert result.zone == "GREEN"
