"""Unit tests for the lifecycle state machine.

After migration 026, stage transitions live in ``memory_lifecycle``.
Tests insert into ``memories`` (the AFTER INSERT trigger auto-creates
the lifecycle row at stage='fresh'), then update ``memory_lifecycle``
to set the desired ``stage_entered_at`` for time-travel setup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ogham.lifecycle import advance_stages, hybrid_decay_factor


def _now():
    return datetime.now(tz=timezone.utc)


def _can_connect() -> bool:
    """Check if Postgres backend is configured and reachable."""
    try:
        from ogham.config import settings

        if settings.database_backend != "postgres":
            return False
        from ogham.backends.postgres import PostgresBackend

        backend = PostgresBackend()
        backend._execute("SELECT 1", fetch="scalar")
        return True
    except Exception:
        return False


_pg_guard = pytest.mark.skipif(
    not _can_connect(), reason="Postgres backend not configured or unreachable"
)


def _seed_memory_with_age(pg_client, profile, content, surprise, importance, age):
    """Insert a memory at the given age; trigger creates the lifecycle row,
    then backdate ``memory_lifecycle.stage_entered_at`` to simulate dwell."""
    entered = _now() - age
    pg_client.execute(
        """INSERT INTO memories (id, profile, content, surprise, importance, created_at)
           VALUES (gen_random_uuid(), %(profile)s, %(content)s,
                   %(surprise)s, %(importance)s, %(created)s)""",
        {
            "profile": profile,
            "content": content,
            "surprise": surprise,
            "importance": importance,
            "created": entered,
        },
    )
    # Trigger inserted a lifecycle row with stage_entered_at = created_at.
    # Force it to our target age explicitly so this test is robust even
    # if the trigger logic changes.
    pg_client.execute(
        """UPDATE memory_lifecycle SET stage_entered_at = %(entered)s
            WHERE profile = %(profile)s AND stage = 'fresh'""",
        {"profile": profile, "entered": entered},
    )


@pytest.mark.postgres_integration
@_pg_guard
def test_advance_fresh_to_stable_dwell_and_surprise(pg_test_profile, pg_client):
    _seed_memory_with_age(
        pg_client,
        pg_test_profile,
        "test content here fresh→stable surprise",
        0.5,
        0.1,
        timedelta(hours=2),
    )
    report = advance_stages(pg_test_profile)
    assert report.fresh_to_stable == 1
    row = pg_client.fetchone(
        "SELECT stage FROM memory_lifecycle WHERE profile = %(p)s",
        {"p": pg_test_profile},
    )
    assert row["stage"] == "stable"


@pytest.mark.postgres_integration
@_pg_guard
def test_advance_fresh_to_stable_dwell_and_importance(pg_test_profile, pg_client):
    _seed_memory_with_age(
        pg_client,
        pg_test_profile,
        "test content here fresh→stable importance",
        0.1,
        0.6,
        timedelta(hours=2),
    )
    report = advance_stages(pg_test_profile)
    assert report.fresh_to_stable == 1


@pytest.mark.postgres_integration
@_pg_guard
def test_advance_fresh_stays_fresh_when_gate_fails(pg_test_profile, pg_client):
    _seed_memory_with_age(
        pg_client,
        pg_test_profile,
        "test content here stays fresh low signal",
        0.1,
        0.1,
        timedelta(hours=2),
    )
    report = advance_stages(pg_test_profile)
    assert report.fresh_to_stable == 0


@pytest.mark.postgres_integration
@_pg_guard
def test_advance_fresh_stays_fresh_when_dwell_too_short(pg_test_profile, pg_client):
    _seed_memory_with_age(
        pg_client,
        pg_test_profile,
        "test content here dwell too short",
        0.9,
        0.9,
        timedelta(minutes=30),
    )
    report = advance_stages(pg_test_profile)
    assert report.fresh_to_stable == 0


def test_decay_exponential_under_3d():
    """At day 0, factor = 1.0. At day 2, factor = exp(-0.2) ≈ 0.819."""
    assert hybrid_decay_factor(0.0) == pytest.approx(1.0)
    assert hybrid_decay_factor(2.0) == pytest.approx(0.8187, rel=1e-3)


def test_decay_powerlaw_over_3d():
    """At day 3, curves touch at factor = 1.0 (both branches agree).
    At day 30, factor = (30/3)^(-0.4) ≈ 0.398."""
    assert hybrid_decay_factor(30.0) == pytest.approx(0.398, rel=1e-2)


def test_decay_custom_params():
    """Per-profile params override defaults."""
    assert hybrid_decay_factor(2.0, decay_lambda=0.2) == pytest.approx(0.6703, rel=1e-3)
