"""Search triggers an EDITING window on STABLE hits; FRESH hits stay FRESH.

Lifecycle state lives in ``memory_lifecycle`` after migration 026, so
this test reads and writes stage/stage_entered_at via memory_id there.
"""

import time

import pytest

from ogham.lifecycle_executor import flush as _lifecycle_flush
from ogham.service import search_memories_enriched, store_memory_enriched


@pytest.mark.postgres_integration
def test_hybrid_search_opens_editing_window(pg_test_profile, pg_client):
    """A memory returned by hybrid_search moves to stage=EDITING if it was STABLE."""
    result = store_memory_enriched(
        content="editable content worth retrieving",
        profile=pg_test_profile,
        source="t",
        tags=[],
    )
    mid = str(result["id"])
    pg_client.execute(
        "UPDATE memory_lifecycle SET stage = 'stable' WHERE memory_id = %(id)s::uuid",
        {"id": mid},
    )
    results = search_memories_enriched(
        query="editable content",
        profile=pg_test_profile,
        limit=5,
    )
    _lifecycle_flush()
    found_ids = {str(r["id"]) for r in results}
    assert mid in found_ids, f"expected {mid} in results, got {found_ids}"
    row = pg_client.fetchone(
        "SELECT stage FROM memory_lifecycle WHERE memory_id = %(id)s::uuid",
        {"id": mid},
    )
    assert row["stage"] == "editing"


@pytest.mark.postgres_integration
def test_hybrid_search_leaves_fresh_alone(pg_test_profile, pg_client):
    """FRESH memories stay FRESH even if retrieved -- they're too new to edit."""
    result = store_memory_enriched(
        content="fresh content newly inserted",
        profile=pg_test_profile,
        source="t",
        tags=[],
    )
    mid = str(result["id"])
    # Stage is 'fresh' by default after insert (trigger creates lifecycle row).
    search_memories_enriched(
        query="fresh content",
        profile=pg_test_profile,
        limit=5,
    )
    _lifecycle_flush()
    row = pg_client.fetchone(
        "SELECT stage FROM memory_lifecycle WHERE memory_id = %(id)s::uuid",
        {"id": mid},
    )
    assert row["stage"] == "fresh"


@pytest.mark.postgres_integration
def test_search_returns_before_lifecycle_side_effects_complete(pg_test_profile, pg_client):
    """Search must return before open_editing_window / strengthen_edges
    complete their DB writes. Validates fire-and-forget semantics.
    """
    # Seed a few memories so search has hits
    for n in range(4):
        result = store_memory_enriched(
            content=f"background timing test memory {n}",
            profile=pg_test_profile,
            source="t",
            tags=[],
        )
        mid = str(result["id"])
        pg_client.execute(
            "UPDATE memory_lifecycle SET stage = 'stable' WHERE memory_id = %(id)s::uuid",
            {"id": mid},
        )

    # Time just the search call. Side-effects should not be in this window.
    t0 = time.perf_counter()
    results = search_memories_enriched(
        query="background timing",
        profile=pg_test_profile,
        limit=4,
    )
    search_ms = (time.perf_counter() - t0) * 1000
    assert len(results) >= 2

    # Before flush: stage transitions may or may not have happened yet.
    # After flush: they must all be done.
    _lifecycle_flush()

    # Side-effects completed
    stages = pg_client.fetchone(
        """SELECT count(*) FILTER (WHERE stage = 'editing') AS editing_count
             FROM memory_lifecycle
            WHERE profile = %(p)s""",
        {"p": pg_test_profile},
    )
    assert stages["editing_count"] >= 2

    # Sanity: search itself was fast (<500ms on warm connection).
    # Not a strict latency SLO -- just prove we're not secretly waiting on
    # the lifecycle writes synchronously.
    assert search_ms < 500, (
        f"search took {search_ms:.1f}ms -- suggests side-effects are on hot path"
    )
