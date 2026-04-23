"""Hebbian edge strengthening on co-retrieval."""

import pytest

from ogham.graph import strengthen_edges
from ogham.lifecycle_executor import flush as _lifecycle_flush
from ogham.service import search_memories_enriched, store_memory_enriched


@pytest.mark.postgres_integration
def test_existing_edge_strengthens(pg_test_profile, pg_client):
    a_result = store_memory_enriched(
        content="memory A first pair", profile=pg_test_profile, source="t", tags=[]
    )
    b_result = store_memory_enriched(
        content="memory B first pair", profile=pg_test_profile, source="t", tags=[]
    )
    # Canonicalize so the manual INSERT and strengthen_edges agree on orientation.
    a, b = sorted([str(a_result["id"]), str(b_result["id"])])

    pg_client.execute(
        """INSERT INTO memory_relationships
               (source_id, target_id, relationship, strength)
           VALUES (%(a)s::uuid, %(b)s::uuid, 'related', 0.5)""",
        {"a": a, "b": b},
    )

    strengthen_edges([a, b])

    row = pg_client.fetchone(
        """SELECT strength FROM memory_relationships
            WHERE source_id = %(a)s::uuid
              AND target_id = %(b)s::uuid
              AND relationship = 'related'""",
        {"a": a, "b": b},
    )
    # eta=0.01: new = min(1.0, 0.5 * 1.01) = 0.505
    assert row["strength"] == pytest.approx(0.505, rel=1e-3)


@pytest.mark.postgres_integration
def test_missing_edge_auto_creates_at_0_1(pg_test_profile, pg_client):
    a_result = store_memory_enriched(
        content="memory A second pair", profile=pg_test_profile, source="t", tags=[]
    )
    b_result = store_memory_enriched(
        content="memory B second pair", profile=pg_test_profile, source="t", tags=[]
    )
    # Canonicalize to match strengthen_edges' sorted-pair orientation.
    a, b = sorted([str(a_result["id"]), str(b_result["id"])])

    strengthen_edges([a, b])

    row = pg_client.fetchone(
        """SELECT strength FROM memory_relationships
            WHERE source_id = %(a)s::uuid
              AND target_id = %(b)s::uuid
              AND relationship = 'related'""",
        {"a": a, "b": b},
    )
    assert row is not None
    assert row["strength"] == pytest.approx(0.1, rel=1e-3)


def test_single_memory_is_noop():
    """Strengthening requires 2+ memories -- no DB needed, no exception."""
    assert strengthen_edges(["abc-123"]) == 0
    assert strengthen_edges([]) == 0


@pytest.mark.postgres_integration
def test_search_triggers_edge_strengthening(pg_test_profile, pg_client):
    """hybrid_search firing on 2+ memories creates/strengthens an edge between them."""
    a_result = store_memory_enriched(
        content="apple red crunchy fruit",
        profile=pg_test_profile,
        source="t",
        tags=[],
    )
    b_result = store_memory_enriched(
        content="apple green tart fruit",
        profile=pg_test_profile,
        source="t",
        tags=[],
    )
    a = str(a_result["id"])
    b = str(b_result["id"])

    results = search_memories_enriched(
        query="apple fruit",
        profile=pg_test_profile,
        limit=5,
    )
    _lifecycle_flush()
    ids_returned = {str(r["id"]) for r in results}
    assert a in ids_returned and b in ids_returned, f"expected both in results: {ids_returned}"

    # Edge exists -- order-insensitive check since combinations() yields (a,b) but
    # a separate test run might have the opposite order if IDs sort differently.
    row = pg_client.fetchone(
        """SELECT strength FROM memory_relationships
            WHERE relationship = 'related'
              AND (
                    (source_id = %(a)s::uuid AND target_id = %(b)s::uuid)
                 OR (source_id = %(b)s::uuid AND target_id = %(a)s::uuid)
              )""",
        {"a": a, "b": b},
    )
    assert row is not None, "expected edge between a and b"
    # Strength should be at least bootstrap (0.1) since this is the first co-retrieval.
    assert row["strength"] >= 0.1
