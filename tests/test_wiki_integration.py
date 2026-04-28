"""Live-Postgres integration tests for wiki Tier 1 (T1.1, T1.5, T1.6).

Runs against the local scratch DB (ogham-postgres-scratch on port 5433)
when DATABASE_URL contains 'scratch' OR OGHAM_TEST_ALLOW_DESTRUCTIVE=1
is set. Skipped otherwise — these fixtures DROP/DELETE rows so they
cannot run against Supabase or any prod DB.

What's covered here that the unit tests in test_compile_wiki /
test_walk_knowledge / test_wiki_lint don't:
  * walk_memory_graph recursive CTE actually returning correct neighbours
    per direction against real memory_relationships rows
  * search_summaries hitting the partial HNSW index (status='fresh') with
    real cosine ordering
  * find_summary_drift detecting hash mismatch after a real tag mutation
  * find_orphans honouring the 5-min grace window on real timestamps
  * find_contradictions surfacing real relationship_type='contradicts' rows

End-to-end (synthesize + embed are still mocked because the live DB
fixture doesn't have an LLM or embedding provider configured) but the
SQL surface is fully exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

MIG_025 = Path(__file__).parent.parent / "sql/migrations/025_memory_lifecycle.sql"
MIG_026 = Path(__file__).parent.parent / "sql/migrations/026_memory_lifecycle_split.sql"
MIG_028 = Path(__file__).parent.parent / "sql/migrations/028_topic_summaries.sql"
MIG_030 = Path(__file__).parent.parent / "sql/migrations/030_topic_summaries_dim_agnostic.sql"
MIG_031 = Path(__file__).parent.parent / "sql/migrations/031_wiki_rpc_functions.sql"
MIG_033 = Path(__file__).parent.parent / "sql/migrations/033_topic_summaries_tldr.sql"
MIG_034 = Path(__file__).parent.parent / "sql/migrations/034_wiki_topic_search_tldr.sql"
DANGER_028 = Path(__file__).parent.parent / "sql/migrations/rollback/DANGER_028_topic_summaries.sql"


def _can_connect() -> bool:
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


pytestmark = [
    pytest.mark.postgres_integration,
    pytest.mark.skipif(not _can_connect(), reason="Postgres backend not configured or unreachable"),
]


def _apply_028(pg_fresh_db):
    """Apply lifecycle + summaries + wiki RPC migrations idempotently on the scratch DB.

    DANGER_028 is the canonical rollback; it has the session-variable
    guard, so we have to set ``ogham.confirm_rollback`` and run the
    rollback in the same _execute call (one connection) before applying
    028 fresh. 030 + 031 follow because the rollback's `wiki_*` function
    drops + topic_summaries drop wipe everything they install.
    """
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)
    pg_fresh_db.apply_rollback(DANGER_028)
    pg_fresh_db.apply_sql(MIG_028)
    pg_fresh_db.apply_sql(MIG_030)
    pg_fresh_db.apply_sql(MIG_031)
    # v0.13: migration 033 grew wiki_topic_upsert RPC's signature; the
    # Python backend always passes the new params (NULL is fine), so the
    # RPC must exist at the new arity.
    pg_fresh_db.apply_sql(MIG_033)
    # v0.13 + 034: wiki_topic_search RETURNS TABLE widened to include the
    # tldr columns so service._wiki_injection_results can route by level.
    pg_fresh_db.apply_sql(MIG_034)


def _seed_memories_with_tag(
    n: int = 3,
    profile: str = "test-025",
    tag: str = "wiki-int-quantum",
    content_prefix: str = "wiki integration seed",
) -> list[str]:
    from ogham.database import get_backend

    backend = cast(Any, get_backend())
    rows = backend._execute(
        """INSERT INTO memories (content, profile, source, tags)
           SELECT %(prefix)s || ' ' || i::text, %(profile)s, 't', ARRAY[%(tag)s]
             FROM generate_series(1, %(n)s) AS i
           RETURNING id::text AS id""",
        {"n": n, "profile": profile, "tag": tag, "prefix": content_prefix},
        fetch="all",
    )
    return [r["id"] for r in rows]


def _add_relationship(source_id: str, target_id: str, rel: str, strength: float = 0.9):
    from ogham.database import get_backend

    backend = cast(Any, get_backend())
    backend._execute(
        """INSERT INTO memory_relationships
              (source_id, target_id, relationship, strength, created_by)
           VALUES (%(s)s::uuid, %(t)s::uuid, %(r)s::relationship_type, %(strength)s, 'test')
           ON CONFLICT DO NOTHING""",
        {"s": source_id, "t": target_id, "r": rel, "strength": strength},
        fetch="none",
    )


# --------------------------------------------------------------------- #
# walk_memory_graph (T1.5) — direction-aware recursive CTE
# --------------------------------------------------------------------- #


def test_walk_memory_graph_outgoing_returns_target_neighbours(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(3, tag="walk-out")
    # A -> B, A -> C
    _add_relationship(ids[0], ids[1], "similar")
    _add_relationship(ids[0], ids[2], "similar")
    # B -> A reverse edge — must NOT show up on outgoing-from-A
    _add_relationship(ids[1], ids[0], "supports")

    from ogham.database import walk_memory_graph

    out = walk_memory_graph(ids[0], depth=1, direction="outgoing")
    out_ids = sorted(str(r["id"]) for r in out)
    assert out_ids == sorted([ids[1], ids[2]])
    assert all(r["direction_used"] == "outgoing" for r in out)


def test_walk_memory_graph_incoming_returns_source_neighbours(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(3, tag="walk-in")
    # B -> A, C -> A; A -> D won't exist (we only have 3)
    _add_relationship(ids[1], ids[0], "similar")
    _add_relationship(ids[2], ids[0], "similar")
    # A -> B forward edge — must NOT surface on incoming-to-A
    _add_relationship(ids[0], ids[1], "supports")

    from ogham.database import walk_memory_graph

    out = walk_memory_graph(ids[0], depth=1, direction="incoming")
    out_ids = sorted(str(r["id"]) for r in out)
    assert out_ids == sorted([ids[1], ids[2]])
    assert all(r["direction_used"] == "incoming" for r in out)


def test_walk_memory_graph_both_unions_directions(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(3, tag="walk-both")
    _add_relationship(ids[0], ids[1], "similar")  # outgoing
    _add_relationship(ids[2], ids[0], "similar")  # incoming

    from ogham.database import walk_memory_graph

    out = walk_memory_graph(ids[0], depth=1, direction="both")
    out_ids = sorted(str(r["id"]) for r in out)
    assert out_ids == sorted([ids[1], ids[2]])


def test_walk_memory_graph_relationship_types_filter(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(3, tag="walk-typed")
    _add_relationship(ids[0], ids[1], "similar")
    _add_relationship(ids[0], ids[2], "contradicts")

    from ogham.database import walk_memory_graph

    out = walk_memory_graph(
        ids[0], depth=1, direction="outgoing", relationship_types=["contradicts"]
    )
    out_ids = [str(r["id"]) for r in out]
    assert out_ids == [ids[2]]
    assert out[0]["relationship"] == "contradicts"


def test_walk_memory_graph_min_strength_filter(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(3, tag="walk-strong")
    _add_relationship(ids[0], ids[1], "similar", strength=0.95)
    _add_relationship(ids[0], ids[2], "similar", strength=0.30)  # below cutoff

    from ogham.database import walk_memory_graph

    out = walk_memory_graph(ids[0], depth=1, direction="outgoing", min_strength=0.5)
    out_ids = [str(r["id"]) for r in out]
    assert out_ids == [ids[1]]


# --------------------------------------------------------------------- #
# search_summaries (injection layer SQL)
# --------------------------------------------------------------------- #


def test_search_summaries_filters_to_fresh_status(pg_fresh_db):
    """A stale summary must not surface even if it matches the embedding closely."""
    _apply_028(pg_fresh_db)
    seed_ids = _seed_memories_with_tag(2, tag="search-fresh")

    from ogham.topic_summaries import (
        get_summary_by_topic,
        mark_stale,
        search_summaries,
        upsert_summary,
    )

    upsert_summary(
        profile="test-025",
        topic_key="search-fresh",
        content="fresh summary body",
        embedding=[0.1] * 512,
        source_memory_ids=seed_ids,
        model_used="test/test",
    )
    upsert_summary(
        profile="test-025",
        topic_key="search-stale",
        content="stale summary body",
        embedding=[0.1] * 512,
        source_memory_ids=seed_ids,
        model_used="test/test",
    )
    stale_row = get_summary_by_topic("test-025", "search-stale")
    assert stale_row is not None
    mark_stale(stale_row["id"], reason="test")

    results = search_summaries("test-025", [0.1] * 512, top_k=10)
    found_topics = {r["topic_key"] for r in results}
    assert "search-fresh" in found_topics
    assert "search-stale" not in found_topics  # filtered by status='fresh'


def test_search_summaries_ranks_closer_embedding_first(pg_fresh_db):
    _apply_028(pg_fresh_db)
    seed_ids = _seed_memories_with_tag(2, tag="search-rank")

    from ogham.topic_summaries import search_summaries, upsert_summary

    near = [0.5] * 512
    far = [0.5] * 256 + [-0.5] * 256

    upsert_summary(
        profile="test-025",
        topic_key="rank-near",
        content="closer",
        embedding=near,
        source_memory_ids=seed_ids,
        model_used="test/test",
    )
    upsert_summary(
        profile="test-025",
        topic_key="rank-far",
        content="further",
        embedding=far,
        source_memory_ids=seed_ids,
        model_used="test/test",
    )

    results = search_summaries("test-025", near, top_k=2, min_similarity=0.0)
    assert len(results) == 2
    assert results[0]["topic_key"] == "rank-near"
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_search_summaries_returns_tldr_columns(pg_fresh_db):
    """v0.13: wiki_topic_search RPC must surface tldr_one_line / tldr_short.

    Regression guard for the 034 migration. Before 034, the RPC's RETURNS
    TABLE shape only exposed the body content, so service._wiki_injection_results
    silently fell back to body for level=short / level=one_line on every
    request -- the BEAM scorer didn't catch this because it reads results,
    not wiki_preamble.
    """
    _apply_028(pg_fresh_db)
    seed_ids = _seed_memories_with_tag(1, tag="tldr-rpc-shape")

    from ogham.topic_summaries import search_summaries, upsert_summary

    upsert_summary(
        profile="test-025",
        topic_key="tldr-rpc-shape",
        content="full body content",
        embedding=[0.5] * 512,
        source_memory_ids=seed_ids,
        model_used="test/test",
        tldr_one_line="one line summary",
        tldr_short="short paragraph summary",
    )

    rows = search_summaries("test-025", [0.5] * 512, top_k=1, min_similarity=0.0)
    assert len(rows) == 1
    row = rows[0]
    # Without 034, "tldr_one_line" and "tldr_short" would be missing from
    # the returned dict (RPC RETURNS TABLE didn't include them). 034 widens
    # the return shape so the values flow through.
    assert "tldr_one_line" in row, "wiki_topic_search must return tldr_one_line (migration 034)"
    assert "tldr_short" in row, "wiki_topic_search must return tldr_short (migration 034)"
    assert row["tldr_one_line"] == "one line summary"
    assert row["tldr_short"] == "short paragraph summary"


# --------------------------------------------------------------------- #
# wiki_lint (T1.6) — find_X functions against real DB
# --------------------------------------------------------------------- #


def test_find_contradictions_surfaces_real_rows(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(2, tag="lint-contra")
    _add_relationship(ids[0], ids[1], "contradicts")

    from ogham.wiki_lint import find_contradictions

    out = find_contradictions("test-025")
    assert out["count"] == 1
    assert out["sample"][0]["source_id"] == ids[0]
    assert out["sample"][0]["target_id"] == ids[1]


def test_find_orphans_excludes_recent_grace_window(pg_fresh_db):
    """A just-stored memory shouldn't register as an orphan even with no edges."""
    _apply_028(pg_fresh_db)
    _seed_memories_with_tag(3, tag="lint-orphan-fresh")

    from ogham.wiki_lint import find_orphans

    out = find_orphans("test-025")
    # All 3 memories are <5min old, so they're in the grace window
    assert out["count"] == 0


def test_find_orphans_surfaces_old_unlinked_memory(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(2, tag="lint-orphan-old")

    # Backdate one row past the 5-min grace; leave the other inside it.
    from ogham.database import get_backend

    backend = cast(Any, get_backend())
    backend._execute(
        "UPDATE memories SET created_at = now() - interval '10 minutes' WHERE id = %(id)s::uuid",
        {"id": ids[0]},
        fetch="none",
    )

    from ogham.wiki_lint import find_orphans

    out = find_orphans("test-025")
    assert out["count"] == 1
    assert out["sample"][0]["id"] == ids[0]


def test_find_summary_drift_detects_added_source_memory(pg_fresh_db):
    """Cache says {a, b}; real state has {a, b, c} -> drift detected."""
    _apply_028(pg_fresh_db)
    initial_ids = _seed_memories_with_tag(2, tag="lint-drift")

    from ogham.topic_summaries import upsert_summary

    upsert_summary(
        profile="test-025",
        topic_key="lint-drift",
        content="cached body",
        embedding=[0.1] * 512,
        source_memory_ids=initial_ids,
        model_used="test/test",
    )

    # Add a third memory carrying the same tag — cache hash now stale.
    _seed_memories_with_tag(1, tag="lint-drift", content_prefix="late add")

    from ogham.wiki_lint import find_summary_drift

    out = find_summary_drift("test-025")
    assert out["count"] == 1
    assert out["sample"][0]["topic_key"] == "lint-drift"
    assert out["sample"][0]["current_source_count"] == 3


def test_lint_report_aggregates_real_data(pg_fresh_db):
    _apply_028(pg_fresh_db)
    ids = _seed_memories_with_tag(2, tag="lint-aggr")
    _add_relationship(ids[0], ids[1], "contradicts")

    from ogham.wiki_lint import lint_report

    out = lint_report("test-025", include_drift=False)
    assert out["profile"] == "test-025"
    assert out["healthy"] is False
    assert out["contradictions"]["count"] == 1
    assert out["summary_drift"]["skipped"] is True


# --------------------------------------------------------------------- #
# compile_wiki end-to-end (synthesize + embed mocked, DB live)
# --------------------------------------------------------------------- #


def test_compile_wiki_end_to_end_writes_to_topic_summaries(pg_fresh_db):
    _apply_028(pg_fresh_db)
    seed_ids = _seed_memories_with_tag(3, tag="e2e-compile")

    from ogham.tools import wiki

    three_forms = {
        "body": "## Overview\n\ncompiled body",
        "tldr_short": "Compiled body summary in one paragraph.",
        "tldr_one_line": "Compiled body summary.",
    }

    with (
        patch.object(wiki, "get_active_profile", return_value="test-025"),
        patch("ogham.recompute.synthesize_json", return_value=three_forms),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
    ):
        out = wiki.compile_wiki(topic="e2e-compile")

    assert out["action"] == "recomputed"
    assert out["topic_key"] == "e2e-compile"
    assert out["source_count"] == 3
    assert "## Overview" in out["markdown"]
    assert out["markdown"].startswith("---\n")  # frontmatter present

    # Cache row really exists with all three forms populated.
    from ogham.topic_summaries import get_summary_by_topic

    row = get_summary_by_topic("test-025", "e2e-compile")
    assert row is not None
    assert row["status"] == "fresh"
    assert row["source_count"] == 3
    assert row["content"] == "## Overview\n\ncompiled body"
    # v0.13: TLDR forms persisted via the migration-033 columns.
    assert row["tldr_short"] == "Compiled body summary in one paragraph."
    assert row["tldr_one_line"] == "Compiled body summary."
    assert sorted(seed_ids)  # silence unused-var warning; we already used the ids


def test_compile_wiki_three_forms_written_in_single_llm_call(pg_fresh_db):
    """v0.13: synthesize_json is called exactly once and produces all three forms.

    The end-to-end path must NOT make three separate LLM calls -- the whole
    point of the JSON-structured-output design is one call, three values.
    """
    _apply_028(pg_fresh_db)
    _seed_memories_with_tag(2, tag="single-call")

    from ogham.tools import wiki

    three_forms = {
        "body": "Body here. [Sources: ...]",
        "tldr_short": "Short paragraph form.",
        "tldr_one_line": "One-liner form.",
    }

    with (
        patch.object(wiki, "get_active_profile", return_value="test-025"),
        patch("ogham.recompute.synthesize_json", return_value=three_forms) as mock_synth,
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
    ):
        wiki.compile_wiki(topic="single-call")

    assert mock_synth.call_count == 1, "v0.13 contract: one LLM call per recompute, not three"
    # The call sent the three-form schema to the LLM.
    schema = mock_synth.call_args.kwargs["json_schema"]
    required = schema.get("required", [])
    assert "body" in required
    assert "tldr_short" in required
    assert "tldr_one_line" in required


def test_compile_wiki_no_sources_doesnt_create_row(pg_fresh_db):
    _apply_028(pg_fresh_db)

    from ogham.tools import wiki
    from ogham.topic_summaries import get_summary_by_topic

    with patch.object(wiki, "get_active_profile", return_value="test-025"):
        out = wiki.compile_wiki(topic="ghost-no-mems")

    assert out["status"] == "no_sources"
    assert get_summary_by_topic("test-025", "ghost-no-mems") is None


# --------------------------------------------------------------------- #
# v0.13: query_topic_summary level= against a live DB row
# --------------------------------------------------------------------- #


def test_query_topic_summary_level_short_live_returns_tldr_short(pg_fresh_db):
    """End-to-end: compile_wiki populates all three forms; query at level='short'
    returns the tldr_short column from the persisted row.
    """
    _apply_028(pg_fresh_db)
    _seed_memories_with_tag(2, tag="level-int")

    from ogham.tools import wiki

    three_forms = {
        "body": "## Long form\n\nLots of detail here.",
        "tldr_short": "Short paragraph form, ~150 tokens.",
        "tldr_one_line": "One sentence.",
    }

    with (
        patch.object(wiki, "get_active_profile", return_value="test-025"),
        patch("ogham.recompute.synthesize_json", return_value=three_forms),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
    ):
        wiki.compile_wiki(topic="level-int")

        # Now query at each level.
        body_resp = wiki.query_topic_summary(topic="level-int")
        short_resp = wiki.query_topic_summary(topic="level-int", level="short")
        one_line_resp = wiki.query_topic_summary(topic="level-int", level="one_line")

    assert body_resp["level"] == "body"
    assert body_resp["content"] == three_forms["body"]

    assert short_resp["level"] == "short"
    assert short_resp["content"] == three_forms["tldr_short"]

    assert one_line_resp["level"] == "one_line"
    assert one_line_resp["content"] == three_forms["tldr_one_line"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
