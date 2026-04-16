"""Tests for per-category gated query reformulation (v0.10.1 item 1.4).

Query reformulation was disabled globally in v0.9.2 after it regressed MRR
(strip filler words hurt temporal + multi-hop paths). v0.10.1 re-enables it
but ONLY for simple-lookup queries that don't trip any specialised intent.

These tests lock in the gating so it can't silently re-regress.
"""

from unittest.mock import patch

import pytest

FAKE_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_PROFILE", "default")

    from ogham.config import settings

    settings._reset()
    yield
    settings._reset()


def _standard_path_mocks():
    """Patch everything outside the gating decision so the reformulation
    branch is the only live behavior in the test."""
    return [
        patch("ogham.service.is_ordering_query", return_value=False),
        patch("ogham.service.is_multi_hop_temporal", return_value=False),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.record_access"),
    ]


def test_reformulation_applied_on_simple_lookup():
    """Simple info-extraction queries get reformulated query_text in hybrid_search."""
    from ogham.service import search_memories_enriched

    query = "What is the database we settled on for production?"
    reformulated_expected = "database settled production"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch(
            "ogham.service.reformulate_query",
            return_value=reformulated_expected,
        ) as reformulate,
        patch("ogham.service.hybrid_search_memories", return_value=[]) as search,
    ):
        with (
            _standard_path_mocks()[0],
            _standard_path_mocks()[1],
            _standard_path_mocks()[2],
            _standard_path_mocks()[3],
            _standard_path_mocks()[4],
            _standard_path_mocks()[5],
        ):
            search_memories_enriched(query, profile="default")

    reformulate.assert_called_once_with(query)
    search.assert_called()
    # The first hybrid_search call on the standard path uses the reformulated text
    assert search.call_args.kwargs["query_text"] == reformulated_expected


def test_reformulation_skipped_for_temporal_intent():
    """Temporal intent -> keep original query (temporal rerank needs filler words)."""
    from ogham.service import search_memories_enriched

    query = "When did we switch to voyage-3?"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.has_temporal_intent", return_value=True),
        patch("ogham.service.reformulate_query") as reformulate,
        patch("ogham.service.hybrid_search_memories", return_value=[]) as search,
    ):
        with (
            _standard_path_mocks()[0],
            _standard_path_mocks()[1],
            _standard_path_mocks()[2],
            _standard_path_mocks()[3],
            _standard_path_mocks()[4],
            _standard_path_mocks()[5],
        ):
            search_memories_enriched(query, profile="default")

    # Reformulation function should not even be called when temporal intent fires
    reformulate.assert_not_called()
    # Standard path uses original query verbatim
    assert search.call_args.kwargs["query_text"] == query


def test_reformulation_skipped_for_ordering_intent():
    """Ordering intent routes to its own path, reformulation never considered."""
    from ogham.service import search_memories_enriched

    query = "List the order of tasks we tackled last week"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.is_ordering_query", return_value=True),
        patch("ogham.service.is_multi_hop_temporal", return_value=False),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=False),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.reformulate_query") as reformulate,
        patch("ogham.service._strided_retrieval", side_effect=lambda r, limit: r),
        patch("ogham.service._merge_activation_results", side_effect=lambda r, *a, **k: r),
        patch("ogham.service.hybrid_search_memories", return_value=[]) as search,
        patch("ogham.service.record_access"),
    ):
        search_memories_enriched(query, profile="default")

    reformulate.assert_not_called()
    assert search.call_args.kwargs["query_text"] == query


def test_reformulation_skipped_for_multi_hop_temporal():
    """Multi-hop temporal returns early via bridge retrieval; reformulation bypassed."""
    from ogham.service import search_memories_enriched

    query = "How many days between the migration and the benchmark release?"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.is_ordering_query", return_value=False),
        patch("ogham.service.is_multi_hop_temporal", return_value=True),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=False),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.extract_query_anchors", return_value=[]),
        patch("ogham.service.reformulate_query") as reformulate,
        patch("ogham.service._bridge_retrieval", return_value=[]),
        patch("ogham.service._merge_bridge_results", side_effect=lambda b, *a, **k: b),
        patch("ogham.service.hybrid_search_memories", return_value=[]),
        patch("ogham.service.record_access"),
    ):
        search_memories_enriched(query, profile="default")

    reformulate.assert_not_called()


def test_reformulation_skipped_for_broad_summary():
    """Broad summary intent takes strided path, reformulation not applied."""
    from ogham.service import search_memories_enriched

    query = "Summarize everything we decided about the architecture"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.is_ordering_query", return_value=False),
        patch("ogham.service.is_multi_hop_temporal", return_value=False),
        patch("ogham.service.is_cross_reference_query", return_value=False),
        patch("ogham.service.is_broad_summary_query", return_value=True),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch("ogham.service.extract_entities", return_value=[]),
        patch("ogham.service.reformulate_query") as reformulate,
        patch("ogham.service._strided_retrieval", side_effect=lambda r, limit: r),
        patch("ogham.service._merge_activation_results", side_effect=lambda r, *a, **k: r),
        patch("ogham.service.hybrid_search_memories", return_value=[]),
        patch("ogham.service.record_access"),
    ):
        search_memories_enriched(query, profile="default")

    reformulate.assert_not_called()


def test_reformulation_noop_when_result_equals_original():
    """If reformulate_query returns the original query unchanged (short query,
    no filler), the search still uses that string -- no behavior change."""
    from ogham.service import search_memories_enriched

    query = "PostgreSQL"

    with (
        patch("ogham.service.generate_embedding", return_value=FAKE_EMBEDDING),
        patch("ogham.service.has_temporal_intent", return_value=False),
        patch(
            "ogham.service.reformulate_query",
            return_value=query,
        ),
        patch("ogham.service.hybrid_search_memories", return_value=[]) as search,
    ):
        with (
            _standard_path_mocks()[0],
            _standard_path_mocks()[1],
            _standard_path_mocks()[2],
            _standard_path_mocks()[3],
            _standard_path_mocks()[4],
            _standard_path_mocks()[5],
        ):
            search_memories_enriched(query, profile="default")

    assert search.call_args.kwargs["query_text"] == query
