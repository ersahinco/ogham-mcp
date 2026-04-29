"""v0.13.1 Supabase parity regression tests for lifecycle / graph / density /
suggest_connections.

Pattern mirrors tests/test_health_dimensions.py:117-194 — a fake backend
with no `_execute` attribute (matching the SupabaseBackend's PostgREST
surface) is patched in via `get_backend()`. Asserts that each call site
goes through the public facade method, never `_execute` directly.

Without these regressions, the same anti-pattern that bit v0.11+ on health
could resurface: a future refactor adds a `_execute` call from outside
`backends/`, Supabase silently swallows the AttributeError in a
fire-and-forget background task, and we don't notice for several releases.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────


class _FakeSupabaseBackend:
    """Backend without `_execute` -- exposes only the public facade
    methods migration 035 added.

    Tests pass per-method canned responses via the constructor; calls are
    recorded so we can assert the right facade was hit with the right
    args. This is the same shape `tests/test_health_dimensions.py` uses
    for the v0.13 health refactor precedent.
    """

    def __init__(
        self,
        advance_stages_count: int = 0,
        close_editing_windows_count: int = 0,
        pipeline_counts: dict[str, int] | None = None,
        strengthen_edges_count: int = 0,
        density: tuple[float, float] = (0.0, 0.0),
        suggestions: list[dict] | None = None,
    ):
        self._advance_stages_count = advance_stages_count
        self._close_editing_windows_count = close_editing_windows_count
        self._pipeline_counts = pipeline_counts or {"fresh": 0, "stable": 0, "editing": 0}
        self._strengthen_edges_count = strengthen_edges_count
        self._density = density
        self._suggestions = suggestions or []
        # Call recorders -- list-of-dicts so we can assert payloads.
        self.calls: dict[str, list[dict]] = {
            "lifecycle_advance_stages": [],
            "lifecycle_close_editing_windows": [],
            "lifecycle_open_editing_window": [],
            "lifecycle_pipeline_counts": [],
            "hebbian_strengthen_edges": [],
            "entity_graph_density": [],
            "suggest_unlinked_by_shared_entities": [],
        }

    # Asserting that this attribute is NOT defined is the regression: any
    # call site that tries `backend._execute(...)` will AttributeError, the
    # tests will catch it, and the fix can land before users do.
    # (Don't add `_execute` here -- the absence is load-bearing.)

    def lifecycle_advance_stages(
        self,
        profile: str,
        cutoff_iso: str,
        surprise_gate: float,
        importance_gate: float,
    ) -> int:
        self.calls["lifecycle_advance_stages"].append(
            {
                "profile": profile,
                "cutoff_iso": cutoff_iso,
                "surprise_gate": surprise_gate,
                "importance_gate": importance_gate,
            }
        )
        return self._advance_stages_count

    def lifecycle_close_editing_windows(self, profile: str, cutoff_iso: str) -> int:
        self.calls["lifecycle_close_editing_windows"].append(
            {"profile": profile, "cutoff_iso": cutoff_iso}
        )
        return self._close_editing_windows_count

    def lifecycle_open_editing_window(self, memory_ids: list[str]) -> None:
        self.calls["lifecycle_open_editing_window"].append({"memory_ids": memory_ids})

    def lifecycle_pipeline_counts(self, profile: str) -> dict[str, int]:
        self.calls["lifecycle_pipeline_counts"].append({"profile": profile})
        return dict(self._pipeline_counts)

    def hebbian_strengthen_edges(
        self,
        sources: list[str],
        targets: list[str],
        bootstrap: float,
        rate: float,
    ) -> int:
        self.calls["hebbian_strengthen_edges"].append(
            {"sources": sources, "targets": targets, "bootstrap": bootstrap, "rate": rate}
        )
        return self._strengthen_edges_count

    def entity_graph_density(self, profile: str) -> tuple[float, float]:
        self.calls["entity_graph_density"].append({"profile": profile})
        return self._density

    def suggest_unlinked_by_shared_entities(
        self,
        memory_id: str,
        profile: str,
        min_shared: int,
        limit: int,
    ) -> list[dict]:
        self.calls["suggest_unlinked_by_shared_entities"].append(
            {
                "memory_id": memory_id,
                "profile": profile,
                "min_shared": min_shared,
                "limit": limit,
            }
        )
        return list(self._suggestions)


@contextmanager
def _patch_backend(backend):
    """Patch every site that imports get_backend so the fake stays in scope."""
    with (
        patch("ogham.lifecycle.get_backend", return_value=backend),
        patch("ogham.graph.get_backend", return_value=backend),
        patch("ogham.database.get_backend", return_value=backend),
    ):
        yield


# ── 1. lifecycle.advance_stages ───────────────────────────────────────


def test_lifecycle_advance_stages_routes_through_facade():
    from ogham.lifecycle import advance_stages

    backend = _FakeSupabaseBackend(advance_stages_count=3, close_editing_windows_count=2)
    with _patch_backend(backend):
        report = advance_stages(profile="work")

    assert report.fresh_to_stable == 3
    assert report.editing_closed == 2
    assert len(backend.calls["lifecycle_advance_stages"]) == 1
    call = backend.calls["lifecycle_advance_stages"][0]
    assert call["profile"] == "work"
    assert call["surprise_gate"] == pytest.approx(0.3)
    assert call["importance_gate"] == pytest.approx(0.5)
    # cutoff is computed at call time; just sanity-check the shape.
    assert isinstance(call["cutoff_iso"], str)
    assert "T" in call["cutoff_iso"]


def test_lifecycle_advance_stages_no_attribute_error_on_supabase():
    """Regression: must NOT touch _execute on a SupabaseBackend-shaped object."""
    from ogham.lifecycle import advance_stages

    backend = _FakeSupabaseBackend()
    with _patch_backend(backend):
        report = advance_stages(profile="work")

    assert report.fresh_to_stable == 0
    assert not hasattr(backend, "_execute")


# ── 2. lifecycle.close_editing_windows ────────────────────────────────


def test_lifecycle_close_editing_windows_routes_through_facade():
    from ogham.lifecycle import close_editing_windows

    backend = _FakeSupabaseBackend(close_editing_windows_count=5)
    with _patch_backend(backend):
        result = close_editing_windows(profile="work")

    assert result == 5
    assert len(backend.calls["lifecycle_close_editing_windows"]) == 1
    assert backend.calls["lifecycle_close_editing_windows"][0]["profile"] == "work"


# ── 3. lifecycle.open_editing_window ──────────────────────────────────


def test_lifecycle_open_editing_window_routes_through_facade():
    from ogham.lifecycle import open_editing_window

    backend = _FakeSupabaseBackend()
    ids = ["uuid-1", "uuid-2"]
    with _patch_backend(backend):
        open_editing_window(ids)

    assert len(backend.calls["lifecycle_open_editing_window"]) == 1
    assert backend.calls["lifecycle_open_editing_window"][0]["memory_ids"] == ids


def test_lifecycle_open_editing_window_empty_list_short_circuits():
    """Empty input must not call the backend at all (was the v0.11 behaviour
    on Postgres; preserve it on Supabase too)."""
    from ogham.lifecycle import open_editing_window

    backend = _FakeSupabaseBackend()
    with _patch_backend(backend):
        open_editing_window([])

    assert backend.calls["lifecycle_open_editing_window"] == []


# ── 4. lifecycle.lifecycle_pipeline_counts ────────────────────────────


def test_lifecycle_pipeline_counts_routes_through_facade():
    from ogham.lifecycle import lifecycle_pipeline_counts

    backend = _FakeSupabaseBackend(pipeline_counts={"fresh": 7, "stable": 42, "editing": 3})
    with _patch_backend(backend):
        result = lifecycle_pipeline_counts(profile="work")

    assert result == {"fresh": 7, "stable": 42, "editing": 3}
    assert backend.calls["lifecycle_pipeline_counts"][0]["profile"] == "work"


# ── 5. graph.strengthen_edges ─────────────────────────────────────────


def test_strengthen_edges_routes_through_facade():
    from ogham.graph import BOOTSTRAP_STRENGTH, HEBBIAN_RATE, strengthen_edges

    backend = _FakeSupabaseBackend(strengthen_edges_count=3)
    # 3 IDs -> 3 pairs after canonicalisation.
    with _patch_backend(backend):
        result = strengthen_edges(["a", "b", "c"])

    assert result == 3
    assert len(backend.calls["hebbian_strengthen_edges"]) == 1
    call = backend.calls["hebbian_strengthen_edges"][0]
    # Pairs canonicalised to (min, max) and globally sorted -- the deadlock
    # + idempotency invariant from graph.py's docstring.
    assert call["sources"] == ["a", "a", "b"]
    assert call["targets"] == ["b", "c", "c"]
    assert call["bootstrap"] == pytest.approx(BOOTSTRAP_STRENGTH)
    assert call["rate"] == pytest.approx(HEBBIAN_RATE)


def test_strengthen_edges_short_circuits_for_under_two_ids():
    from ogham.graph import strengthen_edges

    backend = _FakeSupabaseBackend()
    with _patch_backend(backend):
        assert strengthen_edges([]) == 0
        assert strengthen_edges(["only-one"]) == 0

    assert backend.calls["hebbian_strengthen_edges"] == []


# ── 6. service._profile_graph_density ─────────────────────────────────


def test_profile_graph_density_routes_through_facade():
    # Density cache is module-global; clear it so the test is deterministic.
    from ogham import service

    service._DENSITY_CACHE.clear()

    backend = _FakeSupabaseBackend(density=(50.0, 100.0))
    with _patch_backend(backend):
        density = service._profile_graph_density(profile="work-fresh")

    assert density == 2.0  # 100 edges / 50 entities
    assert backend.calls["entity_graph_density"][0]["profile"] == "work-fresh"


def test_profile_graph_density_zero_entities_falls_back_to_neutral():
    from ogham import service

    service._DENSITY_CACHE.clear()

    backend = _FakeSupabaseBackend(density=(0.0, 0.0))
    with _patch_backend(backend):
        density = service._profile_graph_density(profile="empty-profile")

    assert density == 2.0  # neutral fallback when there's no signal


# ── 7. tools/memory.suggest_connections ───────────────────────────────


def test_suggest_connections_routes_through_facade():
    from ogham.tools import memory as memory_tool

    sample_rows = [
        {
            "id": "m-2",
            "shared_count": 3,
            "shared_entities": ["person:alice", "project:foo", "tool:slack"],
            "content": "Alice mentioned the foo project on Slack.",
            "created_at": "2026-04-28T10:00:00Z",
            "tags": ["type:decision"],
        }
    ]
    backend = _FakeSupabaseBackend(suggestions=sample_rows)

    with patch.object(memory_tool, "get_active_profile", return_value="work"):
        # tools/memory.py imports get_backend lazily at call time
        with patch("ogham.database.get_backend", return_value=backend):
            result = memory_tool.suggest_connections(
                memory_id="m-1",
                min_shared_entities=2,
                limit=10,
            )

    assert result == sample_rows
    call = backend.calls["suggest_unlinked_by_shared_entities"][0]
    assert call["memory_id"] == "m-1"
    assert call["profile"] == "work"
    assert call["min_shared"] == 2
    assert call["limit"] == 10


def test_suggest_connections_returns_empty_on_backend_error():
    """Pre-existing try/except behaviour: don't let backend errors crash
    the tool. Preserve that on Supabase even though we no longer touch
    _execute -- the facade itself can still raise for other reasons (RLS,
    transient network, etc.)."""
    from ogham.tools import memory as memory_tool

    class _ExplodingBackend:
        def suggest_unlinked_by_shared_entities(self, **kwargs):
            raise RuntimeError("simulated PostgREST 502")

    with patch.object(memory_tool, "get_active_profile", return_value="work"):
        with patch("ogham.database.get_backend", return_value=_ExplodingBackend()):
            result = memory_tool.suggest_connections(memory_id="m-1")

    assert result == []
