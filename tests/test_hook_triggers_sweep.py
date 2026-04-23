"""SessionStart hook triggers the lifecycle advancement sweep.

Sweep is fire-and-forget via the lifecycle_executor. Session must still
start even if advance_stages blows up.
"""

from __future__ import annotations

from unittest.mock import patch

from ogham.hooks import session_start


def test_session_start_triggers_advance_stages():
    """Session start schedules advance_stages(profile) on the lifecycle executor."""
    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=[]),
        patch("ogham.hooks.lifecycle_submit") as submit_mock,
        patch("ogham.hooks.advance_stages") as advance_mock,
    ):
        session_start(cwd="/tmp/somewhere", profile="work", limit=4)

    # Exactly one submit, of advance_stages with profile='work'
    assert submit_mock.call_count == 1
    submitted_fn, submitted_profile = submit_mock.call_args[0]
    assert submitted_fn is advance_mock  # same callable was passed
    assert submitted_profile == "work"


def test_session_start_survives_advance_failure():
    """If advance_stages raises, session_start still returns its markdown.

    Uses the real (non-mocked) lifecycle_submit so the exception would
    actually propagate if not caught by the executor. submit runs the
    call on a background thread where the exception is logged, not
    raised. session_start must simply return.
    """

    def boom(*a, **kw):
        raise RuntimeError("advance_stages exploded")

    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=[]),
        patch("ogham.hooks.advance_stages", side_effect=boom),
    ):
        result = session_start(cwd="/tmp/somewhere", profile="work", limit=4)

    # Returns empty string because hybrid_search_memories returned []
    assert isinstance(result, str)


def test_session_start_returns_markdown_when_search_succeeds():
    """Unchanged legacy behavior: non-empty search -> markdown blob."""
    fake_hits = [
        {"content": "first memory body", "tags": ["type:fact"]},
        {"content": "second memory body", "tags": []},
    ]
    with (
        patch("ogham.embeddings.generate_embedding", return_value=[0.0] * 512),
        patch("ogham.database.hybrid_search_memories", return_value=fake_hits),
        patch("ogham.hooks.lifecycle_submit"),
        patch("ogham.hooks.advance_stages"),
    ):
        result = session_start(cwd="/tmp/proj", profile="work", limit=4)

    assert "## Session Context" in result
    assert "first memory body" in result
    assert "second memory body" in result
