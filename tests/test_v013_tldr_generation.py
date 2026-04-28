"""v0.13 progressive recall: single-LLM-call three-form generation tests.

The Phase 2.2 contract: recompute_topic_summary calls synthesize_json
exactly once and the response carries body + tldr_short + tldr_one_line
together. Replaces three-separate-call alternatives so the LLM bill
stays one call deep and the three forms share voice.

These are unit tests -- no DB. They patch the persistence boundary
(upsert_summary) and the LLM boundary (synthesize_json) so they don't
need scratch Postgres or an LLM provider. The integration counterpart
lives in tests/test_wiki_integration.py (test_compile_wiki_three_forms_*).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_recompute_calls_synthesize_json_exactly_once(monkeypatch):
    """Single LLM call produces all three forms via JSON-structured output."""
    json_response = {
        "body": "## Auth Redesign\n\nFull body content here, ~500 words.",
        "tldr_short": (
            "The auth redesign moved session tokens to httpOnly cookies and added refresh rotation."
        ),
        "tldr_one_line": ("Auth redesign: tokens moved to httpOnly cookies + refresh rotation."),
    }

    backend = MagicMock()
    backend.wiki_recompute_get_source_ids.return_value = ["uuid-1", "uuid-2", "uuid-3"]
    backend.wiki_recompute_get_source_content.return_value = [
        {"id": "uuid-1", "content": "Memory 1: We decided httpOnly cookies"},
        {"id": "uuid-2", "content": "Memory 2: Refresh rotation logic added"},
        {"id": "uuid-3", "content": "Memory 3: Old token endpoint deprecated"},
    ]
    # No prior summary -> no hash short-circuit
    upsert_returned = {"id": "summary-uuid", "version": 1}
    backend.wiki_topic_upsert.return_value = upsert_returned

    synthesize_mock = MagicMock(return_value=json_response)

    with (
        patch("ogham.recompute.get_backend", return_value=backend),
        patch("ogham.topic_summaries.get_backend", return_value=backend),
        patch("ogham.recompute.synthesize_json", synthesize_mock),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
        patch("ogham.recompute.get_summary_by_topic", return_value=None),
    ):
        from ogham.recompute import recompute_topic_summary

        result = recompute_topic_summary(
            profile="test-tldr",
            topic_key="auth-redesign",
            provider="openai",
            model="gpt-4o-mini",
        )

    # Single LLM call.
    assert synthesize_mock.call_count == 1
    assert result["action"] == "recomputed"

    # The upsert received all three forms.
    call_kwargs = backend.wiki_topic_upsert.call_args.kwargs
    assert call_kwargs["content"] == json_response["body"]
    assert call_kwargs["tldr_short"] == json_response["tldr_short"]
    assert call_kwargs["tldr_one_line"] == json_response["tldr_one_line"]


def test_recompute_passes_three_form_schema_to_synthesize_json(monkeypatch):
    """Schema sent to synthesize_json requires body + tldr_short + tldr_one_line."""
    backend = MagicMock()
    backend.wiki_recompute_get_source_ids.return_value = ["m1"]
    backend.wiki_recompute_get_source_content.return_value = [
        {"id": "m1", "content": "memory body"},
    ]
    backend.wiki_topic_upsert.return_value = {"id": "s1"}

    captured = {}

    def capture(prompt, **kwargs):
        captured["kwargs"] = kwargs
        return {"body": "x", "tldr_short": "y", "tldr_one_line": "z"}

    with (
        patch("ogham.recompute.get_backend", return_value=backend),
        patch("ogham.topic_summaries.get_backend", return_value=backend),
        patch("ogham.recompute.synthesize_json", side_effect=capture),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
        patch("ogham.recompute.get_summary_by_topic", return_value=None),
    ):
        from ogham.recompute import recompute_topic_summary

        recompute_topic_summary(
            profile="test",
            topic_key="x",
            provider="openai",
            model="gpt-4o-mini",
        )

    schema = captured["kwargs"]["json_schema"]
    assert schema["type"] == "object"
    required = schema.get("required", [])
    assert "body" in required
    assert "tldr_short" in required
    assert "tldr_one_line" in required
    # Per-field descriptions guide the LLM toward the intended form.
    props = schema.get("properties", {})
    assert "tldr_short" in props
    assert "tldr_one_line" in props
    # body's description anchors length expectations.
    assert "markdown" in props["body"]["description"].lower()


def test_recompute_validates_body_for_emptiness_not_tldrs(monkeypatch):
    """Empty body still raises -- TLDRs being short-but-present is OK."""
    backend = MagicMock()
    backend.wiki_recompute_get_source_ids.return_value = ["m1"]
    backend.wiki_recompute_get_source_content.return_value = [
        {"id": "m1", "content": "memory"},
    ]

    # body whitespace-only -> _validate_synthesize_output should raise.
    json_response = {
        "body": "   \n   ",
        "tldr_short": "summary",
        "tldr_one_line": "one line",
    }

    with (
        patch("ogham.recompute.get_backend", return_value=backend),
        patch("ogham.topic_summaries.get_backend", return_value=backend),
        patch("ogham.recompute.synthesize_json", return_value=json_response),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
        patch("ogham.recompute.get_summary_by_topic", return_value=None),
    ):
        from ogham.recompute import recompute_topic_summary

        with pytest.raises(ValueError, match="empty"):
            recompute_topic_summary(
                profile="test",
                topic_key="x",
                provider="openai",
                model="gpt-4o-mini",
            )

    # The upsert NEVER ran -- Letta #3270 guard applies to validation
    # failures the same as it does to synthesize failures.
    backend.wiki_topic_upsert.assert_not_called()


def test_recompute_propagates_synthesize_json_value_error(monkeypatch):
    """If synthesize_json raises (invalid JSON, missing field), recompute propagates.

    The atomic-upsert contract: the previous fresh row stays intact when the
    LLM call fails for ANY reason -- network error, malformed JSON, missing
    required field. That's the Letta #3270 guard, ported through the new
    JSON-structured-output path.
    """
    backend = MagicMock()
    backend.wiki_recompute_get_source_ids.return_value = ["m1"]
    backend.wiki_recompute_get_source_content.return_value = [
        {"id": "m1", "content": "memory"},
    ]

    with (
        patch("ogham.recompute.get_backend", return_value=backend),
        patch("ogham.topic_summaries.get_backend", return_value=backend),
        patch(
            "ogham.recompute.synthesize_json",
            side_effect=ValueError("missing required fields ['tldr_short']"),
        ),
        patch("ogham.recompute.generate_embedding", return_value=[0.1] * 512),
        patch("ogham.recompute.get_summary_by_topic", return_value=None),
    ):
        from ogham.recompute import recompute_topic_summary

        with pytest.raises(ValueError, match="missing required fields"):
            recompute_topic_summary(
                profile="test",
                topic_key="x",
                provider="openai",
                model="gpt-4o-mini",
            )

    # Upsert never ran.
    backend.wiki_topic_upsert.assert_not_called()


def test_recompute_no_sources_short_circuits_before_synthesize(monkeypatch):
    """If no memories carry the topic tag, recompute returns no_sources without LLM call."""
    backend = MagicMock()
    backend.wiki_recompute_get_source_ids.return_value = []

    synth_mock = MagicMock()

    with (
        patch("ogham.recompute.get_backend", return_value=backend),
        patch("ogham.recompute.synthesize_json", synth_mock),
    ):
        from ogham.recompute import recompute_topic_summary

        out = recompute_topic_summary(
            profile="test",
            topic_key="ghost",
            provider="openai",
            model="gpt-4o-mini",
        )

    assert out["action"] == "no_sources"
    assert synth_mock.call_count == 0
