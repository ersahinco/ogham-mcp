"""Unit tests for the advance_lifecycle MCP tool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def test_advance_lifecycle_returns_report_dict():
    from ogham.tools.memory import advance_lifecycle

    fake_report = SimpleNamespace(fresh_to_stable=3, editing_closed=1)
    with patch("ogham.tools.memory.advance_stages", return_value=fake_report):
        result = advance_lifecycle(profile="work")

    assert result == {
        "profile": "work",
        "fresh_to_stable": 3,
        "editing_closed": 1,
    }


def test_advance_lifecycle_defaults_to_active_profile():
    """When profile is None, tool falls back to get_active_profile()."""
    from ogham.tools.memory import advance_lifecycle

    fake_report = SimpleNamespace(fresh_to_stable=0, editing_closed=0)
    with (
        patch("ogham.tools.memory.advance_stages", return_value=fake_report),
        patch("ogham.tools.memory.get_active_profile", return_value="alt"),
    ):
        result = advance_lifecycle(profile=None)

    assert result["profile"] == "alt"
