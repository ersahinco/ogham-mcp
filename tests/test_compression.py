"""Tests for memory compression."""


def test_compress_to_gist_short():
    """Short content should be returned as-is."""
    from ogham.compression import compress_to_gist

    content = "This is short. Just two sentences."
    result = compress_to_gist(content)
    assert "This is short" in result
    assert "Just two sentences" in result


def test_compress_to_gist_preserves_code():
    """Code blocks should be preserved verbatim."""
    from ogham.compression import compress_to_gist

    content = (
        "We fixed the bug. "
        "The error was in config. "
        "We changed the setting. "
        "Then we tested it. "
        "```python\nprint('hello')\n```"
        " It worked after the change. "
        "The team was happy. "
        "We deployed it. "
        "No more errors. "
        "End of story."
    )
    result = compress_to_gist(content)
    assert "```python" in result
    assert "print('hello')" in result


def test_compress_to_gist_keeps_first_last():
    """Should keep first and last sentences."""
    from ogham.compression import compress_to_gist

    sentences = [f"Sentence {i} about something." for i in range(20)]
    content = " ".join(sentences)
    result = compress_to_gist(content)
    assert "Sentence 0" in result
    assert "Sentence 19" in result


def test_compress_to_gist_shorter_than_original():
    """Gist should be shorter than original."""
    from ogham.compression import compress_to_gist

    sentences = [f"Sentence {i} about something unimportant." for i in range(20)]
    content = " ".join(sentences)
    result = compress_to_gist(content)
    assert len(result) < len(content)


def test_compress_to_tags():
    """Should produce a short summary with tags."""
    from ogham.compression import compress_to_tags

    content = "We decided to use PostgreSQL for the database backend."
    tags = ["type:decision", "project:ogham"]
    result = compress_to_tags(content, tags)
    assert len(result) <= 200
    assert "Tags:" in result
    assert "PostgreSQL" in result or "decided" in result


def test_compress_to_tags_truncates():
    """Long content should be truncated to under 200 chars."""
    from ogham.compression import compress_to_tags

    content = "x" * 300 + ". Second sentence here."
    result = compress_to_tags(content, ["tag1", "tag2"])
    assert len(result) <= 200


def test_get_compression_target_recent():
    """Recent memories should not be compressed."""
    from datetime import datetime, timezone

    from ogham.compression import get_compression_target

    memory = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "importance": 0.5,
        "confidence": 0.5,
        "access_count": 0,
        "compression_level": 0,
    }
    assert get_compression_target(memory) == 0


def test_get_compression_target_high_importance_resists():
    """High-importance memories should resist compression."""
    from datetime import datetime, timedelta, timezone

    from ogham.compression import get_compression_target

    # 10 days old but high importance
    memory = {
        "created_at": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        "importance": 0.9,
        "confidence": 0.5,
        "access_count": 0,
        "compression_level": 0,
    }
    # With importance 0.9, resistance = 2.0, gist threshold = 14 days
    # 10 days < 14 days, so should still be level 0
    assert get_compression_target(memory) == 0
