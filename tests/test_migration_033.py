"""Migration 033 dry-run + rollback test.

Verifies (a) 033 forward adds tldr_one_line + tldr_short columns to
topic_summaries and replaces wiki_topic_upsert with a 12-param signature;
(b) 033 rollback drops the columns and restores the 10-param signature;
(c) the rollback's session-variable guard sits *inside* the transaction
so naive `psql -f file.sql` (without ON_ERROR_STOP) cannot bypass it --
the destructive ops never run.

The "guard inside BEGIN" pattern is hardened relative to the pre-033
DANGER files (DANGER_025/026/028/030) where the guard sits before BEGIN
and only protects via ON_ERROR_STOP=1. That's a follow-up to harmonise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MIG_025 = Path(__file__).parent.parent / "sql/migrations/025_memory_lifecycle.sql"
MIG_026 = Path(__file__).parent.parent / "sql/migrations/026_memory_lifecycle_split.sql"
MIG_028 = Path(__file__).parent.parent / "sql/migrations/028_topic_summaries.sql"
MIG_030 = Path(__file__).parent.parent / "sql/migrations/030_topic_summaries_dim_agnostic.sql"
MIG_031 = Path(__file__).parent.parent / "sql/migrations/031_wiki_rpc_functions.sql"
MIG_033 = Path(__file__).parent.parent / "sql/migrations/033_topic_summaries_tldr.sql"
ROLLBACK_028 = (
    Path(__file__).parent.parent / "sql/migrations/rollback/DANGER_028_topic_summaries.sql"
)
ROLLBACK_033 = (
    Path(__file__).parent.parent / "sql/migrations/rollback/DANGER_033_topic_summaries_tldr.sql"
)


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
    pytest.mark.skipif(
        not _can_connect(),
        reason="Postgres backend not configured or unreachable",
    ),
]


def _apply_baseline(pg_fresh_db):
    """Bring DB to the post-031 state that 033 builds on."""
    pg_fresh_db.apply_sql(MIG_025)
    pg_fresh_db.apply_sql(MIG_026)
    pg_fresh_db.apply_rollback(ROLLBACK_028)
    pg_fresh_db.apply_sql(MIG_028)
    pg_fresh_db.apply_sql(MIG_030)
    pg_fresh_db.apply_sql(MIG_031)


def _wiki_topic_upsert_arg_count(pg_fresh_db) -> int:
    """Return the parameter count of the live wiki_topic_upsert function."""
    rows = pg_fresh_db.be._execute(
        "SELECT pronargs FROM pg_proc WHERE proname = 'wiki_topic_upsert'",
        fetch="all",
    )
    assert rows, "wiki_topic_upsert function not found"
    return rows[0][0] if isinstance(rows[0], tuple) else rows[0]["pronargs"]


def test_migration_033_forward_adds_tldr_columns(pg_fresh_db):
    """After 033, topic_summaries gains tldr_one_line + tldr_short columns."""
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_033)

    cols = set(pg_fresh_db.column_names("topic_summaries"))
    assert "tldr_one_line" in cols
    assert "tldr_short" in cols


def test_migration_033_forward_extends_wiki_topic_upsert_signature(pg_fresh_db):
    """Forward replaces wiki_topic_upsert with a 12-param signature (was 10)."""
    _apply_baseline(pg_fresh_db)
    assert _wiki_topic_upsert_arg_count(pg_fresh_db) == 10

    pg_fresh_db.apply_sql(MIG_033)
    assert _wiki_topic_upsert_arg_count(pg_fresh_db) == 12


def test_migration_033_rollback_drops_columns_and_restores_signature(pg_fresh_db):
    """DANGER_033 with explicit session-var opt-in undoes the forward migration."""
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_033)
    assert "tldr_one_line" in set(pg_fresh_db.column_names("topic_summaries"))
    assert _wiki_topic_upsert_arg_count(pg_fresh_db) == 12

    pg_fresh_db.apply_rollback(ROLLBACK_033)

    cols_after = set(pg_fresh_db.column_names("topic_summaries"))
    assert "tldr_one_line" not in cols_after
    assert "tldr_short" not in cols_after
    assert _wiki_topic_upsert_arg_count(pg_fresh_db) == 10


def test_migration_033_rollback_guard_blocks_naive_invocation(pg_fresh_db):
    """Without the session-variable opt-in, the rollback aborts the whole
    transaction -- including the destructive ALTER TABLE / DROP COLUMN at
    the bottom of the file. Verifies the guard sits inside BEGIN, not
    before it.
    """
    _apply_baseline(pg_fresh_db)
    pg_fresh_db.apply_sql(MIG_033)
    assert "tldr_one_line" in set(pg_fresh_db.column_names("topic_summaries"))

    # The connection pool can carry an opt-in SET from earlier fixture
    # calls (apply_rollback(DANGER_028) above) into a recycled connection.
    # RESET to simulate a fresh psql -f invocation with no prior session
    # state -- which is the actual threat model the guard defends against.
    pg_fresh_db.be._execute("RESET ogham.confirm_rollback", fetch="none")

    rollback_sql = ROLLBACK_033.read_text()
    import psycopg

    with pytest.raises(psycopg.errors.RaiseException):
        pg_fresh_db.be._execute(rollback_sql, fetch="none")

    # Columns must still be present -- the guard aborted the transaction
    # before the destructive ALTER could execute.
    cols_after = set(pg_fresh_db.column_names("topic_summaries"))
    assert "tldr_one_line" in cols_after
    assert "tldr_short" in cols_after
