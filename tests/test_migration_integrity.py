"""Upgrade-path migration integrity guards.

Defends against the v0.9.2 regression class: a stray or misnumbered migration
file sorting after the canonical RRF fix and silently overwriting it. See
CHANGELOG [0.9.2] for the full incident write-up.

Also enforces a positive scope guard after Phase B (v0.13): the duplicate
``src/ogham/sql/`` tree must NOT exist. Phase A's hash-parity gate was the
temporary fix; Phase B (this commit) deleted the duplicate tree entirely.
The wheel ships migrations from canonical ``sql/migrations/`` via hatchling
``force-include``. The scope guard below prevents the duplicate from being
silently re-introduced by a careless commit or a mis-typed Makefile rsync.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "sql" / "migrations"
ROLLBACK_DIR = MIGRATIONS_DIR / "rollback"


def _top_level_migrations() -> list[Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())


def test_no_unnumbered_migrations():
    """Every top-level migration filename must start with a digit.

    `sql/upgrade.sh` applies migrations in alphabetical order. Any .sql file
    whose name begins with a non-digit character (e.g. ``update_search_function.sql``)
    sorts after numbered migrations and silently overrides them.
    """
    offenders = [p.name for p in _top_level_migrations() if not p.name[0].isdigit()]
    assert not offenders, (
        f"Unnumbered migration(s) found in sql/migrations/: {offenders}. "
        "Unnumbered files sort after numbered ones and override them on upgrade."
    )


def test_later_hybrid_search_migrations_preserve_rrf():
    """Later migrations may evolve hybrid search, but must keep true RRF.

    017 introduced the v0.9.2 RRF fix. Later migrations such as 021 may
    legitimately re-define hybrid_search_memories, but they must preserve
    the position-based RRF formula rather than silently reintroducing the
    older raw-score fusion regression.
    """
    migrations = _top_level_migrations()
    rrf_fix = next((p for p in migrations if p.name == "017_rrf_bm25.sql"), None)
    assert rrf_fix is not None, "expected 017_rrf_bm25.sql at top-level sql/migrations/"

    later = [p.name for p in migrations if p.name > rrf_fix.name]
    broken_pattern = "semantic_weight * coalesce(s.similarity"

    for name in later:
        content = (MIGRATIONS_DIR / name).read_text().lower()
        if "create or replace function hybrid_search_memories" not in content:
            continue
        assert "1.0 / (rrf_k + coalesce(" in content, (
            f"{name} redefines hybrid_search_memories but does not preserve "
            "the true Reciprocal Rank Fusion formula"
        )
        assert broken_pattern not in content, (
            f"{name} redefines hybrid_search_memories with the broken raw-score fusion pattern"
        )


def test_017_rrf_bm25_is_functional_and_uses_rrf():
    """017 must contain a real RRF formula, not just a docs comment.

    The v0.8.3–v0.9.1 version of this file was comment-only. The v0.9.2 rewrite
    restores it to a functional migration with position-based RRF.
    """
    content = (MIGRATIONS_DIR / "017_rrf_bm25.sql").read_text()
    assert "create or replace function hybrid_search_memories" in content.lower(), (
        "017_rrf_bm25.sql must define hybrid_search_memories, not just document it"
    )
    assert "1.0 / (rrf_k + coalesce(" in content, (
        "017_rrf_bm25.sql must use true Reciprocal Rank Fusion: "
        "1.0 / (rrf_k + rank_ix), not raw-score linear combination"
    )
    broken_pattern = "semantic_weight * coalesce(s.similarity"
    assert broken_pattern not in content, (
        "017_rrf_bm25.sql contains the broken raw-score fusion pattern"
    )


def test_update_search_function_sql_does_not_exist():
    """The v0.9.1-era stray migration must stay removed."""
    stray = MIGRATIONS_DIR / "update_search_function.sql"
    assert not stray.exists(), (
        "sql/migrations/update_search_function.sql was removed in v0.9.2 "
        "because it silently overrode true RRF. Do not reintroduce it."
    )


def test_danger_rollback_guards_live_inside_transaction():
    """Every DANGER_*.sql rollback must put its session-variable guard
    AFTER the first ``BEGIN;`` statement, not before.

    Pre-033 files put the guard before BEGIN, which means a naive
    ``psql $URL -f file.sql`` without ``ON_ERROR_STOP=1`` prints the
    ERROR from the failed DO block and then keeps running the
    destructive ops below. Inside BEGIN, the abort is transactional --
    all-or-nothing -- which matches the threat model the file header
    docstring claims ("Piping this file naively will FAIL by design").

    Static contract test: scan every DANGER_*.sql file under
    ``sql/migrations/rollback/`` and assert the index of the first
    ``BEGIN`` keyword comes before the first ``current_setting``
    reference (the guard's distinguishing token).
    """
    danger_files = sorted(ROLLBACK_DIR.glob("DANGER_*.sql"))
    assert danger_files, "expected at least one DANGER_*.sql under sql/migrations/rollback/"

    offenders = []
    for path in danger_files:
        text = path.read_text()
        # Strip line comments so a `-- BEGIN` in a header docstring
        # doesn't confuse the index search.
        stripped = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("--")
        )
        begin_idx = stripped.find("BEGIN;")
        guard_idx = stripped.find("current_setting('ogham.confirm_rollback'")
        if begin_idx == -1 or guard_idx == -1:
            offenders.append(f"{path.name}: missing BEGIN; or guard reference")
            continue
        if guard_idx < begin_idx:
            offenders.append(
                f"{path.name}: guard appears at byte {guard_idx} BEFORE BEGIN; "
                f"at byte {begin_idx}. Move the DO $$ guard $$; block to "
                f"sit AFTER BEGIN; so a missing session variable aborts the "
                f"whole transaction (otherwise naive psql -f will run the "
                f"destructive ops despite the ERROR)."
            )

    assert not offenders, "DANGER rollback guard placement issue:\n  " + "\n  ".join(offenders)


def test_no_duplicate_sql_tree_under_src():
    """After Phase B (v0.13), src/ogham/sql/ must NOT exist.

    Phase A added a hash-parity gate between sql/migrations/ and
    src/ogham/sql/migrations/. Phase B deleted the duplicate tree
    entirely. This test prevents the duplicate from being silently
    re-introduced by a careless commit or mis-typed Makefile rsync rule.
    """
    duplicate = REPO_ROOT / "src" / "ogham" / "sql"
    assert not duplicate.exists(), (
        f"Duplicate SQL tree found at {duplicate.relative_to(REPO_ROOT)}. "
        f"After Phase B (v0.13), migrations live ONLY at sql/migrations/. "
        f"The wheel ships them via hatchling force-include in pyproject.toml. "
        f"Either delete the new tree or check why it was re-introduced."
    )
