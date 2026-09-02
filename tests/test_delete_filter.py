"""Tests for the delete filter that incremental_index sends to LanceDB.

Background: the filter interpolated each changed path with double quotes
(`file_path = "notes/a.md"`). LanceDB parses filters as SQL, where double quotes
mean an *identifier*, so the planner looked for a column named after the path and
raised `Schema error: No field named "notes/a.md"`. Every incremental reindex
that had at least one changed or deleted file aborted, and the index froze.

The bug was latent from the first commit and surfaced when the server's venv was
rebuilt onto lancedb 0.38.0.

The builder is pure and tested directly. The SQL dialect itself is the thing that
was misjudged, so `test_delete_filter_executes_against_lancedb` also runs the
generated filter through a real scratch table -- a replicated formula would have
happily reproduced the same wrong assumption.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.indexer import build_path_delete_filter, sql_string_literal


# --- sql_string_literal ---


def test_uses_single_quotes_not_double():
    """Double quotes are the bug: SQL reads them as a column name."""
    literal = sql_string_literal("_AI/relationships-archive.md")
    assert literal == "'_AI/relationships-archive.md'"
    assert '"' not in literal


def test_embedded_apostrophe_is_doubled():
    """149 vault notes have an apostrophe in the filename; unescaped, they close
    the literal early and the whole predicate fails to parse."""
    assert (
        sql_string_literal("5. Archive/Brett's PopClip Extensions.md")
        == "'5. Archive/Brett''s PopClip Extensions.md'"
    )


def test_multiple_apostrophes_all_doubled():
    assert sql_string_literal("a'b'c") == "'a''b''c'"


def test_empty_string():
    assert sql_string_literal("") == "''"


@settings(max_examples=200)
@given(value=st.text(max_size=200))
def test_literal_is_balanced_and_recoverable(value):
    """Whatever goes in, the result is a well-formed SQL literal that decodes
    back to the original."""
    literal = sql_string_literal(value)

    assert literal.startswith("'") and literal.endswith("'")
    assert len(literal) >= 2

    body = literal[1:-1]
    # Every quote in the body is part of a doubled pair, so the literal can't
    # terminate early.
    assert body.replace("''", "").count("'") == 0
    assert body.replace("''", "'") == value


# --- build_path_delete_filter ---


def test_filter_shape():
    assert (
        build_path_delete_filter(["b.md", "a.md"])
        == "file_path IN ('a.md', 'b.md')"
    )


def test_filter_is_sorted_and_stable():
    """Input is a set at the call site, whose iteration order varies per run."""
    paths = {"z.md", "a.md", "m.md"}
    assert build_path_delete_filter(paths) == build_path_delete_filter(list(paths))
    assert build_path_delete_filter(paths) == "file_path IN ('a.md', 'm.md', 'z.md')"


def test_filter_names_the_column_as_a_bare_identifier():
    """file_path stays unquoted (it really is a column); only values are quoted."""
    assert build_path_delete_filter(["a.md"]).startswith("file_path IN (")


def test_filter_escapes_apostrophes():
    assert (
        build_path_delete_filter(["Bet's birthday.md"])
        == "file_path IN ('Bet''s birthday.md')"
    )


# --- Contract with LanceDB itself ---


def test_delete_filter_executes_against_lancedb(tmp_path):
    """The regression, end to end: build a filter over paths that previously blew
    up, run it against a real table, and confirm it deletes exactly those rows."""
    lancedb = pytest.importorskip("lancedb")

    doomed = [
        "_AI/relationships-archive.md",
        "4. Resources/Meeting Notes/2026-09-01 1100 - Design Review Tue.md",
        "5. Archive/Brett's PopClip Extensions.md",
    ]
    survivor = "_AI/keep-me.md"

    db = lancedb.connect(str(tmp_path / "probe.lance"))
    table = db.create_table(
        "vault_chunks",
        data=[
            {"file_path": p, "chunk_index": 0}
            for p in [*doomed, survivor]
        ],
    )

    table.delete(build_path_delete_filter(doomed))

    remaining = [row["file_path"] for row in table.search().to_list()]
    assert remaining == [survivor]
