"""Tests for src/chunker.py — all pure functions, no I/O or network deps."""

import pytest

from src.chunker import (
    parse_frontmatter,
    extract_metadata_fields,
    build_metadata_header,
    strip_html,
    chunk_markdown,
    _split_by_headings,
    _split_section,
    TARGET_CHUNK_SIZE,
)

try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False


# --- parse_frontmatter ---


def test_parse_frontmatter_valid():
    """Standard YAML frontmatter returns (dict, body)."""
    content = "---\ntitle: Hello\ntags:\n  - one\n---\nBody text here."
    meta, body = parse_frontmatter(content)

    assert meta["title"] == "Hello"
    assert meta["tags"] == ["one"]
    assert body == "Body text here."


def test_parse_frontmatter_no_frontmatter():
    """Content not starting with --- returns ({}, full_content)."""
    content = "Just a note with no frontmatter."
    meta, body = parse_frontmatter(content)

    assert meta == {}
    assert body == content


def test_parse_frontmatter_unclosed():
    """Only opening --- with no closing --- returns ({}, full_content)."""
    content = "---\ntitle: Hello\nThis never closes."
    meta, body = parse_frontmatter(content)

    assert meta == {}
    assert body == content


def test_parse_frontmatter_invalid_yaml():
    """Malformed YAML between delimiters returns ({}, body)."""
    content = "---\n: : : invalid yaml [[\n---\nBody text."
    meta, body = parse_frontmatter(content)

    assert meta == {}
    assert body == "Body text."


def test_parse_frontmatter_empty():
    """Empty frontmatter (---\\n---) returns ({}, body)."""
    content = "---\n---\nBody text."
    meta, body = parse_frontmatter(content)

    assert meta == {}
    assert body == "Body text."


# --- extract_metadata_fields ---


def test_extract_metadata_fields_with_wikilinks(sample_frontmatter):
    """[[WikiLink]] values in tags/projects/area are cleaned to plain text."""
    fields = extract_metadata_fields(sample_frontmatter["wikilinks_only"])

    assert fields["tags"] == ["tag-one", "tag-two"]
    assert fields["projects"] == ["Project A"]
    assert fields["area"] == "Area X"


def test_extract_metadata_fields_none_values(sample_frontmatter):
    """None values for all fields produce empty strings or empty lists."""
    fields = extract_metadata_fields(sample_frontmatter["nones"])

    assert fields["title"] == ""
    assert fields["tags"] == []
    assert fields["projects"] == []
    assert fields["status"] == ""
    assert fields["area"] == ""


def test_extract_metadata_fields_full(sample_frontmatter):
    """All fields present — each is extracted correctly."""
    fields = extract_metadata_fields(sample_frontmatter["full"])

    assert fields["title"] == "Test Note"
    assert fields["tags"] == ["python", "testing"]
    assert fields["projects"] == ["SuperFit", "ARC"]
    assert fields["status"] == "open"
    assert fields["area"] == "Engineering"
    assert fields["source"] == "Meeting Notes"


# --- build_metadata_header ---


def test_build_metadata_header_full():
    """All fields populated produces bracketed pipe-separated header."""
    metadata = {
        "title": "Sprint Retro",
        "tags": ["meeting", "retro"],
        "projects": ["SuperFit"],
        "area": "Engineering",
        "source": "Notes",
        "date": "2026-04-10",
    }
    header = build_metadata_header(metadata, "Summary")

    assert header.startswith("[")
    assert header.endswith("]")
    assert "Title: Sprint Retro" in header
    assert "Section: Summary" in header
    assert "Tags: meeting, retro" in header
    assert "Projects: SuperFit" in header
    assert "Area: Engineering" in header
    assert "Date: 2026-04-10" in header


def test_build_metadata_header_empty():
    """Empty metadata returns empty string."""
    header = build_metadata_header({}, None)
    assert header == ""


def test_build_metadata_header_title_only():
    """Only title populated — no trailing pipes or empty fields."""
    header = build_metadata_header({"title": "Test"}, None)
    assert header == "[Title: Test]"


# --- strip_html ---


def test_strip_html_basic_tags():
    """Standard HTML tags are removed, text preserved."""
    assert strip_html("<p>hello</p>") == "hello"
    assert strip_html("<b>bold</b> text") == "bold text"


def test_strip_html_no_html():
    """Plain text passes through unchanged (fast path: no '<')."""
    text = "Just a plain paragraph with no tags."
    assert strip_html(text) == text


def test_strip_html_preserves_link_text():
    """Link text is kept, tag and attributes removed."""
    result = strip_html('<a href="https://example.com">click here</a>')
    assert result == "click here"


def test_strip_html_nested():
    """Nested tags are fully stripped."""
    result = strip_html("<div><p>inner <em>emphasis</em></p></div>")
    assert result == "inner emphasis"


# --- _split_by_headings ---


def test_split_by_headings_multiple_levels():
    """Content with h1, h2, h3 headings splits at each boundary."""
    body = "# Intro\nParagraph 1\n## Details\nParagraph 2\n### Sub\nParagraph 3"
    sections = _split_by_headings(body)

    assert len(sections) == 3
    assert sections[0][0] == "Intro"
    assert "Paragraph 1" in sections[0][1]
    assert sections[1][0] == "Details"
    assert sections[2][0] == "Sub"


def test_split_by_headings_no_headings():
    """Content with no headings returns single (None, body) tuple."""
    body = "Just a paragraph.\n\nAnother paragraph."
    sections = _split_by_headings(body)

    assert len(sections) == 1
    assert sections[0][0] is None
    assert sections[0][1] == body


def test_split_by_headings_content_before_first_heading():
    """Pre-heading content is captured as a (None, text) section."""
    body = "Preamble text\n\n## First Section\nContent here."
    sections = _split_by_headings(body)

    assert len(sections) == 2
    assert sections[0][0] is None
    assert "Preamble" in sections[0][1]
    assert sections[1][0] == "First Section"


# --- _split_section ---


def test_split_section_small():
    """Section under TARGET_CHUNK_SIZE stays as one chunk."""
    text = "Short paragraph."
    chunks = _split_section(text, "Heading")
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_section_large_with_overlap():
    """Large section splits into overlapping chunks."""
    # Create a section well over TARGET_CHUNK_SIZE
    paragraphs = [f"Paragraph {i}. " + "x" * 400 for i in range(10)]
    text = "\n\n".join(paragraphs)
    assert len(text) > TARGET_CHUNK_SIZE

    chunks = _split_section(text, "Big Section")
    assert len(chunks) > 1

    # Each chunk after the first should start with overlap content from prev chunk
    # Verify the second chunk contains text that also appears in the first
    if len(chunks) >= 2:
        # Last paragraph of first chunk should appear at start of second chunk
        first_paras = chunks[0].split("\n\n")
        second_paras = chunks[1].split("\n\n")
        # The overlap should include at least one paragraph from the first chunk
        assert any(p in second_paras for p in first_paras), \
            "Expected overlapping paragraphs between adjacent chunks"


# --- chunk_markdown (integration) ---


def test_chunk_markdown_full_note(sample_markdown):
    """A complete note with headings produces multiple chunks with metadata."""
    chunks = chunk_markdown("notes/retro.md", sample_markdown)

    assert len(chunks) > 0
    # Every chunk should have the file_path and sequential chunk_index
    for i, chunk in enumerate(chunks):
        assert chunk["file_path"] == "notes/retro.md"
        assert chunk["chunk_index"] == i
        assert chunk["title"] == "Sprint Retrospective"
        assert "text_to_embed" in chunk
        assert len(chunk["text_to_embed"]) > 0


def test_chunk_markdown_empty_body():
    """Frontmatter-only note (no body text) returns empty list."""
    content = "---\ntitle: Empty\n---\n"
    chunks = chunk_markdown("notes/empty.md", content)
    assert chunks == []


def test_chunk_markdown_no_frontmatter():
    """Note with no frontmatter still chunks the body."""
    content = "# My Note\n\nSome content here.\n\n## Section 2\n\nMore content."
    chunks = chunk_markdown("notes/bare.md", content)

    assert len(chunks) >= 2
    assert chunks[0]["title"] == ""


def test_chunk_markdown_preserves_metadata_in_embed():
    """Metadata header is prepended to text_to_embed."""
    content = "---\ntitle: Test\ntags:\n  - python\n---\n\n## Intro\n\nHello world."
    chunks = chunk_markdown("test.md", content)

    assert len(chunks) >= 1
    embed_text = chunks[0]["text_to_embed"]
    assert "Title: Test" in embed_text
    assert "Tags: python" in embed_text
    assert "Hello world" in embed_text


# --- Property-based tests ---


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(body=st.text(min_size=10, max_size=5000, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z"),
)))
@settings(max_examples=100)
def test_parse_frontmatter_never_crashes(body):
    """parse_frontmatter should never raise on arbitrary input."""
    meta, result_body = parse_frontmatter(body)
    assert isinstance(meta, dict)
    assert isinstance(result_body, str)


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=100)
def test_strip_html_never_crashes(text):
    """strip_html should never raise on arbitrary input."""
    result = strip_html(text)
    assert isinstance(result, str)
