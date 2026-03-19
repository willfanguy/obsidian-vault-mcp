"""Markdown-aware chunking that respects note structure."""

import re
import yaml
import logging

logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE = 2000  # chars, roughly 512 tokens
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and return (metadata, body)."""
    if not content.startswith("---"):
        return {}, content

    end = content.find("\n---", 3)
    if end == -1:
        return {}, content

    yaml_str = content[3:end].strip()
    body = content[end + 4 :].strip()

    try:
        metadata = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, body


def extract_metadata_fields(fm: dict) -> dict:
    """Pull out the fields we care about for search metadata."""

    def clean_wikilinks(val):
        """Strip [[]] from wikilink values."""
        if isinstance(val, str):
            return re.sub(r"\[\[([^\]]+)\]\]", r"\1", val)
        if isinstance(val, list):
            return [clean_wikilinks(v) for v in val]
        return val

    def safe_str(val):
        if val is None:
            return ""
        return str(val)

    return {
        "title": safe_str(fm.get("title", "")),
        "tags": clean_wikilinks(fm.get("tags", [])) or [],
        "projects": clean_wikilinks(fm.get("projects", [])) or [],
        "status": safe_str(fm.get("status")),
        "area": safe_str(clean_wikilinks(fm.get("area", ""))),
        "source": safe_str(clean_wikilinks(fm.get("source", ""))),
    }


def chunk_markdown(file_path: str, content: str) -> list[dict]:
    """Split a markdown file into chunks respecting heading structure."""
    metadata, body = parse_frontmatter(content)
    fields = extract_metadata_fields(metadata)

    if not body.strip():
        return []

    # Split by headings
    sections = _split_by_headings(body)

    chunks = []
    for heading, section_text in sections:
        if not section_text.strip():
            continue

        # If section is small enough, keep as one chunk
        if len(section_text) <= TARGET_CHUNK_SIZE:
            chunks.append(
                _make_chunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    heading=heading,
                    content=section_text.strip(),
                    metadata=fields,
                )
            )
        else:
            # Split large sections by paragraphs
            for sub_chunk in _split_section(section_text, heading):
                chunks.append(
                    _make_chunk(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        heading=heading,
                        content=sub_chunk.strip(),
                        metadata=fields,
                    )
                )

    return chunks


def _split_by_headings(body: str) -> list[tuple[str | None, str]]:
    """Split body into (heading, content) pairs."""
    matches = list(HEADING_PATTERN.finditer(body))

    if not matches:
        return [(None, body)]

    sections = []
    # Content before first heading
    if matches[0].start() > 0:
        pre = body[: matches[0].start()]
        if pre.strip():
            sections.append((None, pre))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append((heading, body[start:end]))

    return sections


def _split_section(text: str, heading: str | None) -> list[str]:
    """Split a large section into paragraph-based chunks."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > TARGET_CHUNK_SIZE and current:
            chunks.append(current)
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current)

    return chunks


def _make_chunk(
    file_path: str,
    chunk_index: int,
    heading: str | None,
    content: str,
    metadata: dict,
) -> dict:
    """Create a chunk dict ready for embedding and storage."""
    return {
        "file_path": file_path,
        "chunk_index": chunk_index,
        "heading": heading or "",
        "content": content,
        "title": metadata.get("title", ""),
        "tags": metadata.get("tags", []),
        "projects": metadata.get("projects", []),
        "area": metadata.get("area", ""),
        "status": metadata.get("status", ""),
        "source": metadata.get("source", ""),
    }
