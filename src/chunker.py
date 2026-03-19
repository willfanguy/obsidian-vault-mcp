"""Markdown-aware chunking that respects note structure."""

import re
from html.parser import HTMLParser
import yaml
import logging

logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE = 2000  # chars, roughly 512 tokens
OVERLAP_SIZE = 200  # chars of overlap between adjacent chunks
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class _HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML, preserving link text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(text: str) -> str:
    """Remove HTML tags from text, keeping the readable content."""
    if "<" not in text:
        return text
    extractor = _HTMLTextExtractor()
    try:
        extractor.feed(text)
        return extractor.get_text()
    except Exception:
        # If parsing fails, fall back to regex strip
        return re.sub(r"<[^>]+>", "", text)


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


def build_metadata_header(metadata: dict, heading: str | None) -> str:
    """Build a bracketed metadata header for embedding context.

    Prepended to chunk content before embedding so the vector captures
    metadata signals (title, tags, projects, etc.) alongside body text.
    """
    parts = []
    if metadata.get("title"):
        parts.append(f"Title: {metadata['title']}")
    if heading:
        parts.append(f"Section: {heading}")
    tags = metadata.get("tags", [])
    if tags and isinstance(tags, list):
        parts.append(f"Tags: {', '.join(str(t) for t in tags if t)}")
    projects = metadata.get("projects", [])
    if projects and isinstance(projects, list):
        parts.append(f"Projects: {', '.join(str(p) for p in projects if p)}")
    if metadata.get("area"):
        parts.append(f"Area: {metadata['area']}")
    if metadata.get("source"):
        parts.append(f"Source: {metadata['source']}")
    if metadata.get("date"):
        parts.append(f"Date: {metadata['date']}")
    return f"[{' | '.join(parts)}]" if parts else ""


def chunk_markdown(file_path: str, content: str, file_mtime: float | None = None) -> list[dict]:
    """Split a markdown file into chunks respecting heading structure."""
    metadata, body = parse_frontmatter(content)
    fields = extract_metadata_fields(metadata)

    # Extract date for embedding context (frontmatter > mtime)
    date_str = ""
    for key in ("date created", "dateCreated", "date", "created"):
        val = metadata.get(key)
        if val:
            date_str = str(val).split(",")[0].split("T")[0].strip()
            break
    if not date_str and file_mtime:
        from datetime import datetime
        date_str = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
    fields["date"] = date_str

    if not body.strip():
        return []

    # Strip HTML from clipped content before chunking
    body = strip_html(body)

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
    """Split a large section into paragraph-based chunks with overlap.

    Adjacent chunks share ~OVERLAP_SIZE chars at boundaries so that
    ideas straddling a split aren't lost to either embedding.
    """
    paragraphs = re.split(r"\n\s*\n", text)

    # Build non-overlapping groups of paragraphs first
    groups: list[list[str]] = []
    current_paras: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > TARGET_CHUNK_SIZE and current_paras:
            groups.append(current_paras)
            current_paras = [para]
            current_len = len(para)
        else:
            current_paras.append(para)
            current_len += len(para) + 2  # +2 for \n\n join

    if current_paras:
        groups.append(current_paras)

    if len(groups) <= 1:
        return ["\n\n".join(g) for g in groups]

    # Build overlapping chunks: pull trailing paragraphs from the
    # previous group into the start of the next group
    chunks = []
    for i, group in enumerate(groups):
        if i == 0:
            chunks.append("\n\n".join(group))
        else:
            # Grab overlap from the end of the previous group
            # Always include at least one paragraph for continuity
            prev = groups[i - 1]
            overlap_paras = []
            overlap_len = 0
            for para in reversed(prev):
                if overlap_paras and overlap_len + len(para) > OVERLAP_SIZE:
                    break
                overlap_paras.insert(0, para)
                overlap_len += len(para) + 2

            chunks.append("\n\n".join(overlap_paras + group))

    return chunks


def _make_chunk(
    file_path: str,
    chunk_index: int,
    heading: str | None,
    content: str,
    metadata: dict,
) -> dict:
    """Create a chunk dict ready for embedding and storage."""
    header = build_metadata_header(metadata, heading)
    return {
        "file_path": file_path,
        "chunk_index": chunk_index,
        "heading": heading or "",
        "content": content,
        "text_to_embed": f"{header}\n\n{content}" if header else content,
        "title": metadata.get("title", ""),
        "tags": metadata.get("tags", []),
        "projects": metadata.get("projects", []),
        "area": metadata.get("area", ""),
        "status": metadata.get("status", ""),
        "source": metadata.get("source", ""),
    }
