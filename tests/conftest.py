"""Shared fixtures for obsidian-vault-mcp tests."""

import pytest


@pytest.fixture
def sample_markdown():
    """A realistic Obsidian note with frontmatter, headings, and mixed content."""
    return """\
---
title: "Sprint Retrospective"
date: 2026-04-10
tags:
  - "[[meeting]]"
  - "[[retro]]"
projects:
  - "[[SuperFit]]"
area: "[[Glassdoor]]"
status: summarized
source: "[[Meeting Notes]]"
---

## Summary

The team discussed what went well and what to improve.

## What Went Well

- Shipped the new search feature on time
- Good collaboration between design and engineering
- Test coverage improved by 15%

## What To Improve

- Deploy pipeline is too slow (45 min average)
- Need better documentation for the API
- Some PRs sat in review for 3+ days

## Action Items

1. Set up parallel CI builds
2. Create API docs template
3. Add review SLA to team agreement
"""


@pytest.fixture
def sample_frontmatter():
    """Various frontmatter dicts for testing extract_metadata_fields."""
    return {
        "full": {
            "title": "Test Note",
            "tags": ["python", "testing"],
            "projects": ["[[SuperFit]]", "[[ARC]]"],
            "status": "open",
            "area": "[[Engineering]]",
            "source": "[[Meeting Notes]]",
        },
        "wikilinks_only": {
            "title": "Wiki Note",
            "tags": ["[[tag-one]]", "[[tag-two]]"],
            "projects": ["[[Project A]]"],
            "area": "[[Area X]]",
        },
        "empty": {},
        "nones": {
            "title": None,
            "tags": None,
            "projects": None,
            "status": None,
            "area": None,
        },
    }
