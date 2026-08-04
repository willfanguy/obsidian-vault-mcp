"""Tests for APIKeyMiddleware — the only thing standing between a public
Tailscale funnel and the entire vault.

These drive the raw ASGI callable directly (no pytest-asyncio: asyncio.run is
enough for a single request) and assert on observable behaviour — the status
sent downstream, and the scope the inner app actually receives — rather than on
the middleware's internals.

The two credentials under test:
  1. Authorization: Bearer <API_KEY>
  2. A secret leading path segment (<FUNNEL_PREFIX>), stripped before forwarding

API_KEY and FUNNEL_PREFIX are read from env at import time and bound as module
globals, so tests monkeypatch the module attributes, not the environment.
"""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from src import server
from src.server import APIKeyMiddleware


SECRET_PREFIX = "/v-0123456789abcdef0123456789abcdef"
TOKEN = "s3cret-token"


class RecordingApp:
    """Inner ASGI app that records the scope it was handed and returns 200."""

    def __init__(self):
        self.scopes = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"FORWARDED"})


def drive(path, headers=None, scope_type="http", raw_path=None):
    """Send one request through the middleware.

    Returns (status, body, inner_app). status is None when the inner app
    handled it, since RecordingApp reports its own 200 the same way — check
    inner_app.scopes to distinguish forwarded from rejected.
    """
    inner = RecordingApp()
    mw = APIKeyMiddleware(inner)

    scope = {"type": scope_type, "path": path}
    if raw_path is not None:
        scope["raw_path"] = raw_path
    scope["headers"] = [
        (k.encode(), v.encode()) for k, v in (headers or {}).items()
    ]

    sent = []

    async def send(msg):
        sent.append(msg)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    asyncio.run(mw(scope, receive, send))

    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body, inner


def forwarded_path(inner):
    """The path the inner app saw, or None if it was never called."""
    return inner.scopes[0]["path"] if inner.scopes else None


# --- Bearer credential ---------------------------------------------------


def test_valid_bearer_forwards_path_untouched(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    status, body, inner = drive("/mcp", {"authorization": f"Bearer {TOKEN}"})

    assert body == b"FORWARDED"
    assert forwarded_path(inner) == "/mcp"
    assert status == 200


def test_wrong_bearer_gets_401_not_404(monkeypatch):
    """A caller that tried and failed deserves 'your token is wrong'."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    status, _, inner = drive("/mcp", {"authorization": "Bearer wrong"})

    assert status == 401
    assert inner.scopes == []


def test_anonymous_gets_404_not_401(monkeypatch):
    """404 keeps MCP clients from inferring an OAuth server that isn't here."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    status, _, inner = drive("/mcp")

    assert status == 404
    assert inner.scopes == []


def test_oauth_discovery_probe_at_root_gets_404(monkeypatch):
    """Claude probes these when adding a connector; a 401 broke setup before."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    for probe in (
        "/.well-known/oauth-authorization-server",
        "/.well-known/oauth-protected-resource",
    ):
        status, _, inner = drive(probe)
        assert status == 404, probe
        assert inner.scopes == []


def test_health_check_is_open(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    status, body, inner = drive("/")

    assert status == 200
    assert b"obsidian-vault-search" in body
    assert inner.scopes == []  # answered by the middleware, never forwarded


def test_empty_api_key_does_not_admit_bare_bearer(monkeypatch):
    """With no key configured, 'Bearer ' must not become a valid credential."""
    monkeypatch.setattr(server, "API_KEY", "")
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    status, _, inner = drive("/mcp", {"authorization": "Bearer "})

    assert inner.scopes == []
    assert status == 401


# --- Secret-path credential ----------------------------------------------


def test_secret_prefix_is_stripped_before_forwarding(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    status, body, inner = drive(f"{SECRET_PREFIX}/mcp")

    assert body == b"FORWARDED"
    assert forwarded_path(inner) == "/mcp"
    assert status == 200


def test_trailing_slash_is_normalised_away(monkeypatch):
    """FastMCP mounts at /mcp, not /mcp/. Forwarding '/mcp/' would earn a 307
    whose Location is rebuilt from the rewritten path — dropping the secret and
    sending the client into a 401. Normalise instead."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, body, inner = drive(f"{SECRET_PREFIX}/mcp/")

    assert body == b"FORWARDED"
    assert forwarded_path(inner) == "/mcp"


def test_raw_path_is_rewritten_alongside_path(monkeypatch):
    """Starlette routes on raw_path when present; a stale one re-leaks the
    secret into the app and breaks routing."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, _, inner = drive(
        f"{SECRET_PREFIX}/mcp", raw_path=f"{SECRET_PREFIX}/mcp".encode()
    )

    assert inner.scopes[0]["raw_path"] == b"/mcp"
    assert SECRET_PREFIX.encode() not in inner.scopes[0]["raw_path"]


def test_prefix_alone_forwards_root(monkeypatch):
    """Boundary: no tail at all. Must not crash or produce an empty path."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, _, inner = drive(SECRET_PREFIX)

    assert forwarded_path(inner) == "/"


def test_segment_merely_starting_with_secret_is_rejected(monkeypatch):
    """The startswith bug: /v-abc...defEXTRA must not authenticate."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    status, _, inner = drive(f"{SECRET_PREFIX}EXTRA/mcp")

    assert inner.scopes == []
    assert status == 404


def test_secret_in_later_segment_is_rejected(monkeypatch):
    """Only the FIRST segment is a credential."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    status, _, inner = drive(f"/mcp{SECRET_PREFIX}")

    assert inner.scopes == []
    assert status == 404


def test_prefix_disabled_rejects_everything_unauthenticated(monkeypatch):
    """Blank FUNNEL_PREFIX must not turn into a match-anything credential."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", "")

    for path in (f"{SECRET_PREFIX}/mcp", "//mcp", "/mcp", "/", "//"):
        if path == "/":
            continue  # open health check, covered separately
        status, _, inner = drive(path)
        assert inner.scopes == [], path
        assert status == 404, path


def test_prefix_credential_does_not_disturb_bearer_clients(monkeypatch):
    """Enabling the prefix must leave header-based LAN clients untouched."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, body, inner = drive("/mcp", {"authorization": f"Bearer {TOKEN}"})

    assert body == b"FORWARDED"
    assert forwarded_path(inner) == "/mcp"


# --- Hostile / malformed input -------------------------------------------


def test_non_ascii_authorization_header_does_not_raise(monkeypatch):
    """hmac.compare_digest rejects non-ASCII str; the encode() guard matters."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    status, _, inner = drive("/mcp", {"authorization": "Bearer ünïcodé-🔑"})

    assert status == 401
    assert inner.scopes == []


def test_non_ascii_path_segment_does_not_raise(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    status, _, inner = drive("/ünïcodé-🔑/mcp")

    assert status == 404
    assert inner.scopes == []


def test_lifespan_scope_passes_through_untouched(monkeypatch):
    """FastMCP's http_app has a lifespan; swallowing it would break startup."""
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, _, inner = drive("/mcp", scope_type="lifespan")

    assert len(inner.scopes) == 1
    assert inner.scopes[0]["type"] == "lifespan"


def test_websocket_upgrade_still_requires_a_credential(monkeypatch):
    monkeypatch.setattr(server, "API_KEY", TOKEN)
    monkeypatch.setattr(server, "FUNNEL_PREFIX", SECRET_PREFIX)

    _, _, inner = drive("/mcp", scope_type="websocket")

    assert inner.scopes == []


@settings(max_examples=200, deadline=None)
@given(
    segment=st.text(
        alphabet=st.characters(blacklist_characters="/"), min_size=0, max_size=60
    )
)
def test_no_segment_other_than_the_secret_ever_authenticates(segment):
    """Property: for any first segment that isn't exactly the secret, the
    request is never forwarded. Covers empty, whitespace, near-misses,
    case variants and control characters in one sweep."""
    original_key = server.API_KEY
    original_prefix = server.FUNNEL_PREFIX
    server.API_KEY = TOKEN
    server.FUNNEL_PREFIX = SECRET_PREFIX
    try:
        if "/" + segment == SECRET_PREFIX:
            return
        _, _, inner = drive(f"/{segment}/mcp")
        assert inner.scopes == []
    finally:
        server.API_KEY = original_key
        server.FUNNEL_PREFIX = original_prefix
