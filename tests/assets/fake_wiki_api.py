#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD-2-Clause License
#

"""A stand-in knowledge base HTTP API, served on loopback.

Tests and eval scenarios point the application's ``WIKI_API_BASE_URL`` at this
server instead of at an ``llm_wiki`` desktop application, so they run without
one installed. It answers the two routes the application uses:

- ``POST /api/v1/projects/current/search``, with the ranked hits, each carrying
  the text of its page.
- ``GET /api/v1/projects/current/files/content``, with one page's text.

A test runs it in its own event loop through :class:`FakeWikiAPI`. A process
that has to be started from outside a test — an eval scenario driving a real bot
— runs :func:`main` instead, and configures the server through the environment:

- ``FAKE_WIKI_PORT``: port to listen on. ``0``, the default, binds a free one and
  writes it to ``FAKE_WIKI_PORT_FILE``.
- ``FAKE_WIKI_TOKEN``: the token the API accepts. Unset accepts any token.
- ``FAKE_WIKI_RESULTS``: JSON array of the hits the search returns. Defaults to
  :data:`DEFAULT_RESULTS`.
- ``FAKE_WIKI_STATUS``: status to answer with, for ``401`` standing in for a
  token the API rejects.
- ``FAKE_WIKI_OMIT_CONTENT``: answer the search without page text, the way an API
  that ignores ``includeContent`` would.
- ``FAKE_WIKI_DELAY_SECS``: how long to take before answering, for a lookup that
  has to be abandoned.
- ``FAKE_WIKI_REQUESTS``: path to append one JSON line per request to.

A knowledge base that cannot be reached needs no server at all: point the
application at a port nothing listens on.
"""

import asyncio
import json
import os
from pathlib import Path
from types import TracebackType
from typing import Any

from aiohttp import web

SEARCH_ROUTE = "/api/v1/projects/current/search"
CONTENT_ROUTE = "/api/v1/projects/current/files/content"

# The range the real API accepts a search size in, and rejects anything outside
# of. Enforced here so a test's search request is the one the real API would take
# rather than one it would refuse.
TOPK_MIN = 1
TOPK_MAX = 50

# The answer line sits in the page text and not in the snippet: a lookup that
# reads snippets alone cannot answer this question, which is the failure the
# full-page read exists to prevent.
DEFAULT_RESULTS = [
    {
        "path": "wiki/entities/axle-counter.md",
        "title": "Axle counter",
        "snippet": (
            "## Alarms in the source\n\n| Alarm | Level | Effect |\n| --- | --- | --- |\n"
            "| Axle counter and MSS communication | 3 | BDMS reports n"
        ),
        "score": 30.0,
        "content": (
            "---\ntype: entity\ntitle: Axle counter\n---\n\n# Axle counter\n\n"
            "## Overview\n\n- Output boards, wheel sensors S1/S2, R1/R0 cubicles.\n\n"
            "## Alarms in the source\n\n| Alarm | Level | Effect |\n| --- | --- | --- |\n"
            "| Axle counter and MSS communication | 3 | BDMS reports no axle counter alarm |\n"
            "| Output board failure | 2 | The section this board serves reads occupied |\n"
            "| Axle counter section unknown | 4 | Every section of the cabinet reads "
            "occupied |\n"
        ),
    },
    {
        "path": "wiki/sources/bdms-alarms.md",
        "title": "BDMS alarms",
        "snippet": "| Axle counter section unknown | 4 |",
        "score": 16.0,
        "content": (
            "# BDMS alarms\n\n"
            "## Levels\n\nLevels run from 1, the most severe, to 4.\n\n"
            "## Axle counter\n\nAxle counter section unknown is level 4.\n"
        ),
    },
    {
        "path": "wiki/index.md",
        "title": "Wiki index",
        "snippet": "- [[Axle counter]]\n- [[BDMS]]",
        "score": 3.0,
        "content": "# Wiki index\n\n- [[Axle counter]]\n- [[BDMS]]\n",
    },
]


class FakeWikiAPI:
    """A knowledge base API a test can point the application at."""

    def __init__(
        self,
        *,
        token: str | None = None,
        results: list[dict[str, Any]] | None = None,
        status: int = 200,
        include_content: bool = True,
        delay_secs: float = 0.0,
        request_log: str | None = None,
    ) -> None:
        """Initialize the API.

        Args:
            token: Token the API accepts. Any token is accepted when ``None``.
            results: Hits the search returns, in the order it returns them. Each
                hit is a mapping with ``path``, ``title``, ``snippet``, ``score``,
                optionally ``content``, and optionally ``content_status``, the
                status the content route answers that page's read with, for a page
                that cannot be read.
            status: Status every request is answered with, for a failure the
                application has to survive.
            include_content: Whether the search carries each hit's page text. When
                ``False`` the text is served by the content route alone, which is
                the shape of an API that ignores ``includeContent``.
            delay_secs: How long to wait before answering, so a lookup can be
                abandoned mid-flight.
            request_log: Path to append one JSON line per request to. A test reads
                :attr:`requests` instead.
        """
        self._token = token
        self._results = DEFAULT_RESULTS if results is None else results
        self._status = status
        self._include_content = include_content
        self._delay_secs = delay_secs
        self._request_log = request_log
        self._runner: web.AppRunner | None = None
        self._base_url: str | None = None
        self.requests: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        """Root of the API, once it is listening.

        Raises:
            RuntimeError: The API has not been started.
        """
        if self._base_url is None:
            raise RuntimeError("the fake knowledge base API is not listening")
        return self._base_url

    async def __aenter__(self) -> "FakeWikiAPI":
        """Start listening."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop listening."""
        await self.aclose()

    async def start(self, port: int = 0) -> str:
        """Start the API on ``port`` and return its base URL.

        Args:
            port: Port to bind on loopback. ``0`` binds a free one.

        Returns:
            The API's base URL.
        """
        # A handler that is still waiting out its delay is abandoned rather than
        # waited for: a caller that stopped listening has no use for the answer,
        # and the test that stopped it does not wait out the delay either.
        self._runner = web.AppRunner(self._app(), shutdown_timeout=0.25)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", port)
        await site.start()
        self._base_url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"
        return self._base_url

    async def aclose(self) -> None:
        """Stop listening."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def _app(self) -> web.Application:
        """Build the application with the two routes the API serves."""
        app = web.Application()
        app.router.add_post("/api/v1/projects/current/search", self._search)
        app.router.add_get("/api/v1/projects/current/files/content", self._content)
        return app

    async def _search(self, request: web.Request) -> web.Response:
        """Answer a query with the ranked hits."""
        body = await request.json()
        self._record(request, body)
        if not self._authorized(request):
            return web.json_response({"ok": False, "error": "Unauthorized"}, status=401)
        if self._status != 200:
            return web.json_response({"ok": False, "error": "failed"}, status=self._status)
        top_k = body.get("topK")
        if not isinstance(top_k, int) or not TOPK_MIN <= top_k <= TOPK_MAX:
            # Answered apart from the rejection an unaccepted token gets, so a
            # search the real API would refuse for its size is not read as an
            # authentication failure.
            return web.json_response(
                {"ok": False, "error": f"topK must be between {TOPK_MIN} and {TOPK_MAX}"},
                status=400,
            )
        await asyncio.sleep(self._delay_secs)
        results = [{**hit} for hit in self._results]
        if not self._include_content:
            for hit in results:
                hit.pop("content", None)
        return web.json_response({"ok": True, "mode": "hybrid", "results": results})

    async def _content(self, request: web.Request) -> web.Response:
        """Answer with one page's text."""
        self._record(request, None)
        if not self._authorized(request):
            return web.json_response({"ok": False, "error": "Unauthorized"}, status=401)
        if self._status != 200:
            return web.json_response({"ok": False, "error": "failed"}, status=self._status)
        path = request.query.get("path")
        for hit in self._results:
            if hit["path"] == path:
                status = hit.get("content_status", 200)
                if status != 200:
                    return web.json_response({"ok": False, "error": "failed"}, status=status)
                return web.json_response(
                    {
                        "ok": True,
                        "path": path,
                        "projectId": "current",
                        "content": hit.get("content", ""),
                    }
                )
        return web.json_response({"ok": False, "error": "not found"}, status=404)

    def _authorized(self, request: web.Request) -> bool:
        """Whether the request carries the token the API accepts."""
        if self._token is None:
            return True
        return request.headers.get("Authorization") == f"Bearer {self._token}"

    def _record(self, request: web.Request, body: Any) -> None:
        """Record one request, for a test to assert what the application sent."""
        record = {
            "method": request.method,
            "path": request.path,
            "authorization": request.headers.get("Authorization"),
            "query": dict(request.query),
            "body": body,
        }
        self.requests.append(record)
        if self._request_log:
            # Opened per request so a test or a scenario can read the log while
            # the server is running.
            with Path(self._request_log).open("a", encoding="utf-8") as log:
                log.write(json.dumps(record) + "\n")


async def serve(api: FakeWikiAPI, port: int, port_file: str | None) -> None:
    """Serve the API until the process is stopped.

    Args:
        api: The API to serve.
        port: Port to bind on loopback. ``0`` binds a free one.
        port_file: Path to write the bound port to, once it is bound.
    """
    base_url = await api.start(port)
    if port_file:
        Path(port_file).write_text(base_url.rsplit(":", 1)[1], encoding="utf-8")
    print(f"fake knowledge base API listening on {base_url}", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    """Run the API with the settings the environment declares."""
    results = os.environ.get("FAKE_WIKI_RESULTS")
    omit_content = os.environ.get("FAKE_WIKI_OMIT_CONTENT", "").lower() in ("1", "true", "yes")
    api = FakeWikiAPI(
        token=os.environ.get("FAKE_WIKI_TOKEN") or None,
        results=json.loads(results) if results else None,
        status=int(os.environ.get("FAKE_WIKI_STATUS", "200")),
        include_content=not omit_content,
        delay_secs=float(os.environ.get("FAKE_WIKI_DELAY_SECS", "0")),
        request_log=os.environ.get("FAKE_WIKI_REQUESTS"),
    )
    asyncio.run(
        serve(
            api,
            port=int(os.environ.get("FAKE_WIKI_PORT", "0")),
            port_file=os.environ.get("FAKE_WIKI_PORT_FILE"),
        )
    )


if __name__ == "__main__":
    main()
