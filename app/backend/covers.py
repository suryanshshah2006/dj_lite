"""Cover-art URLs from Spotify's public oEmbed endpoint, cached in SQLite.

- Cache first; fetch only on a miss (callers only ask for tracks being displayed).
- Self rate-limited to ~N requests/second across all callers.
- Failures are cached too: "not found" permanently, transient errors for a TTL.
- Only URLs are stored, never images.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx

TRACK_ID = re.compile(r"^[A-Za-z0-9]{22}$")
OEMBED = "https://open.spotify.com/oembed?url=https://open.spotify.com/track/{id}"


@dataclass
class FetchResult:
    url: str | None
    status: str  # "ok" | "not_found" | "error"


Fetcher = Callable[[str], Awaitable[FetchResult]]


def oembed_fetcher(timeout_s: float) -> Fetcher:
    client = httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": "DJ-Lite (local)"},
                               follow_redirects=True)

    async def fetch(track_id: str) -> FetchResult:
        try:
            r = await client.get(OEMBED.format(id=track_id))
        except httpx.HTTPError:
            return FetchResult(None, "error")
        if r.status_code >= 500 or r.status_code == 429:
            return FetchResult(None, "error")
        if r.status_code != 200:
            return FetchResult(None, "not_found")
        try:
            url = r.json().get("thumbnail_url")
        except ValueError:
            return FetchResult(None, "error")
        return FetchResult(url, "ok") if url else FetchResult(None, "not_found")

    return fetch


class RateLimiter:
    """Spaces calls at least 1/rate seconds apart (async, shared by all callers)."""

    def __init__(self, rate_per_s: float, clock: Callable[[], float] = time.monotonic):
        self.interval = 1.0 / rate_per_s
        self.clock = clock
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = self.clock()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class CoverCache:
    def __init__(self, db_path: Path, fetch: Fetcher, rate_per_s: float = 5.0,
                 failure_ttl_s: float = 24 * 3600):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS covers ("
            " track_id TEXT PRIMARY KEY, url TEXT, status TEXT NOT NULL,"
            " fetched_at REAL NOT NULL)")
        self.db.commit()
        self._db_lock = threading.Lock()
        self.fetch = fetch
        self.limiter = RateLimiter(rate_per_s)
        self.failure_ttl_s = failure_ttl_s
        self._inflight: dict[str, asyncio.Future] = {}
        self.fetch_count = 0

    def _get(self, track_id: str) -> tuple[str | None, str, float] | None:
        with self._db_lock:
            row = self.db.execute("SELECT url, status, fetched_at FROM covers WHERE track_id = ?",
                                  (track_id,)).fetchone()
        return row

    def _put(self, track_id: str, res: FetchResult) -> None:
        with self._db_lock:
            self.db.execute(
                "INSERT OR REPLACE INTO covers (track_id, url, status, fetched_at) "
                "VALUES (?, ?, ?, ?)", (track_id, res.url, res.status, time.time()))
            self.db.commit()

    async def get(self, track_id: str) -> dict:
        """{'url': str | None, 'cached': bool, 'status': str}."""
        if not TRACK_ID.match(track_id):
            raise ValueError("invalid track id")
        row = self._get(track_id)
        if row is not None:
            url, status, fetched_at = row
            fresh = status != "error" or time.time() - fetched_at < self.failure_ttl_s
            if fresh:
                return {"url": url, "cached": True, "status": status}
        # De-duplicate concurrent requests for the same id.
        if track_id in self._inflight:
            res = await self._inflight[track_id]
            return {"url": res.url, "cached": True, "status": res.status}
        fut = asyncio.get_running_loop().create_future()
        self._inflight[track_id] = fut
        try:
            await self.limiter.wait()
            self.fetch_count += 1
            res = await self.fetch(track_id)
            self._put(track_id, res)
            fut.set_result(res)
        except BaseException as exc:  # pragma: no cover - surfaced to waiters
            fut.set_exception(exc)
            raise
        finally:
            self._inflight.pop(track_id, None)
        return {"url": res.url, "cached": False, "status": res.status}

    def close(self) -> None:
        with self._db_lock:
            self.db.close()
