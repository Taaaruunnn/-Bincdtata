"""Binance USDS-M futures REST helpers shared by capture.py (snapshot
bootstrap, embedded raw into the capture stream) and reconstruct.py
(verification mode).

Fetching and storing a REST response VERBATIM is not "parsing" in the sense
the market-data-integrity skill warns against for the capture process — no
field of the response is read or interpreted here, the raw body is simply
moved to disk (capture.py) or handed to the caller as text (reconstruct.py's
verification mode parses it, but that process's whole job is interpretation).

Endpoint, query params, response shape and per-limit weight verified live
2026-09-13 from:
  https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book
    GET /fapi/v1/depth?symbol=<SYM>&limit=<N>
    valid limits: 5, 10, 20, 50, 100, 500, 1000
    weights:      2, 2,  2,  2,  5,   10,  20

Account-wide budget, confirmed live via GET /fapi/v1/exchangeInfo ->
rateLimits on the same date:
    REQUEST_WEIGHT: 2400 per MINUTE per IP.
A 418 (IP auto-ban, 2min-3day escalating) follows repeated 429s — see
https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info
(retrieved 2026-09-13). Resync storms during volatile periods are exactly
when you are most tempted to hammer this endpoint and exactly when getting
banned costs the most data — budget checks below are deliberately
conservative.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

REST_BASE_URL = "https://fapi.binance.com"
DEPTH_ENDPOINT = "/fapi/v1/depth"

DEPTH_LIMIT_WEIGHTS: dict[int, int] = {5: 2, 10: 2, 20: 2, 50: 2, 100: 5, 500: 10, 1000: 20}
REQUEST_WEIGHT_PER_MINUTE_LIMIT = 2400  # confirmed live 2026-09-13, see module docstring


class SnapshotFetchError(RuntimeError):
    pass


def depth_snapshot_url(symbol: str, limit: int = 1000) -> str:
    if limit not in DEPTH_LIMIT_WEIGHTS:
        raise ValueError(f"limit must be one of {sorted(DEPTH_LIMIT_WEIGHTS)}, got {limit}")
    return f"{REST_BASE_URL}{DEPTH_ENDPOINT}?symbol={symbol.upper()}&limit={limit}"


def fetch_depth_snapshot_sync(symbol: str, limit: int = 1000, timeout: float = 10.0) -> str:
    """Blocking fetch of the raw JSON response body. Call via
    `fetch_depth_snapshot` (asyncio.to_thread wrapper) from async code —
    urllib is used here specifically so this module adds no new dependency
    beyond the standard library."""
    url = depth_snapshot_url(symbol, limit)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SnapshotFetchError(f"{symbol}: HTTP {exc.code} fetching depth snapshot: {body}") from exc
    except urllib.error.URLError as exc:
        raise SnapshotFetchError(f"{symbol}: failed to fetch depth snapshot: {exc}") from exc


async def fetch_depth_snapshot(symbol: str, limit: int = 1000, timeout: float = 10.0) -> str:
    return await asyncio.to_thread(fetch_depth_snapshot_sync, symbol, limit, timeout)


def projected_weight_per_minute(n_symbols: int, interval_s: float, limit: int = 1000) -> float:
    """Weight/minute consumed by periodically snapshotting `n_symbols` every
    `interval_s` seconds. Used to fail loudly at config time rather than
    discover a 429/418 mid-capture."""
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    cycles_per_minute = 60.0 / interval_s
    return n_symbols * DEPTH_LIMIT_WEIGHTS[limit] * cycles_per_minute
