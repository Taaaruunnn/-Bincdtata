"""Binance USDS-M futures raw market data capture daemon.

This process does exactly one thing: receive bytes from Binance's combined
WebSocket stream and write them to disk, unparsed, with two timestamps
attached. It performs no book maintenance, no sequence validation, and no
analysis (see the market-data-integrity skill's "capture raw, reconstruct
offline" split). All of that lives in reconstruct.py, which replays these
files.

Why this split matters in practice, not just in principle:
  - A bug in reconstruction logic costs you a re-run. A bug in capture-time
    parsing costs you the data — raw history from Binance's public streams is
    not re-downloadable after the fact.
  - The hot path (recv -> timestamp -> write) stays trivially simple and
    fast regardless of how complex the offline analysis becomes later.

CAPTURE FORMAT (see also README.md, which is the durable reference — this
docstring is a pointer to it, not a substitute for it, since the code will
change and the file format needs to keep meaning something):

Each output file is a zstd-compressed, newline-delimited JSON (JSONL) file.
The zstd stream is periodically closed with FLUSH_FRAME (see
_RotatingZstdWriter), producing a file made of multiple concatenated zstd
frames rather than one long-lived frame — verified empirically: a bare
`ZstdDecompressor().decompress(data)` on such a file only returns the FIRST
frame. Readers MUST use `ZstdDecompressor().stream_reader(fh)`, which reads
across all concatenated frames transparently. This is deliberate: it bounds
data loss on an unclean shutdown to at most one flush interval, at the cost
of requiring a streaming reader rather than a one-shot decompress() call.

Each JSONL line is one envelope object:
    {
      "schema": "capture.v1",
      "type": "message" | "gap" | "snapshot",
      "stream": "<binance stream name>" | "<symbol>@depth_snapshot" | null,
      "recv_wall_ns": <int, time.time_ns() at receipt>,
      "recv_mono_ns": <int, time.monotonic_ns() at receipt>,
      "raw": "<verbatim WS frame text, or verbatim REST response body>",  # type in {"message","snapshot"}
      "reason": "<initial_connect|reconnect>"                             # type=="gap" only
    }

`raw` is the exact text Binance sent — for `type=="message"`, the combined
stream envelope (`{"stream":"<name>","data":{...}}`); for `type=="snapshot"`,
the REST `/fapi/v1/depth` response body. We never call json.loads on either
in this process (see binance_rest.py for why storing a REST body verbatim is
not "parsing" in the sense that matters here). The "stream" field at our
envelope level is filled in only by cheaply slicing the leading
`"stream":"..."` prefix (see `_extract_stream_name`), which is string
slicing for filing/logging convenience, not interpretation of the payload;
if it fails for any reason the message is still captured with stream=null
rather than dropped.

`type=="snapshot"` records exist because Binance's REST depth endpoint only
ever returns the CURRENT book — there is no historical snapshot endpoint. An
offline reconstructor replaying old capture files therefore needs a snapshot
that was captured close in time to the diff stream it's bootstrapping, which
is why the capture daemon fetches one itself (see `_snapshot_task`) rather
than leaving that to whoever runs the reconstructor later.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import websockets
import zstandard as zstd

import binance_rest

LOG = logging.getLogger("capture")

CAPTURE_SCHEMA_VERSION = "capture.v1"
DEFAULT_CHANNELS: tuple[str, ...] = ("depth@100ms", "aggTrade")
COMBINED_STREAM_BASE_URL = "wss://fstream.binance.com/stream"

#: Binance closes any single WS connection at the 24h mark and allows at most
#: 1024 streams on one connection; see
#: https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Connect
#: (retrieved 2026-09-13). We use one combined-stream connection for the
#: whole symbol/channel list, which comfortably fits under 1024 for any
#: realistic symbol count, and treat the mandatory 24h disconnect the same
#: way as any other disconnect: reconnect with backoff, emit a gap marker.
MAX_STREAMS_PER_CONNECTION = 1024


@dataclass(frozen=True)
class CaptureConfig:
    symbols: Sequence[str]
    out_dir: Path
    channels: Sequence[str] = DEFAULT_CHANNELS
    base_url: str = COMBINED_STREAM_BASE_URL
    rotate_interval_hours: float = 1.0
    max_queue_size: int = 20_000
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    backoff_multiplier: float = 2.0
    flush_interval_s: float = 5.0
    stats_log_interval_s: float = 30.0
    ping_timeout_s: float = 20.0
    snapshot_interval_s: float = 900.0
    snapshot_depth_limit: int = 1000

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if not self.channels:
            raise ValueError("channels must be non-empty")
        if len(self.symbols) * len(self.channels) > MAX_STREAMS_PER_CONNECTION:
            raise ValueError(
                f"{len(self.symbols) * len(self.channels)} streams requested, "
                f"exceeds Binance's {MAX_STREAMS_PER_CONNECTION}-stream-per-connection limit; "
                "split across multiple CaptureConfig/run_capture instances"
            )
        projected = binance_rest.projected_weight_per_minute(
            len(self.symbols), self.snapshot_interval_s, self.snapshot_depth_limit
        )
        # Conservative cap: snapshot polling alone should not use more than
        # half the account's request-weight budget, leaving headroom for
        # reconstruct.py's verification-mode fetches and any other REST use
        # sharing the same IP. See binance_rest.py module docstring for the
        # source of the 2400/min figure.
        budget = binance_rest.REQUEST_WEIGHT_PER_MINUTE_LIMIT / 2
        if projected > budget:
            raise ValueError(
                f"periodic snapshot fetch would use {projected:.0f} weight/min "
                f"(>{budget:.0f}, half of Binance's {binance_rest.REQUEST_WEIGHT_PER_MINUTE_LIMIT}/min IP budget); "
                "increase snapshot_interval_s, lower snapshot_depth_limit, or fetch fewer symbols per process"
            )

    @property
    def stream_names(self) -> list[str]:
        return [f"{s.lower()}@{c}" for s in self.symbols for c in self.channels]

    @property
    def ws_url(self) -> str:
        return f"{self.base_url}?streams={'/'.join(self.stream_names)}"


@dataclass
class CaptureStats:
    """Mutable counters, logged periodically. `messages_dropped` counts
    messages the receive loop had to discard because the disk writer could
    not keep up (the internal queue was full) — this is a CAPTURE-layer
    backpressure metric, distinct from an exchange-side sequence gap, which
    is detected later by reconstruct.py from the `u`/`pu` fields it (and
    only it) parses."""

    messages_received: int = 0
    bytes_received: int = 0
    messages_written: int = 0
    messages_dropped: int = 0
    reconnects: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "messages_received": self.messages_received,
            "bytes_received": self.bytes_received,
            "messages_written": self.messages_written,
            "messages_dropped": self.messages_dropped,
            "reconnects": self.reconnects,
        }


def _extract_stream_name(raw: str) -> str | None:
    """Best-effort extraction of the `"stream":"..."` value from a combined-
    stream envelope, via string slicing rather than json.loads — a filing
    convenience, not a parse of message content. Returns None (never raises)
    on anything unexpected, so a malformed or unexpected frame is still
    captured with stream=null instead of being dropped."""
    marker = '"stream":"'
    i = raw.find(marker)
    if i == -1:
        return None
    start = i + len(marker)
    end = raw.find('"', start)
    if end == -1:
        return None
    return raw[start:end]


def _envelope_bytes(
    type_: str,
    *,
    recv_wall_ns: int,
    recv_mono_ns: int,
    stream: str | None = None,
    raw: str | None = None,
    reason: str | None = None,
) -> bytes:
    obj: dict[str, object] = {
        "schema": CAPTURE_SCHEMA_VERSION,
        "type": type_,
        "stream": stream,
        "recv_wall_ns": recv_wall_ns,
        "recv_mono_ns": recv_mono_ns,
    }
    if raw is not None:
        obj["raw"] = raw
    if reason is not None:
        obj["reason"] = reason
    return (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


class _RotatingZstdWriter:
    """Owns exactly one open (file, zstd stream_writer) pair at a time,
    rotating to a new file on an hour boundary (UTC) and periodically
    flushing a complete zstd frame so a killed process loses at most
    `flush_interval_s` of data rather than an unreadable partial frame."""

    def __init__(self, out_dir: Path, rotate_interval_hours: float, flush_interval_s: float) -> None:
        self._out_dir = out_dir
        self._rotate_s = rotate_interval_hours * 3600.0
        self._flush_interval_s = flush_interval_s
        self._compressor = zstd.ZstdCompressor(level=9)
        self._fh = None
        self._writer = None
        self._window: int | None = None
        self._last_flush = 0.0

    def _path_for(self, now: float) -> Path:
        gmt = time.gmtime(now)
        day_dir = self._out_dir / time.strftime("%Y-%m-%d", gmt)
        day_dir.mkdir(parents=True, exist_ok=True)
        fname = time.strftime("%Y-%m-%dT%H0000Z.jsonl.zst", gmt)
        return day_dir / fname

    def _rotate_if_needed(self, now: float) -> None:
        window = int(now // self._rotate_s)
        if self._window is not None and window == self._window:
            return
        self._close()
        path = self._path_for(now)
        self._fh = open(path, "ab")
        self._writer = self._compressor.stream_writer(self._fh)
        self._window = window
        self._last_flush = now
        LOG.info("rotated capture file -> %s", path)

    def write(self, data: bytes) -> None:
        now = time.time()
        self._rotate_if_needed(now)
        assert self._writer is not None
        self._writer.write(data)
        if (now - self._last_flush) >= self._flush_interval_s:
            self._writer.flush(zstd.FLUSH_FRAME)
            self._last_flush = now

    def _close(self) -> None:
        if self._writer is not None:
            self._writer.flush(zstd.FLUSH_FRAME)
            self._writer.close()  # closes the underlying file handle too
            self._writer = None
            self._fh = None

    def close(self) -> None:
        self._close()


async def _writer_task(queue: asyncio.Queue[bytes], writer: _RotatingZstdWriter, stats: CaptureStats) -> None:
    while True:
        item = await queue.get()
        if item is None:  # sentinel: shut down
            queue.task_done()
            break
        writer.write(item)
        stats.messages_written += 1
        queue.task_done()


async def _stats_logger_task(stats: CaptureStats, interval_s: float) -> None:
    prev = stats.snapshot()
    while True:
        await asyncio.sleep(interval_s)
        cur = stats.snapshot()
        rate = (cur["messages_received"] - prev["messages_received"]) / interval_s
        LOG.info(
            "throughput: %.1f msg/s | cumulative received=%d written=%d dropped=%d reconnects=%d",
            rate,
            cur["messages_received"],
            cur["messages_written"],
            cur["messages_dropped"],
            cur["reconnects"],
        )
        prev = cur


async def _connection_loop(
    config: CaptureConfig, queue: asyncio.Queue[bytes], stats: CaptureStats, connected_event: asyncio.Event
) -> None:
    attempt = 0
    first_connect = True
    while True:
        try:
            async with websockets.connect(
                config.ws_url,
                ping_interval=180,  # Binance server pings every ~3min; respond via library default pong
                ping_timeout=config.ping_timeout_s,
                max_size=None,  # depth snapshots for illiquid pairs can be large; do not truncate
            ) as ws:
                attempt = 0  # reset backoff after a successful connect
                reason = "initial_connect" if first_connect else "reconnect"
                # Every connect (including the first) means the reconstructor
                # cannot assume continuity with anything captured before it —
                # emit a gap marker unconditionally rather than special-casing
                # "this is the very first one, surely it's fine."
                await _enqueue_gap_marker_async(queue, reason)
                first_connect = False
                connected_event.set()
                LOG.info("connected (%s): %s", reason, config.ws_url)

                async for message in ws:
                    recv_wall_ns = time.time_ns()
                    recv_mono_ns = time.monotonic_ns()
                    raw_text = message if isinstance(message, str) else message.decode("utf-8", errors="replace")
                    stats.messages_received += 1
                    stats.bytes_received += len(raw_text)
                    envelope = _envelope_bytes(
                        "message",
                        recv_wall_ns=recv_wall_ns,
                        recv_mono_ns=recv_mono_ns,
                        stream=_extract_stream_name(raw_text),
                        raw=raw_text,
                    )
                    try:
                        queue.put_nowait(envelope)
                    except asyncio.QueueFull:
                        stats.messages_dropped += 1

        except (websockets.ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
            stats.reconnects += 1
            attempt += 1
            delay = min(config.backoff_max_s, config.backoff_initial_s * (config.backoff_multiplier ** (attempt - 1)))
            delay *= 1.0 + random.random() * 0.1  # jitter, avoid thundering herd on shared infra
            LOG.warning("connection lost (%s), reconnecting in %.1fs (attempt %d)", exc, delay, attempt)
            await asyncio.sleep(delay)


async def _snapshot_task(
    config: CaptureConfig, queue: asyncio.Queue[bytes], stats: CaptureStats, connected_event: asyncio.Event
) -> None:
    """Periodically fetch the REST depth snapshot for each symbol and write
    it into the capture stream verbatim (type="snapshot"). This is what
    makes OFFLINE reconstruction possible at all: Binance's REST endpoint
    only ever returns the book as of right now, so a historical replay
    needs a snapshot that was captured at/near the same time as the diff
    stream, not one fetched after the fact. Storing the raw response body
    unparsed keeps this consistent with "capture only" — see binance_rest.py.

    Runs once immediately after the first successful WS connect (so
    reconstruction near the start of a capture window has a nearby
    bootstrap point) and then every `config.snapshot_interval_s`. Waiting
    for `connected_event` avoids a race against the initial-connect gap
    marker: if the first snapshot fetch completed and was queued BEFORE
    that gap marker, the reconstructor would (correctly, but pointlessly)
    treat the gap marker as invalidating a snapshot it had just applied.
    """
    await connected_event.wait()
    first = True
    while True:
        if not first:
            await asyncio.sleep(config.snapshot_interval_s)
        first = False
        for symbol in config.symbols:
            try:
                raw = await binance_rest.fetch_depth_snapshot(symbol, limit=config.snapshot_depth_limit)
            except binance_rest.SnapshotFetchError as exc:
                LOG.warning("snapshot fetch failed for %s: %s", symbol, exc)
                continue
            envelope = _envelope_bytes(
                "snapshot",
                recv_wall_ns=time.time_ns(),
                recv_mono_ns=time.monotonic_ns(),
                stream=f"{symbol.lower()}@depth_snapshot",
                raw=raw,
            )
            try:
                queue.put_nowait(envelope)
            except asyncio.QueueFull:
                stats.messages_dropped += 1


async def _enqueue_gap_marker_async(queue: asyncio.Queue[bytes], reason: str) -> None:
    envelope = _envelope_bytes(
        "gap",
        recv_wall_ns=time.time_ns(),
        recv_mono_ns=time.monotonic_ns(),
        reason=reason,
    )
    await queue.put(envelope)  # block if full: never silently drop a gap marker


async def run_capture(config: CaptureConfig, *, stop_event: asyncio.Event | None = None) -> None:
    """Run the capture daemon until `stop_event` is set (or forever, if not
    given — the caller is expected to cancel the task / send SIGINT)."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=config.max_queue_size)
    stats = CaptureStats()
    writer = _RotatingZstdWriter(config.out_dir, config.rotate_interval_hours, config.flush_interval_s)

    connected_event = asyncio.Event()
    writer_t = asyncio.create_task(_writer_task(queue, writer, stats))
    stats_t = asyncio.create_task(_stats_logger_task(stats, config.stats_log_interval_s))
    conn_t = asyncio.create_task(_connection_loop(config, queue, stats, connected_event))
    snapshot_t = asyncio.create_task(_snapshot_task(config, queue, stats, connected_event))

    stop_event = stop_event or asyncio.Event()
    try:
        await stop_event.wait()
    finally:
        conn_t.cancel()
        stats_t.cancel()
        snapshot_t.cancel()
        for t in (conn_t, stats_t, snapshot_t):
            try:
                await t
            except asyncio.CancelledError:
                pass
        await queue.put(None)  # sentinel drains remaining items then stops writer_t
        await writer_t
        writer.close()
        LOG.info("capture stopped: %s", stats.snapshot())


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            # add_signal_handler is unavailable on Windows' default event
            # loop; SIGINT still raises KeyboardInterrupt, which main() below
            # catches and translates into a clean stop_event.set().
            pass


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", required=True, dest="symbols", help="repeatable, e.g. --symbol BTCUSDT")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rotate-hours", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = CaptureConfig(symbols=args.symbols, out_dir=args.out_dir, rotate_interval_hours=args.rotate_hours)

    stop_event = asyncio.Event()

    async def _run() -> None:
        _install_signal_handlers(asyncio.get_running_loop(), stop_event)
        await run_capture(config, stop_event=stop_event)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
