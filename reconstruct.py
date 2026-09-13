"""Offline order book reconstructor for Binance USDS-M futures captures.

Replays capture.py's raw JSONL/zstd files and rebuilds full L2 book state
plus a trades table, following Binance's own documented local-order-book
procedure EXACTLY — see BookState, which quotes it. Do not "improve" the
apply_diff state machine without re-reading that source; the ordering of the
staleness check vs. the first-event check is load-bearing, not incidental.

ARCHITECTURE (market-data-integrity): this process does the interpretation
that capture.py deliberately does not. It is the only place json.loads()
is called on message/snapshot payload content in this codebase's ingestion
path. Re-running this file against already-captured data after fixing a bug
here is exactly the point of keeping capture and reconstruction separate —
the raw files are unaffected by bugs found here.

SEQUENCE GAPS: on ANY violation of the documented invariants (see
BookState.apply_diff) or on a capture-level `gap` record (WS reconnect —
messages may have been missed while reconnecting), the book is marked
INVALID immediately. No interpolation, no "close enough" patch, no reuse of
stale levels. A `type="gap"` capture record forces every tracked symbol's
book to INVALID because the underlying WS connection is shared across
symbols on one combined stream — a reconnect is a break for all of them, not
just one.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

LOG = logging.getLogger("reconstruct")

BOOK_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("event_type", pa.string()),  # "snapshot" | "update" | "gap"
        ("event_time_ns", pa.int64()),  # exchange timestamp; null for gap rows
        ("update_id", pa.int64()),  # Binance `u`; null for gap rows
        ("recv_wall_ns", pa.int64()),
        ("recv_mono_ns", pa.int64()),
        ("valid", pa.bool_()),
        ("bid_prices", pa.list_(pa.float64())),
        ("bid_qtys", pa.list_(pa.float64())),
        ("ask_prices", pa.list_(pa.float64())),
        ("ask_qtys", pa.list_(pa.float64())),
    ]
)

TRADE_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("agg_trade_id", pa.int64()),
        ("trade_time_ns", pa.int64()),
        ("recv_wall_ns", pa.int64()),
        ("recv_mono_ns", pa.int64()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
        ("is_buyer_maker", pa.bool_()),  # Binance `m`: True => buyer is the resting/maker side => SELLER was the aggressor
        ("first_trade_id", pa.int64()),
        ("last_trade_id", pa.int64()),
    ]
)


# --------------------------------------------------------------------------
# Book state machine
# --------------------------------------------------------------------------


#: Depth diff events (@100ms cadence) to keep per symbol so a snapshot's
#: `lastUpdateId` can be bracketed by an event that already streamed past
#: BEFORE the snapshot's REST response arrived. This is not optional
#: robustness — the documented procedure's step 2 ("buffer the events you
#: receive from the stream" -- BEFORE step 3, fetching the snapshot) exists
#: precisely because REST round-trip latency routinely means the correct
#: bootstrap event is already behind you by the time the snapshot response
#: lands, not still ahead of you. A forward-only search (look only at
#: events arriving after the snapshot) misses this and produces spurious,
#: avoidable invalidations. 200 events at 100ms cadence is a 20s window,
#: comfortably wider than realistic REST latency.
RECENT_EVENTS_BUFFER = 200


@dataclass
class BookState:
    """One symbol's local order book, maintained per Binance's documented
    procedure for USDS-M futures diff-depth streams:
    https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly
    (retrieved 2026-09-13; quoted verbatim in _apply below). This is NOT the
    spot procedure — spot events don't carry `pu` and use a different
    continuity check; do not reuse this class for spot data.
    """

    symbol: str
    depth_limit: int
    valid: bool = False
    awaiting_first_event: bool = False
    last_update_id: int | None = None
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    _recent_events: deque = field(default_factory=lambda: deque(maxlen=RECENT_EVENTS_BUFFER), repr=False)

    def load_snapshot(self, last_update_id: int, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> None:
        self.last_update_id = last_update_id
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}
        self.valid = True
        self.awaiting_first_event = True
        self._bootstrap_from_buffer()

    def _bootstrap_from_buffer(self) -> None:
        """Replay already-buffered events (oldest first) through the normal
        state machine before falling back to waiting for new live events.
        Safe to call unconditionally: `_apply` independently validates each
        event's U/u/pu against the CURRENT last_update_id before mutating
        anything, so replaying events from a previous, now-irrelevant book
        state is a no-op (they fail the `u < last_update_id` staleness check
        and are dropped without side effects) rather than corrupting the
        freshly-loaded snapshot."""
        for U, u, pu, bid_updates, ask_updates in list(self._recent_events):
            if not self.valid:
                break
            self._apply(U, u, pu, bid_updates, ask_updates)

    def invalidate(self) -> None:
        self.valid = False
        self.awaiting_first_event = False
        self.last_update_id = None
        self.bids.clear()
        self.asks.clear()

    def apply_diff(
        self,
        U: int,
        u: int,
        pu: int,
        bid_updates: list[tuple[float, float]],
        ask_updates: list[tuple[float, float]],
    ) -> bool:
        """Record the event in the recent-events buffer (see
        RECENT_EVENTS_BUFFER) and, if the book is currently valid, apply it.
        Returns True if applied, False if dropped/invalidated. See _apply
        for the documented procedure this implements.
        """
        self._recent_events.append((U, u, pu, bid_updates, ask_updates))
        return self._apply(U, u, pu, bid_updates, ask_updates)

    def _apply(
        self,
        U: int,
        u: int,
        pu: int,
        bid_updates: list[tuple[float, float]],
        ask_updates: list[tuple[float, float]],
    ) -> bool:
        """Quoting the documented procedure:
          Step 4: "Drop any event where `u` is < `lastUpdateId` in the snapshot."
          Step 5: "The first processed event should have `U` <= `lastUpdateId`
                   AND `u` >= `lastUpdateId`."
          Step 6: "While listening to the stream, each new event's `pu`
                   should be equal to the previous event's `u`, otherwise
                   initialize the process from step 3 [refetch snapshot]."
          Steps 7-9: values are ABSOLUTE quantity per price level; qty==0
                     removes the level; removing a level not currently held
                     locally is normal, not an error.

        Returns True if the event was applied, False if it was dropped
        (stale, per step 4) or the book was/became invalid (steps 5/6). On
        any violation of step 5 or step 6 this sets self.valid = False and
        clears book state — it does NOT raise, so the caller (which is
        writing one output row per event regardless) can record the
        transition to invalid rather than losing the event's timestamp.
        """
        if not self.valid:
            return False
        assert self.last_update_id is not None

        if self.awaiting_first_event:
            if u < self.last_update_id:
                return False  # stale relative to snapshot; drop, keep waiting (step 4)
            if not (U <= self.last_update_id <= u):
                self.invalidate()  # snapshot and stream start don't overlap correctly (step 5)
                return False
            self.awaiting_first_event = False
        else:
            if pu != self.last_update_id:
                self.invalidate()  # sequence break (step 6)
                return False

        for price, qty in bid_updates:
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price, qty in ask_updates:
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty
        self.last_update_id = u
        return True

    def top_levels(self, n: int) -> tuple[list[float], list[float], list[float], list[float]]:
        # NOTE: O(k log k) per call on the full book depth k. Fine at the
        # scale this module targets (research replay); a production
        # low-latency reconstructor would keep bids/asks in a sorted
        # structure (e.g. a skip list / SortedDict) instead of re-sorting a
        # plain dict on every event.
        bid_prices = sorted(self.bids, reverse=True)[:n]
        ask_prices = sorted(self.asks)[:n]
        return bid_prices, [self.bids[p] for p in bid_prices], ask_prices, [self.asks[p] for p in ask_prices]


# --------------------------------------------------------------------------
# Capture file reading
# --------------------------------------------------------------------------


def _iter_envelopes(path: Path) -> Iterator[dict]:
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            # stream_reader() reads across ALL concatenated zstd frames in
            # the file transparently -- see capture.py module docstring.
            # A bare ZstdDecompressor().decompress(...) would silently
            # return only the first frame and truncate the file.
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _symbol_from_stream(stream: str) -> str:
    return stream.split("@", 1)[0].upper()


# --------------------------------------------------------------------------
# Parquet batch writer
# --------------------------------------------------------------------------


class _ParquetBatchWriter:
    """Buffers rows and writes them as Parquet row groups in batches, so
    reconstruction of tens-of-millions-of-rows captures does not hold the
    entire output table in memory."""

    def __init__(self, path: Path, schema: pa.Schema, row_group_size: int = 100_000) -> None:
        self._path = path
        self._schema = schema
        self._row_group_size = row_group_size
        self._rows: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

    def add(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._row_group_size:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self._path, self._schema)
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self._flush()
        if self._writer is None:
            # Nothing was ever written (e.g. zero aggTrade events in the
            # replayed window). Still produce a schema-only empty file so
            # downstream readers can always expect the file to exist rather
            # than special-casing "this symbol/channel had zero events."
            self._path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(self._schema.empty_table(), self._path)
        else:
            self._writer.close()
            self._writer = None


def _book_row(
    book: BookState,
    *,
    event_type: str,
    event_time_ms: int | None,
    update_id: int | None,
    recv_wall_ns: int,
    recv_mono_ns: int,
) -> dict:
    if book.valid:
        bid_p, bid_q, ask_p, ask_q = book.top_levels(book.depth_limit)
    else:
        bid_p, bid_q, ask_p, ask_q = [], [], [], []
    return {
        "symbol": book.symbol,
        "event_type": event_type,
        "event_time_ns": None if event_time_ms is None else event_time_ms * 1_000_000,
        "update_id": update_id,
        "recv_wall_ns": recv_wall_ns,
        "recv_mono_ns": recv_mono_ns,
        "valid": book.valid,
        "bid_prices": bid_p,
        "bid_qtys": bid_q,
        "ask_prices": ask_p,
        "ask_qtys": ask_q,
    }


def _trade_row(symbol: str, data: dict, recv_wall_ns: int, recv_mono_ns: int) -> dict:
    return {
        "symbol": symbol,
        "agg_trade_id": int(data["a"]),
        "trade_time_ns": int(data["T"]) * 1_000_000,
        "recv_wall_ns": recv_wall_ns,
        "recv_mono_ns": recv_mono_ns,
        "price": float(data["p"]),
        "quantity": float(data["q"]),
        "is_buyer_maker": bool(data["m"]),
        "first_trade_id": int(data["f"]),
        "last_trade_id": int(data["l"]),
    }


# --------------------------------------------------------------------------
# Main replay driver
# --------------------------------------------------------------------------


@dataclass
class ReconstructionStats:
    events_processed: int = 0
    events_dropped_stale: int = 0
    events_dropped_while_invalid: int = 0
    gaps_detected: int = 0
    snapshots_applied: int = 0
    trades_processed: int = 0
    unrecognized_events: int = 0


def reconstruct(
    capture_paths: Sequence[Path],
    symbols: Sequence[str],
    *,
    out_dir: Path,
    depth_levels: int = 20,
    row_group_size: int = 100_000,
) -> ReconstructionStats:
    """Replay `capture_paths` (processed in the given order — pass them
    already sorted chronologically; capture.py's hourly filenames sort
    correctly by construction) and write book_state.parquet and
    trades.parquet under `out_dir`.
    """
    books = {s.upper(): BookState(symbol=s.upper(), depth_limit=depth_levels) for s in symbols}
    stats = ReconstructionStats()

    book_writer = _ParquetBatchWriter(out_dir / "book_state.parquet", BOOK_SCHEMA, row_group_size)
    trade_writer = _ParquetBatchWriter(out_dir / "trades.parquet", TRADE_SCHEMA, row_group_size)

    try:
        for path in capture_paths:
            for env in _iter_envelopes(path):
                _process_envelope(env, books, book_writer, trade_writer, stats)
    finally:
        book_writer.close()
        trade_writer.close()

    return stats


def _process_envelope(
    env: dict,
    books: dict[str, BookState],
    book_writer: _ParquetBatchWriter,
    trade_writer: _ParquetBatchWriter,
    stats: ReconstructionStats,
) -> None:
    if env["type"] == "gap":
        # A gap on the shared combined-stream connection breaks continuity
        # for every symbol on it, not just one -- invalidate all of them and
        # emit a marker row per symbol so the hole is visible per-symbol in
        # book_state.parquet rather than only in a global log line.
        stats.gaps_detected += 1
        for book in books.values():
            book.invalidate()
            book_writer.add(
                _book_row(
                    book,
                    event_type="gap",
                    event_time_ms=None,
                    update_id=None,
                    recv_wall_ns=env["recv_wall_ns"],
                    recv_mono_ns=env["recv_mono_ns"],
                )
            )
        return

    stream = env.get("stream")
    if not stream:
        stats.unrecognized_events += 1
        return
    symbol = _symbol_from_stream(stream)
    book = books.get(symbol)
    if book is None:
        return  # not one of the symbols we were asked to reconstruct

    if env["type"] == "snapshot":
        snap = json.loads(env["raw"])
        bids = [(float(p), float(q)) for p, q in snap["bids"]]
        asks = [(float(p), float(q)) for p, q in snap["asks"]]
        book.load_snapshot(int(snap["lastUpdateId"]), bids, asks)
        stats.snapshots_applied += 1
        book_writer.add(
            _book_row(
                book,
                event_type="snapshot",
                event_time_ms=snap.get("T"),
                update_id=int(snap["lastUpdateId"]),
                recv_wall_ns=env["recv_wall_ns"],
                recv_mono_ns=env["recv_mono_ns"],
            )
        )
        return

    if env["type"] != "message":
        stats.unrecognized_events += 1
        return

    wrapper = json.loads(env["raw"])
    data = wrapper.get("data", {})
    event = data.get("e")

    if event == "depthUpdate":
        bid_updates = [(float(p), float(q)) for p, q in data["b"]]
        ask_updates = [(float(p), float(q)) for p, q in data["a"]]
        was_valid = book.valid
        applied = book.apply_diff(int(data["U"]), int(data["u"]), int(data["pu"]), bid_updates, ask_updates)
        stats.events_processed += 1
        if not applied:
            if was_valid and book.valid:
                stats.events_dropped_stale += 1  # stale per step 4, book otherwise unaffected
            elif was_valid and not book.valid:
                stats.gaps_detected += 1  # THIS event is what triggered invalidation
            else:
                # book was already invalid before this event arrived -- not a
                # new gap, just an event we can't apply until the next
                # snapshot resync. Counted separately so gaps_detected
                # reflects distinct invalidation events, not every dropped
                # message during a long invalid stretch.
                stats.events_dropped_while_invalid += 1
        book_writer.add(
            _book_row(
                book,
                event_type="update",
                event_time_ms=int(data["E"]),
                update_id=int(data["u"]),
                recv_wall_ns=env["recv_wall_ns"],
                recv_mono_ns=env["recv_mono_ns"],
            )
        )
    elif event == "aggTrade":
        trade_writer.add(_trade_row(symbol, data, env["recv_wall_ns"], env["recv_mono_ns"]))
        stats.trades_processed += 1
    else:
        stats.unrecognized_events += 1


# --------------------------------------------------------------------------
# Verification mode (report-only, never auto-corrects)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceReport:
    symbol: str
    snapshot_recv_wall_ns: int
    reconstructed_recv_wall_ns: int | None
    reconstructed_row_valid: bool
    levels_compared: int
    levels_matched: int
    max_bid_price_diff: float
    max_ask_price_diff: float
    max_bid_qty_diff: float
    max_ask_qty_diff: float
    note: str


def verify_against_snapshot(
    book_df: pl.DataFrame,
    symbol: str,
    snapshot_raw: str,
    snapshot_recv_wall_ns: int,
    *,
    levels: int = 10,
) -> DivergenceReport:
    """Compare one independently-fetched REST snapshot against the most
    recently reconstructed row at-or-before the snapshot's receipt time for
    `symbol`. This function only REPORTS a divergence — it never writes back
    to `book_df` or otherwise corrects the reconstructed state (see
    market-data-integrity: an auto-correcting verifier hides the defect it
    exists to surface).

    Comparing against the row at-or-before (not nearest) the snapshot time
    is deliberate: the snapshot is a point-in-time truth, and the
    reconstructed book at that instant should equal whatever the last
    applied event made it, without peeking at anything the reconstructor
    processed after the snapshot was taken.
    """
    snap = json.loads(snapshot_raw)
    ref_bids = [(float(p), float(q)) for p, q in snap["bids"][:levels]]
    ref_asks = [(float(p), float(q)) for p, q in snap["asks"][:levels]]

    candidates = book_df.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("recv_wall_ns") <= snapshot_recv_wall_ns)
        & (pl.col("valid"))
    )
    if candidates.is_empty():
        return DivergenceReport(
            symbol=symbol,
            snapshot_recv_wall_ns=snapshot_recv_wall_ns,
            reconstructed_recv_wall_ns=None,
            reconstructed_row_valid=False,
            levels_compared=0,
            levels_matched=0,
            max_bid_price_diff=float("nan"),
            max_ask_price_diff=float("nan"),
            max_bid_qty_diff=float("nan"),
            max_ask_qty_diff=float("nan"),
            note="no valid reconstructed row at or before snapshot receipt time",
        )

    row = candidates.sort("recv_wall_ns").row(-1, named=True)
    rec_bids = list(zip(row["bid_prices"][:levels], row["bid_qtys"][:levels]))
    rec_asks = list(zip(row["ask_prices"][:levels], row["ask_qtys"][:levels]))

    def _diffs(ref: list[tuple[float, float]], rec: list[tuple[float, float]]) -> tuple[int, float, float]:
        n = min(len(ref), len(rec))
        matched = 0
        max_p = 0.0
        max_q = 0.0
        for (rp, rq), (cp, cq) in zip(ref[:n], rec[:n]):
            dp, dq = abs(rp - cp), abs(rq - cq)
            max_p = max(max_p, dp)
            max_q = max(max_q, dq)
            if dp < 1e-9 and dq < 1e-9:
                matched += 1
        return matched, max_p, max_q

    bid_matched, max_bp, max_bq = _diffs(ref_bids, rec_bids)
    ask_matched, max_ap, max_aq = _diffs(ref_asks, rec_asks)
    compared = len(ref_bids) + len(ref_asks)
    matched = bid_matched + ask_matched

    return DivergenceReport(
        symbol=symbol,
        snapshot_recv_wall_ns=snapshot_recv_wall_ns,
        reconstructed_recv_wall_ns=row["recv_wall_ns"],
        reconstructed_row_valid=True,
        levels_compared=compared,
        levels_matched=matched,
        max_bid_price_diff=max_bp,
        max_ask_price_diff=max_ap,
        max_bid_qty_diff=max_bq,
        max_ask_qty_diff=max_aq,
        note="ok" if matched == compared else f"{compared - matched}/{compared} levels diverged",
    )


def verify_capture_snapshots(
    book_df: pl.DataFrame,
    capture_paths: Sequence[Path],
    symbols: Sequence[str],
    *,
    levels: int = 10,
) -> list[DivergenceReport]:
    """Convenience wrapper: pulls every `type=="snapshot"` record for the
    given symbols out of the capture files and verifies the reconstructed
    book against each one.

    CAVEAT this function does not hide from you: if these are the SAME
    snapshot records reconstruct() used for bootstrap/resync, comparing
    against them is partially circular -- it will always match at the exact
    instant a snapshot was applied. It is still useful for catching drift
    between resyncs (most rows are NOT bootstrap instants), but a stronger
    check is to fetch fresh snapshots independently (see
    binance_rest.fetch_depth_snapshot) while reconstruction is running near
    real time and pass those to verify_against_snapshot directly instead.
    """
    symbol_set = {s.upper() for s in symbols}
    reports: list[DivergenceReport] = []
    for path in capture_paths:
        for env in _iter_envelopes(path):
            if env.get("type") != "snapshot":
                continue
            stream = env.get("stream") or ""
            symbol = _symbol_from_stream(stream)
            if symbol not in symbol_set:
                continue
            reports.append(verify_against_snapshot(book_df, symbol, env["raw"], env["recv_wall_ns"], levels=levels))
    return reports


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True, help="directory of *.jsonl.zst capture files (searched recursively)")
    parser.add_argument("--symbol", action="append", required=True, dest="symbols")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="also run verify_capture_snapshots and print a report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    paths = sorted(args.capture_dir.rglob("*.jsonl.zst"))
    if not paths:
        raise SystemExit(f"no *.jsonl.zst files found under {args.capture_dir}")

    stats = reconstruct(paths, args.symbols, out_dir=args.out_dir, depth_levels=args.depth_levels)
    LOG.info("reconstruction stats: %s", stats)

    if args.verify:
        book_df = pl.read_parquet(args.out_dir / "book_state.parquet")
        reports = verify_capture_snapshots(book_df, paths, args.symbols)
        diverged = [r for r in reports if r.note != "ok"]
        LOG.info("verification: %d snapshots checked, %d diverged", len(reports), len(diverged))
        for r in diverged:
            LOG.warning("divergence: %s", r)


if __name__ == "__main__":
    main()
