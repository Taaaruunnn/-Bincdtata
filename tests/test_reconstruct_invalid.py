"""Proves the offline reconstructor marks book state INVALID -- rather than
producing a plausible-but-wrong book -- when fed a sequence gap. Exercises
the real on-disk capture format end to end (zstd multi-frame JSONL ->
reconstruct.reconstruct() -> Parquet), not just the BookState unit in
isolation (see tests/test_gap_detector.py for that)."""

from __future__ import annotations

import polars as pl

import reconstruct
from tests.helpers import depth_update_envelope, snapshot_envelope, write_capture_file


def test_gap_produces_invalid_rows_with_no_stale_levels(tmp_path):
    symbol = "BTCUSDT"
    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=1, mono_ns=1, event_time_ms=1000),
        # brackets lastUpdateId=100 -> applies cleanly
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=2, mono_ns=2),
        # pu should be 105; it's not -> sequence break -> INVALID
        depth_update_envelope(symbol, U=200, u=210, pu=199, bids=[["99.00", "999.0"]], asks=[], event_time_ms=1020, wall_ns=3, mono_ns=3),
        # another event while still invalid -- must also be rejected, not patched in
        depth_update_envelope(symbol, U=210, u=220, pu=210, bids=[["99.00", "5.0"]], asks=[], event_time_ms=1030, wall_ns=4, mono_ns=4),
        # fresh snapshot resyncs
        snapshot_envelope(symbol, 9000, bids=[["199.00", "3.0"]], asks=[["200.00", "3.0"]], wall_ns=5, mono_ns=5, event_time_ms=1040),
    ]
    capture_path = tmp_path / "capture" / "file1.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=5)

    assert stats.gaps_detected == 1
    assert stats.snapshots_applied == 2

    df = pl.read_parquet(out_dir / "book_state.parquet").sort("recv_wall_ns")
    rows = df.to_dicts()
    assert len(rows) == 5

    # row 0: first snapshot bootstrap
    assert rows[0]["event_type"] == "snapshot"
    assert rows[0]["valid"] is True
    assert rows[0]["bid_prices"] == [99.0]

    # row 1: clean update applied
    assert rows[1]["event_type"] == "update"
    assert rows[1]["valid"] is True
    assert rows[1]["bid_prices"] == [99.0]
    assert rows[1]["bid_qtys"] == [2.0]

    # row 2: the event that triggered invalidation -- MUST be marked
    # invalid with EMPTY levels, never the old [99.0]/2.0 values dressed up
    # as if they were still trustworthy, and never the incoming (unapplied)
    # 999.0 either.
    assert rows[2]["event_type"] == "update"
    assert rows[2]["valid"] is False
    assert rows[2]["bid_prices"] == []
    assert rows[2]["bid_qtys"] == []

    # row 3: still invalid, still empty -- no silent patching mid-gap
    assert rows[3]["valid"] is False
    assert rows[3]["bid_prices"] == []

    # row 4: resync snapshot restores a valid, fresh book
    assert rows[4]["event_type"] == "snapshot"
    assert rows[4]["valid"] is True
    assert rows[4]["bid_prices"] == [199.0]
    assert rows[4]["bid_qtys"] == [3.0]


def test_snapshot_row_update_id_reflects_post_bootstrap_state_not_raw_snapshot(tmp_path):
    """Regression test: BookState.load_snapshot() replays any already-
    buffered diff events (see _bootstrap_from_buffer) BEFORE the snapshot's
    output row is built. If a buffered event brackets the snapshot's
    lastUpdateId, the row's bid/ask content ends up current as of that
    event's `u`, not the raw snapshot's lastUpdateId -- so the row's
    `update_id` column must report book.last_update_id (post-replay), or it
    mislabels content that has already moved past it."""
    symbol = "BTCUSDT"
    envs = [
        # Buffered while the book is still invalid (no snapshot loaded yet)
        # -- apply_diff returns False, but the event is still recorded into
        # BookState._recent_events.
        depth_update_envelope(symbol, U=100, u=110, pu=99, bids=[["50.00", "2.0"]], asks=[], event_time_ms=900, wall_ns=1, mono_ns=1),
        # lastUpdateId=105 falls inside the buffered event's [100, 110]
        # bracket, so bootstrap will apply that event on top of this
        # snapshot's own (999.0) quantity for the same price.
        snapshot_envelope(symbol, 105, bids=[["50.00", "999.0"]], asks=[["51.00", "1.0"]], wall_ns=2, mono_ns=2, event_time_ms=1000),
    ]
    capture_path = tmp_path / "capture" / "file1.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=5)

    df = pl.read_parquet(out_dir / "book_state.parquet").sort("recv_wall_ns")
    row = df.filter(pl.col("event_type") == "snapshot").row(-1, named=True)

    assert row["valid"] is True
    # content is current as of the buffered event's u=110, not the raw
    # snapshot's lastUpdateId=105
    assert row["update_id"] == 110
    # and the content itself reflects that same overwrite: qty 2.0 (from
    # the buffered event), not the snapshot's own 999.0
    assert row["bid_prices"] == [50.0]
    assert row["bid_qtys"] == [2.0]


def test_capture_level_gap_marker_invalidates_all_tracked_symbols(tmp_path):
    from tests.helpers import gap_envelope

    envs = [
        snapshot_envelope("BTCUSDT", 100, bids=[["1.0", "1.0"]], asks=[["1.1", "1.0"]], wall_ns=1, mono_ns=1, event_time_ms=1000),
        snapshot_envelope("ETHUSDT", 200, bids=[["2.0", "1.0"]], asks=[["2.1", "1.0"]], wall_ns=2, mono_ns=2, event_time_ms=1000),
        depth_update_envelope("BTCUSDT", U=95, u=105, pu=90, bids=[], asks=[], event_time_ms=1010, wall_ns=3, mono_ns=3),
        depth_update_envelope("ETHUSDT", U=195, u=205, pu=190, bids=[], asks=[], event_time_ms=1010, wall_ns=4, mono_ns=4),
        gap_envelope("reconnect", wall_ns=5, mono_ns=5),
    ]
    capture_path = tmp_path / "capture" / "file1.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    reconstruct.reconstruct([capture_path], ["BTCUSDT", "ETHUSDT"], out_dir=out_dir, depth_levels=5)

    df = pl.read_parquet(out_dir / "book_state.parquet")
    gap_rows = df.filter(pl.col("event_type") == "gap")
    # ONE ws-level gap marker invalidates BOTH symbols sharing the connection
    assert set(gap_rows["symbol"].to_list()) == {"BTCUSDT", "ETHUSDT"}
    assert gap_rows["valid"].to_list() == [False, False]


def test_market_category_gap_does_not_invalidate_the_book(tmp_path):
    """A gap on the "market" (aggTrade) connection has no bearing on
    whether any depth event was missed -- since capture.py now routes
    depth and aggTrade over separate connections (see its module docstring),
    invalidating the book on every aggTrade hiccup would be needless,
    incorrect caution. Regression test for exactly that: a public-category
    gap invalidates, a market-category one does not, using the SAME book
    state before and after to prove nothing changed."""
    from tests.helpers import gap_envelope

    symbol = "BTCUSDT"
    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=1, mono_ns=1, event_time_ms=1000),
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=2, mono_ns=2),
        gap_envelope("market:reconnect", wall_ns=3, mono_ns=3),
        # book must still be valid and continuable after a market-category gap
        depth_update_envelope(symbol, U=106, u=110, pu=105, bids=[], asks=[["100.00", "1.5"]], event_time_ms=1020, wall_ns=4, mono_ns=4),
    ]
    capture_path = tmp_path / "capture" / "file1.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=5)

    assert stats.gaps_detected == 0  # no BOOK-affecting gap occurred
    assert stats.non_book_gaps_detected == 1  # the market-category gap is still counted, just separately

    df = pl.read_parquet(out_dir / "book_state.parquet").sort("recv_wall_ns")
    rows = df.to_dicts()
    # no "gap" event_type row was written to book_state.parquet at all --
    # a market-category gap doesn't touch book state, so there's nothing
    # book-related to mark
    assert "gap" not in [r["event_type"] for r in rows]
    # the update AFTER the market-category gap applied normally, proving
    # the book was never invalidated
    last = rows[-1]
    assert last["valid"] is True
    assert last["ask_prices"] == [100.0]
    assert last["ask_qtys"] == [1.5]


def test_public_category_gap_still_invalidates_the_book(tmp_path):
    """Companion to the market-category test: a public-category (depth)
    gap must still invalidate, exactly as before the routing split."""
    from tests.helpers import gap_envelope

    symbol = "BTCUSDT"
    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=1, mono_ns=1, event_time_ms=1000),
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=2, mono_ns=2),
        gap_envelope("public:reconnect", wall_ns=3, mono_ns=3),
    ]
    capture_path = tmp_path / "capture" / "file1.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=5)

    assert stats.gaps_detected == 1
    assert stats.non_book_gaps_detected == 0

    df = pl.read_parquet(out_dir / "book_state.parquet").sort("recv_wall_ns")
    last = df.to_dicts()[-1]
    assert last["event_type"] == "gap"
    assert last["valid"] is False
