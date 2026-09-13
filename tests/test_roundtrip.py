"""Round-trip test: capture (real on-disk format) -> reconstruct -> verify
against snapshot (quant-backtest-discipline testing requirement)."""

from __future__ import annotations

import polars as pl

import reconstruct
from tests.helpers import agg_trade_envelope, depth_update_envelope, snapshot_envelope, write_capture_file


def test_capture_reconstruct_verify_round_trip(tmp_path):
    symbol = "BTCUSDT"
    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=100, mono_ns=100, event_time_ms=1000),
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=200, mono_ns=200),
        depth_update_envelope(symbol, U=106, u=110, pu=105, bids=[], asks=[["100.00", "1.5"]], event_time_ms=1020, wall_ns=300, mono_ns=300),
        agg_trade_envelope(symbol, agg_trade_id=1, price="99.50", qty="0.01", first_id=1, last_id=1, is_buyer_maker=True, trade_time_ms=1015, wall_ns=250, mono_ns=250),
        # A SECOND, later snapshot -- independently verifiable against the
        # reconstructed state at that point in time (the book should have
        # continued applying diffs correctly since bootstrap).
        depth_update_envelope(symbol, U=111, u=115, pu=110, bids=[["98.50", "0.5"]], asks=[], event_time_ms=1030, wall_ns=400, mono_ns=400),
        snapshot_envelope(
            symbol,
            115,
            bids=[["99.00", "2.0"], ["98.50", "0.5"]],
            asks=[["100.00", "1.5"]],
            wall_ns=500,
            mono_ns=500,
            event_time_ms=1040,
        ),
    ]
    capture_path = tmp_path / "capture" / "2026-01-01T000000Z.jsonl.zst"
    write_capture_file(capture_path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=10)

    assert stats.gaps_detected == 0  # this capture has a clean, unbroken sequence
    assert stats.trades_processed == 1

    book_df = pl.read_parquet(out_dir / "book_state.parquet")
    trades_df = pl.read_parquet(out_dir / "trades.parquet")

    assert trades_df.height == 1
    trade = trades_df.row(0, named=True)
    assert trade["price"] == 99.50
    assert trade["is_buyer_maker"] is True

    reports = reconstruct.verify_capture_snapshots(book_df, [capture_path], [symbol])
    assert len(reports) == 2  # both snapshot records in the file

    for report in reports:
        assert report.reconstructed_row_valid is True
        assert report.note == "ok"
        assert report.levels_matched == report.levels_compared
        assert report.max_bid_price_diff == 0.0
        assert report.max_ask_price_diff == 0.0


def test_verify_reports_divergence_without_correcting_anything(tmp_path):
    """verify_against_snapshot must REPORT a mismatch, and must not alter
    the reconstructed book_df in any way (market-data-integrity:
    "Verification reports. It never auto-corrects.")."""
    symbol = "BTCUSDT"
    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=100, mono_ns=100, event_time_ms=1000),
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=200, mono_ns=200),
    ]
    capture_path = tmp_path / "capture" / "file.jsonl.zst"
    write_capture_file(capture_path, envs)
    out_dir = tmp_path / "out"
    reconstruct.reconstruct([capture_path], [symbol], out_dir=out_dir, depth_levels=10)

    book_df = pl.read_parquet(out_dir / "book_state.parquet")
    before = book_df.clone()

    # A snapshot claiming a DIFFERENT price than what was reconstructed
    import json

    bogus_snapshot_raw = json.dumps({"lastUpdateId": 105, "E": 1010, "T": 1010, "bids": [["50.00", "9.0"]], "asks": [["51.00", "9.0"]]})
    report = reconstruct.verify_against_snapshot(book_df, symbol, bogus_snapshot_raw, snapshot_recv_wall_ns=300)

    assert report.note != "ok"
    assert report.max_bid_price_diff > 0.0
    assert book_df.equals(before)  # verification must not mutate the reconstructed frame
