"""Tests for reconstruct.py's handling of a malformed/truncated JSONL line
-- observed in practice from capture.py being killed (Ctrl+C) mid-write,
leaving a truncated final line in a capture file. A single corrupt line
must not crash the whole reconstruction: it gets logged (file, line
number, decompressed byte offset), counted in
ReconstructionStats.corrupt_envelopes, and reconstruction continues with
everything else in the file."""

from __future__ import annotations

import polars as pl

import reconstruct
from tests.helpers import depth_update_envelope, gap_envelope, snapshot_envelope, write_capture_file


def test_iter_envelopes_yields_corrupt_sentinel_and_continues(tmp_path):
    good1 = gap_envelope("public:initial_connect", wall_ns=1, mono_ns=1)
    good2 = depth_update_envelope("BTCUSDT", U=1, u=5, pu=0, bids=[], asks=[], event_time_ms=1000, wall_ns=2, mono_ns=2)
    # simulates a process killed mid-write: real envelope bytes always end
    # in "\n" (see capture._envelope_bytes), so a genuine truncation cuts
    # off BEFORE that newline. Re-adding "\n" here isolates the corrupt
    # content onto its own line rather than merging it with whatever comes
    # next -- without it, the next envelope's bytes would run onto the
    # same line and get swallowed into the same JSON-decode error, testing
    # a different (and less realistic) failure than intended.
    truncated = good2[: len(good2) // 2] + b"\n"
    good3 = gap_envelope("public:reconnect", wall_ns=3, mono_ns=3)

    path = tmp_path / "capture.jsonl.zst"
    write_capture_file(path, [good1, truncated, good3])

    envelopes = list(reconstruct._iter_envelopes(path))
    assert len(envelopes) == 3
    assert envelopes[0]["type"] == "gap"
    assert envelopes[0]["reason"] == "public:initial_connect"

    corrupt = envelopes[1]
    assert corrupt["type"] == "corrupt"
    assert corrupt["line_no"] == 2
    assert "error" in corrupt
    assert isinstance(corrupt["decompressed_byte_offset"], int)

    # crucially: the envelope AFTER the corrupt one still comes through
    assert envelopes[2]["type"] == "gap"
    assert envelopes[2]["reason"] == "public:reconnect"


def test_reconstruct_does_not_crash_on_mid_file_corruption(tmp_path):
    symbol = "BTCUSDT"
    good_update = depth_update_envelope(symbol, U=1, u=5, pu=0, bids=[], asks=[], event_time_ms=1000, wall_ns=1, mono_ns=1)
    truncated = good_update[: len(good_update) // 2] + b"\n"  # isolate onto its own line -- see comment above

    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=10, mono_ns=10, event_time_ms=1000),
        depth_update_envelope(symbol, U=95, u=105, pu=90, bids=[["99.00", "2.0"]], asks=[], event_time_ms=1010, wall_ns=20, mono_ns=20),
        truncated,  # corrupt line in the middle of the file
        # a fully valid event AFTER the corruption must still be processed
        depth_update_envelope(symbol, U=106, u=110, pu=105, bids=[], asks=[["100.00", "1.5"]], event_time_ms=1020, wall_ns=30, mono_ns=30),
    ]
    path = tmp_path / "capture.jsonl.zst"
    write_capture_file(path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([path], [symbol], out_dir=out_dir, depth_levels=10)

    assert stats.corrupt_envelopes == 1
    assert stats.gaps_detected == 0  # the corruption is not a "gap" record, so it must not invalidate the book

    df = pl.read_parquet(out_dir / "book_state.parquet").sort("recv_wall_ns")
    last = df.to_dicts()[-1]
    # the update AFTER the corrupt line applied normally -- proof the
    # corruption didn't kill (or silently break) the rest of the run
    assert last["valid"] is True
    assert last["ask_prices"] == [100.0]
    assert last["ask_qtys"] == [1.5]


def test_corrupt_line_at_end_of_file_does_not_crash(tmp_path):
    """The specific real-world scenario reported: capture.py killed
    mid-write, leaving the truncated line as the LAST thing in the file
    (nothing valid after it to prove continuation against, unlike the
    mid-file test above -- this just proves reconstruct() returns
    normally instead of raising)."""
    symbol = "BTCUSDT"
    good = depth_update_envelope(symbol, U=1, u=5, pu=0, bids=[], asks=[], event_time_ms=1000, wall_ns=1, mono_ns=1)
    truncated_last = good[: len(good) // 2]

    envs = [
        snapshot_envelope(symbol, 100, bids=[["99.00", "1.0"]], asks=[["100.00", "1.0"]], wall_ns=10, mono_ns=10, event_time_ms=1000),
        truncated_last,
    ]
    path = tmp_path / "capture.jsonl.zst"
    write_capture_file(path, envs)

    out_dir = tmp_path / "out"
    stats = reconstruct.reconstruct([path], [symbol], out_dir=out_dir, depth_levels=10)  # must not raise

    assert stats.corrupt_envelopes == 1
    assert stats.snapshots_applied == 1
