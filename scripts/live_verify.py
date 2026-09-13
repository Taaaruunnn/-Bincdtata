"""Non-circular verification: compare a near-real-time reconstruction
against a FRESH, independently-fetched REST snapshot -- never a snapshot
record already sitting in the capture file.

Why this is a different (and stronger) check than
verify_capture_snapshots(): that path, by construction, only ever compares
a reconstructed row against the very snapshot reconstruct.py bootstrapped
it from -- ground truth relative to this codebase's OWN capture, not to
Binance's live book (see reconstruct.verify_capture_snapshots' own
docstring, and the conversation that led to diagnose_snapshot_divergence.py
for why that matters). This script's snapshot comes from a fresh
binance_rest.fetch_depth_snapshot() call made at run time, so the
comparison is against Binance's actual live state.

DESIGN NOTE on ordering: the snapshot is fetched FIRST, then the script
waits `--settle-seconds` (default matches capture.py's flush_interval_s,
5s) before reconstructing, so the capture file has had a chance to flush
diff events covering the snapshot's instant to disk before we read them.
Reconstructing first and fetching second would compare the snapshot against
whatever reconstructed row happened to be the latest one already on disk --
which could be stale by an uncontrolled amount and would manufacture
"divergence" that's really just capture lag, not a reconstruction defect.

DESIGN NOTE on reading a currently-open capture file: capture.py's
_RotatingZstdWriter only writes bytes to the underlying file when it calls
zstd's FLUSH_FRAME (periodically, and on rotation/close) -- verified
empirically during development that data written to the zstd stream_writer
without an intervening flush is NOT yet visible on disk at all, so a reader
opening the file concurrently just sees however many complete frames have
been flushed so far and reaches a clean EOF, not a truncated/corrupt tail.
No special exception handling is needed here for that reason, but it is a
real assumption (small writes are effectively atomic at the OS level) --
if the capture volume/environment ever changes, re-verify.

Usage (while capture.py is running against the same --capture-dir):
    python scripts/live_verify.py \
        --capture-dir smoke_capture \
        --symbol BTCUSDT \
        --out-dir live_verify_out
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import reconstruct` works run from anywhere

import polars as pl

import binance_rest
import reconstruct

LOG = logging.getLogger("live_verify")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", type=Path, required=True, help="directory capture.py is currently writing to")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--depth-levels", type=int, default=20, help="levels reconstructed per event (matches reconstruct.py)")
    ap.add_argument("--verify-levels", type=int, default=20, help="levels compared against the live snapshot")
    ap.add_argument("--snapshot-limit", type=int, default=1000, choices=sorted(binance_rest.DEPTH_LIMIT_WEIGHTS), help="REST depth limit param")
    ap.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="wait this long after fetching the snapshot before reconstructing, so capture.py's periodic "
        "FLUSH_FRAME has a chance to make diff events covering the snapshot's instant visible on disk",
    )
    ap.add_argument("--show-levels", action="store_true", help="print a rank-by-rank side-by-side table (debugging)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    LOG.info("fetching independent live snapshot for %s (limit=%d)...", args.symbol, args.snapshot_limit)
    snapshot_raw = binance_rest.fetch_depth_snapshot_sync(args.symbol, limit=args.snapshot_limit)
    snapshot_recv_wall_ns = time.time_ns()
    LOG.info("  fetched at recv_wall_ns=%d", snapshot_recv_wall_ns)

    LOG.info("waiting %.1fs for capture to flush data past this instant...", args.settle_seconds)
    time.sleep(args.settle_seconds)

    paths = sorted(args.capture_dir.rglob("*.jsonl.zst"))
    if not paths:
        raise SystemExit(f"no capture files found under {args.capture_dir} -- is capture.py running with this --out-dir?")
    LOG.info("reconstructing %s from %d capture file(s)...", args.symbol, len(paths))

    stats = reconstruct.reconstruct(paths, [args.symbol], out_dir=args.out_dir, depth_levels=args.depth_levels)
    LOG.info("reconstruction stats: %s", stats)

    book_df = pl.read_parquet(args.out_dir / "book_state.parquet")
    if book_df.filter(pl.col("symbol") == args.symbol).is_empty():
        raise SystemExit(f"no reconstructed rows for {args.symbol} -- check the symbol and capture-dir")

    report = reconstruct.verify_against_snapshot(
        book_df, args.symbol, snapshot_raw, snapshot_recv_wall_ns, levels=args.verify_levels
    )

    # lag against the row verify_against_snapshot ACTUALLY matched (recv_wall_ns
    # <= snapshot time, closest below) -- not just the latest row in book_df,
    # which could be from after the snapshot and would report a nonsensical
    # negative "lag".
    if report.reconstructed_recv_wall_ns is not None:
        lag_ms = (snapshot_recv_wall_ns - report.reconstructed_recv_wall_ns) / 1e6
        LOG.info("matched reconstructed row is %.1fms before the snapshot fetch", lag_ms)
    else:
        lag_ms = float("nan")

    print()
    print(f"symbol:                          {report.symbol}")
    print(f"snapshot recv_wall_ns:           {report.snapshot_recv_wall_ns}")
    print(f"reconstructed recv_wall_ns:      {report.reconstructed_recv_wall_ns}")
    print(f"reconstructed row valid:         {report.reconstructed_row_valid}")
    print(f"levels compared / matched:       {report.levels_compared} / {report.levels_matched}")
    print(f"bid only-in-snapshot/-reconstr.: {report.bid_only_in_snapshot} / {report.bid_only_in_reconstruction}")
    print(f"ask only-in-snapshot/-reconstr.: {report.ask_only_in_snapshot} / {report.ask_only_in_reconstruction}")
    print(f"max same-price qty diff (bid/ask): {report.max_bid_qty_diff} / {report.max_ask_qty_diff}")
    print(f"note:                            {report.note}")
    print()

    if args.show_levels and report.reconstructed_row_valid:
        import json

        snap = json.loads(snapshot_raw)
        candidates = book_df.filter(
            (pl.col("symbol") == args.symbol) & (pl.col("recv_wall_ns") <= snapshot_recv_wall_ns) & (pl.col("valid"))
        ).sort("recv_wall_ns")
        row = candidates.row(-1, named=True)

        def _print_side(label: str, ref: list, rec_prices: list, rec_qtys: list) -> None:
            print(f"  {label:>4}  {'rank':<5}{'snapshot(px,qty)':<28}{'reconstructed(px,qty)':<28}")
            n = max(len(ref), len(rec_prices))
            for i in range(min(n, args.verify_levels)):
                ref_p, ref_q = (float(ref[i][0]), float(ref[i][1])) if i < len(ref) else (None, None)
                rec_p, rec_q = (rec_prices[i], rec_qtys[i]) if i < len(rec_prices) else (None, None)
                mark = "  " if (ref_p == rec_p and ref_q == rec_q) else "<<"
                print(f"        {i:<5}{f'{ref_p}, {ref_q}':<28}{f'{rec_p}, {rec_q}':<28}{mark}")

        print("side-by-side (<< marks a diverging rank):")
        _print_side("BID", snap["bids"], row["bid_prices"], row["bid_qtys"])
        _print_side("ASK", snap["asks"], row["ask_prices"], row["ask_qtys"])
        print()

    if not report.reconstructed_row_valid:
        print(
            "RESULT: no valid reconstructed row exists at/before the snapshot's receipt time. This is not a "
            "divergence finding -- it means the book was INVALID (mid-resync, or capture just started and hasn't "
            "bootstrapped yet) at the moment being checked. Re-run once capture has had more time, or check "
            "reconstruction stats above for gaps_detected."
        )
    elif report.note == "ok":
        print(
            "RESULT: every compared level matches the live snapshot exactly. This is the check that "
            "diagnose_snapshot_divergence.py's own result could not provide -- ground truth here is Binance's "
            "live book, not this codebase's own capture. Trustworthy for this symbol at this point in time."
        )
    else:
        qty_mismatched = report.levels_compared - report.levels_matched - report.bid_only_in_snapshot - report.ask_only_in_snapshot
        only_count = report.bid_only_in_snapshot + report.bid_only_in_reconstruction + report.ask_only_in_snapshot + report.ask_only_in_reconstruction
        if qty_mismatched == 0:
            print(
                f"RESULT: divergence is entirely {only_count} level(s) only present on one side (insertions/"
                f"removals in the {lag_ms:.1f}ms + REST round-trip gap between the reconstructed row and the "
                "snapshot fetch), with ZERO same-price quantity disagreement. That is the expected, benign "
                "signature of a live book moving between two fetches -- use --show-levels to see exactly which "
                "level(s) and confirm they're a plausible single order arriving/leaving, not a pattern."
            )
        else:
            print(
                f"RESULT: {qty_mismatched} level(s) show the SAME price on both sides but a DIFFERENT quantity -- "
                "this is not explained by an insertion or removal (those show up as only-in-snapshot/-reconstruction "
                "instead, reported separately above) and is not a rank-shift artifact, since matching is by price. "
                "This is the signal worth investigating as a real reconstruction defect. Use --show-levels, re-run "
                "a few times, and check whether it's the same price/magnitude repeatedly (bug) or different each "
                "time (still just fast-moving quantity at that price, which is possible but less likely at a "
                f"{lag_ms:.1f}ms gap -- judge by magnitude relative to typical book depth at that level)."
            )


if __name__ == "__main__":
    main()
