"""Diagnostic: explain a book_state.parquet snapshot row's divergence from
the raw REST snapshot it was bootstrapped from.

verify_against_snapshot() / verify_capture_snapshots() match rows by
recv_wall_ns, so a snapshot row will trivially have recv_wall_ns identical
to the very snapshot it diverges from -- that check alone cannot tell you
whether the divergence is legitimate (the row's content was correctly
advanced past the snapshot by buffered-event replay in
BookState._bootstrap_from_buffer, per the documented "buffer events before
fetching snapshot" step) or a real application bug in apply_diff.

This script closes that gap for a specific row WITHOUT needing a fresh live
snapshot: it reconstructs, by hand, the exact replay chain
_bootstrap_from_buffer would have walked (same bracket check, same pu-chain
check -- see reconstruct.BookState._apply), and for every level where the
row disagrees with the raw snapshot, reports which captured depthUpdate
event (by its `u`) was the last one to touch that price, and whether that
event's own recorded quantity matches what ended up in the row. If every
diverging level traces back cleanly to a real, correctly-applied captured
event, the divergence is the "book is validly more current" signature, not
corruption. If any level does NOT trace back cleanly, that is evidence of a
real reconstruction bug and should not be dismissed as buffer replay.

This is an ad hoc investigation tool, not part of the library or test
suite -- it reuses reconstruct.py's private _iter_envelopes() reader
because it needs the same raw envelopes reconstruct.py itself works from,
not a public API.

Usage:
    python scripts/diagnose_snapshot_divergence.py \
        --capture smoke_capture/2026-09-13/2026-09-13T090000Z.jsonl.zst \
        --book-state reconstructed/book_state.parquet \
        --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `import reconstruct` works run from anywhere

import polars as pl

import reconstruct

TOLERANCE = 1e-9


def load_events_for_symbol(capture_path: Path, symbol: str) -> tuple[list[dict], list[dict]]:
    """Returns (depth_update_events, raw_snapshots), both in file order,
    each event/snapshot as a plain dict with prices/qtys already cast to
    float. Only json.loads'd here -- this script, like reconstruct.py, is
    an offline interpretation step, not the capture process."""
    depth_stream = f"{symbol.lower()}@depth@100ms"
    snapshot_stream = f"{symbol.lower()}@depth_snapshot"
    events: list[dict] = []
    snapshots: list[dict] = []

    for env in reconstruct._iter_envelopes(capture_path):
        if env.get("type") == "message" and env.get("stream") == depth_stream:
            data = json.loads(env["raw"])["data"]
            if data.get("e") != "depthUpdate":
                continue
            events.append(
                {
                    "U": int(data["U"]),
                    "u": int(data["u"]),
                    "pu": int(data["pu"]),
                    "b": [(float(p), float(q)) for p, q in data["b"]],
                    "a": [(float(p), float(q)) for p, q in data["a"]],
                    "recv_wall_ns": env["recv_wall_ns"],
                }
            )
        elif env.get("type") == "snapshot" and env.get("stream") == snapshot_stream:
            snap = json.loads(env["raw"])
            snapshots.append(
                {
                    "lastUpdateId": int(snap["lastUpdateId"]),
                    "bids": [(float(p), float(q)) for p, q in snap["bids"]],
                    "asks": [(float(p), float(q)) for p, q in snap["asks"]],
                    "recv_wall_ns": env["recv_wall_ns"],
                }
            )

    return events, snapshots


def find_replay_chain(events: list[dict], snapshot_last_update_id: int, target_update_id: int) -> list[dict]:
    """Reproduces BookState._bootstrap_from_buffer's event selection by
    hand: the same bracket check (U <= lastUpdateId <= u) to find the first
    applicable event, then the same pu == previous-u chaining forward,
    exactly mirroring reconstruct.BookState._apply's logic. Raises
    AssertionError if the events captured on disk don't actually support
    the chain reconstruct.py must have walked to produce target_update_id --
    that would itself be a finding, not something to swallow silently.
    """
    chain: list[dict] = []
    started = False
    prev_u: int | None = None

    for ev in events:
        if not started:
            if ev["u"] < snapshot_last_update_id:
                continue  # stale relative to the snapshot; would have been dropped, not applied
            if not (ev["U"] <= snapshot_last_update_id <= ev["u"]):
                raise AssertionError(
                    f"no event brackets lastUpdateId={snapshot_last_update_id} before "
                    f"hitting u={ev['u']} > it; reconstruct.py's replay could not have "
                    f"looked like this from these captured events"
                )
            started = True
        else:
            if ev["pu"] != prev_u:
                raise AssertionError(f"pu-chain broken: event pu={ev['pu']} != previous u={prev_u}")

        chain.append(ev)
        prev_u = ev["u"]
        if prev_u == target_update_id:
            return chain
        if prev_u is not None and prev_u > target_update_id:
            raise AssertionError(f"replay chain overshot target_update_id={target_update_id} at u={prev_u}")

    raise AssertionError(
        f"replay chain never reached update_id={target_update_id}; only reached u={prev_u} "
        f"(row's update_id may be stale relative to this capture file, or the file is incomplete)"
    )


def diverging_levels(ref: list[tuple[float, float]], rec_prices: list[float], rec_qtys: list[float]) -> list[tuple[float, float, float]]:
    ref_map = dict(ref[: len(rec_prices)])
    rec_map = dict(zip(rec_prices, rec_qtys))
    out = []
    for price, ref_qty in ref_map.items():
        rec_qty = rec_map.get(price)
        if rec_qty is not None and abs(rec_qty - ref_qty) > TOLERANCE:
            out.append((price, ref_qty, rec_qty))
    return out


def explain_level(chain: list[dict], side: str, price: float) -> tuple[int, float] | None:
    """Walks the replay chain in order; the LAST event touching this exact
    price on this side determines the final quantity (absolute-quantity
    semantics -- later events overwrite earlier ones at the same price)."""
    last: tuple[int, float] | None = None
    key = "b" if side == "bid" else "a"
    for ev in chain:
        for p, q in ev[key]:
            if p == price:
                last = (ev["u"], q)
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--book-state", type=Path, required=True)
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()

    events, snapshots = load_events_for_symbol(args.capture, args.symbol)
    print(f"loaded {len(events)} depthUpdate event(s), {len(snapshots)} raw snapshot(s) for {args.symbol}")

    book_df = pl.read_parquet(args.book_state)
    snapshot_rows = book_df.filter(
        (pl.col("symbol") == args.symbol) & (pl.col("event_type") == "snapshot")
    ).sort("recv_wall_ns")

    if snapshot_rows.is_empty():
        raise SystemExit(f"no snapshot-type rows for {args.symbol} in {args.book_state}")

    any_mismatch = False

    for row in snapshot_rows.to_dicts():
        raw_snap = next((s for s in snapshots if s["recv_wall_ns"] == row["recv_wall_ns"]), None)
        print(f"\n=== row recv_wall_ns={row['recv_wall_ns']} row.update_id={row['update_id']} valid={row['valid']} ===")

        if raw_snap is None:
            print("  [skip] no raw snapshot envelope in this capture file matches this row's recv_wall_ns")
            continue

        if not row["valid"]:
            print(f"  row is INVALID -- book was cleared, not a divergence to explain (raw lastUpdateId={raw_snap['lastUpdateId']})")
            continue

        bid_div = diverging_levels(raw_snap["bids"], row["bid_prices"], row["bid_qtys"])
        ask_div = diverging_levels(raw_snap["asks"], row["ask_prices"], row["ask_qtys"])

        if not bid_div and not ask_div:
            print(f"  no divergence vs raw snapshot (raw lastUpdateId={raw_snap['lastUpdateId']}, row update_id={row['update_id']})")
            continue

        print(
            f"  {len(bid_div) + len(ask_div)} diverging level(s) vs raw snapshot "
            f"(raw lastUpdateId={raw_snap['lastUpdateId']} -> row update_id={row['update_id']})"
        )

        try:
            chain = find_replay_chain(events, raw_snap["lastUpdateId"], row["update_id"])
        except AssertionError as exc:
            print(f"  [UNEXPLAINED] could not reconstruct the replay chain: {exc}")
            any_mismatch = True
            continue

        print(f"  replay chain: {len(chain)} event(s), u sequence = {[e['u'] for e in chain]}")

        for side, divs in (("bid", bid_div), ("ask", ask_div)):
            for price, ref_qty, rec_qty in divs:
                touch = explain_level(chain, side, price)
                if touch is None:
                    print(f"  [UNEXPLAINED] {side} {price}: raw_qty={ref_qty} row_qty={rec_qty} -- no event in the replay chain touches this price")
                    any_mismatch = True
                elif abs(touch[1] - rec_qty) < TOLERANCE:
                    print(f"  [OK] {side} {price}: raw_qty={ref_qty} row_qty={rec_qty} == event(u={touch[0]}).qty={touch[1]}")
                else:
                    print(f"  [MISMATCH] {side} {price}: row_qty={rec_qty} != event(u={touch[0]}).qty={touch[1]} (raw_qty={ref_qty})")
                    any_mismatch = True

    print()
    if any_mismatch:
        print("RESULT: at least one diverging level did NOT trace back cleanly to a captured event. "
              "Treat this as a real reconstruction discrepancy, not benign buffer replay -- do not dismiss it.")
    else:
        print("RESULT: every diverging level traces back exactly to a real captured depthUpdate event's own "
              "recorded quantity. This is the expected 'book validly more current than the raw snapshot' "
              "signature, not corruption. It does NOT independently confirm the book matches Binance's true "
              "live state -- only that apply_diff transcribed the captured events correctly. The non-circular "
              "check (independent live snapshot vs. near-real-time reconstruction) is still the only check that "
              "validates against ground truth Binance itself has, rather than against this codebase's own capture.")


if __name__ == "__main__":
    main()
