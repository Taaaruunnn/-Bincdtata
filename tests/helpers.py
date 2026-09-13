"""Test-only helpers for building synthetic capture.py-format files without
touching the network. Reuses capture.py's own envelope encoder and zstd
framing so tests exercise the real on-disk format, not a hand-rolled
approximation of it."""

from __future__ import annotations

import json
from pathlib import Path

import zstandard as zstd

import capture


def write_capture_file(path: Path, envelopes: list[bytes]) -> None:
    """Write `envelopes` (each produced by capture._envelope_bytes) as a
    multi-frame zstd JSONL file, flushing a frame after every envelope --
    mirroring capture.py's periodic FLUSH_FRAME behavior so tests exercise
    the same multi-frame-concatenation reader path reconstruct.py uses in
    production."""
    path.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstd.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        writer = compressor.stream_writer(fh)
        for env in envelopes:
            writer.write(env)
            writer.flush(zstd.FLUSH_FRAME)
        writer.close()


def gap_envelope(reason: str, *, wall_ns: int, mono_ns: int) -> bytes:
    return capture._envelope_bytes("gap", recv_wall_ns=wall_ns, recv_mono_ns=mono_ns, reason=reason)


def snapshot_envelope(symbol: str, last_update_id: int, bids: list[list], asks: list[list], *, wall_ns: int, mono_ns: int, event_time_ms: int) -> bytes:
    body = json.dumps({"lastUpdateId": last_update_id, "E": event_time_ms, "T": event_time_ms, "bids": bids, "asks": asks})
    return capture._envelope_bytes(
        "snapshot",
        recv_wall_ns=wall_ns,
        recv_mono_ns=mono_ns,
        stream=f"{symbol.lower()}@depth_snapshot",
        raw=body,
    )


def depth_update_envelope(
    symbol: str,
    *,
    U: int,
    u: int,
    pu: int,
    bids: list[list],
    asks: list[list],
    event_time_ms: int,
    wall_ns: int,
    mono_ns: int,
) -> bytes:
    data = {"e": "depthUpdate", "E": event_time_ms, "T": event_time_ms, "s": symbol.upper(), "U": U, "u": u, "pu": pu, "b": bids, "a": asks}
    wrapper = json.dumps({"stream": f"{symbol.lower()}@depth@100ms", "data": data})
    return capture._envelope_bytes(
        "message",
        recv_wall_ns=wall_ns,
        recv_mono_ns=mono_ns,
        stream=f"{symbol.lower()}@depth@100ms",
        raw=wrapper,
    )


def agg_trade_envelope(
    symbol: str,
    *,
    agg_trade_id: int,
    price: str,
    qty: str,
    first_id: int,
    last_id: int,
    is_buyer_maker: bool,
    trade_time_ms: int,
    wall_ns: int,
    mono_ns: int,
) -> bytes:
    data = {
        "e": "aggTrade",
        "E": trade_time_ms,
        "a": agg_trade_id,
        "s": symbol.upper(),
        "p": price,
        "q": qty,
        "f": first_id,
        "l": last_id,
        "T": trade_time_ms,
        "m": is_buyer_maker,
    }
    wrapper = json.dumps({"stream": f"{symbol.lower()}@aggTrade", "data": data})
    return capture._envelope_bytes(
        "message",
        recv_wall_ns=wall_ns,
        recv_mono_ns=mono_ns,
        stream=f"{symbol.lower()}@aggTrade",
        raw=wrapper,
    )
