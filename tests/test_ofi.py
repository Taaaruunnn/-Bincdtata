"""Tests for ofi.py: correct null placement at reconstruction-gap
boundaries (OFI must never be computed across a gap), the aggressor-flag
mapping, and that the cost-hurdle overlay is wired to the real cost model."""

from __future__ import annotations

import polars as pl
import pytest

import ofi
from costs import CryptoPerpChargeSchedule, Side


def _book_row(symbol, event_type, valid, bid_price=None, bid_qty=None, ask_price=None, ask_qty=None, t=0):
    return {
        "symbol": symbol,
        "event_type": event_type,
        "event_time_ns": t,
        "update_id": None,
        "recv_wall_ns": t,
        "recv_mono_ns": t,
        "valid": valid,
        "bid_prices": [] if bid_price is None else [bid_price],
        "bid_qtys": [] if bid_qty is None else [bid_qty],
        "ask_prices": [] if ask_price is None else [ask_price],
        "ask_qtys": [] if ask_qty is None else [ask_qty],
    }


def test_ofi_is_null_at_first_event_of_each_segment():
    rows = [
        _book_row("BTCUSDT", "snapshot", True, 100.0, 1.0, 101.0, 1.0, t=0),
        _book_row("BTCUSDT", "update", True, 100.0, 2.0, 101.0, 1.0, t=1),  # 2nd event of segment 1: real OFI
        _book_row("BTCUSDT", "gap", False, t=2),
        _book_row("BTCUSDT", "snapshot", True, 50.0, 5.0, 51.0, 5.0, t=3),  # 1st event of segment 2
        _book_row("BTCUSDT", "update", True, 50.0, 6.0, 51.0, 5.0, t=4),  # 2nd event of segment 2: real OFI
    ]
    df = pl.DataFrame(rows)
    events = ofi.ofi_events(df, "BTCUSDT")

    assert events.height == 4  # gap row excluded, only snapshot/update rows
    ofi_values = events["ofi_event"].to_list()
    # first event of segment 1 (the bootstrap snapshot) -> null
    assert ofi_values[0] is None
    # second event of segment 1 -> real OFI, not null
    assert ofi_values[1] is not None
    # first event of segment 2 (post-gap resync) -> null again, NOT computed
    # against segment 1's last observation
    assert ofi_values[2] is None
    assert ofi_values[3] is not None


def test_ofi_event_formula_hand_computed():
    # bid price increases 100 -> 101: e_bid = new bid qty (3.0)
    # ask price unchanged 105 -> 105: e_ask = new_ask_qty - old_ask_qty = 2.0 - 1.0 = 1.0
    # OFI = e_bid - e_ask = 3.0 - 1.0 = 2.0
    rows = [
        _book_row("BTCUSDT", "snapshot", True, 100.0, 1.0, 105.0, 1.0, t=0),
        _book_row("BTCUSDT", "update", True, 101.0, 3.0, 105.0, 2.0, t=1),
    ]
    events = ofi.ofi_events(pl.DataFrame(rows), "BTCUSDT")
    assert events["ofi_event"].to_list()[1] == pytest.approx(2.0)


def test_ofi_event_formula_bid_price_decrease_and_ask_price_decrease():
    # bid price decreases 100 -> 99: e_bid = -old_bid_qty = -1.0
    # ask price decreases 105 -> 104: e_ask = new_ask_qty = 2.0
    # OFI = e_bid - e_ask = -1.0 - 2.0 = -3.0
    rows = [
        _book_row("BTCUSDT", "snapshot", True, 100.0, 1.0, 105.0, 1.0, t=0),
        _book_row("BTCUSDT", "update", True, 99.0, 5.0, 104.0, 2.0, t=1),
    ]
    events = ofi.ofi_events(pl.DataFrame(rows), "BTCUSDT")
    assert events["ofi_event"].to_list()[1] == pytest.approx(-3.0)


def test_aggressor_side_mapping_matches_binance_semantics():
    # m=True: buyer is the maker -> seller crossed the spread -> aggressor SOLD
    assert ofi.aggressor_side(True) == Side.SELL
    # m=False: buyer is NOT the maker -> buyer crossed the spread -> aggressor BOUGHT
    assert ofi.aggressor_side(False) == Side.BUY


def test_annotate_trades_signed_quantity():
    trades = pl.DataFrame({"quantity": [1.0, 2.0], "is_buyer_maker": [True, False]})
    out = ofi.annotate_trades(trades)
    assert out["aggressor_side"].to_list() == ["SELL", "BUY"]
    assert out["signed_quantity"].to_list() == [-1.0, 2.0]


def test_round_trip_taker_cost_bps_matches_schedule():
    sched = CryptoPerpChargeSchedule(maker_bps=1.0, taker_bps=4.0)
    assert ofi.round_trip_taker_cost_bps(sched) == pytest.approx(8.0)  # 2 * taker_bps, price/qty-independent


def test_overlay_marks_predicted_edge_below_cost_as_not_exceeding():
    sched = CryptoPerpChargeSchedule()  # round trip = 10 bps
    cells = [ofi.DecayCell(window_ms=1000, horizon_ms=1000, correlation=0.01, n_obs=1000, predicted_edge_bps=0.5)]
    overlaid = ofi.overlay_cost_hurdle(cells, sched)
    assert overlaid[0].exceeds_cost is False
    assert overlaid[0].round_trip_cost_bps == pytest.approx(10.0)


def test_overlay_marks_predicted_edge_above_cost_as_exceeding():
    sched = CryptoPerpChargeSchedule()
    cells = [ofi.DecayCell(window_ms=1000, horizon_ms=1000, correlation=0.5, n_obs=1000, predicted_edge_bps=50.0)]
    overlaid = ofi.overlay_cost_hurdle(cells, sched)
    assert overlaid[0].exceeds_cost is True


# --------------------------------------------------------------------------
# trade_flow_decay_matrix
# --------------------------------------------------------------------------


def _synthetic_trades(n: int = 3000, seed: int = 11) -> pl.DataFrame:
    import random

    rng = random.Random(seed)
    price = 100.0
    t_ns = 0
    drift = 0.0
    rows = []
    for i in range(n):
        t_ns += rng.randint(50_000_000, 150_000_000)  # 50-150ms between trades
        is_buyer_maker = rng.random() < 0.5
        qty = rng.uniform(0.01, 1.0)
        signed = -qty if is_buyer_maker else qty  # matches annotate_trades' convention
        # `drift` carries recent flow forward with decay, so price keeps
        # moving for several trades AFTER a burst of one-sided flow --
        # i.e. a genuinely PREDICTIVE relationship between trailing flow
        # and FORWARD returns, not just a same-instant price bump (which
        # trailing-flow-vs-forward-return correlation has no reason to
        # pick up at all, since it's already priced into price(t) itself).
        drift = 0.97 * drift + 0.1 * signed  # ~2.3s half-life at ~100ms/trade -- persists across a ~1s horizon
        price += 0.002 * drift + rng.gauss(0, 0.01)
        rows.append(
            {
                "symbol": "BTCUSDT",
                "agg_trade_id": i,
                "trade_time_ns": t_ns,
                "recv_wall_ns": t_ns,
                "recv_mono_ns": t_ns,
                "price": price,
                "quantity": qty,
                "is_buyer_maker": is_buyer_maker,
                "first_trade_id": i,
                "last_trade_id": i,
            }
        )
    return pl.DataFrame(rows)


def test_trade_flow_decay_matrix_empty_input():
    empty = pl.DataFrame(
        {
            "trade_time_ns": pl.Series([], dtype=pl.Int64),
            "price": pl.Series([], dtype=pl.Float64),
            "signed_quantity": pl.Series([], dtype=pl.Float64),
        }
    )
    assert ofi.trade_flow_decay_matrix(empty) == []


def test_trade_flow_decay_matrix_shape_and_grid():
    trades = ofi.annotate_trades(_synthetic_trades())
    windows = [500.0, 5_000.0]
    horizons = [500.0, 5_000.0]
    cells = ofi.trade_flow_decay_matrix(trades, windows_ms=windows, horizons_ms=horizons)

    assert len(cells) == len(windows) * len(horizons)
    seen = {(c.window_ms, c.horizon_ms) for c in cells}
    assert seen == {(w, h) for w in windows for h in horizons}
    for c in cells:
        assert c.n_obs > 0
        assert not (c.correlation != c.correlation)  # not NaN, given n_obs > 0 here


def test_trade_flow_decay_matrix_detects_the_planted_signal():
    # window/horizon chosen to sit within the synthetic drift's ~2.3s
    # persistence half-life (see _synthetic_trades) -- at 5000ms the drift
    # has already decayed to noise by the time the forward window starts,
    # which is itself a real property of decay measurement (a horizon far
    # past the signal's persistence looks like no edge), not a reason to
    # weaken this assertion.
    trades = ofi.annotate_trades(_synthetic_trades())
    cells = ofi.trade_flow_decay_matrix(trades, windows_ms=[1_000.0], horizons_ms=[1_000.0])
    assert len(cells) == 1
    # weak but real planted signal (trailing flow drives a persistent
    # drift) -- should show up as a small POSITIVE correlation, not noise
    # scattered around 0
    assert cells[0].correlation > 0.0


def test_trade_flow_decay_matrix_output_works_with_overlay_and_sanity_check_unchanged():
    """Reuse overlay_cost_hurdle/check_hurdle_sanity as-is on
    trade_flow_decay_matrix's output -- both operate on Sequence[DecayCell]
    generically and require no changes for this new function."""
    trades = ofi.annotate_trades(_synthetic_trades())
    cells = ofi.trade_flow_decay_matrix(trades, windows_ms=[500.0, 5_000.0], horizons_ms=[500.0, 5_000.0])
    sched = CryptoPerpChargeSchedule()

    overlaid = ofi.overlay_cost_hurdle(cells, sched)
    assert len(overlaid) == len(cells)
    assert all(isinstance(c, ofi.HurdleCell) for c in overlaid)
    assert all(c.round_trip_cost_bps == pytest.approx(10.0) for c in overlaid)

    ofi.check_hurdle_sanity(overlaid)  # must not raise
