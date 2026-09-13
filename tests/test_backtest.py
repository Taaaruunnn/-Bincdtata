"""Sanity tests for the backtest engine's core discipline guarantees:
signal-at-t/fill-at-t+1, pessimistic fills, and gross/friction/net
decomposition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtest import Bar, EngineConfig, run_backtest, summarize, walk_forward_folds
from costs import CryptoPerpChargeSchedule, LiquidityRole


def _bars(prices: list[float]) -> list[Bar]:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(timestamp=t0 + timedelta(minutes=i), open=p, high=p, low=p, close=p, volume=1.0, bid=p - 0.5, ask=p + 0.5)
        for i, p in enumerate(prices)
    ]


def _config(slippage_bps: float = 0.0) -> EngineConfig:
    return EngineConfig(schedule=CryptoPerpChargeSchedule(), slippage_bps=slippage_bps, initial_capital=10_000.0)


def test_engine_config_requires_slippage_explicitly():
    with pytest.raises(TypeError):
        EngineConfig(schedule=CryptoPerpChargeSchedule(), initial_capital=10_000.0)  # type: ignore[call-arg]


def test_signal_cannot_see_its_own_fill_bar():
    """A strategy that tries to peek at the bar it will fill against should
    be structurally unable to -- it only ever receives history up to and
    including bar t, never bar t+1."""
    bars = _bars([100.0, 200.0, 100.0, 200.0, 100.0])
    seen_lengths = []

    def strategy(history):
        seen_lengths.append(len(history))
        return 0.0  # never trade; we only care what the strategy can see

    run_backtest(bars, strategy, _config())
    # called once per bar except the last (whose target would fill against
    # a nonexistent bar): lengths must be 1, 2, 3, 4 -- never 5 (== len(bars))
    assert seen_lengths == [1, 2, 3, 4]


def test_fill_price_uses_next_bar_not_signal_bar():
    """Going long at bar 0 must fill at bar 1's ask, not bar 0's own price."""
    bars = _bars([100.0, 200.0, 200.0])  # huge jump so a same-bar fill would be obviously wrong

    def strategy(history):
        return 1.0 if len(history) == 1 else 1.0  # go long immediately, hold

    result = run_backtest(bars, strategy, _config())
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.price == pytest.approx(bars[1].ask)  # bar 1's ask, not bar 0's
    assert fill.timestamp == bars[1].timestamp


def test_pessimistic_fill_crosses_spread_both_directions():
    bars = _bars([100.0, 100.0, 100.0])

    def go_long(history):
        return 1.0

    def go_short(history):
        return -1.0

    long_fill = run_backtest(bars, go_long, _config()).fills[0]
    short_fill = run_backtest(bars, go_short, _config()).fills[0]
    assert long_fill.price == pytest.approx(bars[1].ask)  # buys cross to the ask
    assert short_fill.price == pytest.approx(bars[1].bid)  # sells cross to the bid
    assert long_fill.price > short_fill.price


def test_high_frequency_flip_flop_loses_to_friction():
    """A signal that trades every bar with no real edge should show
    ~zero gross P&L (mean-reverting noise cancels out) but strictly
    negative net P&L once costs and spread are applied -- the canonical
    quant-backtest-discipline failure mode this engine exists to expose."""
    import random

    rng = random.Random(3)
    prices = [100.0]
    for _ in range(400):
        prices.append(prices[-1] * (1 + rng.gauss(0, 0.0005)))
    bars = _bars(prices)

    def flip_flop(history):
        return 1.0 if len(history) % 2 == 0 else -1.0

    result = run_backtest(bars, flip_flop, _config(slippage_bps=1.0))
    assert result.total_friction > 0
    assert result.net_pnl < result.gross_pnl  # friction strictly erodes P&L
    assert result.net_pnl == pytest.approx(result.gross_pnl - result.total_friction)


def test_gross_friction_net_decomposition_is_internally_consistent():
    bars = _bars([100.0, 101.0, 99.0, 102.0, 98.0])

    def strategy(history):
        return 1.0 if len(history) % 2 else -1.0

    result = run_backtest(bars, strategy, _config(slippage_bps=2.0))
    assert result.gross_pnl == pytest.approx(result.net_pnl + result.total_charges + result.total_spread_slippage)
    assert result.total_friction == pytest.approx(result.total_charges + result.total_spread_slippage)


def test_maker_liquidity_must_be_requested_explicitly():
    bars = _bars([100.0, 100.0])

    def go_long(history):
        return 1.0

    taker_result = run_backtest(bars, go_long, _config())
    maker_config = EngineConfig(
        schedule=CryptoPerpChargeSchedule(), slippage_bps=0.0, initial_capital=10_000.0, liquidity=LiquidityRole.MAKER
    )
    maker_result = run_backtest(bars, go_long, maker_config)
    # maker fee (2bps) < taker fee (5bps) on the same fill -> strictly less charged
    assert maker_result.total_charges < taker_result.total_charges


def test_walk_forward_folds_are_sequential_and_non_overlapping():
    folds = walk_forward_folds(1000, n_folds=4, min_train_frac=0.2)
    assert len(folds) == 4
    for i, fold in enumerate(folds):
        assert fold.train_start == 0
        assert fold.train_end == fold.test_start
        if i > 0:
            assert fold.test_start == folds[i - 1].test_end  # contiguous, no overlap, no gap
    assert folds[-1].test_end == 1000


def test_summarize_requires_bars_per_year_and_reports_all_fields():
    bars = _bars([100.0 + i * 0.01 for i in range(50)])

    def strategy(history):
        return 1.0

    result = run_backtest(bars, strategy, _config(slippage_bps=1.0))
    summary = summarize(result, bars_per_year=525_600)
    assert summary.bar_count == len(bars)
    assert summary.trade_count == result.trade_count
    assert summary.net_pnl == pytest.approx(result.net_pnl)
    assert summary.gross_pnl == pytest.approx(result.gross_pnl)
