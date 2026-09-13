"""Bar-level backtest engine.

Every design choice here exists to prevent a backtest from silently
overstating performance (see the quant-backtest-discipline non-negotiables):

  1. Signal at bar t, fill at bar t+1 — enforced STRUCTURALLY. The strategy
     callback is only ever handed `bars[: t + 1]`; it physically cannot see
     the bar its fill executes against, so a same-bar-close fill (the most
     common source of fake alpha) is not a bug a strategy author can
     introduce even by accident.
  2. Fills are pessimistic by default: buys cross to the ask (or open, with
     adverse slippage, if no ask is in the bar), sells cross to the bid.
     There is no maker/passive-fill path here at all — with bar-level data
     you cannot know your queue position, so "the limit order got filled at
     the touch" is not a supportable assumption. If you have book data
     precise enough to justify a passive fill, that belongs in a different,
     explicitly-labeled fill model, not a silent option on this one.
  3. Costs are itemized per order via costs.order_charges() and reported
     separately from spread/slippage cost, which is itself reported
     separately from gross (frictionless) P&L. See BacktestResult.
  4. slippage_bps has no default — see EngineConfig.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from costs import ChargeBreakdown, ChargeScheduleT, LiquidityRole, Side, order_charges


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar. bid/ask are the best quotes observed at/near the bar's
    close, when available (e.g. from a quote feed aligned to the same bar
    grid) — used for pessimistic fills on the *next* bar. When absent, fills
    fall back to that next bar's open with slippage applied against the
    trader; see _fill_price."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None


#: A strategy receives ONLY the bars observed so far (through bar t,
#: inclusive) and returns a target position to be filled at bar t+1. It
#: never receives bar t+1 itself. Position units are caller-defined (e.g.
#: contracts, shares, or a -1..+1 fraction of `EngineConfig.lot_size`); the
#: engine only cares about the delta between successive targets.
StrategyFn = Callable[[Sequence[Bar]], float]


@dataclass(frozen=True)
class EngineConfig:
    """Engine configuration.

    Field order matters: `slippage_bps` has NO default, deliberately. A
    silent slippage default (whether zero or some remembered "reasonable"
    number) becomes an inherited guess nobody chose for this run. If you
    want frictionless fills for a sensitivity check, pass slippage_bps=0.0
    and say so in the report — that is a stated choice, not a default.
    """

    schedule: ChargeScheduleT
    slippage_bps: float
    initial_capital: float
    liquidity: LiquidityRole = LiquidityRole.TAKER
    lot_size: float = 1.0

    def __post_init__(self) -> None:
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0 (it is applied against the trader)")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")


@dataclass(frozen=True)
class Fill:
    timestamp: datetime
    side: Side
    price: float
    reference_price: float  # bar.open at the fill bar — the frictionless baseline
    quantity: float
    charges: ChargeBreakdown

    @property
    def spread_slippage_cost(self) -> float:
        """Cost attributable to crossing the spread + slippage, i.e. the gap
        between the pessimistic fill price and the frictionless reference
        price. Reported separately from `charges` so a backtest showing
        unexpected drag can distinguish "the spread ate it" from "the
        exchange fees ate it" — they demand different fixes."""
        return abs(self.price - self.reference_price) * self.quantity


@dataclass(frozen=True)
class BacktestResult:
    bars: Sequence[Bar]
    fills: list[Fill]
    equity_curve: np.ndarray  # length len(bars); equity_curve[0] == initial_capital
    initial_capital: float

    @property
    def net_pnl(self) -> float:
        return float(self.equity_curve[-1] - self.initial_capital)

    @property
    def total_charges(self) -> float:
        return sum(f.charges.total for f in self.fills)

    @property
    def total_spread_slippage(self) -> float:
        return sum(f.spread_slippage_cost for f in self.fills)

    @property
    def total_friction(self) -> float:
        return self.total_charges + self.total_spread_slippage

    @property
    def gross_pnl(self) -> float:
        """P&L with charges and spread/slippage added back — what the
        strategy would have made in a frictionless world. Comparing this to
        net_pnl is what separates "no gross edge, abandon the idea" from
        "edge exists but frequency/costs are wrong, extend holding period
        and retest" (quant-backtest-discipline rule #4). Never report only
        net_pnl."""
        return self.net_pnl + self.total_friction

    @property
    def trade_count(self) -> int:
        return len(self.fills)


def _fill_price(bar: Bar, direction: int, slippage_bps: float) -> float:
    if direction > 0:
        base = bar.ask if bar.ask is not None else bar.open
        return base * (1.0 + slippage_bps / 10_000.0)
    elif direction < 0:
        base = bar.bid if bar.bid is not None else bar.open
        return base * (1.0 - slippage_bps / 10_000.0)
    raise ValueError("direction must be nonzero")


def run_backtest(bars: Sequence[Bar], strategy_fn: StrategyFn, config: EngineConfig) -> BacktestResult:
    """Run one backtest. See module docstring for the fill/timing model.

    `bars` must be in ascending timestamp order; this is the caller's
    responsibility (the engine does not re-sort, since silently re-ordering
    input the caller believes is already time-ordered can mask a data bug
    rather than surface it).
    """
    if len(bars) < 2:
        raise ValueError("need at least 2 bars: signal computed at t, filled at t+1")

    position = 0.0
    cash = config.initial_capital
    equity = np.empty(len(bars))
    equity[0] = config.initial_capital
    fills: list[Fill] = []

    for t in range(len(bars) - 1):
        # Strategy sees bars[0 : t+1] ONLY (i.e. through bar t). Bar t+1,
        # the one the resulting target will be filled against, is not
        # passed — this is what makes signal-at-t/fill-at-t+1 structural
        # rather than conventional.
        target = strategy_fn(bars[: t + 1])
        fill_bar = bars[t + 1]
        delta = target - position

        if delta != 0.0:
            direction = 1 if delta > 0 else -1
            price = _fill_price(fill_bar, direction, config.slippage_bps)
            qty = abs(delta) * config.lot_size
            side = Side.BUY if direction > 0 else Side.SELL
            charges = order_charges(
                config.schedule,
                side=side,
                price=price,
                quantity=qty,
                liquidity=config.liquidity,
            )
            cash -= direction * price * qty
            cash -= charges.total
            fills.append(
                Fill(
                    timestamp=fill_bar.timestamp,
                    side=side,
                    price=price,
                    reference_price=fill_bar.open,
                    quantity=qty,
                    charges=charges,
                )
            )
            position = target

        equity[t + 1] = cash + position * fill_bar.close * config.lot_size

    return BacktestResult(bars=bars, fills=fills, equity_curve=equity, initial_capital=config.initial_capital)


# --------------------------------------------------------------------------
# Walk-forward splitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardFold:
    """Half-open index ranges into the bar sequence. train_start is always 0
    (expanding window): fold i's train set is everything before its test
    window, never data that comes after it, and test windows are
    sequential and non-overlapping — a single train/test split hides regime
    dependence, so results MUST be reported per fold, not only aggregated.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def walk_forward_folds(n_bars: int, *, n_folds: int, min_train_frac: float = 0.3) -> list[WalkForwardFold]:
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0.0 < min_train_frac < 1.0):
        raise ValueError("min_train_frac must be in (0, 1)")

    min_train = int(n_bars * min_train_frac)
    remaining = n_bars - min_train
    if remaining < n_folds:
        raise ValueError(
            f"not enough bars ({n_bars}) for {n_folds} folds at min_train_frac={min_train_frac}"
        )

    test_size = remaining // n_folds
    folds: list[WalkForwardFold] = []
    train_end = min_train
    for i in range(n_folds):
        test_start = train_end
        test_end = n_bars if i == n_folds - 1 else test_start + test_size
        folds.append(WalkForwardFold(train_start=0, train_end=train_end, test_start=test_start, test_end=test_end))
        train_end = test_end
    return folds


def run_walk_forward(
    bars: Sequence[Bar],
    strategy_factory: Callable[[Sequence[Bar]], StrategyFn],
    config: EngineConfig,
    *,
    n_folds: int,
    min_train_frac: float = 0.3,
) -> list[BacktestResult]:
    """`strategy_factory(train_bars)` is called once per fold to produce a
    StrategyFn (e.g. fit any parameters on train_bars), which is then run
    against that fold's held-out test_bars only. Returns one BacktestResult
    per fold, in fold order — report each one, not just their sum. Folds
    alternating between strongly positive and strongly negative results are
    the signature of noise, and that pattern is invisible in an aggregate.
    """
    folds = walk_forward_folds(len(bars), n_folds=n_folds, min_train_frac=min_train_frac)
    results = []
    for fold in folds:
        train_bars = bars[fold.train_start : fold.train_end]
        test_bars = bars[fold.test_start : fold.test_end]
        strategy_fn = strategy_factory(train_bars)
        results.append(run_backtest(test_bars, strategy_fn, config))
    return results


# --------------------------------------------------------------------------
# Random-signal control
# --------------------------------------------------------------------------


def _make_random_strategy(
    position_values: Sequence[float], trade_frequency: float, rng: random.Random
) -> StrategyFn:
    # Mutable closure state: a random strategy still HOLDS a position between
    # bars like a real one would, it just decides direction randomly at the
    # same frequency as the real signal. Re-rolling a fresh random target
    # every single bar would not match a real strategy's turnover profile
    # and would make the control comparison meaningless.
    state = {"target": 0.0}

    def strategy_fn(history: Sequence[Bar]) -> float:
        if rng.random() < trade_frequency:
            state["target"] = rng.choice(position_values)
        return state["target"]

    return strategy_fn


def random_signal_control(
    bars: Sequence[Bar],
    config: EngineConfig,
    *,
    n_trials: int,
    trade_frequency: float,
    position_values: Sequence[float] = (-1.0, 0.0, 1.0),
    seed: int | None = None,
) -> np.ndarray:
    """Run `n_trials` random-entry backtests at the given trade_frequency
    (fraction of bars on which the random strategy reconsiders its target —
    match this to the real strategy's actual turnover) and direction mix,
    returning an array of net P&L outcomes.

    A real strategy's result should be compared against this distribution
    (see `percentile_rank`). Landing inside the cloud means nothing was
    found, regardless of how the equity curve looks; this is necessary, not
    sufficient, validation (quant-backtest-discipline).
    """
    if not (0.0 < trade_frequency <= 1.0):
        raise ValueError("trade_frequency must be in (0, 1]")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    master_rng = random.Random(seed)
    outcomes = np.empty(n_trials)
    for i in range(n_trials):
        trial_rng = random.Random(master_rng.randrange(2**32))
        strategy_fn = _make_random_strategy(position_values, trade_frequency, trial_rng)
        outcomes[i] = run_backtest(bars, strategy_fn, config).net_pnl
    return outcomes


def percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Fraction of the control distribution at or below `value`. A claimed
    edge should sit far in the tail; a value near 0.5 means the real result
    is statistically indistinguishable from random entries at the same
    frequency."""
    return float(np.mean(distribution <= value))


# --------------------------------------------------------------------------
# Performance summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceSummary:
    bar_count: int
    total_return: float
    cagr: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    trade_count: int
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    gross_pnl: float
    total_friction: float
    total_charges: float
    total_spread_slippage: float
    net_pnl: float


def summarize(result: BacktestResult, *, bars_per_year: float, risk_free_rate: float = 0.0) -> PerformanceSummary:
    """Compute the standard reporting block (quant-backtest-discipline
    "Reporting format"). `bars_per_year` must be supplied explicitly — it is
    the annualization basis (e.g. 525_600 for 1-minute crypto bars trading
    24/7, ~93_750 for 1-minute NSE bars) and silently assuming one when the
    data is the other misstates every annualized figure by ~5-6x.
    """
    equity = result.equity_curve
    n = len(equity)
    total_return = float(equity[-1] / equity[0] - 1.0)
    years = (n - 1) / bars_per_year if bars_per_year > 0 else float("nan")
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    rets = np.diff(equity) / equity[:-1]
    ann_factor = math.sqrt(bars_per_year)
    annualized_vol = float(np.std(rets, ddof=1) * ann_factor) if len(rets) > 1 else 0.0

    excess = rets - risk_free_rate / bars_per_year
    sharpe = float(np.mean(excess) / np.std(rets, ddof=1) * ann_factor) if len(rets) > 1 and np.std(rets, ddof=1) > 0 else 0.0

    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
    sortino = float(np.mean(excess) / downside_std * ann_factor) if downside_std > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(drawdown.min())

    trade_pnls = _per_trade_pnls(result)
    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    hit_rate = float(len(wins) / len(trade_pnls)) if trade_pnls else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    return PerformanceSummary(
        bar_count=n,
        total_return=total_return,
        cagr=cagr,
        annualized_vol=annualized_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        trade_count=result.trade_count,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        gross_pnl=result.gross_pnl,
        total_friction=result.total_friction,
        total_charges=result.total_charges,
        total_spread_slippage=result.total_spread_slippage,
        net_pnl=result.net_pnl,
    )


def _per_trade_pnls(result: BacktestResult) -> list[float]:
    """Realized P&L per round-trip (a closing fill that flattens or reverses
    the running position), net of that closing fill's own charges. Used only
    for hit-rate/profit-factor reporting, not for net_pnl (which comes from
    the equity curve directly and is authoritative)."""
    pnls: list[float] = []
    position = 0.0
    avg_price = 0.0
    for f in result.fills:
        signed_qty = f.quantity if f.side is Side.BUY else -f.quantity
        new_position = position + signed_qty
        if position == 0.0 or (signed_qty > 0) == (position > 0):
            # opening or adding to a position
            total_cost = avg_price * abs(position) + f.price * abs(signed_qty)
            avg_price = total_cost / abs(new_position) if new_position != 0 else 0.0
        else:
            closing_qty = min(abs(signed_qty), abs(position))
            direction = 1 if position > 0 else -1
            pnl = (f.price - avg_price) * closing_qty * direction - f.charges.total
            pnls.append(pnl)
        position = new_position
    return pnls
