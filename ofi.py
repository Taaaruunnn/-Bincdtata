"""Order Flow Imbalance (OFI) and its decay against forward returns.

OFI is computed exactly as defined in:
    R. Cont, A. Kukanov, S. Stoikov, "The Price Impact of Order Book
    Events," Journal of Financial Econometrics, 12(1), 2014.
    (also SSRN working paper 1712062)

using ONLY best-bid/best-ask price and size changes between consecutive
VALID top-of-book snapshots from reconstruct.py's book_state.parquet — see
ofi_events(). This module never uses tick-rule/Lee-Ready inference for trade
side: wherever a trade's aggressor side matters (aggressor_side()), it comes
directly from Binance aggTrade's `m` field, which is ground truth (see the
market-data-integrity skill's "Aggressor side: use the flag, not
inference").

VALIDITY: only rows with valid=True (as written by reconstruct.py) are used.
OFI is never computed across a reconstruction gap: the event immediately
following a resync is deliberately given a null OFI (see the segment-id
logic in ofi_events()), because comparing its best bid/ask to whatever the
book showed BEFORE the gap would measure "the book got resynced," not real
order flow. See market-data-integrity: "Analysis code filters on [the
validity flag] explicitly."

LOOK-AHEAD WARNING: `overlay_cost_hurdle` expects MOST (w, h) cells to show
predicted edge below round-trip cost. If it doesn't, check decay_matrix()
and its use of `strategy="forward"` in the as-of join for forward returns —
a `strategy="backward"` typo there would leak the future return into the
"current" OFI window and manufacture edge that isn't real.

`trade_flow_decay_matrix()` is the trade-flow analog of `decay_matrix()`:
same grid, same DecayCell output (so `overlay_cost_hurdle`/
`check_hurdle_sanity` work on either unchanged), but computed from signed
trade volume (`annotate_trades()`) and trade-price returns rather than
book OFI and mid-price returns. It is a cross-check, not a substitute — see
its own docstring for the two concrete ways it differs from `decay_matrix()`
(trade-price bid-ask-bounce noise, and no gap/validity awareness at all
since trades.parquet doesn't carry one).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from costs import CryptoPerpChargeSchedule, LiquidityRole, Side, order_charges

LOG = logging.getLogger("ofi")

#: Default window/horizon grid requested for the decay matrix, in
#: milliseconds. {100ms, 500ms, 1s, 5s, 30s, 1m, 5m, 15m}.
DEFAULT_GRID_MS: tuple[float, ...] = (100.0, 500.0, 1_000.0, 5_000.0, 30_000.0, 60_000.0, 300_000.0, 900_000.0)


# --------------------------------------------------------------------------
# Trade aggressor classification (ground truth, not inferred)
# --------------------------------------------------------------------------


def aggressor_side(is_buyer_maker: bool) -> Side:
    """Binance aggTrade's `m` field ("is the buyer the market maker?") is
    True when the BUYER posted the resting order, which means the SELLER
    crossed the spread and was the aggressor on that trade -- and vice
    versa. This is ground truth from the exchange; never derive trade side
    via tick-rule/Lee-Ready inference when this flag is available (both
    misclassify a meaningful fraction of trades, for no reason, when a
    field already tells you)."""
    return Side.SELL if is_buyer_maker else Side.BUY


def annotate_trades(trades_df: pl.DataFrame) -> pl.DataFrame:
    """Adds `aggressor_side` ("BUY"/"SELL") and `signed_quantity` (+qty if
    the aggressor bought, -qty if sold) columns, both derived solely from
    the `is_buyer_maker` flag. Useful as a cross-check against book-derived
    OFI, or as an input to a trade-flow-imbalance variant, without ever
    falling back to tick-rule inference."""
    return trades_df.with_columns(
        pl.when(pl.col("is_buyer_maker")).then(pl.lit("SELL")).otherwise(pl.lit("BUY")).alias("aggressor_side"),
        pl.when(pl.col("is_buyer_maker")).then(-pl.col("quantity")).otherwise(pl.col("quantity")).alias("signed_quantity"),
    )


# --------------------------------------------------------------------------
# OFI event series from reconstructed book state
# --------------------------------------------------------------------------


def ofi_events(book_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Build the per-event OFI series for one symbol from book_state rows.

    Returns columns: event_time_ns, mid_price, ofi_event (nullable — null
    at the first event of each valid segment, i.e. right after a resync,
    where there is no valid predecessor to compare against).

    Cont-Kukanov-Stoikov event-level OFI, for consecutive top-of-book
    observations (P^b, q^b) = best bid price/size and (P^a, q^a) = best ask
    price/size, comparing observation n to n-1:

        e^bid_n =  q^b_n            if P^b_n >  P^b_{n-1}
                =  q^b_n - q^b_{n-1} if P^b_n == P^b_{n-1}
                = -q^b_{n-1}         if P^b_n <  P^b_{n-1}

        e^ask_n = -q^a_{n-1}         if P^a_n >  P^a_{n-1}
                =  q^a_n - q^a_{n-1} if P^a_n == P^a_{n-1}
                =  q^a_n             if P^a_n <  P^a_{n-1}

        OFI_n = e^bid_n - e^ask_n
    """
    # IMPORTANT: segment_id is computed over ALL rows for this symbol,
    # including "gap" rows (valid=False), BEFORE filtering down to the
    # valid update/snapshot rows used for OFI itself. Computing it after
    # filtering would silently discard the very valid=False rows the
    # boundary detection depends on, making every gap invisible to the
    # shift-based segment logic below.
    df = book_df.filter(pl.col("symbol") == symbol).sort("event_time_ns")
    df = df.with_columns(
        (pl.col("valid") & ~pl.col("valid").shift(1, fill_value=False)).cum_sum().alias("_segment_id")
    )

    valid = df.filter(
        pl.col("valid")
        & pl.col("event_type").is_in(["snapshot", "update"])
        & (pl.col("bid_prices").list.len() > 0)
        & (pl.col("ask_prices").list.len() > 0)
    )

    valid = valid.with_columns(
        pl.col("bid_prices").list.get(0).alias("best_bid_price"),
        pl.col("bid_qtys").list.get(0).alias("best_bid_qty"),
        pl.col("ask_prices").list.get(0).alias("best_ask_price"),
        pl.col("ask_qtys").list.get(0).alias("best_ask_qty"),
    )

    valid = valid.with_columns(
        pl.col("best_bid_price").shift(1).over("_segment_id").alias("_prev_bid_price"),
        pl.col("best_bid_qty").shift(1).over("_segment_id").alias("_prev_bid_qty"),
        pl.col("best_ask_price").shift(1).over("_segment_id").alias("_prev_ask_price"),
        pl.col("best_ask_qty").shift(1).over("_segment_id").alias("_prev_ask_qty"),
    )

    e_bid = (
        pl.when(pl.col("best_bid_price") > pl.col("_prev_bid_price"))
        .then(pl.col("best_bid_qty"))
        .when(pl.col("best_bid_price") == pl.col("_prev_bid_price"))
        .then(pl.col("best_bid_qty") - pl.col("_prev_bid_qty"))
        .otherwise(-pl.col("_prev_bid_qty"))
    )
    e_ask = (
        pl.when(pl.col("best_ask_price") > pl.col("_prev_ask_price"))
        .then(-pl.col("_prev_ask_qty"))
        .when(pl.col("best_ask_price") == pl.col("_prev_ask_price"))
        .then(pl.col("best_ask_qty") - pl.col("_prev_ask_qty"))
        .otherwise(pl.col("best_ask_qty"))
    )

    valid = valid.with_columns(
        ((pl.col("best_bid_price") + pl.col("best_ask_price")) / 2.0).alias("mid_price"),
        (e_bid - e_ask).alias("ofi_event"),
    )

    return valid.select(["event_time_ns", "mid_price", "ofi_event"])


# --------------------------------------------------------------------------
# Decay matrix: corr(OFI over window w, forward return over horizon h)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayCell:
    window_ms: float
    horizon_ms: float
    correlation: float
    n_obs: int
    predicted_edge_bps: float  # correlation * std(forward_return) * 1e4 -- see decay_matrix docstring


def decay_matrix(
    events: pl.DataFrame,
    *,
    windows_ms: Sequence[float] = DEFAULT_GRID_MS,
    horizons_ms: Sequence[float] = DEFAULT_GRID_MS,
) -> list[DecayCell]:
    """For every (window, horizon) pair, compute:
      - OFI_w(t): trailing sum of ofi_event over the w-millisecond window
        ending at each event time t (time-based rolling sum, NOT
        event-count based, since events are irregularly spaced).
      - forward_return_h(t): (mid_price at the first event at/after t+h,
        minus mid_price(t)) / mid_price(t) -- found via a forward as-of
        join. This is deliberately a FORWARD-looking join
        (strategy="forward"): it is the one place in this module where
        getting the direction backwards would silently manufacture
        look-ahead edge. See module docstring.
      - correlation(OFI_w(t), forward_return_h(t)) across all valid t.
      - predicted_edge_bps: the return move (in bps) associated with a
        1-standard-deviation OFI shock, i.e. correlation * std(return) *
        1e4 -- the OLS-implied predicted y at x = std(x), which is the
        natural "how big an edge does this correlation actually imply"
        number to compare against a cost hurdle in the SAME units.
    """
    if events.is_empty():
        return []

    base = events.drop_nulls(["ofi_event"]).with_columns(
        pl.from_epoch(pl.col("event_time_ns"), time_unit="ns").alias("ts")
    ).sort("ts").with_columns(pl.arange(0, pl.len()).alias("_idx"))

    cells: list[DecayCell] = []
    for w_ms in windows_ms:
        window_str = f"{int(w_ms)}ms"
        with_ofi_w = base.with_columns(
            pl.col("ofi_event").rolling_sum_by("ts", window_size=window_str).alias("ofi_w")
        )
        future = with_ofi_w.select(["ts", "mid_price"]).rename({"ts": "future_ts", "mid_price": "future_mid"}).sort("future_ts")

        for h_ms in horizons_ms:
            # Horizon arithmetic done in integer nanoseconds, THEN cast to
            # datetime[ns] -- pl.duration() defaults to microsecond
            # resolution, which mismatches the ns-resolution `ts`/`future_ts`
            # columns and makes join_asof refuse to join (SchemaError) rather
            # than silently truncating precision.
            target = with_ofi_w.with_columns(
                pl.from_epoch(pl.col("event_time_ns") + int(h_ms * 1_000_000), time_unit="ns").alias("target_ts")
            ).sort("target_ts")

            joined = target.join_asof(future, left_on="target_ts", right_on="future_ts", strategy="forward").sort("_idx")

            fwd_return = (joined["future_mid"] - joined["mid_price"]) / joined["mid_price"]
            pair = pl.DataFrame({"ofi_w": joined["ofi_w"], "fwd_return": fwd_return}).drop_nulls()

            if pair.height < 3:
                cells.append(DecayCell(w_ms, h_ms, float("nan"), pair.height, float("nan")))
                continue

            corr = pair.select(pl.corr("ofi_w", "fwd_return")).item()
            ret_std = float(pair["fwd_return"].std())
            predicted_edge_bps = (corr or 0.0) * ret_std * 10_000.0
            cells.append(DecayCell(w_ms, h_ms, float(corr) if corr is not None else float("nan"), pair.height, predicted_edge_bps))

    return cells


def trade_flow_decay_matrix(
    trades: pl.DataFrame,
    *,
    windows_ms: Sequence[float] = DEFAULT_GRID_MS,
    horizons_ms: Sequence[float] = DEFAULT_GRID_MS,
) -> list[DecayCell]:
    """Trade-flow analog of decay_matrix(): corr(signed trade volume over
    window w, forward return over horizon h), for the same grid. Same
    structure and same output type (DecayCell), so overlay_cost_hurdle()
    and check_hurdle_sanity() work on its output unchanged.

    `trades` must already carry `signed_quantity` -- i.e. the caller has
    already run it through annotate_trades() -- exactly the same "build
    the event series first, then measure decay" split decay_matrix() uses
    with ofi_events(). Trailing volume is a time-based rolling sum
    (rolling_sum_by), not event-count based, for the same reason as
    decay_matrix(): trades are irregularly spaced.

    TWO REAL DIFFERENCES FROM decay_matrix(), not implementation
    shortcuts:
      1. Forward returns here are computed from TRADE PRICES, not book
         mid-price. A trade alternates between hitting the bid and the
         ask depending on aggressor side, so a trade-price return carries
         bid-ask-bounce noise that a mid-price return does not. Treat this
         matrix as a cross-check against decay_matrix(), not a like-for-
         like replacement -- a real signal should show up in both, and a
         signal that only shows up here is a reason to suspect bounce
         noise before believing it.
      2. There is no validity/segment concept here the way ofi_events()
         nulls OFI at a reconstruction gap boundary. trades.parquet
         currently has no gap-awareness column at all (see README's Known
         Limitations) -- a "market"-category connection gap during a
         window is invisible to this function. Check
         `ReconstructionStats.non_book_gaps_detected` for the capture that
         produced `trades` before trusting a window that might have
         missed one.

    Same forward-only join direction as decay_matrix() -- see that
    function's docstring for the look-ahead warning. Getting this
    direction backwards is the one place this function could silently
    manufacture edge that isn't real.
    """
    if trades.is_empty():
        return []

    base = (
        trades.drop_nulls(["signed_quantity"])
        .with_columns(pl.from_epoch(pl.col("trade_time_ns"), time_unit="ns").alias("ts"))
        .sort("ts")
        .with_columns(pl.arange(0, pl.len()).alias("_idx"))
    )

    cells: list[DecayCell] = []
    for w_ms in windows_ms:
        window_str = f"{int(w_ms)}ms"
        with_flow_w = base.with_columns(
            pl.col("signed_quantity").rolling_sum_by("ts", window_size=window_str).alias("flow_w")
        )
        future = with_flow_w.select(["ts", "price"]).rename({"ts": "future_ts", "price": "future_price"}).sort("future_ts")

        for h_ms in horizons_ms:
            # Same reasoning as decay_matrix(): horizon arithmetic in
            # integer nanoseconds THEN cast to datetime[ns], since
            # pl.duration() defaults to microsecond resolution and
            # mismatches the ns-resolution `ts`/`future_ts` columns.
            target = with_flow_w.with_columns(
                pl.from_epoch(pl.col("trade_time_ns") + int(h_ms * 1_000_000), time_unit="ns").alias("target_ts")
            ).sort("target_ts")

            joined = target.join_asof(future, left_on="target_ts", right_on="future_ts", strategy="forward").sort("_idx")

            fwd_return = (joined["future_price"] - joined["price"]) / joined["price"]
            pair = pl.DataFrame({"flow_w": joined["flow_w"], "fwd_return": fwd_return}).drop_nulls()

            if pair.height < 3:
                cells.append(DecayCell(w_ms, h_ms, float("nan"), pair.height, float("nan")))
                continue

            corr = pair.select(pl.corr("flow_w", "fwd_return")).item()
            ret_std = float(pair["fwd_return"].std())
            predicted_edge_bps = (corr or 0.0) * ret_std * 10_000.0
            cells.append(DecayCell(w_ms, h_ms, float(corr) if corr is not None else float("nan"), pair.height, predicted_edge_bps))

    return cells


# --------------------------------------------------------------------------
# Cost hurdle overlay
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HurdleCell(DecayCell):
    round_trip_cost_bps: float
    exceeds_cost: bool


def round_trip_taker_cost_bps(schedule: CryptoPerpChargeSchedule) -> float:
    """Round-trip cost in bps of notional, at TAKER pricing (the default
    fill assumption throughout this codebase). Independent of price/qty for
    a pure-bps crypto fee schedule -- computed via order_charges() on a
    nominal $100 x 1 order rather than duplicating the bps arithmetic, so
    this stays correct if CryptoPerpChargeSchedule ever grows a
    non-proportional component."""
    price, qty = 100.0, 1.0
    entry = order_charges(schedule, side=Side.BUY, price=price, quantity=qty, liquidity=LiquidityRole.TAKER)
    exit_ = order_charges(schedule, side=Side.SELL, price=price, quantity=qty, liquidity=LiquidityRole.TAKER)
    notional = price * qty
    return (entry.total + exit_.total) / notional * 10_000.0


def overlay_cost_hurdle(cells: Sequence[DecayCell], schedule: CryptoPerpChargeSchedule) -> list[HurdleCell]:
    """Mark which (window, horizon) cells have |predicted_edge_bps|
    exceeding round-trip taker cost. Per quant-backtest-discipline, MOST
    cells should fail (exceeds_cost=False): if none fail, that is a
    reason to suspect look-ahead in decay_matrix(), not a reason to
    celebrate -- see check_hurdle_sanity().
    """
    cost_bps = round_trip_taker_cost_bps(schedule)
    out = []
    for c in cells:
        exceeds = (not np.isnan(c.predicted_edge_bps)) and abs(c.predicted_edge_bps) > cost_bps
        out.append(
            HurdleCell(
                window_ms=c.window_ms,
                horizon_ms=c.horizon_ms,
                correlation=c.correlation,
                n_obs=c.n_obs,
                predicted_edge_bps=c.predicted_edge_bps,
                round_trip_cost_bps=cost_bps,
                exceeds_cost=exceeds,
            )
        )
    return out


def check_hurdle_sanity(cells: Sequence[HurdleCell]) -> None:
    """Logs a warning if the overlay found zero passing cells (expected —
    most edges don't survive cost) OR if it found that EVERY cell passes
    (suspicious — see module docstring's look-ahead warning). Does not
    raise: this is a signal for a human to check decay_matrix(), not an
    error the program should refuse to report."""
    valid_cells = [c for c in cells if not np.isnan(c.predicted_edge_bps)]
    if not valid_cells:
        LOG.warning("no valid decay cells to sanity-check (insufficient data everywhere)")
        return
    passing = sum(1 for c in valid_cells if c.exceeds_cost)
    if passing == len(valid_cells):
        LOG.warning(
            "ALL %d (window, horizon) cells show predicted edge exceeding round-trip cost. "
            "This is NOT the expected result -- re-check decay_matrix()'s forward-return join "
            "for look-ahead before trusting this.",
            len(valid_cells),
        )
    else:
        LOG.info("%d/%d cells exceed round-trip cost (expected: most should not)", passing, len(valid_cells))


def decay_matrix_to_frame(cells: Sequence[HurdleCell]) -> pl.DataFrame:
    """Convenience: cells -> a tidy Polars DataFrame for display/export."""
    return pl.DataFrame(
        {
            "window_ms": [c.window_ms for c in cells],
            "horizon_ms": [c.horizon_ms for c in cells],
            "correlation": [c.correlation for c in cells],
            "n_obs": [c.n_obs for c in cells],
            "predicted_edge_bps": [c.predicted_edge_bps for c in cells],
            "round_trip_cost_bps": [c.round_trip_cost_bps for c in cells],
            "exceeds_cost": [c.exceeds_cost for c in cells],
        }
    )
