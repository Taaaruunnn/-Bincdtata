"""Per-order transaction cost model.

Design principle (see quant-backtest-discipline rule #3): every charge is
itemized and returned as a named breakdown, never collapsed into a single
blended number before the caller has seen the components. A blended bps
figure hides the fact that flat fees (brokerage, SEBI/IPFT per-crore charges)
behave nothing like percentages at small trade size, and that different
charges apply to different bases (premium vs. notional).

RATE SOURCES AND VERIFICATION STATUS
-------------------------------------
Every rate below is a named constant in exactly one place (the two
ChargeSchedule dataclasses). RATE_SCHEDULE_VERSION is bumped whenever any
rate changes, so a backtest run under one schedule is never silently compared
against a run under another.

NSE equity/F&O rates — retrieved 2026-09-13, SECONDARY sources (broker fee
pages, not the CBDT notification or NSE circular text directly). Cross-checked
across two independent pages and found consistent; verify against your own
contract note before relying on these for live P&L:
  - https://zerodha.com/charges (STT, exchange transaction charges, stamp
    duty, GST, brokerage — reflects the Union Budget 2026 STT hike on
    equity F&O effective 2026-04-01: futures STT 0.02% -> 0.05% sell side,
    options premium STT 0.10% -> 0.15% sell side)
  - IPFT (Investor Protection Fund Trust) equity-cash rate, Rs 10/crore
    (0.0001%), effective 2023-04-01 per NSE circular, cross-referenced via
    Groww/Shoonya broker FAQ pages, retrieved 2026-09-13. F&O IPFT is applied
    here at the same per-crore rate as equity cash; this has NOT been
    independently confirmed against an NSE F&O-specific circular and should
    be checked if IPFT precision matters for your use case.

Binance USDS-M Futures crypto rates — retrieved 2026-09-13:
  - https://www.binance.com/en/support/faq/detail/360033544231 — Binance's
    own fee-rate FAQ (page reports last update 2026-05-01). VIP0 (Regular
    User): maker 0.0200% (2 bps), taker 0.0500% (5 bps). PRIMARY source
    (exchange's own published page).
  - Funding interval is NOT universally 8h. Binance moved a subset of
    USDS-M perpetuals to 4h settlement starting 2023-10-12; the interval is
    symbol-specific. Read it per symbol from GET /fapi/v1/fundingInfo
    (`fundingIntervalHours` field) rather than assuming a constant. The
    default of 8h below MUST be overridden per symbol for correctness.
    https://www.binance.com/en/support/announcement/important-updates-on-funding-rates-of-usd%E2%93%A2-m-perpetual-contracts-98d6b24d3e5c4f84a8ed04087997d8d0
  - IP request-weight limit (relevant to funding-rate/snapshot polling, not
    to this module directly, but recorded here since it was verified in the
    same research pass): 2400/minute/IP, confirmed live from
    GET /fapi/v1/exchangeInfo -> rateLimits, retrieved 2026-09-13.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto

RATE_SCHEDULE_VERSION = "2026-09-13"


class Segment(Enum):
    """Instrument/venue segment. Each segment has its own ChargeSchedule type
    (IndiaEquityChargeSchedule or CryptoPerpChargeSchedule) — there is no
    shared flat schedule with fields zeroed out per segment, because a zero
    that should be an absence (e.g. "STT" on a crypto perp) is easy to misread
    as a verified zero rather than "does not apply here"."""

    EQUITY_DELIVERY = auto()
    EQUITY_INTRADAY = auto()
    EQUITY_FUTURES = auto()
    EQUITY_OPTIONS = auto()
    CRYPTO_PERP = auto()


INDIA_SEGMENTS = frozenset(
    {
        Segment.EQUITY_DELIVERY,
        Segment.EQUITY_INTRADAY,
        Segment.EQUITY_FUTURES,
        Segment.EQUITY_OPTIONS,
    }
)


class Side(Enum):
    BUY = auto()
    SELL = auto()


class LiquidityRole(Enum):
    """Whether an order added liquidity (maker) or removed it (taker).

    order_charges() defaults every crypto order to TAKER. Assuming maker
    fills is the single most common way a crypto backtest manufactures fake
    alpha: without full order-book queue-position data you cannot know
    whether a passive order actually filled, so a caller must ask for MAKER
    pricing explicitly and be able to justify it (see
    quant-backtest-discipline rule #2)."""

    MAKER = auto()
    TAKER = auto()


@dataclass(frozen=True)
class ChargeBreakdown:
    """Named, itemized charges for a single order. `total` is a derived
    property, not a stored field, so the itemization is always the source of
    truth — a caller who only reads `.total` has to actively discard the
    breakdown rather than never having seen it."""

    segment: Segment
    side: Side
    notional: float
    components: Mapping[str, float]

    @property
    def total(self) -> float:
        return sum(self.components.values())


# --------------------------------------------------------------------------
# India equity / F&O
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndiaEquityChargeSchedule:
    """Charge schedule for NSE equity cash and equity F&O segments.

    Rates are percentages expressed as fractions (0.0003 == 0.03%) applied to
    a segment-specific base: notional (price * quantity) for delivery,
    intraday and futures; premium (price * quantity, where `price` is the
    option premium) for options. See module docstring for sourcing.
    """

    segment: Segment
    brokerage_rate: float
    brokerage_cap: float  # flat ceiling per order, in currency units (0.0 = unlimited)
    stt_rate_buy: float
    stt_rate_sell: float
    exchange_txn_rate: float  # both sides
    sebi_rate: float  # both sides, per-crore-of-turnover fraction
    ipft_rate: float  # both sides, per-crore-of-turnover fraction
    stamp_duty_rate_buy: float  # buy side only, per Indian Stamp Act practice
    gst_rate: float = 0.18
    version: str = RATE_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.segment not in INDIA_SEGMENTS:
            raise ValueError(
                f"IndiaEquityChargeSchedule requires an India segment, got {self.segment}"
            )


def _capped_brokerage(schedule: IndiaEquityChargeSchedule, notional: float) -> float:
    pct = schedule.brokerage_rate * notional
    if schedule.brokerage_cap <= 0.0:
        return pct
    return min(pct, schedule.brokerage_cap)


#: Rates as sourced 2026-09-13 — see module docstring for provenance and
#: caveats. Each is a standalone named constant so a rate revision is a
#: one-line edit rather than a hunt through arithmetic.
NSE_EQUITY_DELIVERY = IndiaEquityChargeSchedule(
    segment=Segment.EQUITY_DELIVERY,
    brokerage_rate=0.0,
    brokerage_cap=0.0,
    stt_rate_buy=0.001,
    stt_rate_sell=0.001,
    exchange_txn_rate=0.0000307,
    sebi_rate=0.000001,
    ipft_rate=0.000001,
    stamp_duty_rate_buy=0.00015,
)

NSE_EQUITY_INTRADAY = IndiaEquityChargeSchedule(
    segment=Segment.EQUITY_INTRADAY,
    brokerage_rate=0.0003,
    brokerage_cap=20.0,
    stt_rate_buy=0.0,
    stt_rate_sell=0.00025,
    exchange_txn_rate=0.0000307,
    sebi_rate=0.000001,
    ipft_rate=0.000001,
    stamp_duty_rate_buy=0.00003,
)

NSE_EQUITY_FUTURES = IndiaEquityChargeSchedule(
    segment=Segment.EQUITY_FUTURES,
    brokerage_rate=0.0003,
    brokerage_cap=20.0,
    stt_rate_buy=0.0,
    stt_rate_sell=0.0005,  # post Budget-2026 hike, effective 2026-04-01
    exchange_txn_rate=0.0000183,
    sebi_rate=0.000001,
    ipft_rate=0.000001,
    stamp_duty_rate_buy=0.00002,
)

NSE_EQUITY_OPTIONS = IndiaEquityChargeSchedule(
    segment=Segment.EQUITY_OPTIONS,
    brokerage_rate=0.0,  # flat per-order fee below, not a rate on premium
    brokerage_cap=20.0,
    stt_rate_buy=0.0,
    stt_rate_sell=0.0015,  # on premium, sell side; post Budget-2026 hike
    exchange_txn_rate=0.0003553,  # on premium
    sebi_rate=0.000001,  # on premium
    ipft_rate=0.000001,  # on premium
    stamp_duty_rate_buy=0.00003,  # on premium, buy side
)

#: NSE options STT on exercised contracts is charged to the BUYER on
#: intrinsic value, not premium — distinct base and distinct rate trigger
#: from the sell-side premium STT above. Applied only when the caller passes
#: option_exercised=True and an intrinsic_value_per_unit.
NSE_OPTIONS_EXERCISE_STT_RATE = 0.0015


# --------------------------------------------------------------------------
# Crypto perpetuals (Binance USDS-M futures)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CryptoPerpChargeSchedule:
    """Charge schedule for USDS-M perpetual futures. Flat maker/taker bps on
    notional. Deliberately has NO stt/stamp_duty/gst/sebi/ipft fields: those
    are India-specific charges with no crypto-perp equivalent, and giving
    this dataclass a zeroed field for each would let a caller read
    `schedule.stt_rate` and get a *verified* zero instead of an AttributeError
    that immediately says "wrong schedule type for this instrument."
    """

    segment: Segment = Segment.CRYPTO_PERP
    maker_bps: float = 2.0  # Binance VIP0 USDS-M, see module docstring
    taker_bps: float = 5.0
    tier_name: str = "VIP0"
    funding_interval_hours: float = 8.0  # SYMBOL-SPECIFIC — see module docstring
    version: str = RATE_SCHEDULE_VERSION

    def __post_init__(self) -> None:
        if self.segment is not Segment.CRYPTO_PERP:
            raise ValueError("CryptoPerpChargeSchedule.segment must be Segment.CRYPTO_PERP")


# --------------------------------------------------------------------------
# order_charges
# --------------------------------------------------------------------------

ChargeScheduleT = IndiaEquityChargeSchedule | CryptoPerpChargeSchedule


def order_charges(
    schedule: ChargeScheduleT,
    *,
    side: Side,
    price: float,
    quantity: float,
    liquidity: LiquidityRole = LiquidityRole.TAKER,
    option_exercised: bool = False,
    intrinsic_value_per_unit: float | None = None,
) -> ChargeBreakdown:
    """Compute the itemized charge breakdown for one executed order.

    `liquidity` defaults to TAKER. This matters for crypto (maker vs. taker
    fee tiers) and is a required, explicit choice rather than an inferred one
    for the same reason fills default to crossing the spread in backtest.py:
    assuming a fill added liquidity is an assumption about queue position
    that most market data cannot support.
    """
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if price <= 0:
        raise ValueError("price must be positive")

    if isinstance(schedule, CryptoPerpChargeSchedule):
        if option_exercised or intrinsic_value_per_unit is not None:
            raise ValueError("option_exercised/intrinsic_value_per_unit are not meaningful for CRYPTO_PERP")
        return _crypto_order_charges(schedule, side=side, price=price, quantity=quantity, liquidity=liquidity)

    if isinstance(schedule, IndiaEquityChargeSchedule):
        return _india_order_charges(
            schedule,
            side=side,
            price=price,
            quantity=quantity,
            option_exercised=option_exercised,
            intrinsic_value_per_unit=intrinsic_value_per_unit,
        )

    raise TypeError(f"Unsupported charge schedule type: {type(schedule)!r}")


def _crypto_order_charges(
    schedule: CryptoPerpChargeSchedule,
    *,
    side: Side,
    price: float,
    quantity: float,
    liquidity: LiquidityRole,
) -> ChargeBreakdown:
    notional = price * quantity
    rate_bps = schedule.maker_bps if liquidity is LiquidityRole.MAKER else schedule.taker_bps
    fee = notional * (rate_bps / 10_000.0)
    components = {
        f"exchange_fee_{'maker' if liquidity is LiquidityRole.MAKER else 'taker'}": fee,
    }
    return ChargeBreakdown(segment=schedule.segment, side=side, notional=notional, components=components)


def _india_order_charges(
    schedule: IndiaEquityChargeSchedule,
    *,
    side: Side,
    price: float,
    quantity: float,
    option_exercised: bool,
    intrinsic_value_per_unit: float | None,
) -> ChargeBreakdown:
    if option_exercised and schedule.segment is not Segment.EQUITY_OPTIONS:
        raise ValueError("option_exercised is only meaningful for EQUITY_OPTIONS")
    if option_exercised and intrinsic_value_per_unit is None:
        raise ValueError("option_exercised=True requires intrinsic_value_per_unit")

    notional = price * quantity  # `price` is the option premium for EQUITY_OPTIONS
    components: dict[str, float] = {}

    brokerage = _capped_brokerage(schedule, notional)
    if brokerage:
        components["brokerage"] = brokerage

    stt_rate = schedule.stt_rate_buy if side is Side.BUY else schedule.stt_rate_sell
    stt = stt_rate * notional
    if stt:
        components["stt"] = stt

    if option_exercised:
        assert intrinsic_value_per_unit is not None
        components["stt_on_exercise"] = NSE_OPTIONS_EXERCISE_STT_RATE * intrinsic_value_per_unit * quantity

    exchange_txn = schedule.exchange_txn_rate * notional
    components["exchange_transaction_charge"] = exchange_txn

    sebi = schedule.sebi_rate * notional
    components["sebi_turnover_fee"] = sebi

    ipft = schedule.ipft_rate * notional
    components["ipft_charge"] = ipft

    if side is Side.BUY:
        stamp_duty = schedule.stamp_duty_rate_buy * notional
        if stamp_duty:
            components["stamp_duty"] = stamp_duty

    # GST applies to brokerage + SEBI + exchange transaction charge, NOT to
    # STT or stamp duty (both are themselves taxes/levies, not services).
    gst_base = brokerage + exchange_txn + sebi
    gst = schedule.gst_rate * gst_base
    if gst:
        components["gst"] = gst

    return ChargeBreakdown(segment=schedule.segment, side=side, notional=notional, components=components)


# --------------------------------------------------------------------------
# Funding (crypto perp holding cost — NOT part of order_charges)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FundingEvent:
    """One observed funding settlement. `funding_rate` must come from
    exchange data (GET /fapi/v1/fundingRate or the aggTrade-adjacent funding
    stream), never assumed constant — funding rate is market data, not a
    schedule parameter."""

    timestamp: datetime
    funding_rate: float  # signed: positive means longs pay shorts


def funding_timestamps(
    start: datetime,
    end: datetime,
    *,
    interval_hours: float,
) -> list[datetime]:
    """Funding settlement instants in [start, end), anchored to 00:00 UTC.

    `interval_hours` is symbol-specific on Binance (see costs.py module
    docstring) and must be supplied explicitly — there is no default here,
    because silently assuming 8h for a 4h-interval symbol misprices funding
    by roughly 2x.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end must be timezone-aware (use UTC)")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")

    step = timedelta(hours=interval_hours)
    epoch = start.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    n = math.floor((start - epoch) / step)
    ts = epoch + n * step
    out: list[datetime] = []
    while ts < end:
        if ts >= start:
            out.append(ts)
        ts += step
    return out


def holding_cost_funding(
    position_notional_at: Callable[[datetime], float],
    funding_events: Iterable[FundingEvent],
) -> float:
    """Total funding paid (positive) or received (negative) across the given
    funding events, for a position whose signed notional at each event
    timestamp is given by `position_notional_at` (positive = long).

    Kept entirely separate from order_charges(): funding accrues to an OPEN
    position independent of whether any order was placed at that instant, so
    folding it into per-trade cost would misattribute a holding cost to a
    trade and corrupt the cost-hurdle-per-trade calculation in
    cost_hurdle_table().
    """
    total = 0.0
    for ev in funding_events:
        notional = position_notional_at(ev.timestamp)
        if notional != 0.0:
            total += notional * ev.funding_rate
    return total


# --------------------------------------------------------------------------
# Cost hurdle
# --------------------------------------------------------------------------


def breakeven_move(
    schedule: ChargeScheduleT,
    *,
    price: float,
    quantity: float,
    liquidity: LiquidityRole = LiquidityRole.TAKER,
    round_trip: bool = True,
) -> float:
    """Minimum favorable price move (in price units, not bps) needed to
    cover order charges. Excludes funding (a separate holding cost — see
    holding_cost_funding()) since funding depends on holding duration, not
    on the entry/exit orders themselves.
    """
    entry = order_charges(schedule, side=Side.BUY, price=price, quantity=quantity, liquidity=liquidity)
    total_cost = entry.total
    if round_trip:
        exit_ = order_charges(schedule, side=Side.SELL, price=price, quantity=quantity, liquidity=liquidity)
        total_cost += exit_.total
    return total_cost / quantity


#: Trading-time annualization basis, minutes per year. NSE cash/F&O trades
#: ~6h15m/day (375 min) across ~250 sessions/year; crypto perps trade
#: continuously. Used only to convert an annualized vol into a horizon-sigma
#: — has no effect on any actually-executed cost, only on how the hurdle
#: table is displayed, but is still made explicit per segment rather than
#: silently assumed, since a wrong denominator misstates the sigma by 5-6x
#: between these two segments.
_ANNUALIZATION_MINUTES: dict[Segment, float] = {
    Segment.EQUITY_DELIVERY: 375 * 250,
    Segment.EQUITY_INTRADAY: 375 * 250,
    Segment.EQUITY_FUTURES: 375 * 250,
    Segment.EQUITY_OPTIONS: 375 * 250,
    Segment.CRYPTO_PERP: 365 * 24 * 60,
}


@dataclass(frozen=True)
class HurdleRow:
    """One row of the cost-hurdle-vs-horizon table. `cost_over_sigma` is the
    number that decides viable holding period: values >> 1 mean the typical
    move at that horizon does not cover round-trip cost, so a strategy
    trading at that horizon needs an edge many multiples of a random move
    just to break even before it is even net profitable."""

    horizon_minutes: float
    one_sigma_move: float
    round_trip_cost: float
    cost_over_sigma: float


def cost_hurdle_table(
    schedule: ChargeScheduleT,
    *,
    price: float,
    quantity: float,
    annualized_vol: float,
    horizons_minutes: Sequence[float] = (1.0, 5.0, 15.0, 60.0),
    liquidity: LiquidityRole = LiquidityRole.TAKER,
    trading_minutes_per_year: float | None = None,
) -> list[HurdleRow]:
    """For a given annualized vol, show the 1-sigma price move against
    round-trip order cost at each horizon, and cost as a multiple of sigma.

    This is computed BEFORE any strategy work: if round-trip cost exceeds a
    few sigma at every horizon you would plausibly hold, no signal quality
    can rescue the strategy at that holding period, and there is no reason to
    backtest it.
    """
    if annualized_vol <= 0:
        raise ValueError("annualized_vol must be positive")

    minutes_per_year = trading_minutes_per_year or _ANNUALIZATION_MINUTES[schedule.segment]
    rt_cost = breakeven_move(schedule, price=price, quantity=quantity, liquidity=liquidity, round_trip=True)

    rows = []
    for h in horizons_minutes:
        sigma_h = price * annualized_vol * math.sqrt(h / minutes_per_year)
        rows.append(
            HurdleRow(
                horizon_minutes=h,
                one_sigma_move=sigma_h,
                round_trip_cost=rt_cost,
                cost_over_sigma=(rt_cost / sigma_h) if sigma_h > 0 else math.inf,
            )
        )
    return rows
