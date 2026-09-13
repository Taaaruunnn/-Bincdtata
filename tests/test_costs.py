"""Cost model tests with hand-computed expected values (quant-backtest-
discipline: "Cost model tests with hand-computed expected values.")."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

import costs
from costs import (
    NSE_EQUITY_DELIVERY,
    NSE_EQUITY_FUTURES,
    NSE_EQUITY_INTRADAY,
    CryptoPerpChargeSchedule,
    FundingEvent,
    IndiaEquityChargeSchedule,
    LiquidityRole,
    Segment,
    Side,
    breakeven_move,
    cost_hurdle_table,
    funding_timestamps,
    holding_cost_funding,
    order_charges,
)


# --------------------------------------------------------------------------
# Crypto perp
# --------------------------------------------------------------------------


def test_crypto_taker_default():
    sched = CryptoPerpChargeSchedule()  # maker=2bps, taker=5bps
    bd = order_charges(sched, side=Side.BUY, price=50_000.0, quantity=0.1)  # liquidity omitted -> TAKER
    notional = 50_000.0 * 0.1
    expected_fee = notional * (5.0 / 10_000.0)
    assert bd.total == pytest.approx(expected_fee)
    assert bd.components == {"exchange_fee_taker": pytest.approx(expected_fee)}


def test_crypto_maker_must_be_explicit():
    sched = CryptoPerpChargeSchedule()
    bd = order_charges(sched, side=Side.BUY, price=50_000.0, quantity=0.1, liquidity=LiquidityRole.MAKER)
    notional = 50_000.0 * 0.1
    expected_fee = notional * (2.0 / 10_000.0)
    assert bd.total == pytest.approx(expected_fee)
    assert bd.total < order_charges(sched, side=Side.BUY, price=50_000.0, quantity=0.1, liquidity=LiquidityRole.TAKER).total


def test_crypto_custom_tier():
    sched = CryptoPerpChargeSchedule(maker_bps=0.0, taker_bps=1.7, tier_name="VIP9")
    bd = order_charges(sched, side=Side.SELL, price=1000.0, quantity=2.0)
    assert bd.total == pytest.approx(2000.0 * 1.7 / 10_000.0)


def test_crypto_rejects_india_only_params():
    sched = CryptoPerpChargeSchedule()
    with pytest.raises(ValueError):
        order_charges(sched, side=Side.BUY, price=100.0, quantity=1.0, option_exercised=True, intrinsic_value_per_unit=5.0)


def test_crypto_schedule_rejects_wrong_segment():
    with pytest.raises(ValueError):
        CryptoPerpChargeSchedule(segment=Segment.EQUITY_DELIVERY)


# --------------------------------------------------------------------------
# India equity delivery
# --------------------------------------------------------------------------


def test_india_delivery_buy_hand_computed():
    notional = 100.0 * 100  # price=100, qty=100
    bd = order_charges(NSE_EQUITY_DELIVERY, side=Side.BUY, price=100.0, quantity=100.0)

    stt = 0.001 * notional  # 10.0
    exch = 0.0000307 * notional  # 0.307
    sebi = 0.000001 * notional  # 0.01
    ipft = 0.000001 * notional  # 0.01
    stamp = 0.00015 * notional  # 1.5
    gst = 0.18 * (0.0 + exch + sebi)  # brokerage is 0 for delivery

    assert bd.components["stt"] == pytest.approx(stt)
    assert bd.components["exchange_transaction_charge"] == pytest.approx(exch)
    assert bd.components["sebi_turnover_fee"] == pytest.approx(sebi)
    assert bd.components["ipft_charge"] == pytest.approx(ipft)
    assert bd.components["stamp_duty"] == pytest.approx(stamp)
    assert bd.components["gst"] == pytest.approx(gst)
    assert "brokerage" not in bd.components  # zero brokerage is cleanly absent, not a zero entry

    expected_total = stt + exch + sebi + ipft + stamp + gst
    assert bd.total == pytest.approx(expected_total)
    assert expected_total == pytest.approx(11.88406, abs=1e-5)


def test_india_delivery_sell_has_no_stamp_duty():
    bd = order_charges(NSE_EQUITY_DELIVERY, side=Side.SELL, price=100.0, quantity=100.0)
    assert "stamp_duty" not in bd.components
    notional = 10_000.0
    expected_total = (
        0.001 * notional  # stt
        + 0.0000307 * notional  # exchange txn
        + 0.000001 * notional  # sebi
        + 0.000001 * notional  # ipft
        + 0.18 * (0.0000307 * notional + 0.000001 * notional)  # gst on exch+sebi (brokerage=0)
    )
    assert bd.total == pytest.approx(expected_total)


# --------------------------------------------------------------------------
# India equity intraday (brokerage rate + cap)
# --------------------------------------------------------------------------


def test_india_intraday_brokerage_percentage_below_cap():
    notional = 100.0 * 100  # 10,000 -> 0.03% = 3.0, below the 20 cap
    bd = order_charges(NSE_EQUITY_INTRADAY, side=Side.SELL, price=100.0, quantity=100.0)
    assert bd.components["brokerage"] == pytest.approx(0.0003 * notional)
    assert bd.components["stt"] == pytest.approx(0.00025 * notional)


def test_india_intraday_brokerage_capped():
    notional = 1000.0 * 1000  # 1,000,000 -> 0.03% = 300, capped at 20
    bd = order_charges(NSE_EQUITY_INTRADAY, side=Side.BUY, price=1000.0, quantity=1000.0)
    assert bd.components["brokerage"] == pytest.approx(20.0)
    assert "stt" not in bd.components  # buy side STT is zero -> cleanly absent, not a zero entry


def test_india_intraday_buy_has_no_stt_but_has_stamp_duty():
    bd = order_charges(NSE_EQUITY_INTRADAY, side=Side.BUY, price=100.0, quantity=100.0)
    assert "stt" not in bd.components
    assert bd.components["stamp_duty"] == pytest.approx(0.00003 * 10_000.0)


# --------------------------------------------------------------------------
# India equity futures (post Budget-2026 STT hike)
# --------------------------------------------------------------------------


def test_india_futures_stt_only_on_sell():
    notional = 100.0 * 100
    buy = order_charges(NSE_EQUITY_FUTURES, side=Side.BUY, price=100.0, quantity=100.0)
    sell = order_charges(NSE_EQUITY_FUTURES, side=Side.SELL, price=100.0, quantity=100.0)
    assert "stt" not in buy.components
    assert sell.components["stt"] == pytest.approx(0.0005 * notional)  # 0.05% post-hike rate


# --------------------------------------------------------------------------
# Options exercise STT (distinct base: intrinsic value, not premium)
# --------------------------------------------------------------------------


def test_options_exercise_stt_uses_intrinsic_value_not_premium():
    bd = order_charges(
        costs.NSE_EQUITY_OPTIONS,
        side=Side.BUY,
        price=5.0,  # premium
        quantity=50.0,
        option_exercised=True,
        intrinsic_value_per_unit=20.0,  # deep ITM, intrinsic >> premium
    )
    expected_exercise_stt = 0.0015 * 20.0 * 50.0
    assert bd.components["stt_on_exercise"] == pytest.approx(expected_exercise_stt)


def test_options_exercise_requires_intrinsic_value():
    with pytest.raises(ValueError):
        order_charges(costs.NSE_EQUITY_OPTIONS, side=Side.BUY, price=5.0, quantity=50.0, option_exercised=True)


def test_options_exercise_rejected_for_non_options_segment():
    with pytest.raises(ValueError):
        order_charges(
            NSE_EQUITY_FUTURES, side=Side.BUY, price=100.0, quantity=1.0, option_exercised=True, intrinsic_value_per_unit=1.0
        )


# --------------------------------------------------------------------------
# Schedule construction guards
# --------------------------------------------------------------------------


def test_india_schedule_rejects_crypto_segment():
    with pytest.raises(ValueError):
        IndiaEquityChargeSchedule(
            segment=Segment.CRYPTO_PERP,
            brokerage_rate=0.0,
            brokerage_cap=0.0,
            stt_rate_buy=0.0,
            stt_rate_sell=0.0,
            exchange_txn_rate=0.0,
            sebi_rate=0.0,
            ipft_rate=0.0,
            stamp_duty_rate_buy=0.0,
        )


def test_order_charges_rejects_unsupported_type():
    with pytest.raises(TypeError):
        order_charges(object(), side=Side.BUY, price=1.0, quantity=1.0)  # type: ignore[arg-type]


def test_order_charges_rejects_nonpositive_price_or_quantity():
    sched = CryptoPerpChargeSchedule()
    with pytest.raises(ValueError):
        order_charges(sched, side=Side.BUY, price=0.0, quantity=1.0)
    with pytest.raises(ValueError):
        order_charges(sched, side=Side.BUY, price=1.0, quantity=-1.0)


# --------------------------------------------------------------------------
# Funding (crypto holding cost)
# --------------------------------------------------------------------------


def test_funding_timestamps_8h_boundaries():
    start = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc)
    ts = funding_timestamps(start, end, interval_hours=8.0)
    assert ts == [
        datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
    ]


def test_funding_timestamps_requires_tz_aware():
    with pytest.raises(ValueError):
        funding_timestamps(datetime(2026, 1, 1), datetime(2026, 1, 2), interval_hours=8.0)


def test_funding_timestamps_4h_interval_symbol():
    # Some USDS-M perps settle every 4h, not 8h -- see costs.py module
    # docstring. interval_hours must be supplied explicitly per symbol.
    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
    ts = funding_timestamps(start, end, interval_hours=4.0)
    assert ts == [
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
    ]


def test_holding_cost_funding_hand_computed():
    ts1 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
    events = [FundingEvent(ts1, 0.0001), FundingEvent(ts2, -0.00005)]
    total = holding_cost_funding(lambda t: 10_000.0, events)
    assert total == pytest.approx(10_000.0 * 0.0001 + 10_000.0 * -0.00005)
    assert total == pytest.approx(0.5)


def test_holding_cost_funding_zero_when_flat():
    ts1 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    events = [FundingEvent(ts1, 0.0001)]
    total = holding_cost_funding(lambda t: 0.0, events)
    assert total == 0.0


# --------------------------------------------------------------------------
# breakeven_move / cost_hurdle_table
# --------------------------------------------------------------------------


def test_breakeven_move_crypto_round_trip():
    sched = CryptoPerpChargeSchedule()  # taker 5bps each side
    move = breakeven_move(sched, price=100.0, quantity=1.0)
    # entry fee = 100*1*0.0005 = 0.05; exit fee = 0.05; total 0.10; /qty(1) = 0.10
    assert move == pytest.approx(0.10)


def test_breakeven_move_single_leg():
    sched = CryptoPerpChargeSchedule()
    move = breakeven_move(sched, price=100.0, quantity=1.0, round_trip=False)
    assert move == pytest.approx(0.05)


def test_cost_hurdle_table_hand_computed():
    sched = CryptoPerpChargeSchedule()
    rt_cost = breakeven_move(sched, price=100.0, quantity=1.0)  # 0.10
    rows = cost_hurdle_table(sched, price=100.0, quantity=1.0, annualized_vol=0.5, horizons_minutes=(1.0, 60.0))
    minutes_per_year = 365 * 24 * 60
    expected_sigma_1 = 100.0 * 0.5 * math.sqrt(1.0 / minutes_per_year)
    expected_sigma_60 = 100.0 * 0.5 * math.sqrt(60.0 / minutes_per_year)

    assert rows[0].horizon_minutes == 1.0
    assert rows[0].one_sigma_move == pytest.approx(expected_sigma_1)
    assert rows[0].round_trip_cost == pytest.approx(rt_cost)
    assert rows[0].cost_over_sigma == pytest.approx(rt_cost / expected_sigma_1)

    assert rows[1].one_sigma_move == pytest.approx(expected_sigma_60)
    assert rows[1].cost_over_sigma == pytest.approx(rt_cost / expected_sigma_60)
    # a 1-minute horizon should be a much worse hurdle multiple than 60 minutes
    assert rows[0].cost_over_sigma > rows[1].cost_over_sigma


def test_cost_hurdle_table_rejects_nonpositive_vol():
    sched = CryptoPerpChargeSchedule()
    with pytest.raises(ValueError):
        cost_hurdle_table(sched, price=100.0, quantity=1.0, annualized_vol=0.0)
