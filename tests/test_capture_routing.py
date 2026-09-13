"""Tests for capture.py's routed-category split (public/market) -- see its
module docstring's "LEGACY URL RETIREMENT AND ROUTING" section for why this
exists: a single unrouted legacy connection was silently dropping aggTrade
(market-category) data while depth (public-category) kept flowing with no
error at all."""

from __future__ import annotations

from pathlib import Path

import pytest

import capture


def test_categorize_channel_depth_variants_are_public():
    assert capture._categorize_channel("depth@100ms") == "public"
    assert capture._categorize_channel("depth") == "public"
    assert capture._categorize_channel("depth@500ms") == "public"


def test_categorize_channel_aggtrade_is_market():
    assert capture._categorize_channel("aggTrade") == "market"


def test_categorize_channel_rejects_unmapped_channel():
    with pytest.raises(ValueError):
        capture._categorize_channel("markPrice")


def test_stream_names_by_category_splits_correctly():
    cfg = capture.CaptureConfig(symbols=["BTCUSDT", "ETHUSDT"], out_dir=Path("x"))
    grouped = cfg.stream_names_by_category
    assert grouped == {
        "public": ["btcusdt@depth@100ms", "ethusdt@depth@100ms"],
        "market": ["btcusdt@aggTrade", "ethusdt@aggTrade"],
    }


def test_ws_url_for_routes_to_the_correct_base_url():
    cfg = capture.CaptureConfig(symbols=["BTCUSDT"], out_dir=Path("x"))
    assert cfg.ws_url_for("public") == "wss://fstream.binance.com/public/stream?streams=btcusdt@depth@100ms"
    assert cfg.ws_url_for("market") == "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade"


def test_depth_only_config_has_no_market_category():
    cfg = capture.CaptureConfig(symbols=["BTCUSDT"], out_dir=Path("x"), channels=("depth@100ms",))
    assert cfg.stream_names_by_category == {"public": ["btcusdt@depth@100ms"]}
    assert "market" not in cfg.stream_names_by_category


def test_max_streams_per_connection_is_checked_per_category_not_combined():
    # 600 symbols x 1 depth channel = 600 streams on "public" alone, under
    # the 1024 cap -- must NOT be rejected just because adding a second
    # (aggTrade) channel would push the OLD combined-connection total over
    # 1024; each category is its own connection now.
    symbols = [f"SYM{i}USDT" for i in range(600)]
    cfg = capture.CaptureConfig(symbols=symbols, out_dir=Path("x"), channels=("depth@100ms", "aggTrade"))
    assert len(cfg.stream_names_by_category["public"]) == 600
    assert len(cfg.stream_names_by_category["market"]) == 600


def test_max_streams_per_connection_still_rejects_a_single_category_over_the_limit():
    symbols = [f"SYM{i}USDT" for i in range(1025)]
    with pytest.raises(ValueError, match="1024-stream-per-connection"):
        capture.CaptureConfig(symbols=symbols, out_dir=Path("x"), channels=("depth@100ms",))
