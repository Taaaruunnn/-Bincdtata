# Fintech backtest + market-data infrastructure

Two parts, built to different rules:

1. **`costs.py` / `backtest.py`** — a bar-level backtest harness (NSE equity/F&O
   and Binance USDS-M crypto perps) whose whole design exists to make a
   backtest capable of returning "no." See **Backtest discipline** below.
2. **`capture.py` / `reconstruct.py` / `ofi.py`** — read-only Binance USDS-M
   futures market-data infrastructure: capture raw WebSocket/REST bytes,
   reconstruct a validated local order book offline, and measure order-flow
   imbalance and its decay against cost. See **Market data pipeline** below.

**Non-goals, deliberately not built here:** no order placement, no API keys,
no trading logic, no live signal path from the reconstructor to anything
executable. This is measurement infrastructure and a backtest harness, not a
trading system.

## Setup

```
pip install -r requirements.txt
python -m pytest tests/ -q
```

Python 3.11+ (uses `X | Y` union type hints throughout). All tick/event data
uses `polars`, not `pandas` — row counts are expected to reach tens of
millions for a few days of `@100ms` depth-diff capture on a liquid symbol.

---

## Backtest discipline (costs.py, backtest.py)

- **Signal at bar t, fill at bar t+1 is structural, not conventional.**
  `backtest.StrategyFn` only ever receives `bars[: t+1]` — bar `t+1`, the one
  the resulting order fills against, is never passed to the strategy. A
  same-bar-close fill (the most common source of fake alpha) is not a bug a
  strategy author can introduce even deliberately.
- **Fills are pessimistic by default and there is no passive-fill path.**
  Buys cross to the ask, sells cross to the bid (falling back to the next
  bar's open, adjusted against the trader, when no quote is available). This
  engine has no maker/limit-fill model at all: bar-level data cannot tell you
  your queue position, so "the limit order got filled at the touch" is not a
  supportable assumption here.
- **`EngineConfig.slippage_bps` has no default.** Passing it is mandatory —
  see the class docstring. A silent slippage default is an inherited guess
  nobody chose for a given run.
- **Costs are itemized per order** via `costs.order_charges()`, which returns
  a named `ChargeBreakdown`, never a blended total. `BacktestResult` reports
  `gross_pnl`, `total_charges`, `total_spread_slippage`, and `net_pnl`
  separately — never only `net_pnl`. "No gross edge" (abandon the idea) and
  "edge eaten by friction" (extend the holding period, retest) are different
  diagnoses and must never be collapsed into one number.
- **`Segment.CRYPTO_PERP` has no India-specific fields at all** —
  `CryptoPerpChargeSchedule` simply has no `stt_rate`/`stamp_duty_rate`/etc.
  attributes, so a caller reading `schedule.stt_rate` on a crypto schedule
  gets `AttributeError`, not a verified-looking zero. `order_charges()`
  dispatches on the schedule's type, and a maker/taker mismatch (e.g. passing
  `option_exercised` to a crypto schedule) raises rather than being ignored.
- **`liquidity` defaults to `TAKER`.** A caller wanting maker pricing must
  ask for it explicitly — assuming maker fills is the single most common way
  a crypto backtest manufactures fake alpha.
- **Funding is a separate holding cost** (`holding_cost_funding`,
  `funding_timestamps`), not folded into `order_charges()`. It accrues to an
  open position independent of whether any order was placed, and Binance's
  funding interval is **symbol-specific, not universally 8h** — some
  USDS-M perpetuals settle every 4h. `funding_interval_hours` must be read
  per symbol from `GET /fapi/v1/fundingInfo`, never assumed.
- **`walk_forward_folds`** produces sequential, non-overlapping, expanding-
  window folds — report each fold, not just the aggregate. **
  `random_signal_control`** runs the same strategy shape with random entries
  at the same frequency/direction mix; compare a real result's
  `percentile_rank` against that distribution before believing it.
- **Rate provenance and version**: every India/crypto rate is a named
  constant with a single source of truth in `costs.py`, tagged with
  `RATE_SCHEDULE_VERSION`. Full sourcing (URLs, retrieval dates, what's
  primary vs. secondary, and what's NOT independently verified — see the
  IPFT F&O caveat) is in `costs.py`'s module docstring. **These are
  SECONDARY sources for the India rates** (broker fee pages, not the raw
  CBDT/NSE circular text) — verify against your own contract note before
  relying on them for live P&L. Binance's crypto fee rates come from
  Binance's own published FAQ (primary).

---

## Market data pipeline (capture.py, reconstruct.py, ofi.py)

Architecture: **capture raw, reconstruct offline.** `capture.py` does nothing
but receive bytes and timestamp them — no parsing, no book maintenance, no
analysis. `reconstruct.py` is a separate, re-runnable offline process that
replays capture files and does all the interpretation. This split matters
because raw Binance history is not re-downloadable after the fact: a bug in
reconstruction logic costs a re-run against data you still have; a bug in
capture-time parsing costs the data itself.

### Capture format

Each capture file is **zstd-compressed, newline-delimited JSON (JSONL)**,
rotated hourly:

```
<out_dir>/<UTC date>/<UTC hour>0000Z.jsonl.zst
e.g. smoke_capture/2026-09-13/2026-09-13T060000Z.jsonl.zst
```

**The zstd stream is periodically closed with `FLUSH_FRAME`** (every
`flush_interval_s`, default 5s), producing a file made of multiple
concatenated zstd frames rather than one long-lived frame. This was verified
empirically during development: a bare `ZstdDecompressor().decompress(data)`
on such a file silently returns only the FIRST frame and truncates
everything after it.

> **Readers MUST use `ZstdDecompressor().stream_reader(fh)`**, which reads
> across all concatenated frames transparently. Never use the one-shot
> `decompress()` call on a capture file.

**Unclean shutdown can still leave a malformed line or a corrupt tail.**
`FLUSH_FRAME` bounds *unflushed* data loss to one flush interval, but that's
not the only way killing `capture.py` mid-write shows up on disk:

- **Truncated final line inside a valid frame** — observed in practice from
  Ctrl+C mid-write. `reconstruct._iter_envelopes` handles this: catches the
  per-line JSON error, logs the file/line/decompressed-byte-offset, counts
  it in `ReconstructionStats.corrupt_envelopes`, and keeps processing the
  rest of the file rather than crashing the whole reconstruction.
- **Corrupt trailing zstd frame** (the OS-level write behind a flush itself
  interrupted) — a rarer, worse case: verified empirically that
  `stream_reader` does **not** raise on a deliberately truncated compressed
  frame, it just silently stops yielding further content with **no error
  and no signal at all**. `reconstruct.py` does not currently detect this
  case. If a reconstruction's row/trade counts look implausibly low for a
  capture window with no corresponding `corrupt_envelopes` or
  `gaps_detected`, this untracked failure mode is one explanation worth
  checking by hand (e.g. comparing decompressed output size against
  expectations) before trusting the output.

Each JSONL line is one envelope object:

```json
{
  "schema": "capture.v1",
  "type": "message" | "gap" | "snapshot",
  "stream": "<binance stream name>" | "<symbol>@depth_snapshot" | null,
  "recv_wall_ns": 1789280301249752700,
  "recv_mono_ns": 223777860188200,
  "raw": "<verbatim text>",
  "reason": "<category>:<initial_connect|reconnect>"
}
```

| Field | Present when | Meaning |
|---|---|---|
| `type: "message"` | always | `raw` is the exact WS frame text Binance sent for that routed connection: `{"stream":"<name>","data":{...}}`. Never `json.loads`'d by capture.py. |
| `type: "snapshot"` | always | `raw` is the exact REST `/fapi/v1/depth` response body. Exists because Binance's REST endpoint only ever returns the CURRENT book — an offline reconstructor replaying old files needs a snapshot captured close in time to the diff stream, which is why `capture.py` fetches one itself (`_snapshot_task`) rather than leaving that to whoever runs `reconstruct.py` later. |
| `type: "gap"` | always | No `raw`. `reason` is `"<category>:initial_connect"` or `"<category>:reconnect"` — `category` is `"public"` or `"market"` (see **Routed connections** below), telling `reconstruct.py` which connection actually broke. Emitted on EVERY connect, not just reconnects after a failure. |
| `recv_wall_ns` | always | `time.time_ns()` at receipt — wall clock, for correlating with other sources. |
| `recv_mono_ns` | always | `time.monotonic_ns()` at receipt — immune to clock adjustment, for interval arithmetic (this minus the exchange timestamp is your measured latency). |
| `stream` | `message`/`snapshot` | Cheaply sliced from the leading `"stream":"..."` text (not a JSON parse) for filing/logging; `null` if extraction fails, but the record is still kept. |

Exchange timestamps (Binance's `E`/`T` fields, milliseconds) live **inside**
`raw` and are only extracted by `reconstruct.py`, which is the only process
in this codebase that calls `json.loads` on message/snapshot *content*.

### Routed connections (public / market)

Binance restructured USDS-M futures WebSocket delivery into three
category-routed base URLs — **Public** (high-frequency: order book depth,
book tickers), **Market** (regular: aggregate trades, mark price, klines,
...), and **Private** (user data) — and retired the old unrouted URL for
everything except Public data. `capture.py` opens **one WebSocket
connection per category actually in use** (depth on `/public`, aggTrade on
`/market`), each with its own independent reconnect/backoff cycle and its
own category-tagged gap markers. See `capture.py`'s module docstring
("LEGACY URL RETIREMENT AND ROUTING") for the full story, including an
ambiguity in Binance's own documentation that's deliberately left
unresolved and flagged in a comment rather than silently assumed one way.

**Why this matters operationally**: before this split, a single unrouted
connection silently delivered depth data while aggTrade was dropped with
**no error, no disconnect, and no gap marker** — the connection looked
completely healthy by every metric this project logs. Two full capture
sessions ran with `trades_processed` stuck at 0 before this was caught, by
noticing the *absence* of trades rather than any explicit failure signal.
That failure mode — a connection that stays open and looks fine while one
category's data silently stops — is also why `reconstruct.py`'s gap
handling and `capture.py`'s stats are now category-aware (see below): a
gap that doesn't affect book state must not be reported as if it does, and
per-category health needs its own signal distinct from "is the connection
still open."

`reconstruct.py` uses the category prefix on `reason` to decide whether a
gap invalidates order-book state at all: a `"market"`-category gap (an
aggTrade hiccup) has no bearing on whether any depth event was missed and
does **not** invalidate the book (tracked separately in
`ReconstructionStats.non_book_gaps_detected`); only a `"public"`-category
gap does (`ReconstructionStats.gaps_detected`). A `reason` with no
category prefix — from capture files written before this split — is
treated conservatively as book-affecting, since which connection it came
from can't be determined after the fact.

### Capture operational notes

- One WebSocket connection per routed category in use (see above), each up
  to Binance's 1024-streams-per-connection limit — `CaptureConfig` raises
  at construction time if any single category's stream count exceeds it.
- Reconnects use exponential backoff (`backoff_initial_s` → `backoff_max_s`,
  with jitter) **independently per category** — one connection's backoff
  state has no effect on the other's. Binance's own mandatory 24h
  connection expiry is handled identically to any other disconnect, per
  connection.
- `messages_dropped` (logged every `stats_log_interval_s`) counts messages
  the receive loop discarded because the internal queue was full — a
  **capture-layer backpressure metric**, distinct from an exchange-side
  sequence gap (which only `reconstruct.py` can detect, from `u`/`pu`).
- `CaptureStats.last_message_recv_wall_ns` is tracked **per category** and
  logged as an age (seconds since last message) alongside throughput. This
  exists to catch a failure mode reconnects and gap markers cannot see: a
  connection that stays open (keeps passing ping/pong) but silently stops
  receiving new messages from Binance's side. That looks completely
  healthy on every other metric — watch this one specifically if you
  suspect one category has gone quiet without the other.
- Periodic REST snapshots (`snapshot_interval_s`, default 15 min) are
  weight-budgeted at construction time against Binance's confirmed
  `2400`/minute/IP limit (`binance_rest.py`), capped conservatively at half
  that budget to leave headroom for `reconstruct.py`'s verification-mode
  fetches sharing the same IP.

### Order book reconstruction (reconstruct.py)

Implements Binance's own documented procedure for USDS-M futures diff-depth
streams **exactly** — see the URL and quoted steps in `reconstruct.BookState`
(retrieved 2026-09-13). This is NOT the spot procedure (spot lacks the `pu`
field and uses a different check) — don't reuse `BookState` for spot data.

Key points that are easy to get subtly wrong (both found and fixed during
this project's own development — see git history / commit messages if this
repo is later put under version control):

- **The documented "buffer events" step (before fetching the snapshot)
  matters in practice, not just in principle.** REST round-trip latency
  routinely means the correct bootstrap event (`U <= lastUpdateId <= u`) is
  already sitting a few events *behind* the snapshot's arrival in the
  stream, not still ahead of it. `BookState` keeps a small rolling buffer
  (`RECENT_EVENTS_BUFFER`, 200 events / ~20s at `@100ms`) and replays it on
  every `load_snapshot()` before falling back to waiting for new live
  events. Without this, a healthy connection with a realistic snapshot
  cadence spuriously invalidates on almost every resync.
- **A single WS-connection gap invalidates every symbol on it**, not just
  one — the combined stream is shared, so a reconnect is a break for all
  tracked symbols simultaneously.
- **Every `snapshot`-type row's content already includes buffered diff
  events replayed on top of the raw REST payload**, per the documented
  "buffer events before fetching the snapshot" step — `load_snapshot()`
  unconditionally replays `RECENT_EVENTS_BUFFER` on every snapshot, not just
  after a gap. Its `update_id` column reports `book.last_update_id`
  *after* that replay, not the raw snapshot's own `lastUpdateId` — the two
  can differ, and using the raw value there would mislabel the row's actual
  content. Comparing a snapshot row's bids/asks byte-for-byte against the
  raw REST response it was bootstrapped from (e.g. via
  `verify_capture_snapshots`, which is circular in exactly this way) will
  therefore often show quantity-only divergence at a handful of levels with
  zero price diff — that's this replay, not corruption.
- **On any violation, the book is marked INVALID and cleared — never
  patched, never interpolated.** Rows written while invalid carry empty
  level lists (`bid_prices: []`, not stale or guessed values) and
  `valid: false`. See `tests/test_reconstruct_invalid.py`, which asserts
  this directly: a plausible-but-wrong book is strictly worse than a
  documented hole, because nothing downstream can tell the difference
  between real and patched data, but a hole is visible and can be excluded.

**Output**: `book_state.parquet` (one row per depth-diff/snapshot/gap event)
and `trades.parquet` (one row per aggTrade), both written incrementally via
`pyarrow.parquet.ParquetWriter` in configurable row-group batches so
reconstruction of a multi-million-row capture doesn't hold the whole output
table in memory. A channel with zero events still produces an empty
schema-only Parquet file, never a missing one.

`book_state.parquet` columns: `symbol`, `event_type`
(`snapshot`/`update`/`gap`), `event_time_ns` (exchange timestamp, null for
gap rows), `update_id`, `recv_wall_ns`, `recv_mono_ns`, `valid`,
`bid_prices`/`bid_qtys`/`ask_prices`/`ask_qtys` (top-N levels, N =
`depth_levels`, empty lists when invalid).

`trades.parquet` columns: `symbol`, `agg_trade_id`, `trade_time_ns`,
`recv_wall_ns`, `recv_mono_ns`, `price`, `quantity`, `is_buyer_maker`,
`first_trade_id`, `last_trade_id`.

### Verification mode

`reconstruct.verify_against_snapshot()` / `verify_capture_snapshots()`
compare reconstructed book state against an independently-obtained REST
snapshot and **report** divergence (a `DivergenceReport`) — they never
write back to or otherwise alter the reconstructed data. An
auto-correcting verifier would hide the exact defect it exists to surface.

**Caveat this code does not hide from you**: `verify_capture_snapshots()`
checks against the SAME snapshot records `reconstruct()` used for bootstrap
by default, which is partially circular (it will trivially match at the
instant a snapshot was applied). It still catches drift between resyncs. For
a non-circular check, fetch a snapshot independently
(`binance_rest.fetch_depth_snapshot`) while reconstruction is running near
real time and pass it to `verify_against_snapshot` directly.

### OFI and decay (ofi.py)

- OFI is computed per Cont, Kukanov & Stoikov (2014), *The Price Impact of
  Order Book Events*, from best-bid/best-ask changes between consecutive
  **valid** top-of-book observations only (`ofi.ofi_events`). OFI is never
  computed across a reconstruction gap — the first event of every valid
  segment (i.e. right after a resync) gets a null OFI, not a value computed
  against a book state from before the gap.
- Trade-side classification uses Binance aggTrade's `is_buyer_maker` flag
  directly (`ofi.aggressor_side`) — never tick-rule/Lee-Ready inference.
- `ofi.decay_matrix()` computes `corr(OFI over window w, forward return over
  horizon h)` for the full `{100ms, 500ms, 1s, 5s, 30s, 1m, 5m, 15m}` grid,
  plus a `predicted_edge_bps` (the OLS-implied return at a 1-std OFI shock).
  The forward return is found via a **forward** `join_asof` — this is the
  one place in the module where a directional mistake would silently
  manufacture look-ahead edge; see the module docstring's explicit warning.
- `ofi.overlay_cost_hurdle()` marks which `(w, h)` cells show predicted edge
  exceeding round-trip TAKER cost from the real `costs.py` model (not a
  duplicated bps constant). `check_hurdle_sanity()` logs a warning if
  **every** cell passes — per quant-backtest-discipline, most should fail,
  and "none fail" is a reason to suspect look-ahead, not celebrate.

---

## Known limitations (state these wherever results are reported)

- **Depth is throttled, not tick-by-tick.** `@depth@100ms` is Binance
  consolidating book changes into 100ms buckets. Multiple adds/cancels/trades
  between observations are netted into one visible delta — sub-100ms
  microstructure is not observable through this feed at all. Don't build
  analysis that assumes finer resolution than the feed carries.
- **Top-N levels are not the whole book.** `depth_levels` (default 20)
  truncates; liquidity beyond that depth is invisible to `reconstruct.py`'s
  output even though the in-memory `BookState` tracks the full book Binance
  sends.
- **This is market-by-price, not market-by-order.** Individual order queue
  position within a price level is fundamentally unknowable from this feed —
  this is exactly why `backtest.py` never assumes a passive/maker fill.
- **India rate figures are SECONDARY-sourced** (broker fee pages) and the
  F&O IPFT rate specifically is not independently confirmed against an
  NSE F&O circular — see `costs.py`'s module docstring.
- **Funding interval defaults to 8h but is symbol-specific** — always
  confirm per symbol via `GET /fapi/v1/fundingInfo` before relying on
  `holding_cost_funding` for a specific instrument.
- **No margin, liquidation, or position-limit modeling anywhere in
  `backtest.py`.** It models order-level cost and pessimistic fills only.
- **`trades.parquet` has no validity/gap-awareness column at all.** Unlike
  `book_state.parquet`, an aggTrade-side (`"market"`-category) gap or
  connection stall does not mark any trade rows as suspect or produce a
  visible hole — trades are written as parsed, with no signal that some
  window's worth might be missing. A trade-flow calculation (OFI
  cross-check, signed volume, etc.) spanning a `"market"`-category
  reconnect or an extended silent stall (see `last_message_recv_wall_ns`
  above) could silently undercount real trade activity in that window with
  nothing in the output to flag it. If you build trade-flow analysis on a
  capture that had any `non_book_gaps_detected` or a large
  `last_recv_age_s` gap for the `"market"` category, treat that window's
  trade counts as a lower bound, not a measurement.
- **The legacy-vs-routed-URL ambiguity documented in `capture.py` is
  unresolved, not fixed.** This project migrated to the routed `/public`
  and `/market` URLs specifically because Binance's own materials disagree
  on whether the old unrouted URL keeps serving Public-category data
  indefinitely. That migration removes the question for this project; it
  does not mean the underlying ambiguity has an answer.
- **A corrupt trailing zstd frame (not just a truncated JSON line) is a
  known, currently-undetected failure mode.** `reconstruct.py` catches and
  counts a malformed JSON line (`ReconstructionStats.corrupt_envelopes`),
  but a deliberately truncated *compressed* frame produces no exception at
  all when read — it silently stops yielding content. There is no counter
  or log line for this case. See `capture.py`'s module docstring.
