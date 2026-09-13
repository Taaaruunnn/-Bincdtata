"""Unit tests for reconstruct.BookState's sequence-gap detector, against
synthetic event streams with deliberately injected gaps, out-of-order
events, and duplicates (quant-backtest-discipline / market-data-integrity
testing requirements)."""

from __future__ import annotations

from reconstruct import BookState


def _book() -> BookState:
    return BookState(symbol="BTCUSDT", depth_limit=10)


def test_snapshot_then_clean_bootstrap_and_continuation():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    assert b.valid

    # First event brackets lastUpdateId=100 (U<=100<=u)
    applied = b.apply_diff(U=95, u=105, pu=90, bid_updates=[(99.0, 2.0)], ask_updates=[])
    assert applied is True
    assert b.valid is True
    assert b.bids[99.0] == 2.0
    assert b.last_update_id == 105

    # Continuation: pu matches previous u
    applied = b.apply_diff(U=106, u=110, pu=105, bid_updates=[], ask_updates=[(100.0, 3.0)])
    assert applied is True
    assert b.asks[100.0] == 3.0
    assert b.last_update_id == 110


def test_stale_event_dropped_during_bootstrap_without_invalidating():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])

    # u < lastUpdateId: per documented step 4, DROP (not invalidate) -- the
    # book should still be waiting for a qualifying event, not broken.
    applied = b.apply_diff(U=40, u=50, pu=39, bid_updates=[(99.0, 999.0)], ask_updates=[])
    assert applied is False
    assert b.valid is True  # still fine, just dropped as stale
    assert b.awaiting_first_event is True
    assert b.bids[99.0] == 1.0  # untouched by the dropped event


def test_bootstrap_gap_invalidates_when_U_exceeds_last_update_id():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])

    # U > lastUpdateId: no update ID before this event's start ever existed
    # in the stream at lastUpdateId -- a real gap between snapshot and
    # stream start. Must invalidate, not guess.
    applied = b.apply_diff(U=150, u=160, pu=149, bid_updates=[(99.0, 5.0)], ask_updates=[])
    assert applied is False
    assert b.valid is False
    # book state must be CLEARED on invalidation, not left holding the
    # pre-gap snapshot values as if they were still trustworthy
    assert b.bids == {}
    assert b.asks == {}


def test_mid_stream_gap_pu_mismatch_invalidates():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    assert b.apply_diff(U=95, u=105, pu=90, bid_updates=[], ask_updates=[]) is True
    assert b.last_update_id == 105

    # pu should equal 105 (previous u); it doesn't -> a real gap occurred
    # between the two events.
    applied = b.apply_diff(U=120, u=130, pu=119, bid_updates=[(99.0, 42.0)], ask_updates=[])
    assert applied is False
    assert b.valid is False
    assert b.bids == {}  # NOT left at {99.0: 1.0} or partially updated -- cleared


def test_duplicate_event_after_bootstrap_invalidates():
    """A byte-for-byte re-delivery of an already-applied event has pu equal
    to ITS OWN predecessor, which no longer matches the book's current
    last_update_id (which has since advanced). Per the documented pu-chain
    check, this correctly triggers invalidation rather than being silently
    re-applied (idempotent replay is not something the documented procedure
    promises, and pretending it is would risk double-counting a level
    update that used a delta rather than an absolute value elsewhere)."""
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    assert b.apply_diff(U=95, u=105, pu=90, bid_updates=[(99.0, 2.0)], ask_updates=[]) is True
    assert b.apply_diff(U=106, u=110, pu=105, bid_updates=[(99.0, 3.0)], ask_updates=[]) is True
    assert b.last_update_id == 110

    # Re-deliver the FIRST event again (duplicate)
    applied = b.apply_diff(U=95, u=105, pu=90, bid_updates=[(99.0, 2.0)], ask_updates=[])
    assert applied is False
    assert b.valid is False


def test_out_of_order_event_after_bootstrap_invalidates():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    assert b.apply_diff(U=95, u=105, pu=90, bid_updates=[], ask_updates=[]) is True
    assert b.apply_diff(U=106, u=110, pu=105, bid_updates=[], ask_updates=[]) is True
    assert b.apply_diff(U=111, u=115, pu=110, bid_updates=[], ask_updates=[]) is True
    assert b.last_update_id == 115

    # An event that should have arrived BEFORE u=110 shows up late/out of
    # order; its pu will not match the current last_update_id (115).
    applied = b.apply_diff(U=106, u=109, pu=105, bid_updates=[], ask_updates=[])
    assert applied is False
    assert b.valid is False


def test_events_while_invalid_are_all_dropped_until_resync():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    b.apply_diff(U=150, u=160, pu=149, bid_updates=[], ask_updates=[])  # forces invalid
    assert b.valid is False

    # Further events -- even ones that "look" continuous with each other --
    # must NOT be applied while invalid. No patching, no silent resumption.
    applied1 = b.apply_diff(U=160, u=170, pu=159, bid_updates=[(99.0, 77.0)], ask_updates=[])
    applied2 = b.apply_diff(U=170, u=180, pu=170, bid_updates=[(99.0, 88.0)], ask_updates=[])
    assert applied1 is False
    assert applied2 is False
    assert b.valid is False
    assert b.bids == {}


def test_resync_after_gap_restores_valid_book_with_fresh_state():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0)], asks=[(100.0, 1.0)])
    b.apply_diff(U=150, u=160, pu=149, bid_updates=[], ask_updates=[])  # forces invalid
    assert b.valid is False

    # Fresh snapshot resync
    b.load_snapshot(last_update_id=500, bids=[(199.0, 4.0)], asks=[(200.0, 4.0)])
    assert b.valid is True
    assert b.bids == {199.0: 4.0}  # old (cleared) state is gone, not merged with new


def test_zero_quantity_removes_level_and_missing_level_removal_is_a_noop():
    b = _book()
    b.load_snapshot(last_update_id=100, bids=[(99.0, 1.0), (98.0, 2.0)], asks=[(100.0, 1.0)])
    b.apply_diff(U=95, u=105, pu=90, bid_updates=[(99.0, 0.0), (97.0, 0.0)], ask_updates=[])
    # 99.0 removed (qty 0); 97.0 wasn't present -- removing it is a normal no-op
    assert 99.0 not in b.bids
    assert b.bids == {98.0: 2.0}


def test_late_bootstrap_from_recent_events_buffer():
    """Reproduces the real bug found during development: if diff events
    arrive (and are buffered, since the book is invalid/not yet bootstrapped)
    BEFORE load_snapshot() is called, the correct first event to apply may
    already be sitting in that buffer rather than arriving fresh afterward.
    This matches the documented step 2 ("buffer events") occurring BEFORE
    step 3 (fetch snapshot)."""
    b = _book()
    # No snapshot yet -- book invalid, these are buffered but not applied.
    assert b.apply_diff(U=90, u=99, pu=85, bid_updates=[(50.0, 1.0)], ask_updates=[]) is False
    assert b.apply_diff(U=100, u=110, pu=99, bid_updates=[(50.0, 2.0)], ask_updates=[]) is False
    assert b.valid is False

    # Snapshot's lastUpdateId falls inside the SECOND buffered event's
    # [U, u] range (100 <= 105 <= 110) -- the correct bootstrap point is
    # already in the buffer, not in whatever arrives next.
    b.load_snapshot(last_update_id=105, bids=[(50.0, 999.0)], asks=[(51.0, 1.0)])
    assert b.valid is True
    assert b.last_update_id == 110
    assert b.bids[50.0] == 2.0  # applied from the buffered event, not left at the snapshot's 999.0


def test_recent_events_buffer_replay_is_safe_when_nothing_qualifies():
    """If the buffer contains only events irrelevant to the new snapshot
    (all stale, u < new lastUpdateId), bootstrap must fall back to waiting
    for live events rather than mis-firing on a stale one."""
    b = _book()
    b.apply_diff(U=1, u=5, pu=0, bid_updates=[], ask_updates=[])  # buffered while invalid
    b.load_snapshot(last_update_id=1000, bids=[(50.0, 1.0)], asks=[(51.0, 1.0)])
    assert b.valid is True
    assert b.awaiting_first_event is True  # buffer replay found nothing usable; still waiting
    assert b.last_update_id == 1000
