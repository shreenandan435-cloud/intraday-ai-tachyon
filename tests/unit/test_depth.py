"""Level-2 depth & Order Book Imbalance integration — CLAUDE.md §2, §3.1, §4, §6.

The depth path reduces Angel One's five-level SNAP_QUOTE ladder to the fixed 8-float
shared-memory slot ``(b1p, b1q, b2p, b2q, s1p, s1q, s2p, s2q)`` and, at parse time, also
computes the top-of-book Order Book Imbalance ``obi_l1 = (bid_qty - ask_qty) /
(bid_qty + ask_qty)``. The imbalance cannot ride inside the 56-byte slot (the C++ consumer
compiles against that layout), so it is published out-of-band — ``NormalizedTick.obi_l1``,
``AngelOneWebSocketClient.latest_obi`` and the optional ``obi_sink`` — and the execution
boundary re-derives it from the top-of-book it reads back.

What these tests prove:

* the OBI formula matches the math engine's ``calculate_obi`` kernel, including the
  empty-book → ``0.0`` convention;
* the depth mapping is positional, paise prices scale to rupees, and a thin book pads with
  zeros so field positions never shift;
* a crossed book (bid > ask after mapping) is counted by the depth audit;
* the imbalance survives the SHM ring round-trip via the top-of-book quantities;
* the router's OBI entry gate refuses zero-, below-threshold- and opposed-imbalance books
  and admits only an aligned, above-threshold imbalance.

The OBI gate tests build a fully-passing risk stack (ACTIVE state, temp daily-lock, fresh
feed monitor, empty registry) so that the *only* thing that can veto an aligned entry is the
gate under test, and the only thing that can admit a bad one is a bug in the gate.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import Settings, StrategySettings, WatchlistItem
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.execution.router import (
    ACTION_BUY,
    ACTION_SELL,
    ActionIntent,
    ExecutionRouter,
    ShmTopOfBookReader,
    TopOfBook,
)
from tachyon.ingestion.angel_adapter import (
    AngelOneWebSocketClient,
    normalize_depth,
    order_book_imbalance_l1,
    tick_from_dict,
)
from tachyon.ingestion.shm_writer import SHMWriter
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math_engine import warmup
from tachyon.math_engine.indicators import calculate_obi
from tachyon.risk.engine import RiskEngine
from tachyon.risk.tracker import PnLTracker, PositionRegistry

SYMBOL: Final[str] = "RELIANCE"
TOKEN: Final[str] = "2885"
TOKEN_INT: Final[int] = 2885


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock(hh: int = 9, mm: int = 30, mono: float = 1000.0) -> ManualClock:
    # 09:30 IST → GOLDEN_WINDOW, the regime where momentum signals are allowed.
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=mono)


def _unique_segment() -> str:
    return f"tachyon_ut_depth_{os.getpid()}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def shm_name() -> Iterator[str]:
    """Isolated segment per test: never touch a live sidecar's ring."""
    yield _unique_segment()


def _settings(obi_threshold: float = 0.3) -> Settings:
    return Settings(
        watchlist=(WatchlistItem(symbol=SYMBOL, token=TOKEN, exchange="NSE"),),
        strategy=StrategySettings(obi_threshold=obi_threshold),
    )


def _passing_risk(tmp_path: Path, clock: ManualClock, settings: Settings) -> RiskEngine:
    """A fully-wired risk stack that admits ``SYMBOL`` — mirrors test_risk._Harness.

    ACTIVE state, unengaged temp daily-lock, fresh feed monitor, empty position registry and
    no wired Sentinel/margin/quote providers: every check passes, so any veto observed in the
    router tests below comes from the OBI gate, not from the twelve-check gate behind it.
    """
    machine = StateMachine(TradingState.ACTIVE, clock=clock)
    lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
    pnl = PnLTracker(machine, daily_lock=lock, clock=clock)
    monitor = FeedMonitor(clock=clock)
    monitor.record()  # feed is alive by default
    return RiskEngine(machine, pnl, monitor, PositionRegistry(), settings=settings, clock=clock)


def _book(bid_qty: float, ask_qty: float, *, bid: float = 100.0, ask: float = 101.0) -> TopOfBook:
    return TopOfBook(bid=bid, bid_qty=bid_qty, ask=ask, ask_qty=ask_qty, ts_epoch=1.0)


def _router(
    *,
    tmp_path: Path,
    book: TopOfBook | None,
    settings: Settings,
    clock: ManualClock,
) -> ExecutionRouter:
    return ExecutionRouter(
        risk=_passing_risk(tmp_path, clock, settings),
        settings=settings,
        paper_trade=True,
        tokens={TOKEN: SYMBOL},
        book_provider=lambda _token: book,
        clock=clock,
    )


def _entry(action: int) -> ActionIntent:
    return ActionIntent(action=action, instrument_token=TOKEN_INT, timestamp_ns=1)


# ──────────────────────────────────────────────────────────────────────────────
# The OBI formula — order_book_imbalance_l1
# ──────────────────────────────────────────────────────────────────────────────


class TestOrderBookImbalanceL1:
    """The parse-time imbalance must be exactly the math engine's kernel."""

    def test_balanced_book_is_zero(self) -> None:
        assert order_book_imbalance_l1(100.0, 100.0) == 0.0

    def test_bid_heavy_is_positive(self) -> None:
        assert order_book_imbalance_l1(300.0, 100.0) == pytest.approx(0.5)

    def test_ask_heavy_is_negative(self) -> None:
        assert order_book_imbalance_l1(100.0, 300.0) == pytest.approx(-0.5)

    def test_one_sided_buy_book_is_plus_one(self) -> None:
        assert order_book_imbalance_l1(500.0, 0.0) == pytest.approx(1.0)

    def test_one_sided_sell_book_is_minus_one(self) -> None:
        assert order_book_imbalance_l1(0.0, 500.0) == pytest.approx(-1.0)

    def test_empty_book_is_zero_not_nan(self) -> None:
        """No size on either side is *no evidence*, never an exception or NaN."""
        result = order_book_imbalance_l1(0.0, 0.0)
        assert result == 0.0

    @pytest.mark.parametrize(
        ("bid_qty", "ask_qty"),
        [(1, 1), (10, 5), (5, 10), (1234, 4321), (0, 7), (7, 0), (0, 0)],
    )
    def test_agrees_with_the_math_engine_kernel(self, bid_qty: int, ask_qty: int) -> None:
        assert order_book_imbalance_l1(float(bid_qty), float(ask_qty)) == pytest.approx(
            calculate_obi(bid_qty, ask_qty)
        )

    def test_bounded_to_unit_interval(self) -> None:
        for bid in (0.0, 1.0, 10.0, 1e6):
            for ask in (0.0, 1.0, 10.0, 1e6):
                assert -1.0 <= order_book_imbalance_l1(bid, ask) <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Depth mapping — normalize_depth / tick_from_dict
# ──────────────────────────────────────────────────────────────────────────────


class TestDepthMapping:
    def test_top_two_levels_map_positionally(self) -> None:
        payload = {
            "token": TOKEN_INT,
            "exchange_timestamp": 1_755_000_000_000,
            "depth": {
                "buy": [
                    {"price": 100.5, "quantity": 500},
                    {"price": 100.4, "quantity": 1200},
                    {"price": 100.3, "quantity": 40},  # third level dropped by design
                ],
                "sell": [
                    {"price": 100.7, "quantity": 300},
                    {"price": 100.8, "quantity": 900},
                ],
            },
        }
        tick = tick_from_dict(payload)
        assert tick.floats == (
            100.5,
            500.0,
            100.4,
            1200.0,  # top two bids
            100.7,
            300.0,
            100.8,
            900.0,  # top two asks
        )

    def test_v2_paise_prices_scale_to_rupees(self) -> None:
        payload = {
            "token": TOKEN_INT,
            "exchange_timestamp": 1,
            "best_5_buy_data": [{"flag": 1, "quantity": 500, "price": 10_050, "no of orders": 3}],
            "best_5_sell_data": [{"flag": 0, "quantity": 300, "price": 10_070, "no of orders": 2}],
        }
        tick = tick_from_dict(payload)
        assert tick.floats[0] == pytest.approx(100.50)
        assert tick.floats[4] == pytest.approx(100.70)

    def test_thin_book_pads_with_zeros(self) -> None:
        payload = {
            "token": 99,
            "exchange_timestamp": 1,
            "depth": {"buy": [{"price": 10.0, "quantity": 1}]},
            "depth_sell": [],
        }
        tick = tick_from_dict(payload)
        assert tick.floats == (10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert len(tick.floats) == 8

    def test_normalize_depth_always_returns_eight_floats(self) -> None:
        assert len(normalize_depth((), ())) == 8
        assert normalize_depth((), ()) == (0.0,) * 8

    def test_obi_l1_rides_on_the_tick(self) -> None:
        payload = {
            "token": TOKEN_INT,
            "exchange_timestamp": 1,
            "depth": {
                "buy": [{"price": 100.0, "quantity": 300}],
                "sell": [{"price": 100.5, "quantity": 100}],
            },
        }
        tick = tick_from_dict(payload)
        assert tick.obi_l1 == pytest.approx((300 - 100) / (300 + 100))

    def test_empty_book_tick_has_zero_obi(self) -> None:
        tick = tick_from_dict({"token": 1, "exchange_timestamp": 1})
        assert tick.obi_l1 == 0.0
        assert tick.floats == (0.0,) * 8


# ──────────────────────────────────────────────────────────────────────────────
# Client publication — latest_obi, obi_sink, crossed-book audit
# ──────────────────────────────────────────────────────────────────────────────


def _depth_payload(bid_qty: int, ask_qty: int) -> dict[str, Any]:
    return {
        "token": TOKEN_INT,
        "exchange_timestamp": 1_755_000_000_000,
        "depth": {
            "buy": [{"price": 100.0, "quantity": bid_qty}],
            "sell": [{"price": 100.5, "quantity": ask_qty}],
        },
    }


class TestClientObiPublication:
    def test_latest_obi_and_sink_receive_the_value(self, shm_name: str) -> None:
        published: list[tuple[int, float]] = []
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(
                writer,
                tokens=(TOKEN,),
                obi_sink=lambda iid, obi: published.append((iid, obi)),
            )
            client.on_data(None, _depth_payload(300, 100))

            assert client.ticks_written == 1
            assert client.latest_obi[TOKEN_INT] == pytest.approx(0.5)
            assert published == [(TOKEN_INT, pytest.approx(0.5))]
        finally:
            writer.close()

    def test_a_failing_sink_never_takes_down_the_feed(self, shm_name: str) -> None:
        def boom(_iid: int, _obi: float) -> None:
            raise RuntimeError("observer exploded")

        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, tokens=(TOKEN,), obi_sink=boom)
            client.on_data(None, _depth_payload(300, 100))
            # The tick still landed and the imbalance was still retained.
            assert client.ticks_written == 1
            assert client.latest_obi[TOKEN_INT] == pytest.approx(0.5)
        finally:
            writer.close()

    def test_an_uncrossed_book_leaves_the_audit_at_zero(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, tokens=(TOKEN,))
            client.on_data(None, _depth_payload(300, 100))
            assert client.crossed_books == 0
        finally:
            writer.close()

    def test_a_crossed_book_is_counted(self, shm_name: str) -> None:
        """bid > ask after mapping is the standing alarm that side labelling inverted."""
        crossed = {
            "token": TOKEN_INT,
            "exchange_timestamp": 1,
            "depth": {
                "buy": [{"price": 101.0, "quantity": 100}],
                "sell": [{"price": 100.0, "quantity": 50}],
            },
        }
        writer = SHMWriter(shm_name)
        try:
            client = AngelOneWebSocketClient(writer, tokens=(TOKEN,))
            client.on_data(None, crossed)
            assert client.crossed_books == 1
        finally:
            writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Shared-memory round-trip — the imbalance survives via the top-of-book
# ──────────────────────────────────────────────────────────────────────────────


class TestShmRoundTrip:
    def test_the_slot_round_trips_and_obi_is_rederivable(self, shm_name: str) -> None:
        floats = (100.0, 500.0, 99.9, 400.0, 101.0, 300.0, 101.1, 200.0)
        writer = SHMWriter(shm_name)
        try:
            writer.write_tick(TOKEN_INT, 1_755_000_000_000_000_000, floats)
        finally:
            writer.close()

        reader = ShmTopOfBookReader(shm_name)
        try:
            book = reader.read(TOKEN_INT)
            assert book is not None
            assert book.bid == pytest.approx(100.0)
            assert book.bid_qty == pytest.approx(500.0)
            assert book.ask == pytest.approx(101.0)
            assert book.ask_qty == pytest.approx(300.0)
            assert book.usable
            assert book.obi_l1 == pytest.approx((500 - 300) / (500 + 300))
        finally:
            reader.close()

    def test_an_empty_ring_reads_none(self, shm_name: str) -> None:
        writer = SHMWriter(shm_name)
        writer.close()
        reader = ShmTopOfBookReader(shm_name)
        try:
            assert reader.read(TOKEN_INT) is None
        finally:
            reader.close()

    def test_a_crossed_slot_is_unusable_and_reads_none(self, shm_name: str) -> None:
        # bid price above ask price: the reader must refuse to price off it.
        floats = (102.0, 500.0, 101.9, 400.0, 101.0, 300.0, 101.1, 200.0)
        writer = SHMWriter(shm_name)
        try:
            writer.write_tick(TOKEN_INT, 1, floats)
        finally:
            writer.close()
        reader = ShmTopOfBookReader(shm_name)
        try:
            assert reader.read(TOKEN_INT) is None
        finally:
            reader.close()


# ──────────────────────────────────────────────────────────────────────────────
# TopOfBook.obi_l1 — the execution boundary's own reading
# ──────────────────────────────────────────────────────────────────────────────


class TestTopOfBookObi:
    def test_obi_from_quantities(self) -> None:
        assert _book(900.0, 100.0).obi_l1 == pytest.approx(0.8)
        assert _book(100.0, 900.0).obi_l1 == pytest.approx(-0.8)

    def test_empty_book_obi_is_zero(self) -> None:
        assert _book(0.0, 0.0).obi_l1 == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# The router OBI entry gate — CLAUDE.md §4
# ──────────────────────────────────────────────────────────────────────────────


class TestRouterObiGate:
    """The gate runs before the twelve-check risk engine; a fully-passing risk stack is
    wired so the OBI gate is the only thing under test."""

    @staticmethod
    def _dispatch(
        tmp_path: Path,
        *,
        bid_qty: float,
        ask_qty: float,
        action: int,
        obi_threshold: float = 0.3,
    ) -> tuple[ExecutionRouter, Any]:
        settings = _settings(obi_threshold=obi_threshold)
        router = _router(
            tmp_path=tmp_path,
            book=_book(bid_qty, ask_qty),
            settings=settings,
            clock=_clock(),
        )
        return router, router.handle_payload(_entry(action))

    def test_zero_obi_admits_no_entry(self, tmp_path: Path) -> None:
        router, outcome = self._dispatch(tmp_path, bid_qty=100.0, ask_qty=100.0, action=ACTION_BUY)

        assert outcome.event == "REJECTED"
        assert outcome.accepted is False
        assert outcome.reason == "OBI_BELOW_THRESHOLD"
        assert router.stats.obi_blocks == 1
        assert router.stats.paper_entries == 0

    def test_empty_book_obi_is_treated_as_no_evidence(self, tmp_path: Path) -> None:
        router, outcome = self._dispatch(tmp_path, bid_qty=0.0, ask_qty=0.0, action=ACTION_BUY)

        assert outcome.reason == "OBI_BELOW_THRESHOLD"
        assert router.stats.obi_blocks == 1

    def test_below_threshold_obi_is_rejected(self, tmp_path: Path) -> None:
        # OBI = (60 - 40) / 100 = 0.2, below the 0.3 threshold.
        router, outcome = self._dispatch(tmp_path, bid_qty=60.0, ask_qty=40.0, action=ACTION_BUY)

        assert outcome.reason == "OBI_BELOW_THRESHOLD"
        assert router.stats.obi_blocks == 1
        assert router.stats.paper_entries == 0

    def test_opposed_obi_is_a_direction_conflict(self, tmp_path: Path) -> None:
        # Strongly bid-heavy book (OBI = +0.8) but the action is a SELL.
        router, outcome = self._dispatch(
            tmp_path, bid_qty=900.0, ask_qty=100.0, action=ACTION_SELL
        )

        assert outcome.event == "REJECTED"
        assert outcome.reason == "OBI_DIRECTION_CONFLICT"
        assert router.stats.obi_blocks == 1
        assert router.stats.paper_entries == 0

    def test_aligned_above_threshold_obi_enters(self, tmp_path: Path) -> None:
        # BUY facing a bid-heavy book (OBI = +0.8): clears magnitude and direction.
        router, outcome = self._dispatch(tmp_path, bid_qty=900.0, ask_qty=100.0, action=ACTION_BUY)

        assert outcome.event == "ENTRY"
        assert outcome.accepted is True
        assert outcome.mode == "PAPER"
        assert outcome.side == "BUY"
        assert router.stats.obi_blocks == 0
        assert router.stats.paper_entries == 1
        assert router.stats.risk_vetoes == 0

    def test_aligned_sell_enters_against_an_ask_heavy_book(self, tmp_path: Path) -> None:
        router, outcome = self._dispatch(
            tmp_path, bid_qty=100.0, ask_qty=900.0, action=ACTION_SELL
        )

        assert outcome.event == "ENTRY"
        assert outcome.side == "SELL"
        assert router.stats.obi_blocks == 0
        assert router.stats.paper_entries == 1

    def test_obi_blocks_are_reported_in_stats_dict(self, tmp_path: Path) -> None:
        router, _outcome = self._dispatch(
            tmp_path, bid_qty=100.0, ask_qty=100.0, action=ACTION_BUY
        )

        assert router.stats_dict()["obi_blocks"] == 1

    def test_an_unusable_book_is_dropped_before_the_gate(self, tmp_path: Path) -> None:
        settings = _settings()
        router = _router(tmp_path=tmp_path, book=None, settings=settings, clock=_clock())
        outcome = router.handle_payload(_entry(ACTION_BUY))

        assert outcome.event == "DROPPED"
        assert outcome.reason == "NO_TOP_OF_BOOK"
        assert router.stats.obi_blocks == 0


# ──────────────────────────────────────────────────────────────────────────────
# Time-of-day gate (CLAUDE.md §3.1, Step 3) — momentum/ORB hard-rejected
# outside the morning windows, and VWAP pullback is its own signal type.
# ──────────────────────────────────────────────────────────────────────────────


def _entry_with_signal_type(action: int, signal_type: str) -> ActionIntent:
    return ActionIntent(
        action=action, instrument_token=TOKEN_INT, timestamp_ns=1, signal_type=signal_type
    )


class TestTimeGateBlocks:
    """Momentum and ORB signals must be hard-rejected during MIDDAY_CHOP and SQUARE_OFF.

    A real risk: the C++ sidecar may legitimately produce a momentum long during the
    noon lull if the offline model is biased toward trending behaviour. The router's
    job is to refuse it without invoking the OBI gate or the risk engine — the
    discipline lives at the time-of-day layer, not at the risk layer.
    """

    def test_momentum_long_blocked_during_midday_chop(self, tmp_path: Path) -> None:
        clock = _clock(hh=12, mm=0)  # 12:00 IST → MIDDAY_CHOP
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),  # OBI would pass on its own
            settings=_settings(),
            clock=clock,
        )
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )

        assert outcome.event == "REJECTED"
        assert outcome.reason == "TIME_GATE"
        assert router.stats.time_gate_blocks == 1
        # The OBI gate must not have run.
        assert router.stats.obi_blocks == 0
        assert router.stats.paper_entries == 0

    def test_orb_short_blocked_during_square_off(self, tmp_path: Path) -> None:
        clock = _clock(hh=15, mm=10)  # 15:10 IST → SQUARE_OFF
        router = _router(
            tmp_path=tmp_path,
            book=_book(100.0, 900.0),
            settings=_settings(),
            clock=clock,
        )
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_SELL, "ORB")
        )

        assert outcome.reason == "TIME_GATE"
        assert router.stats.time_gate_blocks == 1

    def test_momentum_long_allowed_in_golden_window(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)  # 09:30 → GOLDEN_WINDOW
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        # Should pass the time gate, fail the OBI / enter cleanly.
        assert outcome.reason != "TIME_GATE"
        assert router.stats.time_gate_blocks == 0

    def test_vwap_pullback_passes_the_time_gate_in_midday_chop(
        self, tmp_path: Path
    ) -> None:
        """The strategy owns its regime; the router does not add a second gate."""
        clock = _clock(hh=12, mm=0)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "VWAP_PULLBACK")
        )
        assert outcome.reason != "TIME_GATE"
        assert router.stats.time_gate_blocks == 0

    def test_default_signal_type_is_momentum(self, tmp_path: Path) -> None:
        """Wire frames without a ``signal_type`` field default to MOMENTUM and
        are blocked in the same way a typed MOMENTUM frame would be — the
        backward-compat path is no softer than the explicit one."""
        clock = _clock(hh=12, mm=0)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        outcome = router.handle_payload(_entry(ACTION_BUY))  # no signal_type
        assert outcome.reason == "TIME_GATE"
        assert router.stats.time_gate_blocks == 1


class TestSubmitPullback:
    """The programmatic entry point for the VWAP pullback strategy.

    Same gate as the sidecar path (inventory ceiling, OBI, risk), but a
    ``VWAP_PULLBACK`` signal_type is *not* time-gated by the router — the strategy
    itself only fires during MIDDAY_CHOP, and the router trusts the caller.
    """

    def test_paper_fill_priced_at_marketable_limit(self, tmp_path: Path) -> None:
        clock = _clock(hh=12, mm=0)
        book = _book(900.0, 100.0)  # OBI passes
        router = _router(
            tmp_path=tmp_path, book=book, settings=_settings(), clock=clock
        )

        async def _run() -> Any:
            return await router.submit_pullback(
                symbol=SYMBOL,
                side="BUY",
                instrument_token=TOKEN_INT,
                signal_price=100.0,
                ltp=100.5,
                atr_5m=1.0,
                vwap=100.0,
                band=99.5,
                std_dev=0.5,
                paper_trade=True,
            )

        outcome = asyncio.run(_run())
        assert outcome.event == "ENTRY"
        assert outcome.accepted is True
        assert outcome.mode == "PAPER"
        assert outcome.side == "BUY"
        # Paper fill is at the touch, not the signal's price.
        assert outcome.price == pytest.approx(book.ask)
        assert router.stats.paper_entries == 1
        assert router.stats.pullback_entries == 1

    def test_no_book_drops_the_pullback(self, tmp_path: Path) -> None:
        clock = _clock(hh=12, mm=0)
        router = _router(
            tmp_path=tmp_path, book=None, settings=_settings(), clock=clock
        )

        async def _run() -> Any:
            return await router.submit_pullback(
                symbol=SYMBOL,
                side="BUY",
                instrument_token=TOKEN_INT,
                signal_price=100.0,
                ltp=100.5,
                atr_5m=1.0,
                vwap=100.0,
                band=99.5,
                std_dev=0.5,
            )

        outcome = asyncio.run(_run())
        assert outcome.event == "DROPPED"
        assert outcome.reason == "NO_TOP_OF_BOOK"
        assert router.stats.pullback_entries == 1
        assert router.stats.paper_entries == 0

    def test_pullback_in_midday_chop_is_not_time_gated(self, tmp_path: Path) -> None:
        """A pullback submitted at 12:00 IST is allowed even though MOMENTUM is not.

        The strategy is exclusive to MIDDAY_CHOP, so by the time the router sees a
        ``submit_pullback`` call the strategy has already done its gating. The
        router does not add a second regime check on top.
        """
        clock = _clock(hh=12, mm=0)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )

        async def _run() -> Any:
            return await router.submit_pullback(
                symbol=SYMBOL,
                side="BUY",
                instrument_token=TOKEN_INT,
                signal_price=100.0,
                ltp=100.5,
                atr_5m=1.0,
                vwap=100.0,
                band=99.5,
                std_dev=0.5,
            )

        outcome = asyncio.run(_run())
        assert outcome.event == "ENTRY"
        assert router.stats.time_gate_blocks == 0


# ──────────────────────────────────────────────────────────────────────────────
# Risk:Reward gate (CLAUDE.md §6.1, Step 4) — INSUFFICIENT_RR hard-rejection
# when the projected target/stop gives worse than 1:1.
# ──────────────────────────────────────────────────────────────────────────────


def _high_rr_context(symbol: str = SYMBOL) -> Any:
    """Trade context that produces an RR well above 1.0.

    atr=30, HOD-LOD=10 → remaining=20 → atr_target = entry+20 = 120.
    structural = 120×0.9995 = 119.94. l2 = inf. min = 119.94.
    risk = 0.5×30 = 15. reward = 19.94. RR = 1.33.
    """
    from tachyon.math_engine.targets import TradeContext

    return TradeContext(
        entry=100.0,
        intraday_atr=30.0,
        hod=120.0,
        lod=110.0,
        ask_qty=(100, 0, 0, 0, 0),
        bid_qty=(100, 0, 0, 0, 0),
    )


def _low_rr_context(symbol: str = SYMBOL) -> Any:
    """Trade context that produces an RR well below 1.0."""
    from tachyon.math_engine.targets import TradeContext

    return TradeContext(
        entry=100.0,
        intraday_atr=10.0,
        hod=101.0,
        lod=99.0,
        ask_qty=(100, 0, 0, 0, 0),
        bid_qty=(100, 0, 0, 0, 0),
    )


def _context_provider(context: Any) -> Any:
    return lambda _symbol: context


class TestRiskRewardGate:
    """The RR gate is the layer above OBI, below the twelve-check risk engine."""

    def test_high_rr_attaches_target_and_stop(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _context_provider(_high_rr_context())  # type: ignore[attr-defined]

        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "ENTRY"
        # Realistic target > entry; stop < entry; both attached.
        assert outcome.target > 0.0
        assert outcome.stop > 0.0
        assert outcome.target > 100.0  # entry
        assert outcome.stop < 100.0
        assert router.stats.rr_blocks == 0

    def test_low_rr_is_hard_rejected(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _context_provider(_low_rr_context())  # type: ignore[attr-defined]

        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "REJECTED"
        assert outcome.reason == "INSUFFICIENT_RR"
        assert "projected RR" in outcome.detail
        assert router.stats.rr_blocks == 1
        assert router.stats.paper_entries == 0

    def test_missing_provider_is_soft_pass(self, tmp_path: Path) -> None:
        """No context provider wired → RR gate skipped, trade allowed.

        The alternative (refusing on "I cannot tell") would silently disable
        the strategy on the first 09:15 tick before the math engine has
        produced its first snapshot. Soft pass is the right default; tests
        that want a stricter policy inject a provider that returns a
        low-RR context.
        """
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        # No provider.
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "ENTRY"
        # No target attached when no provider — the gate ran but had no data.
        assert outcome.target == 0.0
        assert outcome.stop == 0.0
        assert router.stats.rr_blocks == 0

    def test_provider_failure_is_soft_pass(self, tmp_path: Path) -> None:
        """A provider that raises must not crash the gate."""

        def _bad(_symbol: str) -> Any:
            raise RuntimeError("aggregator offline")

        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _bad  # type: ignore[attr-defined]

        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "ENTRY"
        assert router.stats.rr_blocks == 0

    def test_unusable_context_is_soft_pass(self, tmp_path: Path) -> None:
        from tachyon.math_engine.targets import TradeContext

        # Zero ATR — the context is_usable() returns False.
        bad = TradeContext(
            entry=100.0, intraday_atr=0.0, hod=101.0, lod=99.0,
            ask_qty=(100, 0, 0, 0, 0), bid_qty=(100, 0, 0, 0, 0),
        )
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _context_provider(bad)  # type: ignore[attr-defined]

        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "ENTRY"
        assert router.stats.rr_blocks == 0

    def test_rr_blocks_are_in_stats_dict(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _context_provider(_low_rr_context())  # type: ignore[attr-defined]

        router.handle_payload(_entry_with_signal_type(ACTION_BUY, "MOMENTUM"))
        assert router.stats_dict()["rr_blocks"] == 1

    def test_dispatch_outcome_carries_target_through_sink_record(self, tmp_path: Path) -> None:
        """The trade sink must see target and stop so post-mortem can reconcile."""
        clock = _clock(hh=9, mm=30)
        router = _router(
            tmp_path=tmp_path,
            book=_book(900.0, 100.0),
            settings=_settings(),
            clock=clock,
        )
        router._trade_context_provider = _context_provider(_high_rr_context())  # type: ignore[attr-defined]

        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        record = outcome.sink_record(latency_ms=1.0, ts_epoch=0.0)
        assert "target" in record
        assert "stop" in record
        assert record["target"] > 0.0
        assert record["stop"] > 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Trade lifecycle integration (CLAUDE.md §6.1, Step 5) — entries register
# with the lifecycle manager; lifecycle evaluation promotes breakeven, kills
# stagnant trades, and trails stops on profit.
# ──────────────────────────────────────────────────────────────────────────────


def _entry_router_with_context(
    tmp_path: Path,
    *,
    clock: Any,
    context: Any,
) -> Any:
    """Build a router wired with a single-symbol context provider."""
    router = _router(
        tmp_path=tmp_path,
        book=_book(900.0, 100.0),
        settings=_settings(),
        clock=clock,
    )
    router._trade_context_provider = _context_provider(context)  # type: ignore[attr-defined]
    return router


class TestLifecycleRegistration:
    def test_entry_registers_active_trade(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )
        outcome = router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        assert outcome.event == "ENTRY"
        # The trade is registered with the lifecycle manager.
        from tachyon.execution.trade_manager import Side
        trade = router._lifecycle.get_by_symbol(SYMBOL)  # type: ignore[attr-defined]
        assert trade is not None
        assert trade.side is Side.LONG
        assert trade.quantity == 1
        assert trade.entry_price == pytest.approx(101.0)  # filled at ask
        assert trade.current_stop == pytest.approx(outcome.stop)

    def test_pullback_registers_active_trade(self, tmp_path: Path) -> None:
        clock = _clock(hh=12, mm=0)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )

        async def _run() -> Any:
            return await router.submit_pullback(
                symbol=SYMBOL,
                side="BUY",
                instrument_token=TOKEN_INT,
                signal_price=100.0,
                ltp=100.5,
                atr_5m=1.0,
                vwap=100.0,
                band=99.5,
                std_dev=0.5,
            )

        outcome = asyncio.run(_run())
        assert outcome.event == "ENTRY"
        trade = router._lifecycle.get_by_symbol(SYMBOL)  # type: ignore[attr-defined]
        assert trade is not None


class TestLifecycleBreakeven:
    def test_breakeven_promotion_via_router(self, tmp_path: Path) -> None:
        from tachyon.execution.trade_manager import ActiveTrade, ExitReason, Side
        import time as _time

        clock = _clock(hh=9, mm=30)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )
        router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        # Re-register the trade with an entry time well in the future (relative
        # to test time) so the time-stop branch does not fire — we only want
        # to assert the breakeven promotion.
        old = router._lifecycle.get_by_symbol(SYMBOL)  # type: ignore[attr-defined]
        router._lifecycle.deregister(old.trade_id)  # type: ignore[attr-defined]
        new = ActiveTrade(
            trade_id=old.trade_id,
            symbol=old.symbol,
            side=Side.LONG,
            quantity=old.quantity,
            entry_price=old.entry_price,
            entry_time=_time.time() + 1_000_000,  # well after 45 minutes
            initial_stop=old.initial_stop,
            initial_target=old.initial_target,
            current_stop=old.current_stop,
        )
        router._lifecycle.register(new)  # type: ignore[attr-defined]
        # entry=101 (book.ask fill), initial_risk=0.5*30=15 (target.stop=86,
        # entry=100, risk=16). LTP=111.7 gives r = 10.7/16 = 0.668. For
        # breakeven to fire we need r >= 0.7 → LTP >= 100 + 0.7*16 = 111.2.
        # Use LTP=113 to be safely above.
        intents = router.evaluate_lifecycle(_time.time() + 1_000_001, {SYMBOL: 113.0})
        assert any(i.reason is ExitReason.BREAKEVEN_PROMOTED for i in intents)
        assert router.stats.breakeven_promotions == 1

    def test_market_exit_removes_inventory(self, tmp_path: Path) -> None:
        from tachyon.execution.trade_manager import ActiveTrade, ExitReason, Side, TIME_STOP_SECONDS
        import time as _time

        clock = _clock(hh=9, mm=30)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )
        router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        # Re-register with an entry time well in the past so the time-stop fires.
        old = router._lifecycle.get_by_symbol(SYMBOL)  # type: ignore[attr-defined]
        router._lifecycle.deregister(old.trade_id)  # type: ignore[attr-defined]
        new = ActiveTrade(
            trade_id=old.trade_id,
            symbol=old.symbol,
            side=Side.LONG,
            quantity=old.quantity,
            entry_price=old.entry_price,
            entry_time=_time.time() - TIME_STOP_SECONDS - 60,
            initial_stop=old.initial_stop,
            initial_target=old.initial_target,
            current_stop=old.current_stop,
        )
        router._lifecycle.register(new)  # type: ignore[attr-defined]
        # r ≈ 0.07 → below 0.5 → TIME_STOP_STAGNANT
        intents = router.evaluate_lifecycle(_time.time(), {SYMBOL: 102.0})
        assert any(i.reason is ExitReason.TIME_STOP_STAGNANT for i in intents)
        assert router.stats.time_stop_stagnant_exits == 1
        # Inventory cleared.
        assert SYMBOL not in router.inventory

    def test_lifecycle_counters_in_stats_dict(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )
        router.handle_payload(
            _entry_with_signal_type(ACTION_BUY, "MOMENTUM")
        )
        d = router.stats_dict()
        assert "breakeven_promotions" in d
        assert "time_stop_stagnant_exits" in d
        assert "time_stop_profit_chokes" in d

    def test_lifecycle_evaluate_on_empty_manager(self, tmp_path: Path) -> None:
        clock = _clock(hh=9, mm=30)
        router = _entry_router_with_context(
            tmp_path, clock=clock, context=_high_rr_context()
        )
        # No active trades.
        intents = router.evaluate_lifecycle(0.0, {SYMBOL: 100.0})
        assert intents == []
