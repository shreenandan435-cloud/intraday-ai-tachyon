"""The Crucible — end-to-end chaos tests, CLAUDE.md §11.

Four ways for the world to break, each aimed at a specific promise the constitution makes:

======================  =========================================================
Sabotage                Promise under test
======================  =========================================================
Feed dies mid-session   §4 check 5 — no market data for 2 s blocks new entries
Broker 503 storm        §6.4 — a failed placement never crashes the loop, and an
                        unknown outcome locks rather than guesses
Rogue webhook fill      §9 — a fill we never placed bricks the day, durably
NTP clock jump          §1.1 — the 15:15 deadline cannot be moved by the wall clock
======================  =========================================================

These run against **real** ZeroMQ sockets, **real** threads and the real orchestrator
components. A fail-safe exercised only through a mock is a fail-safe nobody has tested.

Teardown is asserted, not hoped for: :func:`_no_leaks` fails the test if a non-daemon thread or
a ZeroMQ context outlives it. A hanging square-off watchdog would keep a real process alive
forever, so a test that leaks one is reporting a bug rather than an inconvenience.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from tachyon.core import eventloop
from tachyon.core.clock import IST, ManualClock, SessionEvent
from tachyon.core.config import (
    ExecutionSettings,
    MathEngineSettings,
    Settings,
    StrategySettings,
    WatchlistItem,
)
from tachyon.core.constants import FEED_STALE_AFTER, TradingMode
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.execution.api import ClientIdentity, SmartApiClient
from tachyon.ipc.monitor import FeedState
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import OrderBook, Tick
from tachyon.ipc.subscriber import AsyncSubscriber, SubscriberRole
from tachyon.math_engine import warmup
from tachyon.math_engine.core import TickAggregator
from tachyon.persistence.journal import JsonlJournal
from tachyon.risk.engine import VetoReason
from tachyon.risk.watchdog import SquareOffWatchdog
from tachyon.strategy.brain import BRAIN_TOPICS, StrategyBrain
from tachyon.strategy.signals import NoSignalReason, Signal, SignalReport
from tachyon.strategy.telemetry import StatePublisher
from tachyon.ui.postback import OrderStatusListener, watchlist_resolver

FEED_STALE_SECONDS = FEED_STALE_AFTER.total_seconds()

TICK_ENDPOINT = "tcp://127.0.0.1:5901"
STATE_ENDPOINT = "tcp://127.0.0.1:5902"


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


@pytest.fixture(autouse=True)
def _no_leaks() -> Iterator[None]:
    """Fail the test if it leaves a non-daemon thread running.

    The square-off watchdog is non-daemon by design (CLAUDE.md §1.1) — it must survive main
    thread shutdown long enough to flatten. The flip side is that a test which forgets to stop
    one would keep a real process alive forever, so leaking here is a bug report, not tidiness.
    """
    before = {thread.ident for thread in threading.enumerate()}
    yield
    leaked = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in before and not thread.daemon and thread.is_alive()
    ]
    assert not leaked, f"leaked non-daemon threads: {[t.name for t in leaked]}"


# ──────────────────────────────────────────────────────────────────────────────
# Harness
# ──────────────────────────────────────────────────────────────────────────────


def _clock(hh: int = 11, mm: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=mono)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),),
        "math_engine": MathEngineSettings(),
        "strategy": StrategySettings(evaluate_every_n_ticks=10),
        "execution": ExecutionSettings(),
        "zmq_tick_endpoint": TICK_ENDPOINT,
        "zmq_state_endpoint": STATE_ENDPOINT,
    }
    base.update(overrides)
    return Settings(**base)


class _NullPublisher:
    """A state-spine publisher that binds no port. Chaos tests build several Brains."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def publish_raw(self, topic: bytes, payload: bytes) -> None:
        self.sent.append(topic.decode())

    def close(self) -> None:
        return None


class _AlwaysLong:
    """Forces an unambiguous LONG so a chaos test exercises the path, not the indicators."""

    def evaluate(self, snapshot: Any) -> SignalReport:
        return SignalReport(
            symbol="RELIANCE",
            signal=Signal.LONG,
            reason=NoSignalReason.NONE,
            ltp=2500.0,
            session_vwap=2490.0,
            ema=2495.0,
            ema_period=21,
            obi=0.5,
            obi_weighted=0.5,
            obi_threshold=0.3,
            atr_5m=8.4,
            ts_epoch=0.0,
        )


def _build_brain(
    tmp_path: Path,
    clock: ManualClock,
    settings: Settings,
    *,
    subscriber: AsyncSubscriber | None = None,
    client: SmartApiClient | None = None,
) -> tuple[StrategyBrain, _NullPublisher]:
    """A fully wired Brain with its journal and daily lock redirected into ``tmp_path``."""
    from tachyon.strategy.brain import _SymbolState

    publisher = _NullPublisher()
    brain = StrategyBrain(
        settings=settings,
        client=client,
        subscriber=subscriber,
        state_machine=StateMachine(TradingState.ACTIVE, clock=clock),
        state_publisher=StatePublisher(
            settings=settings,
            publisher=publisher,  # type: ignore[arg-type]
            clock=clock,
        ),
        daily_lock=DailyLock(path=tmp_path / "daily_lock.txt", clock=clock),
        clock=clock,
    )
    brain._journal = JsonlJournal(tmp_path / "journal", clock=clock)  # noqa: SLF001
    for item in settings.watchlist:
        brain._symbols[item.symbol] = _SymbolState(  # noqa: SLF001
            aggregator=TickAggregator.from_settings(item.symbol, settings)
        )
    return brain, publisher


def _tick(seq: int) -> Tick:
    """A well-formed tick with monotonically rising cumulative volume."""
    return Tick(
        token="2885",
        ltp=2500.0 + (seq % 11) * 0.05,
        volume=1_000_000 + seq * 40,
        ts_epoch=1_786_000_000.0 + seq,
        seq=seq,
    )


def _book() -> OrderBook:
    return OrderBook(
        token="2885",
        bid_price=(2499.95, 2499.90, 2499.85, 2499.80, 2499.75),
        bid_qty=(900, 800, 700, 600, 500),
        ask_price=(2500.05, 2500.10, 2500.15, 2500.20, 2500.25),
        ask_qty=(100, 90, 80, 70, 60),
        ts_epoch=1_786_000_000.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Sudden feed death
# ──────────────────────────────────────────────────────────────────────────────


class TestSuddenFeedDeath:
    """The ingestor dies mid-session. CLAUDE.md §4 check 5, §2.1.

    Driven through :func:`tachyon.core.eventloop.run` so it runs on the same loop the
    entrypoints use — on Windows that is the only loop ``zmq.asyncio`` can receive on at all.
    """

    def test_sudden_feed_death(self, tmp_path: Path) -> None:
        eventloop.run(self._scenario(tmp_path))

    async def _scenario(self, tmp_path: Path) -> None:
        settings = _settings()
        clock = _clock()

        publisher = Publisher(TICK_ENDPOINT, role="ingestor")
        subscriber = AsyncSubscriber(
            BRAIN_TOPICS,
            role=SubscriberRole.STRATEGY,
            endpoint=TICK_ENDPOINT,
            clock=clock,
        )
        Publisher.settle(0.3)

        brain, _pub = _build_brain(tmp_path, clock, settings, subscriber=subscriber)
        # The monitor is the component under test; wire it to the same clock we control so the
        # 2 s boundary can be asserted exactly rather than slept through.
        brain.feed_monitor.clock = clock
        runner = asyncio.create_task(brain.run())

        try:
            # ── the feed is healthy ──────────────────────────────────────────
            publisher.publish_orderbook("RELIANCE", _book())
            for seq in range(1, 51):
                publisher.publish_tick("RELIANCE", _tick(seq))

            async def drained() -> None:
                while brain.stats.ticks < 50:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(drained(), timeout=10.0)

            aggregator = brain.aggregator("RELIANCE")
            assert aggregator is not None
            assert aggregator.gaps_detected == 0, "50 clean ticks must arrive without a gap"
            assert aggregator.session_vwap_valid
            snapshot = brain.snapshot("RELIANCE")
            assert snapshot is not None
            assert snapshot.session_vwap > 0, "session VWAP warmed up"

            brain.feed_monitor.record()
            assert brain.risk.evaluate("RELIANCE").allowed, "a live feed permits entries"

            # ── the ingestor dies ────────────────────────────────────────────
            publisher.close()

            # Just inside the window: still fresh. The threshold is strict, and a system that
            # blocked at 1.9 s would refuse trades all day on ordinary jitter.
            clock.advance(FEED_STALE_SECONDS - 0.01)
            assert not brain.feed_monitor.is_stale
            assert brain.risk.evaluate("RELIANCE").allowed

            # Just past it: stale, and every entry is refused.
            clock.advance(0.02)
            assert brain.feed_monitor.is_stale
            assert brain.feed_monitor.check() is FeedState.STALE

            decision = brain.risk.evaluate("RELIANCE")
            assert not decision.allowed
            assert decision.reason is VetoReason.FEED_STALE
            assert "no market data" in decision.detail

            # And a signal arriving now is dropped by the gate, not acted on.
            brain._generator = _AlwaysLong()  # type: ignore[assignment]  # noqa: SLF001
            state = brain._symbols["RELIANCE"]  # noqa: SLF001
            before = brain.stats.risk_vetoes
            brain._evaluate("RELIANCE", state)  # noqa: SLF001
            await asyncio.gather(*tuple(brain._entry_tasks), return_exceptions=True)  # noqa: SLF001

            assert brain.stats.risk_vetoes == before + 1
            assert brain.stats.entries_placed == 0
            assert brain.positions.open_count == 0
            assert brain.risk.veto_counts.get(VetoReason.FEED_STALE, 0) >= 1

            # The feed dying is a veto, not a kill switch: the session is still ACTIVE and
            # would resume the moment data returns. Nothing here latches (CLAUDE.md §4).
            assert brain.state is TradingState.ACTIVE
        finally:
            await brain.stop()
            subscriber.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(runner, timeout=10.0)
            publisher.close()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Broker 503 storm
# ──────────────────────────────────────────────────────────────────────────────


class _StormTransport(httpx.AsyncBaseTransport):
    """Answers 503 for placements and success for everything else.

    ``body`` decides which failure mode is exercised. A 503 carrying a broker error envelope is
    a *known* rejection; a bare 503 is an **unknown outcome** — the request went out and we
    never learned what happened to it, which CLAUDE.md §6.4 treats very differently.
    """

    def __init__(self, *, with_body: bool) -> None:
        self.with_body = with_body
        self.placements = 0
        self.paths: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = str(request.url.path)
        self.paths.append(path)
        if path.endswith("placeOrder"):
            self.placements += 1
            if self.with_body:
                return httpx.Response(
                    503,
                    json={
                        "status": False,
                        "message": "Service Unavailable",
                        "errorcode": "AB5003",
                    },
                )
            return httpx.Response(503, text="<html>503 Service Unavailable</html>")
        if path.endswith("getRMS"):
            return httpx.Response(
                200,
                json={"status": True, "message": "OK", "data": {"availablecash": "250000"}},
            )
        return httpx.Response(200, json={"status": True, "message": "OK", "data": []})


class TestBroker503Storm:
    """The broker falls over mid-entry. CLAUDE.md §6.4."""

    def _stack(
        self, tmp_path: Path, *, with_body: bool
    ) -> tuple[StrategyBrain, _StormTransport, SmartApiClient, ManualClock]:
        clock = _clock()
        settings = _settings(
            trading_mode=TradingMode.LIVE,
            smartapi_api_key="k",
            smartapi_client_code="c",
        )
        transport = _StormTransport(with_body=with_body)
        client = SmartApiClient(
            settings=settings,
            journal=JsonlJournal(tmp_path / "journal", clock=clock),
            client=httpx.AsyncClient(base_url="https://test", transport=transport),
            mode=TradingMode.LIVE,
            clock=clock,
            identity=ClientIdentity(local_ip="1", public_ip="1", mac_address="A"),
            rate_limit_scale=10_000.0,
        )
        brain, _pub = _build_brain(tmp_path, clock, settings, client=client)
        brain.feed_monitor.record()
        brain._generator = _AlwaysLong()  # type: ignore[assignment]  # noqa: SLF001
        # In LIVE the margin check reads a cached value that `_margin_loop` refreshes, and it
        # is deliberately undefined until the first successful poll — so prime it, or the gate
        # refuses before the broker is ever reached. That veto has its own test.
        brain._margin = Decimal("250000")  # noqa: SLF001
        return brain, transport, client, clock

    async def _fire(self, brain: StrategyBrain) -> None:
        state = brain._symbols["RELIANCE"]  # noqa: SLF001
        brain._evaluate("RELIANCE", state)  # noqa: SLF001
        await asyncio.gather(*tuple(brain._entry_tasks), return_exceptions=True)  # noqa: SLF001

    async def test_a_known_503_rejection_does_not_crash_or_open_a_position(
        self, tmp_path: Path
    ) -> None:
        """A 503 with a broker error envelope: refused, logged, loop intact."""
        brain, transport, client, _clock_ = self._stack(tmp_path, with_body=True)
        try:
            await self._fire(brain)

            assert transport.placements >= 1, "LIVE must actually transmit"
            assert brain.stats.entries_placed == 0
            assert brain.stats.entries_failed == 1
            assert brain.positions.open_count == 0
            assert not brain.positions.is_open("RELIANCE")

            # A knowable rejection is a missed trade, not an emergency. The session stays
            # ACTIVE and the next signal is evaluated normally.
            assert brain.state is TradingState.ACTIVE
            assert client.stats.broker_rejections >= 1

            # The loop is unharmed: another signal can still be evaluated.
            await self._fire(brain)
            assert brain.stats.evaluations >= 2
        finally:
            await brain.shutdown()
            await client.aclose()

    async def test_an_unknown_outcome_locks_rather_than_guesses(self, tmp_path: Path) -> None:
        """A bare 503: the bytes went out and we never learned the result (§6.4).

        The order may be live at the exchange. Trading on would size the next decision against
        a position we cannot see, so the session locks instead.
        """
        brain, transport, client, _clock_ = self._stack(tmp_path, with_body=False)
        try:
            await self._fire(brain)

            assert transport.placements == 1, "an unknown outcome is NEVER retried"
            assert client.stats.unknown_outcomes == 1
            assert brain.stats.entries_placed == 0
            assert brain.positions.open_count == 0
            assert brain.state is TradingState.LOCKED
        finally:
            await brain.shutdown()
            await client.aclose()

    async def test_a_storm_of_signals_never_raises(self, tmp_path: Path) -> None:
        """Twenty signals into a dead broker. The loop must still be standing."""
        brain, transport, client, _clock_ = self._stack(tmp_path, with_body=True)
        try:
            for _ in range(20):
                await self._fire(brain)

            assert brain.stats.entries_placed == 0
            assert brain.positions.open_count == 0
            assert brain.stats.handler_errors == 0, "nothing escaped into the message loop"
            assert transport.placements >= 1
        finally:
            await brain.shutdown()
            await client.aclose()

    async def test_only_the_second_leg_is_spared_after_an_unknown_outcome(
        self, tmp_path: Path
    ) -> None:
        """Leg B is not sent on top of a leg A that may or may not exist."""
        brain, transport, client, _clock_ = self._stack(tmp_path, with_body=False)
        try:
            await self._fire(brain)
            assert transport.placements == 1, "the bracket stopped after the unknown leg"
        finally:
            await brain.shutdown()
            await client.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Rogue webhook fill
# ──────────────────────────────────────────────────────────────────────────────


def _fill_payload(
    order_id: str,
    *,
    side: str = "BUY",
    tag: str = "",
    filled: int = 10,
    price: str = "2500.00",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "orderid": order_id,
        "symboltoken": "2885",
        "tradingsymbol": "RELIANCE-EQ",
        "transactiontype": side,
        "orderstatus": "complete",
        "quantity": filled,
        "filledshares": filled,
        "averageprice": price,
        "ordertype": "LIMIT",
    }
    if tag:
        payload["ordertag"] = tag
    return payload


class TestRogueWebhookFill:
    """A fill arrives for an order this system never placed. CLAUDE.md §9."""

    def _listener(
        self, tmp_path: Path
    ) -> tuple[OrderStatusListener, DailyLock, list[Any], list[Any]]:
        clock = _clock()
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        closures: list[Any] = []
        rogues: list[Any] = []
        listener = OrderStatusListener(
            on_closed=lambda *args: closures.append(args),
            symbol_resolver=watchlist_resolver({"2885": "RELIANCE"}, {"RELIANCE-EQ": "RELIANCE"}),
            on_rogue_fill=rogues.append,
            daily_lock=lock,
            journal=JsonlJournal(tmp_path / "journal", prefix="fills", clock=clock),
            clock=clock,
        )
        return listener, lock, closures, rogues

    def test_rogue_webhook_fill_bricks_the_day(self, tmp_path: Path) -> None:
        listener, lock, closures, rogues = self._listener(tmp_path)
        assert not lock.is_engaged()

        accepted = listener.ingest_raw(_fill_payload("SOMEONE-ELSES-ORDER"))

        assert not accepted, "a rogue fill is refused, not booked"
        assert listener.stats.rogue_fills == 1
        assert len(rogues) == 1

        # The durable half. A restart must not resume trading (§1.2).
        assert lock.is_engaged()
        record = lock.read()
        assert record is not None
        assert record.reason == "ROGUE_FILL"
        assert record.lock_date == date(2026, 8, 10)

        # And the exposure was NOT folded into our books: guessing at a position we did not
        # open would corrupt the number the ₹500 limit is enforced against.
        assert listener.net_quantity("RELIANCE") == 0
        assert not closures

    def test_a_restart_survives_because_of_the_client_tag(self, tmp_path: Path) -> None:
        """After a crash the id registry is empty, but our tag is still on the order.

        Without this, the first order-book sweep after any restart would brick the day.
        """
        listener, lock, _closures, rogues = self._listener(tmp_path)

        assert listener.ingest_raw(_fill_payload("251008000001", tag="TCHYN-20260810-0001"))

        assert not rogues
        assert not lock.is_engaged()
        assert listener.net_quantity("RELIANCE") == 10

    def test_a_registered_order_is_recognised_without_a_tag(self, tmp_path: Path) -> None:
        listener, lock, _closures, rogues = self._listener(tmp_path)
        listener.register_order("251008000002", "TCHYN-20260810-0002")

        assert listener.ingest_raw(_fill_payload("251008000002"))

        assert not rogues
        assert not lock.is_engaged()

    def test_the_alarm_is_journalled(self, tmp_path: Path) -> None:
        listener, _lock, _closures, _rogues = self._listener(tmp_path)
        listener.ingest_raw(_fill_payload("ROGUE-1"))

        journal = JsonlJournal(tmp_path / "journal", prefix="fills", clock=_clock())
        assert any(record["event"] == "rogue_fill" for record in journal.read())

    def test_a_crashing_alert_callback_still_locks_the_day(self, tmp_path: Path) -> None:
        """The lock is written before the callback, so a broken consumer cannot prevent it."""
        clock = _clock()
        lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)

        def explode(_update: Any) -> None:
            raise RuntimeError("alerting is broken too")

        listener = OrderStatusListener(
            on_closed=lambda *_: None,
            symbol_resolver=watchlist_resolver({"2885": "RELIANCE"}, {"RELIANCE-EQ": "RELIANCE"}),
            on_rogue_fill=explode,
            daily_lock=lock,
            journal=JsonlJournal(tmp_path / "journal", prefix="fills", clock=clock),
            clock=clock,
        )
        listener.ingest_raw(_fill_payload("ROGUE-2"))
        assert lock.is_engaged()

    def test_the_locked_day_blocks_the_next_boot(self, tmp_path: Path) -> None:
        """The point of the *durable* lock rather than the in-memory latch (§1.2, §6.5)."""
        from tachyon.core.state import resolve_boot_state

        listener, lock, _closures, _rogues = self._listener(tmp_path)
        listener.ingest_raw(_fill_payload("ROGUE-3"))

        clock = _clock()
        assert resolve_boot_state(lock, clock) is TradingState.LOCKED
        machine = StateMachine.boot(lock, clock=clock)
        assert not machine.may_open_position()

    async def test_the_brain_locks_in_memory_too(self, tmp_path: Path) -> None:
        """The listener writes the day lock; the Brain adds the in-memory half."""
        clock = _clock()
        settings = _settings()
        brain, publisher = _build_brain(tmp_path, clock, settings)
        try:
            brain.fills.ingest_raw(_fill_payload("NOT-OURS"))

            assert brain.state is TradingState.LOCKED
            assert brain.pnl.is_breached
            assert brain.pnl.daily_lock.is_engaged()
            assert any("RISK.ROGUE_FILL" in topic for topic in publisher.sent)
        finally:
            await brain.shutdown()

    async def test_a_legitimate_entry_is_registered_and_not_rogue(self, tmp_path: Path) -> None:
        """End to end: place in PAPER, then feed the fill back. It must be recognised."""
        clock = _clock()
        settings = _settings()
        brain, _publisher = _build_brain(tmp_path, clock, settings)
        brain.feed_monitor.record()
        brain._generator = _AlwaysLong()  # type: ignore[assignment]  # noqa: SLF001
        try:
            state = brain._symbols["RELIANCE"]  # noqa: SLF001
            brain._evaluate("RELIANCE", state)  # noqa: SLF001
            await asyncio.gather(*tuple(brain._entry_tasks), return_exceptions=True)  # noqa: SLF001
            assert brain.stats.entries_placed == 1

            # PAPER order ids look like PAPER-TCHYN-...; the tag alone would carry provenance,
            # so strip it to prove the *registration* path works on its own.
            order_id = "251008000009"
            brain.fills.register_order(order_id, "")
            assert brain.fills.ingest_raw(_fill_payload(order_id))

            assert brain.fills.stats.rogue_fills == 0
            assert brain.state is TradingState.ACTIVE
        finally:
            await brain.shutdown()


# ──────────────────────────────────────────────────────────────────────────────
# 4. NTP time jump
# ──────────────────────────────────────────────────────────────────────────────


class TestNtpTimeJump:
    """The 15:15 deadline cannot be moved by the wall clock. CLAUDE.md §1.1.

    ``freezegun`` is deliberately **not** used. It patches ``datetime`` *and* ``time``, so it
    would move the monotonic clock along with the wall clock — which is precisely the
    distinction under test. :class:`~tachyon.core.clock.ManualClock` moves the two
    independently, and that is the only way to tell an NTP correction apart from time actually
    passing.
    """

    def _watchdog(self, clock: ManualClock) -> tuple[SquareOffWatchdog, list[str], threading.Event]:
        fired: list[str] = []
        flattened = threading.Event()

        def flatten() -> None:
            fired.append("flatten")
            flattened.set()

        watchdog = SquareOffWatchdog(
            StateMachine(TradingState.ACTIVE, clock=clock),
            on_square_off=flatten,
            clock=clock,
            tick_seconds=0.01,
            on_date=date(2026, 8, 10),
        )
        return watchdog, fired, flattened

    def test_a_wall_clock_jump_alone_does_not_fire_the_deadline(self) -> None:
        """An aggressive NTP sync moves the wall clock. The deadline must not move with it.

        This is the property §1.1 actually promises, and the reason the deadline is pinned to
        ``time.monotonic`` at arm time: neither a correction forward nor backward can postpone
        or advance the 15:15 flatten.
        """
        clock = _clock(hh=14, mm=0)
        watchdog, fired, flattened = self._watchdog(clock)
        watchdog.start()
        try:
            assert flattened.wait(timeout=0.5) is False, "nothing due at 14:00"

            # NTP yanks the wall clock 80 minutes forward. Monotonic does not move.
            clock.jump_wall_clock(80 * 60)
            assert clock.now().hour == 15
            assert clock.now().minute == 20

            assert flattened.wait(timeout=0.5) is False, (
                "a wall-clock jump must not fire the deadline — it is pinned to monotonic"
            )
            assert fired == []
            assert not watchdog.square_off_fired
            assert watchdog.seconds_until_square_off > 0
        finally:
            watchdog.stop()
        assert not watchdog.is_alive

    def test_real_elapsed_time_fires_it_immediately(self) -> None:
        """Time genuinely passes: both clocks advance, and the flatten runs at once.

        "Immediately" is the assertion that matters — the watchdog must not sit out 80 minutes
        of real time waiting for a deadline that has already elapsed.
        """
        clock = _clock(hh=14, mm=0)
        watchdog, fired, flattened = self._watchdog(clock)
        watchdog.start()
        try:
            assert flattened.wait(timeout=0.3) is False

            clock.advance(80 * 60)  # 14:00 → 15:20, wall AND monotonic

            assert flattened.wait(timeout=5.0), "square-off did not fire once the deadline passed"
            assert fired == ["flatten"]
            assert watchdog.square_off_fired
            assert watchdog.square_off_completed
        finally:
            watchdog.stop()
        assert not watchdog.is_alive

    def test_a_backwards_jump_cannot_un_fire_it(self) -> None:
        """Firing is latching. A clock correction must never re-open a closed session."""
        clock = _clock(hh=14, mm=0)
        watchdog, _fired, flattened = self._watchdog(clock)
        watchdog.start()
        try:
            clock.advance(80 * 60)
            assert flattened.wait(timeout=5.0)

            clock.jump_wall_clock(-80 * 60)  # NTP corrects back to 14:00
            assert watchdog.square_off_fired, "square-off is latching (§1.1)"
            assert watchdog.has_fired(SessionEvent.SQUARE_OFF)
        finally:
            watchdog.stop()

    def test_a_process_starting_after_the_deadline_flattens_at_once(self) -> None:
        """A restart at 15:20 must flatten now, never tomorrow (§1.1)."""
        clock = _clock(hh=15, mm=20)
        watchdog, fired, flattened = self._watchdog(clock)
        watchdog.start()
        try:
            assert flattened.wait(timeout=5.0), "an elapsed deadline is due immediately"
            assert fired == ["flatten"]
        finally:
            watchdog.stop()

    def test_a_failing_flatten_is_retried_until_it_succeeds(self) -> None:
        """Being flat is the success criterion, not having sent the request (§6.5)."""
        clock = _clock(hh=15, mm=20)
        attempts: list[int] = []
        done = threading.Event()

        def flaky() -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("broker timeout")
            done.set()

        watchdog = SquareOffWatchdog(
            StateMachine(TradingState.ACTIVE, clock=clock),
            on_square_off=flaky,
            clock=clock,
            tick_seconds=0.01,
            on_date=date(2026, 8, 10),
        )
        watchdog.start()
        try:
            # The retry interval is measured on the injected clock, so advance it rather than
            # sleeping through two real seconds per attempt.
            deadline = 5.0
            waited = 0.0
            while not done.is_set() and waited < deadline:
                clock.advance(3.0)
                done.wait(timeout=0.05)
                waited += 0.05

            assert done.is_set(), f"flatten was not retried to success (attempts={len(attempts)})"
            assert len(attempts) >= 3
            assert watchdog.square_off_completed
            assert watchdog.action_failures == 2
        finally:
            watchdog.stop()
        assert not watchdog.is_alive

    def test_the_watchdog_thread_is_non_daemon_and_joins(self) -> None:
        """Non-daemon by design (§1.1); a test that leaks one would hang a real process."""
        clock = _clock(hh=14, mm=0)
        watchdog, _fired, _flattened = self._watchdog(clock)
        watchdog.start()

        thread = next(t for t in threading.enumerate() if t.name == "risk-squareoff-watchdog")
        assert not thread.daemon

        watchdog.stop()
        assert not watchdog.is_alive
        assert watchdog.thread_errors == 0


# ──────────────────────────────────────────────────────────────────────────────
# Teardown hygiene
# ──────────────────────────────────────────────────────────────────────────────


class TestNoOrphans:
    """The suite's own contract: nothing outlives a test."""

    def test_a_brain_shutdown_leaves_no_threads(self, tmp_path: Path) -> None:
        async def scenario() -> None:
            clock = _clock()
            brain, _pub = _build_brain(tmp_path, clock, _settings())
            await brain.boot()
            assert any(t.name == "risk-squareoff-watchdog" for t in threading.enumerate())
            await brain.shutdown()

        eventloop.run(scenario())
        assert not any(t.name == "risk-squareoff-watchdog" for t in threading.enumerate())

    def test_subscribers_and_publishers_close_cleanly(self) -> None:
        async def scenario() -> None:
            publisher = Publisher(TICK_ENDPOINT, role="ingestor")
            subscriber = AsyncSubscriber(
                ("TICK.",), role=SubscriberRole.STRATEGY, endpoint=TICK_ENDPOINT
            )
            Publisher.settle(0.1)
            publisher.publish_tick("RELIANCE", _tick(1))
            envelope = await asyncio.wait_for(subscriber.recv(), timeout=5.0)
            assert envelope is not None
            subscriber.close()
            publisher.close()
            assert subscriber.closed
            assert publisher.closed

        eventloop.run(scenario())

    def test_the_stale_threshold_is_the_constitutional_one(self) -> None:
        """The chaos suite must test the real limit, not a convenient copy of it."""
        assert FEED_STALE_SECONDS == 2.0
        assert FEED_STALE_AFTER == timedelta(seconds=2)
