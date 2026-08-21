"""Phase 8 integration — the Brain against a real ZeroMQ tick spine, CLAUDE.md §2.

The unit tests drive :meth:`StrategyBrain.on_message` directly, which proves the routing and
the decision chain but says nothing about whether ``zmq.asyncio`` actually delivers on this
platform. That is worth proving separately: the Brain's entire non-blocking design rests on it,
the Windows event loop is a Proactor loop, and pyzmq's asyncio integration behaves differently
there than on POSIX.

So these tests bind a real publisher, connect a real :class:`AsyncSubscriber`, and run the
actual loop:

* every published tick arrives, in order, with no gaps — a gap would void the session VWAP
  for the rest of the day (CLAUDE.md §3.3);
* the loop keeps draining while a deliberately slow order is in flight;
* shutdown is clean, including the non-daemon watchdog thread.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tachyon.core import eventloop
from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import ExecutionSettings, Settings, StrategySettings, WatchlistItem
from tachyon.core.state import StateMachine, TradingState
from tachyon.execution.executor import ExecutionReport, LegResult
from tachyon.execution.journal import OrderJournal
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import OrderBook, Tick
from tachyon.ipc.subscriber import AsyncSubscriber, SubscriberRole
from tachyon.math_engine import warmup
from tachyon.math_engine.core import TickAggregator
from tachyon.strategy.brain import BRAIN_TOPICS, StrategyBrain

ENDPOINT = "tcp://127.0.0.1:5793"


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock() -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, 11, 0, tzinfo=IST), mono=1000.0)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),),
        "strategy": StrategySettings(evaluate_every_n_ticks=10),
        "execution": ExecutionSettings(),
        "zmq_tick_endpoint": ENDPOINT,
    }
    base.update(overrides)
    return Settings(**base)


class _SlowExecutor:
    """Stands in for the broker, taking a configurable and deliberately long time."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def place_robo_order(self, plan: Any, decision: Any) -> ExecutionReport:
        await asyncio.sleep(self.delay)
        self.calls += 1
        return ExecutionReport(
            symbol=plan.symbol,
            side=plan.side,
            at_ist="now",
            plan=plan,
            results=tuple(
                LegResult(
                    leg=leg.leg,
                    order_tag=leg.order_tag,
                    quantity=leg.quantity,
                    accepted=True,
                    order_id="INT",
                )
                for leg in plan.legs
            ),
            simulated=True,
        )

    def square_off_action(self, loop: Any, timeout: float = 20.0) -> Any:
        return lambda: None


def _build_brain(tmp_path: Path, subscriber: AsyncSubscriber, settings: Settings) -> StrategyBrain:
    from tachyon.strategy.brain import _SymbolState

    clock = _clock()
    brain = StrategyBrain(
        settings=settings,
        client=None,
        subscriber=subscriber,
        state_machine=StateMachine(TradingState.ACTIVE, clock=clock),
        clock=clock,
    )
    brain._journal = OrderJournal(tmp_path / "journal", clock=clock)  # noqa: SLF001
    lock_type = type(brain._pnl.daily_lock)  # noqa: SLF001
    brain._pnl.daily_lock = lock_type(path=tmp_path / "daily_lock.txt", clock=clock)  # noqa: SLF001
    for item in settings.watchlist:
        brain._symbols[item.symbol] = _SymbolState(  # noqa: SLF001
            aggregator=TickAggregator.from_settings(item.symbol, settings)
        )
    return brain


@pytest.fixture
def spine() -> Any:
    """A bound publisher and a connected async subscriber, in that order.

    Binding first matters: ZeroMQ's slow-joiner problem means a SUB that connects before the
    PUB binds silently loses the first messages, which reads exactly like a broken decoder.
    """
    publisher = Publisher(ENDPOINT, role="test")
    subscriber = AsyncSubscriber(BRAIN_TOPICS, role=SubscriberRole.STRATEGY, endpoint=ENDPOINT)
    Publisher.settle(0.3)
    try:
        yield publisher, subscriber
    finally:
        subscriber.close()
        publisher.close()


def _tick(seq: int) -> Tick:
    return Tick(
        token="2885",
        ltp=2500.0 + (seq % 7) * 0.05,
        volume=1_000_000 + seq * 25,
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


class TestBrainOverRealSpine:
    """Driven through :func:`tachyon.core.eventloop.run`, not pytest-asyncio's default loop.

    That is the point of these tests. pytest-asyncio would give us the stdlib loop, and on
    Windows that is the *Proactor* loop, on which ``zmq.asyncio`` cannot receive at all. Running
    the same loop factory the entrypoints use is what makes this an integration test rather
    than a second set of unit tests.
    """

    def test_every_published_tick_is_consumed_in_order(self, tmp_path: Path, spine: Any) -> None:
        eventloop.run(self._test_every_published_tick_is_consumed_in_order(tmp_path, spine))

    def test_ingestion_continues_while_an_order_is_in_flight(
        self, tmp_path: Path, spine: Any
    ) -> None:
        eventloop.run(self._test_ingestion_continues_while_an_order_is_in_flight(tmp_path, spine))

    def test_shutdown_stops_the_watchdog_thread(self, tmp_path: Path, spine: Any) -> None:
        eventloop.run(self._test_shutdown_stops_the_watchdog_thread(tmp_path, spine))

    def test_a_locked_session_consumes_ticks_but_places_nothing(
        self, tmp_path: Path, spine: Any
    ) -> None:
        eventloop.run(
            self._test_a_locked_session_consumes_ticks_but_places_nothing(tmp_path, spine)
        )

    async def _test_every_published_tick_is_consumed_in_order(
        self, tmp_path: Path, spine: Any
    ) -> None:
        publisher, subscriber = spine
        settings = _settings()
        brain = _build_brain(tmp_path, subscriber, settings)

        total = 300
        runner = asyncio.create_task(brain.run())
        try:
            publisher.publish_orderbook("RELIANCE", _book())
            for seq in range(1, total + 1):
                publisher.publish_tick("RELIANCE", _tick(seq))
                if seq % 50 == 0:
                    await asyncio.sleep(0)  # let the loop drain

            async def drained() -> None:
                while brain.stats.ticks < total:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(drained(), timeout=10.0)
        finally:
            await brain.stop()
            subscriber.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(runner, timeout=10.0)

        assert brain.stats.ticks == total
        aggregator = brain.aggregator("RELIANCE")
        assert aggregator is not None
        # A gap would have been logged CRITICAL and voided the session VWAP permanently.
        assert aggregator.gaps_detected == 0
        assert aggregator.session_vwap_valid

        snapshot = brain.snapshot("RELIANCE")
        assert snapshot is not None
        assert snapshot.obi > 0  # the depth frame arrived too

    async def _test_ingestion_continues_while_an_order_is_in_flight(
        self, tmp_path: Path, spine: Any
    ) -> None:
        """The property the whole async design exists for."""
        publisher, subscriber = spine
        settings = _settings()
        brain = _build_brain(tmp_path, subscriber, settings)

        slow = _SlowExecutor(delay=0.4)
        brain._executor = slow  # type: ignore[assignment]  # noqa: SLF001

        class AlwaysLong:
            def evaluate(self, snapshot: Any) -> Any:
                from tachyon.strategy.signals import NoSignalReason, Signal, SignalReport

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

        brain._generator = AlwaysLong()  # type: ignore[assignment]  # noqa: SLF001
        brain.feed_monitor.record()

        total = 200
        runner = asyncio.create_task(brain.run())
        try:
            for seq in range(1, total + 1):
                publisher.publish_tick("RELIANCE", _tick(seq))
                if seq % 50 == 0:
                    await asyncio.sleep(0)

            async def drained() -> None:
                while brain.stats.ticks < total:
                    await asyncio.sleep(0.005)

            # Comfortably shorter than would be possible if a 0.4 s placement blocked the loop
            # once per 10-tick evaluation window (that would be 8 s of stalls).
            await asyncio.wait_for(drained(), timeout=3.0)
            assert brain.stats.ticks == total
        finally:
            await brain.stop()
            subscriber.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(runner, timeout=10.0)

        assert slow.calls == 1, "only one entry may be in flight per symbol"
        assert brain.aggregator("RELIANCE") is not None
        assert brain.aggregator("RELIANCE").gaps_detected == 0  # type: ignore[union-attr]

    async def _test_shutdown_stops_the_watchdog_thread(self, tmp_path: Path, spine: Any) -> None:
        """A non-daemon thread left running keeps the process alive forever."""
        publisher, subscriber = spine
        brain = _build_brain(tmp_path, subscriber, _settings())
        brain._watchdog.set_square_off_action(lambda: None)  # noqa: SLF001
        brain._watchdog.start()  # noqa: SLF001
        assert brain._watchdog.is_alive  # noqa: SLF001

        runner = asyncio.create_task(brain.run())
        publisher.publish_tick("RELIANCE", _tick(1))
        await asyncio.sleep(0.1)

        await brain.stop()
        subscriber.close()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(runner, timeout=10.0)

        assert not brain._watchdog.is_alive  # noqa: SLF001

    async def _test_a_locked_session_consumes_ticks_but_places_nothing(
        self, tmp_path: Path, spine: Any
    ) -> None:
        """Read-only still means telemetry flows and the 15:15 deadline stays armed."""
        publisher, subscriber = spine
        brain = _build_brain(tmp_path, subscriber, _settings())
        brain.pnl.trip("TEST", "forced")
        assert brain.state is TradingState.LOCKED

        calls = _SlowExecutor(delay=0.0)
        brain._executor = calls  # type: ignore[assignment]  # noqa: SLF001

        runner = asyncio.create_task(brain.run())
        try:
            for seq in range(1, 41):
                publisher.publish_tick("RELIANCE", _tick(seq))

            async def drained() -> None:
                while brain.stats.ticks < 40:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(drained(), timeout=5.0)
        finally:
            await brain.stop()
            subscriber.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(runner, timeout=10.0)

        assert brain.stats.ticks == 40
        assert calls.calls == 0
        assert brain.pnl.total == Decimal("0")
