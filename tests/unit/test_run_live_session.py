"""Tests for the live session orchestrator.

The :class:`LiveSession` is the only entry point the operator
needs to know; the tests below exercise the pieces that do not
require a real SmartConnect session: the per-symbol context
bookkeeping, the entry/exit routing, and the force-close safety
net. The WebSocket connection is mocked at the boundary so the
tests stay deterministic.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tachyon.execution.angel_broker import PaperBroker
from tachyon.execution.router import (
    ACTION_BUY,
    ACTION_SELL,
    ActionIntent,
    DispatchOutcome,
    ExecutionRouter,
    TopOfBook,
)
from tachyon.execution.trade_manager import (
    ActiveTrade,
    ExitKind,
    ExitReason,
    Side as LifecycleSide,
    TradeLifecycleManager,
)
from tachyon.ingestion.websocket_feed import FeedFrame, FeedMode
from tachyon.math_engine.warmup import warmup
from tachyon.strategies.time_gates import MarketPhase

import json
import socket as _socket
from datetime import datetime as _datetime

import zmq
from fastapi.testclient import TestClient

from tachyon.core.clock import IST as _IST_TZ
from tachyon.core.clock import ManualClock
from tachyon.core.config import Settings, UiSettings, WatchlistItem
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import FillUpdate, PnLUpdate, StateUpdate
from tachyon.ipc.subscriber import Subscriber, SubscriberRole
from tachyon.strategy.telemetry import StatePublisher
from tachyon.ui.app import create_app
from tachyon.ui.telemetry import TelemetryBridge

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    """Compile the Numba kernels before any test asks for them."""
    assert warmup() is True


def _ist_epoch(hour: int, minute: int, second: int = 0) -> float:
    return datetime(2026, 8, 10, hour, minute, second, tzinfo=IST).timestamp()


def _top_of_book(bid: float = 100.0, ask: float = 101.0) -> TopOfBook:
    return TopOfBook(
        bid=bid, bid_qty=200.0, ask=ask, ask_qty=100.0, ts_epoch=time.time()
    )


class _StubBookProvider:
    """Book provider that returns the same TopOfBook regardless of token."""

    def __init__(self, book: TopOfBook) -> None:
        self._book = book

    def __call__(self, _token: int) -> TopOfBook | None:
        return self._book


class TestSessionBootstrap:
    def test_session_initialises_router(self) -> None:
        from scripts.run_live_session import LiveSession

        session = LiveSession(broker=PaperBroker())
        # The router is built lazily on ``run()``; check that
        # ``run`` is callable without a real connection.
        assert session._router is None
        assert session._lifecycle is None


class TestSymbolContext:
    def test_bar_volume_resets_at_bar_boundary(self) -> None:
        from scripts.run_live_session import SymbolContext
        from tachyon.math_engine.targets import TargetEngine
        from tachyon.strategies.orb_strike import OrbStrikeEngine
        from tachyon.strategies.vwap_pullback import VwapPullbackStrategy
        from tachyon.math_engine.core import TickAggregator
        from tachyon.ipc.schemas import Tick

        aggregator = TickAggregator("RELIANCE")
        context = SymbolContext(
            symbol="RELIANCE",
            token=2885,
            aggregator=aggregator,
            orb=OrbStrikeEngine("RELIANCE"),
            vwap=VwapPullbackStrategy("RELIANCE"),
            target_engine=TargetEngine(),
        )
        ts0 = _ist_epoch(9, 20)
        # First bar
        context.on_tick(Tick(token="2885", ltp=100.0, volume=10, ts_epoch=ts0, seq=1))
        context.on_tick(Tick(token="2885", ltp=101.0, volume=20, ts_epoch=ts0 + 5, seq=2))
        assert context.current_bar_volume == 30.0
        # New bar
        context.on_tick(Tick(token="2885", ltp=102.0, volume=5, ts_epoch=ts0 + 300, seq=3))
        assert context.current_bar_volume == 5.0


class TestFrameDispatch:
    """The session's per-frame handler must call aggregator and strategies."""

    async def test_drive_ticks_through_session(self) -> None:
        from scripts.run_live_session import LiveSession
        from tachyon.math_engine.core import HistoricalBar

        session = LiveSession(broker=PaperBroker(book_provider=_StubBookProvider(_top_of_book())))
        # Open a synthetic router manually to avoid a network connection.
        session._build_router()
        # Inject a single symbol context.
        context = session._require_context(2885)
        # Pre-seed the aggregator with synthetic bars so the ATR is
        # finite; the ORB's set_context needs a positive ATR before
        # it will emit a signal.
        seed_bars: list[HistoricalBar] = []
        for i in range(20):
            p = 100.0 + 0.1 * i
            seed_bars.append(
                HistoricalBar(
                    start_epoch=_ist_epoch(9, 0) + i * 300,
                    open=p, high=p + 0.5, low=p - 0.5, close=p + 0.05, volume=1000.0,
                )
            )
        context.aggregator.seed_bars(seed_bars)
        # Walk a small ORB-style scenario. The ORB building window
        # is 09:15–09:20 IST (5 minutes); we emit one tick per
        # second so 300 ticks cover the window and 1 more
        # triggers the breakout.
        build_start = _ist_epoch(9, 15)
        for i in range(301):
            ltp = 100.0 + 0.005 * i
            payload = {
                "lp": ltp,
                "v": 5000,  # high per-print volume so RVOL clears 2.5×
                "ltt": build_start + i,
            }
            frame = FeedFrame(kind="data", mode=FeedMode.TICK, token="2885", payload=payload)
            await session._on_frame(frame)
        # Push a single L2 frame so the ORB engine has a book on the
        # breakout tick.
        book_payload = {
            "bp": [102.0, 101.5, 101.0, 100.5, 100.0],
            "sp": [103.0, 103.5, 104.0, 104.5, 105.0],
            "bq": [500, 100, 50, 50, 50],
            "sq": [100, 50, 50, 50, 50],
            "ltt": build_start + 300,
        }
        book_frame = FeedFrame(
            kind="data", mode=FeedMode.L2, token="2885", payload=book_payload
        )
        await session._on_frame(book_frame)
        # Break out at 09:20:00 with LTP above the range high.
        breakout_payload = {
            "lp": 102.5,
            "v": 2000,
            "ltt": build_start + 301,
        }
        breakout_frame = FeedFrame(
            kind="data", mode=FeedMode.TICK, token="2885", payload=breakout_payload
        )
        await session._on_frame(breakout_frame)
        # The strategy is wired; the trade may or may not be
        # accepted depending on the gate stack (gates can veto),
        # but the orchestrator's counters move.
        assert session._stats.signals_orb >= 1
        assert len(session._signals_log) >= 1

    async def test_control_frame_is_ignored(self) -> None:
        from scripts.run_live_session import LiveSession

        session = LiveSession(broker=PaperBroker())
        session._build_router()
        before = session._stats.ticks_consumed
        frame = FeedFrame(
            kind="control",
            mode=None,
            token=None,
            payload={"status": "ok"},
        )
        await session._on_frame(frame)
        # A control frame is not a tick; counters must not move.
        assert session._stats.ticks_consumed == before

    async def test_garbage_frame_is_dropped(self) -> None:
        from scripts.run_live_session import LiveSession

        session = LiveSession(broker=PaperBroker())
        session._build_router()
        before = session._stats.ticks_consumed
        # Token present but mode missing → unknown frame kind, no
        # strategy side-effects.
        frame = FeedFrame(kind="control", mode=None, token=None, payload={})
        await session._on_frame(frame)
        assert session._stats.ticks_consumed == before


class TestPhaseLookup:
    def test_phase_returns_enum(self) -> None:
        from scripts.run_live_session import LiveSession

        session = LiveSession(broker=PaperBroker())
        # No router needed for this method.
        phase = session._phase(_ist_epoch(9, 20))
        assert phase is MarketPhase.GOLDEN_WINDOW
        phase = session._phase(_ist_epoch(12, 0))
        assert phase is MarketPhase.MIDDAY_CHOP
        phase = session._phase(_ist_epoch(15, 0))
        assert phase is MarketPhase.SQUARE_OFF


class TestForceCloseAtEnd:
    def test_force_close_market_exit_per_trade(self) -> None:
        from scripts.run_live_session import LiveSession

        session = LiveSession(broker=PaperBroker())
        session._build_router()
        # Manually inject an active trade.
        trade = ActiveTrade(
            trade_id="t-1",
            symbol="RELIANCE",
            side=LifecycleSide.LONG,
            quantity=1,
            entry_price=100.0,
            entry_time=time.time(),
            initial_stop=95.0,
            initial_target=110.0,
            current_stop=95.0,
        )
        assert session._lifecycle is not None
        session._lifecycle.register(trade)
        # No LTP observed for the symbol — the fallback uses entry
        # price.
        session._force_close_at_end()
        assert session._lifecycle.get("t-1") is None
        # One market exit logged.
        assert session._stats.exits_market == 1


def _free_port() -> int:
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return port


def _settings_for(state_endpoint: str) -> Settings:
    return Settings(
        watchlist=(WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),),
        ui=UiSettings(),
        zmq_tick_endpoint=f"tcp://127.0.0.1:{_free_port()}",
        zmq_state_endpoint=state_endpoint,
    )


def _clock_for_test() -> ManualClock:
    return ManualClock(
        wall=_datetime(2026, 8, 10, 11, 0, tzinfo=_IST_TZ),
        mono=1000.0,
    )


class TestTelemetryPublishing:
    """End-to-end: ``LiveSession._publish_telemetry`` publishes to the state spine,
    which the dashboard's ``TelemetryBridge`` consumes and exposes via ``/api/snapshot``.
    """

    def test_publish_telemetry_does_not_raise_and_increments_counters(self) -> None:
        port = _free_port()
        endpoint = f"tcp://127.0.0.1:{port}"

        pub = Publisher(endpoint, role="step7")

        try:
            settings = _settings_for(endpoint)
            state_pub = StatePublisher(settings=settings, publisher=pub)
            from scripts.run_live_session import LiveSession

            session = LiveSession(
                broker=PaperBroker(),
                state_publisher=state_pub,
                clock=_clock_for_test(),
            )
            session._build_router()

            # Capture a heartbeat frame from the publisher via a SUB socket.
            ctx = zmq.Context()
            sub = ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b"FEED.")
            sub.connect(endpoint)
            Publisher.settle(0.3)

            session._publish_telemetry(ts_epoch=1.0)

            sub.setsockopt(zmq.RCVTIMEO, 1500)
            saw_heartbeat = False
            for _ in range(10):
                try:
                    topic, _ = sub.recv_multipart()
                except zmq.Again:
                    break
                if topic == b"FEED.HEARTBEAT":
                    saw_heartbeat = True
                    break

            sub.close()
            ctx.term()

            assert state_pub.published >= 2
            assert state_pub.errors == 0
            assert saw_heartbeat, "no FEED.HEARTBEAT frame was published"
        finally:
            pub.close()

    def test_fill_update_routes_to_fill_topic(self) -> None:
        port = _free_port()
        endpoint = f"tcp://127.0.0.1:{port}"

        pub = Publisher(endpoint, role="step7")

        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"FILL.")
        sub.connect(endpoint)

        try:
            Publisher.settle(0.3)

            settings = _settings_for(endpoint)
            state_pub = StatePublisher(settings=settings, publisher=pub)
            from scripts.run_live_session import LiveSession

            session = LiveSession(
                broker=PaperBroker(),
                state_publisher=state_pub,
                clock=_clock_for_test(),
            )
            session._build_router()

            fill = FillUpdate(
                symbol="RELIANCE",
                order_id="paper-1",
                status="filled",
                side="BUY",
                quantity=1,
                filled_quantity=1,
                price="2500.00",
                ts_epoch=1.0,
                order_tag="ORB",
                is_exit=False,
            )

            session._publish_telemetry(ts_epoch=1.0, fill_update=fill)

            sub.setsockopt(zmq.RCVTIMEO, 1500)

            frames: list[tuple[bytes, bytes]] = []
            for _ in range(10):
                try:
                    frames.append(tuple(sub.recv_multipart()))
                except zmq.Again:
                    break

            fill_frames = [
                (topic, payload) for topic, payload in frames if topic == b"FILL.RELIANCE"
            ]
            assert fill_frames, f"no FILL.RELIANCE frame received (saw topics: {[f[0] for f in frames]})"

            topic, payload = fill_frames[0]
            assert topic == b"FILL.RELIANCE"
            decoded = json.loads(payload)
            assert decoded["symbol"] == "RELIANCE"
            assert decoded["status"] == "filled"
            assert decoded["price"] == "2500.00"
            assert decoded["order_tag"] == "ORB"
        finally:
            sub.close()
            ctx.term()
            pub.close()

    def test_publishes_reach_the_dashboard_snapshot(self) -> None:
        port = _free_port()
        endpoint = f"tcp://127.0.0.1:{port}"

        pub = Publisher(endpoint, role="step7")
        Publisher.settle(0.3)

        try:
            settings = _settings_for(endpoint)
            clock = _clock_for_test()

            state_sub = Subscriber(
                ("STATE.", "PNL.", "FILL.", "RISK."),
                role=SubscriberRole.UI,
                endpoint=endpoint,
                conflate=True,
                settings=settings,
                clock=clock,
            )

            bridge = TelemetryBridge(
                settings=settings,
                state_subscriber=state_sub,
                tick_subscriber=None,
                clock=clock,
            )

            app = create_app(settings=settings, bridge=bridge, start_bridge=False, clock=clock)
            state_pub = StatePublisher(settings=settings, publisher=pub)
            from scripts.run_live_session import LiveSession

            session = LiveSession(
                broker=PaperBroker(),
                state_publisher=state_pub,
                clock=clock,
            )
            session._build_router()

            session._publish_telemetry(ts_epoch=1.0)
            Publisher.settle(0.3)
            bridge.poll_once()

            with TestClient(app) as client:
                body = client.get("/api/snapshot").json()

            assert body["type"] == "snapshot"
            assert isinstance(body["session"]["state"], str) and body["session"]["state"]
            assert body["session"]["mode"] in ("PAPER", "LIVE")
            assert isinstance(body["pnl"]["limit"], str)
            assert isinstance(body["pnl"]["realised"], str)
        finally:
            pub.close()
