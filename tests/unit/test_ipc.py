"""Phase 3 IPC layer tests — CLAUDE.md §2.

These exercise real ZeroMQ sockets over real TCP rather than mocks. The socket-option
contract (HWM, LINGER) and the conflation behaviour are precisely the things a mock would
happily lie about, and they are the things that decide whether the strategy sees every tick.
"""

from __future__ import annotations

import gc
import socket
from collections.abc import Iterator
from datetime import datetime

import msgspec
import pytest
import zmq

from tachyon.core.clock import IST, ManualClock
from tachyon.ipc.monitor import (
    FeedMonitor,
    FeedStaleEvent,
    FeedStaleException,
    FeedState,
)
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import (
    HEARTBEAT_TOPIC_BYTES,
    SCHEMA_VERSION,
    TOPIC_HEARTBEAT,
    Heartbeat,
    OrderBook,
    Tick,
    UnknownTopicError,
    decode_for_topic,
    depth_topic,
    encode,
    symbol_from_topic,
    tick_topic,
)
from tachyon.ipc.subscriber import (
    ConflationForbiddenError,
    SchemaVersionMismatchError,
    Subscriber,
    SubscriberRole,
)

RECV_TIMEOUT_MS = 2000


def _free_endpoint() -> str:
    """Reserve an ephemeral TCP port and hand back a ZMQ endpoint for it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def _tick(ltp: float = 100.0, volume: int = 10, ts: float = 1_786_000_000.0, seq: int = 1) -> Tick:
    return Tick(token="2885", ltp=ltp, volume=volume, ts_epoch=ts, seq=seq)


def _book(ts: float = 1_786_000_000.0) -> OrderBook:
    return OrderBook(
        token="2885",
        bid_price=(99.95, 99.90, 99.85, 99.80, 99.75),
        bid_qty=(100, 200, 300, 400, 500),
        ask_price=(100.05, 100.10, 100.15, 100.20, 100.25),
        ask_qty=(120, 220, 320, 420, 520),
        ts_epoch=ts,
    )


@pytest.fixture
def endpoint() -> str:
    return _free_endpoint()


@pytest.fixture
def pubsub(endpoint: str) -> Iterator[tuple[Publisher, Subscriber]]:
    """A connected PUB/SUB pair on a private port, settled past the slow-joiner window."""
    publisher = Publisher(endpoint, role="test-ingestor")
    subscriber = Subscriber((), role=SubscriberRole.STRATEGY, endpoint=endpoint)
    Publisher.settle(0.3)
    try:
        yield publisher, subscriber
    finally:
        subscriber.close()
        publisher.close()


# ──────────────────────────────────────────────────────────────────────────────
# schemas.py
# ──────────────────────────────────────────────────────────────────────────────


class TestSchemas:
    def test_structs_are_frozen(self) -> None:
        tick = _tick()
        with pytest.raises(AttributeError):
            tick.ltp = 200.0  # type: ignore[misc]

    def test_structs_are_untracked_by_gc(self) -> None:
        """gc=False keeps the hot path out of every collection pass (CLAUDE.md §3.2)."""
        assert not gc.is_tracked(_tick())
        assert not gc.is_tracked(_book())
        assert not gc.is_tracked(Heartbeat(ts_epoch=1.0, seq=1))

    def test_tick_round_trip(self) -> None:
        original = _tick(ltp=1234.55, volume=98_765)
        assert decode_for_topic("TICK.RELIANCE", encode(original)) == original

    def test_orderbook_round_trip(self) -> None:
        original = _book()
        decoded = decode_for_topic("DEPTH.RELIANCE", encode(original))
        assert decoded == original
        assert isinstance(decoded, OrderBook)
        assert decoded.best_bid == 99.95
        assert decoded.best_ask == 100.05
        assert decoded.spread == pytest.approx(0.10)
        assert not decoded.is_crossed()

    def test_heartbeat_round_trip_carries_schema_version(self) -> None:
        decoded = decode_for_topic(TOPIC_HEARTBEAT, encode(Heartbeat(ts_epoch=1.0, seq=7)))
        assert isinstance(decoded, Heartbeat)
        assert decoded.seq == 7
        assert decoded.schema_version == SCHEMA_VERSION

    def test_timestamps_render_in_ist(self) -> None:
        moment = datetime(2026, 8, 10, 10, 30, tzinfo=IST)
        assert _tick(ts=moment.timestamp()).ts_ist == moment
        assert _tick(ts=moment.timestamp()).ts_ist.utcoffset() == IST.utcoffset(moment)

    def test_truncated_depth_is_rejected(self) -> None:
        """A four-level book must fail loudly, not silently skew OBI."""
        payload = msgspec.json.encode(
            {
                "token": "2885",
                "bid_price": [1.0, 2.0, 3.0, 4.0],
                "bid_qty": [1, 2, 3, 4],
                "ask_price": [1.0, 2.0, 3.0, 4.0],
                "ask_qty": [1, 2, 3, 4],
                "ts_epoch": 1.0,
            }
        )
        with pytest.raises(msgspec.ValidationError):
            decode_for_topic("DEPTH.RELIANCE", payload)

    def test_wrong_type_is_rejected(self) -> None:
        payload = msgspec.json.encode(
            {"token": "2885", "ltp": "not-a-number", "volume": 1, "ts_epoch": 1.0}
        )
        with pytest.raises(msgspec.ValidationError):
            decode_for_topic("TICK.RELIANCE", payload)

    def test_unknown_topic_raises(self) -> None:
        with pytest.raises(UnknownTopicError):
            decode_for_topic("MYSTERY.THING", b"{}")

    def test_topic_helpers(self) -> None:
        assert tick_topic("RELIANCE") == b"TICK.RELIANCE"
        assert depth_topic("RELIANCE") == b"DEPTH.RELIANCE"
        assert symbol_from_topic("TICK.RELIANCE") == "RELIANCE"

    def test_payload_stays_compact(self) -> None:
        """A tick is the highest-frequency message; guard against field creep."""
        assert len(encode(_tick())) < 100


# ──────────────────────────────────────────────────────────────────────────────
# publisher.py / subscriber.py — socket contract
# ──────────────────────────────────────────────────────────────────────────────


class TestSocketContract:
    def test_subscriber_enforces_hwm_and_linger(self, endpoint: str) -> None:
        """CLAUDE.md §2.1: RCVHWM 10_000, LINGER 0 — read back off the live socket."""
        with Subscriber((), role=SubscriberRole.STRATEGY, endpoint=endpoint) as sub:
            assert sub._socket.getsockopt(zmq.RCVHWM) == 10_000
            assert sub._socket.getsockopt(zmq.LINGER) == 0

    def test_publisher_enforces_hwm_and_linger(self, endpoint: str) -> None:
        with Publisher(endpoint) as pub:
            assert pub._socket.getsockopt(zmq.SNDHWM) == 10_000
            assert pub._socket.getsockopt(zmq.LINGER) == 0

    def test_zmq_conflate_is_never_set(self, endpoint: str) -> None:
        """Setting ZMQ_CONFLATE would silently discard every multipart message."""
        with Subscriber((), role=SubscriberRole.UI, endpoint=endpoint, conflate=True) as sub:
            assert sub._socket.getsockopt(zmq.CONFLATE) == 0

    def test_publisher_binds_and_subscriber_connects(self, endpoint: str) -> None:
        """Reversing this makes restarts fail intermittently and is forbidden."""
        with Publisher(endpoint) as first:
            assert not first.closed
            with pytest.raises(zmq.ZMQError):
                Publisher(endpoint)  # second bind to the same port must fail


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end delivery
# ──────────────────────────────────────────────────────────────────────────────


class TestDelivery:
    def test_tick_survives_the_wire(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        publisher, subscriber = pubsub
        sent = _tick(ltp=2456.75, volume=41_200)
        publisher.publish_tick("RELIANCE", sent)

        envelope = subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
        assert envelope is not None
        assert envelope.topic == "TICK.RELIANCE"
        assert envelope.message == sent
        assert isinstance(envelope.message, Tick)
        assert envelope.message.ltp == 2456.75

    def test_orderbook_survives_the_wire(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        publisher, subscriber = pubsub
        publisher.publish_orderbook("RELIANCE", _book())

        envelope = subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
        assert envelope is not None
        assert envelope.topic == "DEPTH.RELIANCE"
        assert isinstance(envelope.message, OrderBook)
        assert envelope.message.bid_qty == (100, 200, 300, 400, 500)

    def test_heartbeat_sequence_increments(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        publisher, subscriber = pubsub
        first = publisher.publish_heartbeat()
        second = publisher.publish_heartbeat()
        assert (first.seq, second.seq) == (1, 2)
        assert publisher.heartbeat_seq == 2

        received = [subscriber.recv(timeout_ms=RECV_TIMEOUT_MS) for _ in range(2)]
        seqs = [
            e.message.seq for e in received if e is not None and isinstance(e.message, Heartbeat)
        ]
        assert seqs == [1, 2]

    def test_topic_filtering(self, endpoint: str) -> None:
        publisher = Publisher(endpoint)
        subscriber = Subscriber(("TICK.INFY",), role=SubscriberRole.STRATEGY, endpoint=endpoint)
        Publisher.settle(0.3)
        try:
            publisher.publish_tick("RELIANCE", _tick(ltp=1.0))
            publisher.publish_tick("INFY", _tick(ltp=2.0))

            envelope = subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
            assert envelope is not None
            assert envelope.topic == "TICK.INFY"
            assert subscriber.recv(timeout_ms=200) is None, "RELIANCE must be filtered out"
        finally:
            subscriber.close()
            publisher.close()

    def test_strategy_receives_every_tick_in_order(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        """VWAP and OBI are accumulators — a gap here silently corrupts them."""
        publisher, subscriber = pubsub
        for i in range(200):
            publisher.publish_tick("RELIANCE", _tick(ltp=float(i), volume=i, seq=i + 1))

        prices = []
        for _ in range(200):
            envelope = subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
            assert envelope is not None
            assert isinstance(envelope.message, Tick)
            prices.append(envelope.message.ltp)

        assert prices == [float(i) for i in range(200)]

    def test_recv_timeout_returns_none(self, pubsub: tuple[Publisher, Subscriber]) -> None:
        _, subscriber = pubsub
        assert subscriber.recv(timeout_ms=100) is None

    def test_malformed_payload_is_counted_not_raised(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        """One corrupt frame must not take down the strategy process."""
        publisher, subscriber = pubsub
        publisher.publish_raw(tick_topic("RELIANCE"), b"{not valid json")
        assert subscriber.recv(timeout_ms=RECV_TIMEOUT_MS) is None
        assert subscriber.decode_errors == 1

        publisher.publish_tick("RELIANCE", _tick())
        assert subscriber.recv(timeout_ms=RECV_TIMEOUT_MS) is not None

    def test_schema_version_mismatch_stops_the_process(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        publisher, subscriber = pubsub
        rogue = msgspec.json.encode(
            {"ts_epoch": 1.0, "seq": 1, "role": "ingestor", "schema_version": 99}
        )
        publisher.publish_raw(HEARTBEAT_TOPIC_BYTES, rogue)

        with pytest.raises(SchemaVersionMismatchError, match="v99"):
            subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)

    def test_monitor_is_fed_by_the_subscriber(self, endpoint: str) -> None:
        monitor = FeedMonitor()
        publisher = Publisher(endpoint)
        subscriber = Subscriber(
            (), role=SubscriberRole.STRATEGY, endpoint=endpoint, monitor=monitor
        )
        Publisher.settle(0.3)
        try:
            publisher.publish_tick("RELIANCE", _tick())
            subscriber.recv(timeout_ms=RECV_TIMEOUT_MS)
            assert monitor.messages_seen == 1
        finally:
            subscriber.close()
            publisher.close()


# ──────────────────────────────────────────────────────────────────────────────
# Conflation policy
# ──────────────────────────────────────────────────────────────────────────────


class TestConflationPolicy:
    @pytest.mark.parametrize(
        "role", [SubscriberRole.STRATEGY, SubscriberRole.RISK, SubscriberRole.EXECUTION]
    )
    def test_conflation_forbidden_for_decision_roles(
        self, role: SubscriberRole, endpoint: str
    ) -> None:
        with pytest.raises(ConflationForbiddenError, match="VWAP"):
            Subscriber((), role=role, endpoint=endpoint, conflate=True)

    @pytest.mark.parametrize("role", [SubscriberRole.UI, SubscriberRole.TELEMETRY])
    def test_conflation_permitted_for_presentation_roles(
        self, role: SubscriberRole, endpoint: str
    ) -> None:
        with Subscriber((), role=role, endpoint=endpoint, conflate=True) as sub:
            assert not sub.closed

    def test_drain_latest_refused_without_conflation(
        self, pubsub: tuple[Publisher, Subscriber]
    ) -> None:
        _, subscriber = pubsub
        with pytest.raises(ConflationForbiddenError, match="discards messages"):
            subscriber.drain_latest()

    def test_drain_latest_keeps_newest_per_topic(self, endpoint: str) -> None:
        publisher = Publisher(endpoint)
        ui = Subscriber((), role=SubscriberRole.UI, endpoint=endpoint, conflate=True)
        Publisher.settle(0.3)
        try:
            for i in range(50):
                publisher.publish_tick("RELIANCE", _tick(ltp=float(i)))
                publisher.publish_tick("INFY", _tick(ltp=float(100 + i)))
            Publisher.settle(0.3)

            latest = ui.drain_latest()
            assert set(latest) == {"TICK.RELIANCE", "TICK.INFY"}

            reliance = latest["TICK.RELIANCE"].message
            infy = latest["TICK.INFY"].message
            assert isinstance(reliance, Tick)
            assert isinstance(infy, Tick)
            assert reliance.ltp == 49.0, "per-topic conflation must keep the newest"
            assert infy.ltp == 149.0

            assert ui.drain_latest() == {}, "the queue must be empty after draining"
        finally:
            ui.close()
            publisher.close()


# ──────────────────────────────────────────────────────────────────────────────
# monitor.py
# ──────────────────────────────────────────────────────────────────────────────


class TestFeedMonitor:
    @staticmethod
    def _clock() -> ManualClock:
        return ManualClock(wall=datetime(2026, 8, 10, 10, 0, tzinfo=IST), mono=1000.0)

    def test_starts_armed_so_a_feed_that_never_arrives_goes_stale(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        assert not monitor.is_stale

        clock.advance(2.01)
        assert monitor.is_stale, "a feed that never delivers must not look healthy forever"

    def test_recording_keeps_it_fresh(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        for _ in range(10):
            clock.advance(1.0)
            monitor.record()
            assert not monitor.is_stale
        assert monitor.messages_seen == 10

    def test_goes_stale_exactly_at_the_threshold(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        monitor.record()

        clock.advance(2.0)
        assert not monitor.is_stale, "exactly at the threshold is still fresh"
        clock.advance(0.001)
        assert monitor.is_stale

    def test_assert_fresh_raises_when_stale(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        monitor.record()
        monitor.assert_fresh()

        clock.advance(2.5)
        with pytest.raises(FeedStaleException, match="stale for 2.5"):
            monitor.assert_fresh()

    def test_stale_exception_carries_context(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        clock.advance(3.0)
        try:
            monitor.assert_fresh()
        except FeedStaleException as exc:
            assert exc.stale_for == pytest.approx(3.0)
            assert exc.threshold == pytest.approx(2.0)
        else:
            pytest.fail("expected FeedStaleException")

    def test_callbacks_are_edge_triggered(self) -> None:
        """The watchdog polls 4x/second; a halt callback must not fire 4x/second."""
        clock = self._clock()
        stale_events: list[FeedStaleEvent] = []
        recover_events: list[FeedStaleEvent] = []
        monitor = FeedMonitor(
            clock=clock, on_stale=stale_events.append, on_recover=recover_events.append
        )
        monitor.record()

        assert monitor.check() is FeedState.FRESH
        assert stale_events == []

        clock.advance(2.5)
        for _ in range(10):
            assert monitor.check() is FeedState.STALE
        assert len(stale_events) == 1, "on_stale must fire once per transition"

        monitor.record()
        for _ in range(10):
            assert monitor.check() is FeedState.FRESH
        assert len(recover_events) == 1

    def test_first_arrival_is_not_reported_as_a_recovery(self) -> None:
        """STARTING -> FRESH is the feed coming up, not recovering from an outage.

        A handler that un-blocks entries on recovery must not be invoked before the feed has
        ever delivered anything.
        """
        clock = self._clock()
        recover_events: list[FeedStaleEvent] = []
        monitor = FeedMonitor(clock=clock, on_recover=recover_events.append)

        monitor.record()
        assert monitor.check() is FeedState.FRESH
        assert recover_events == []

    def test_stale_event_payload(self) -> None:
        clock = self._clock()
        captured: list[FeedStaleEvent] = []
        monitor = FeedMonitor(clock=clock, on_stale=captured.append)
        monitor.record()
        clock.advance(2.5)
        monitor.check()

        event = captured[0]
        assert event.state is FeedState.STALE
        assert event.stale_for == pytest.approx(2.5)
        assert event.messages_seen == 1
        assert event.at_ist.tzinfo is not None

    def test_a_failing_callback_cannot_stall_the_watchdog(self) -> None:
        clock = self._clock()

        def boom(_event: FeedStaleEvent) -> None:
            raise RuntimeError("listener exploded")

        monitor = FeedMonitor(clock=clock, on_stale=boom)
        monitor.record()
        clock.advance(2.5)

        assert monitor.check() is FeedState.STALE
        assert monitor.state is FeedState.STALE

    def test_lag_is_reported_separately_from_staleness(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        assert monitor.lag_seconds is None

        monitor.record(wire_epoch=clock.now().timestamp() - 0.75)
        assert monitor.lag_seconds == pytest.approx(0.75, abs=1e-6)
        assert not monitor.is_stale, "clock lag is not the same fault as a stalled feed"

    def test_reset_rearms(self) -> None:
        clock = self._clock()
        monitor = FeedMonitor(clock=clock)
        monitor.record()
        clock.advance(5.0)
        assert monitor.is_stale

        monitor.reset()
        assert not monitor.is_stale
        assert monitor.messages_seen == 0
        assert monitor.state is FeedState.STARTING

    def test_threshold_matches_the_constitution(self) -> None:
        assert FeedMonitor().stale_after == 2.0
