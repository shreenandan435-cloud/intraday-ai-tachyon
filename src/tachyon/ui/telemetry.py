"""Telemetry bridge — ZeroMQ to WebSocket, CLAUDE.md §2.1, §7.3.

Reads both spines, folds them into one snapshot, and pushes it to browsers at 10 Hz. Three
independence rules shape the whole module, and each has a specific failure it prevents.

**The publisher must never wait on the UI.** The subscriber connects with ``LINGER = 0`` and a
bounded high-water mark, so a UI that stalls is dropped by ZeroMQ rather than back-pressuring
the tick feed (CLAUDE.md §2.1). Nothing here can slow the Brain down.

**The UI must never wait on a browser.** Each WebSocket client owns a bounded queue. A client
that cannot keep up loses *its own* frames and nobody else's; the fan-out never awaits a send.
A laptop that goes to sleep with the dashboard open must not stall the bridge for everyone.

**Conflation is the point, not an optimisation.** The bridge uses
:meth:`~tachyon.ipc.subscriber.Subscriber.drain_latest` — the userspace conflation built in
Phase 3 — because the UI renders the newest value and nothing else. Draining a 4000-message
backlog to render one frame would be pure waste, and would make the display lag reality by
exactly as long as the backlog is deep. ``ZMQ_CONFLATE`` is *not* used: it silently discards
multipart messages entirely (measured against libzmq 4.3.5).

Conflation is permitted here and only here because this consumer is presentation-only. VWAP and
OBI accumulate over every print, so the strategy path must see them all — but the *rendered*
numbers are computed by the Brain and arrive already-aggregated.

``FILL.`` and ``RISK.`` are events, not states
----------------------------------------------
A dropped tick frame costs one repaint. A dropped fill is a trade the operator never sees. So
the two event topics are drained per *message* into a bounded ring rather than conflated per
topic, and every one is forwarded. They are low-rate by nature — a handful a day.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Final

import msgspec

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.ipc.schemas import (
    TOPIC_DEPTH,
    TOPIC_FILL,
    TOPIC_PNL,
    TOPIC_RISK,
    TOPIC_STATE,
    TOPIC_TICK,
    FillUpdate,
    Heartbeat,
    OrderBook,
    PnLUpdate,
    RiskEvent,
    StateUpdate,
    Tick,
    symbol_from_topic,
)
from tachyon.ipc.subscriber import Envelope, Subscriber, SubscriberRole

_log = get_logger(__name__)

_ENCODER: Final[msgspec.json.Encoder] = msgspec.json.Encoder()

#: Topics on the tick spine the UI cares about. All conflatable — each carries a full value.
TICK_TOPICS: Final[tuple[str, ...]] = (TOPIC_TICK, TOPIC_DEPTH, "FEED.")

#: Topics on the state spine.
STATE_TOPICS: Final[tuple[str, ...]] = (TOPIC_STATE, TOPIC_PNL, TOPIC_FILL, TOPIC_RISK)

#: How many recent events each ring keeps. Small: the panel shows a handful, and an unbounded
#: buffer in a process that runs all day is a leak with extra steps.
EVENT_RING: Final[int] = 50

#: Frames a slow client may fall behind before it starts losing them. Two seconds at 10 Hz —
#: long enough to ride out a garbage collection, short enough that a wedged tab is not
#: rendering a stale frame from a minute ago.
CLIENT_QUEUE_DEPTH: Final[int] = 20


@dataclass(slots=True)
class SymbolView:
    """Latest rendered numbers for one symbol."""

    symbol: str
    ltp: float = 0.0
    obi: float = 0.0
    obi_weighted: float = 0.0
    spread: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    ts_epoch: float = 0.0
    """Wire timestamp of the newest message. The browser measures staleness against
    *its own* clock and this value's arrival, never against wall-clock arithmetic."""

    updated_mono: float = 0.0


@dataclass(slots=True)
class TelemetryStats:
    """Counters for observability."""

    polls: int = 0
    tick_frames: int = 0
    state_frames: int = 0
    events: int = 0
    broadcasts: int = 0
    frames_dropped: int = 0
    clients_connected: int = 0
    clients_dropped: int = 0
    poll_errors: int = 0


class ClientChannel:
    """One WebSocket client's outbound queue.

    Bounded and lossy on purpose. :meth:`offer` never awaits and never blocks: when the queue
    is full the *oldest* frame is discarded, because a UI showing the newest available state is
    strictly more useful than one replaying a backlog it is already behind on.
    """

    __slots__ = ("_queue", "dropped", "name")

    def __init__(self, name: str, depth: int = CLIENT_QUEUE_DEPTH) -> None:
        self.name = name
        self.dropped = 0
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=depth)

    def offer(self, frame: bytes) -> bool:
        """Enqueue a frame, dropping the oldest if full. Returns False if a frame was dropped."""
        try:
            self._queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(frame)
            return False

    async def get(self) -> bytes:
        return await self._queue.get()

    def try_get(self) -> bytes | None:
        """Take a frame if one is waiting, without awaiting. ``None`` if the queue is empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class TelemetryBridge:
    """Polls both ZeroMQ spines and fans the result out to WebSocket clients.

    Args:
        settings: resolved config. Supplies the endpoints and ``ui.ws_max_hz``.
        tick_subscriber / state_subscriber: injected for tests.

    Example::

        bridge = TelemetryBridge(settings=settings)
        task = asyncio.create_task(bridge.run())
        channel = bridge.register("127.0.0.1:54321")
        ...
        bridge.unregister(channel)
    """

    __slots__ = (
        "_clients",
        "_clock",
        "_events",
        "_hz",
        "_owns_subscribers",
        "_pnl",
        "_session",
        "_settings",
        "_state_sub",
        "_stopping",
        "_symbols",
        "_tick_sub",
        "stats",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        tick_subscriber: Subscriber | None = None,
        state_subscriber: Subscriber | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._hz = max(1, min(60, self._settings.ui.ws_max_hz))

        self._owns_subscribers = tick_subscriber is None and state_subscriber is None
        # role=UI is what unlocks conflation. STRATEGY or RISK would be refused at
        # construction, which is the guard that keeps a sampled VWAP out of the trading path.
        self._tick_sub = (
            tick_subscriber
            if tick_subscriber is not None
            else Subscriber(
                TICK_TOPICS,
                role=SubscriberRole.UI,
                endpoint=self._settings.zmq_tick_endpoint,
                conflate=True,
                settings=self._settings,
                clock=clock,
            )
        )
        self._state_sub = (
            state_subscriber
            if state_subscriber is not None
            else Subscriber(
                STATE_TOPICS,
                role=SubscriberRole.UI,
                endpoint=self._settings.zmq_state_endpoint,
                conflate=True,
                settings=self._settings,
                clock=clock,
            )
        )

        self._symbols: dict[str, SymbolView] = {}
        self._session: StateUpdate | None = None
        self._pnl: PnLUpdate | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=EVENT_RING)
        self._clients: set[ClientChannel] = set()
        self._stopping = asyncio.Event()
        self.stats = TelemetryStats()

    # ── clients ──────────────────────────────────────────────────────────────

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def interval_seconds(self) -> float:
        return 1.0 / self._hz

    def register(self, name: str) -> ClientChannel:
        """Attach a client and prime it with the current snapshot.

        Priming matters: a browser connecting at 11:07 would otherwise render blank panels
        until the next value changed, and blank is indistinguishable from zero on a P&L display.
        """
        channel = ClientChannel(name)
        self._clients.add(channel)
        self.stats.clients_connected += 1
        channel.offer(self.snapshot_frame())
        _log.info("ui.client_connected", client=name, clients=len(self._clients))
        return channel

    def unregister(self, channel: ClientChannel) -> None:
        self._clients.discard(channel)
        self.stats.clients_dropped += 1
        _log.info(
            "ui.client_disconnected",
            client=channel.name,
            dropped_frames=channel.dropped,
            clients=len(self._clients),
        )

    # ── snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """The complete current view. Also served by ``GET /api/snapshot``."""
        return {
            "type": "snapshot",
            "server_epoch": now_ist(self._clock).timestamp(),
            "session": _struct_to_dict(self._session),
            "pnl": _struct_to_dict(self._pnl),
            "symbols": [
                {
                    "symbol": view.symbol,
                    "ltp": view.ltp,
                    "obi": view.obi,
                    "obi_weighted": view.obi_weighted,
                    "spread": view.spread,
                    "best_bid": view.best_bid,
                    "best_ask": view.best_ask,
                    "ts_epoch": view.ts_epoch,
                }
                for view in sorted(self._symbols.values(), key=lambda v: v.symbol)
            ],
            "events": list(self._events),
            "stats": {
                "clients": len(self._clients),
                "frames_dropped": self.stats.frames_dropped,
                "tick_frames": self.stats.tick_frames,
            },
        }

    def snapshot_frame(self) -> bytes:
        return _ENCODER.encode(self.snapshot())

    # ── the loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Poll at ``ui.ws_max_hz`` and broadcast. Never raises."""
        _log.info(
            "ui.telemetry_started",
            hz=self._hz,
            tick_endpoint=self._tick_sub.endpoint,
            state_endpoint=self._state_sub.endpoint,
        )
        while not self._stopping.is_set():
            try:
                self.poll_once()
                self.broadcast()
            except Exception as exc:  # noqa: BLE001 - telemetry must not take down the UI
                self.stats.poll_errors += 1
                _log.error(
                    "ui.telemetry_poll_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_seconds)

        _log.info("ui.telemetry_stopped", polls=self.stats.polls, clients=len(self._clients))

    def poll_once(self) -> None:
        """Drain both sockets and fold everything into the snapshot.

        Synchronous and non-blocking: ``drain_latest`` uses ``NOBLOCK`` and returns whatever is
        already queued. That is what lets a 10 Hz async loop read a socket without a thread and
        without ever awaiting on the network.
        """
        self.stats.polls += 1
        self._apply(self._tick_sub.drain_latest().values(), tick_spine=True)
        self._apply(self._state_sub.drain_latest().values(), tick_spine=False)

    def _apply(self, envelopes: Iterable[Envelope], *, tick_spine: bool) -> None:
        for envelope in envelopes:
            message = envelope.message
            if isinstance(message, Tick):
                self.stats.tick_frames += 1
                view = self._view(symbol_from_topic(envelope.topic))
                view.ltp = message.ltp
                view.ts_epoch = message.ts_epoch
                view.updated_mono = envelope.received_mono
            elif isinstance(message, OrderBook):
                self.stats.tick_frames += 1
                view = self._view(symbol_from_topic(envelope.topic))
                view.best_bid = message.best_bid
                view.best_ask = message.best_ask
                view.spread = message.spread
                view.ts_epoch = message.ts_epoch
                view.updated_mono = envelope.received_mono
            elif isinstance(message, StateUpdate):
                self.stats.state_frames += 1
                self._session = message
            elif isinstance(message, PnLUpdate):
                self.stats.state_frames += 1
                self._pnl = message
            elif isinstance(message, (FillUpdate, RiskEvent)):
                # Events, never conflated away — see the module docstring.
                self.stats.events += 1
                self._events.appendleft({"topic": envelope.topic, **_struct_to_dict(message)})
            elif isinstance(message, Heartbeat) and tick_spine:
                pass  # liveness only; the browser measures staleness itself

    def _view(self, symbol: str) -> SymbolView:
        view = self._symbols.get(symbol)
        if view is None:
            view = SymbolView(symbol=symbol)
            self._symbols[symbol] = view
        return view

    # ── fan-out ──────────────────────────────────────────────────────────────

    def broadcast(self) -> None:
        """Offer the current snapshot to every client. Never awaits, never raises.

        Encoded **once** for all clients. Per-client encoding would be the one place a
        twenty-tab dashboard could start costing real CPU on the same box as the Brain.
        """
        if not self._clients:
            return
        frame = self.snapshot_frame()
        self.stats.broadcasts += 1
        for channel in tuple(self._clients):
            if not channel.offer(frame):
                self.stats.frames_dropped += 1

    async def stop(self) -> None:
        """Stop polling and close the sockets we own."""
        self._stopping.set()
        if self._owns_subscribers:
            self._tick_sub.close()
            self._state_sub.close()

    # ── direct injection ─────────────────────────────────────────────────────

    def record_event(self, topic: str, **fields: Any) -> None:
        """Push an event raised inside the UI process itself (a panic, a listener fault).

        The parameter is ``topic``, not ``kind``, so a caller may pass ``kind=`` as one of the
        event fields without colliding with it.
        """
        self.stats.events += 1
        self._events.appendleft(
            {
                "topic": f"{topic}.LOCAL",
                "kind": topic,
                "ts_epoch": now_ist(self._clock).timestamp(),
                **fields,
            }
        )


def _struct_to_dict(message: Any) -> dict[str, Any]:
    """Convert a msgspec Struct to a plain dict, or ``{}`` for ``None``."""
    if message is None:
        return {}
    return {name: getattr(message, name) for name in getattr(message, "__struct_fields__", ())}
