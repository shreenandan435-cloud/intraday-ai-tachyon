"""ZeroMQ PUB side of the tick spine — CLAUDE.md §2.1.

Used by the ingestion process (Process A). It contains no business logic: decode the broker
frame, stamp it, publish it. Anything that makes a decision belongs in the Brain.

Contract, non-negotiable:

* **The publisher binds; the subscriber connects.** No exceptions.
* ``SNDHWM = 10_000`` and ``LINGER = 0``. When a subscriber cannot keep up, ZeroMQ drops
  messages for that subscriber. That is the intended behaviour: a stale tick is worthless, and
  back-pressuring the ingestion socket to protect a slow consumer would stall the whole feed.
* Frames are multipart ``[topic][payload]`` so the topic can be matched without touching the
  payload, and the payload needs no delimiter escaping.

**Sockets are not thread-safe.** One :class:`Publisher` per thread. Sharing one across the
WebSocket callback and a heartbeat timer will corrupt the stream; give the timer its own
publisher or marshal both onto one thread.
"""

from __future__ import annotations

import time as _time
from types import TracebackType
from typing import Final, Self

import zmq

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.ipc.schemas import (
    HEARTBEAT_TOPIC_BYTES,
    Heartbeat,
    OrderBook,
    Tick,
    WireMessage,
    depth_topic,
    encode,
    tick_topic,
)

_log = get_logger(__name__)

#: CLAUDE.md §2.1. Deep enough to absorb a burst, shallow enough that a wedged subscriber
#: cannot make the publisher hoard memory.
DEFAULT_HWM: Final[int] = 10_000

#: Discard immediately on close. A shutting-down process must never block trying to flush
#: ticks that are already worthless.
DEFAULT_LINGER_MS: Final[int] = 0

#: PUB/SUB "slow joiner": a subscriber's SUBSCRIBE takes a moment to reach the publisher, and
#: anything sent before then is silently dropped. Callers that must not lose the first message
#: should :meth:`Publisher.settle` after binding.
DEFAULT_SETTLE_SECONDS: Final[float] = 0.25


class Publisher:
    """Binds a PUB socket and publishes typed messages.

    Args:
        endpoint: ``tcp://...`` to bind. Defaults to ``ZMQ_TICK_ENDPOINT`` from config.
        role: publisher identity stamped on heartbeats.
        hwm: send high-water mark.
        linger_ms: socket linger on close.
        context: share an existing context, or let the publisher own a private one.
        clock: injected for testability.
    """

    __slots__ = ("_clock", "_ctx", "_owns_ctx", "_role", "_seq", "_socket", "endpoint")

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        role: str = "ingestor",
        hwm: int = DEFAULT_HWM,
        linger_ms: int = DEFAULT_LINGER_MS,
        context: zmq.Context[zmq.Socket[bytes]] | None = None,
        settings: Settings | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if endpoint is not None:
            self.endpoint: str = endpoint
        else:
            resolved = settings if settings is not None else get_settings()
            self.endpoint = resolved.zmq_tick_endpoint
        self._role = role
        self._clock = clock
        self._seq = 0

        self._owns_ctx = context is None
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context() if context is None else context
        self._socket: zmq.Socket[bytes] = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, hwm)
        # LINGER is set once, honouring the caller's value. libzmq already applies
        # SO_REUSEADDR to TCP binds internally, and pyzmq exposes no SO_REUSEADDR socket
        # option (hasattr(zmq, "SO_REUSEADDR") is False), so a manual set is dead code that
        # only pretends to guard against EADDRINUSE on rapid restarts.
        self._socket.setsockopt(zmq.LINGER, linger_ms)
        self._socket.bind(self.endpoint)

        _log.info("ipc.publisher.bound", endpoint=self.endpoint, role=role, hwm=hwm)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the socket, and the context if we own it. Safe to call twice."""
        if not self._socket.closed:
            self._socket.close()
        if self._owns_ctx and not self._ctx.closed:
            self._ctx.term()
        _log.info("ipc.publisher.closed", endpoint=self.endpoint, role=self._role)

    @staticmethod
    def settle(seconds: float = DEFAULT_SETTLE_SECONDS) -> None:
        """Wait out the PUB/SUB slow-joiner window after binding.

        Only meaningful at startup. Never call this on the tick path.
        """
        _time.sleep(seconds)

    @property
    def closed(self) -> bool:
        return bool(self._socket.closed)

    @property
    def heartbeat_seq(self) -> int:
        """Number of heartbeats published since process start."""
        return self._seq

    # ── publishing ───────────────────────────────────────────────────────────

    def publish_raw(self, topic: bytes, payload: bytes) -> None:
        """Send a pre-encoded multipart frame.

        Uses ``NOBLOCK``: if the send buffer is full the message is dropped rather than
        blocking the ingestion thread. Blocking here would let a stalled consumer apply
        back-pressure all the way to the WebSocket, which is exactly what the HWM exists to
        prevent.
        """
        try:
            self._socket.send_multipart([topic, payload], flags=zmq.NOBLOCK)
        except zmq.Again:
            _log.warning("ipc.publisher.dropped", topic=topic.decode(errors="replace"))

    def publish(self, topic: bytes, message: WireMessage) -> None:
        """Encode and publish a message on ``topic``."""
        self.publish_raw(topic, encode(message))

    def publish_tick(self, symbol: str, tick: Tick) -> None:
        """Publish a tick on ``TICK.<symbol>``."""
        self.publish_raw(tick_topic(symbol), encode(tick))

    def publish_orderbook(self, symbol: str, book: OrderBook) -> None:
        """Publish L2 depth on ``DEPTH.<symbol>``."""
        self.publish_raw(depth_topic(symbol), encode(book))

    def publish_heartbeat(self) -> Heartbeat:
        """Publish a liveness ping on ``FEED.HEARTBEAT`` and return what was sent.

        Call this on a timer — at least twice per
        :data:`~tachyon.core.constants.FEED_STALE_AFTER` window, so a single dropped heartbeat
        cannot on its own be mistaken for a dead feed. It is what lets the Brain tell "the
        market is quiet" apart from "the WebSocket died", and only one of those halts trading.
        """
        self._seq += 1
        beat = Heartbeat(ts_epoch=self._clock.now().timestamp(), seq=self._seq, role=self._role)
        self.publish_raw(HEARTBEAT_TOPIC_BYTES, encode(beat))
        return beat
