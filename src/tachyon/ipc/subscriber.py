"""ZeroMQ SUB side of the tick spine — CLAUDE.md §2.1.

Contract:

* ``RCVHWM = 10_000``, ``LINGER = 0``. A subscriber that falls behind loses messages. That is
  correct: a stale tick has no value, and queueing them would only delay the strategy further
  while consuming memory.
* The subscriber **connects**; the publisher binds.
* Frames are multipart ``[topic][payload]``.

Conflation
----------
``ZMQ_CONFLATE`` keeps only the most recent message on a socket. It is appropriate for the UI,
which only ever renders the latest value, and **forbidden** for the strategy and risk
subscribers — VWAP and OBI are accumulators, and a VWAP computed from a sampled subset of
prints is not a VWAP (CLAUDE.md §3.1). :class:`Subscriber` refuses to construct a conflating
socket for those roles.

We implement conflation in userspace rather than setting the ``ZMQ_CONFLATE`` socket option,
because that option **does not support multipart messages** — measured against libzmq 4.3.5,
a conflating SUB socket receives *zero* multipart frames rather than the latest one. Silently
receiving nothing is the worst available failure mode for a liveness-sensitive UI, so
:meth:`Subscriber.drain_latest` provides latest-only semantics over the multipart wire.
It is also strictly better than the socket option: conflation is per *topic* rather than
per socket, so a UI watching five symbols keeps the newest tick for each instead of only the
newest tick overall.

Sync and async
--------------
Two classes, one wire contract and one decoder:

* :class:`Subscriber` — blocking. For threads and scripts.
* :class:`AsyncSubscriber` — ``zmq.asyncio``. For the Brain (CLAUDE.md §2.2), where a blocking
  ``recv`` would stall tick ingestion, the P&L update, the broker call and the UI push at once.

They share :func:`decode_frames`, so there is exactly one implementation of "bytes on the wire
become a typed message". A second copy would eventually disagree with the first about a
malformed frame, and the disagreement would be silent.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Final, Self

import msgspec
import zmq
import zmq.asyncio

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.ipc.monitor import FeedMonitor
from tachyon.ipc.schemas import (
    SCHEMA_VERSION,
    TOPIC_HEARTBEAT,
    AnyMessage,
    Heartbeat,
    UnknownTopicError,
    decode_for_topic,
)

_log = get_logger(__name__)

DEFAULT_HWM: Final[int] = 10_000
DEFAULT_LINGER_MS: Final[int] = 0


class SubscriberRole(StrEnum):
    """Who is consuming, which determines whether conflation is permissible."""

    STRATEGY = "STRATEGY"
    RISK = "RISK"
    EXECUTION = "EXECUTION"
    UI = "UI"
    TELEMETRY = "TELEMETRY"


#: Only presentation-layer consumers may skip messages. Everything that computes state from
#: the sequence of ticks must see all of them.
CONFLATION_ALLOWED: Final[frozenset[SubscriberRole]] = frozenset(
    {SubscriberRole.UI, SubscriberRole.TELEMETRY}
)


class ConflationForbiddenError(RuntimeError):
    """Conflation was requested by a role that must observe every message."""


class SchemaVersionMismatchError(RuntimeError):
    """The publisher is speaking a different wire version than this process understands."""


@dataclass(frozen=True, slots=True)
class Envelope:
    """One decoded message plus its delivery metadata."""

    topic: str
    message: AnyMessage
    """Any message this system publishes — tick spine or state spine. Consumers narrow with
    ``isinstance``; the topic prefix already told the decoder which type to produce."""

    received_mono: float


@dataclass(slots=True)
class DecodeStats:
    """Per-subscriber counters, mutated by :func:`decode_frames`."""

    received: int = 0
    decode_errors: int = 0
    version_checked: bool = field(default=False)


def decode_frames(
    frames: Sequence[bytes],
    stats: DecodeStats,
    *,
    monitor: FeedMonitor | None = None,
    clock: Clock = SYSTEM_CLOCK,
) -> Envelope | None:
    """Decode one multipart frame pair, or ``None`` if it is unusable.

    A malformed payload is counted and dropped rather than raised: one corrupt frame must not
    take down the strategy process, and :class:`~tachyon.ipc.monitor.FeedMonitor` will catch a
    publisher that has genuinely stopped producing valid data.

    Raises:
        SchemaVersionMismatchError: the first heartbeat announces a wire version this process
            does not understand. That one *is* fatal — continuing would mean interpreting
            prices under the wrong schema.
    """
    if len(frames) != 2:
        stats.decode_errors += 1
        _log.error("ipc.subscriber.malformed_frame", frame_count=len(frames))
        return None

    topic = frames[0].decode(errors="replace")
    try:
        message = decode_for_topic(topic, frames[1])
    except (msgspec.DecodeError, msgspec.ValidationError, UnknownTopicError) as exc:
        stats.decode_errors += 1
        _log.error(
            "ipc.subscriber.decode_failed",
            topic=topic,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return None

    if isinstance(message, Heartbeat) and not stats.version_checked:
        stats.version_checked = True
        if message.schema_version != SCHEMA_VERSION:
            raise SchemaVersionMismatchError(
                f"Publisher {message.role!r} speaks schema v{message.schema_version}, "
                f"this process expects v{SCHEMA_VERSION}. Redeploy both sides together."
            )

    stats.received += 1
    if monitor is not None:
        monitor.record(wire_epoch=message.ts_epoch)

    return Envelope(topic=topic, message=message, received_mono=clock.monotonic())


class Subscriber:
    """Connects a SUB socket and yields decoded, typed messages.

    Args:
        topics: topic prefixes to subscribe to. ``("TICK.",)`` takes every symbol's ticks;
            ``("TICK.INFY",)`` takes one. An empty sequence subscribes to everything.
        role: consumer role. Governs whether conflation may be enabled.
        endpoint: ``tcp://...`` to connect to. Defaults to ``ZMQ_TICK_ENDPOINT`` from config.
        conflate: keep only the newest message per topic. UI/telemetry only.
        monitor: if supplied, :meth:`FeedMonitor.record` is called on every received message,
            so liveness tracking cannot drift out of sync with actual delivery.

    Raises:
        ConflationForbiddenError: ``conflate=True`` with a role that must see every message.

    Sockets are not thread-safe — one subscriber per thread.
    """

    __slots__ = (
        "_clock",
        "_conflate",
        "_ctx",
        "_monitor",
        "_owns_ctx",
        "_role",
        "_socket",
        "_stats",
        "endpoint",
    )

    def __init__(
        self,
        topics: Sequence[str],
        *,
        role: SubscriberRole,
        endpoint: str | None = None,
        conflate: bool = False,
        hwm: int = DEFAULT_HWM,
        linger_ms: int = DEFAULT_LINGER_MS,
        context: zmq.Context[zmq.Socket[bytes]] | None = None,
        settings: Settings | None = None,
        monitor: FeedMonitor | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if conflate and role not in CONFLATION_ALLOWED:
            raise ConflationForbiddenError(
                f"Conflation requested for role {role}. VWAP and OBI are accumulated over "
                f"every print (CLAUDE.md §3.1); sampling the stream silently corrupts them. "
                f"Conflation is permitted only for {sorted(CONFLATION_ALLOWED)}."
            )

        if endpoint is not None:
            self.endpoint: str = endpoint
        else:
            resolved = settings if settings is not None else get_settings()
            self.endpoint = resolved.zmq_tick_endpoint
        self._role = role
        self._conflate = conflate
        self._clock = clock
        self._monitor = monitor
        self._stats = DecodeStats()

        self._owns_ctx = context is None
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context() if context is None else context
        self._socket: zmq.Socket[bytes] = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, hwm)
        self._socket.setsockopt(zmq.LINGER, linger_ms)

        # NOTE: zmq.CONFLATE is deliberately never set — see the module docstring. It silently
        # discards multipart messages entirely.

        for topic in topics or ("",):
            self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._socket.connect(self.endpoint)

        _log.info(
            "ipc.subscriber.connected",
            endpoint=self.endpoint,
            role=role,
            topics=list(topics),
            conflate=conflate,
            hwm=hwm,
        )

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
        _log.info(
            "ipc.subscriber.closed",
            endpoint=self.endpoint,
            role=self._role,
            received=self._stats.received,
            decode_errors=self._stats.decode_errors,
        )

    @property
    def closed(self) -> bool:
        return bool(self._socket.closed)

    @property
    def received(self) -> int:
        return self._stats.received

    @property
    def decode_errors(self) -> int:
        return self._stats.decode_errors

    def subscribe(self, topic: str) -> None:
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())

    def unsubscribe(self, topic: str) -> None:
        self._socket.setsockopt(zmq.UNSUBSCRIBE, topic.encode())

    # ── receiving ────────────────────────────────────────────────────────────

    def _decode(self, frames: Sequence[bytes]) -> Envelope | None:
        return decode_frames(frames, self._stats, monitor=self._monitor, clock=self._clock)

    def recv(self, timeout_ms: int | None = None) -> Envelope | None:
        """Receive the next message.

        Args:
            timeout_ms: ``None`` blocks indefinitely; ``0`` polls; otherwise wait that long.

        Returns:
            The decoded envelope, or ``None`` on timeout or on an undecodable message.
        """
        if timeout_ms is not None and not self._socket.poll(timeout=timeout_ms):
            return None
        frames = self._socket.recv_multipart()
        return self._decode(frames)

    def drain_latest(self) -> dict[str, Envelope]:
        """Consume everything queued and return only the newest message per topic.

        This is the conflation path (see the module docstring) and is restricted to roles
        permitted to skip messages.

        Raises:
            ConflationForbiddenError: called on a non-conflating subscriber. Draining discards
                messages, and for the strategy that would corrupt VWAP and OBI silently.
        """
        if not self._conflate:
            raise ConflationForbiddenError(
                f"drain_latest() discards messages and is not available to role {self._role}. "
                f"Use recv() and process every message."
            )

        latest: dict[str, Envelope] = {}
        while True:
            try:
                frames = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return latest
            envelope = self._decode(frames)
            if envelope is not None:
                latest[envelope.topic] = envelope

    def __iter__(self) -> Iterator[Envelope]:
        """Yield messages until the socket closes.

        Undecodable messages are skipped, having already been counted and logged.
        """
        while not self._socket.closed:
            try:
                frames = self._socket.recv_multipart()
            except zmq.ContextTerminated:
                return
            envelope = self._decode(frames)
            if envelope is not None:
                yield envelope


class AsyncSubscriber:
    """asyncio-native SUB socket for the Brain — CLAUDE.md §2.2.

    Same wire, same decoder, same HWM/LINGER contract as :class:`Subscriber`; the only
    difference is that ``recv`` yields to the event loop instead of blocking it. That matters
    because the Brain multiplexes tick ingestion, broker calls and the UI push on one loop, and
    a blocking ``recv`` would stall all three.

    Args:
        topics: topic prefixes to subscribe to. Empty subscribes to everything.
        role: consumer role, recorded for diagnostics.
        endpoint: ``tcp://...`` to connect to. Defaults to ``ZMQ_TICK_ENDPOINT``.
        monitor: fed on every received message, so liveness tracking cannot drift out of sync
            with actual delivery.

    **No conflation, at any role.** The async subscriber exists for the strategy path, and
    VWAP and OBI accumulate over every print (CLAUDE.md §3.1). A UI that wants latest-only
    should use the sync :meth:`Subscriber.drain_latest`.

    Example::

        async with AsyncSubscriber(("TICK.", "DEPTH."), role=SubscriberRole.STRATEGY) as sub:
            async for envelope in sub:
                brain.on_message(envelope)
    """

    __slots__ = (
        "_clock",
        "_ctx",
        "_monitor",
        "_owns_ctx",
        "_role",
        "_socket",
        "_stats",
        "endpoint",
    )

    def __init__(
        self,
        topics: Sequence[str],
        *,
        role: SubscriberRole = SubscriberRole.STRATEGY,
        endpoint: str | None = None,
        hwm: int = DEFAULT_HWM,
        linger_ms: int = DEFAULT_LINGER_MS,
        context: zmq.asyncio.Context | None = None,
        settings: Settings | None = None,
        monitor: FeedMonitor | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        if endpoint is not None:
            self.endpoint: str = endpoint
        else:
            resolved = settings if settings is not None else get_settings()
            self.endpoint = resolved.zmq_tick_endpoint

        self._role = role
        self._clock = clock
        self._monitor = monitor
        self._stats = DecodeStats()

        self._owns_ctx = context is None
        self._ctx: zmq.asyncio.Context = zmq.asyncio.Context() if context is None else context
        self._socket: Any = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, hwm)
        self._socket.setsockopt(zmq.LINGER, linger_ms)

        for topic in topics or ("",):
            self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._socket.connect(self.endpoint)

        _log.info(
            "ipc.async_subscriber.connected",
            endpoint=self.endpoint,
            role=role,
            topics=list(topics),
            hwm=hwm,
        )

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
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
        _log.info(
            "ipc.async_subscriber.closed",
            endpoint=self.endpoint,
            role=self._role,
            received=self._stats.received,
            decode_errors=self._stats.decode_errors,
        )

    @property
    def closed(self) -> bool:
        return bool(self._socket.closed)

    @property
    def received(self) -> int:
        return self._stats.received

    @property
    def decode_errors(self) -> int:
        return self._stats.decode_errors

    def subscribe(self, topic: str) -> None:
        self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())

    # ── receiving ────────────────────────────────────────────────────────────

    async def recv(self, timeout_ms: int | None = None) -> Envelope | None:
        """Await the next message.

        Args:
            timeout_ms: ``None`` waits indefinitely (yielding to the loop throughout);
                otherwise poll for at most that long and return ``None`` if nothing arrived.

        Returns:
            The decoded envelope, or ``None`` on timeout or on an undecodable message.
        """
        if timeout_ms is not None:
            events = await self._socket.poll(timeout=timeout_ms)
            if not events:
                return None
        frames = await self._socket.recv_multipart()
        return decode_frames(frames, self._stats, monitor=self._monitor, clock=self._clock)

    async def __aiter__(self) -> AsyncIterator[Envelope]:
        """Yield messages until the socket closes.

        Undecodable messages are skipped, having already been counted and logged.
        """
        while not self._socket.closed:
            try:
                frames = await self._socket.recv_multipart()
            except zmq.ContextTerminated:
                return
            except zmq.ZMQError as exc:
                # A socket closed underneath a pending recv is an ordinary shutdown, not a
                # fault. Anything else is re-raised so it cannot be mistaken for one.
                if exc.errno in (zmq.ENOTSOCK, zmq.ENOTSUP):
                    return
                raise
            envelope = decode_frames(frames, self._stats, monitor=self._monitor, clock=self._clock)
            if envelope is not None:
                yield envelope


def heartbeat_subscriber(
    *,
    role: SubscriberRole = SubscriberRole.RISK,
    endpoint: str | None = None,
    monitor: FeedMonitor | None = None,
) -> Subscriber:
    """Subscriber for liveness only.

    A consumer that subscribes to ticks alone cannot distinguish a quiet market from a dead
    feed, so every process that makes trading decisions should also watch this topic.
    """
    return Subscriber((TOPIC_HEARTBEAT,), role=role, endpoint=endpoint, monitor=monitor)
