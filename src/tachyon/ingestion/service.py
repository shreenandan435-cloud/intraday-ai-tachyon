"""Ingestion daemon — Process A, CLAUDE.md §2.

Wires the SmartAPI WebSocket to the ZeroMQ tick spine:

``binary frame -> decoder -> SequenceManager -> Tick / OrderBook -> Publisher``

This process contains **no business logic**. It computes no indicators, forms no opinions and
places no orders. It must keep publishing when the Brain is dead, and it must survive anything
downstream of the socket failing.

Reconnect state continuity
--------------------------
Everything cumulative lives **here**, not in the WebSocket client: per-symbol sequence
numbers (:class:`SequenceManager`), the token→symbol map, and these counters. The client
reconnects and re-subscribes transparently, so a network blip costs a few ticks — never the
session's running volume, VWAP inputs or monotonic ordering. The client additionally runs a
tick-heartbeat watchdog (5 s without market data during market hours ⇒ forced reconnect);
its ``stale_reconnects`` counter rides along in the silent-feed warning below.


The heartbeat is gated on the subscription
---------------------------------------------
A heartbeat published while the WebSocket is down would be actively dangerous. The Brain's
:class:`~tachyon.ipc.monitor.FeedMonitor` treats *any* message — tick or heartbeat — as proof
of life, so a process-liveness heartbeat during an outage would hold the feed "fresh" while no
market data flowed at all, and the strategy would keep trading against a frozen book. That is
precisely the failure the 2-second staleness rule exists to catch.

So heartbeats are emitted only while :attr:`SmartApiFeedClient.subscribed` is true. During a
reconnect the stream goes silent, the Brain's monitor trips after 2 s, and entries are blocked
until real data resumes. Silence is the correct signal; a reassuring heartbeat is a lie.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict, dataclass
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, is_market_open
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.ingestion.decoder import (
    DEPTH_LEVELS,
    DepthLevel,
    PacketDecodeError,
    SnapQuote,
    SubscriptionMode,
    decode_snap_quote,
    iter_packets,
    timestamp_is_plausible,
)
from tachyon.ingestion.sequence import SequenceManager
from tachyon.ingestion.ws_client import (
    FeedCredentials,
    SmartApiFeedClient,
    subscriptions_from_settings,
)
from tachyon.ipc.publisher import DEFAULT_SETTLE_SECONDS, Publisher
from tachyon.ipc.schemas import OrderBook, Tick

_log = get_logger(__name__)

#: Two heartbeats per FEED_STALE_AFTER window, so one lost heartbeat cannot alone look like a
#: dead feed (CLAUDE.md §1, FEED_STALE_AFTER = 2 s).
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 1.0

#: How long a live subscription may deliver nothing, during market hours, before we say so.
#: Generous: a thin symbol can genuinely go quiet, and this must not become background noise.
SILENT_SUBSCRIPTION_WARN_SECONDS: Final[float] = 30.0

#: Repeat the silent-subscription warning every N intervals rather than every second.
_WARN_EVERY: Final[int] = 30


@dataclass(slots=True)
class IngestionStats:
    """Counters for observability. Never used for a trading decision."""

    frames: int = 0
    packets: int = 0
    ticks_published: int = 0
    books_published: int = 0
    heartbeats: int = 0
    heartbeats_suppressed: int = 0
    decode_errors: int = 0
    unknown_tokens: int = 0
    implausible_timestamps: int = 0


def _pad_depth(
    levels: tuple[DepthLevel, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Normalise a ladder to exactly five levels of prices and quantities.

    The wire schema fixes depth at five rungs so the Numba OBI kernel sees a constant shape
    (CLAUDE.md §3.1). A short ladder is padded with zeros — which the kernel already treats as
    an empty rung — and a long one is truncated. Never reordered: index 0 must stay the touch.
    """
    prices = [level.price for level in levels[:DEPTH_LEVELS]]
    quantities = [level.quantity for level in levels[:DEPTH_LEVELS]]
    while len(prices) < DEPTH_LEVELS:
        prices.append(0.0)
        quantities.append(0)
    return tuple(prices), tuple(quantities)


class IngestionService:
    """The ingestion daemon.

    Args:
        credentials: SmartAPI feed credentials (feed token comes from login).
        settings: resolved configuration; defaults to the process singleton.
        publisher: ZMQ publisher; one is created and owned if omitted.
        heartbeat_interval: seconds between heartbeats while subscribed.
    """

    __slots__ = (
        "_client",
        "_clock",
        "_heartbeat_interval",
        "_owns_publisher",
        "_publisher",
        "_sequences",
        "_settings",
        "_stopping",
        "_token_to_symbol",
        "stats",
    )

    def __init__(
        self,
        credentials: FeedCredentials,
        *,
        settings: Settings | None = None,
        publisher: Publisher | None = None,
        heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._heartbeat_interval = heartbeat_interval
        self._sequences = SequenceManager()
        self.stats = IngestionStats()
        self._stopping = asyncio.Event()

        # Tokens are the wire identity; symbols are what everything else speaks.
        self._token_to_symbol = {item.token: item.symbol for item in self._settings.watchlist}

        self._owns_publisher = publisher is None
        self._publisher = (
            publisher
            if publisher is not None
            else Publisher(self._settings.zmq_tick_endpoint, role="ingestor", clock=clock)
        )

        self._client = SmartApiFeedClient(
            credentials,
            subscriptions_from_settings(self._settings),
            on_binary=self.handle_binary,
            mode=SubscriptionMode.SNAP_QUOTE,
            url=self._settings.feed.stream_url,
            backoff_seconds=self._settings.feed.reconnect_backoff_seconds,
            max_attempts=self._settings.feed.max_reconnect_attempts,
        )

    @property
    def client(self) -> SmartApiFeedClient:
        return self._client

    @property
    def sequences(self) -> SequenceManager:
        return self._sequences

    # ── the hot path ─────────────────────────────────────────────────────────

    def handle_binary(self, payload: bytes) -> None:
        """Decode one WebSocket frame and publish everything in it.

        Called from the event loop. Errors are counted and swallowed per packet: one corrupt
        packet must not cost us the rest of the frame, and must never drop the subscription.
        """
        self.stats.frames += 1
        try:
            packets = iter_packets(payload)
        except PacketDecodeError as exc:
            self.stats.decode_errors += 1
            _log.error("ingestion.frame_misaligned", error=str(exc), bytes=len(payload))
            return

        for packet in packets:
            self.stats.packets += 1
            try:
                self._publish_packet(packet)
            except PacketDecodeError as exc:
                self.stats.decode_errors += 1
                _log.error("ingestion.packet_decode_failed", error=str(exc), bytes=len(packet))

    def _publish_packet(self, packet: bytes) -> None:
        """Decode one packet and emit its Tick and OrderBook."""
        mode = packet[0]
        if mode != SubscriptionMode.SNAP_QUOTE:
            # We only ever subscribe in Snap Quote; anything else means the subscription or
            # the protocol changed underneath us, and depth would be missing.
            self.stats.decode_errors += 1
            _log.error("ingestion.unexpected_mode", mode=mode)
            return

        quote = decode_snap_quote(packet)

        symbol = self._token_to_symbol.get(quote.token)
        if symbol is None:
            # Never publish an instrument absent from the watchlist (CLAUDE.md §8.1).
            self.stats.unknown_tokens += 1
            _log.warning("ingestion.unknown_token", token=quote.token)
            return

        if not timestamp_is_plausible(quote.exchange_timestamp_ms):
            self.stats.implausible_timestamps += 1
            _log.warning(
                "ingestion.implausible_timestamp",
                token=quote.token,
                exchange_timestamp_ms=quote.exchange_timestamp_ms,
                hint="expected epoch milliseconds — verify the protocol version",
            )

        self._publish_tick(symbol, quote)
        self._publish_book(symbol, quote)

    def _publish_tick(self, symbol: str, quote: SnapQuote) -> None:
        tick = Tick(
            token=quote.token,
            ltp=quote.ltp,
            volume=quote.volume,
            ts_epoch=quote.ts_epoch,
            seq=self._sequences.next_for(quote.token),
        )
        self._publisher.publish_tick(symbol, tick)
        self.stats.ticks_published += 1

    def _publish_book(self, symbol: str, quote: SnapQuote) -> None:
        bid_prices, bid_qtys = _pad_depth(quote.bids)
        ask_prices, ask_qtys = _pad_depth(quote.asks)
        book = OrderBook(
            token=quote.token,
            bid_price=bid_prices,  # type: ignore[arg-type]
            bid_qty=bid_qtys,  # type: ignore[arg-type]
            ask_price=ask_prices,  # type: ignore[arg-type]
            ask_qty=ask_qtys,  # type: ignore[arg-type]
            ts_epoch=quote.ts_epoch,
        )
        self._publisher.publish_orderbook(symbol, book)
        self.stats.books_published += 1

    # ── loops ────────────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Emit a heartbeat every interval, but only while genuinely subscribed."""
        silent_intervals = 0
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self._heartbeat_interval)
            if self._stopping.is_set():
                return

            if self._client.subscribed:
                self._publisher.publish_heartbeat()
                self.stats.heartbeats += 1
                silent_intervals = self._warn_if_silent(silent_intervals)
            else:
                # Deliberate silence: let the Brain's FeedMonitor go stale and block entries.
                self.stats.heartbeats_suppressed += 1
                silent_intervals = 0

    def _warn_if_silent(self, silent_intervals: int) -> int:
        """Escalate a subscription that is up but has never delivered a packet.

        ``stats.heartbeats`` counts *our own* liveness, not the broker's: it increments once a
        second purely because the socket is open and subscribed. A rising heartbeat count
        alongside ``packets=0`` therefore looks reassuring and means the opposite — the
        subscription was accepted (or silently dropped) and no market data is flowing. Saying
        so out loud is the difference between a five-minute diagnosis and an hour of one.

        Only warns while the market is actually open: no packets at 09:05 is correct, not a
        fault, and a warning that cries wolf pre-open is a warning nobody reads at 09:20.
        """
        if self.stats.packets > 0 or not is_market_open(clock=self._clock):
            return 0

        silent_intervals += 1
        elapsed = silent_intervals * self._heartbeat_interval
        if elapsed >= SILENT_SUBSCRIPTION_WARN_SECONDS and silent_intervals % _WARN_EVERY == 0:
            _log.warning(
                "ingestion.subscribed_but_silent",
                seconds=round(elapsed),
                binary_frames=self._client.stats.binary_frames,
                text_frames=self._client.stats.text_frames,
                heartbeats=self.stats.heartbeats,
                seconds_since_last_tick=self._client.seconds_since_last_tick,
                stale_reconnects=self._client.stats.stale_reconnects,
                meaning="the socket is up and subscribed but the broker has sent no market "
                "data; heartbeats count our own liveness, not the feed's",
                check="look for a feed.text_frame warning carrying the broker's rejection",
            )
        return silent_intervals

    async def run(self) -> None:
        """Run until stopped or the feed gives up. Never returns normally on its own."""
        _log.info(
            "ingestion.starting",
            endpoint=self._publisher.endpoint,
            symbols=sorted(self._token_to_symbol.values()),
            heartbeat_interval=self._heartbeat_interval,
        )
        # Let subscribers finish connecting before the first tick (PUB/SUB slow joiner).
        # Await an async sleep: Publisher.settle() is a blocking time.sleep and must never
        # run on the event loop, which is the tick path itself.
        await asyncio.sleep(DEFAULT_SETTLE_SECONDS)

        feed = asyncio.create_task(self._client.run(), name="feed")
        heartbeat = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")

        try:
            done, pending = await asyncio.wait(
                (feed, heartbeat), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            for task in done:
                task.result()  # re-raise whatever ended the service
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the feed, flush counters and release the publisher if we own it."""
        self._stopping.set()
        await self._client.stop()
        # asdict, not vars(): IngestionStats uses slots=True and has no __dict__.
        _log.info("ingestion.stopped", **asdict(self.stats))
        if self._owns_publisher and not self._publisher.closed:
            self._publisher.close()
