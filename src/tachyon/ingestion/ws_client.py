"""SmartAPI WebSocket 2.0 async client — CLAUDE.md §2, §5.

Owns exactly one thing: keeping a live subscription to Angel One's binary feed and handing
raw frames to a callback. It performs no decoding, no sequencing and no publishing — the
ingestion process must keep running even when everything downstream of it is broken.

Reconnection
------------
A dropped socket is routine, not exceptional: networks blip, brokers restart, laptops sleep.
It must never take the process down. Failures are retried on an exponential backoff schedule.

The backoff resets only after a connection has been **stably** up for
:data:`STABLE_CONNECTION_SECONDS`, not merely on connect. A server that accepts the handshake
and immediately drops it would otherwise reset the counter every cycle and turn the backoff
into a hot reconnect loop — the exact behaviour that gets an API key throttled or banned.

Subscription
------------
Requests going *up* the socket are **Text** frames; market data comes back as **Binary**.
``websockets`` chooses the opcode from the argument type — ``send(str)`` is text,
``send(bytes)`` is binary — so :meth:`SmartApiFeedClient.subscription_payload` deliberately
returns ``str``. SmartStream discards a JSON request that arrives in a binary frame and sends
no error in reply, which looks identical to a subscription that was accepted but has nothing
to report: connection up, pongs flowing, zero packets forever.

Liveness
--------
Angel One expects an application-level ``ping`` text frame; the library's own protocol-level
ping is disabled because the server is not guaranteed to answer it, and an unanswered
protocol ping makes ``websockets`` close a perfectly healthy connection.

Because our ping draws a ``pong``, silence is genuinely diagnostic: no frame of any kind
within :data:`READ_TIMEOUT_SECONDS` means the connection is dead even if TCP has not noticed,
so we tear it down and reconnect rather than sit on a socket that will never deliver again.
A quiet market still produces pongs, so this does not misfire at lunchtime.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Final, Self

import msgspec
import websockets
from websockets.asyncio.client import ClientConnection, connect

from tachyon.core.config import Settings
from tachyon.core.logger import get_logger
from tachyon.ingestion.decoder import SubscriptionMode

_log = get_logger(__name__)

SMART_STREAM_URL: Final[str] = "wss://smartapisocket.angelone.in/smart-stream"

#: Application-level keepalive cadence. Angel One's documented guidance is every 30 s.
PING_INTERVAL_SECONDS: Final[float] = 25.0

#: No frame at all within this window means the connection is dead. Comfortably longer than
#: two ping cycles, so an ordinary pong gap cannot trip it.
READ_TIMEOUT_SECONDS: Final[float] = 60.0

#: How long a connection must survive before its success "counts" and the backoff resets.
STABLE_CONNECTION_SECONDS: Final[float] = 30.0

ACTION_SUBSCRIBE: Final[int] = 1
ACTION_UNSUBSCRIBE: Final[int] = 0

_ENCODER: Final[msgspec.json.Encoder] = msgspec.json.Encoder()

#: Callback for a raw binary frame. Must be fast and non-blocking — it runs on the event loop.
BinaryHandler = Callable[[bytes], None]


class FeedAuthenticationError(RuntimeError):
    """The broker rejected our credentials. Retrying will not help."""


@dataclass(frozen=True, slots=True)
class FeedCredentials:
    """Credentials for the streaming endpoint.

    ``generateSession`` issues *two* distinct credentials and the handshake wants both in
    different places: ``jwtToken`` in ``Authorization`` and ``feedToken`` in ``x-feed-token``.
    They are not interchangeable — sending the feed token as the bearer is the shape of failure
    where the socket opens, the subscription is acknowledged, and no market data ever arrives.

    ``jwt_token`` defaults to empty for the manual ``SMARTAPI_FEED_TOKEN`` fallback path, which
    has no login and therefore no JWT. In that case the feed token is sent as the bearer, which
    is the old behaviour — degraded, but no worse than before.
    """

    api_key: str
    client_code: str
    feed_token: str
    jwt_token: str = ""

    @classmethod
    def from_settings(cls, settings: Settings, feed_token: str, jwt_token: str = "") -> Self:
        """Take the API key and client code from config, both tokens from login."""
        return cls(
            api_key=settings.smartapi_api_key.get_secret_value(),
            client_code=settings.smartapi_client_code.get_secret_value(),
            feed_token=feed_token,
            jwt_token=jwt_token,
        )

    def headers(self) -> dict[str, str]:
        """Handshake headers. Never log the result — every value is a credential."""
        return {
            "Authorization": f"Bearer {self.jwt_token}" if self.jwt_token else self.feed_token,
            "x-api-key": self.api_key,
            "x-client-code": self.client_code,
            "x-feed-token": self.feed_token,
        }

    def is_complete(self) -> bool:
        return bool(self.api_key and self.client_code and self.feed_token)

    def has_bearer(self) -> bool:
        """True when a real JWT is available for the ``Authorization`` header."""
        return bool(self.jwt_token)


@dataclass(frozen=True, slots=True)
class TokenSubscription:
    """Instrument tokens to subscribe to, grouped by exchange segment.

    ``tokens`` are strings on the wire and validated as such here. SmartStream quietly ignores
    a ``tokenList`` entry whose tokens are JSON numbers, and an ignored subscription is
    indistinguishable from a quiet market: the socket stays up, pongs keep arriving, and no
    packet ever lands.
    """

    exchange_type: int
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("a TokenSubscription must carry at least one token")
        offenders = [token for token in self.tokens if not isinstance(token, str)]
        if offenders:
            raise TypeError(
                f"instrument tokens must be strings on the wire, got {offenders!r}; "
                "a JSON number here is silently dropped by SmartStream"
            )
        if any(not token.strip() for token in self.tokens):
            raise ValueError("instrument tokens must be non-empty")
        if not isinstance(self.exchange_type, int) or isinstance(self.exchange_type, bool):
            raise TypeError(f"exchange_type must be an int, got {self.exchange_type!r}")


@dataclass(slots=True)
class FeedStats:
    """Counters for observability. Not used for any trading decision."""

    connects: int = 0
    reconnects: int = 0
    binary_frames: int = 0
    text_frames: int = 0
    handler_errors: int = 0
    read_timeouts: int = 0
    last_error: str | None = None


class SmartApiFeedClient:
    """Maintains a subscribed SmartAPI WebSocket and streams binary frames to a callback.

    Args:
        credentials: API key, client code and feed token.
        subscriptions: tokens grouped by exchange type.
        on_binary: called with every binary frame. Must not block the event loop.
        mode: subscription mode. Snap Quote (3) is required for L2 depth.
        backoff_seconds: reconnect delay schedule; the last value repeats.
        max_attempts: consecutive failures before giving up. ``None`` retries forever.
    """

    __slots__ = (
        "_backoff",
        "_connection",
        "_credentials",
        "_max_attempts",
        "_mode",
        "_on_binary",
        "_ping_interval",
        "_read_timeout",
        "_stopping",
        "_subscribed",
        "_subscriptions",
        "_url",
        "stats",
    )

    def __init__(
        self,
        credentials: FeedCredentials,
        subscriptions: Sequence[TokenSubscription],
        *,
        on_binary: BinaryHandler,
        mode: SubscriptionMode = SubscriptionMode.SNAP_QUOTE,
        url: str = SMART_STREAM_URL,
        backoff_seconds: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 30.0),
        max_attempts: int | None = 20,
        ping_interval: float = PING_INTERVAL_SECONDS,
        read_timeout: float = READ_TIMEOUT_SECONDS,
    ) -> None:
        if not subscriptions:
            raise ValueError("at least one TokenSubscription is required")
        if not backoff_seconds:
            raise ValueError("backoff_seconds must not be empty")

        self._credentials = credentials
        self._subscriptions = tuple(subscriptions)
        self._on_binary = on_binary
        self._mode = mode
        self._url = url
        self._backoff = tuple(backoff_seconds)
        self._max_attempts = max_attempts
        self._ping_interval = ping_interval
        self._read_timeout = read_timeout

        self._stopping = asyncio.Event()
        self._connection: ClientConnection | None = None
        self._subscribed = False
        self.stats = FeedStats()

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def subscribed(self) -> bool:
        """True only while a live connection has an accepted subscription.

        This, not :attr:`connected`, is what gates the heartbeat: a socket that is open but
        not subscribed delivers no market data, and telling the Brain otherwise would keep it
        trading against a frozen view of the book.
        """
        return self._subscribed

    def subscription_payload(self, action: int = ACTION_SUBSCRIBE) -> str:
        """Build the subscribe/unsubscribe request.

        Returns ``str``, and that is load-bearing rather than cosmetic. ``websockets`` picks
        the frame opcode from the argument type: ``str`` sends a **Text** frame, ``bytes``
        sends a **Binary** frame. SmartStream's control channel is text — market data flows
        back as binary, but requests going *up* must be text. A JSON request delivered in a
        binary frame is discarded without an error reply, which presents exactly as a healthy
        socket that never delivers a tick.

        ``exchangeType`` is coerced to a plain ``int`` (1 = NSE cash) and tokens are emitted as
        JSON strings; :class:`TokenSubscription` enforces the latter at construction.
        """
        return _ENCODER.encode(
            {
                "correlationID": "tachyon",
                "action": action,
                "params": {
                    "mode": int(self._mode),
                    "tokenList": [
                        {"exchangeType": int(sub.exchange_type), "tokens": list(sub.tokens)}
                        for sub in self._subscriptions
                    ],
                },
            }
        ).decode()

    def backoff_for(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (0-based). The final value repeats indefinitely."""
        return self._backoff[min(attempt, len(self._backoff) - 1)]

    async def stop(self) -> None:
        """Ask the run loop to exit and close the socket."""
        self._stopping.set()
        connection = self._connection
        if connection is not None:
            with contextlib.suppress(Exception):
                await connection.close()

    async def run(self) -> None:
        """Connect, subscribe and stream until :meth:`stop`, reconnecting as needed.

        Raises:
            FeedAuthenticationError: credentials are missing or rejected. Not retried —
                backing off against a permanent rejection just burns the rate limit.
            ConnectionError: ``max_attempts`` consecutive failures.
        """
        if not self._credentials.is_complete():
            raise FeedAuthenticationError(
                "SmartAPI feed credentials incomplete: api_key, client_code and feed_token "
                "are all required (feed_token comes from generateSession at login)."
            )

        attempt = 0
        while not self._stopping.is_set():
            loop = asyncio.get_running_loop()
            connected_at = loop.time()
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except FeedAuthenticationError:
                raise
            except (OSError, TimeoutError, websockets.WebSocketException) as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._connection = None

            if self._stopping.is_set():
                break

            uptime = loop.time() - connected_at
            if uptime >= STABLE_CONNECTION_SECONDS:
                # The connection proved itself; treat the next failure as the first.
                attempt = 0
            else:
                attempt += 1

            if self._max_attempts is not None and attempt >= self._max_attempts:
                raise ConnectionError(
                    f"SmartAPI feed failed {attempt} consecutive times; "
                    f"last error: {self.stats.last_error}"
                )

            delay = self.backoff_for(attempt)
            self.stats.reconnects += 1
            _log.warning(
                "feed.reconnecting",
                attempt=attempt,
                delay_seconds=delay,
                uptime_seconds=round(uptime, 1),
                error=self.stats.last_error,
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)

        # asdict, not vars(): FeedStats uses slots=True and therefore has no __dict__.
        _log.info("feed.stopped", **asdict(self.stats))

    # ── one connection ───────────────────────────────────────────────────────

    async def _session(self) -> None:
        """Hold a single connection open until it fails or we are asked to stop."""
        async with connect(
            self._url,
            additional_headers=self._credentials.headers(),
            # Library-level pings are disabled in favour of the application-level "ping"
            # Angel One expects; see the module docstring.
            ping_interval=None,
            close_timeout=5.0,
            max_size=None,
        ) as connection:
            self._connection = connection
            self.stats.connects += 1
            _log.info(
                "feed.connected",
                url=self._url,
                mode=int(self._mode),
                tokens=sum(len(sub.tokens) for sub in self._subscriptions),
            )

            # A str, so websockets emits a Text frame. Sending bytes here would be a Binary
            # frame, which SmartStream drops silently — see the module docstring.
            await connection.send(self.subscription_payload(ACTION_SUBSCRIBE))
            self._subscribed = True
            _log.info("feed.subscribed", mode=int(self._mode))

            keepalive = asyncio.create_task(self._keepalive(connection))
            try:
                await self._consume(connection)
            finally:
                self._subscribed = False
                keepalive.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keepalive

    async def _consume(self, connection: ClientConnection) -> None:
        """Read frames until the connection dies or goes silent."""
        while not self._stopping.is_set():
            try:
                message = await asyncio.wait_for(connection.recv(), timeout=self._read_timeout)
            except TimeoutError:
                self.stats.read_timeouts += 1
                _log.warning(
                    "feed.read_timeout",
                    seconds=self._read_timeout,
                    action="treating connection as dead and reconnecting",
                )
                raise

            if isinstance(message, bytes):
                self.stats.binary_frames += 1
                self._dispatch(message)
            else:
                self.stats.text_frames += 1
                self._handle_text(message)

    def _dispatch(self, payload: bytes) -> None:
        """Hand a frame to the consumer, absorbing handler failures.

        A malformed packet or a downstream bug must not drop the subscription: reconnecting
        loses market data, and the ingestion process is the one component with nothing to fall
        back on.
        """
        try:
            self._on_binary(payload)
        except Exception as exc:  # noqa: BLE001 - a handler fault must not drop the feed
            self.stats.handler_errors += 1
            _log.error(
                "feed.handler_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                bytes=len(payload),
                exc_info=True,
            )

    def _handle_text(self, message: str) -> None:
        """Text frames are keepalive replies and errors; they never carry market data."""
        text = message.strip()
        if text.lower() == "pong":
            return
        _log.warning("feed.text_frame", message=text[:500])

    async def _keepalive(self, connection: ClientConnection) -> None:
        """Send the application-level ``ping`` Angel One expects."""
        while not self._stopping.is_set():
            await asyncio.sleep(self._ping_interval)
            try:
                await connection.send("ping")
            except OSError, websockets.WebSocketException:
                # The read loop owns reconnection; just stop pinging a dead socket.
                return


def subscriptions_from_settings(settings: Settings) -> tuple[TokenSubscription, ...]:
    """Group the configured watchlist into per-exchange subscription requests.

    Only watchlist symbols are ever subscribed — trading an instrument absent from the
    watchlist is forbidden (CLAUDE.md §8.1), and subscribing to one invites exactly that.
    """
    from tachyon.ingestion.decoder import ExchangeType

    exchange_codes = {
        "NSE": ExchangeType.NSE_CM,
        "BSE": ExchangeType.BSE_CM,
        "NFO": ExchangeType.NSE_FO,
    }

    grouped: dict[int, list[str]] = {}
    for item in settings.watchlist:
        code = int(exchange_codes[item.exchange])
        grouped.setdefault(code, []).append(item.token)

    return tuple(
        TokenSubscription(exchange_type=code, tokens=tuple(tokens))
        for code, tokens in sorted(grouped.items())
    )
