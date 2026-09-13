"""Angel One SmartAPI REST client — CLAUDE.md §6.4.

**The only module in the system that talks to the broker.** Nothing else opens a socket to
Angel One; everything goes through one rate-limited client so the limits are enforced in one
place rather than hoped for in five.

Why not the ``smartapi-python`` SDK
-----------------------------------
The SDK is synchronous and built on ``requests``. CLAUDE.md §2.2 bans blocking HTTP in an async
path, and the Brain is an asyncio process: a 2-second broker call on the event loop stalls tick
ingestion, the P&L update and the UI push simultaneously. So the REST protocol is spoken
directly over ``httpx.AsyncClient``. The SDK remains a declared dependency for its instrument
master helpers; it is never used on the order path.

Rate limiting
-------------
Angel One publishes per-endpoint limits and enforces them with temporary bans. A token bucket
guards every call: one bucket per endpoint, plus a global ceiling from
``settings.execution.rate_limit_per_second``. Callers *wait*; they are never rejected locally,
because a locally-dropped cancel is far worse than a slightly late one.

The buckets refill on :func:`time.monotonic`, not the wall clock — an NTP correction must not
hand out a burst of tokens.

Retry discipline — the part that matters
----------------------------------------
CLAUDE.md §6.4: *never retry a placement whose outcome is unknown.* A duplicate live order is
worse than a missed fill, and "the response timed out" does **not** mean "the order was not
placed" — it very often means the opposite.

So retries are classified by whether the request provably never reached the broker:

* ``ConnectError`` / ``ConnectTimeout`` / ``PoolTimeout`` — the request was never sent. Safe to
  retry, for any endpoint.
* Anything after the bytes went out — read timeout, dropped connection, 5xx — is **unknown**.
  For a read-only endpoint that is harmless and retried. For a placement, modification or
  cancellation it raises :class:`UnknownOrderOutcomeError`, which the caller must resolve by
  reconciling against the order book, never by re-sending.

Secrets
-------
The JWT, refresh token, feed token, password and TOTP are held in memory, sent in headers, and
never logged, never journaled, never published over IPC (CLAUDE.md §5, §8).
"""

from __future__ import annotations

import asyncio
import re
import socket
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from types import TracebackType
from typing import Any, Final, Self

import httpx
import pyotp

from tachyon.core import token_cache
from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import TradingMode
from tachyon.core.logger import get_logger
from tachyon.execution.journal import OrderJournal

_log = get_logger(__name__)

BASE_URL: Final[str] = "https://apiconnect.angelone.in"

#: Angel One error codes meaning "your JWT is no longer valid". Distinguished from ordinary
#: failures because they are the one class of error where retrying *after a refresh* is correct.
_TOKEN_ERROR_CODES: Final[frozenset[str]] = frozenset({"AG8001", "AG8002", "AG8003", "AB8050"})

#: Wall-clock ceiling on any single broker call. Longer than this and we are better off not
#: knowing the answer than blocking the event loop waiting for it.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 7.0

#: Transport failures that prove the request never left this machine. Only these are safe to
#: retry on a non-idempotent endpoint. See the module docstring.
_NEVER_SENT: Final[tuple[type[Exception], ...]] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

#: Order statuses in which an order can no longer execute. Anything *not* listed is treated as
#: live — fail-safe, because mistaking a working order for a dead one leaves it unmanaged, and
#: Angel One's status vocabulary has grown before.
TERMINAL_ORDER_STATUSES: Final[frozenset[str]] = frozenset(
    {"complete", "cancelled", "canceled", "rejected"}
)

#: Prefix every order we place carries in its ``ordertag`` (CLAUDE.md §6.4).
ORDER_TAG_PREFIX: Final[str] = "TCHYN-"

#: Endpoint names intercepted in PAPER mode — the three that move money.
#:
#: Listed explicitly rather than derived from ``Endpoint.idempotent``, because login and token
#: refresh are also non-idempotent and **must** reach the network in PAPER: a paper session
#: still needs a real feed token and a real order book, or its telemetry is fiction.
PAPER_INTERCEPTED: Final[frozenset[str]] = frozenset(
    {"place_order", "modify_order", "cancel_order"}
)

#: Prefix on every simulated order id. Angel One order ids are numeric strings, so this can
#: never collide with a real one — in a log line, in the journal, or in the order book.
PAPER_ORDER_PREFIX: Final[str] = "PAPER-"

#: Longest slice of a non-JSON error page allowed to reach a log line or the journal. WAF
#: interstitials are mostly markup; the rejection reason lives in the first few words.
_NON_JSON_SNIPPET_LIMIT: Final[int] = 300


def _normalize_totp_secret(secret: str) -> str:
    """Canonicalise a pasted TOTP secret before handing it to ``pyotp``.

    pyotp 2.10 restores missing ``=`` padding itself and casefolds on decode, but a secret
    pasted from an authenticator-app export with embedded whitespace still fails
    ``base64.b32decode`` ("Non-base32 digit found") — at login time, the worst moment to
    discover it. Strip whitespace, upper-case and pad explicitly so the decode cannot fail
    for formatting reasons.
    """
    cleaned = "".join(secret.upper().split())
    return cleaned + "=" * (-len(cleaned) % 8)


def _summarize_non_json_body(text: str) -> str:
    """Reduce a WAF/gateway error page to one log-safe line.

    Angel One's edge firewall answers blocked requests with an HTML interstitial rather
    than JSON. Stripping tags and collapsing whitespace surfaces the actual rejection
    ("Access Denied", "Request rejected") instead of a wall of markup, and the truncation
    keeps a full-page block from flooding the journal.
    """
    cleaned = " ".join(re.sub(r"<[^>]+>", " ", text).split())
    if len(cleaned) > _NON_JSON_SNIPPET_LIMIT:
        cleaned = cleaned[:_NON_JSON_SNIPPET_LIMIT] + "..."
    return cleaned or "<empty body>"


# ──────────────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────────────


class SmartApiError(RuntimeError):
    """The broker rejected a call, or the transport failed in a knowable way."""

    def __init__(self, endpoint: str, message: str, *, error_code: str = "") -> None:
        self.endpoint = endpoint
        self.error_code = error_code
        super().__init__(f"{endpoint}: {message}" + (f" [{error_code}]" if error_code else ""))


class SmartApiAuthError(SmartApiError):
    """Login failed or the session could not be refreshed. Retrying will not help."""


class UnknownOrderOutcomeError(SmartApiError):
    """A mutating call failed *after* it was transmitted. The broker's state is unknown.

    **Never retry on this.** The order may be live. The only correct response is to reconcile
    against the order book (:class:`~tachyon.execution.reconciliation.StateReconciler`) and act
    on what is actually there.
    """

    def __init__(self, endpoint: str, message: str, *, order_tag: str = "") -> None:
        self.order_tag = order_tag
        super().__init__(endpoint, f"OUTCOME UNKNOWN — {message} (order_tag={order_tag!r})")


class PaperModeError(RuntimeError):
    """A mutating broker call was attempted in PAPER with simulation disabled.

    The default PAPER behaviour is to *intercept and simulate* (see :class:`PaperBroker`), so
    this is only raised when a caller has explicitly asked for strict refusal instead. Either
    way the request never reaches the network.
    """


@dataclass(slots=True)
class PaperBroker:
    """Issues simulated responses for the three endpoints that move money.

    This is the **interception layer**, and it sits at the last point before ``httpx``: no
    mutating request can reach the network while ``TRADING_MODE`` is not LIVE, whatever the
    caller believes. :class:`~tachyon.execution.executor.RoboExecutor` already short-circuits
    in PAPER, so in the normal path this never fires — which is exactly the property worth
    having. It is the backstop for every *other* caller: the reconciler, the square-off path,
    a script, and whatever Phase 12 adds.

    Read-only endpoints are deliberately **not** intercepted. A paper session that invents its
    own order book and margin is not a rehearsal, it is a fiction, and the point of PAPER is to
    exercise the real code against the real broker state (CLAUDE.md §9).

    **PAPER does not simulate fills.** A simulated placement is accepted and then nothing
    happens to it: no fill arrives, so the position never closes and P&L stays at zero. PAPER
    validates the *path* — payload construction, tick alignment, rate limiting, journalling,
    response handling — not the *outcome*. Anyone reading a flat paper P&L should know it means
    "nothing was simulated", not "nothing was lost".
    """

    prefix: str = PAPER_ORDER_PREFIX
    issued: int = 0
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    def next_order_id(self) -> str:
        self.issued += 1
        return f"{self.prefix}{self.issued:06d}"

    def simulate(self, endpoint: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
        """Build the response the broker would have returned.

        Every reply carries ``"simulated": true``. A downstream consumer that cannot tell a
        paper fill from a real one is one refactor away from an expensive surprise.
        """
        if endpoint.name == "place_order":
            order_id = self.next_order_id()
            self.orders[order_id] = dict(payload)
            return {
                "script": payload.get("tradingsymbol", ""),
                "orderid": order_id,
                "uniqueorderid": order_id,
                "simulated": True,
            }

        order_id = str(payload.get("orderid", "")) or self.next_order_id()
        if endpoint.name == "cancel_order":
            self.orders.pop(order_id, None)
        else:
            self.orders[order_id] = {**self.orders.get(order_id, {}), **payload}
        return {"orderid": order_id, "uniqueorderid": order_id, "simulated": True}

    def reset_session(self) -> None:
        self.issued = 0
        self.orders.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Token bucket
# ──────────────────────────────────────────────────────────────────────────────


class TokenBucket:
    """Classic token bucket, refilled on the monotonic clock.

    Args:
        rate: sustained tokens per second.
        capacity: burst size. Defaults to ``rate`` — one second of allowance.
        clock: injected for testing.

    ``acquire`` waits rather than failing: a locally-dropped broker call is a decision this
    class is not entitled to make. The caller's own timeout is the backstop.
    """

    __slots__ = ("_capacity", "_clock", "_lock", "_rate", "_tokens", "_updated", "waits")

    def __init__(self, rate: float, *, capacity: float | None = None, clock: Clock = SYSTEM_CLOCK):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        self._clock = clock
        self._tokens = self._capacity
        self._updated = clock.monotonic()
        self._lock = asyncio.Lock()
        self.waits = 0

    @property
    def tokens(self) -> float:
        """Tokens available *right now*, without consuming any."""
        return min(self._capacity, self._tokens + (self._elapsed() * self._rate))

    def _elapsed(self) -> float:
        return max(0.0, self._clock.monotonic() - self._updated)

    def _refill(self) -> None:
        now = self._clock.monotonic()
        self._tokens = min(
            self._capacity, self._tokens + max(0.0, now - self._updated) * self._rate
        )
        self._updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Take ``tokens`` if available. Non-blocking; returns False if they are not."""
        self._refill()
        if self._tokens < tokens:
            return False
        self._tokens -= tokens
        return True

    def delay_for(self, tokens: float = 1.0) -> float:
        """Seconds until ``tokens`` would be available. Zero if they already are."""
        self._refill()
        deficit = tokens - self._tokens
        return 0.0 if deficit <= 0 else deficit / self._rate

    async def acquire(self, tokens: float = 1.0) -> float:
        """Wait until ``tokens`` are available, then take them. Returns seconds waited."""
        waited = 0.0
        async with self._lock:
            while True:
                delay = self.delay_for(tokens)
                if delay <= 0.0:
                    self._tokens -= tokens
                    return waited
                self.waits += 1
                waited += delay
                await asyncio.sleep(delay)


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One SmartAPI route, with the rate limit and retry class it belongs to.

    ``idempotent`` is not a REST nicety here — it is the switch that decides whether a failure
    after transmission may be retried. Every order-mutating route is False.
    """

    name: str
    method: str
    path: str
    limit_per_second: float
    idempotent: bool = False


#: Published Angel One limits, taken conservatively. Where a documented limit is generous the
#: value below is the smaller of it and what this system could plausibly need — a single-account
#: intraday system placing a handful of brackets does not benefit from sailing close to a ban.
LOGIN = Endpoint("login", "POST", "/rest/auth/angelbroking/user/v1/loginByPassword", 1.0, False)
REFRESH = Endpoint("refresh", "POST", "/rest/auth/angelbroking/jwt/v1/generateTokens", 1.0, False)
LOGOUT = Endpoint("logout", "POST", "/rest/secure/angelbroking/user/v1/logout", 1.0, False)
PLACE_ORDER = Endpoint("place_order", "POST", "/rest/secure/angelbroking/order/v1/placeOrder", 10.0)
MODIFY_ORDER = Endpoint(
    "modify_order", "POST", "/rest/secure/angelbroking/order/v1/modifyOrder", 10.0
)
CANCEL_ORDER = Endpoint(
    "cancel_order", "POST", "/rest/secure/angelbroking/order/v1/cancelOrder", 10.0
)
ORDER_BOOK = Endpoint(
    "order_book", "GET", "/rest/secure/angelbroking/order/v1/getOrderBook", 1.0, True
)
TRADE_BOOK = Endpoint(
    "trade_book", "GET", "/rest/secure/angelbroking/order/v1/getTradeBook", 1.0, True
)
POSITIONS = Endpoint(
    "positions", "GET", "/rest/secure/angelbroking/order/v1/getPosition", 1.0, True
)
RMS = Endpoint("rms", "GET", "/rest/secure/angelbroking/user/v1/getRMS", 2.0, True)
PROFILE = Endpoint("profile", "GET", "/rest/secure/angelbroking/user/v1/getProfile", 1.0, True)


# ──────────────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class BrokerSession:
    """Credentials issued by ``generateSession``. Every field is a secret."""

    jwt_token: str = ""
    refresh_token: str = ""
    feed_token: str = ""
    client_code: str = ""
    issued_at_ist: datetime | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.jwt_token and self.feed_token)

    def clear(self) -> None:
        self.jwt_token = ""
        self.refresh_token = ""
        self.feed_token = ""
        self.issued_at_ist = None


@dataclass(slots=True)
class ApiStats:
    """Counters for observability. Never used for a trading decision."""

    requests: int = 0
    responses: int = 0
    broker_rejections: int = 0
    transport_retries: int = 0
    unknown_outcomes: int = 0
    token_refreshes: int = 0
    rate_limit_waits: int = 0
    paper_intercepts: int = 0
    last_error: str | None = None


#: Static egress IP for the Giganode SSH tunnel. Angel One's WAF whitelists this
#: address; reporting the LAN IP in ``X-ClientPublicIP`` produces an HTTP 401/403
#: rejection (CLAUDE.md §6). Override via ``ANGEL_PUBLIC_IP`` for a different tunnel.
_GIGANODE_PUBLIC_IP: Final[str] = "87.76.191.175"


@dataclass(frozen=True, slots=True)
class ClientIdentity:
    """The ``X-Client*`` headers SmartAPI requires on every call.

    ``public_ip`` is the **egress** IP, not the local LAN address: Angel One's WAF
    whitelists a single static address (the Giganode tunnel endpoint at
    :data:`_GIGANODE_PUBLIC_IP`) and rejects every other source with HTTP 401/403.
    ``local_ip`` stays as the LAN address — it is informational, not whitelisted.

    The ``X-ClientPublicIP`` value is read from the ``ANGEL_PUBLIC_IP`` env var when set
    (so a non-Giganode tunnel can be wired without a code change), then from the constant,
    then — only if both are unavailable — falls back to the LAN address with a warning.
    Loopback must never be reported: Angel One's WAF rejects ``127.0.0.1`` in
    ``X-ClientLocalIP``/``X-ClientPublicIP`` with an HTTP 403 before the request reaches
    the API, so :meth:`detect` falls back to a standard dummy LAN address instead.

    Resolved once at construction so nothing on the async path performs a blocking DNS
    lookup or env lookup.
    """

    local_ip: str
    public_ip: str
    mac_address: str

    @classmethod
    def detect(cls) -> Self:
        local = cls._outbound_ip()
        public = cls._public_ip(local)
        node = uuid.getnode()
        mac = ":".join(f"{(node >> shift) & 0xFF:02X}" for shift in range(40, -1, -8))
        return cls(local_ip=local, public_ip=public, mac_address=mac)

    @staticmethod
    def _public_ip(fallback: str) -> str:
        """Resolve the static egress IP that the broker WAF whitelists.

        Order of precedence:

        1. ``ANGEL_PUBLIC_IP`` env var — explicit override (lets a non-Giganode tunnel
           be wired without a code change; ``angel_adapter.fetch_historical_candles``
           reads the same constant, so both REST and WebSocket egress match).
        2. :data:`_GIGANODE_PUBLIC_IP` — the Giganode tunnel endpoint. Hard-coded so a
           blank environment (a launcher that forgot to set the var) still reports the
           correct egress IP. Reporting the LAN IP here is the failure mode that
           produces the HTTP 401/403 from the broker.
        3. The detected LAN address, with a warning — only reached when the constant
           is somehow empty, which is a configuration error worth surfacing.
        """
        import os

        override = os.environ.get("ANGEL_PUBLIC_IP", "").strip()
        if override:
            return override
        if _GIGANODE_PUBLIC_IP:
            return _GIGANODE_PUBLIC_IP
        _log.warning(
            "execution.public_ip_fallback — ANGEL_PUBLIC_IP unset and the Giganode constant "
            "is empty; reporting the LAN address %s; Angel One will reject every call with "
            "HTTP 401/403 until the static IP is configured",
            fallback,
        )
        return fallback

    @staticmethod
    def _outbound_ip() -> str:
        """The LAN address the OS would route outbound traffic from.

        A UDP ``connect`` sends no packets — it only selects the egress interface — so this
        is a cheap one-shot lookup, safe to run once at construction.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.settimeout(1.0)
                probe.connect(("203.0.113.1", 80))  # TEST-NET-3: unroutable, nothing is sent
                candidate = str(probe.getsockname()[0])
            if candidate and not candidate.startswith("127."):
                return candidate
        except OSError:
            pass
        return "192.168.1.1"


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────


class SmartApiClient:
    """Rate-limited async client for Angel One SmartAPI.

    Args:
        settings: resolved config. Supplies credentials and the global rate ceiling.
        journal: append-only order journal. Every mutating call is recorded before it is made.
        transport: injected ``httpx`` transport, for tests.
        mode: PAPER refuses every mutating call outright.

    Example::

        async with SmartApiClient(settings=settings, journal=journal) as client:
            await client.login()
            book = await client.order_book()
    """

    __slots__ = (
        "_buckets",
        "_client",
        "_clock",
        "_global_bucket",
        "_identity",
        "_journal",
        "_max_retries",
        "_mode",
        "_owns_client",
        "_paper",
        "_rate_scale",
        "_refresh_lock",
        "_settings",
        "_simulate_paper",
        "_timeout",
        "session",
        "stats",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        journal: OrderJournal | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = BASE_URL,
        mode: TradingMode | None = None,
        clock: Clock = SYSTEM_CLOCK,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        identity: ClientIdentity | None = None,
        rate_limit_scale: float = 1.0,
        simulate_paper_orders: bool = True,
    ) -> None:
        """See the class docstring.

        ``rate_limit_scale`` multiplies every bucket's rate. It exists so a test suite against a
        mock transport does not spend real seconds waiting on a 1-request-per-second endpoint.
        Production never passes it — and because it is a constructor argument rather than a
        config key, it cannot be loosened by editing a YAML file (CLAUDE.md §1).
        """
        self._settings = settings if settings is not None else get_settings()
        self._journal = journal if journal is not None else OrderJournal(clock=clock)
        self._clock = clock
        self._mode = mode if mode is not None else self._settings.trading_mode
        self._timeout = timeout
        self._identity = identity if identity is not None else ClientIdentity.detect()
        self._max_retries = self._settings.execution.max_order_retries

        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)
        )

        # The rate limiter deliberately runs on the *system* monotonic clock, not on the
        # injected one. A broker's rate counter advances in real seconds regardless of what any
        # test double believes, and a frozen or fast-forwarded clock must not be able to switch
        # the limiter off — that is how an API key gets banned. `clock` still drives session
        # timestamps and latency measurement, which are ours to fake.
        if rate_limit_scale <= 0:
            raise ValueError("rate_limit_scale must be positive")
        self._rate_scale = rate_limit_scale
        ceiling = float(self._settings.execution.rate_limit_per_second) * rate_limit_scale
        self._global_bucket = TokenBucket(ceiling)
        self._buckets: dict[str, TokenBucket] = {}
        self._refresh_lock = asyncio.Lock()

        self._simulate_paper = simulate_paper_orders
        self._paper = PaperBroker()

        client_code = self._settings.smartapi_client_code.get_secret_value()
        self.session = BrokerSession(client_code=client_code)
        self.stats = ApiStats()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the transport if we own it. Idempotent."""
        if self._owns_client:
            await self._client.aclose()

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_authenticated(self) -> bool:
        return self.session.is_active

    @property
    def paper(self) -> PaperBroker:
        """The simulated broker. Populated only while ``TRADING_MODE`` is not LIVE."""
        return self._paper

    @property
    def intercepts_orders(self) -> bool:
        """True when a mutating call would be simulated rather than transmitted."""
        return self._mode is not TradingMode.LIVE

    @property
    def feed_token(self) -> str:
        """Streaming credential for the ingestion process. Never log the result."""
        return self.session.feed_token

    # ── authentication ───────────────────────────────────────────────────────

    def current_totp(self) -> str:
        """Generate the current TOTP from the configured secret.

        Raises:
            SmartApiAuthError: no TOTP secret is configured, or it is not valid base32.
        """
        secret = _normalize_totp_secret(self._settings.smartapi_totp_secret.get_secret_value())
        if not secret:
            raise SmartApiAuthError(LOGIN.name, "SMARTAPI_TOTP_SECRET is not set")
        try:
            # pyotp ships no type information, so .now() is Any; str() pins it at the boundary.
            return str(pyotp.TOTP(secret).now())
        except Exception as exc:  # noqa: BLE001 - pyotp raises bare binascii errors on bad base32
            raise SmartApiAuthError(LOGIN.name, f"TOTP generation failed: {exc}") from exc

    async def login(self) -> BrokerSession:
        """Full ``generateSession`` flow: client code + password/PIN + TOTP → tokens.

        Populates :attr:`session` with the JWT, refresh token and **feed token** — the last of
        which the ingestion process needs for its WebSocket handshake, and which is the reason
        ``SMARTAPI_FEED_TOKEN`` no longer has to be pasted into ``.env`` by hand.

        Session cache
        -------------
        A same-day session cached on disk is adopted **without touching the network**: Angel
        One rate-limits ``loginByPassword`` hard, and every supervisor restart used to fire a
        fresh login. Only a missing, stale or 401-rejected cache re-authenticates — the 401
        path is handled in :meth:`_call`, which invalidates the cache and re-enters here.

        Raises:
            SmartApiAuthError: credentials missing, or the broker rejected them.
        """
        client_code = self._settings.smartapi_client_code.get_secret_value().strip()
        password = self._settings.smartapi_password.get_secret_value().strip()
        api_key = self._settings.smartapi_api_key.get_secret_value().strip()

        cached = (
            token_cache.load_session_cache(api_key, client_code, clock=self._clock)
            if client_code
            else None
        )
        if cached is not None:
            self.session = BrokerSession(
                jwt_token=cached.jwt_token,
                refresh_token=cached.refresh_token,
                feed_token=cached.feed_token,
                client_code=cached.client_code,
                issued_at_ist=cached.issued_at_ist,
            )
            _log.info("broker.login_from_cache", client_code=client_code, mode=self._mode)
            return self.session

        if not client_code or not password:
            raise SmartApiAuthError(
                LOGIN.name,
                "SMARTAPI_CLIENT_CODE and SMARTAPI_PASSWORD are both required to log in",
            )

        # The journal records that a login was attempted; scrub_secrets removes the
        # password and totp values before anything reaches the disk.
        self._journal.request(LOGIN.name, {"clientcode": client_code})

        data = await self._login_with_backoff(client_code, password)
        tokens = data if isinstance(data, dict) else {}
        jwt = str(tokens.get("jwtToken", "")).removeprefix("Bearer ").strip()
        feed = str(tokens.get("feedToken", "")).strip()
        if not jwt or not feed:
            raise SmartApiAuthError(
                LOGIN.name, "login succeeded but the response carried no jwtToken/feedToken"
            )

        self.session = BrokerSession(
            jwt_token=jwt,
            refresh_token=str(tokens.get("refreshToken", "")).strip(),
            feed_token=feed,
            client_code=client_code,
            issued_at_ist=now_ist(self._clock),
        )
        token_cache.save_session_cache(
            api_key=api_key,
            client_code=client_code,
            jwt_token=jwt,
            refresh_token=self.session.refresh_token,
            feed_token=feed,
            clock=self._clock,
        )
        _log.info("broker.login_ok", client_code=client_code, mode=self._mode)
        return self.session

    async def _login_with_backoff(self, client_code: str, password: str) -> Any:
        """``loginByPassword`` with exponential backoff on the login rate limit.

        Angel One answers a login burst with HTTP 403 "Access denied because of exceeding
        access rate". Retrying immediately re-arms the ban — a supervisor restart loop can
        turn one ban into a day-long lockout — so each retry waits on an exponential
        schedule instead of killing the process. Any *other* refusal (wrong password, bad
        TOTP) propagates at once: retrying bad credentials burns the account's attempt
        budget and can lock the account.
        """
        attempt = 0
        while True:
            # Rebuilt per attempt: a 30-120 s sleep crosses TOTP windows.
            payload = {
                "clientcode": client_code,
                "password": password,
                "totp": self.current_totp(),
            }
            try:
                return await self._call(
                    LOGIN, json=payload, authenticated=False, allow_refresh=False
                )
            except SmartApiError as exc:
                if attempt >= len(
                    token_cache.LOGIN_RATE_LIMIT_BACKOFFS
                ) or not token_cache.is_rate_limited_message(str(exc)):
                    raise
                delay = token_cache.LOGIN_RATE_LIMIT_BACKOFFS[attempt]
                attempt += 1
                self.stats.rate_limit_waits += 1
                _log.warning(
                    "broker.login_rate_limited",
                    attempt=attempt,
                    retry_in_seconds=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)

    async def refresh_session(self) -> BrokerSession:
        """Exchange the refresh token for a new JWT.

        Serialised behind a lock: a burst of concurrent calls all seeing "token expired" must
        produce one refresh, not five — five would trip the login rate limit and lock us out
        of our own account mid-session.
        """
        async with self._refresh_lock:
            if not self.session.refresh_token:
                raise SmartApiAuthError(REFRESH.name, "no refresh token — a full login is needed")

            try:
                data = await self._call(
                    REFRESH,
                    json={"refreshToken": self.session.refresh_token},
                    authenticated=True,
                    allow_refresh=False,
                )
            except SmartApiAuthError:
                # The broker refused the refresh token itself — the cached session is dead
                # server-side. Drop the cache so the next start re-authenticates cleanly.
                token_cache.invalidate_session_cache()
                raise
            tokens = data if isinstance(data, dict) else {}
            jwt = str(tokens.get("jwtToken", "")).removeprefix("Bearer ").strip()
            if not jwt:
                token_cache.invalidate_session_cache()
                raise SmartApiAuthError(REFRESH.name, "refresh returned no jwtToken")

            self.session.jwt_token = jwt
            self.session.refresh_token = (
                str(tokens.get("refreshToken", "")).strip() or self.session.refresh_token
            )
            self.session.feed_token = (
                str(tokens.get("feedToken", "")).strip() or self.session.feed_token
            )
            self.session.issued_at_ist = now_ist(self._clock)
            self.stats.token_refreshes += 1
            token_cache.save_session_cache(
                api_key=self._settings.smartapi_api_key.get_secret_value().strip(),
                client_code=self.session.client_code,
                jwt_token=self.session.jwt_token,
                refresh_token=self.session.refresh_token,
                feed_token=self.session.feed_token,
                clock=self._clock,
            )
            _log.info("broker.session_refreshed")
            return self.session

    async def logout(self) -> None:
        """End the broker session. Failure is logged, never raised — we are shutting down."""
        if not self.session.is_active:
            return
        try:
            await self._call(LOGOUT, json={"clientcode": self.session.client_code})
        except SmartApiError as exc:
            _log.warning("broker.logout_failed", error=str(exc))
        finally:
            self.session.clear()
            # The broker killed this session; a cached copy must not outlive it.
            token_cache.invalidate_session_cache()

    # ── order operations ─────────────────────────────────────────────────────

    async def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Place one order. **Never retried after transmission** (CLAUDE.md §6.4).

        Raises:
            PaperModeError: ``TRADING_MODE`` is PAPER.
            UnknownOrderOutcomeError: the call failed after being sent. Reconcile; do not
                re-send.
        """
        self._require_live(PLACE_ORDER.name)
        tag = str(payload.get("ordertag", ""))
        self._journal.request(PLACE_ORDER.name, payload, order_tag=tag)
        data = await self._call(PLACE_ORDER, json=payload, order_tag=tag)
        result = data if isinstance(data, dict) else {}
        self._journal.response(PLACE_ORDER.name, result, order_tag=tag)
        return result

    async def modify_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Modify a live order — used only to move a stop *toward profit* (CLAUDE.md §8.1)."""
        self._require_live(MODIFY_ORDER.name)
        tag = str(payload.get("ordertag", ""))
        self._journal.request(MODIFY_ORDER.name, payload, order_tag=tag)
        data = await self._call(MODIFY_ORDER, json=payload, order_tag=tag)
        result = data if isinstance(data, dict) else {}
        self._journal.response(MODIFY_ORDER.name, result, order_tag=tag)
        return result

    async def cancel_order(self, order_id: str, variety: str = "ROBO") -> dict[str, Any]:
        """Cancel one order by broker order id."""
        self._require_live(CANCEL_ORDER.name)
        payload = {"variety": variety, "orderid": order_id}
        self._journal.request(CANCEL_ORDER.name, payload, order_tag=order_id)
        data = await self._call(CANCEL_ORDER, json=payload, order_tag=order_id)
        result = data if isinstance(data, dict) else {}
        self._journal.response(CANCEL_ORDER.name, result, order_tag=order_id)
        return result

    # ── read-only queries ────────────────────────────────────────────────────

    async def order_book(self) -> tuple[dict[str, Any], ...]:
        """Every order for the session. Empty tuple when the broker reports none.

        SmartAPI returns ``data: null`` for an empty book, which is not an error — the
        distinction between "no orders" and "the call failed" is carried by the exception,
        never by the return value.
        """
        return await self._rows(ORDER_BOOK)

    async def trade_book(self) -> tuple[dict[str, Any], ...]:
        """Every execution for the session."""
        return await self._rows(TRADE_BOOK)

    async def positions(self) -> tuple[dict[str, Any], ...]:
        """Open and closed positions for the session."""
        return await self._rows(POSITIONS)

    async def profile(self) -> dict[str, Any]:
        data = await self._call(PROFILE)
        return data if isinstance(data, dict) else {}

    async def available_margin(self) -> Decimal:
        """Free cash available for a new position, in rupees.

        Feeds the risk engine's ``INSUFFICIENT_MARGIN`` check. Returned as ``Decimal`` because
        it is money; an unparseable or missing value raises rather than defaulting to zero or
        infinity — the caller's guard turns that into a veto (CLAUDE.md §4).
        """
        data = await self._call(RMS)
        rms = data if isinstance(data, dict) else {}
        raw = rms.get("availablecash", rms.get("net"))
        if raw is None:
            raise SmartApiError(RMS.name, "RMS response carried no availablecash/net field")
        try:
            value = Decimal(str(raw))
        except InvalidOperation as exc:
            raise SmartApiError(RMS.name, f"unparseable margin value {raw!r}") from exc
        if not value.is_finite():
            raise SmartApiError(RMS.name, f"margin value is undefined: {value}")
        return value

    async def _rows(self, endpoint: Endpoint) -> tuple[dict[str, Any], ...]:
        data = await self._call(endpoint)
        if data is None:
            return ()
        if not isinstance(data, list):
            raise SmartApiError(endpoint.name, f"expected a list, got {type(data).__name__}")
        return tuple(row for row in data if isinstance(row, dict))

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _require_live(self, endpoint: str) -> None:
        """Refuse a mutating call in PAPER when simulation has been switched off.

        With simulation enabled (the default) this is a no-op and :meth:`_intercept` handles
        the call instead. Either way nothing reaches the network.
        """
        if self._mode is not TradingMode.LIVE and not self._simulate_paper:
            raise PaperModeError(
                f"{endpoint} refused: TRADING_MODE is {self._mode} and paper simulation is "
                f"disabled. No broker-mutating call is permitted outside LIVE (CLAUDE.md §9)."
            )

    def _intercept(self, endpoint: Endpoint, payload: dict[str, Any] | None) -> Any | None:
        """Return a simulated response, or ``None`` to let the call proceed to the network.

        The single choke point for the PAPER guarantee. It runs **before** the rate limiter and
        before any header is assembled, so a paper placement costs no quota, no token and no
        socket — and there is exactly one line in this system that decides whether an order
        leaves the machine.
        """
        if self._mode is TradingMode.LIVE or endpoint.name not in PAPER_INTERCEPTED:
            return None

        body = payload if payload is not None else {}
        data = self._paper.simulate(endpoint, body)
        self.stats.paper_intercepts += 1
        self._journal.record(
            "DECISION",
            f"paper_{endpoint.name}",
            order_tag=str(body.get("ordertag", "")),
            order_id=data.get("orderid", ""),
            payload=body,
            note="INTERCEPTED — TRADING_MODE is PAPER; nothing was transmitted",
        )
        _log.info(
            "broker.paper_intercept",
            endpoint=endpoint.name,
            order_id=data.get("orderid", ""),
            order_tag=body.get("ordertag", ""),
            mode=self._mode,
        )
        return data

    def _bucket(self, endpoint: Endpoint) -> TokenBucket:
        bucket = self._buckets.get(endpoint.name)
        if bucket is None:
            # System clock, for the reason given in __init__.
            bucket = TokenBucket(endpoint.limit_per_second * self._rate_scale)
            self._buckets[endpoint.name] = bucket
        return bucket

    def _headers(self, authenticated: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": self._identity.local_ip,
            "X-ClientPublicIP": self._identity.public_ip,
            "X-MACAddress": self._identity.mac_address,
            "X-PrivateKey": self._settings.smartapi_api_key.get_secret_value(),
        }
        if authenticated and self.session.jwt_token:
            headers["Authorization"] = f"Bearer {self.session.jwt_token}"
        return headers

    async def _throttle(self, endpoint: Endpoint) -> None:
        waited = await self._bucket(endpoint).acquire()
        waited += await self._global_bucket.acquire()
        if waited > 0:
            self.stats.rate_limit_waits += 1
            _log.debug(
                "broker.rate_limited", endpoint=endpoint.name, waited_seconds=round(waited, 3)
            )

    async def _call(
        self,
        endpoint: Endpoint,
        *,
        json: dict[str, Any] | None = None,
        authenticated: bool = True,
        allow_refresh: bool = True,
        order_tag: str = "",
    ) -> Any:
        """Execute one endpoint call with rate limiting, retries and error classification.

        Returns the broker's ``data`` field, which may legitimately be ``None``.
        """
        intercepted = self._intercept(endpoint, json)
        if intercepted is not None:
            return intercepted

        attempt = 0
        while True:
            await self._throttle(endpoint)
            self.stats.requests += 1
            started = self._clock.monotonic()

            try:
                response = await self._client.request(
                    endpoint.method,
                    endpoint.path,
                    json=json,
                    headers=self._headers(authenticated),
                    timeout=self._timeout,
                )
            except _NEVER_SENT as exc:
                # Provably never transmitted, so re-sending cannot duplicate anything.
                attempt += 1
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                if attempt > self._max_retries:
                    self._journal.error(endpoint.name, self.stats.last_error, order_tag=order_tag)
                    raise SmartApiError(
                        endpoint.name, f"unreachable after {attempt} attempts: {exc}"
                    ) from exc
                self.stats.transport_retries += 1
                _log.warning(
                    "broker.retrying",
                    endpoint=endpoint.name,
                    attempt=attempt,
                    reason="request was never transmitted",
                    error=self.stats.last_error,
                )
                await asyncio.sleep(min(2.0**attempt * 0.25, 4.0))
                continue
            except httpx.HTTPError as exc:
                # Sent, but we never learned the outcome.
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self._journal.error(endpoint.name, self.stats.last_error, order_tag=order_tag)
                if endpoint.idempotent:
                    attempt += 1
                    if attempt > self._max_retries:
                        raise SmartApiError(
                            endpoint.name, f"failed after {attempt} attempts: {exc}"
                        ) from exc
                    self.stats.transport_retries += 1
                    await asyncio.sleep(min(2.0**attempt * 0.25, 4.0))
                    continue
                self.stats.unknown_outcomes += 1
                _log.critical(
                    "broker.outcome_unknown",
                    endpoint=endpoint.name,
                    order_tag=order_tag,
                    error=self.stats.last_error,
                    action="NOT retrying — reconcile against the order book (CLAUDE.md §6.4)",
                )
                raise UnknownOrderOutcomeError(
                    endpoint.name, str(exc), order_tag=order_tag
                ) from exc

            latency_ms = (self._clock.monotonic() - started) * 1000.0
            self.stats.responses += 1

            try:
                body = response.json()
            except ValueError as exc:
                # WAF blocks and gateway errors arrive as HTML, not JSON. Extract the
                # rejection text so the operator sees *why* the call was refused instead
                # of a bare status code.
                snippet = _summarize_non_json_body(response.text)
                detail = f"non-JSON response (HTTP {response.status_code}): {snippet}"
                _log.error(
                    "broker.non_json_response",
                    endpoint=endpoint.name,
                    status_code=response.status_code,
                    body=snippet,
                )
                self._journal.error(endpoint.name, detail, order_tag=order_tag)
                if endpoint in (LOGIN, REFRESH):
                    # A session request rejected at the edge has no unknown *order* outcome;
                    # classify it as auth so the login backoff can inspect the refusal.
                    raise SmartApiAuthError(endpoint.name, detail) from exc
                if endpoint.idempotent:
                    raise SmartApiError(endpoint.name, detail) from exc
                self.stats.unknown_outcomes += 1
                raise UnknownOrderOutcomeError(
                    endpoint.name,
                    detail,
                    order_tag=order_tag,
                ) from exc

            payload = body if isinstance(body, dict) else {}
            error_code = str(payload.get("errorcode", "") or "")
            message = str(payload.get("message", "") or f"HTTP {response.status_code}")

            if payload.get("status") is True:
                _log.debug("broker.ok", endpoint=endpoint.name, latency_ms=round(latency_ms, 1))
                return payload.get("data")

            # "Token expired" arrives either as a documented error code or as a bare HTTP
            # 401; both mean the same thing — the cached/refreshed JWT is no longer valid.
            token_expired = error_code in _TOKEN_ERROR_CODES or response.status_code == 401
            if token_expired and allow_refresh and self.session.refresh_token:
                _log.warning(
                    "broker.token_expired",
                    endpoint=endpoint.name,
                    error_code=error_code,
                    http_status=response.status_code,
                )
                try:
                    await self.refresh_session()
                except SmartApiAuthError:
                    # The refresh token is dead too — e.g. a cached session the broker
                    # revoked. refresh_session already dropped the cache; one full
                    # re-login, then the retried call below gets the fresh JWT.
                    if endpoint in (LOGIN, REFRESH):
                        raise
                    await self.login()
                allow_refresh = False
                continue

            self.stats.broker_rejections += 1
            self.stats.last_error = f"{message} [{error_code}]"
            self._journal.error(
                endpoint.name,
                message,
                order_tag=order_tag,
                error_code=error_code,
                http_status=response.status_code,
            )
            _log.error(
                "broker.rejected",
                endpoint=endpoint.name,
                error_code=error_code,
                message=message,
                http_status=response.status_code,
                order_tag=order_tag,
            )
            # A rejected login is not retryable at any layer — surface it as an auth failure so
            # the boot path stops rather than backing off against a permanent refusal.
            if endpoint in (LOGIN, REFRESH):
                raise SmartApiAuthError(endpoint.name, message, error_code=error_code)
            raise SmartApiError(endpoint.name, message, error_code=error_code)


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """One row of the broker order book, normalised to the fields we reason about.

    Deliberately a subset. The raw dictionary is preserved in :attr:`raw` for the journal, but
    every decision is made against typed fields, so a broker adding or renaming a field cannot
    silently change what "open" means.
    """

    order_id: str
    status: str
    trading_symbol: str
    token: str
    transaction_type: str
    quantity: int
    pending_quantity: int
    variety: str
    order_tag: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_open(self) -> bool:
        """True if this order can still fill. Unknown statuses count as open (fail-safe)."""
        return self.status.strip().lower() not in TERMINAL_ORDER_STATUSES

    @property
    def is_ours(self) -> bool:
        """True if the order carries one of our client tags (CLAUDE.md §6.4)."""
        return self.order_tag.startswith(ORDER_TAG_PREFIX)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        def _int(key: str) -> int:
            try:
                return int(Decimal(str(row.get(key, 0) or 0)))
            except InvalidOperation, ValueError:
                return 0

        return cls(
            order_id=str(row.get("orderid", "")),
            status=str(row.get("orderstatus", row.get("status", "")) or ""),
            trading_symbol=str(row.get("tradingsymbol", "")),
            token=str(row.get("symboltoken", "")),
            transaction_type=str(row.get("transactiontype", "")),
            quantity=_int("quantity"),
            pending_quantity=_int("unfilledshares"),
            variety=str(row.get("variety", "")),
            order_tag=str(row.get("ordertag", "") or ""),
            raw=row,
        )


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """One row of the broker position book, normalised.

    ``net_quantity`` is the only field that decides whether we are exposed. It is signed:
    positive is long, negative is short, zero is flat.
    """

    trading_symbol: str
    token: str
    exchange: str
    net_quantity: int
    product_type: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_open(self) -> bool:
        return self.net_quantity != 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        raw_qty = row.get("netqty", row.get("netQty", 0))
        try:
            net = int(Decimal(str(raw_qty or 0)))
        except InvalidOperation, ValueError:
            # An unparseable quantity is not "flat". Treat it as exposure so reconciliation
            # locks the system rather than assuming the best.
            _log.error("broker.unparseable_netqty", value=repr(raw_qty))
            net = 1
        return cls(
            trading_symbol=str(row.get("tradingsymbol", "")),
            token=str(row.get("symboltoken", "")),
            exchange=str(row.get("exchange", "")),
            net_quantity=net,
            product_type=str(row.get("producttype", "")),
            raw=row,
        )
