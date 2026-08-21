"""Wire schemas and topic contract — CLAUDE.md §2.1.

Every message crossing a process boundary is a frozen :class:`msgspec.Struct` encoded with
``msgspec.json``. ``pickle`` is banned: it executes arbitrary code on decode, and a market data
spine is the last place that belongs.

Struct configuration, and why:

``frozen=True``
    Immutable and hashable. A tick that has been published cannot be mutated by a downstream
    consumer, so the strategy and the risk engine provably see identical data.
``gc=False``
    These structs hold only primitives and tuples of primitives, so they cannot participate in
    reference cycles. Opting out of GC tracking removes them from every collection pass — at
    thousands of ticks per second that is the difference between a smooth feed and periodic
    GC pauses landing in the middle of the tick path.
``eq=True`` (default)
    Cheap and useful for dedup and tests.

**Versioning.** ``SCHEMA_VERSION`` is announced on the :class:`Heartbeat` rather than stamped on
every tick — a per-message version field would cost bytes on the hottest path to detect what is
really a deploy-time mismatch. Subscribers validate it on the first heartbeat and refuse to run
against a wire they do not understand. Fields may only ever be **appended**; never reorder,
never repurpose, never change a type. Anything else is a new ``SCHEMA_VERSION``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

import msgspec

from tachyon.core.clock import IST

#: Wire contract version. Bump only on a breaking change (reorder, retype, remove).
#:
#: v2 — added the required :attr:`Tick.seq` for gap detection. A *required* field is a
#: breaking change even though it was appended: a v1 payload has no ``seq`` and will not
#: decode here. Both processes must be deployed together.
SCHEMA_VERSION: Final[int] = 2

#: SmartAPI publishes five levels of market depth.
DEPTH_LEVELS: Final[int] = 5

#: Exactly-five-element depth rows. msgspec enforces the length at decode time for free —
#: a truncated depth frame fails loudly instead of silently skewing Order Book Imbalance.
Depth5Price = tuple[float, float, float, float, float]
Depth5Qty = tuple[int, int, int, int, int]


# ──────────────────────────────────────────────────────────────────────────────
# Messages
# ──────────────────────────────────────────────────────────────────────────────


class Tick(msgspec.Struct, frozen=True, gc=False):
    """A single trade print / LTP update.

    Deliberately minimal — this is the highest-frequency message in the system and every
    field is paid for on every print. The symbol is carried by the *topic*
    (``TICK.RELIANCE``), so it is not repeated in the payload.
    """

    token: str
    """Exchange instrument token, e.g. ``"2885"``. Not a credential — safe to log."""

    ltp: float
    """Last traded price."""

    volume: int
    """Cumulative traded volume for the session, as reported by the exchange."""

    ts_epoch: float
    """Exchange/receipt time, Unix seconds UTC.

    Stored as an epoch float rather than an encoded datetime: RFC-3339 costs ~25 extra bytes
    and a parse on every tick. :attr:`ts_ist` renders it in IST when a human needs it.
    """

    seq: int
    """Per-symbol publisher sequence number, incrementing by exactly 1.

    The transport drops messages by design when a subscriber falls behind (CLAUDE.md §2.1),
    and without this the loss is undetectable: VWAP and OBI would simply become quietly wrong.
    A consumer that sees ``seq != previous + 1`` knows its accumulators are tainted.

    Publishers assign this per symbol, not globally, so one symbol's gap does not implicate
    another's.
    """

    @property
    def ts_ist(self) -> datetime:
        """Timestamp as a timezone-aware IST datetime (CLAUDE.md §8)."""
        return datetime.fromtimestamp(self.ts_epoch, tz=IST)


class OrderBook(msgspec.Struct, frozen=True, gc=False):
    """Level-2 market depth: five bid levels and five ask levels.

    Index 0 is the top of book. Empty levels are published as ``0.0`` price / ``0`` quantity,
    never omitted, so the arrays stay a fixed shape for the Numba OBI kernel (CLAUDE.md §3.1).
    """

    token: str
    bid_price: Depth5Price
    bid_qty: Depth5Qty
    ask_price: Depth5Price
    ask_qty: Depth5Qty
    ts_epoch: float

    @property
    def ts_ist(self) -> datetime:
        return datetime.fromtimestamp(self.ts_epoch, tz=IST)

    @property
    def best_bid(self) -> float:
        return self.bid_price[0]

    @property
    def best_ask(self) -> float:
        return self.ask_price[0]

    @property
    def spread(self) -> float:
        """Top-of-book spread. Negative or zero indicates a crossed or empty book."""
        return self.ask_price[0] - self.bid_price[0]

    def is_crossed(self) -> bool:
        """True if the book is crossed or a side is empty — do not trade on it."""
        return self.bid_price[0] <= 0.0 or self.ask_price[0] <= 0.0 or self.spread <= 0.0


class Heartbeat(msgspec.Struct, frozen=True, gc=False):
    """Liveness ping from a publisher.

    Exists so that :class:`~tachyon.ipc.monitor.FeedMonitor` can distinguish "the market is
    quiet" from "the feed is dead". Without it, a symbol that simply is not trading looks
    identical to a dropped WebSocket — and one of those must halt trading.

    ``seq`` increments monotonically from process start, so a subscriber can detect both gaps
    (dropped datagrams) and resets (the publisher restarted underneath it).
    """

    ts_epoch: float
    seq: int
    role: str = "ingestor"
    schema_version: int = SCHEMA_VERSION

    @property
    def ts_ist(self) -> datetime:
        return datetime.fromtimestamp(self.ts_epoch, tz=IST)


# ──────────────────────────────────────────────────────────────────────────────
# State spine — tcp://127.0.0.1:5556, published by the Brain (CLAUDE.md §2.1)
#
# These are a new *family* of messages on a different endpoint, not a change to the tick wire.
# SCHEMA_VERSION is deliberately NOT bumped: the ingestor's tick and heartbeat bytes are
# identical, and bumping would make an unchanged v2 ingestor fail the Brain's version check for
# a reason that has nothing to do with it. Versioning guards the wire a subscriber decodes, and
# no existing field has moved.
# ──────────────────────────────────────────────────────────────────────────────


class StateUpdate(msgspec.Struct, frozen=True, gc=False):
    """Session state, published on every transition and on a slow keepalive.

    Carries the *whole* view rather than a delta. The UI is a conflating consumer by design
    (CLAUDE.md §2.1) — it may miss frames, so a frame that only says "what changed" would leave
    it permanently wrong the first time one is dropped.
    """

    state: str
    """``TradingState`` value: BOOTING, PRE_MARKET, ACTIVE, ... LOCKED."""

    mode: str
    """PAPER or LIVE. Rendered prominently — an operator must never have to guess."""

    ts_epoch: float
    feed_stale: bool = False
    feed_age_seconds: float = 0.0
    open_symbols: tuple[str, ...] = ()
    cooling_symbols: tuple[str, ...] = ()
    macro_regime: str = "NEUTRAL"
    macro_confidence: int = 0
    macro_blocks_entries: bool = False
    macro_degraded: bool = False
    daily_lock_engaged: bool = False
    reason: str = ""


class PnLUpdate(msgspec.Struct, frozen=True, gc=False):
    """Running P&L against the daily limit.

    Money crosses the wire as **strings**, not floats. ``Decimal`` is the type at the risk
    boundary (CLAUDE.md §8) and a float round-trip would quietly reintroduce the representation
    error the whole system avoids — on the number the operator watches most closely.
    """

    realised: str
    floating: str
    charges: str
    total: str
    headroom: str
    limit: str
    breached: bool
    ts_epoch: float
    trades_today: int = 0

    #: Appended in the dynamic-budget change; defaulted so an older subscriber still decodes
    #: (§2.1 — add fields, never reorder or repurpose).
    #: Session trading capital, or "0" when the §1 constants are in force.
    capital: str = "0"
    #: ``limit`` as a percentage of ``capital``. "0" when static.
    drawdown_pct: str = "0"
    #: Where the limit came from. The UI shows this so a wrong number is traceable at a glance
    #: rather than being mistaken for the ₹500 everyone has memorised.
    limit_source: str = ""


class FillUpdate(msgspec.Struct, frozen=True, gc=False):
    """One order-status transition from the broker."""

    symbol: str
    order_id: str
    status: str
    side: str
    quantity: int
    filled_quantity: int
    price: str
    ts_epoch: float
    order_tag: str = ""
    is_exit: bool = False


class RiskEvent(msgspec.Struct, frozen=True, gc=False):
    """A veto, a kill switch, or a square-off — anything the operator must see."""

    kind: str
    """VETO, KILL_SWITCH, SQUARE_OFF, RECONCILIATION, SENTINEL."""

    symbol: str
    reason: str
    detail: str
    ts_epoch: float
    severity: str = "INFO"
    """INFO, WARNING or CRITICAL. Drives the colour the UI renders it in (§7.2)."""


#: Anything that can arrive on the tick spine.
WireMessage = Tick | OrderBook | Heartbeat

#: Anything that can arrive on the state spine.
StateMessage = StateUpdate | PnLUpdate | FillUpdate | RiskEvent | Heartbeat

#: Every message this system puts on a socket.
AnyMessage = WireMessage | StateUpdate | PnLUpdate | FillUpdate | RiskEvent


# ──────────────────────────────────────────────────────────────────────────────
# Topics — CLAUDE.md §2.1
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_TICK: Final[str] = "TICK."
TOPIC_DEPTH: Final[str] = "DEPTH."
TOPIC_FEED: Final[str] = "FEED."

#: State spine (tcp://127.0.0.1:5556). Conflation is permitted on ``STATE.`` and ``PNL.`` —
#: both carry a complete view, so the newest frame is always sufficient. It is **forbidden** on
#: ``FILL.`` and ``RISK.`` (CLAUDE.md §2.1): those are events, and a dropped fill is a position
#: the operator never learns about.
TOPIC_STATE: Final[str] = "STATE."
TOPIC_FILL: Final[str] = "FILL."
TOPIC_PNL: Final[str] = "PNL."
TOPIC_RISK: Final[str] = "RISK."
TOPIC_LOG: Final[str] = "LOG."

#: Well-known single topics on the state spine, pre-encoded — published on a timer.
TOPIC_STATE_SESSION: Final[str] = "STATE.SESSION"
TOPIC_PNL_SESSION: Final[str] = "PNL.SESSION"

STATE_TOPIC_BYTES: Final[bytes] = TOPIC_STATE_SESSION.encode()
PNL_TOPIC_BYTES: Final[bytes] = TOPIC_PNL_SESSION.encode()

#: Single well-known topic for liveness. Subscribers that care about nothing else still
#: subscribe to this one, otherwise they cannot tell a dead feed from a quiet market.
TOPIC_HEARTBEAT: Final[str] = "FEED.HEARTBEAT"

_TICK_PREFIX: Final[bytes] = TOPIC_TICK.encode()
_DEPTH_PREFIX: Final[bytes] = TOPIC_DEPTH.encode()

#: Pre-encoded because it is published on a timer and never changes.
HEARTBEAT_TOPIC_BYTES: Final[bytes] = TOPIC_HEARTBEAT.encode()


def tick_topic(symbol: str) -> bytes:
    """Topic frame for a symbol's ticks, e.g. ``b"TICK.RELIANCE"``.

    Callers on the hot path should hoist this out of the loop — the per-symbol result is
    constant for the life of the session.
    """
    return _TICK_PREFIX + symbol.encode()


def depth_topic(symbol: str) -> bytes:
    """Topic frame for a symbol's L2 depth, e.g. ``b"DEPTH.RELIANCE"``."""
    return _DEPTH_PREFIX + symbol.encode()


def symbol_from_topic(topic: str) -> str:
    """Extract the symbol from a topic frame. ``"TICK.RELIANCE"`` -> ``"RELIANCE"``."""
    _, _, symbol = topic.partition(".")
    return symbol


# ──────────────────────────────────────────────────────────────────────────────
# Codec
# ──────────────────────────────────────────────────────────────────────────────

#: Reusable encoder. This is the object form of ``msgspec.json.encode`` — identical output,
#: but it avoids rebuilding encoder state on every call, which matters at tick rates.
_ENCODER: Final[msgspec.json.Encoder] = msgspec.json.Encoder()

_TICK_DECODER: Final[msgspec.json.Decoder[Tick]] = msgspec.json.Decoder(Tick)
_DEPTH_DECODER: Final[msgspec.json.Decoder[OrderBook]] = msgspec.json.Decoder(OrderBook)
_HEARTBEAT_DECODER: Final[msgspec.json.Decoder[Heartbeat]] = msgspec.json.Decoder(Heartbeat)
_STATE_DECODER: Final[msgspec.json.Decoder[StateUpdate]] = msgspec.json.Decoder(StateUpdate)
_PNL_DECODER: Final[msgspec.json.Decoder[PnLUpdate]] = msgspec.json.Decoder(PnLUpdate)
_FILL_DECODER: Final[msgspec.json.Decoder[FillUpdate]] = msgspec.json.Decoder(FillUpdate)
_RISK_DECODER: Final[msgspec.json.Decoder[RiskEvent]] = msgspec.json.Decoder(RiskEvent)


class UnknownTopicError(ValueError):
    """A payload arrived on a topic with no registered schema."""


def encode(message: AnyMessage) -> bytes:
    """Serialise a message for the wire."""
    return _ENCODER.encode(message)


def decode_tick(payload: bytes) -> Tick:
    return _TICK_DECODER.decode(payload)


def decode_orderbook(payload: bytes) -> OrderBook:
    return _DEPTH_DECODER.decode(payload)


def decode_heartbeat(payload: bytes) -> Heartbeat:
    return _HEARTBEAT_DECODER.decode(payload)


def decode_for_topic(topic: str, payload: bytes) -> AnyMessage:
    """Decode a payload using the schema implied by its topic.

    Raises:
        UnknownTopicError: the topic has no registered schema. Treated as a defect rather
            than ignored — an unrecognised topic means the two processes disagree about the
            wire, and guessing is worse than stopping.
        msgspec.DecodeError: the payload does not match its schema.
    """
    if topic.startswith(TOPIC_TICK):
        return _TICK_DECODER.decode(payload)
    if topic.startswith(TOPIC_DEPTH):
        return _DEPTH_DECODER.decode(payload)
    if topic.startswith(TOPIC_FEED):
        return _HEARTBEAT_DECODER.decode(payload)
    if topic.startswith(TOPIC_STATE):
        return _STATE_DECODER.decode(payload)
    if topic.startswith(TOPIC_PNL):
        return _PNL_DECODER.decode(payload)
    if topic.startswith(TOPIC_FILL):
        return _FILL_DECODER.decode(payload)
    if topic.startswith(TOPIC_RISK):
        return _RISK_DECODER.decode(payload)
    raise UnknownTopicError(f"No schema registered for topic {topic!r}")
