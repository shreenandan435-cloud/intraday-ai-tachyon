"""Broker order-status listener — the fill source, CLAUDE.md §6, §9.

Until now nothing told the system that a position had *closed*. Entries were booked, brackets
went to the exchange, and then the loop went quiet: P&L never moved off zero, the re-entry
cooldown never started, and ``POSITION_ALREADY_OPEN`` blocked the symbol forever. This module
closes that loop.

Two sources, one path
---------------------
Angel One offers a **postback webhook** (push) and an **order book** (pull). Both are supported
and both feed :meth:`OrderStatusListener.ingest`, because neither is trustworthy alone:

* a webhook is fast but silently lossy — a dropped POST is invisible to the receiver;
* polling is reliable but rate-limited to roughly one call a second, which is far too slow to
  notice a T1 fill promptly.

Running both means the fast path is usually first and the slow path is the backstop. That makes
**idempotency the central requirement**, not a nicety: the same fill will arrive twice, in
either order, and booking it twice would double the realised P&L on the number the ₹500 kill
switch is enforced against.

Idempotency
-----------
Every update is keyed by ``(order_id, status, filled_quantity)``. A repeat is dropped before it
reaches the position book. Progressive fills of one order (100 → 250 → 400) are distinct keys
and each is applied as a *delta*, so a partial fill sequence books exactly the traded quantity
however many times the broker repeats itself.

Closing a position
------------------
A symbol is closed when its net quantity returns to zero. At that moment, and once only, the
listener calls the ``on_closed`` callback — the Brain's ``on_position_closed`` — which books
realised P&L, starts the 30-minute cooldown, and frees the symbol.

**A stop-out is identified by the order that closed the position**, not by whether the trade
lost money. CLAUDE.md §8.1's cooldown exists because being stopped out means the market
disagreed with our entry; a position closed at T2 for a small loss on charges is not the same
event and must not be treated as one.

Failure posture
---------------
Every public entry point absorbs its own exceptions. A malformed postback from the broker — or
a field they renamed last week — must not take down the process that owns the fill loop, and a
listener that dies silently would leave positions permanently open in local state while the
broker shows flat.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import TradingMode
from tachyon.core.logger import get_logger
from tachyon.core.state import DailyLock
from tachyon.execution.charges import DEFAULT_SCHEDULE, ChargeSchedule, estimate_charges
from tachyon.persistence.journal import JsonlJournal

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")

#: Statuses that mean quantity actually changed hands.
FILL_STATUSES: Final[frozenset[str]] = frozenset({"complete", "filled", "traded"})

#: Statuses in which an order is finished and will not fill further.
TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"complete", "filled", "traded", "cancelled", "canceled", "rejected"}
)

#: Order types that indicate the exit was a stop-out rather than a target or a square-off.
#: Matched as substrings because Angel One spells these several ways across endpoints.
_STOP_MARKERS: Final[tuple[str, ...]] = ("stoploss", "stop_loss", "sl-m", "sl_m", "slm")

#: Callback fired once, when a symbol's net quantity returns to zero.
#: ``(symbol, realised_inr, charges_inr, was_stop_out, at_ist)``.
ClosedCallback = Callable[[str, Decimal, Decimal, bool, datetime], None]

#: Callback fired for every accepted update, for telemetry. Must not raise.
FillCallback = Callable[["OrderUpdate"], None]

#: Callback fired when a fill arrives for an order this system never placed.
RogueCallback = Callable[["OrderUpdate"], None]

#: Callback fired **after** a genuine fill has been booked into the ledger:
#: ``(update, realised_net_inr_or_None)``. The net is present only on the fill that flattened
#: the symbol, and is ``realised − charges``.
#:
#: Distinct from :data:`FillCallback`, which fires *before* booking and therefore cannot know
#: whether the fill closed anything. Persisting a trade row needs both facts at once, so it
#: needs this hook rather than that one. Must not raise.
BookedFillCallback = Callable[["OrderUpdate", "Decimal | None"], None]

#: Client-tag prefix that identifies an order as ours (CLAUDE.md §6.4). A fill carrying this
#: is ours even if the in-memory registry was lost to a restart.
OUR_ORDER_TAG_PREFIX: Final[str] = "TCHYN-"

#: Reason written into the day lock when a rogue fill is detected.
ROGUE_FILL_REASON: Final[str] = "ROGUE_FILL"


def _decimal(value: object, default: Decimal = ZERO) -> Decimal:
    """Parse money from a broker field. Never raises, never returns NaN."""
    if value is None or value == "":
        return default
    try:
        parsed = Decimal(str(value))
    except InvalidOperation, ValueError, TypeError:
        return default
    return parsed if parsed.is_finite() else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(Decimal(str(value or 0)))
    except InvalidOperation, ValueError, TypeError:
        return default


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    """One normalised order-status transition, from a webhook or the order book."""

    order_id: str
    symbol: str
    status: str
    side: str
    quantity: int
    filled_quantity: int
    average_price: Decimal
    order_type: str = ""
    order_tag: str = ""
    text: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def key(self) -> tuple[str, str, int]:
        """Idempotency key. A repeat of this exact triple is a duplicate delivery."""
        return (self.order_id, self.status.strip().lower(), self.filled_quantity)

    @property
    def is_fill(self) -> bool:
        return self.status.strip().lower() in FILL_STATUSES and self.filled_quantity > 0

    @property
    def is_terminal(self) -> bool:
        return self.status.strip().lower() in TERMINAL_STATUSES

    @property
    def is_buy(self) -> bool:
        return self.side.strip().upper() == "BUY"

    @property
    def is_stop_order(self) -> bool:
        """True if this order is a stop-loss, by type or by the broker's own text."""
        haystack = f"{self.order_type} {self.text}".lower()
        return any(marker in haystack for marker in _STOP_MARKERS)

    @classmethod
    def from_broker(cls, row: dict[str, Any], symbol_for: Callable[[str, str], str]) -> OrderUpdate:
        """Normalise an Angel One order row or postback body.

        Field names differ between the postback payload and ``getOrderBook``; both spellings
        are accepted. An unrecognised shape yields an update with an empty symbol, which
        :meth:`OrderStatusListener.ingest` drops — better than guessing which instrument a
        malformed message referred to.
        """
        token = str(row.get("symboltoken", row.get("symbolToken", "")) or "")
        trading_symbol = str(row.get("tradingsymbol", row.get("tradingSymbol", "")) or "")
        return cls(
            order_id=str(row.get("orderid", row.get("orderId", "")) or ""),
            symbol=symbol_for(token, trading_symbol),
            status=str(row.get("orderstatus", row.get("status", "")) or ""),
            side=str(row.get("transactiontype", row.get("transactionType", "")) or ""),
            quantity=_int(row.get("quantity")),
            filled_quantity=_int(row.get("filledshares", row.get("filledShares"))),
            average_price=_decimal(row.get("averageprice", row.get("averagePrice"))),
            order_type=str(row.get("ordertype", row.get("orderType", "")) or ""),
            order_tag=str(row.get("ordertag", row.get("orderTag", "")) or ""),
            text=str(row.get("text", "") or ""),
            raw=row,
        )


@dataclass(slots=True)
class PositionLedger:
    """Running fills for one symbol, and the arithmetic that closes it.

    Quantities are signed: positive is long, negative is short, zero is flat. Turnover is
    accumulated separately per side so the charge estimate has the numbers it needs without
    re-deriving them from an average price that has already been rounded.
    """

    symbol: str
    net_quantity: int = 0
    buy_quantity: int = 0
    sell_quantity: int = 0
    buy_turnover: Decimal = ZERO
    sell_turnover: Decimal = ZERO
    orders: int = 0
    opened_at_ist: datetime | None = None
    closed_by_stop: bool = False

    @property
    def is_flat(self) -> bool:
        return self.net_quantity == 0

    @property
    def gross_realised(self) -> Decimal:
        """Sell turnover minus buy turnover on the *matched* quantity.

        Only meaningful once flat. While a position is open the two turnovers cover different
        quantities and their difference is not a P&L — which is exactly why the close callback
        fires on ``net_quantity == 0`` and not on every fill.
        """
        return self.sell_turnover - self.buy_turnover

    def apply(self, update: OrderUpdate, delta: int) -> None:
        """Book ``delta`` newly-filled shares from ``update``."""
        value = update.average_price * delta
        if update.is_buy:
            self.net_quantity += delta
            self.buy_quantity += delta
            self.buy_turnover += value
        else:
            self.net_quantity -= delta
            self.sell_quantity += delta
            self.sell_turnover += value

    def reset(self) -> None:
        self.net_quantity = 0
        self.buy_quantity = 0
        self.sell_quantity = 0
        self.buy_turnover = ZERO
        self.sell_turnover = ZERO
        self.orders = 0
        self.opened_at_ist = None
        self.closed_by_stop = False


@dataclass(slots=True)
class ListenerStats:
    """Counters for observability. Never used for a trading decision."""

    received: int = 0
    duplicates: int = 0
    dropped: int = 0
    fills: int = 0
    rogue_fills: int = 0
    closures: int = 0
    stop_outs: int = 0
    callback_errors: int = 0
    malformed: int = 0


class OrderStatusListener:
    """Turns broker order updates into position closures.

    Args:
        on_closed: called once per closed position. Wired to the Brain's
            ``on_position_closed``, which books P&L and starts the cooldown.
        symbol_resolver: maps ``(token, trading_symbol)`` to a watchlist symbol, or ``""``.
        on_fill: optional per-update callback, for telemetry.
        journal: append-only audit record.
        charge_schedule: cost rates for the realised-P&L estimate.

    Thread-safe: a webhook handler and a polling task may both call :meth:`ingest`.

    Example::

        listener = OrderStatusListener(
            on_closed=brain.on_position_closed_from_fills,
            symbol_resolver=resolver,
        )
        listener.ingest(OrderUpdate.from_broker(payload, resolver_pair))
    """

    __slots__ = (
        "_booked",
        "_charges",
        "_clock",
        "_daily_lock",
        "_mode",
        "_on_rogue",
        "_our_tags",
        "_ours",
        "_journal",
        "_ledgers",
        "_lock",
        "_on_closed",
        "_on_booked_fill",
        "_on_fill",
        "_resolve",
        "_seen",
        "stats",
    )

    def __init__(
        self,
        *,
        on_closed: ClosedCallback,
        symbol_resolver: Callable[[str, str], str],
        on_fill: FillCallback | None = None,
        on_booked_fill: BookedFillCallback | None = None,
        on_rogue_fill: RogueCallback | None = None,
        daily_lock: DailyLock | None = None,
        journal: JsonlJournal | None = None,
        charge_schedule: ChargeSchedule = DEFAULT_SCHEDULE,
        mode: TradingMode = TradingMode.PAPER,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._on_closed = on_closed
        self._resolve = symbol_resolver
        self._on_fill = on_fill
        self._on_booked_fill = on_booked_fill
        self._journal = (
            journal if journal is not None else JsonlJournal(prefix="fills", clock=clock)
        )
        self._charges = charge_schedule
        self._on_rogue = on_rogue_fill
        self._daily_lock = daily_lock
        self._mode = mode
        self._clock = clock

        self._ledgers: dict[str, PositionLedger] = {}
        self._seen: set[tuple[str, str, int]] = set()
        #: Highest quantity booked per order id. Separate from the dedupe set because they
        #: answer different questions: `_seen` is "have I processed this exact message", and
        #: this is "how much of this order have I already applied". Deriving the second from
        #: the first looked elegant and got the out-of-order partial case wrong.
        self._booked: dict[str, int] = {}
        #: Order ids this system placed, and the client tags it stamped on them. Either is
        #: proof of provenance; see `_is_ours`.
        self._ours: set[str] = set()
        self._our_tags: set[str] = set()
        self._lock = threading.RLock()
        self.stats = ListenerStats()

    # ── inspection ───────────────────────────────────────────────────────────

    def ledger(self, symbol: str) -> PositionLedger | None:
        with self._lock:
            return self._ledgers.get(symbol)

    def open_symbols(self) -> frozenset[str]:
        with self._lock:
            return frozenset(
                symbol for symbol, ledger in self._ledgers.items() if not ledger.is_flat
            )

    def net_quantity(self, symbol: str) -> int:
        with self._lock:
            ledger = self._ledgers.get(symbol)
        return 0 if ledger is None else ledger.net_quantity

    # ── ingestion ────────────────────────────────────────────────────────────

    def ingest_raw(self, payload: dict[str, Any]) -> bool:
        """Normalise and ingest one broker payload. Never raises.

        Returns True if the update was accepted (not a duplicate, not malformed).
        """
        try:
            update = OrderUpdate.from_broker(payload, self._pair_resolver)
        except Exception as exc:  # noqa: BLE001 - a broker field rename must not kill the loop
            self.stats.malformed += 1
            _log.error(
                "postback.malformed",
                error=str(exc),
                error_type=type(exc).__name__,
                keys=sorted(payload)[:20],
            )
            return False
        return self.ingest(update)

    def _pair_resolver(self, token: str, trading_symbol: str) -> str:
        try:
            return self._resolve(token, trading_symbol)
        except Exception as exc:  # noqa: BLE001 - a bad resolver must not kill the loop
            _log.error("postback.resolver_failed", error=str(exc))
            return ""

    def ingest(self, update: OrderUpdate) -> bool:
        """Apply one order update. Never raises.

        Returns True if it changed anything.
        """
        try:
            return self._ingest(update)
        except Exception as exc:  # noqa: BLE001 - the fill loop must survive its own bugs
            self.stats.dropped += 1
            _log.critical(
                "postback.ingest_failed",
                order_id=update.order_id,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="this fill was NOT booked — P&L and the cooldown may be wrong",
                exc_info=True,
            )
            return False

    def _ingest(self, update: OrderUpdate) -> bool:
        self.stats.received += 1

        if not update.order_id or not update.symbol:
            # Not ours, or unrecognisable. Dropping is correct: guessing which instrument a
            # malformed message referred to is how a fill gets booked against the wrong symbol.
            self.stats.dropped += 1
            _log.warning(
                "postback.unresolved",
                order_id=update.order_id,
                status=update.status,
                reason="no order id, or the instrument is not on the watchlist",
            )
            return False

        with self._lock:
            if update.key in self._seen:
                self.stats.duplicates += 1
                return False
            self._seen.add(update.key)

        self._journal.record(
            "FILL",
            "order_update",
            order_id=update.order_id,
            symbol=update.symbol,
            status=update.status,
            side=update.side,
            filled=update.filled_quantity,
            average_price=str(update.average_price),
            order_tag=update.order_tag,
        )
        self._notify_fill(update)

        if not update.is_fill:
            return True

        if not self._is_ours(update):
            self._rogue_fill(update)
            return False

        closure = self._book_fill(update)
        realised_net: Decimal | None = None
        if closure is not None:
            self._fire_closed(*closure)
            _symbol, realised, charges, _was_stop, _at = closure
            realised_net = realised - charges
        self._notify_booked(update, realised_net)
        return True

    def _book_fill(
        self, update: OrderUpdate
    ) -> tuple[str, Decimal, Decimal, bool, datetime] | None:
        """Apply the newly-filled quantity. Returns closure arguments if the symbol went flat."""
        with self._lock:
            ledger = self._ledgers.setdefault(update.symbol, PositionLedger(symbol=update.symbol))
            previously_filled = self._booked.get(update.order_id, 0)
            delta = update.filled_quantity - previously_filled
            if delta <= 0:
                # An out-of-order delivery reporting less than we have already booked. The
                # later, larger fill is authoritative; re-applying a smaller one would unwind
                # quantity that genuinely traded.
                return None
            self._booked[update.order_id] = update.filled_quantity

            was_flat = ledger.is_flat
            ledger.apply(update, delta)
            ledger.orders += 1
            if was_flat and not ledger.is_flat:
                ledger.opened_at_ist = now_ist(self._clock)
            self.stats.fills += 1

            if not ledger.is_flat:
                return None

            if update.is_stop_order:
                ledger.closed_by_stop = True

            charges = estimate_charges(
                buy_turnover=ledger.buy_turnover,
                sell_turnover=ledger.sell_turnover,
                orders=ledger.orders,
                schedule=self._charges,
            ).total
            realised = ledger.gross_realised
            was_stop_out = ledger.closed_by_stop
            ledger.reset()

        self.stats.closures += 1
        if was_stop_out:
            self.stats.stop_outs += 1

        return (update.symbol, realised, charges, was_stop_out, now_ist(self._clock))

    def _fire_closed(
        self,
        symbol: str,
        realised: Decimal,
        charges: Decimal,
        was_stop_out: bool,
        at: datetime,
    ) -> None:
        _log.info(
            "postback.position_closed",
            symbol=symbol,
            realised_inr=str(realised),
            charges_inr=str(charges),
            net_inr=str(realised - charges),
            was_stop_out=was_stop_out,
        )
        self._journal.decision(
            "position_closed",
            symbol=symbol,
            realised_inr=str(realised),
            charges_inr=str(charges),
            was_stop_out=was_stop_out,
            at_ist=at.isoformat(timespec="milliseconds"),
        )
        try:
            self._on_closed(symbol, realised, charges, was_stop_out, at)
        except Exception as exc:  # noqa: BLE001 - a bad consumer must not lose the fill loop
            self.stats.callback_errors += 1
            _log.critical(
                "postback.closed_callback_failed",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="P&L and the re-entry cooldown may not reflect this close",
                exc_info=True,
            )

    def _notify_fill(self, update: OrderUpdate) -> None:
        if self._on_fill is None:
            return
        try:
            self._on_fill(update)
        except Exception as exc:  # noqa: BLE001 - telemetry must never affect bookkeeping
            self.stats.callback_errors += 1
            _log.error("postback.fill_callback_failed", error=str(exc), exc_info=True)

    def _notify_booked(self, update: OrderUpdate, realised_net: Decimal | None) -> None:
        """Fire the post-booking hook. A failure here loses a persistence row, never a fill."""
        if self._on_booked_fill is None:
            return
        try:
            self._on_booked_fill(update, realised_net)
        except Exception as exc:  # noqa: BLE001 - persistence must never affect bookkeeping
            self.stats.callback_errors += 1
            _log.error("postback.booked_callback_failed", error=str(exc), exc_info=True)

    # ── provenance ───────────────────────────────────────────────────────────

    def register_order(self, order_id: str, order_tag: str = "") -> None:
        """Record an order this system placed, so its fills are recognised as ours.

        Called by the Brain the moment a leg is accepted. An order id we never registered and
        whose tag is not ours is a **rogue fill** — see :meth:`_rogue_fill`.
        """
        if not order_id:
            return
        with self._lock:
            self._ours.add(order_id)
            if order_tag:
                self._our_tags.add(order_tag)

    def _is_ours(self, update: OrderUpdate) -> bool:
        """Did this system place the order that produced this fill?

        Two independent proofs, and either suffices:

        * the order id was registered when we placed it;
        * the order carries one of our ``TCHYN-`` client tags (CLAUDE.md §6.4).

        The tag is what makes a **restart survivable**. After a crash the registry is empty,
        but our live orders at the exchange still carry the tag we stamped on them, so the
        boot-time order-book sweep recognises them instead of bricking the day.
        """
        with self._lock:
            if update.order_id in self._ours:
                return True
        return update.order_tag.startswith(OUR_ORDER_TAG_PREFIX)

    def _rogue_fill(self, update: OrderUpdate) -> None:
        """A fill on a watchlist symbol from an order this system never placed.

        This is not the same fault as a stale position found at boot, and it does not get the
        same response. A boot mismatch is a fact about the broker's *past* that a manual
        flatten resolves, so :class:`~tachyon.execution.reconciliation.StateReconciler` engages
        only the in-memory latch (CLAUDE.md §6.5). A fill arriving **mid-session** for an order
        we never placed means something else is trading this account right now, or our record
        of what we placed is wrong. Restarting cannot fix either, so the response is the
        **durable** day lock: this account stops trading today, and a restart does not resume.

        The fill is **not** booked. Folding a position we did not open into our P&L would
        corrupt the number the ₹500 limit is enforced against — and it is the last number that
        should be guessed at while an unexplained order is executing.

        Bricking the day on a false positive is the cost. It is bounded by the two proofs in
        :meth:`_is_ours`, and it is the right side to err on: the alternative is trading on
        against an account we demonstrably do not understand.
        """
        self.stats.rogue_fills += 1
        _log.critical(
            "postback.rogue_fill",
            order_id=update.order_id,
            symbol=update.symbol,
            side=update.side,
            filled=update.filled_quantity,
            price=str(update.average_price),
            order_tag=update.order_tag or "(none)",
            action="DAY LOCKED — a fill arrived for an order this system never placed. "
            "The fill was NOT booked. Reconcile the account manually before restarting.",
        )
        self._journal.record(
            "ERROR",
            "rogue_fill",
            order_id=update.order_id,
            symbol=update.symbol,
            order_tag=update.order_tag,
            filled=update.filled_quantity,
            payload=update.raw,
        )

        if self._daily_lock is not None:
            try:
                self._daily_lock.engage(
                    reason=ROGUE_FILL_REASON,
                    realised_pnl_inr=None,
                    mode=self._mode,
                )
            except Exception as exc:  # noqa: BLE001 - a disk fault must not lose the alarm
                _log.critical(
                    "postback.rogue_lock_failed",
                    error=str(exc),
                    impact="THE DAY LOCK WAS NOT WRITTEN — a restart would resume trading",
                )

        if self._on_rogue is not None:
            try:
                self._on_rogue(update)
            except Exception as exc:  # noqa: BLE001 - a bad consumer must not lose the alarm
                self.stats.callback_errors += 1
                _log.critical("postback.rogue_callback_failed", error=str(exc), exc_info=True)

    # ── polling backstop ─────────────────────────────────────────────────────

    def ingest_order_book(self, rows: list[dict[str, Any]]) -> int:
        """Feed a full ``getOrderBook`` response. Returns how many updates were accepted.

        The pull path. Every row is re-offered on every poll; idempotency is what makes that
        free rather than catastrophic.
        """
        return sum(1 for row in rows if self.ingest_raw(row))

    def reset_session(self) -> None:
        """Clear ledgers and the dedupe set for a new trading day."""
        with self._lock:
            self._ledgers.clear()
            self._seen.clear()
            self._booked.clear()
            self._ours.clear()
            self._our_tags.clear()
        self.stats = ListenerStats()


def watchlist_resolver(
    symbol_by_token: dict[str, str], symbol_by_trading_symbol: dict[str, str]
) -> Callable[[str, str], str]:
    """Build a resolver over the configured watchlist.

    Returns ``""`` for anything not on it. That is the §8.1 rule applied to the *inbound*
    direction: an instrument we may not trade is also one whose fills we must not book, because
    doing so would move a P&L number the kill switch is enforced against.
    """

    def resolve(token: str, trading_symbol: str) -> str:
        return symbol_by_token.get(token) or symbol_by_trading_symbol.get(trading_symbol) or ""

    return resolve
