"""P&L tracking, the latching kill switch, and position bookkeeping — CLAUDE.md §1.2, §4.

Money is ``Decimal`` throughout. ``float`` is banned at the risk boundary (CLAUDE.md §8):
binary floating point cannot represent ₹0.05 exactly, and a limit that is enforced as
"₹499.99999999" rather than "₹500" is not a limit anyone can reason about.

The kill switch
---------------
On breach the tracker does three things, in this order:

1. Flips its own latch, so every later query reports breached even if the numbers move back.
2. Transitions the state machine to ``LOCKED``.
3. Writes ``data/journal/daily_lock.txt`` so a **restart cannot resume trading**.

Step 3 can fail — a full disk, a read-only directory, a permissions problem. That must not
prevent steps 1 and 2. A process that cannot persist the lock is still locked for this
session; it simply cannot guarantee the lock survives a restart, which is logged at
``CRITICAL`` because the operator needs to know the durable guarantee is gone.

What counts as a breach
-----------------------
Either of these trips it:

* **realised** P&L at or below ``−DAILY_LOSS_LIMIT_INR``
* **total** (realised + floating + charges) at or below the same limit

Tripping on realised alone would let an open loser run far past the limit before anything
fires. Tripping on total alone would miss the case where a booked loss is masked by an
unrealised gain that then evaporates. Requiring both to breach would be weaker than either.
So: whichever trips first wins.

A ``NaN`` anywhere in the arithmetic is treated as a breach, never as "unknown, carry on".
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import DAILY_LOSS_LIMIT_INR, REENTRY_COOLDOWN, TradingMode
from tachyon.core.logger import get_logger
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.core.symbols import normalize_symbol

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")

DIRECTIONS: Final[tuple[str, str]] = ("LONG", "SHORT")


def to_decimal(value: Decimal | str | int | float) -> Decimal:
    """Convert broker/exchange values to ``Decimal`` without float contamination.

    This is the choke point that kills the ``"+Rs.0.00"`` class of bug: prices arriving
    as JSON strings (``"890.70"``), floats from indicators, or ints from quantities all
    collapse to one exact type. ``float`` goes through ``str()`` first so ``Decimal(0.1)``
    becomes ``Decimal('0.1')`` rather than its binary expansion.

    Raises:
        ValueError: on anything unparseable — callers at the risk boundary treat a bad
            price as unknown state, never as zero.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("₹", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation as exc:
            raise ValueError(f"cannot parse {value!r} as a Decimal price") from exc
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ValueError(f"boolean is not a price: {value!r}")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    raise ValueError(f"unsupported price type {type(value).__name__}: {value!r}")


def _is_undefined(value: Decimal) -> bool:
    """True if a Decimal cannot be compared meaningfully.

    ``Decimal('NaN') <= x`` raises ``InvalidOperation`` rather than returning False, so an
    undefined P&L would surface as an exception deep inside a comparison. Detect it up front
    and treat it as a breach.
    """
    return value.is_nan() or value.is_infinite()


@dataclass(frozen=True, slots=True)
class TripEvent:
    """What the kill switch latched on, handed to the trip observer after the fact.

    Frozen because an observer must not be able to reach back and alter the reason the
    session was locked.
    """

    reason: str
    detail: str
    realised: Decimal
    total: Decimal
    limit: Decimal

    lock_durable: bool
    """True if ``daily_lock.txt`` was written. **False means a restart would NOT be locked**
    — the operator has to be told to stay away from the process for the rest of the day."""


#: Notified after a lockdown completes. Observation only: it cannot veto or delay the latch,
#: and :meth:`PnLTracker.trip` swallows anything it raises.
TripObserver = Callable[[TripEvent], None]


@dataclass(frozen=True, slots=True)
class PnLSnapshot:
    """Immutable read of the P&L state."""

    realised: Decimal
    floating: Decimal
    charges: Decimal
    total: Decimal
    headroom: Decimal
    breached: bool
    limit: Decimal


class PnLTracker:
    """Running P&L with a latching daily loss limit.

    Args:
        state_machine: transitioned to ``LOCKED`` on breach.
        daily_lock: on-disk latch, written on breach.
        limit: loss limit as a positive rupee amount. Defaults to the frozen constant.
        mode: recorded in the lock file so a PAPER lock is never mistaken for a LIVE one.
        on_trip: notified **after** the lockdown completes — see :meth:`trip`. Observation
            only; it cannot veto or delay the latch, and an exception from it is swallowed.

    Thread-safe: the strategy loop books fills while the watchdog thread reads totals.
    """

    __slots__ = (
        "_charges",
        "_clock",
        "_floating",
        "_limit",
        "_lock",
        "_lock_file_written",
        "_mode",
        "_on_trip",
        "_realised",
        "_state_machine",
        "_tripped",
        "daily_lock",
    )

    def __init__(
        self,
        state_machine: StateMachine,
        *,
        daily_lock: DailyLock | None = None,
        limit: Decimal = DAILY_LOSS_LIMIT_INR,
        mode: TradingMode = TradingMode.PAPER,
        on_trip: TripObserver | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._state_machine = state_machine
        self.daily_lock = daily_lock if daily_lock is not None else DailyLock(clock=clock)
        self._limit = limit
        self._mode = mode
        self._on_trip = on_trip
        self._clock = clock

        self._realised = ZERO
        self._floating = ZERO
        self._charges = ZERO
        self._tripped = False
        self._lock_file_written = False
        self._lock = threading.RLock()

    # ── reads ────────────────────────────────────────────────────────────────

    @property
    def realised(self) -> Decimal:
        with self._lock:
            return self._realised

    @property
    def floating(self) -> Decimal:
        with self._lock:
            return self._floating

    @property
    def charges(self) -> Decimal:
        with self._lock:
            return self._charges

    @property
    def total(self) -> Decimal:
        """Realised + floating − charges. The number the limit is enforced against."""
        with self._lock:
            return self._realised + self._floating - self._charges

    @property
    def limit(self) -> Decimal:
        return self._limit

    @property
    def headroom(self) -> Decimal:
        """Rupees of further loss available before the limit trips. Never negative."""
        remaining = self._limit + self.total
        return remaining if remaining > ZERO else ZERO

    @property
    def is_breached(self) -> bool:
        """True once the limit has tripped. Latching — never returns to False."""
        with self._lock:
            return self._tripped

    @property
    def lock_file_written(self) -> bool:
        """False if the breach could not be persisted, so it will not survive a restart."""
        with self._lock:
            return self._lock_file_written

    def snapshot(self) -> PnLSnapshot:
        with self._lock:
            total = self._realised + self._floating - self._charges
            return PnLSnapshot(
                realised=self._realised,
                floating=self._floating,
                charges=self._charges,
                total=total,
                headroom=self.headroom,
                breached=self._tripped,
                limit=self._limit,
            )

    # ── writes ───────────────────────────────────────────────────────────────

    def book_realised(self, amount: Decimal, charges: Decimal = ZERO) -> None:
        """Record a closed trade. ``amount`` is signed; ``charges`` is a positive cost."""
        with self._lock:
            self._realised += amount
            self._charges += charges
        self._evaluate()

    def update_floating(self, amount: Decimal) -> None:
        """Set the current mark-to-market on open positions (signed, absolute not delta)."""
        with self._lock:
            self._floating = amount
        self._evaluate()

    def add_charges(self, amount: Decimal) -> None:
        """Record brokerage, STT and friends as a positive cost."""
        with self._lock:
            self._charges += amount
        self._evaluate()

    def reset_session(self) -> None:
        """Clear P&L for a new trading day. Does not clear an engaged on-disk lock."""
        with self._lock:
            self._realised = ZERO
            self._floating = ZERO
            self._charges = ZERO
            self._tripped = False
            self._lock_file_written = False

    # ── the latch ────────────────────────────────────────────────────────────

    def _evaluate(self) -> None:
        """Trip the kill switch if either breach condition holds."""
        with self._lock:
            if self._tripped:
                return

            realised = self._realised
            total = self._realised + self._floating - self._charges

            if _is_undefined(realised) or _is_undefined(total):
                reason = "PNL_UNDEFINED"
                detail = f"realised={realised} total={total}"
            else:
                threshold = -self._limit
                try:
                    breached_realised = realised <= threshold
                    breached_total = total <= threshold
                except InvalidOperation:
                    breached_realised = breached_total = True
                    detail = f"comparison failed: realised={realised} total={total}"
                    reason = "PNL_UNCOMPARABLE"
                else:
                    if not (breached_realised or breached_total):
                        return
                    reason = (
                        "DAILY_LOSS_LIMIT_REALISED"
                        if breached_realised
                        else "DAILY_LOSS_LIMIT_TOTAL"
                    )
                    detail = f"realised={realised} total={total} limit={self._limit}"

        self.trip(reason, detail)

    def trip(self, reason: str, detail: str = "") -> None:
        """Engage the kill switch. Idempotent, and safe to call from any thread.

        Never raises. A kill switch that can fail is not a kill switch: whatever goes wrong
        while locking down, the process must end up in ``LOCKED``.
        """
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
            realised = self._realised
            total = self._realised + self._floating - self._charges

        _log.critical(
            "risk.kill_switch",
            reason=reason,
            detail=detail,
            realised=str(realised),
            total=str(total),
            limit=str(self._limit),
            action="LOCKED — cancel all, exit all, no further entries this session",
        )

        # In-memory lock first: it is what stops this process trading.
        #
        # Catches Exception, not merely StateTransitionError. Whatever goes wrong here must not
        # abort the lockdown before the durable lock below is written: the latch is already
        # set, so the worst case is a mislabelled state — far better than an unwritten lock
        # file that would let a restart resume trading after a breach.
        try:
            if not self._state_machine.is_terminal():
                self._state_machine.lock_out(reason=f"{reason}: {detail}")
        except Exception as exc:  # noqa: BLE001 - a kill switch that can fail is not one
            _log.critical(
                "risk.kill_switch.state_transition_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                impact="state label may be wrong; the in-memory latch is engaged regardless",
            )

        # Then the durable lock, which can fail on a full or read-only disk.
        try:
            self.daily_lock.engage(reason=reason, realised_pnl_inr=realised, mode=self._mode)
            with self._lock:
                self._lock_file_written = True
        except Exception as exc:  # noqa: BLE001 - never let a disk fault propagate
            _log.critical(
                "risk.kill_switch.lock_file_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                path=str(self.daily_lock.path),
                impact="this session is locked, but a RESTART WILL NOT BE — fix the disk "
                "and do not restart the process today",
            )

        # Dead last, and outside every lock. The latch, the state transition and the durable
        # lock have all already happened, so an observer cannot delay or prevent any of them
        # — it only reports. Reading `_lock_file_written` here is what lets the notification
        # say whether a RESTART would still be locked, which is the one fact the operator
        # cannot infer from the P&L number.
        with self._lock:
            durable = self._lock_file_written
        self._notify_trip(reason, detail, realised, total, durable=durable)

    def _notify_trip(
        self,
        reason: str,
        detail: str,
        realised: Decimal,
        total: Decimal,
        *,
        durable: bool,
    ) -> None:
        """Fire the trip observer. Swallows everything — :meth:`trip` never raises."""
        if self._on_trip is None:
            return
        try:
            self._on_trip(
                TripEvent(
                    reason=reason,
                    detail=detail,
                    realised=realised,
                    total=total,
                    limit=self._limit,
                    lock_durable=durable,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a kill switch that can fail is not one
            _log.critical(
                "risk.kill_switch.observer_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                impact="the lockdown is fully engaged; only the notification was lost",
            )


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """One open position's bookkeeping, including its protective levels.

    ``stop_loss`` / ``target`` are what the per-tick exit evaluation
    (:meth:`PositionRegistry.evaluate_exits`) compares the live LTP against. They live on
    the record — not in a strategy-local variable — precisely so that *some* monitored
    component can act on them even when the strategy loop itself is stuck.
    """

    symbol: str
    opened_at: datetime
    quantity: int
    entry_price: Decimal | None = None
    direction: str = "LONG"  # "LONG" | "SHORT"
    stop_loss: Decimal | None = None
    target: Decimal | None = None


@dataclass(frozen=True, slots=True)
class ExitSignal:
    """A protective exit the tick evaluator demands. Raised by the risk gate."""

    symbol: str
    reason: str  # "STOP_LOSS_HIT" | "TARGET_HIT"
    direction: str
    quantity: int
    ltp: Decimal
    threshold: Decimal


def evaluate_exits(
    records: Mapping[str, PositionRecord],
    ltp_by_symbol: Mapping[str, Decimal | str | int | float],
) -> tuple[ExitSignal, ...]:
    """Check every open position's LTP against its Stop Loss / Target.

    Direction-aware: for a LONG the SL is breached *below* and the TP *above*; for a SHORT
    both invert. The SHORT case is exactly what failed live (a short from ₹890.70 with an
    SL at ₹894.85 rode all the way to ₹920+ because nothing inverted the comparison).

    LTP values pass through :func:`to_decimal`, so broker strings and indicator floats
    are handled identically. Symbols without a usable LTP are skipped here — the caller
    decides fail-safe policy for unknown prices; this function only reports provable hits.

    Both key sets pass through :func:`~tachyon.core.symbols.normalize_symbol` before
    matching: a broker-spelled ``"RELIANCE-EQ"`` tick must evaluate against a record
    stored as ``"RELIANCE"``, never silently miss it.
    """
    signals: list[ExitSignal] = []
    ltp_by_key = {normalize_symbol(symbol): raw for symbol, raw in ltp_by_symbol.items()}
    for symbol, record in records.items():
        raw_ltp = ltp_by_key.get(normalize_symbol(symbol))
        if raw_ltp is None:
            continue
        try:
            ltp = to_decimal(raw_ltp)
        except ValueError:
            _log.warning("risk.exit_eval.unusable_ltp", symbol=symbol, raw=repr(raw_ltp))
            continue

        if record.stop_loss is not None:
            breached = (
                ltp <= record.stop_loss if record.direction == "LONG" else ltp >= record.stop_loss
            )
            if breached:
                signals.append(
                    ExitSignal(
                        symbol=symbol,
                        reason="STOP_LOSS_HIT",
                        direction=record.direction,
                        quantity=record.quantity,
                        ltp=ltp,
                        threshold=record.stop_loss,
                    )
                )
                continue

        if record.target is not None:
            breached = ltp >= record.target if record.direction == "LONG" else ltp <= record.target
            if breached:
                signals.append(
                    ExitSignal(
                        symbol=symbol,
                        reason="TARGET_HIT",
                        direction=record.direction,
                        quantity=record.quantity,
                        ltp=ltp,
                        threshold=record.target,
                    )
                )
    return tuple(signals)


def _to_direction(value: str) -> str:
    upper = value.strip().upper()
    if upper not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {value!r}")
    return upper


class PositionRegistry:
    """What we are currently exposed to, and what we may not re-enter yet.

    Two jobs the risk gate depends on:

    * how many positions are open (CLAUDE.md §4 — one trade at a time)
    * when a symbol was last stopped out (CLAUDE.md §8.1 — 30-minute cooldown, so a
      chopping market cannot stop us out of the same name repeatedly)

    Every symbol argument is canonicalised through
    :func:`~tachyon.core.symbols.normalize_symbol` before it touches the dictionaries,
    so a caller holding a broker-spelled ``"RELIANCE-EQ"`` and one holding the bare
    ``"RELIANCE"`` address the same record. A lookup that misses because of an ``-EQ``
    suffix is exactly how square-off booked ₹0.00 while a position was open.
    """

    __slots__ = ("_cooldown", "_lock", "_open", "_stopped_out_at")

    def __init__(self, cooldown: timedelta = REENTRY_COOLDOWN) -> None:
        self._open: dict[str, PositionRecord] = {}
        self._stopped_out_at: dict[str, datetime] = {}
        self._cooldown = cooldown
        self._lock = threading.RLock()

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self._open)

    def is_open(self, symbol: str) -> bool:
        with self._lock:
            return normalize_symbol(symbol) in self._open

    def open_symbols(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._open)

    def record_entry(
        self,
        symbol: str,
        quantity: int,
        *,
        at: datetime | None = None,
        entry_price: Decimal | str | int | float | None = None,
        direction: str = "LONG",
        stop_loss: Decimal | str | int | float | None = None,
        target: Decimal | str | int | float | None = None,
    ) -> None:
        """Open a position. Protective levels are stored for per-tick exit evaluation."""
        moment = at if at is not None else now_ist()
        key = normalize_symbol(symbol)
        record = PositionRecord(
            symbol=key,
            opened_at=moment,
            quantity=quantity,
            entry_price=None if entry_price is None else to_decimal(entry_price),
            direction=_to_direction(direction),
            stop_loss=None if stop_loss is None else to_decimal(stop_loss),
            target=None if target is None else to_decimal(target),
        )
        with self._lock:
            self._open[key] = record
        _log.info("risk.position_opened", symbol=key, quantity=quantity)

    def get(self, symbol: str) -> PositionRecord | None:
        with self._lock:
            return self._open.get(normalize_symbol(symbol))

    def records_snapshot(self) -> dict[str, PositionRecord]:
        """Point-in-time copy of the open-position table, safe to iterate freely."""
        with self._lock:
            return dict(self._open)

    def record_exit(self, symbol: str, *, was_stop_out: bool, at: datetime | None = None) -> None:
        moment = at if at is not None else now_ist()
        key = normalize_symbol(symbol)
        with self._lock:
            self._open.pop(key, None)
            if was_stop_out:
                self._stopped_out_at[key] = moment
        _log.info("risk.position_closed", symbol=key, was_stop_out=was_stop_out)

    def in_cooldown(self, symbol: str, at: datetime | None = None) -> bool:
        """True if ``symbol`` was stopped out within the cooldown window."""
        moment = at if at is not None else now_ist()
        with self._lock:
            stopped = self._stopped_out_at.get(normalize_symbol(symbol))
        return stopped is not None and (moment - stopped) < self._cooldown

    def cooldown_until(self, symbol: str) -> datetime | None:
        with self._lock:
            stopped = self._stopped_out_at.get(normalize_symbol(symbol))
        return None if stopped is None else stopped + self._cooldown

    def reset_session(self) -> None:
        with self._lock:
            self._open.clear()
            self._stopped_out_at.clear()


def state_is_active(machine: StateMachine) -> bool:
    """Convenience predicate used by the gate and by tests."""
    return machine.state is TradingState.ACTIVE
