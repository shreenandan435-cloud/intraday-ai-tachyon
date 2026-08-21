"""Trading state machine and the latching daily lock — CLAUDE.md §1.2, §4.

The state machine is **monotonic**: every transition moves forward through the session, and
:data:`TradingState.LOCKED` is absorbing. There is deliberately no path back to
:data:`TradingState.ACTIVE`. A system that can un-halt itself is a system that will.

The daily lock is the on-disk half of the kill switch. Once the ₹500 limit trips, a marker is
written to ``data/journal/daily_lock.txt``; any process starting later that day reads it and
boots straight into ``LOCKED`` (read-only: telemetry and UI, zero order capability). Restarting
the process is therefore not a way to keep trading after a bad day.

**The lock fails safe.** A lock file that cannot be read or parsed is treated as *locked*, not
as absent. If we cannot prove trading is permitted, it is not permitted.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist, today_ist
from tachyon.core.constants import (
    AUTO_SQUAREOFF_IST,
    DAILY_LOCK_FILE,
    MARKET_CLOSE_IST,
    MARKET_OPEN_IST,
    NO_NEW_ENTRIES_IST,
    TradingMode,
)
from tachyon.core.logger import get_logger

_log = get_logger(__name__)


class TradingState(StrEnum):
    """Session lifecycle. Order of declaration is chronological."""

    BOOTING = "BOOTING"
    PRE_MARKET = "PRE_MARKET"
    ACTIVE = "ACTIVE"
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"
    SQUARING_OFF = "SQUARING_OFF"
    SQUARED_OFF = "SQUARED_OFF"
    LOCKED = "LOCKED"


#: Legal transitions. Anything absent is illegal and raises :class:`StateTransitionError`.
#:
#: BOOTING fans out to every state because the boot-time resolver may legitimately land
#: anywhere: a fresh 09:00 start, a 15:20 restart, or a restart into an engaged kill switch.
#: LOCKED is reachable from everywhere and escapes to nowhere.
_ALLOWED: Final[dict[TradingState, frozenset[TradingState]]] = {
    TradingState.BOOTING: frozenset(
        {
            TradingState.PRE_MARKET,
            TradingState.ACTIVE,
            TradingState.NO_NEW_ENTRIES,
            TradingState.SQUARING_OFF,
            TradingState.SQUARED_OFF,
            TradingState.LOCKED,
        }
    ),
    TradingState.PRE_MARKET: frozenset(
        {
            TradingState.ACTIVE,
            TradingState.SQUARING_OFF,
            TradingState.SQUARED_OFF,
            TradingState.LOCKED,
        }
    ),
    TradingState.ACTIVE: frozenset(
        {TradingState.NO_NEW_ENTRIES, TradingState.SQUARING_OFF, TradingState.LOCKED}
    ),
    TradingState.NO_NEW_ENTRIES: frozenset({TradingState.SQUARING_OFF, TradingState.LOCKED}),
    TradingState.SQUARING_OFF: frozenset({TradingState.SQUARED_OFF, TradingState.LOCKED}),
    TradingState.SQUARED_OFF: frozenset({TradingState.LOCKED}),
    TradingState.LOCKED: frozenset(),
}

#: States in which a *new* position may be opened. Everything else is exit-only.
_ENTRY_STATES: Final[frozenset[TradingState]] = frozenset({TradingState.ACTIVE})

#: States from which no further transition is possible.
TERMINAL_STATES: Final[frozenset[TradingState]] = frozenset({TradingState.LOCKED})


class StateTransitionError(RuntimeError):
    """An illegal state transition was attempted."""


# ──────────────────────────────────────────────────────────────────────────────
# Daily lock
# ──────────────────────────────────────────────────────────────────────────────


class LockStatus(StrEnum):
    """Result of inspecting the daily lock file."""

    ABSENT = "ABSENT"  # no lock — trading permitted
    ENGAGED = "ENGAGED"  # locked today — read-only
    STALE = "STALE"  # a previous session's lock — ignored, archived on next engage
    UNREADABLE = "UNREADABLE"  # corrupt/unparseable — treated as ENGAGED (fail safe)


@dataclass(frozen=True, slots=True)
class LockRecord:
    """Contents of an engaged daily lock."""

    lock_date: date
    engaged_at_ist: str
    reason: str
    realised_pnl_inr: Decimal | None
    mode: TradingMode


@dataclass(frozen=True, slots=True)
class DailyLock:
    """The latching, on-disk kill switch (CLAUDE.md §1.2).

    File format is deliberately trivial — first line is the ISO date, second is a JSON object:

    .. code-block:: text

        2026-08-09
        {"date": "2026-08-09", "engaged_at_ist": "...", "reason": "DAILY_LOSS_LIMIT", ...}

    The date on its own line means an operator can determine lock state with ``head -1``, and
    a truncated or partially-written second line still leaves the safety-critical fact intact.
    """

    path: Path = DAILY_LOCK_FILE
    clock: Clock = SYSTEM_CLOCK

    def status(self) -> LockStatus:
        """Inspect the lock. Unreadable is reported distinctly but treated as engaged."""
        if not self.path.is_file():
            return LockStatus.ABSENT
        try:
            first_line = self.path.read_text(encoding="utf-8").splitlines()[0].strip()
            lock_date = date.fromisoformat(first_line)
        except (OSError, IndexError, ValueError) as exc:
            _log.error(
                "daily_lock.unreadable",
                path=str(self.path),
                error=str(exc),
                action="treating as ENGAGED (fail safe)",
            )
            return LockStatus.UNREADABLE
        return LockStatus.ENGAGED if lock_date == today_ist(self.clock) else LockStatus.STALE

    def is_engaged(self) -> bool:
        """True if this session must boot read-only.

        Both ``ENGAGED`` and ``UNREADABLE`` return True: an unparseable lock is a lock.
        """
        return self.status() in (LockStatus.ENGAGED, LockStatus.UNREADABLE)

    def read(self) -> LockRecord | None:
        """Parse the lock payload, or ``None`` if absent/stale/corrupt.

        Never use this to decide whether trading is permitted — use :meth:`is_engaged`, which
        fails safe. This is for reporting and the UI banner.
        """
        if self.status() is not LockStatus.ENGAGED:
            return None
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[1]) if len(lines) > 1 else {}
            pnl = payload.get("realised_pnl_inr")
            return LockRecord(
                lock_date=date.fromisoformat(lines[0].strip()),
                engaged_at_ist=str(payload.get("engaged_at_ist", "")),
                reason=str(payload.get("reason", "unknown")),
                realised_pnl_inr=Decimal(str(pnl)) if pnl is not None else None,
                mode=TradingMode(payload.get("mode", TradingMode.PAPER.value)),
            )
        except (OSError, IndexError, ValueError):
            # json.JSONDecodeError subclasses ValueError, so it is covered here.
            return None

    def engage(
        self,
        reason: str,
        realised_pnl_inr: Decimal | None = None,
        mode: TradingMode = TradingMode.PAPER,
    ) -> LockRecord:
        """Write the lock. Idempotent within a session; archives a previous day's lock first.

        Writes atomically via a temp file and ``replace`` so a crash mid-write cannot leave a
        half-written lock — though even that case fails safe, since an unparseable lock counts
        as engaged.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.status() is LockStatus.STALE:
            self._archive_stale()

        moment = now_ist(self.clock)
        record = LockRecord(
            lock_date=moment.date(),
            engaged_at_ist=moment.isoformat(timespec="seconds"),
            reason=reason,
            realised_pnl_inr=realised_pnl_inr,
            mode=mode,
        )
        payload = {
            "date": record.lock_date.isoformat(),
            "engaged_at_ist": record.engaged_at_ist,
            "reason": record.reason,
            "realised_pnl_inr": (
                str(record.realised_pnl_inr) if record.realised_pnl_inr is not None else None
            ),
            "mode": record.mode.value,
        }
        body = f"{record.lock_date.isoformat()}\n{json.dumps(payload, sort_keys=True)}\n"

        temp = self.path.with_suffix(".tmp")
        temp.write_text(body, encoding="utf-8")
        temp.replace(self.path)

        _log.critical(
            "daily_lock.engaged",
            reason=reason,
            realised_pnl_inr=str(realised_pnl_inr),
            path=str(self.path),
        )
        return record

    def _archive_stale(self) -> None:
        """Preserve a previous session's lock instead of overwriting it."""
        try:
            previous = self.path.read_text(encoding="utf-8").splitlines()[0].strip()
            self.path.replace(self.path.with_name(f"daily_lock.{previous}.txt"))
        except (OSError, IndexError) as exc:
            _log.warning("daily_lock.archive_failed", error=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Boot resolution
# ──────────────────────────────────────────────────────────────────────────────


def resolve_boot_state(lock: DailyLock | None = None, clock: Clock = SYSTEM_CLOCK) -> TradingState:
    """Determine the state this process should start in.

    An engaged lock wins over everything. Otherwise the state follows the IST wall clock:

    ==========================  ==================
    IST wall clock              State
    ==========================  ==================
    < 09:15                     ``PRE_MARKET``
    09:15 – 15:00               ``ACTIVE``
    15:00 – 15:15               ``NO_NEW_ENTRIES``
    15:15 – 15:30               ``SQUARING_OFF``
    >= 15:30                    ``SQUARED_OFF``
    ==========================  ==================

    A restart between 15:15 and 15:30 resolves to ``SQUARING_OFF``, not ``SQUARED_OFF`` — the
    flatten routine must actually *run*, because we cannot assume the process that died had
    finished (or started) it.
    """
    daily_lock = lock if lock is not None else DailyLock(clock=clock)
    if daily_lock.is_engaged():
        _log.critical("boot.locked", reason="daily lock engaged for today — read-only mode")
        return TradingState.LOCKED

    current = now_ist(clock).time()
    if current < MARKET_OPEN_IST:
        return TradingState.PRE_MARKET
    if current < NO_NEW_ENTRIES_IST:
        return TradingState.ACTIVE
    if current < AUTO_SQUAREOFF_IST:
        return TradingState.NO_NEW_ENTRIES
    if current < MARKET_CLOSE_IST:
        return TradingState.SQUARING_OFF
    return TradingState.SQUARED_OFF


# ──────────────────────────────────────────────────────────────────────────────
# State machine
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Transition:
    """One recorded state change."""

    from_state: TradingState
    to_state: TradingState
    reason: str
    at_ist: datetime


StateListener = Callable[[Transition], None]


class StateMachine:
    """Thread-safe holder of the current :class:`TradingState`.

    Guarded by a lock because the square-off watchdog thread and the strategy loop both drive
    transitions. Reads are cheap; the contended path is only ever a handful of writes a day.
    """

    __slots__ = ("_clock", "_history", "_lock", "_listeners", "_state")

    def __init__(
        self, initial: TradingState = TradingState.BOOTING, clock: Clock = SYSTEM_CLOCK
    ) -> None:
        self._state = initial
        self._clock = clock
        self._lock = threading.RLock()
        self._listeners: list[StateListener] = []
        self._history: list[Transition] = []

    @classmethod
    def boot(cls, lock: DailyLock | None = None, clock: Clock = SYSTEM_CLOCK) -> Self:
        """Construct in ``BOOTING`` and immediately advance to the resolved boot state."""
        machine = cls(TradingState.BOOTING, clock=clock)
        machine.transition_to(resolve_boot_state(lock, clock), reason="boot resolution")
        return machine

    @property
    def state(self) -> TradingState:
        with self._lock:
            return self._state

    @property
    def history(self) -> tuple[Transition, ...]:
        with self._lock:
            return tuple(self._history)

    def can_transition_to(self, target: TradingState) -> bool:
        with self._lock:
            return target in _ALLOWED[self._state]

    def transition_to(self, target: TradingState, reason: str) -> Transition:
        """Move to ``target``.

        Raises:
            StateTransitionError: if the transition is not legal from the current state.
        """
        with self._lock:
            current = self._state
            if target not in _ALLOWED[current]:
                raise StateTransitionError(
                    f"Illegal transition {current} -> {target} ({reason}). "
                    f"Legal targets: {sorted(_ALLOWED[current]) or 'none — terminal state'}."
                )
            transition = Transition(current, target, reason, now_ist(self._clock))
            self._state = target
            self._history.append(transition)

        _log.info("state.transition", from_state=current, to_state=target, reason=reason)
        self._notify(transition)
        return transition

    def lock_out(self, reason: str) -> Transition:
        """Force ``LOCKED`` from any state. The kill switch's in-memory half.

        Never raises on an already-locked machine — a second breach reporting in must not take
        down the process that is trying to shut things off.
        """
        with self._lock:
            if self._state is TradingState.LOCKED:
                _log.warning("state.already_locked", reason=reason)
                return (
                    self._history[-1]
                    if self._history
                    else Transition(
                        TradingState.LOCKED, TradingState.LOCKED, reason, now_ist(self._clock)
                    )
                )
        return self.transition_to(TradingState.LOCKED, reason)

    def may_open_position(self) -> bool:
        """True only in ``ACTIVE``. The Risk Engine gate consults this first (CLAUDE.md §4)."""
        return self.state in _ENTRY_STATES

    def may_exit_position(self) -> bool:
        """Exits are permitted in every state. Closing risk is never blocked."""
        return True

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def add_listener(self, listener: StateListener) -> None:
        """Register a transition callback. Called outside the lock, after the change commits."""
        with self._lock:
            self._listeners.append(listener)

    def _notify(self, transition: Transition) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(transition)
            except Exception as exc:  # noqa: BLE001 - a bad listener must not stall square-off
                _log.error(
                    "state.listener_failed",
                    listener=getattr(listener, "__name__", repr(listener)),
                    error=str(exc),
                    exc_info=True,
                )
