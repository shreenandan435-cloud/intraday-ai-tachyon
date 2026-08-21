"""IST-aware time management and the monotonic square-off watchdog — CLAUDE.md §1.1, §8.

Two rules govern everything here:

1. **Every market-time decision is made in ``Asia/Kolkata``**, resolved through
   :mod:`zoneinfo`, never through the host machine's local time. A laptop set to UTC, or
   travelling, or with a wrong TZ must produce identical trading behaviour.

2. **Every deadline is measured on the monotonic clock.** A deadline is *pinned* once — the
   wall clock is read exactly one time, at arm-time, to compute an offset — and from then on
   only :func:`time.monotonic` is consulted. An NTP correction, a manual clock change, or a
   suspend/resume therefore cannot postpone the 15:15 square-off. Wall clock decides *what
   time it is*; the monotonic clock decides *how long until we act*.

The watchdog *thread* is wired in Phase 6 (``tachyon.risk.watchdog``). All of its time-checking
mechanics live here, fully implemented and testable without starting a thread.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from typing import Final, Protocol, Self
from zoneinfo import ZoneInfo

from tachyon.core.constants import (
    AUTO_SQUAREOFF_IST,
    FIRST_ENTRY_IST,
    IST_ZONE_NAME,
    MARKET_CLOSE_IST,
    MARKET_OPEN_IST,
    NO_NEW_ENTRIES_IST,
)

#: The only timezone this system reasons in.
IST: Final[ZoneInfo] = ZoneInfo(IST_ZONE_NAME)


# ──────────────────────────────────────────────────────────────────────────────
# Clock abstraction — injectable so time-dependent logic is testable deterministically
# ──────────────────────────────────────────────────────────────────────────────


class Clock(Protocol):
    """Source of truth for time. Inject a fake in tests; never patch the stdlib."""

    def now(self) -> datetime:
        """Current wall-clock time, timezone-aware, in IST."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary fixed point. Never decreases; unaffected by NTP."""
        ...


class SystemClock:
    """The real clock. Default everywhere in production."""

    __slots__ = ()

    def now(self) -> datetime:
        return datetime.now(tz=IST)

    def monotonic(self) -> float:
        return _time.monotonic()


@dataclass(slots=True)
class ManualClock:
    """Test double. Wall time and monotonic time advance only when you say so.

    Deliberately allows the two to be advanced *independently*, so tests can simulate an NTP
    jump (wall moves, monotonic does not) and assert that square-off still fires on schedule.
    """

    wall: datetime
    mono: float = 0.0

    def now(self) -> datetime:
        return self.wall.astimezone(IST)

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        """Advance both clocks together — the normal passage of time."""
        self.wall += timedelta(seconds=seconds)
        self.mono += seconds

    def jump_wall_clock(self, seconds: float) -> None:
        """Move wall time only, leaving monotonic untouched (simulates an NTP correction)."""
        self.wall += timedelta(seconds=seconds)


#: Process-wide default. Swappable in tests via the ``clock`` argument on every API here.
SYSTEM_CLOCK: Final[SystemClock] = SystemClock()


# ──────────────────────────────────────────────────────────────────────────────
# Wall-clock helpers
# ──────────────────────────────────────────────────────────────────────────────


def now_ist(clock: Clock = SYSTEM_CLOCK) -> datetime:
    """Current IST time, timezone-aware."""
    return clock.now().astimezone(IST)


def today_ist(clock: Clock = SYSTEM_CLOCK) -> date:
    """Today's *trading* date in IST — not the host's local date."""
    return now_ist(clock).date()


def to_ist(moment: datetime) -> datetime:
    """Convert any datetime to IST.

    Raises:
        ValueError: if ``moment`` is naive. Naive datetimes are banned (CLAUDE.md §8) —
            silently assuming a timezone is how a square-off ends up an hour late.
    """
    if moment.tzinfo is None:
        raise ValueError(
            "Naive datetime rejected: every timestamp must be timezone-aware (CLAUDE.md §8)."
        )
    return moment.astimezone(IST)


def ist_at(on_date: date, at_time: time) -> datetime:
    """Combine a date and a wall-clock time into an aware IST datetime."""
    return datetime.combine(on_date, at_time, tzinfo=IST)


def is_market_open(at: datetime | None = None, clock: Clock = SYSTEM_CLOCK) -> bool:
    """True during regular NSE hours, 09:15–15:30 IST. Does not consider holidays."""
    moment = to_ist(at) if at is not None else now_ist(clock)
    return MARKET_OPEN_IST <= moment.time() < MARKET_CLOSE_IST


def is_entry_window_open(at: datetime | None = None, clock: Clock = SYSTEM_CLOCK) -> bool:
    """True only within ``[09:20, 15:00)`` IST — the window in which *new* positions may open.

    Exits are never gated by this; they are always permitted.
    """
    moment = to_ist(at) if at is not None else now_ist(clock)
    return FIRST_ENTRY_IST <= moment.time() < NO_NEW_ENTRIES_IST


def is_past_squareoff(at: datetime | None = None, clock: Clock = SYSTEM_CLOCK) -> bool:
    """True at or after 15:15 IST. Used for the *boot-time* check only.

    In a running session, elapse is determined by :class:`SessionWatchdog` on the monotonic
    clock, not by re-reading the wall clock.
    """
    moment = to_ist(at) if at is not None else now_ist(clock)
    return moment.time() >= AUTO_SQUAREOFF_IST


# ──────────────────────────────────────────────────────────────────────────────
# Monotonic deadlines
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MonotonicDeadline:
    """A point in time pinned to the monotonic clock at arm-time.

    After construction the wall clock is never consulted again. This is the mechanism that
    makes CLAUDE.md §1.1 true: "if the data feed dies, square-off still fires" — and equally,
    if the *clock* is corrected, square-off still fires at the right moment.

    A deadline whose target has already passed at arm-time is immediately elapsed. It never
    rolls forward to the next day: this is an intraday system, and a missed deadline must
    surface as "act now", not "act tomorrow".
    """

    label: str
    target_wall: datetime
    deadline_mono: float

    @classmethod
    def arm(
        cls,
        label: str,
        at_time: time,
        clock: Clock = SYSTEM_CLOCK,
        on_date: date | None = None,
    ) -> Self:
        """Pin ``at_time`` on ``on_date`` (default: today IST) to the monotonic clock."""
        wall_now = now_ist(clock)
        mono_now = clock.monotonic()
        target = ist_at(on_date if on_date is not None else wall_now.date(), at_time)
        offset = (target - wall_now).total_seconds()
        return cls(label=label, target_wall=target, deadline_mono=mono_now + offset)

    def remaining(self, clock: Clock = SYSTEM_CLOCK) -> float:
        """Seconds until the deadline. Negative once it has passed."""
        return self.deadline_mono - clock.monotonic()

    def has_elapsed(self, clock: Clock = SYSTEM_CLOCK) -> bool:
        """True once the deadline is reached, on the monotonic clock alone."""
        return self.remaining(clock) <= 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Session watchdog — scaffold for tachyon.risk.watchdog (Phase 6)
# ──────────────────────────────────────────────────────────────────────────────


class SessionEvent(StrEnum):
    """Time-triggered session events, in the order they occur."""

    ENTRIES_OPEN = "ENTRIES_OPEN"  # 09:20 — first entry permitted
    NO_NEW_ENTRIES = "NO_NEW_ENTRIES"  # 15:00 — entry gate closes
    SQUARE_OFF = "SQUARE_OFF"  # 15:15 — cancel all, exit all, latch terminal
    MARKET_CLOSE = "MARKET_CLOSE"  # 15:30 — session over


_EVENT_TIMES: Final[dict[SessionEvent, time]] = {
    SessionEvent.ENTRIES_OPEN: FIRST_ENTRY_IST,
    SessionEvent.NO_NEW_ENTRIES: NO_NEW_ENTRIES_IST,
    SessionEvent.SQUARE_OFF: AUTO_SQUAREOFF_IST,
    SessionEvent.MARKET_CLOSE: MARKET_CLOSE_IST,
}


@dataclass(slots=True)
class SessionWatchdog:
    """Latching, monotonic-clock timer set for the trading session.

    Arms one :class:`MonotonicDeadline` per :class:`SessionEvent` at construction, then
    reports events as they elapse. Firing is **latching**: once an event has been reported it
    stays fired for the life of the object, so a clock adjustment can never "un-fire"
    square-off.

    This class only *decides*. It performs no I/O, holds no lock, and starts no thread —
    Phase 6 (``tachyon.risk.watchdog``) supplies the daemon thread that calls :meth:`poll`
    every :data:`~tachyon.core.constants.WATCHDOG_TICK` and dispatches the results.

    Example::

        wd = SessionWatchdog.arm_for_today(clock)
        while True:
            for event in wd.poll():
                if event is SessionEvent.SQUARE_OFF:
                    risk_engine.square_off_all()
    """

    deadlines: dict[SessionEvent, MonotonicDeadline]
    clock: Clock = SYSTEM_CLOCK
    _fired: set[SessionEvent] = field(default_factory=set)

    @classmethod
    def arm_for_today(cls, clock: Clock = SYSTEM_CLOCK, on_date: date | None = None) -> Self:
        """Arm every session deadline against ``on_date`` (default: today IST)."""
        session_date = on_date if on_date is not None else today_ist(clock)
        deadlines = {
            event: MonotonicDeadline.arm(event.value, at_time, clock=clock, on_date=session_date)
            for event, at_time in _EVENT_TIMES.items()
        }
        return cls(deadlines=deadlines, clock=clock)

    def poll(self) -> tuple[SessionEvent, ...]:
        """Return events that have elapsed since the previous call, in chronological order.

        Idempotent per event: each is returned exactly once, then latched.
        """
        newly_fired = [
            event
            for event, deadline in self.deadlines.items()
            if event not in self._fired and deadline.has_elapsed(self.clock)
        ]
        newly_fired.sort(key=lambda event: self.deadlines[event].target_wall)
        self._fired.update(newly_fired)
        return tuple(newly_fired)

    def has_fired(self, event: SessionEvent) -> bool:
        """True if ``event`` has already been reported by :meth:`poll`."""
        return event in self._fired

    def seconds_until(self, event: SessionEvent) -> float:
        """Monotonic seconds remaining until ``event``. Negative once passed."""
        return self.deadlines[event].remaining(self.clock)

    def next_poll_delay(self, tick: float) -> float:
        """Sleep duration for the driving thread.

        Returns the smaller of ``tick`` and the time to the next unfired deadline, so the
        thread wakes exactly on the deadline rather than up to one tick late — while never
        sleeping past it and never busy-spinning.
        """
        pending = [
            self.seconds_until(event) for event in self.deadlines if not self.has_fired(event)
        ]
        soonest = min((s for s in pending if s > 0.0), default=tick)
        return max(0.0, min(tick, soonest))
