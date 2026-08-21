"""Re-entry cooldown — CLAUDE.md §1, §8.1.

After a position closes, the same symbol is blocked for
:data:`~tachyon.core.constants.REENTRY_COOLDOWN` (30 minutes). One rule, two reasons:

**Thrashing.** A VWAP-crossing strategy is at its worst when price oscillates across the
anchor. Each crossing looks like a fresh confluence, and without a cooldown the system would
re-enter within seconds of a stop-out, at a worse price, into the same chop. Three round trips
of that consumes the entire ₹500 daily budget before lunch.

**Revenge trading, mechanised.** A human doing this has a name for it. A machine doing it has
no name for it and is faster.

Scope: every exit, not only stop-outs
-------------------------------------
CLAUDE.md §8.1 mandates the cooldown after a **stop-out**. This module applies it after **any**
close by default (``strategy.cooldown_on_every_exit``), because the thrashing argument does not
care whether the previous trade won: price oscillating across VWAP produces the same repeated
signal either way, and the second entry is no better founded than the first.

That is strictly more conservative than the constitution requires, and it is deliberately the
only direction a deviation may go. Set ``cooldown_on_every_exit: false`` to fall back to the
literal §8.1 rule.

Two layers, on purpose
----------------------
:class:`~tachyon.risk.tracker.PositionRegistry` keeps its own stop-out cooldown, which the Risk
Engine checks as veto 9. This one lives in the strategy and is checked earlier, so a blocked
symbol never reaches the gate at all. They are not redundant: the registry's is the *floor*
guaranteed by §8.1 and cannot be tuned away, while this one may be widened but never narrowed
below it — :meth:`ReentryManager.__init__` refuses a shorter window.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import REENTRY_COOLDOWN
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

#: CLAUDE.md §1. The frozen constant, in minutes, for readability at call sites and in the UI.
#: Sourced from the constant — never re-declared, so it cannot drift.
REENTRY_COOLDOWN_MINUTES: Final[float] = REENTRY_COOLDOWN.total_seconds() / 60.0


class CooldownTooShortError(ValueError):
    """A cooldown shorter than the constitutional minimum was requested.

    The 30-minute window is a §1 hard constant. It may be *lengthened* by configuration —
    that is a more conservative choice — but never shortened, because shortening it is exactly
    the change a bad afternoon makes attractive.
    """


@dataclass(frozen=True, slots=True)
class CooldownVerdict:
    """Whether a symbol may be re-entered, and if not, when it may be."""

    symbol: str
    allowed: bool
    last_exit_ist: datetime | None
    blocked_until_ist: datetime | None
    remaining_seconds: float
    was_stop_out: bool = False

    def __bool__(self) -> bool:
        return self.allowed

    @property
    def detail(self) -> str:
        if self.allowed:
            return f"{self.symbol}: no cooldown in effect"
        return (
            f"{self.symbol}: blocked until {self.blocked_until_ist:%H:%M:%S} IST "
            f"({self.remaining_seconds:.0f}s remaining"
            f"{', after a stop-out' if self.was_stop_out else ''})"
        )


@dataclass(frozen=True, slots=True)
class ExitRecord:
    """One closed trade."""

    symbol: str
    at_ist: datetime
    was_stop_out: bool


class ReentryManager:
    """Tracks the last exit per symbol and enforces the re-entry window.

    Args:
        cooldown: window length. Defaults to the frozen constant; may be longer, never shorter.
        clock: injected for testing.
        apply_to_every_exit: if False, only stop-outs start a cooldown (the literal §8.1 rule).

    Raises:
        CooldownTooShortError: ``cooldown`` is below :data:`REENTRY_COOLDOWN`.

    Thread-safe: the square-off watchdog books exits while the strategy loop reads.

    Example::

        if not reentry.may_enter("RELIANCE"):
            return
        ...
        reentry.record_exit("RELIANCE", was_stop_out=True)
    """

    __slots__ = ("_apply_to_every_exit", "_clock", "_cooldown", "_exits", "_lock", "_blocked")

    def __init__(
        self,
        cooldown: timedelta = REENTRY_COOLDOWN,
        *,
        clock: Clock = SYSTEM_CLOCK,
        apply_to_every_exit: bool = True,
    ) -> None:
        if cooldown < REENTRY_COOLDOWN:
            raise CooldownTooShortError(
                f"cooldown={cooldown} is shorter than the constitutional minimum "
                f"{REENTRY_COOLDOWN} (CLAUDE.md §8.1). It may be lengthened, never shortened."
            )
        self._cooldown = cooldown
        self._clock = clock
        self._apply_to_every_exit = apply_to_every_exit
        self._exits: dict[str, ExitRecord] = {}
        self._blocked = 0
        self._lock = threading.RLock()

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def cooldown(self) -> timedelta:
        return self._cooldown

    @property
    def blocked_count(self) -> int:
        """How many entries this manager has refused. Telemetry only."""
        return self._blocked

    def last_exit(self, symbol: str) -> ExitRecord | None:
        with self._lock:
            return self._exits.get(symbol)

    def blocked_until(self, symbol: str) -> datetime | None:
        """When ``symbol`` becomes eligible again, or ``None`` if it already is."""
        with self._lock:
            record = self._exits.get(symbol)
        return None if record is None else record.at_ist + self._cooldown

    def cooling_symbols(self, at: datetime | None = None) -> frozenset[str]:
        """Symbols currently in cooldown. For the UI."""
        moment = at if at is not None else now_ist(self._clock)
        with self._lock:
            records = tuple(self._exits.values())
        return frozenset(
            record.symbol for record in records if moment < record.at_ist + self._cooldown
        )

    # ── the decision ─────────────────────────────────────────────────────────

    def check(self, symbol: str, at: datetime | None = None) -> CooldownVerdict:
        """Full verdict for ``symbol``, including how long is left."""
        moment = at if at is not None else now_ist(self._clock)
        with self._lock:
            record = self._exits.get(symbol)

        if record is None:
            return CooldownVerdict(
                symbol=symbol,
                allowed=True,
                last_exit_ist=None,
                blocked_until_ist=None,
                remaining_seconds=0.0,
            )

        until = record.at_ist + self._cooldown
        remaining = (until - moment).total_seconds()

        # `>=` is deliberate: at exactly last_exit + 30:00 the window has elapsed. A strict `>`
        # would leave the boundary permanently blocked on a clock with coarse resolution.
        if remaining <= 0.0:
            return CooldownVerdict(
                symbol=symbol,
                allowed=True,
                last_exit_ist=record.at_ist,
                blocked_until_ist=until,
                remaining_seconds=0.0,
                was_stop_out=record.was_stop_out,
            )

        self._blocked += 1
        return CooldownVerdict(
            symbol=symbol,
            allowed=False,
            last_exit_ist=record.at_ist,
            blocked_until_ist=until,
            remaining_seconds=remaining,
            was_stop_out=record.was_stop_out,
        )

    def may_enter(self, symbol: str, at: datetime | None = None) -> bool:
        """True if ``symbol`` is outside its cooldown window."""
        return self.check(symbol, at).allowed

    # ── recording ────────────────────────────────────────────────────────────

    def record_exit(
        self,
        symbol: str,
        *,
        was_stop_out: bool = False,
        at: datetime | None = None,
    ) -> None:
        """Start the cooldown for ``symbol``.

        A win does not reset the clock any faster than a loss when
        ``apply_to_every_exit`` is set — see the module docstring.
        """
        if not (was_stop_out or self._apply_to_every_exit):
            _log.info("strategy.cooldown_skipped", symbol=symbol, reason="not a stop-out")
            return

        moment = at if at is not None else now_ist(self._clock)
        with self._lock:
            self._exits[symbol] = ExitRecord(
                symbol=symbol, at_ist=moment, was_stop_out=was_stop_out
            )

        _log.info(
            "strategy.cooldown_started",
            symbol=symbol,
            was_stop_out=was_stop_out,
            until_ist=(moment + self._cooldown).isoformat(timespec="seconds"),
            minutes=round(self._cooldown.total_seconds() / 60.0, 1),
        )

    def reset_session(self) -> None:
        """Clear every cooldown for a new trading day.

        Called at 09:15, never mid-session. There is deliberately no ``clear(symbol)``: a
        per-symbol escape hatch is the exact thing an operator reaches for at 14:50 after a
        stop-out, which is when the rule is doing its most valuable work.
        """
        with self._lock:
            self._exits.clear()
            self._blocked = 0
