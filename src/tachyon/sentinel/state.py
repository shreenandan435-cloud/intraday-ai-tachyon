"""Shared macro state — the Sentinel's only channel into the trading path, CLAUDE.md §5.

A single mutable object the daemon writes and the Risk Engine reads. It is deliberately tiny
and deliberately dumb: the Sentinel is advisory, and the surface through which an LLM can
affect real money should be small enough to hold in your head.

Two asymmetries govern everything here, and they point in opposite directions.

**Failure must not block trading.** Everywhere else in this system, unknown state is a veto
(CLAUDE.md §4). Here it is not. A Gemini outage, a timeout, a malformed response, an expired
API key — none of those may halt the session, because the Sentinel is an *advisor* and a broken
advisor must not become a kill switch. An unreachable Sentinel leaves
:attr:`MacroState.is_trading_allowed` exactly as it was, and a fresh process starts permissive.

**Success may only ever restrict.** The Sentinel may veto or downsize. It may never create a
signal, upsize a position, widen a stop, or unlock the loss limit. So the two fields that gate
risk are **ratchets**:

* ``is_trading_allowed`` latches ``False`` for the session and never returns to ``True``.
* ``size_multiplier`` is monotonically non-increasing within a session.

Without the ratchet, a RISK_OFF at 10:00 followed by a RISK_ON at 10:15 would *unblock*
trading — the model increasing risk, which §5 forbids outright. Only
:meth:`MacroState.reset_session` clears them, and that runs at 09:15, not on a model response.

The symbol blacklist is the same shape with an expiry: a news score above the threshold blocks
new entries in that symbol for 30 minutes. It expires because it is about one headline, not
about the day.

Degradation and recovery
------------------------
A failed classification sets ``degraded`` and increments ``consecutive_failures``; the very
next **successful** :meth:`apply` clears both and emits a ``sentinel.recovered`` event. The
poller therefore cannot get stuck in "NEUTRAL (DEGRADED)" forever: every cycle re-attempts,
and one good response ends the degraded state — non-blocking, timeout-guarded upstream in
:mod:`tachyon.sentinel.api`.

What recovery deliberately does **not** do: relax any ratchet. ``is_trading_allowed`` stays
``False`` and ``size_multiplier`` stays down once a confident RISK_OFF has landed, no matter
how many healthy responses follow. Only :meth:`reset_session` (09:15) touches those. A model
that said RISK_OFF at 10:00 does not get to un-say it at 10:15 because connectivity came back.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")
ONE: Final[Decimal] = Decimal("1")

#: Downsizing applied when the model says RISK_OFF but not confidently enough to block.
#: Halving is a judgement call, but it is the *only* direction the Sentinel may move size.
LOW_CONFIDENCE_RISK_OFF_MULTIPLIER: Final[Decimal] = Decimal("0.5")


class Regime(StrEnum):
    """Macro risk regime. ``NEUTRAL`` is the fail-safe value in every degraded path."""

    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


@dataclass(frozen=True, slots=True)
class MacroSnapshot:
    """Immutable read of the macro state, for the UI, the journal and tests."""

    regime: Regime
    confidence: int
    reason: str
    is_trading_allowed: bool
    size_multiplier: Decimal
    updated_at_ist: datetime | None
    blacklisted_symbols: tuple[str, ...]
    updates: int
    failures: int
    consecutive_failures: int
    degraded: bool
    """True when the last classification attempt failed. Trading continues regardless."""


@dataclass(slots=True)
class MacroState:
    """What the Sentinel currently believes, and what the Risk Engine is allowed to do with it.

    Args:
        risk_off_confidence: minimum confidence (0–100) at which RISK_OFF blocks entries.
        blacklist_window: how long a high news score suppresses one symbol.
        clock: injected for testing.

    Thread-safe: the Sentinel task writes while the strategy loop reads.

    Example::

        state = MacroState()
        state.apply(regime=Regime.RISK_OFF, confidence=91, reason="VIX +18%, US -2%")
        blocked, why = state.blocks_entry("RELIANCE")
    """

    risk_off_confidence: int = 80
    blacklist_window: timedelta = timedelta(minutes=30)
    clock: Clock = SYSTEM_CLOCK

    regime: Regime = Regime.NEUTRAL
    confidence: int = 0
    reason: str = "no classification yet"
    updated_at_ist: datetime | None = None

    #: Latching. Once False it stays False until :meth:`reset_session`.
    is_trading_allowed: bool = True

    #: Ratchet. Only ever decreases within a session. Clamped to [0, 1] on every write.
    size_multiplier: Decimal = ONE

    updates: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    degraded: bool = False
    degraded_since: datetime | None = None

    _blacklist: dict[str, datetime] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ── reads ────────────────────────────────────────────────────────────────

    def snapshot(self) -> MacroSnapshot:
        with self._lock:
            return MacroSnapshot(
                regime=self.regime,
                confidence=self.confidence,
                reason=self.reason,
                is_trading_allowed=self.is_trading_allowed,
                size_multiplier=self.size_multiplier,
                updated_at_ist=self.updated_at_ist,
                blacklisted_symbols=tuple(sorted(self._live_blacklist())),
                updates=self.updates,
                failures=self.failures,
                consecutive_failures=self.consecutive_failures,
                degraded=self.degraded,
            )

    @property
    def is_degraded(self) -> bool:
        """True when the last classification attempt failed. Advisory only."""
        return self.degraded

    def _live_blacklist(self, at: datetime | None = None) -> dict[str, datetime]:
        """Blacklist entries that have not yet expired. Caller holds the lock."""
        moment = at if at is not None else now_ist(self.clock)
        return {symbol: until for symbol, until in self._blacklist.items() if moment < until}

    def is_blacklisted(self, symbol: str, at: datetime | None = None) -> bool:
        with self._lock:
            return symbol in self._live_blacklist(at)

    def blacklisted_until(self, symbol: str) -> datetime | None:
        with self._lock:
            return self._blacklist.get(symbol)

    def blocks_entry(self, symbol: str, at: datetime | None = None) -> tuple[bool, str]:
        """The single question the Risk Engine asks. Returns ``(blocked, reason)``.

        Never raises and never consults the network — it is a lock-guarded read of two fields.
        The gate runs on the entry path, so anything slower would put an LLM in the order
        path, which CLAUDE.md §5 forbids outright.
        """
        with self._lock:
            if not self.is_trading_allowed:
                return (
                    True,
                    f"sentinel regime {self.regime} at {self.confidence}% confidence: "
                    f"{self.reason}",
                )
            until = self._live_blacklist(at).get(symbol)
            if until is not None:
                return (True, f"{symbol} news-blacklisted until {until:%H:%M:%S} IST")
        return (False, "")

    # ── writes ───────────────────────────────────────────────────────────────

    def apply(self, regime: Regime, confidence: int, reason: str) -> None:
        """Record a successful classification and derive its consequences.

        ``size_multiplier`` is computed **here**, from the regime and the confidence — it is
        never read from the model's response. CLAUDE.md §5 requires the clamp to live in our
        code; deriving the value outright is stronger than clamping a number the model chose,
        because there is then no number for it to choose.
        """
        confidence = max(0, min(100, confidence))
        blocks = regime is Regime.RISK_OFF and confidence >= self.risk_off_confidence

        if regime is Regime.RISK_OFF:
            target = ZERO if blocks else LOW_CONFIDENCE_RISK_OFF_MULTIPLIER
        else:
            target = ONE

        newly_blocked = False
        with self._lock:
            was_degraded = self.degraded
            degraded_since = self.degraded_since

            self.regime = regime
            self.confidence = confidence
            self.reason = reason
            self.updated_at_ist = now_ist(self.clock)
            self.updates += 1
            self.degraded = False
            self.consecutive_failures = 0
            self.degraded_since = None

            # Ratchet: never let a later, cheerier reading restore size or unblock trading.
            if target < self.size_multiplier:
                self.size_multiplier = target
            if blocks:
                newly_blocked = self.is_trading_allowed
                self.is_trading_allowed = False

        if was_degraded:
            # Recovery telemetry: the poller escaped DEGRADED on its own — no operator
            # action, no latch touched. Ratchets above were deliberately left alone.
            downtime_seconds: float | None = None
            if degraded_since is not None and self.updated_at_ist is not None:
                seconds = (self.updated_at_ist - degraded_since).total_seconds()
                downtime_seconds = round(max(seconds, 0.0), 1)
            _log.info(
                "sentinel.recovered",
                downtime_seconds=downtime_seconds,
                regime=regime,
                confidence=confidence,
                note="degraded state cleared by a successful classification; session "
                "ratchets were intentionally NOT relaxed",
            )
        if newly_blocked:
            _log.critical(
                "sentinel.risk_off",
                confidence=confidence,
                reason=reason,
                threshold=self.risk_off_confidence,
                action="NEW ENTRIES BLOCKED for the remainder of the session",
            )
        else:
            _log.info(
                "sentinel.regime",
                regime=regime,
                confidence=confidence,
                reason=reason,
                size_multiplier=str(self.size_multiplier),
                trading_allowed=self.is_trading_allowed,
            )

    def record_failure(self, detail: str) -> None:
        """Note that a classification attempt failed. **Deliberately changes nothing else.**

        The regime, the multiplier and the trading flag are left exactly as they were. An
        outage must not halt the session — the Sentinel is advisory (CLAUDE.md §5) — and it
        must equally not relax anything that a previous, successful reading tightened.

        The next successful :meth:`apply` clears the degraded state automatically; there is
        no operator step and no latch to break.
        """
        with self._lock:
            self.failures += 1
            self.consecutive_failures += 1
            if self.degraded_since is None:
                self.degraded_since = now_ist(self.clock)
            self.degraded = True
            failures = self.failures
            consecutive = self.consecutive_failures
        _log.warning(
            "sentinel.degraded",
            detail=detail,
            failures=failures,
            consecutive_failures=consecutive,
            retained_regime=self.regime,
            trading_allowed=self.is_trading_allowed,
            impact="trading continues under the last known good report; "
            "the next successful poll recovers automatically",
        )

    def blacklist(self, symbol: str, reason: str, at: datetime | None = None) -> datetime:
        """Suppress new entries in ``symbol`` for :attr:`blacklist_window`.

        Extends an existing block; never shortens one. Returns the expiry.
        """
        moment = at if at is not None else now_ist(self.clock)
        until = moment + self.blacklist_window
        with self._lock:
            existing = self._blacklist.get(symbol)
            if existing is not None and existing > until:
                until = existing
            self._blacklist[symbol] = until
        _log.warning(
            "sentinel.symbol_blacklisted",
            symbol=symbol,
            reason=reason,
            until_ist=until.isoformat(timespec="seconds"),
        )
        return until

    def reset_session(self) -> None:
        """Clear everything for a new trading day. The **only** way out of a latched block.

        Called at 09:15, never in response to a model reply. If a model response could clear
        the latch, the Sentinel could increase risk, which CLAUDE.md §5 forbids.
        """
        with self._lock:
            self.regime = Regime.NEUTRAL
            self.confidence = 0
            self.reason = "no classification yet"
            self.updated_at_ist = None
            self.is_trading_allowed = True
            self.size_multiplier = ONE
            self.updates = 0
            self.failures = 0
            self.consecutive_failures = 0
            self.degraded = False
            self.degraded_since = None
            self._blacklist.clear()
