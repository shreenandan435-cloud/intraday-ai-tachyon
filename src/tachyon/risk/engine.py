"""The risk gate — CLAUDE.md §4.

**The only component permitted to authorise an order.** Strategy code produces intents;
nothing may construct an order without a passing :class:`RiskDecision`.

Fail-safe is the whole design
-----------------------------
Every check runs inside a guard. A check that raises — a NaN comparison, a missing lock file,
a dead socket, a bug we have not found yet — produces a **veto**, not an exception. The
alternative is a traceback propagating into the strategy loop, where the natural handling is
to log and continue, and "continue" after a failed risk check means trading blind.

So: unknown state is never permission. If we cannot prove it is safe to open a position, it
is not safe to open a position.

The gate short-circuits on the first failure and logs exactly which check failed, so a
rejected signal is diagnosable without re-running anything.

The eleven checks
-----------------
========  ==========================  ====================================================
Order     Check                       Vetoes when
========  ==========================  ====================================================
1         ``ENGINE_COLD``             JIT kernels are not compiled (CLAUDE.md §3)
2         ``STATE_NOT_ACTIVE``        state machine is not ``ACTIVE``
3         ``DAILY_LOCK_ENGAGED``      the on-disk kill switch is set for today (§1.2)
4         ``OUTSIDE_ENTRY_WINDOW``    outside ``[09:20, 15:00)`` IST (§1)
5         ``FEED_STALE``              no market data for > 2 s (§2.1)
6         ``LOSS_LIMIT_BREACHED``     P&L at or past ``−₹500``, or undefined (§1.2)
7         ``POSITION_ALREADY_OPEN``   we are not flat (§4)
8         ``SYMBOL_NOT_ALLOWED``      symbol is absent from the watchlist (§8.1)
9         ``REENTRY_COOLDOWN``        stopped out of this symbol < 30 min ago (§8.1)
10        ``SENTINEL_RISK_OFF``       macro regime is RISK_OFF, or the symbol is
                                      news-blacklisted (§5)
11        ``INSUFFICIENT_MARGIN``     the broker reports too little free cash (§4)
========  ==========================  ====================================================

Cheapest and most fundamental first, so the common rejections cost almost nothing. The margin
check is last on purpose: it is the only one that can touch the network, and it should never be
reached for a signal that ten cheaper rules already refuse.

The Sentinel check is the one place in this gate where **absence is permission**. Every other
check treats unknown state as a veto; check 10 passes when no ``MacroState`` is wired, when
Gemini is down, and when the API key is missing. That inversion is deliberate and confined to
this check: the Sentinel is *advisory* (CLAUDE.md §5), and an advisory component that halts
trading when it breaks is a kill switch nobody signed off on. It may restrict; it may never be
the reason we cannot trade at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, is_entry_window_open, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.core.state import StateMachine, TradingState
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math_engine.warmup import is_warm
from tachyon.risk.tracker import PnLTracker, PositionRegistry
from tachyon.sentinel.state import MacroState

_log = get_logger(__name__)


class VetoReason(StrEnum):
    """Why an entry was refused. One value per check, plus the fail-safe catch-all."""

    ENGINE_COLD = "ENGINE_COLD"
    STATE_NOT_ACTIVE = "STATE_NOT_ACTIVE"
    DAILY_LOCK_ENGAGED = "DAILY_LOCK_ENGAGED"
    OUTSIDE_ENTRY_WINDOW = "OUTSIDE_ENTRY_WINDOW"
    FEED_STALE = "FEED_STALE"
    LOSS_LIMIT_BREACHED = "LOSS_LIMIT_BREACHED"
    POSITION_ALREADY_OPEN = "POSITION_ALREADY_OPEN"
    SYMBOL_NOT_ALLOWED = "SYMBOL_NOT_ALLOWED"
    REENTRY_COOLDOWN = "REENTRY_COOLDOWN"
    SENTINEL_RISK_OFF = "SENTINEL_RISK_OFF"
    INSUFFICIENT_MARGIN = "INSUFFICIENT_MARGIN"

    #: A check itself raised. Always a veto — see the module docstring.
    CHECK_FAILED = "CHECK_FAILED"


#: Checks required by CLAUDE.md §4 that later phases must supply. Empty as of Phase 9 — kept
#: so that a future gap is recorded here rather than discovered in a live session.
PENDING_CHECKS: Final[tuple[str, ...]] = ()

#: Supplies the broker's free cash, in rupees. Synchronous by contract: the gate runs on the
#: entry path and must not await a network call there, so the Brain refreshes a cached value
#: on its own schedule and this returns the latest reading. ``None`` means "not wired".
MarginProvider = Callable[[], Decimal]

#: A check returns None to pass, or (reason, detail) to veto.
CheckResult = tuple[VetoReason, str] | None
Check = Callable[[], CheckResult]


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The outcome of the gate. An order may only be built from an allowed decision."""

    allowed: bool
    symbol: str
    at_ist: datetime
    reason: VetoReason | None = None
    detail: str = ""
    failed_check: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class RiskEngine:
    """The eleven-step veto gate.

    Args:
        state_machine: session state.
        pnl: P&L tracker holding the latching loss limit.
        feed_monitor: liveness of the market data feed.
        positions: open positions and stop-out cooldowns.
        settings: resolved config, for the watchlist.

    Example::

        if risk.can_open_position("RELIANCE"):
            ...
        # or, when the reason matters:
        decision = risk.evaluate("RELIANCE")
        if not decision:
            telemetry.publish(decision.reason)
    """

    __slots__ = (
        "_clock",
        "_feed_monitor",
        "_macro",
        "_margin_provider",
        "_pnl",
        "_positions",
        "_settings",
        "_state_machine",
        "_vetoes",
    )

    def __init__(
        self,
        state_machine: StateMachine,
        pnl: PnLTracker,
        feed_monitor: FeedMonitor,
        positions: PositionRegistry,
        *,
        settings: Settings | None = None,
        clock: Clock = SYSTEM_CLOCK,
        margin_provider: MarginProvider | None = None,
        macro_state: MacroState | None = None,
    ) -> None:
        self._state_machine = state_machine
        self._pnl = pnl
        self._feed_monitor = feed_monitor
        self._positions = positions
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._margin_provider = margin_provider
        self._macro = macro_state
        self._vetoes: dict[VetoReason, int] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def can_open_position(self, symbol: str) -> bool:
        """True only if every check passes. Never raises."""
        return self.evaluate(symbol).allowed

    def evaluate(self, symbol: str, *, required_margin: Decimal | None = None) -> RiskDecision:
        """Run the gate and return the full decision. Never raises.

        Args:
            symbol: instrument to evaluate. Must be on the watchlist.
            required_margin: rupees the intended order will block, when the caller knows it.
                Omitted, check 10 verifies only that the broker reports positive, defined free
                cash — see :meth:`_check_margin`.

        Short-circuits on the first failure; later checks are not evaluated.
        """
        at = self._now()

        for name, check in self._checks(symbol, required_margin):
            result = self._guarded(name, check)
            if result is not None:
                reason, detail = result
                self._vetoes[reason] = self._vetoes.get(reason, 0) + 1
                _log.warning(
                    "risk.entry_vetoed",
                    symbol=symbol,
                    check=name,
                    reason=reason,
                    detail=detail,
                )
                return RiskDecision(
                    allowed=False,
                    symbol=symbol,
                    at_ist=at,
                    reason=reason,
                    detail=detail,
                    failed_check=name,
                )

        _log.info("risk.entry_authorised", symbol=symbol, headroom=str(self._pnl.headroom))
        return RiskDecision(allowed=True, symbol=symbol, at_ist=at)

    @property
    def veto_counts(self) -> dict[VetoReason, int]:
        """How often each reason has fired. Telemetry only."""
        return dict(self._vetoes)

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        """Current IST time, degrading to a stub only if the clock itself is broken."""
        try:
            return now_ist(self._clock)
        except Exception:  # noqa: BLE001 - a broken clock must not break the decision record
            return datetime.min

    def _guarded(self, name: str, check: Check) -> CheckResult:
        """Run one check so that any failure becomes a veto.

        ``BaseException`` is deliberately not caught: ``KeyboardInterrupt`` and ``SystemExit``
        mean the operator or the OS wants this process to stop, and swallowing them here would
        fight a shutdown.
        """
        try:
            return check()
        except Exception as exc:  # noqa: BLE001 - unknown state is never permission
            _log.critical(
                "risk.check_raised",
                check=name,
                error=str(exc),
                error_type=type(exc).__name__,
                action="VETO (fail-safe)",
                exc_info=True,
            )
            return (
                VetoReason.CHECK_FAILED,
                f"{name} raised {type(exc).__name__}: {exc}",
            )

    def _checks(
        self, symbol: str, required_margin: Decimal | None = None
    ) -> tuple[tuple[str, Check], ...]:
        """The gate, in evaluation order."""
        return (
            ("engine_warm", self._check_engine_warm),
            ("state_active", self._check_state_active),
            ("daily_lock", self._check_daily_lock),
            ("entry_window", self._check_entry_window),
            ("feed_fresh", self._check_feed_fresh),
            ("loss_limit", self._check_loss_limit),
            ("flat", self._check_flat),
            ("symbol_allowed", lambda: self._check_symbol_allowed(symbol)),
            ("reentry_cooldown", lambda: self._check_cooldown(symbol)),
            ("sentinel", lambda: self._check_sentinel(symbol)),
            ("margin", lambda: self._check_margin(required_margin)),
        )

    # ── the checks ───────────────────────────────────────────────────────────

    def _check_engine_warm(self) -> CheckResult:
        """1. Cold JIT kernels would compile on the entry path (CLAUDE.md §3)."""
        if not is_warm():
            return (VetoReason.ENGINE_COLD, "math engine warmup has not completed")
        return None

    def _check_state_active(self) -> CheckResult:
        """2. Entries are permitted only in ACTIVE."""
        state = self._state_machine.state
        if state is not TradingState.ACTIVE:
            return (VetoReason.STATE_NOT_ACTIVE, f"state is {state}, expected ACTIVE")
        return None

    def _check_daily_lock(self) -> CheckResult:
        """3. The on-disk kill switch, which also fails safe when unreadable (§1.2)."""
        if self._pnl.daily_lock.is_engaged():
            return (
                VetoReason.DAILY_LOCK_ENGAGED,
                f"daily lock engaged at {self._pnl.daily_lock.path}",
            )
        return None

    def _check_entry_window(self) -> CheckResult:
        """4. ``[09:20, 15:00)`` IST. Exits are never gated by this."""
        if not is_entry_window_open(clock=self._clock):
            return (
                VetoReason.OUTSIDE_ENTRY_WINDOW,
                f"{now_ist(self._clock).time().isoformat()} is outside [09:20, 15:00) IST",
            )
        return None

    def _check_feed_fresh(self) -> CheckResult:
        """5. Stale market data means our view of price is frozen (§2.1)."""
        age = self._feed_monitor.age
        if age > self._feed_monitor.stale_after:
            return (
                VetoReason.FEED_STALE,
                f"no market data for {age:.3f}s (threshold {self._feed_monitor.stale_after:.3f}s)",
            )
        return None

    def _check_loss_limit(self) -> CheckResult:
        """6. The daily loss limit, including the undefined-P&L case.

        ``is_breached`` is checked first because it is the latch: once tripped it stays
        tripped even if the numbers recover, which is the entire point of a latching switch.
        """
        if self._pnl.is_breached:
            snapshot = self._pnl.snapshot()
            return (
                VetoReason.LOSS_LIMIT_BREACHED,
                f"kill switch engaged: realised={snapshot.realised} total={snapshot.total} "
                f"limit={snapshot.limit}",
            )

        snapshot = self._pnl.snapshot()
        if snapshot.total.is_nan() or snapshot.realised.is_nan():
            return (
                VetoReason.LOSS_LIMIT_BREACHED,
                f"P&L is undefined (realised={snapshot.realised} total={snapshot.total}); "
                f"treating as breached",
            )
        if snapshot.headroom <= 0:
            return (
                VetoReason.LOSS_LIMIT_BREACHED,
                f"no headroom left: total={snapshot.total} limit={snapshot.limit}",
            )
        return None

    def _check_flat(self) -> CheckResult:
        """7. One trade at a time (CLAUDE.md §4)."""
        open_count = self._positions.open_count
        if open_count != 0:
            return (
                VetoReason.POSITION_ALREADY_OPEN,
                f"{open_count} position(s) open: {sorted(self._positions.open_symbols())}",
            )
        return None

    def _check_symbol_allowed(self, symbol: str) -> CheckResult:
        """8. Trading an instrument absent from the watchlist is forbidden (§8.1)."""
        if self._settings.find_symbol(symbol) is None:
            return (
                VetoReason.SYMBOL_NOT_ALLOWED,
                f"{symbol!r} is not in the configured watchlist",
            )
        return None

    def _check_cooldown(self, symbol: str) -> CheckResult:
        """9. No immediate re-entry after a stop-out (§8.1)."""
        if self._positions.in_cooldown(symbol, now_ist(self._clock)):
            until = self._positions.cooldown_until(symbol)
            return (
                VetoReason.REENTRY_COOLDOWN,
                f"{symbol} was stopped out recently; blocked until {until}",
            )
        return None

    def _check_sentinel(self, symbol: str) -> CheckResult:
        """10. The AI Sentinel's macro veto and news blacklist (CLAUDE.md §5).

        A lock-guarded read of two in-memory fields. It never calls Gemini, never awaits, and
        never touches the network — the Sentinel must not sit in the order path, so the gate
        reads a value the background daemon has already written.

        **No wired state means no veto.** Unlike every other check here, absence is permission:
        the Sentinel may restrict, but an outage, a missing API key or a process that simply
        never started one must not halt trading. A raise still vetoes, via :meth:`_guarded` —
        so a *broken* Sentinel is refused while an *absent* one is not.
        """
        if self._macro is None:
            return None

        blocked, reason = self._macro.blocks_entry(symbol, now_ist(self._clock))
        if blocked:
            return (VetoReason.SENTINEL_RISK_OFF, reason)
        return None

    def _check_margin(self, required: Decimal | None = None) -> CheckResult:
        """10. The broker must actually have the cash (CLAUDE.md §4, Phase 7).

        Two modes, by what the caller knows:

        * ``required`` supplied — the reading must cover it.
        * ``required`` omitted — free cash must merely be positive and defined. This is the
          normal path on the entry gate, which runs *before* a plan exists and therefore
          before the blocked amount is known.

        There is deliberately **no notional-to-margin model here.** Bracket orders are
        leveraged and the ratio is instrument- and broker-specific; inventing a multiplier
        would either wave through orders that get rejected, or refuse every trade the system
        would ever take. The real figures come from a PAPER session (CLAUDE.md §9).

        With no provider wired the check passes. That is an explicit configuration, not
        unknown state: a provider that is absent has told us nothing, whereas a provider that
        *fails* raises and :meth:`_guarded` turns that into a veto.
        """
        if self._margin_provider is None:
            return None

        margin = self._margin_provider()
        if not margin.is_finite():
            return (
                VetoReason.INSUFFICIENT_MARGIN,
                f"broker margin is undefined ({margin}) — treating as unavailable",
            )
        if margin <= 0:
            return (VetoReason.INSUFFICIENT_MARGIN, f"broker reports {margin} free cash")
        if required is not None and margin < required:
            return (
                VetoReason.INSUFFICIENT_MARGIN,
                f"need {required}, broker reports {margin} available",
            )
        return None
