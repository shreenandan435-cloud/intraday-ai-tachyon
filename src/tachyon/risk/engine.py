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
12        ``VWAP_OVEREXTENDED``       LONG entry with LTP > session VWAP by more than
                                      ``strategy.max_vwap_extension_pct`` (default 1.5 %),
                                      or more than +2σ above it in realised session
                                      volatility (the volume-weighted z-score filter)
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

import math
from collections.abc import Callable, Mapping
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
from tachyon.math.vwap_filter import DEFAULT_Z_THRESHOLD, VWAPZScoreFilter, prewarm_vwap_filter
from tachyon.math_engine.warmup import is_warm
from tachyon.risk.tracker import PnLTracker, PositionRegistry, evaluate_exits, to_decimal
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

    #: The overextension guardrail: price has run too far above session VWAP for a LONG.
    #: Chasing a vertical breakout buys the top of the move; mean reversion collects the
    #: position immediately after fill.
    VWAP_OVEREXTENDED = "VWAP_OVEREXTENDED"

    #: A check itself raised. Always a veto — see the module docstring.
    CHECK_FAILED = "CHECK_FAILED"


class ExitReason(StrEnum):
    """Why an open position must be flattened immediately."""

    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TARGET_HIT = "TARGET_HIT"

    #: The position's protective levels could not be evaluated against a usable price.
    #: Treated as a mandatory exit: with an unknown mark we cannot prove the position is
    #: safe, and for *exits* — unlike the Sentinel entry check — absence of information
    #: is NOT permission to keep holding.
    PRICE_UNUSABLE = "PRICE_UNUSABLE"


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """A forced-exit order may only be built from this."""

    symbol: str
    reason: ExitReason
    direction: str
    quantity: int
    ltp: Decimal
    threshold: Decimal
    at_ist: datetime
    action: str = "EXIT_MARKET_IMMEDIATE"

    def __bool__(self) -> bool:
        return True


#: Checks required by CLAUDE.md §4 that later phases must supply. Empty as of Phase 9 — kept
#: so that a future gap is recorded here rather than discovered in a live session.
PENDING_CHECKS: Final[tuple[str, ...]] = ()

#: Supplies the broker's free cash, in rupees. Synchronous by contract: the gate runs on the
#: entry path and must not await a network call there, so the Brain refreshes a cached value
#: on its own schedule and this returns the latest reading. ``None`` means "not wired".
MarginProvider = Callable[[], Decimal]

#: Supplies ``(ltp, session_vwap)`` for a symbol from in-memory indicator state — the same
#: values the signal generator just used. Synchronous and cheap by contract: it reads dicts
#: the Brain already holds, never the broker. Returning ``None`` means "no view of this
#: symbol", which disables the overextension check for that evaluation rather than guessing.
QuoteProvider = Callable[[str], tuple[float, float] | None]

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


@dataclass(slots=True)
class _VWAPZFilter:
    """One symbol's z-score accumulator plus the price of its most recent print.

    The last price is what lets check 12 score an entry that arrives with no
    quote of its own — Track 2's ExecutionRouter evaluates actions without a
    signal-step VWAP pair, so the accumulator's freshest print stands in.
    """

    filter: VWAPZScoreFilter
    last_price: float | None = None


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
        "_quote_provider",
        "_settings",
        "_state_machine",
        "_vetoes",
        "_vwap_filters",
        "_vwap_z_ready",
        "_vwap_z_threshold",
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
        quote_provider: QuoteProvider | None = None,
        vwap_z_threshold: float | None = None,
    ) -> None:
        self._state_machine = state_machine
        self._pnl = pnl
        self._feed_monitor = feed_monitor
        self._positions = positions
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._margin_provider = margin_provider
        self._macro = macro_state
        self._quote_provider = quote_provider
        self._vetoes: dict[StrEnum, int] = {}

        # ── the z-score overextension filter (check 12, volatility-aware clause) ──
        # The jitclass compiles lazily on first call; prewarming here — boot time, never
        # tick time — is what keeps a cold compile off the first live print (CLAUDE.md §3).
        self._vwap_z_threshold = (
            vwap_z_threshold if vwap_z_threshold is not None else DEFAULT_Z_THRESHOLD
        )
        self._vwap_z_ready = prewarm_vwap_filter()
        self._vwap_filters: dict[str, _VWAPZFilter] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def can_open_position(self, symbol: str) -> bool:
        """True only if every check passes. Never raises."""
        return self.evaluate(symbol).allowed

    def observe_tick(self, symbol: str, price: float, volume_delta: float) -> None:
        """Fold one print into the symbol's session z-score accumulator. Never raises.

        Called from the Brain's tick handler with the same differenced per-print volume the
        aggregator just consumed (:attr:`TickAggregator.last_volume_delta`), so both views
        of the session can never disagree about what was traded. The print's price is
        retained so the gate can score a quote-less entry (Track 2's ExecutionRouter path)
        against the freshest evidence it holds.

        The verdict computed here is discarded: the gate scores the judged price itself at
        decision time via :meth:`VWAPZScoreFilter.score`, which is pure — judging a price
        must not fold it into the averages it is judged against.

        Failure policy mirrors the gate itself: a non-finite price or volume is discarded
        (a NaN would poison the running sums permanently), and a kernel that somehow raises
        drops only *this symbol's* accumulator and logs — the feed, the other symbols and
        the gate all keep running.
        """
        if not self._vwap_z_ready:
            return

        if not (math.isfinite(price) and math.isfinite(volume_delta)):
            _log.debug(
                "risk.vwap_z_tick_discarded",
                symbol=symbol,
                price=price,
                volume_delta=volume_delta,
                reason="non-finite input would poison the session sums",
            )
            return

        try:
            track = self._vwap_filters.get(symbol)
            if track is None:
                track = _VWAPZFilter(
                    filter=VWAPZScoreFilter(self._vwap_z_threshold), last_price=price
                )
                self._vwap_filters[symbol] = track
            else:
                track.last_price = price

            # The verdict computed here is discarded: the gate scores the judged price
            # itself at decision time via score(), which is pure. Feeding never scores.
            track.filter.process_tick(price, volume_delta)
        except Exception as exc:  # noqa: BLE001 - fail-safe: drop the accumulator, not the gate
            self._vwap_filters.pop(symbol, None)
            _log.error(
                "risk.vwap_z_feed_failed",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
                action="z-score accumulator dropped for this symbol; percentage check remains",
            )

    def reset_vwap_session(self) -> None:
        """Clear every z-score accumulator at 09:15 IST. VWAP is session-anchored (§3.1).

        Fresh filters are built lazily on the first tick of the new session; the compiled
        code is cached, so this allocates objects but never recompiles.
        """
        self._vwap_filters.clear()

    def check_exits(
        self, ltp_by_symbol: Mapping[str, Decimal | str | int | float]
    ) -> tuple[ExitDecision, ...]:
        """Per-tick protective-exit evaluation. Call this from the live tick path.

        The 15:15 incident — a SHORT held through its Stop Loss all the way past ₹920
        until auto-square-off — happened because SL/TP levels lived in the strategy loop,
        which only compared them occasionally. This method is the fix's contract: feed it
        the latest LTP for every open symbol on **every** tick and flatten whatever comes
        back, immediately and by market order.

        Direction-aware (SHORT inverts both comparisons), ``Decimal``-exact via
        :func:`~tachyon.risk.tracker.to_decimal` — broker strings like ``"890.70"`` and
        indicator floats are handled identically, which is what killed the zero-rupee
        square-off P&L print.

        Fail-safe inversion: a price that was provided but cannot be parsed yields an
        exit decision with :attr:`ExitReason.PRICE_UNUSABLE` for that position. An
        **absent** price means the symbol simply did not tick this instant — that is
        covered by the feed-staleness guards, not by flattening. For entries, unknown
        state vetoes; for exits, corrupt state forces flattening. Holding blind is the
        one thing this engine must never do.
        """
        at = self._now()
        records = self._positions.records_snapshot()

        try:
            signals = evaluate_exits(records, ltp_by_symbol)
        except Exception as exc:  # noqa: BLE001 - fail-safe: flatten everything
            _log.critical(
                "risk.exit_evaluation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                action="EXIT_MARKET_IMMEDIATE for all open positions (fail-safe)",
                exc_info=True,
            )
            return tuple(
                ExitDecision(
                    symbol=symbol,
                    reason=ExitReason.PRICE_UNUSABLE,
                    direction=record.direction,
                    quantity=record.quantity,
                    ltp=Decimal("0"),
                    threshold=Decimal("0"),
                    at_ist=at,
                )
                for symbol, record in records.items()
            )

        decisions: list[ExitDecision] = []
        signalled: set[str] = set()
        for signal in signals:
            signalled.add(signal.symbol)
            self._vetoes[ExitReason(signal.reason)] = (
                self._vetoes.get(ExitReason(signal.reason), 0) + 1
            )
            _log.warning(
                "risk.exit_triggered",
                symbol=signal.symbol,
                reason=signal.reason,
                direction=signal.direction,
                quantity=signal.quantity,
                ltp=str(signal.ltp),
                threshold=str(signal.threshold),
                action="flatten immediately",
            )
            decisions.append(
                ExitDecision(
                    symbol=signal.symbol,
                    reason=ExitReason(signal.reason),
                    direction=signal.direction,
                    quantity=signal.quantity,
                    ltp=signal.ltp,
                    threshold=signal.threshold,
                    at_ist=at,
                )
            )

        # Fail-safe sweep: a provided-but-unparseable LTP on an open position gets a
        # mandatory-exit decision rather than silence. An absent LTP means the symbol
        # simply did not tick this instant and is deliberately not acted on here.
        for symbol, record in records.items():
            if symbol in signalled:
                continue
            raw_ltp = ltp_by_symbol.get(symbol)
            if raw_ltp is None:
                continue
            try:
                to_decimal(raw_ltp)
            except ValueError:
                self._vetoes[ExitReason.PRICE_UNUSABLE] = (
                    self._vetoes.get(ExitReason.PRICE_UNUSABLE, 0) + 1
                )
                _log.critical(
                    "risk.exit_price_unusable",
                    symbol=symbol,
                    raw_ltp=repr(raw_ltp),
                    direction=record.direction,
                    quantity=record.quantity,
                    action="EXIT_MARKET_IMMEDIATE (fail-safe: cannot prove the position is safe)",
                )
                decisions.append(
                    ExitDecision(
                        symbol=symbol,
                        reason=ExitReason.PRICE_UNUSABLE,
                        direction=record.direction,
                        quantity=record.quantity,
                        ltp=Decimal("0"),
                        threshold=Decimal("0"),
                        at_ist=at,
                    )
                )
        return tuple(decisions)

    def evaluate(
        self,
        symbol: str,
        *,
        required_margin: Decimal | None = None,
        direction: str | None = None,
        quote: tuple[float, float] | None = None,
    ) -> RiskDecision:
        """Run the gate and return the full decision. Never raises.

        Args:
            symbol: instrument to evaluate. Must be on the watchlist.
            required_margin: rupees the intended order will block, when the caller knows it.
                Omitted, check 10 verifies only that the broker reports positive, defined free
                cash — see :meth:`_check_margin`.
            direction: ``"LONG"`` or ``"SHORT"`` when the caller knows which way the signal
                points. The overextension check (12) vetoes only LONG entries; without a
                direction the check cannot attribute the signal and passes through.
            quote: ``(ltp, session_vwap)`` as evaluated by the caller's own signal step.
                Preferred over the wired :attr:`QuoteProvider` because it is the exact pair
                the signal was derived from — no chance of drift between the two reads.

        Short-circuits on the first failure; later checks are not evaluated.
        """
        at = self._now()

        for name, check in self._checks(symbol, required_margin, direction, quote):
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
    def veto_counts(self) -> dict[str, int]:
        """How often each reason has fired. Telemetry only."""
        return {reason.value: count for reason, count in self._vetoes.items()}

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
        self,
        symbol: str,
        required_margin: Decimal | None = None,
        direction: str | None = None,
        quote: tuple[float, float] | None = None,
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
            (
                "vwap_overextended",
                lambda: self._check_vwap_extension(symbol, direction, quote),
            ),
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

    def _check_vwap_extension(
        self,
        symbol: str,
        direction: str | None,
        quote: tuple[float, float] | None,
    ) -> CheckResult:
        """12. The overextension guardrail — no chasing a vertical breakout (§4).

        A LONG whose LTP sits more than ``strategy.max_vwap_extension_pct`` above session
        VWAP is refused. Buying there means paying the top of a momentum spike; the fill is
        followed immediately by mean reversion against the position. SHORT entries are
        deliberately untouched — the rule guards the "buying the top" failure, and shorting
        an overextended name is its own strategy decision.

        The volatility-aware clause consults :class:`VWAPZScoreFilter` accumulators fed by
        the tick path (:meth:`observe_tick`): a LONG is also refused when the quote LTP
        itself scores more than ``+vwap_z_threshold`` (default +2σ) above session VWAP in
        the session's own realised, volume-weighted dispersion. Scoring is pure — it reads
        the accumulators, never writes them — so the verdict tracks the price actually
        being authorised, not whichever print arrived last. Percentage and z-score are two
        views of the same failure — chasing exhaustion — so both report under
        ``VWAP_OVEREXTENDED``; either firing vetoes.

        The caller-supplied ``quote`` — the exact ``(ltp, session_vwap)`` the signal step
        evaluated — is preferred over the wired provider, which re-reads live aggregator
        state and could have drifted since. Data policy:

        * **Quote unavailable** (no explicit quote, no wired provider) disables only the
          *percentage* clause for this evaluation. The z-score clause is independent of
          VWAP quotes and still runs: it scores the quote LTP when one exists, otherwise
          the accumulator's most recent print. Track 2's ExecutionRouter arrives exactly
          this way — actions carry a token and a price context, not a signal-step VWAP
          pair — and must still be guarded.
        * **No z-score accumulator** for the symbol (no ticks observed yet, or its feed
          failed) passes the clause — absence of evidence is not evidence of extension,
          and the percentage check still guards the entry when it has inputs.
        * A provider that **raises** still vetoes, via :meth:`_guarded`.
        """
        if direction is None or direction.upper() != "LONG":
            return None

        ltp: float | None = None

        if quote is None and self._quote_provider is not None:
            # A raising provider vetoes upstream, via _guarded.
            quote = self._quote_provider(symbol)
        if quote is not None:
            vwap = quote[1]
            ltp = quote[0]
            if (
                not math.isfinite(ltp)
                or not math.isfinite(vwap)
                or vwap <= 0.0
                or ltp <= 0.0
            ):
                return None

            extension_pct = (ltp - vwap) / vwap * 100.0
            limit_pct = self._settings.strategy.max_vwap_extension_pct
            if extension_pct > limit_pct:
                return (
                    VetoReason.VWAP_OVEREXTENDED,
                    f"LTP {ltp:.2f} is {extension_pct:.2f}% above session VWAP "
                    f"{vwap:.2f} (limit {limit_pct:.2f}%) — refusing to chase the breakout",
                )

        z_track = self._vwap_filters.get(symbol)
        if z_track is None:
            return None

        z_price = ltp if ltp is not None else z_track.last_price
        if z_price is None or not math.isfinite(z_price):
            return None

        overextended, z_vwap, sigma, z_score = z_track.filter.score(z_price)
        if bool(overextended):
            return (
                VetoReason.VWAP_OVEREXTENDED,
                f"price {z_price:.2f} is {float(z_score):.2f}σ above session VWAP "
                f"{float(z_vwap):.2f} (σ={float(sigma):.4f}, limit "
                f"{self._vwap_z_threshold:.2f}σ) — refusing to chase the breakout",
            )
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
