"""Boot-time state reconciliation — CLAUDE.md §9.

*"On boot, reconcile local state against the broker order book and positions before accepting
any signal. **Never assume flat.**"*

This module is the implementation of that sentence. It runs once, before the strategy loop is
allowed to produce anything, and answers one question: does the broker agree with us about what
we are exposed to?

Why it exists
-------------
The dangerous case is not an orderly restart. It is a hard crash — power loss, an OOM kill, a
blue screen — at 11:40 with a live bracket at the exchange. The new process starts with an empty
:class:`~tachyon.risk.tracker.PositionRegistry`, its risk gate happily reports ``flat``, and it
opens a *second* position on top of the first. Two brackets on one instrument is double the
intended risk, and the ₹500 limit was sized against one.

The policy
----------
Any of these locks the session:

* the broker reports an **open position** we do not know about;
* the broker reports a **working order** we do not know about;
* the broker **cannot be queried at all**.

The third is not paranoia. "We could not read the order book" and "the order book is empty" are
different facts, and only one of them permits trading. Unknown state is never permission
(CLAUDE.md §4).

Why ``lock_out`` and not the daily lock file
--------------------------------------------
A mismatch engages the in-memory :data:`~tachyon.core.state.TradingState.LOCKED` latch, **not**
``data/journal/daily_lock.txt``. The daily lock is the loss-limit kill switch: it bricks the
entire day, deliberately, because a ₹500 loss is a fact about the day. A reconciliation mismatch
is a fact about the *broker's current state* — if the operator flattens the stray position
manually, the next boot re-reads the book, finds it clean, and proceeds. Writing the day-lock
here would punish an operator who fixed the problem correctly, and would train them to delete
lock files, which is the last habit anyone should have around a kill switch.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final, Protocol

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import TradingMode
from tachyon.core.logger import get_logger
from tachyon.core.state import StateMachine
from tachyon.execution.api import BrokerOrder, BrokerPosition, SmartApiClient
from tachyon.execution.builder import trading_symbol_for
from tachyon.execution.journal import OrderJournal
from tachyon.risk.tracker import PositionRegistry

_log = get_logger(__name__)


class FillSink(Protocol):
    """What the poller needs from a fill listener.

    A Protocol rather than an import: ``execution`` must not depend on ``ui``. The listener
    lives there because that is where the webhook is, but the dependency only ever points one
    way — execution knows nothing about the observability layer.
    """

    def ingest_order_book(self, rows: list[dict[str, Any]]) -> int:
        """Offer every row; return how many were newly accepted."""
        ...


class ReconcileOutcome(StrEnum):
    """Verdict of a reconciliation pass."""

    CLEAN = "CLEAN"
    """Broker and local state agree. Trading may proceed."""

    MISMATCH = "MISMATCH"
    """The broker holds something we do not know about. Session locked."""

    UNAVAILABLE = "UNAVAILABLE"
    """The broker could not be queried. We cannot prove we are flat, so we are not."""

    SKIPPED = "SKIPPED"
    """PAPER mode with no broker client. Nothing to reconcile against."""


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """What the broker said, and what we concluded."""

    outcome: ReconcileOutcome
    at_ist: datetime
    broker_positions: tuple[BrokerPosition, ...] = ()
    broker_open_orders: tuple[BrokerOrder, ...] = ()
    unknown_positions: tuple[str, ...] = ()
    unknown_orders: tuple[str, ...] = ()
    orphaned_local: tuple[str, ...] = ()
    """Symbols we believe are open but the broker does not report. Logged, never fatal —
    it can only make us refuse a trade, never take one blindly."""

    detail: str = ""

    @property
    def is_clean(self) -> bool:
        return self.outcome is ReconcileOutcome.CLEAN

    @property
    def may_trade(self) -> bool:
        """True only for CLEAN and SKIPPED. Everything else must lock."""
        return self.outcome in (ReconcileOutcome.CLEAN, ReconcileOutcome.SKIPPED)


#: Reason string written into the state transition, so the UI and the journal agree on why.
LOCK_REASON: Final[str] = "RECONCILIATION"


class StateReconciler:
    """Compares broker state against local state at boot.

    Args:
        client: broker client. ``None`` is permitted only in PAPER.
        state_machine: locked out on a mismatch or an unavailable broker.
        positions: local exposure registry — empty at boot, which is the point.
        settings: resolved config, for the watchlist symbol/token mapping.
        journal: append-only audit record.
        mode: PAPER without a client skips; LIVE never does.

    Example::

        report = await reconciler.run_at_boot()
        if not report.may_trade:
            ...  # already locked; the process runs read-only
    """

    __slots__ = (
        "_client",
        "_clock",
        "_journal",
        "_mode",
        "_positions",
        "_settings",
        "_state_machine",
    )

    def __init__(
        self,
        *,
        client: SmartApiClient | None,
        state_machine: StateMachine,
        positions: PositionRegistry,
        settings: Settings | None = None,
        journal: OrderJournal | None = None,
        mode: TradingMode | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._client = client
        self._state_machine = state_machine
        self._positions = positions
        self._settings = settings if settings is not None else get_settings()
        self._journal = journal if journal is not None else OrderJournal(clock=clock)
        self._mode = mode if mode is not None else self._settings.trading_mode
        self._clock = clock

    # ── symbol mapping ───────────────────────────────────────────────────────

    def _symbol_index(self) -> dict[str, str]:
        """``{broker identifier: watchlist symbol}`` for tokens and trading symbols alike.

        Both keys are indexed because the two books disagree about which one they carry, and a
        position matched by neither is by definition something we did not put on.
        """
        index: dict[str, str] = {}
        for item in self._settings.watchlist:
            index[item.token] = item.symbol
            index[trading_symbol_for(item)] = item.symbol
            index[item.symbol] = item.symbol
        return index

    def _resolve(self, token: str, trading_symbol: str) -> str | None:
        """Map a broker row back to a watchlist symbol, or ``None`` if it is not ours."""
        index = self._symbol_index()
        return index.get(token) or index.get(trading_symbol)

    # ── the pass ─────────────────────────────────────────────────────────────

    async def reconcile(self) -> ReconciliationReport:
        """Query the broker and compare. Never raises.

        A failure to query is reported as :data:`ReconcileOutcome.UNAVAILABLE`, not propagated:
        the caller's only sane response to an exception here would be to lock, so the decision
        is made where the context is.
        """
        at = now_ist(self._clock)

        if self._client is None:
            if self._mode is TradingMode.LIVE:
                # Unreachable through the executor, which refuses to construct LIVE without a
                # client — but if it ever happens, it is a lockdown, not a skip.
                return self._unavailable(at, "LIVE mode with no broker client")
            _log.warning(
                "reconcile.skipped",
                mode=self._mode,
                reason="no broker client in PAPER mode — local state is the only state",
            )
            return ReconciliationReport(outcome=ReconcileOutcome.SKIPPED, at_ist=at)

        try:
            order_rows = await self._client.order_book()
            position_rows = await self._client.positions()
        except Exception as exc:  # noqa: BLE001 - any failure here means "cannot prove flat"
            return self._unavailable(at, f"{type(exc).__name__}: {exc}")

        orders = tuple(
            order for order in (BrokerOrder.from_row(row) for row in order_rows) if order.is_open
        )
        positions = tuple(
            position
            for position in (BrokerPosition.from_row(row) for row in position_rows)
            if position.is_open
        )

        known = self._positions.open_symbols()

        unknown_positions = tuple(
            sorted(
                position.trading_symbol or position.token
                for position in positions
                if (self._resolve(position.token, position.trading_symbol) or "") not in known
            )
        )
        unknown_orders = tuple(
            sorted(
                f"{order.order_id}:{order.trading_symbol}"
                for order in orders
                if (self._resolve(order.token, order.trading_symbol) or "") not in known
            )
        )

        broker_symbols = {
            self._resolve(position.token, position.trading_symbol) for position in positions
        }
        orphaned_local = tuple(sorted(known - {s for s in broker_symbols if s is not None}))

        report = ReconciliationReport(
            outcome=(
                ReconcileOutcome.MISMATCH
                if (unknown_positions or unknown_orders)
                else ReconcileOutcome.CLEAN
            ),
            at_ist=at,
            broker_positions=positions,
            broker_open_orders=orders,
            unknown_positions=unknown_positions,
            unknown_orders=unknown_orders,
            orphaned_local=orphaned_local,
        )
        self._record(report)
        return report

    def _unavailable(self, at: datetime, detail: str) -> ReconciliationReport:
        report = ReconciliationReport(
            outcome=ReconcileOutcome.UNAVAILABLE, at_ist=at, detail=detail
        )
        self._record(report)
        return report

    def _record(self, report: ReconciliationReport) -> None:
        """Journal and log the verdict at a severity that matches its consequence."""
        self._journal.decision(
            "reconciliation",
            outcome=report.outcome.value,
            broker_positions=len(report.broker_positions),
            broker_open_orders=len(report.broker_open_orders),
            unknown_positions=list(report.unknown_positions),
            unknown_orders=list(report.unknown_orders),
            orphaned_local=list(report.orphaned_local),
            detail=report.detail,
        )

        if report.outcome is ReconcileOutcome.MISMATCH:
            _log.critical(
                "reconcile.mismatch",
                unknown_positions=list(report.unknown_positions),
                unknown_orders=list(report.unknown_orders),
                action="LOCKING — the broker holds exposure this process did not create; "
                "flatten it manually and restart",
            )
        elif report.outcome is ReconcileOutcome.UNAVAILABLE:
            _log.critical(
                "reconcile.unavailable",
                detail=report.detail,
                action="LOCKING — cannot prove the account is flat, so it is not",
            )
        else:
            _log.info(
                "reconcile.clean",
                outcome=report.outcome,
                broker_positions=len(report.broker_positions),
                broker_open_orders=len(report.broker_open_orders),
            )

        if report.orphaned_local:
            # We think we hold something the broker does not. Not fatal — the consequence is
            # that the gate refuses new entries, which errs in the safe direction — but it does
            # mean our P&L is being marked against a position that no longer exists.
            _log.error(
                "reconcile.orphaned_local_state",
                symbols=list(report.orphaned_local),
                impact="local registry believes these are open; the broker disagrees",
            )

    # ── enforcement ──────────────────────────────────────────────────────────

    def enforce(self, report: ReconciliationReport) -> bool:
        """Apply the lock policy. Returns True if trading may proceed.

        Never raises: this runs on the boot path, and a failure to *lock* must not be reported
        to the caller as a failure to *check*.
        """
        if report.may_trade:
            return True
        try:
            self._state_machine.lock_out(reason=f"{LOCK_REASON}: {report.outcome}")
        except Exception as exc:  # noqa: BLE001 - a lockdown that can fail is not a lockdown
            _log.critical(
                "reconcile.lock_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                impact="state machine may not read LOCKED — do not trade this session",
            )
        return False

    async def run_at_boot(self) -> ReconciliationReport:
        """Reconcile and enforce in one call. The only entry point the boot path needs."""
        report = await self.reconcile()
        self.enforce(report)
        return report


# ──────────────────────────────────────────────────────────────────────────────
# Order-book polling — the backstop under the webhook (CLAUDE.md §6.4, §7.4)
# ──────────────────────────────────────────────────────────────────────────────

#: Default poll cadence. ``getOrderBook`` is rate-limited to roughly 1 req/s, so 3 s leaves
#: headroom for the reconciler and the square-off path to use the same endpoint without
#: queueing behind this loop.
DEFAULT_POLL_SECONDS: Final[float] = 3.0

#: Consecutive failures before the poller starts backing off, and the ceiling it backs off to.
#: A broker that is down does not become available faster for being asked every three seconds.
_BACKOFF_AFTER: Final[int] = 3
_BACKOFF_CEILING_SECONDS: Final[float] = 30.0


@dataclass(slots=True)
class PollerStats:
    """Counters for observability. Never used for a trading decision."""

    polls: int = 0
    rows_seen: int = 0
    updates_accepted: int = 0
    failures: int = 0
    consecutive_failures: int = 0


class OrderBookPoller:
    """Sweeps the broker's order book into the fill listener on a timer.

    The webhook (:mod:`tachyon.ui.postback`) is the fast path and is *silently* lossy — a
    dropped POST looks identical to no event at all. This is the slow path that makes the loss
    survivable: every row is re-offered on every poll, and because the listener is idempotent
    (keyed by ``(order_id, status, filled_quantity)``) re-offering costs nothing and
    double-books nothing.

    Without this, one dropped postback means a position that never closes in local state: P&L
    frozen, cooldown never started, and the symbol blocked by ``POSITION_ALREADY_OPEN`` for the
    rest of the session.

    Args:
        client: broker client. ``None`` disables the poller entirely.
        listener: the fill listener rows are piped into.
        interval_seconds: poll cadence.
        on_degraded: optional callback fired the first time the poller gives up on a cycle.

    The loop never raises. A broker outage is logged and retried on a backoff; it does not stop
    the task, because the task stopping is precisely the failure that leaves fills unbooked
    while everything else looks healthy.
    """

    __slots__ = (
        "_client",
        "_clock",
        "_interval",
        "_listener",
        "_on_degraded",
        "_stopping",
        "stats",
    )

    def __init__(
        self,
        *,
        client: SmartApiClient | None,
        listener: FillSink,
        interval_seconds: float = DEFAULT_POLL_SECONDS,
        clock: Clock = SYSTEM_CLOCK,
        on_degraded: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._listener = listener
        self._interval = max(1.0, interval_seconds)
        self._clock = clock
        self._on_degraded = on_degraded
        self._stopping = asyncio.Event()
        self.stats = PollerStats()

    @property
    def is_enabled(self) -> bool:
        return self._client is not None

    @property
    def interval_seconds(self) -> float:
        return self._interval

    def _delay(self) -> float:
        """Poll interval, backed off while the broker is failing."""
        if self.stats.consecutive_failures < _BACKOFF_AFTER:
            return self._interval
        multiplier = float(2 ** min(self.stats.consecutive_failures - _BACKOFF_AFTER, 5))
        return min(self._interval * multiplier, _BACKOFF_CEILING_SECONDS)

    async def poll_once(self) -> int:
        """Fetch the order book and pipe it into the listener. Never raises.

        Returns the number of updates the listener accepted — zero when nothing changed, which
        is the normal case and the reason this is cheap to run every few seconds.
        """
        if self._client is None:
            return 0
        try:
            rows = await self._client.order_book()
        except Exception as exc:  # noqa: BLE001 - a broker outage must not kill the fill loop
            self.stats.failures += 1
            self.stats.consecutive_failures += 1
            _log.warning(
                "poller.order_book_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                consecutive_failures=self.stats.consecutive_failures,
                next_poll_seconds=self._delay(),
                impact="fills are unswept until this recovers; the webhook is still live",
            )
            if self.stats.consecutive_failures == _BACKOFF_AFTER and self._on_degraded is not None:
                with contextlib.suppress(Exception):
                    self._on_degraded(f"{type(exc).__name__}: {exc}")
            return 0

        self.stats.polls += 1
        self.stats.consecutive_failures = 0
        self.stats.rows_seen += len(rows)

        try:
            accepted = self._listener.ingest_order_book(list(rows))
        except Exception as exc:  # noqa: BLE001 - the listener absorbs its own, but be certain
            self.stats.failures += 1
            _log.error("poller.ingest_failed", error=str(exc), exc_info=True)
            return 0

        self.stats.updates_accepted += accepted
        if accepted:
            _log.info("poller.swept", rows=len(rows), accepted=accepted)
        return accepted

    async def run(self) -> None:
        """Poll until :meth:`stop`. Never raises."""
        if self._client is None:
            _log.warning(
                "poller.disabled",
                reason="no broker client",
                impact="fills arrive only via the postback webhook, which is silently lossy",
            )
            return

        _log.info("poller.started", interval_seconds=self._interval)
        while not self._stopping.is_set():
            await self.poll_once()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self._delay())
        _log.info(
            "poller.stopped",
            polls=self.stats.polls,
            accepted=self.stats.updates_accepted,
            failures=self.stats.failures,
        )

    async def stop(self) -> None:
        self._stopping.set()
