"""The square-off watchdog thread — CLAUDE.md §1.1.

15:15 IST: cancel everything, exit everything, latch terminal. No flag, no config key and no
code path skips it.

Why a thread, and why monotonic
-------------------------------
The deadline is driven by a dedicated thread polling
:class:`~tachyon.core.clock.SessionWatchdog`, which counts down on ``time.monotonic``. Both
choices are load-bearing:

* **A thread, not a tick callback.** If square-off were driven by incoming market data, a dead
  feed would mean a dead deadline — and a dead feed is precisely when we most need to be flat.
* **Monotonic, not wall clock.** The deadline is pinned once at arm time. An NTP correction,
  a manual clock change or a suspend/resume cannot move it in either direction.

Immortality
-----------
The loop catches every ``Exception`` and continues. A watchdog that dies on an unexpected
error is worse than no watchdog, because the system carries on believing it is protected.
``BaseException`` is deliberately not caught — ``KeyboardInterrupt`` and ``SystemExit`` are
the operator asking the process to stop, and fighting that would be its own hazard.

The thread is **non-daemon** (CLAUDE.md §2.2). A daemon thread is killed the instant the main
thread exits, which is exactly the wrong behaviour for the component whose entire job is to
flatten positions during shutdown. Callers must :meth:`SquareOffWatchdog.stop` it explicitly.

Retrying the flatten
--------------------
Transitioning the state machine is bookkeeping; *actually being flat* is the point. If the
square-off callback fails — a broker timeout, a rejected cancel — the watchdog keeps retrying
on an interval until it succeeds. A one-shot attempt that failed at 15:15:00 would leave a
live position overnight with nothing watching it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, SessionEvent, SessionWatchdog, now_ist
from tachyon.core.constants import WATCHDOG_TICK
from tachyon.core.logger import get_logger
from tachyon.core.state import StateMachine, StateTransitionError, TradingState

_log = get_logger(__name__)

#: How often the loop wakes. The clock module shortens this automatically as a deadline
#: approaches, so square-off lands within milliseconds of 15:15:00 rather than up to a tick late.
DEFAULT_TICK_SECONDS: Final[float] = WATCHDOG_TICK.total_seconds()

#: Gap between retries of a failed square-off action. Long enough not to hammer a struggling
#: broker, short enough that we are flat within a minute of the deadline.
SQUARE_OFF_RETRY_SECONDS: Final[float] = 2.0

#: Called when the 15:15 deadline fires. Must cancel all orders and exit all positions.
#: Raising signals failure and schedules a retry.
SquareOffAction = Callable[[], None]

#: Returns current net P&L (realised + floating − charges). Negative is a loss. Must not block:
#: it runs on the watchdog thread between deadline polls.
DrawdownProbe = Callable[[], Decimal]

#: Called once, on the tick that breaches the drawdown limit, with the net P&L that breached it.
DrawdownBreachAction = Callable[[Decimal], None]

ZERO: Final[Decimal] = Decimal("0")

#: State transitions the watchdog drives, in chronological order.
_EVENT_TARGETS: Final[dict[SessionEvent, TradingState]] = {
    SessionEvent.ENTRIES_OPEN: TradingState.ACTIVE,
    SessionEvent.NO_NEW_ENTRIES: TradingState.NO_NEW_ENTRIES,
    SessionEvent.SQUARE_OFF: TradingState.SQUARING_OFF,
    SessionEvent.MARKET_CLOSE: TradingState.SQUARED_OFF,
}


class SquareOffWatchdog:
    """Drives the session's time-based state transitions from its own thread.

    Args:
        state_machine: transitioned as each deadline elapses.
        on_square_off: the flatten action. Retried until it returns without raising.
        clock: injected for testing; production uses the system clock.
        tick_seconds: maximum sleep between polls.
        on_date: session date to arm against. Defaults to today in IST.

    Example::

        watchdog = SquareOffWatchdog(machine, on_square_off=executor.flatten_everything)
        watchdog.start()
        ...
        watchdog.stop()
    """

    __slots__ = (
        "_action_failures",
        "_clock",
        "_iterations",
        "_last_retry_at",
        "_on_square_off",
        "_session",
        "_square_off_done",
        "_square_off_pending",
        "_state_machine",
        "_stop",
        "_drawdown_limit",
        "_drawdown_probe",
        "_drawdown_tripped",
        "_on_drawdown_breach",
        "_thread",
        "_thread_errors",
        "_tick",
    )

    def __init__(
        self,
        state_machine: StateMachine,
        *,
        on_square_off: SquareOffAction | None = None,
        clock: Clock = SYSTEM_CLOCK,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        on_date: date | None = None,
        drawdown_probe: DrawdownProbe | None = None,
        drawdown_limit: Decimal = ZERO,
        on_drawdown_breach: DrawdownBreachAction | None = None,
    ) -> None:
        self._state_machine = state_machine
        self._on_square_off = on_square_off
        self._clock = clock
        self._tick = tick_seconds
        self._session = SessionWatchdog.arm_for_today(clock, on_date=on_date)

        # The drawdown hard-stop (CLAUDE.md §1.2). Polled here as well as on every P&L update
        # because this thread runs on a monotonic clock and is immortal: a Brain that has
        # stopped consuming ticks, or an event loop that has wedged, cannot stop it firing.
        self._drawdown_probe = drawdown_probe
        self._drawdown_limit = drawdown_limit if drawdown_limit > ZERO else ZERO
        self._on_drawdown_breach = on_drawdown_breach
        self._drawdown_tripped = False

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._square_off_pending = False
        self._square_off_done = False
        self._action_failures = 0
        self._thread_errors = 0
        self._iterations = 0
        self._last_retry_at = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def set_square_off_action(self, action: SquareOffAction) -> None:
        """Install the flatten action before starting.

        Exists because the action is usually a bridge onto an asyncio event loop
        (:meth:`~tachyon.execution.executor.RoboExecutor.square_off_action`), which cannot be
        built until that loop is running — later than the watchdog is naturally constructed.

        Raises:
            RuntimeError: the thread is already running. Swapping the flatten action out from
                under a live watchdog, possibly mid-retry, is never what anyone means.
        """
        if self.is_alive:
            raise RuntimeError("cannot replace the square-off action while the watchdog is running")
        self._on_square_off = action

    def start(self) -> None:
        """Start the thread. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        # Non-daemon on purpose: this thread must survive main-thread shutdown long enough
        # to flatten. See the module docstring.
        self._thread = threading.Thread(
            target=self._run, name="risk-squareoff-watchdog", daemon=False
        )
        self._thread.start()
        _log.info(
            "risk.watchdog.started",
            square_off_in_seconds=round(self.seconds_until_square_off, 1),
            tick_seconds=self._tick,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to exit and wait for it. Safe to call more than once."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                _log.error(
                    "risk.watchdog.stop_timeout",
                    timeout=timeout,
                    impact="watchdog thread did not exit; process will not terminate cleanly",
                )
        _log.info(
            "risk.watchdog.stopped",
            iterations=self._iterations,
            thread_errors=self._thread_errors,
            square_off_fired=self._session.has_fired(SessionEvent.SQUARE_OFF),
            square_off_completed=self._square_off_done,
        )

    # ── observability ────────────────────────────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def square_off_fired(self) -> bool:
        """True once the 15:15 deadline has elapsed. Latching."""
        return self._session.has_fired(SessionEvent.SQUARE_OFF)

    @property
    def square_off_completed(self) -> bool:
        """True once the flatten action has succeeded."""
        return self._square_off_done

    @property
    def seconds_until_square_off(self) -> float:
        return self._session.seconds_until(SessionEvent.SQUARE_OFF)

    @property
    def action_failures(self) -> int:
        return self._action_failures

    @property
    def thread_errors(self) -> int:
        """Unexpected exceptions absorbed by the loop. Should always be zero."""
        return self._thread_errors

    @property
    def iterations(self) -> int:
        return self._iterations

    def has_fired(self, event: SessionEvent) -> bool:
        return self._session.has_fired(event)

    # ── the loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Poll deadlines until stopped. Never exits on error."""
        while not self._stop.is_set():
            try:
                self._iterations += 1
                self._check_drawdown()
                for event in self._session.poll():
                    self._handle(event)
                if self._square_off_pending:
                    self._retry_square_off()
            except Exception as exc:  # noqa: BLE001 - the watchdog must never die
                self._thread_errors += 1
                _log.critical(
                    "risk.watchdog.iteration_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    action="absorbed; watchdog continues",
                    exc_info=True,
                )

            delay = self._poll_delay()
            if self._stop.wait(delay):
                return

    def _poll_delay(self) -> float:
        """Sleep until the next deadline, capped at one tick. Never returns 0 forever."""
        try:
            delay = self._session.next_poll_delay(self._tick)
        except Exception:  # noqa: BLE001 - a broken schedule must not become a busy loop
            return self._tick
        return max(delay, 0.001) if self._square_off_pending is False else min(delay, self._tick)

    # ── drawdown hard-stop ───────────────────────────────────────────────────

    @property
    def drawdown_limit(self) -> Decimal:
        """Positive rupee loss at which this watchdog flattens. Zero disables the check."""
        return self._drawdown_limit

    @property
    def drawdown_tripped(self) -> bool:
        """True once breached. Latching — it never returns to False (CLAUDE.md §1.2)."""
        return self._drawdown_tripped

    def _check_drawdown(self) -> None:
        """Flatten and lock if net P&L has breached the configured drawdown.

        Fires **once**. The latch matters more than it looks: without it, every 250 ms tick
        past the threshold would re-enter the flatten path, and §6.5 is explicit that a second
        market exit does not close a position twice — it reverses it, creating fresh naked risk
        at the exact moment the system is trying to have none.

        A probe that raises counts as *no reading*, not as a breach. Unknown P&L must not
        flatten a healthy session; the P&L-update path already treats NaN/Inf as breached
        (§4 check 6), which is where that judgement belongs.
        """
        if self._drawdown_tripped or self._drawdown_limit <= ZERO or self._drawdown_probe is None:
            return

        try:
            net = self._drawdown_probe()
        except Exception as exc:  # noqa: BLE001 - the watchdog must never die
            self._thread_errors += 1
            _log.error(
                "risk.drawdown.probe_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                action="no reading this tick; the P&L update path still enforces the limit",
            )
            return

        if not net.is_finite() or net > -self._drawdown_limit:
            return

        self._drawdown_tripped = True
        _log.critical(
            "risk.drawdown.breached",
            net_inr=str(net),
            limit_inr=str(self._drawdown_limit),
            at_ist=now_ist(self._clock).isoformat(timespec="milliseconds"),
            state=self._state_machine.state,
            action="CANCEL ALL ORDERS, EXIT ALL POSITIONS, LOCK THE SESSION",
        )

        # Latch first, flatten second. If the flatten throws, the session must still be locked:
        # a system that failed to exit must certainly not open anything new.
        if self._on_drawdown_breach is not None:
            try:
                self._on_drawdown_breach(net)
            except Exception as exc:  # noqa: BLE001 - a kill switch that can fail is not one
                self._thread_errors += 1
                _log.critical(
                    "risk.drawdown.breach_action_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    impact="the in-memory latch may not have been written to disk",
                    exc_info=True,
                )

        if self._on_square_off is not None:
            self._square_off_pending = True
            self._retry_square_off(force=True)

    # ── event handling ───────────────────────────────────────────────────────

    def _handle(self, event: SessionEvent) -> None:
        """React to one elapsed deadline."""
        if event is SessionEvent.SQUARE_OFF:
            _log.critical(
                "risk.square_off.deadline",
                at_ist=now_ist(self._clock).isoformat(timespec="milliseconds"),
                state=self._state_machine.state,
                action="CANCEL ALL ORDERS, EXIT ALL POSITIONS — 15:15 IST hard deadline",
            )
            self._square_off_pending = self._on_square_off is not None
        else:
            # NB: the key is `session_event`, not `event` — structlog's first positional
            # parameter is itself named `event`, so that kwarg would collide at runtime.
            _log.info(
                "risk.watchdog.event",
                session_event=event,
                state=self._state_machine.state,
            )

        self._transition(event)

        if event is SessionEvent.SQUARE_OFF and self._square_off_pending:
            self._retry_square_off(force=True)

    def _transition(self, event: SessionEvent) -> None:
        """Move the state machine, tolerating an already-terminal or out-of-order state.

        An illegal transition is logged, never raised: the state label is bookkeeping, and
        losing the thread over it would cost us the deadline that actually matters.
        """
        target = _EVENT_TARGETS[event]
        current = self._state_machine.state

        if current is target:
            return
        if self._state_machine.is_terminal():
            _log.warning(
                "risk.watchdog.transition_skipped",
                session_event=event,
                state=current,
                reason="state machine is terminal",
            )
            return
        if not self._state_machine.can_transition_to(target):
            _log.warning(
                "risk.watchdog.transition_illegal",
                session_event=event,
                from_state=current,
                to_state=target,
            )
            return

        try:
            self._state_machine.transition_to(target, reason=f"watchdog: {event}")
        except StateTransitionError as exc:
            _log.error("risk.watchdog.transition_failed", session_event=event, error=str(exc))

    def _retry_square_off(self, *, force: bool = False) -> None:
        """Attempt the flatten action, honouring the retry interval."""
        if self._on_square_off is None or self._square_off_done:
            self._square_off_pending = False
            return

        now = self._clock.monotonic()
        if not force and (now - self._last_retry_at) < SQUARE_OFF_RETRY_SECONDS:
            return
        self._last_retry_at = now

        try:
            self._on_square_off()
        except Exception as exc:  # noqa: BLE001 - keep trying until we are actually flat
            self._action_failures += 1
            _log.critical(
                "risk.square_off.action_failed",
                attempt=self._action_failures,
                error=str(exc),
                error_type=type(exc).__name__,
                action=f"retrying in {SQUARE_OFF_RETRY_SECONDS}s — positions may still be open",
                exc_info=True,
            )
            return

        self._square_off_pending = False
        self._square_off_done = True
        _log.critical(
            "risk.square_off.completed",
            attempts=self._action_failures + 1,
            at_ist=now_ist(self._clock).isoformat(timespec="milliseconds"),
        )
