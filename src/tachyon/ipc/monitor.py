"""Feed liveness watchdog — CLAUDE.md §4 (veto step 5).

A dead market data feed is more dangerous than a bad signal. With no ticks the strategy still
holds positions, the P&L still moves, and the stop is only enforced at the broker — while our
view of the world silently freezes at whatever the last tick said. So a feed that goes quiet
for more than :data:`~tachyon.core.constants.FEED_STALE_AFTER` must block new entries
immediately and let the operator flatten.

**Staleness is measured on the receiver's own monotonic clock, at arrival.** It is tempting to
compare the tick's ``ts_epoch`` against local wall time, but that conflates two different
faults — a stalled feed and a clock skew — and wall-clock comparisons break under NTP
correction. ``time.monotonic()`` from two *different processes* is not comparable at all
(each has its own arbitrary origin), so the only sound measurement is "how long since *I* last
received anything". Wire-clock lag is still tracked, separately, as a diagnostic.

The monitor is a decision component: it holds no socket and starts no thread. Phase 6 drives
:meth:`FeedMonitor.check` from the risk watchdog thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import FEED_STALE_AFTER
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

#: CLAUDE.md §1. Sourced from the frozen constant, not a config key.
FEED_STALE_SECONDS: Final[float] = FEED_STALE_AFTER.total_seconds()


class FeedStaleException(RuntimeError):  # noqa: N818 - name fixed by the risk-engine contract
    """The market data feed has gone silent beyond the permitted window.

    Raised by :meth:`FeedMonitor.assert_fresh` on the entry path. Callers must treat this as
    "block new entries and consider flattening", never as a retryable I/O hiccup.
    """

    def __init__(self, stale_for: float, threshold: float) -> None:
        self.stale_for = stale_for
        self.threshold = threshold
        super().__init__(
            f"Market data feed stale for {stale_for:.3f}s "
            f"(threshold {threshold:.3f}s) — new entries blocked."
        )


class FeedState(StrEnum):
    """Liveness state of the feed."""

    STARTING = "STARTING"  # armed, nothing received yet
    FRESH = "FRESH"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class FeedStaleEvent:
    """Payload handed to the stale/recovery callbacks."""

    state: FeedState
    stale_for: float
    messages_seen: int
    at_ist: datetime


StaleCallback = Callable[[FeedStaleEvent], None]


@dataclass(slots=True)
class FeedMonitor:
    """Tracks time since the last received :class:`~tachyon.ipc.schemas.Tick` or
    :class:`~tachyon.ipc.schemas.Heartbeat`.

    Callbacks are **edge-triggered**: ``on_stale`` fires once on the fresh→stale transition and
    ``on_recover`` once on stale→fresh. The watchdog polls four times a second, and a callback
    that halts trading must not be invoked four times a second while the feed stays down.

    Example::

        monitor = FeedMonitor(on_stale=lambda e: risk.block_entries(e))
        ...
        monitor.record()          # on every tick / heartbeat received
        monitor.check()           # from the watchdog thread, every WATCHDOG_TICK
        monitor.assert_fresh()    # on the entry path, raises FeedStaleException
    """

    stale_after: float = FEED_STALE_SECONDS
    clock: Clock = SYSTEM_CLOCK
    on_stale: StaleCallback | None = None
    on_recover: StaleCallback | None = None

    _last_seen_mono: float = field(default=0.0, init=False)
    _last_wire_epoch: float | None = field(default=None, init=False)
    _messages: int = field(default=0, init=False)
    _state: FeedState = field(default=FeedState.STARTING, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        # Arm from construction, so a feed that never delivers a first tick still goes stale
        # on schedule instead of looking healthy forever.
        self._last_seen_mono = self.clock.monotonic()

    # ── recording ────────────────────────────────────────────────────────────

    def record(self, wire_epoch: float | None = None) -> None:
        """Note that a message arrived. Call on every tick and every heartbeat.

        Args:
            wire_epoch: the message's ``ts_epoch``, used only for the :attr:`lag_seconds`
                diagnostic. Never used for the staleness decision.

        Deliberately lock-free: this sits directly on the tick path, and the writes are single
        attribute stores which are atomic under the GIL. A torn read in :meth:`check` would at
        worst delay the verdict by one 250 ms poll, whereas taking a lock here would put
        contention on the hottest path in the system.
        """
        self._last_seen_mono = self.clock.monotonic()
        self._last_wire_epoch = wire_epoch
        self._messages += 1

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def age(self) -> float:
        """Seconds since the last received message (or since arming)."""
        return self.clock.monotonic() - self._last_seen_mono

    @property
    def is_stale(self) -> bool:
        """True if the feed has exceeded the staleness window."""
        return self.age > self.stale_after

    @property
    def state(self) -> FeedState:
        """Last state observed by :meth:`check`. Use :attr:`is_stale` for a live read."""
        return self._state

    @property
    def messages_seen(self) -> int:
        return self._messages

    @property
    def lag_seconds(self) -> float | None:
        """Wire clock minus local wall clock, or ``None`` before the first timestamped message.

        Diagnostic only — a large value means the publisher's clock disagrees with ours or the
        broker is behind, neither of which is the same fault as a stalled feed.
        """
        if self._last_wire_epoch is None:
            return None
        return self.clock.now().timestamp() - self._last_wire_epoch

    # ── decisions ────────────────────────────────────────────────────────────

    def check(self) -> FeedState:
        """Evaluate liveness and fire edge-triggered callbacks. Returns the current state.

        A callback that raises is logged and swallowed: a broken listener must never prevent
        the monitor from tracking the feed, and must never take down the watchdog thread.
        """
        age = self.age
        stale = age > self.stale_after

        with self._lock:
            previous = self._state
            current = FeedState.STALE if stale else FeedState.FRESH
            if current is previous:
                return current
            self._state = current
            event = FeedStaleEvent(
                state=current,
                stale_for=age,
                messages_seen=self._messages,
                at_ist=now_ist(self.clock),
            )
            # STARTING -> FRESH is the feed coming up for the first time, not a recovery.
            # Firing on_recover there would announce a recovery from an outage that never
            # happened — and any handler that un-blocks entries on recovery would then be
            # invoked before the feed had ever proven itself.
            recovered = previous is FeedState.STALE
            callback = self.on_stale if stale else (self.on_recover if recovered else None)

        if stale:
            _log.error(
                "feed.stale",
                stale_for=round(age, 3),
                threshold=self.stale_after,
                messages_seen=self._messages,
                action="new entries blocked",
            )
        elif recovered:
            _log.info("feed.recovered", messages_seen=self._messages)
        else:
            _log.info("feed.started", messages_seen=self._messages)

        if callback is not None:
            try:
                callback(event)
            except Exception as exc:  # noqa: BLE001 - a bad listener must not stall the watchdog
                _log.error(
                    "feed.callback_failed",
                    state=current,
                    error=str(exc),
                    exc_info=True,
                )
        return current

    def assert_fresh(self) -> None:
        """Raise :class:`FeedStaleException` if the feed is stale.

        Call this on the entry path, immediately before an order is authorised.
        """
        age = self.age
        if age > self.stale_after:
            raise FeedStaleException(stale_for=age, threshold=self.stale_after)

    def reset(self) -> None:
        """Re-arm as if just constructed. For reconnect handling and tests."""
        with self._lock:
            self._last_seen_mono = self.clock.monotonic()
            self._last_wire_epoch = None
            self._messages = 0
            self._state = FeedState.STARTING
