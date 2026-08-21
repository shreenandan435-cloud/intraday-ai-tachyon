"""The Brain's state-spine publisher — CLAUDE.md §2.1, §7.3.

Publishes ``STATE.``, ``PNL.``, ``FILL.`` and ``RISK.`` on ``tcp://127.0.0.1:5556`` so the UI
can render the session without holding a reference to a single trading object. That separation
is the whole reason this module exists: an in-process dashboard reading ``PnLTracker`` directly
would be simpler and would also mean a UI bug can reach the risk state, and a UI that hangs can
hold a lock the strategy loop needs.

Two publishing cadences, for two different kinds of fact:

* **State and P&L are *states*.** Published on change and on a slow keepalive, each frame
  carrying the complete view. The UI conflates them (§2.1), so it may miss frames — a frame
  carrying only a delta would leave the display permanently wrong the first time one dropped.
* **Fills and risk events are *events*.** Published once, when they happen, and never
  conflated. A dropped tick costs a repaint; a dropped fill is a trade nobody sees.

Publishing never blocks. The socket is ``SNDHWM``-bounded with ``LINGER = 0``, so a UI that
stops reading is dropped by ZeroMQ rather than back-pressuring the strategy loop, and every
call here is fire-and-forget. A telemetry failure is logged and swallowed: the operator losing
their dashboard is bad, and the Brain stopping because of it would be worse.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.ipc.publisher import Publisher
from tachyon.ipc.schemas import (
    PNL_TOPIC_BYTES,
    STATE_TOPIC_BYTES,
    TOPIC_FILL,
    TOPIC_RISK,
    FillUpdate,
    PnLUpdate,
    RiskEvent,
    StateUpdate,
    encode,
)

_log = get_logger(__name__)

#: Republish the full state and P&L at least this often even when nothing changes, so a UI
#: that connects mid-session is not looking at empty panels until the next trade.
KEEPALIVE_SECONDS: Final[float] = 1.0


class StatePublisher:
    """Fire-and-forget publisher for the state spine.

    Args:
        settings: resolved config. Supplies ``zmq_state_endpoint``.
        publisher: injected for tests; constructed and bound when omitted.

    Example::

        telemetry = StatePublisher(settings=settings)
        telemetry.publish_state(state_update)
        telemetry.publish_risk("VETO", "RELIANCE", "FEED_STALE", "no data for 2.4s")
    """

    __slots__ = (
        "_clock",
        "_disabled",
        "_owns_publisher",
        "_publisher",
        "_settings",
        "errors",
        "published",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        publisher: Publisher | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._owns_publisher = publisher is None
        self._publisher = publisher
        self._disabled = False
        self.published = 0
        self.errors = 0

    @property
    def is_bound(self) -> bool:
        return self._publisher is not None

    def close(self) -> None:
        if self._owns_publisher and self._publisher is not None:
            self._publisher.close()
            self._publisher = None

    # ── publishing ───────────────────────────────────────────────────────────

    def _bind(self) -> Publisher | None:
        """Bind the PUB socket on first use, or ``None`` if it cannot be bound.

        Deferred rather than done in ``__init__`` for one blunt reason: **constructing an
        object must not claim an OS resource.** A ``StrategyBrain`` that binds a port the
        moment it exists cannot be built twice in one process — not in a test, not in a
        supervisor that constructs before it decides to start — and the failure shows up as a
        hang at teardown rather than an error at the call site.

        A bind failure disables telemetry permanently and is logged at ``CRITICAL``. It never
        propagates: losing the dashboard is bad, and the Brain stopping because the dashboard
        could not start would be very much worse.
        """
        if self._publisher is not None or self._disabled:
            return self._publisher
        try:
            self._publisher = Publisher(
                self._settings.zmq_state_endpoint,
                role="brain",
                settings=self._settings,
                clock=self._clock,
            )
        except Exception as exc:  # noqa: BLE001 - a bound port must not stop the Brain
            self._disabled = True
            _log.critical(
                "brain.telemetry_bind_failed",
                endpoint=self._settings.zmq_state_endpoint,
                error=str(exc),
                error_type=type(exc).__name__,
                impact="the UI will show nothing; TRADING IS UNAFFECTED",
            )
            return None
        return self._publisher

    def _send(self, topic: bytes, payload: bytes) -> None:
        """One fire-and-forget send. Never raises."""
        publisher = self._bind()
        if publisher is None:
            self.errors += 1
            return
        try:
            publisher.publish_raw(topic, payload)
            self.published += 1
        except Exception as exc:  # noqa: BLE001 - telemetry must never stop the Brain
            self.errors += 1
            _log.error(
                "brain.telemetry_publish_failed",
                topic=topic.decode(errors="replace"),
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def publish_state(self, update: StateUpdate) -> None:
        self._send(STATE_TOPIC_BYTES, encode(update))

    def publish_pnl(self, update: PnLUpdate) -> None:
        self._send(PNL_TOPIC_BYTES, encode(update))

    def publish_fill(self, update: FillUpdate) -> None:
        """One order-status transition. Never conflated — see the module docstring."""
        self._send(f"{TOPIC_FILL}{update.symbol}".encode(), encode(update))

    def publish_risk(
        self,
        kind: str,
        symbol: str,
        reason: str,
        detail: str = "",
        severity: str = "INFO",
    ) -> None:
        """A veto, a kill switch, a square-off — anything the operator must see."""
        event = RiskEvent(
            kind=kind,
            symbol=symbol,
            reason=reason,
            detail=detail,
            ts_epoch=self._clock.now().timestamp(),
            severity=severity,
        )
        self._send(f"{TOPIC_RISK}{kind}".encode(), encode(event))


def money(value: Decimal) -> str:
    """Render a ``Decimal`` for the wire.

    Money crosses as a **string**. ``Decimal`` is the type at the risk boundary (CLAUDE.md §8),
    and a float round-trip would reintroduce exactly the representation error the rest of the
    system is careful to avoid — on the number the operator watches most closely.
    """
    return str(value)
