"""Order execution — Angel One SmartAPI Robo (Bracket) Orders, CLAUDE.md §6.

Geometry, with A = ATR(14) on 5m candles and R = 1.5 × A:
    stoploss  = 1.0 × R          (i.e. 1.5 × ATR)
    Target 1  = 1.5 × R          (1 : 1.5 R/R)   → leg A, 60% of qty
    Target 2  = 2.5 × R          (1 : 2.5 R/R)   → leg B, remainder
A Robo Order carries one target, so a position is split into two Robo legs at entry.
On leg A fill, leg B's stop is trailed to breakeven + 1 tick.

Trailing is BROKER-NATIVE (``trailingStopLoss``, LTP-jump). Never a client-side loop — if our
process dies the trail must still be live at the exchange. All prices Decimal-quantised to the
instrument tick size before transmission.

**Nothing here may place an order on its own authority.** The Risk Engine (Phase 6) is the only
component that authorises one, and :meth:`~tachyon.execution.executor.RoboExecutor.open_position`
both requires a passing :class:`~tachyon.risk.engine.RiskDecision` and re-runs the gate itself
immediately before transmitting.

Modules:
  api.py             SmartApiClient — the only module that talks to the broker; token-bucket
                     rate limits, TOTP login, and the never-retry-an-unknown-outcome rule
  builder.py         OrderBuilder — tick quantisation, R-multiple geometry, floor sizing,
                     the 60/40 two-leg split. Pure arithmetic, no I/O
  executor.py        RoboExecutor — payload construction, stop management, square-off
  reconciliation.py  StateReconciler — boot-time broker-vs-local comparison; locks on mismatch
  journal.py         OrderJournal — append-only audit record, written BEFORE acting
"""

from __future__ import annotations

from tachyon.execution.api import (
    BrokerOrder,
    BrokerPosition,
    BrokerSession,
    PaperModeError,
    SmartApiAuthError,
    SmartApiClient,
    SmartApiError,
    TokenBucket,
    UnknownOrderOutcomeError,
)
from tachyon.execution.builder import (
    BracketPlan,
    Leg,
    LegPlan,
    OrderBuilder,
    OrderGeometry,
    OrderRejected,
    OrderTagSequencer,
    Side,
    SizingResult,
    round_to_tick,
)
from tachyon.execution.executor import (
    ExecutionReport,
    FlattenReport,
    LegResult,
    NotFlatError,
    OffsetSanityError,
    OpenBracket,
    RoboExecutor,
    StopWidenedError,
)
from tachyon.execution.journal import OrderJournal
from tachyon.execution.reconciliation import (
    ReconcileOutcome,
    ReconciliationReport,
    StateReconciler,
)

__all__ = [
    "BracketPlan",
    "BrokerOrder",
    "BrokerPosition",
    "BrokerSession",
    "ExecutionReport",
    "FlattenReport",
    "Leg",
    "LegPlan",
    "LegResult",
    "NotFlatError",
    "OffsetSanityError",
    "OpenBracket",
    "OrderBuilder",
    "OrderGeometry",
    "OrderJournal",
    "OrderRejected",
    "OrderTagSequencer",
    "PaperModeError",
    "ReconcileOutcome",
    "ReconciliationReport",
    "RoboExecutor",
    "Side",
    "SizingResult",
    "SmartApiAuthError",
    "SmartApiClient",
    "SmartApiError",
    "StateReconciler",
    "StopWidenedError",
    "TokenBucket",
    "UnknownOrderOutcomeError",
    "round_to_tick",
]
