"""Hard Risk Engine — CLAUDE.md §1, §4.

THE ONLY COMPONENT PERMITTED TO AUTHORISE AN ORDER. Execution code that builds an order
without a passing :class:`~tachyon.risk.engine.RiskDecision` is a critical defect.

**Everything here fails safe.** A check that raises produces a veto, not an exception. An
unreadable lock file counts as locked. An undefined P&L counts as breached. If we cannot
prove it is safe to open a position, it is not safe to open a position.

Two rules admit no exception, no flag, no config key:

* **15:15 IST** — cancel all, exit all, latch terminal. Driven by a non-daemon thread counting
  down on the monotonic clock, so it fires even if the market data feed is dead and cannot be
  postponed by a clock adjustment.
* **₹500 daily loss** — latching kill switch, persisted to ``data/journal/daily_lock.txt`` so
  a restart boots read-only.

Modules:
  engine.py    RiskEngine — the eleven-step veto gate
  tracker.py   PnLTracker (latching kill switch) and PositionRegistry (exposure + cooldowns)
  watchdog.py  SquareOffWatchdog — the immortal 15:15 thread, with retry until actually flat
"""

from __future__ import annotations

from tachyon.risk.engine import RiskDecision, RiskEngine, VetoReason
from tachyon.risk.tracker import PnLSnapshot, PnLTracker, PositionRegistry
from tachyon.risk.watchdog import SquareOffWatchdog

__all__ = [
    "PnLSnapshot",
    "PnLTracker",
    "PositionRegistry",
    "RiskDecision",
    "RiskEngine",
    "SquareOffWatchdog",
    "VetoReason",
]
