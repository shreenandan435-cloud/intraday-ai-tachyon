"""Strategy and orchestration — CLAUDE.md §3.1, §4, §6.

Produces ``Signal`` *intents* only. A strategy never authorises an order and never decides that
one is safe; it hands intents to the Risk Engine, which alone permits (CLAUDE.md §4).

The rule is a three-way confluence — price above/below **session** VWAP, agreeing with the EMA
trend filter, with Order Book Imbalance beyond a threshold in the same direction. Requiring all
three is what makes the signal rare, and rare is the point: a missed trade costs nothing.

Averaging down, martingale and pyramiding into a loser are forbidden (CLAUDE.md §8.1), and are
structurally impossible here — the gate's ``POSITION_ALREADY_OPEN`` veto and the re-entry
cooldown between them mean there is no code path that adds to an existing position.

Modules:
  signals.py   SignalGenerator — the confluence rule, a pure function of an IndicatorSnapshot
  cooldown.py  ReentryManager — the 30-minute post-exit window that prevents thrashing
  brain.py     StrategyBrain — Process B's async orchestrator; routes, never judges
  scanner.py   PreMarketScanner — chooses the session's watchlist before the socket opens

``scanner`` is deliberately **not** re-exported here. It runs once at boot, before the Brain
exists, and pulling it into this package's import graph would put ``httpx`` and the scrip-master
decoder on the critical path of every process that imports ``tachyon.strategy``. Import it by
module: ``from tachyon.strategy.scanner import PreMarketScanner``.
"""

from __future__ import annotations

from tachyon.strategy.brain import BrainStats, StrategyBrain
from tachyon.strategy.cooldown import (
    REENTRY_COOLDOWN_MINUTES,
    CooldownTooShortError,
    CooldownVerdict,
    ExitRecord,
    ReentryManager,
)
from tachyon.strategy.signals import (
    DEFAULT_OBI_THRESHOLD,
    NoSignalReason,
    Signal,
    SignalGenerator,
    SignalReport,
)

__all__ = [
    "DEFAULT_OBI_THRESHOLD",
    "REENTRY_COOLDOWN_MINUTES",
    "BrainStats",
    "CooldownTooShortError",
    "CooldownVerdict",
    "ExitRecord",
    "NoSignalReason",
    "ReentryManager",
    "Signal",
    "SignalGenerator",
    "SignalReport",
    "StrategyBrain",
]
