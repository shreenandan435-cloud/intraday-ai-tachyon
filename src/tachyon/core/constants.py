"""Immutable risk and system constants — CLAUDE.md §1.

These values are the reason this system is allowed to trade unattended. They are:

* **Not** readable from ``config/settings.yaml``.
* **Not** readable from ``.env``.
* **Not** overridable by CLI flag, environment variable, or monkeypatch.
* Frozen at the module level — rebinding ``constants.DAILY_LOSS_LIMIT_INR`` raises
  :class:`FrozenConstantError` at runtime (see :class:`_FrozenModule` below).

``config.py`` additionally rejects any config file that so much as *mentions* one of these
names, so a typo'd YAML key fails loudly at boot rather than silently doing nothing.

Changing a value here requires an explicit operator instruction and a matching test update
(CLAUDE.md §8). Never infer one. Never "tune" one.
"""

from __future__ import annotations

import sys
from datetime import time, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Final, NoReturn


class FrozenConstantError(RuntimeError):
    """Raised on any attempt to mutate a hard constant at runtime."""


# ──────────────────────────────────────────────────────────────────────────────
# Identity & paths
# ──────────────────────────────────────────────────────────────────────────────

#: Repository root. ``<root>/src/tachyon/core/constants.py`` -> parents[3].
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
JOURNAL_DIR: Final[Path] = DATA_DIR / "journal"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "config"

#: Settings file consumed by :mod:`tachyon.core.config`.
SETTINGS_YAML: Final[Path] = CONFIG_DIR / "settings.yaml"

#: Latching daily kill-switch marker (CLAUDE.md §1.2). Its presence *with today's IST date*
#: forces the system to boot into :data:`~tachyon.core.state.TradingState.LOCKED`.
DAILY_LOCK_FILE: Final[Path] = JOURNAL_DIR / "daily_lock.txt"


# ──────────────────────────────────────────────────────────────────────────────
# Time — all wall-clock decisions are made in IST, never in host local time
# ──────────────────────────────────────────────────────────────────────────────

#: IANA zone. Requires the ``tzdata`` package on Windows — it is a hard dependency.
IST_ZONE_NAME: Final[str] = "Asia/Kolkata"

MARKET_OPEN_IST: Final[time] = time(9, 15)
MARKET_CLOSE_IST: Final[time] = time(15, 30)

#: No entries during the opening auction aftermath — the first 5 minutes are noise.
FIRST_ENTRY_IST: Final[time] = time(9, 20)

#: Entry gate closes. Nothing new may be opened after this (CLAUDE.md §1).
NO_NEW_ENTRIES_IST: Final[time] = time(15, 0)

#: THE hard deadline. Cancel all, exit all, latch terminal state. Driven by a monotonic
#: watchdog so it fires even if the market data feed is dead (CLAUDE.md §1.1).
AUTO_SQUAREOFF_IST: Final[time] = time(15, 15)

#: Watchdog poll interval. Small enough that square-off lands within a second of 15:15:00.
WATCHDOG_TICK: Final[timedelta] = timedelta(milliseconds=250)

#: Beyond this much tick silence the feed is STALE and entries are blocked (CLAUDE.md §4).
FEED_STALE_AFTER: Final[timedelta] = timedelta(seconds=2)

#: Cooldown after a stop-out before the same symbol may be re-entered (CLAUDE.md §8.1).
REENTRY_COOLDOWN: Final[timedelta] = timedelta(minutes=30)


# ──────────────────────────────────────────────────────────────────────────────
# Money — Decimal only. `float` is banned at the risk boundary (CLAUDE.md §8).
# ──────────────────────────────────────────────────────────────────────────────

#: Realised + unrealised + estimated charges. On breach: latching kill switch (CLAUDE.md §1.2).
DAILY_LOSS_LIMIT_INR: Final[Decimal] = Decimal("500")

#: Maximum rupees at risk on a single position. 100/500 => no single loser can consume more
#: than 20% of the day's budget. Sizing derives quantity from this, never the reverse (§6.3).
PER_TRADE_RISK_INR: Final[Decimal] = Decimal("100")

#: Once tripped, the loss limit stays tripped for the remainder of the session.
LOSS_LIMIT_IS_LATCHING: Final[bool] = True


# ──────────────────────────────────────────────────────────────────────────────
# Order geometry — CLAUDE.md §6.1. R = SL_ATR_MULTIPLIER x ATR(ATR_PERIOD, 5m)
# ──────────────────────────────────────────────────────────────────────────────

SL_ATR_MULTIPLIER: Final[Decimal] = Decimal("1.5")
T1_RR: Final[Decimal] = Decimal("1.5")
T2_RR: Final[Decimal] = Decimal("2.5")

ATR_PERIOD: Final[int] = 14
CANDLE_INTERVAL_MINUTES: Final[int] = 5

#: Fraction of quantity assigned to the T1 leg; the remainder runs to T2 (CLAUDE.md §6.1).
T1_QTY_FRACTION: Final[Decimal] = Decimal("0.60")

#: Quantity is always floored. "Just one more share" is how accounts die.
ALLOW_SIZE_ROUNDING_UP: Final[bool] = False


# ──────────────────────────────────────────────────────────────────────────────
# Trading mode — CLAUDE.md §9. The default is, and always will be, PAPER.
# ──────────────────────────────────────────────────────────────────────────────


class TradingMode(StrEnum):
    """Execution mode. LIVE additionally requires interactive confirmation at boot."""

    PAPER = "PAPER"
    LIVE = "LIVE"


#: Fail-safe default. An unset, blank, or unrecognised ``TRADING_MODE`` resolves to PAPER.
#: There is no code path that defaults to LIVE.
TRADING_MODE: Final[TradingMode] = TradingMode.PAPER
DEFAULT_TRADING_MODE: Final[TradingMode] = TradingMode.PAPER


# ──────────────────────────────────────────────────────────────────────────────
# Enforcement
# ──────────────────────────────────────────────────────────────────────────────

#: Names that :mod:`tachyon.core.config` must reject if they appear in any config source.
#: A config file mentioning one of these is a defect, not a preference — fail at boot.
PROTECTED_CONSTANT_NAMES: Final[frozenset[str]] = frozenset(
    {
        "AUTO_SQUAREOFF_IST",
        "NO_NEW_ENTRIES_IST",
        "FIRST_ENTRY_IST",
        "MARKET_OPEN_IST",
        "MARKET_CLOSE_IST",
        "DAILY_LOSS_LIMIT_INR",
        "PER_TRADE_RISK_INR",
        "LOSS_LIMIT_IS_LATCHING",
        "SL_ATR_MULTIPLIER",
        "T1_RR",
        "T2_RR",
        "T1_QTY_FRACTION",
        "ATR_PERIOD",
        "ALLOW_SIZE_ROUNDING_UP",
        "DAILY_LOCK_FILE",
        "WATCHDOG_TICK",
        "REENTRY_COOLDOWN",
        "FEED_STALE_AFTER",
    }
)


class _FrozenModule(ModuleType):
    """Module type that refuses attribute assignment and deletion.

    Installed over this module below. This turns "these constants are immutable" from a
    comment into an enforced runtime property: a stray ``constants.DAILY_LOSS_LIMIT_INR = 5000``
    anywhere in the process — including a test fixture or a debugging session — raises instead
    of quietly widening the risk budget.
    """

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise FrozenConstantError(
            f"{__name__}.{name} is a hard constant (CLAUDE.md §1) and cannot be reassigned "
            f"at runtime. Edit the source and update the corresponding test."
        )

    def __delattr__(self, name: str) -> NoReturn:
        raise FrozenConstantError(
            f"{__name__}.{name} is a hard constant (CLAUDE.md §1) and cannot be deleted."
        )


sys.modules[__name__].__class__ = _FrozenModule
