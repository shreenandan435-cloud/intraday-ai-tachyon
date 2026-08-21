"""AI Sentinel — async Gemini advisory layer, CLAUDE.md §5.

**THE SENTINEL CAN ONLY EVER MAKE THE SYSTEM MORE CONSERVATIVE.**
It may veto or downsize. It may never create a signal, upsize a position, widen a stop, or
unlock the loss limit. Any path where an LLM response increases risk is a critical defect.

It never sits in the order path. The Risk Engine reads
:class:`~tachyon.sentinel.state.MacroState` — a lock-guarded read of two in-memory fields —
while a background daemon does the talking. If Gemini hangs, 503s, returns malformed JSON, or
was never configured, trading continues under the last known good report, and every failure
path resolves to ``NEUTRAL``. **Never to ``RISK_ON``.**

Two safety properties are worth stating plainly:

* ``MacroState.is_trading_allowed`` is a **latch** and ``size_multiplier`` a **ratchet**. A
  RISK_OFF at 10:00 followed by a RISK_ON at 10:15 does not unblock trading — that would be
  the model increasing risk. Only a session reset clears them.
* ``size_multiplier`` is **derived in our code** from the regime and confidence. The model is
  never asked for it and never trusted with it.

Modules:
  state.py       MacroState — the entire surface through which an LLM reaches real money
  api.py         GeminiClient — httpx.AsyncClient, hard deadline, session budget, journalling
  classifier.py  RegimeClassifier + NewsClassifier — hardcoded prompts, defensive parsing
  service.py     SentinelDaemon — the background loop; failing open is correct here, only here
"""

from __future__ import annotations

from tachyon.sentinel.api import (
    GeminiBudgetExhaustedError,
    GeminiClient,
    GeminiError,
    GeminiNotConfiguredError,
    GeminiRejectedError,
    GeminiReply,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from tachyon.sentinel.classifier import (
    NEWS_SYSTEM_PROMPT,
    REGIME_SYSTEM_PROMPT,
    NewsClassifier,
    NewsRisk,
    RegimeClassifier,
    RegimeReport,
)
from tachyon.sentinel.service import SentinelDaemon, SentinelStats
from tachyon.sentinel.state import MacroSnapshot, MacroState, Regime

__all__ = [
    "NEWS_SYSTEM_PROMPT",
    "REGIME_SYSTEM_PROMPT",
    "GeminiBudgetExhaustedError",
    "GeminiClient",
    "GeminiError",
    "GeminiNotConfiguredError",
    "GeminiRejectedError",
    "GeminiReply",
    "GeminiTimeoutError",
    "GeminiUnavailableError",
    "MacroSnapshot",
    "MacroState",
    "NewsClassifier",
    "NewsRisk",
    "Regime",
    "RegimeClassifier",
    "RegimeReport",
    "SentinelDaemon",
    "SentinelStats",
]
