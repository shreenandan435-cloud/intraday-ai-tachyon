"""The Sentinel background daemon — CLAUDE.md §5.

An async task that wakes on a timer, asks Gemini what it thinks, and writes the answer into
:class:`~tachyon.sentinel.state.MacroState`. Nothing else in the system waits on it.

**It must never sit in the order path.** The Risk Engine reads ``MacroState`` — a lock-guarded
read of two fields — and never calls this module. If the Sentinel hangs, is rate-limited, or
was never configured at all, trading continues under the last known good report (or under the
permissive default). The daemon is the only place in this system where a component failing
open is correct, and it is correct because the component is advisory.

Cadence
-------
Job 1 (regime) runs immediately at startup and then every ``interval``. Running at startup
matters: a process that boots at 09:10 should classify before 09:20, not fifteen minutes into
the session. CLAUDE.md §5 describes a ~08:45 pre-market run; the immediate-then-periodic
schedule subsumes it and also covers a mid-session restart, which a fixed 08:45 job would not.

Job 2 (per-symbol news) runs on its own, slower cadence and only for watchlist symbols.

Failure posture
---------------
The loop catches every ``Exception`` and continues. A dead Sentinel that keeps its last verdict
is strictly better than one that stops updating silently — and far better than one that takes
the Brain's event loop down with it. ``asyncio.CancelledError`` is re-raised, because that is
shutdown asking politely.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.persistence.journal import JsonlJournal
from tachyon.sentinel.api import GeminiBudgetExhaustedError, GeminiClient, GeminiNotConfiguredError
from tachyon.sentinel.classifier import NewsClassifier, RegimeClassifier
from tachyon.sentinel.state import MacroState

_log = get_logger(__name__)

#: Supplies the text handed to the model. Async so a real implementation can fetch quotes or
#: headlines; the default returns a static placeholder, which is honest about being one.
BriefProvider = Callable[[], Awaitable[str]]

#: Supplies headlines for one symbol.
HeadlineProvider = Callable[[str], Awaitable[str]]

#: Placeholder brief. Deliberately says so — a Sentinel classifying a hardcoded string must
#: never look like one classifying live data (CLAUDE.md §7.2: truthfulness beats aesthetics).
PLACEHOLDER_BRIEF: Final[str] = (
    "NO LIVE MARKET DATA WIRED. No overnight index closes, India VIX, DXY, crude, FII/DII "
    "flows or event calendar are available for this session."
)


async def _default_brief() -> str:
    return PLACEHOLDER_BRIEF


async def _default_headlines(symbol: str) -> str:  # noqa: ARG001 - fixed provider signature
    return ""


@dataclass(slots=True)
class SentinelStats:
    """Counters for observability. Never used for a trading decision."""

    regime_runs: int = 0
    regime_degraded: int = 0
    news_runs: int = 0
    symbols_blacklisted: int = 0
    loop_errors: int = 0
    disabled_reason: str = ""


class SentinelDaemon:
    """Periodically classifies the macro regime and per-symbol news risk.

    Args:
        state: the shared object the Risk Engine reads.
        client: Gemini client. Omitted, one is built from config.
        interval_minutes: regime cadence. Defaults to ``sentinel.news_scan_interval_minutes``.
        brief_provider: async callable returning the macro brief.
        headline_provider: async callable returning headlines for one symbol.

    Example::

        daemon = SentinelDaemon(state=macro_state, settings=settings)
        task = asyncio.create_task(daemon.run())
        ...
        await daemon.stop()
    """

    __slots__ = (
        "_brief",
        "_client",
        "_clock",
        "_disabled",
        "_headlines",
        "_interval",
        "_journal",
        "_news",
        "_news_interval",
        "_owns_client",
        "_regime",
        "_settings",
        "_state",
        "_stopping",
        "stats",
    )

    def __init__(
        self,
        *,
        state: MacroState,
        client: GeminiClient | None = None,
        settings: Settings | None = None,
        journal: JsonlJournal | None = None,
        interval_minutes: float | None = None,
        brief_provider: BriefProvider = _default_brief,
        headline_provider: HeadlineProvider = _default_headlines,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._state = state
        self._clock = clock
        self._journal = (
            journal if journal is not None else JsonlJournal(prefix="sentinel", clock=clock)
        )

        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else GeminiClient(settings=self._settings, journal=self._journal, clock=clock)
        )
        self._regime = RegimeClassifier(self._client, settings=self._settings)
        self._news = NewsClassifier(self._client, settings=self._settings)

        sentinel = self._settings.sentinel
        minutes = (
            interval_minutes
            if interval_minutes is not None
            else float(sentinel.news_scan_interval_minutes)
        )
        self._interval = max(60.0, minutes * 60.0)
        self._news_interval = max(60.0, float(sentinel.news_scan_interval_minutes) * 60.0)

        self._brief = brief_provider
        self._headlines = headline_provider
        self._stopping = asyncio.Event()
        self._disabled = False
        self.stats = SentinelStats()

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def state(self) -> MacroState:
        return self._state

    @property
    def is_enabled(self) -> bool:
        """False once the Sentinel has switched itself off. Trading is unaffected either way."""
        return not self._disabled and self._settings.sentinel_enabled

    @property
    def interval_seconds(self) -> float:
        return self._interval

    # ── one cycle ────────────────────────────────────────────────────────────

    async def classify_once(self) -> None:
        """Run one regime classification and apply it. Never raises."""
        brief = await self._safe_brief()
        report = await self._regime.classify(brief)
        self.stats.regime_runs += 1

        if report.degraded:
            self.stats.regime_degraded += 1
            self._state.record_failure(report.error or "degraded classification")
        else:
            self._state.apply(report.regime, report.confidence, report.reason)

        self._journal.decision(
            "regime_applied",
            regime=report.regime.value,
            confidence=report.confidence,
            reason=report.reason,
            degraded=report.degraded,
            error=report.error,
            trading_allowed=self._state.is_trading_allowed,
            size_multiplier=str(self._state.size_multiplier),
        )

    async def scan_news_once(self) -> None:
        """Score every watchlist symbol and blacklist the hazardous ones. Never raises."""
        threshold = self._settings.sentinel.news_blacklist_score
        for item in self._settings.watchlist:
            if self._stopping.is_set() or self._disabled:
                return
            headlines = await self._safe_headlines(item.symbol)
            if not headlines:
                continue

            risk = await self._news.score(item.symbol, headlines)
            self.stats.news_runs += 1
            if risk.degraded:
                continue
            if risk.exceeds(threshold):
                self.stats.symbols_blacklisted += 1
                self._state.blacklist(item.symbol, reason=risk.headline or risk.reason)
                self._journal.decision(
                    "news_blacklist",
                    symbol=item.symbol,
                    score=risk.score,
                    headline=risk.headline,
                    reason=risk.reason,
                )

    async def _safe_brief(self) -> str:
        try:
            return await self._brief()
        except Exception as exc:  # noqa: BLE001 - a bad provider must not stop the daemon
            _log.error("sentinel.brief_provider_failed", error=str(exc), exc_info=True)
            return ""

    async def _safe_headlines(self, symbol: str) -> str:
        try:
            return await self._headlines(symbol)
        except Exception as exc:  # noqa: BLE001 - a bad provider must not stop the daemon
            _log.error(
                "sentinel.headline_provider_failed", symbol=symbol, error=str(exc), exc_info=True
            )
            return ""

    # ── the loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Classify immediately, then on the interval, until :meth:`stop`.

        The first run is immediate rather than after one interval: a process booting at 09:10
        must have a verdict before the 09:20 entry gate opens, and a mid-session restart must
        not spend fifteen minutes with no macro view.
        """
        if not self.is_enabled:
            _log.warning(
                "sentinel.disabled",
                reason="sentinel_enabled=false in config",
                impact="no macro veto; trading proceeds unrestricted by the Sentinel",
            )
            return
        if not self._client.is_configured:
            self._disable("GEMINI_API_KEY not set")
            return

        _log.info(
            "sentinel.started",
            model=self._client.model,
            interval_seconds=self._interval,
            news_interval_seconds=self._news_interval,
            symbols=len(self._settings.watchlist),
        )

        next_news = 0.0
        while not self._stopping.is_set():
            try:
                await self.classify_once()
                now = self._clock.monotonic()
                if now >= next_news:
                    await self.scan_news_once()
                    next_news = now + self._news_interval
            except asyncio.CancelledError:
                raise
            except GeminiNotConfiguredError as exc:
                self._disable(str(exc))
                return
            except GeminiBudgetExhaustedError as exc:
                # Out of quota disables the Sentinel, never trading (CLAUDE.md §5).
                self._disable(str(exc))
                return
            except Exception as exc:  # noqa: BLE001 - the advisor must not take down the Brain
                self.stats.loop_errors += 1
                _log.error(
                    "sentinel.cycle_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    action="absorbed; the last known good report stands",
                    exc_info=True,
                )

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)

        _log.info(
            "sentinel.stopped",
            regime_runs=self.stats.regime_runs,
            degraded=self.stats.regime_degraded,
            blacklisted=self.stats.symbols_blacklisted,
            final_regime=self._state.regime,
            trading_allowed=self._state.is_trading_allowed,
        )

    def _disable(self, reason: str) -> None:
        """Switch the Sentinel off for the session. **Never touches the trading flags.**

        A Sentinel that has run out of quota, lost its key, or was never configured leaves
        ``MacroState`` exactly as it found it. Whatever a previous successful classification
        restricted stays restricted; nothing new is restricted, and nothing is released.
        """
        self._disabled = True
        self.stats.disabled_reason = reason
        _log.warning(
            "sentinel.disabled",
            reason=reason,
            at_ist=now_ist(self._clock).isoformat(timespec="seconds"),
            trading_allowed=self._state.is_trading_allowed,
            impact="no further macro updates; trading is UNAFFECTED",
        )
        self._journal.decision("sentinel_disabled", reason=reason)

    async def stop(self) -> None:
        """Ask the loop to exit and release the client if we own it."""
        self._stopping.set()
        if self._owns_client:
            await self._client.aclose()

    def reset_session(self) -> None:
        """Clear the macro latch and the request budget for a new trading day."""
        self._state.reset_session()
        self._client.reset_session()
        self._disabled = False
        self.stats = SentinelStats()
