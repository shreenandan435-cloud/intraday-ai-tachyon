"""VWAP + OBI confluence signal generation — CLAUDE.md §3.1, §8.

A **pure function of an** :class:`~tachyon.math_engine.core.IndicatorSnapshot`. No I/O, no
clock, no broker, no state beyond counters. That is what makes the rule readable and testable:
given these numbers, this signal — every time.

The rule
--------
::

    LONG   ⟸  LTP > session VWAP   AND  LTP > EMA(n)   AND  OBI >  +threshold
    SHORT  ⟸  LTP < session VWAP   AND  LTP < EMA(n)   AND  OBI <  −threshold
    NEUTRAL otherwise

Three independent confirmations, deliberately: VWAP says where the day's volume-weighted fair
value is, the EMA says which way price is trending against it, and OBI says whether the resting
book agrees *right now*. Any two of these can be true in a chop; requiring all three is what
makes the signal rare, and rare is the point — a missed trade costs nothing (CLAUDE.md §0).

**A signal is not an authorisation.** It is a request. :class:`~tachyon.risk.engine.RiskEngine`
is the only component that may permit an order (CLAUDE.md §4), and it can and does refuse
signals produced here.

Session VWAP, not rolling VWAP
------------------------------
The anchor is :attr:`IndicatorSnapshot.session_vwap` — the exact accumulation since 09:15.
The rolling VWAP over the last 2000 ticks is a different quantity that happens to look similar,
and anchoring the day's entries to it while calling it session VWAP would be a silent,
expensive error (CLAUDE.md §3.3). After a sequence gap the session VWAP is permanently ``NaN``,
which means **no signals for the rest of the session** on that symbol. That is intended: the
alternative is trading against a VWAP we know is wrong.

NaN is load-bearing
-------------------
Every indicator returns ``NaN`` when it is not ready, never ``0.0`` (CLAUDE.md §3). Every
comparison against ``NaN`` is false, so an unready indicator produces ``NEUTRAL`` by
construction. The checks below are still written explicitly, because "no edge" and "the engine
is not ready" are different facts and the operator needs to be able to tell them apart in the
log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.math_engine.core import IndicatorSnapshot

_log = get_logger(__name__)

#: Default OBI magnitude required for a directional signal. Overridable via
#: ``strategy.obi_threshold`` in ``config/settings.yaml``.
DEFAULT_OBI_THRESHOLD: Final[float] = 0.3


class Signal(StrEnum):
    """What the strategy would like to do. Not what it is permitted to do."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"

    @property
    def is_directional(self) -> bool:
        return self is not Signal.NEUTRAL


class NoSignalReason(StrEnum):
    """Why a directional signal did not fire. Purely diagnostic, but always populated.

    A strategy that only reports "NEUTRAL" is impossible to debug in a live session: a quiet
    market and a broken indicator look identical from the outside.
    """

    NONE = "NONE"  # a directional signal did fire
    ENGINE_NOT_READY = "ENGINE_NOT_READY"
    SESSION_VWAP_UNAVAILABLE = "SESSION_VWAP_UNAVAILABLE"
    EMA_UNAVAILABLE = "EMA_UNAVAILABLE"
    NO_VWAP_CONFLUENCE = "NO_VWAP_CONFLUENCE"
    NO_TREND_CONFLUENCE = "NO_TREND_CONFLUENCE"
    OBI_BELOW_THRESHOLD = "OBI_BELOW_THRESHOLD"
    OBI_DISAGREEMENT = "OBI_DISAGREEMENT"


@dataclass(frozen=True, slots=True)
class SignalReport:
    """One evaluation: the verdict plus every number it was derived from.

    Carrying the inputs makes a post-mortem possible without replaying the session. A log line
    saying "LONG" is not evidence; a log line saying "LONG because 2501.20 > 2498.55 > 2497.10
    with OBI 0.41/0.38" is.
    """

    symbol: str
    signal: Signal
    reason: NoSignalReason
    ltp: float
    session_vwap: float
    ema: float
    ema_period: int
    obi: float
    obi_weighted: float
    obi_threshold: float
    atr_5m: float
    ts_epoch: float

    def __bool__(self) -> bool:
        return self.signal.is_directional

    @property
    def detail(self) -> str:
        """One-line human summary for the journal and the UI."""
        return (
            f"{self.symbol} {self.signal} ltp={self.ltp:.2f} vwap={self.session_vwap:.2f} "
            f"ema{self.ema_period}={self.ema:.2f} obi={self.obi:.3f}/"
            f"{self.obi_weighted:.3f} (>{self.obi_threshold})"
        )


class SignalGenerator:
    """Evaluates the VWAP + EMA + OBI confluence rule.

    Args:
        settings: resolved config, for the thresholds. Defaults to the process singleton.
        obi_threshold: overrides ``strategy.obi_threshold``.
        ema_period: overrides ``strategy.ema_period``. Must be a period the aggregator computes.
        require_obi_agreement: overrides ``strategy.require_obi_agreement``.

    Example::

        report = generator.evaluate(aggregator.snapshot())
        if report.signal is Signal.LONG:
            ...  # ask the risk engine — never act directly on this
    """

    __slots__ = ("_counts", "_ema_period", "_obi_threshold", "_require_agreement", "_settings")

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        obi_threshold: float | None = None,
        ema_period: int | None = None,
        require_obi_agreement: bool | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        strategy = self._settings.strategy

        self._obi_threshold = obi_threshold if obi_threshold is not None else strategy.obi_threshold
        self._ema_period = ema_period if ema_period is not None else strategy.ema_period
        self._require_agreement = (
            require_obi_agreement
            if require_obi_agreement is not None
            else strategy.require_obi_agreement
        )

        if not 0.0 < self._obi_threshold < 1.0:
            raise ValueError(
                f"obi_threshold={self._obi_threshold} must lie in (0, 1) — OBI is normalised "
                f"to [-1, +1], so a threshold at or above 1 can only fire on a one-sided book."
            )

        self._counts: dict[Signal, int] = dict.fromkeys(Signal, 0)

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def obi_threshold(self) -> float:
        return self._obi_threshold

    @property
    def ema_period(self) -> int:
        return self._ema_period

    @property
    def counts(self) -> dict[Signal, int]:
        """How many of each verdict has been produced. Telemetry only."""
        return dict(self._counts)

    # ── the rule ─────────────────────────────────────────────────────────────

    def evaluate(self, snapshot: IndicatorSnapshot) -> SignalReport:
        """Apply the confluence rule to one indicator snapshot. Never raises."""
        ema = self._ema_from(snapshot)
        signal, reason = self._decide(snapshot, ema)
        self._counts[signal] += 1

        report = SignalReport(
            symbol=snapshot.symbol,
            signal=signal,
            reason=reason,
            ltp=snapshot.ltp,
            session_vwap=snapshot.session_vwap,
            ema=ema,
            ema_period=self._ema_period,
            obi=snapshot.obi,
            obi_weighted=snapshot.obi_weighted,
            obi_threshold=self._obi_threshold,
            atr_5m=snapshot.atr_5m,
            ts_epoch=snapshot.ts_epoch,
        )
        if signal.is_directional:
            _log.info(
                "strategy.signal",
                symbol=snapshot.symbol,
                signal=signal,
                ltp=round(snapshot.ltp, 2),
                session_vwap=round(snapshot.session_vwap, 2),
                ema=round(ema, 2),
                ema_period=self._ema_period,
                obi=round(snapshot.obi, 3),
                obi_weighted=round(snapshot.obi_weighted, 3),
                atr_5m=round(snapshot.atr_5m, 3),
            )
        return report

    def _ema_from(self, snapshot: IndicatorSnapshot) -> float:
        """The configured EMA, or ``NaN`` if this snapshot does not carry it.

        A missing period is ``NaN`` rather than a raise: the aggregator's period set is
        configuration, and a mismatch must degrade to "no signal", never to a crash in the
        tick loop. Config validation catches the mismatch at boot; this is the runtime backstop.
        """
        try:
            index = snapshot.ema_periods.index(self._ema_period)
        except ValueError:
            return math.nan
        return snapshot.emas[index] if index < len(snapshot.emas) else math.nan

    def _decide(self, snapshot: IndicatorSnapshot, ema: float) -> tuple[Signal, NoSignalReason]:
        """The rule itself, separated so the reason coding stays legible."""
        # 1. The engine must be able to produce an order at all. `is_tradeable` covers taint
        #    from a sequence gap and a missing ATR — without ATR there is no stop, no target
        #    and no size (CLAUDE.md §6.1), so a signal would be unactionable anyway.
        if not snapshot.is_tradeable:
            return (Signal.NEUTRAL, NoSignalReason.ENGINE_NOT_READY)

        vwap = snapshot.session_vwap
        if not math.isfinite(vwap):
            # Permanent for the session once a gap has occurred (CLAUDE.md §3.3).
            return (Signal.NEUTRAL, NoSignalReason.SESSION_VWAP_UNAVAILABLE)

        if not math.isfinite(ema):
            return (Signal.NEUTRAL, NoSignalReason.EMA_UNAVAILABLE)

        ltp = snapshot.ltp
        above_vwap = ltp > vwap
        below_vwap = ltp < vwap
        if not (above_vwap or below_vwap):
            return (Signal.NEUTRAL, NoSignalReason.NO_VWAP_CONFLUENCE)

        above_ema = ltp > ema
        below_ema = ltp < ema
        if above_vwap and not above_ema:
            return (Signal.NEUTRAL, NoSignalReason.NO_TREND_CONFLUENCE)
        if below_vwap and not below_ema:
            return (Signal.NEUTRAL, NoSignalReason.NO_TREND_CONFLUENCE)

        return self._obi_verdict(snapshot, long=above_vwap)

    def _obi_verdict(
        self, snapshot: IndicatorSnapshot, *, long: bool
    ) -> tuple[Signal, NoSignalReason]:
        """The book-pressure leg, including the anti-spoofing agreement check."""
        threshold = self._obi_threshold
        obi = snapshot.obi
        weighted = snapshot.obi_weighted

        if long:
            touch_ok, ladder_ok = obi > threshold, weighted > threshold
        else:
            touch_ok, ladder_ok = obi < -threshold, weighted < -threshold

        if not touch_ok:
            return (Signal.NEUTRAL, NoSignalReason.OBI_BELOW_THRESHOLD)

        if self._require_agreement and not ladder_ok:
            # Size at the touch pulling one way while the ladder pulls the other is the classic
            # spoofing shape (CLAUDE.md §3.1). The two variants disagreeing is meaningful
            # signal — it just is not a signal to trade.
            _log.warning(
                "strategy.obi_disagreement",
                symbol=snapshot.symbol,
                direction="LONG" if long else "SHORT",
                obi=round(obi, 3),
                obi_weighted=round(weighted, 3),
                threshold=threshold,
                action="entry suppressed — touch and ladder disagree (possible spoof)",
            )
            return (Signal.NEUTRAL, NoSignalReason.OBI_DISAGREEMENT)

        return (Signal.LONG if long else Signal.SHORT, NoSignalReason.NONE)

    def reset_session(self) -> None:
        """Clear the telemetry counters for a new trading day."""
        self._counts = dict.fromkeys(Signal, 0)
