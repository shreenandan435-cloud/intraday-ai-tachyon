"""Pre-market gap scanner — picks the session's watchlist before the socket opens.

What it does
------------
Between the close of NSE's pre-open auction (09:08 IST) and the market open (09:15), this
module ranks the cash-equity universe by **gap percentage** — how far the auction's discovered
price sits from yesterday's close — filters out what the operator's budget cannot trade, and
returns the top ``n`` symbols as validated :class:`~tachyon.core.config.WatchlistItem` objects
ready to be written to ``config/settings.yaml``.

::

    gap_pct = (pre_open_price - previous_close) / previous_close * 100

Ranked by **absolute** gap: a −6 % gap down is the same magnitude of overnight repricing as a
+6 % gap up, and the strategy (§4.1) is symmetric — it takes shorts on the same three-way
confluence it takes longs on.

One request, not two thousand
-----------------------------
NSE publishes the entire pre-open book as a single JSON document. That is the whole reason this
source was chosen over per-symbol broker quotes: Angel One's ``getMarketData`` accepts at most
50 tokens per call at 1 call/second, so the same universe would cost ~40 sequential requests
and roughly 40 seconds of a 7-minute window — while a naive per-symbol loop would be 2,000+
requests and would get the API key throttled long before it finished.

The endpoint is **unofficial**. It is NSE's own website JSON, not a documented API: it needs a
browser-shaped ``User-Agent`` and a primed cookie, and it can change without notice. That is an
accepted, contained risk — see the failure policy below.

Failure policy — this module may only ever produce *less*
---------------------------------------------------------
A scan that fails produces **no candidates**, and the caller's contract (see
:mod:`tachyon.core.config_writer` and ``scripts/boot_tachyon.py``) is to then leave the existing
watchlist untouched. That direction is deliberate and mirrors §5.1's Sentinel asymmetry: the
scanner exists to *choose among* symbols, so its absence must not manufacture one. The dangerous
failure would be a half-successful scan writing a one-symbol watchlist, or a watchlist of
symbols whose tokens were guessed.

So every candidate must survive six independent proofs before it is selected:

1. **It has real pre-open data.** A symbol with no discovered price is dropped, never treated
   as a 0 → which would compute a −100 % gap and sort straight to the top of the ranking. This
   is a correctness rule, not a preference: it is the one data fault that actively rewards
   itself under an ``abs(gap)`` sort.
2. **It is ``EQ`` series.** ``BE``/``BZ`` are trade-to-trade — delivery-settled, with intraday
   square-off not permitted at all, so a position in one could not be flattened at 15:15 (§1.1).
   See :data:`EQUITY_SERIES`.
3. **It clears the price floor.** Below ₹100 the tick is a larger fraction of the ATR-derived
   stop and the book is thinner than the OBI kernel assumes (§3.1).
4. **It clears the liquidity floor.** ₹1 crore matched in the auction, and this is the proof
   that stops the ranking selecting for illiquidity — see :data:`MIN_PREOPEN_TURNOVER_INR`,
   which explains why a gap without depth behind it is not a signal at all.
5. **The budget can margin it.** ``pre_open_price / 5 > budget`` is rejected — see
   :data:`INTRADAY_LEVERAGE`.
6. **Its token resolves in Angel One's scrip master**, to a cash-equity row whose ``name``
   matches the symbol exactly. A token is the only instrument identity that reaches the wire
   (§2.3), and the ingestor will refuse to boot on a token it cannot verify. Resolving *from*
   the master is what makes a machine-written watchlist safe to hand to a process that
   validates tokens fatally.

Selection walks the ranked list and accepts until ``n`` symbols have passed all six, rather
than slicing the top ``n`` and then resolving them. Slicing first would silently return three
symbols whenever the fourth-ranked one failed to resolve.

Dynamic liquidity floor
-----------------------
The static ₹1 crore floor (:data:`MIN_PREOPEN_TURNOVER_INR`) is rigid in both directions: on a
thin day it rejects the entire board and the session starts with no watchlist at all, while on
an unusually liquid day it admits symbols that are the board's small change. A scanner
constructed with ``turnover_percentile`` (e.g. ``90.0`` = top decile) instead derives the floor
from the board itself — the nearest-rank percentile of published positive turnovers — so the
liquidity bar moves with market conditions. The percentile never acts alone: it is bounded below
by :data:`TURNOVER_BACKSTOP_INR`, because on a universally dead board the top decile is still
dead, and the two-share book of the :data:`MIN_PREOPEN_TURNOVER_INR` note would sail through.
When no symbol publishes a turnover the percentile is uncomputable and the scan falls back to
the static floor — this module may only ever produce *less*.

Composite ranking score
-----------------------
Selection ranks by a cross-sectional composite, not by gap alone::

    S_i = w_gap * Z(|GapPct_i|) + w_turnover * Z(log10 Turnover_i)
          + w_rvol * Z(RVOL_i)  - w_spread * SpreadBps_i

``Z`` is the population z-score across the surviving cross-section (a degenerate section —
fewer than two distinct values — yields zeros, so the term drops out rather than dividing by
zero). Three deliberate departures from the raw formula, each documented where it is
implemented: the gap enters as ``|gap|`` because §4.1 is symmetric (a −6 % gap down is the same
opportunity as a +6 % gap up, and a signed z would rank the day's biggest faller *last*);
turnover is log-scaled before standardisation because one mega-cap would otherwise dominate the
raw cross-section; and spread enters in raw basis points, already a naturally scaled penalty.
RVOL is the symbol's matched volume relative to the cross-sectional mean volume; a source that
publishes no volume drops the term rather than inventing one. On a source that publishes
neither volume nor spread — NSE's pre-open document — the score collapses to the gap-and-
turnover terms, and when turnovers tie it collapses further to the absolute-gap order the
original scanner used.

Dual-source ingestion
---------------------
NSE's unofficial pre-open document is the primary source; Angel One's REST gainers/losers
screener (:class:`AngelMoversSource`) is the secondary. :class:`DualSourcePreOpen` tries the
primary first and falls back to the secondary only when the primary fails — a fallback is an
acknowledged degradation, never a blend: mixing two boards with different liquidity semantics
into one cross-section would corrupt every z-score on it.

Intraday rotation
-----------------
The same maths rescans the live board during the session. :class:`RotationScheduler` fires on
fixed IST slots (default 09:30, 11:30, 13:30); :func:`plan_rotation` decides which watched
symbols may be swapped — only symbols that are flat, not cooling and have nothing in flight —
and pairs the weakest of those with the highest-scoring unwatched mover. The orchestrator
(:mod:`tachyon.main`) applies the plan by re-pointing subscription slots on the live socket.

What this module is not
-----------------------
It does not authorise anything. It chooses *which instruments the strategy is allowed to look
at* — §8.1's watchlist gate — and every one of them still faces the full §4 veto checklist on
every signal. A gap is not an edge; it is a reason to be watching.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final, Protocol, Self

import httpx

from tachyon.core.clock import SYSTEM_CLOCK, Clock, ist_at, now_ist
from tachyon.core.config import WatchlistItem
from tachyon.core.logger import get_logger
from tachyon.ingestion.instruments import InstrumentMaster, InstrumentRecord

_log = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Screening parameters
# ──────────────────────────────────────────────────────────────────────────────

#: SEBI's standard intraday margin for cash equity. One share blocks ``price / 5``.
#:
#: This is a *screening* multiplier and nothing more. The risk engine deliberately has no
#: notional-to-margin model (§4) because real leverage is instrument- and broker-specific, and
#: an invented multiplier would either wave through orders the broker rejects or refuse every
#: trade the system would ever take. Nothing downstream sizes a position from this number:
#: §6.3 sizes from ``per_trade_risk / R``, and the broker's own RMS reading is what the
#: ``INSUFFICIENT_MARGIN`` check consults.
INTRADAY_LEVERAGE: Final[Decimal] = Decimal("5")

#: Below this, a symbol is not worth trading on this system's geometry. The ₹0.05 tick is 0.05 %
#: of a ₹100 stock and 0.5 % of a ₹10 one, so the rounding in §6.2 starts to move the realised
#: risk/reward ratio meaningfully; the book is also thinner than the OBI kernel assumes.
PENNY_PRICE_FLOOR_INR: Final[Decimal] = Decimal("100")

#: Minimum value matched in the pre-open auction, in rupees. ₹1 crore.
#:
#: This is the filter that stops the ranking from selecting for illiquidity. ``|gap %|`` with no
#: liquidity floor does not find the day's biggest movers — it finds the day's thinnest books,
#: because a thin book is exactly where two shares can move the print 16 %. Measured on a real
#: board: the top-ranked symbol was discovered on **2 shares, ₹510 of turnover**, and outranked
#: every genuine mover on the exchange. That is not an unlucky day, it is what the sort does.
#:
#: The consequences are specific and all bad. §3.1's OBI kernel reads five levels of depth, and
#: on a book that matched two shares it is reading noise — so the confluence rule that is
#: supposed to make signals rare stops filtering anything. §6.2's bracket stop can gap straight
#: through, and §1.2 enforces the daily limit against *realised* P&L, not intended. And a
#: microcap gapping 16 % is plausibly sitting on its circuit band, where the 15:15 flatten §1.1
#: calls unconditional may simply not fill.
#:
#: NSE publishes this in the same document as the price, so the floor costs no extra request.
MIN_PREOPEN_TURNOVER_INR: Final[Decimal] = Decimal("10000000")

#: Default dynamic liquidity floor: the top decile of published pre-open turnovers.
#:
#: Expressed as the percentile *value* that bounds the top 10 % — i.e. a symbol must clear the
#: turnover that 90 % of the board sits below. See the module docstring ("Dynamic liquidity
#: floor") for why a rigid rupee floor is wrong in both directions, and why the percentile
#: never acts alone.
DEFAULT_TURNOVER_PERCENTILE: Final[float] = 90.0

#: Absolute lower bound on a percentile-derived floor, in rupees. ₹10 lakh.
#:
#: The percentile adapts the floor to the day's liquidity; the backstop stops the adaptation
#: from bottoming out on a universally dead board, where the top decile is still dead and the
#: two-share books the percentile exists to exclude would sail through. An order of magnitude
#: below :data:`MIN_PREOPEN_TURNOVER_INR` on purpose: it is a data-integrity guard against
#: noise books, not a liquidity opinion — the percentile carries that.
TURNOVER_BACKSTOP_INR: Final[Decimal] = Decimal("1000000")

#: A percentile computed from fewer published turnovers than this is noise, not a floor.
#: Below it the scan falls back to the static floor rather than trusting a quantile of a
#: handful of rows.
MIN_TURNOVER_SAMPLE: Final[int] = 10

#: The only NSE series this system may trade.
#:
#: Not a preference — a prerequisite for §1.1. ``BE`` and ``BZ`` are trade-to-trade: every
#: transaction is settled by delivery and **intraday square-off is not permitted on them at
#: all**. A position opened in a T2T scrip cannot be flattened at 15:15; it must be taken to
#: delivery. That is the one outcome this system exists to make impossible, so the series is
#: checked before anything else about the symbol is considered. ``SM``/``ST`` are SME-platform
#: scrips with their own lot sizes and liquidity profile, and ``IV`` is not a normal cash line
#: either. Only ``EQ`` is the ordinary rolling-settlement equity segment.
EQUITY_SERIES: Final[str] = "EQ"

#: How many symbols the watchlist gets. Four, against ``positions.max_concurrent = 2``.
DEFAULT_SELECTION_SIZE: Final[int] = 4

#: Cash-equity series on NSE. Anything else in the master is a derivative or a non-tradable row.
EQUITY_SERIES_SUFFIX: Final[str] = "-EQ"

_ZERO: Final[Decimal] = Decimal("0")
_HUNDRED: Final[Decimal] = Decimal("100")
_PAISE: Final[Decimal] = Decimal("0.01")
_BASIS_POINT: Final[Decimal] = Decimal("0.0001")

#: Intraday universe rescans fire on these IST slots. 09:30 lets the opening auction settle
#: for fifteen minutes first; 13:30 is the last slot that leaves a rotated symbol a meaningful
#: window before the 15:15 square-off (§1.1) — a rotation at 14:30 would subscribe to a symbol
#: the system must flatten an hour later.
DEFAULT_ROTATION_TIMES: Final[tuple[time, ...]] = (time(9, 30), time(11, 30), time(13, 30))

#: Angel One SmartAPI base URL — mirrored from ``execution.api.BASE_URL`` rather than imported
#: so the scanner (strategy layer) does not grow a dependency on the order path.
ANGEL_API_BASE_URL: Final[str] = "https://apiconnect.angelone.in"

#: Angel One's market screener: today's gainers and losers on an exchange segment. The
#: secondary/fallback universe source — see :class:`AngelMoversSource`.
ANGEL_GAINERS_LOSERS_PATH: Final[str] = "/rest/secure/angelbroking/marketData/v1/gainersLosers"


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PreOpenQuote:
    """One symbol's pre-open auction result, as published.

    Only what the screen needs. A source that cannot supply a finite, positive value for both
    prices must not emit a quote at all — see :class:`RejectReason`.

    ``turnover`` and ``series`` default to *unknown*, and unknown is rejected rather than
    waved through. That direction is the point: a source that does not publish liquidity cannot
    prove a symbol is liquid, and this module's contract (see the failure policy in the module
    docstring) is that it may only ever produce less. A default of ``Decimal(0)`` would read as
    "zero turnover" and reject too; a default of "assume liquid" would silently reinstate
    exactly the defect these fields exist to close.
    """

    symbol: str
    previous_close: Decimal
    pre_open_price: Decimal

    #: Value matched in the pre-open auction, in **rupees**. ``None`` means the source did not
    #: publish it, which is treated as failing the floor.
    turnover: Decimal | None = None

    #: NSE series — ``EQ``, ``BE``, ``SM``, ``ST``, ``BZ``, ``IV``. Empty means unknown.
    series: str = ""

    #: Matched volume in shares, when the source publishes it. Feeds the RVOL term of the
    #: composite score; ``None`` drops the term for this symbol (see :func:`rvol_values`).
    volume: Decimal | None = None

    #: Top-of-book spread in basis points, when the source publishes one. The composite score's
    #: penalty term — wide-spread books cost more to cross, so they rank down. ``None`` means
    #: no penalty: NSE's pre-open document carries no live quote, and inventing a spread would
    #: be worse than ignoring the term.
    spread_bps: Decimal | None = None


class RejectReason(StrEnum):
    """Why a symbol did not reach the watchlist. Every rejection is counted and logged."""

    #: The auction discovered no price — usually a symbol with no pre-open participation.
    NO_PRE_OPEN_PRICE = "NO_PRE_OPEN_PRICE"
    #: No usable previous close, so the gap denominator is undefined.
    NO_PREVIOUS_CLOSE = "NO_PREVIOUS_CLOSE"
    #: Not the ``EQ`` series — trade-to-trade, SME or another non-intraday line (§1.1).
    NOT_EQUITY_SERIES = "NOT_EQUITY_SERIES"
    #: Priced below :data:`PENNY_PRICE_FLOOR_INR`.
    BELOW_PRICE_FLOOR = "BELOW_PRICE_FLOOR"
    #: Matched less than :data:`MIN_PREOPEN_TURNOVER_INR` in the auction, or published nothing.
    BELOW_TURNOVER_FLOOR = "BELOW_TURNOVER_FLOOR"
    #: One share's margin exceeds the session budget.
    UNAFFORDABLE = "UNAFFORDABLE"
    #: No cash-equity row in the scrip master carries this name.
    NOT_IN_MASTER = "NOT_IN_MASTER"
    #: The master row has no usable tick size, so §6.2 could not round a price for it.
    UNKNOWN_TICK_SIZE = "UNKNOWN_TICK_SIZE"
    #: Ranked below the cut. Not a fault.
    OUTRANKED = "OUTRANKED"


@dataclass(frozen=True, slots=True)
class Rejection:
    """One screened-out symbol, kept for the boot report."""

    symbol: str
    reason: RejectReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A symbol that cleared every filter and resolved to a verified instrument."""

    symbol: str
    token: str
    previous_close: Decimal
    pre_open_price: Decimal
    gap_pct: Decimal
    margin_per_share: Decimal
    tick_size: Decimal
    lot_size: int
    #: Rupees matched in the pre-open auction. Printed at boot so the operator can see the
    #: evidence behind the gap, not just the gap.
    turnover: Decimal = _ZERO
    series: str = EQUITY_SERIES
    exchange: str = "NSE"
    #: Cross-sectional composite score that ranked this candidate (see
    #: :func:`composite_scores`). Reporting only — nothing downstream gates on it except the
    #: intraday rotation planner, which compares scores within one scan and never across scans.
    score: float = 0.0

    @property
    def direction(self) -> str:
        """``UP`` or ``DOWN``. Reporting only — the strategy decides its own side (§4.1)."""
        return "UP" if self.gap_pct >= _ZERO else "DOWN"

    def to_watchlist_item(self) -> WatchlistItem:
        """Render as the validated config model the rest of the system consumes."""
        return WatchlistItem(
            symbol=self.symbol,
            token=self.token,
            exchange=self.exchange,  # type: ignore[arg-type]
            tick_size=self.tick_size,
            lot_size=self.lot_size,
        )


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Everything one scan produced, including what it threw away and why."""

    selected: tuple[Candidate, ...]
    rejected: tuple[Rejection, ...]
    considered: int
    budget: Decimal
    #: The liquidity floor this scan applied, in rupees. Carried so the boot report can state
    #: the threshold beside the count it rejected, rather than leaving the operator to guess.
    min_turnover: Decimal = MIN_PREOPEN_TURNOVER_INR
    #: The series this scan required. Reported for the same reason.
    required_series: str = EQUITY_SERIES
    #: Where :attr:`min_turnover` came from: ``static`` (the configured rupee floor),
    #: ``percentile`` (derived from the board), ``backstop`` (percentile below the absolute
    #: backstop) or ``fallback-static`` (percentile requested but uncomputable — no usable
    #: turnover sample on the board). A dynamic floor the operator cannot see is a silent
    #: re-tuning of the strategy, so the provenance travels with the value.
    floor_source: str = "static"

    @property
    def is_usable(self) -> bool:
        """True when at least one symbol survived. See the module's failure policy."""
        return bool(self.selected)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(candidate.symbol for candidate in self.selected)

    def watchlist(self) -> tuple[WatchlistItem, ...]:
        return tuple(candidate.to_watchlist_item() for candidate in self.selected)

    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[str(rejection.reason)] = counts.get(str(rejection.reason), 0) + 1
        return counts

    @property
    def affordability_ceiling(self) -> Decimal:
        """The highest share price this budget admits. Worth printing — see §1.3's note.

        At any realistic session budget this is far above every NSE cash price, which means the
        affordability filter rejects nothing. That is not a bug, but the operator should be able
        to see it rather than assume the filter is doing work.
        """
        return (self.budget * INTRADAY_LEVERAGE).quantize(_PAISE)


# ──────────────────────────────────────────────────────────────────────────────
# Pure screening maths
# ──────────────────────────────────────────────────────────────────────────────


def gap_percent(previous_close: Decimal, pre_open_price: Decimal) -> Decimal:
    """``(pre_open - prev_close) / prev_close × 100``, to a basis point.

    Raises:
        ValueError: ``previous_close`` is not strictly positive. Callers screen for that first;
            it is an assertion, not a path — a zero denominator here would otherwise surface as
            a ``DivisionByZero`` deep inside the sort.
    """
    if previous_close <= _ZERO:
        raise ValueError(f"previous_close must be positive, got {previous_close}")
    return (((pre_open_price - previous_close) / previous_close) * _HUNDRED).quantize(_BASIS_POINT)


def margin_per_share(price: Decimal, *, leverage: Decimal = INTRADAY_LEVERAGE) -> Decimal:
    """Blocked margin for a single share at the assumed intraday leverage."""
    if leverage <= _ZERO:
        raise ValueError(f"leverage must be positive, got {leverage}")
    return (price / leverage).quantize(_PAISE)


def percentile_turnover_floor(turnovers: Iterable[Decimal], percentile: float) -> Decimal | None:
    """Nearest-rank percentile of the published positive turnovers, or ``None``.

    ``None`` means the floor is uncomputable — no positive turnover was published, or the
    sample is below :data:`MIN_TURNOVER_SAMPLE` rows, where a quantile is noise. Callers fall
    back to the static floor rather than trust it.

    The nearest-rank method returns an *observed* value, never an interpolation: a floor the
    board itself produced, and one the operator can find again in the same document.

    Raises:
        ValueError: ``percentile`` outside ``(0, 100]``.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    positive = sorted(t for t in turnovers if t > _ZERO)
    if len(positive) < MIN_TURNOVER_SAMPLE:
        return None
    rank = math.ceil(percentile / 100.0 * len(positive))
    rank = min(max(rank, 1), len(positive))
    return positive[rank - 1]


def cross_sectional_z(values: Sequence[float]) -> tuple[float, ...]:
    """Population z-scores across one cross-section. Pure.

    A degenerate section — fewer than two values, or no dispersion — yields all zeros. That is
    the important edge: a z-score with zero standard deviation is undefined, and the only safe
    reading of "every symbol has the same turnover" is "turnover distinguishes nobody", i.e.
    the term contributes nothing. Returning zeros drops the term; returning NaNs or raising
    would take the whole scan down on a perfectly ordinary board.
    """
    n = len(values)
    if n < 2:
        return tuple(0.0 for _ in values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    std = math.sqrt(variance)
    if std < 1e-12:
        return tuple(0.0 for _ in values)
    return tuple((value - mean) / std for value in values)


def rvol_values(quotes: Sequence[PreOpenQuote]) -> tuple[float, ...]:
    """Cross-sectional relative volume: ``volume_i / mean(volume)``, aligned to ``quotes``.

    Only *published* volumes count. The tempting proxy ``turnover / price`` is refused: it is
    turnover divided by price, so it would double-count the turnover term while smuggling a
    price correlation in — cheap symbols would rank as "high RVOL" on no evidence of volume
    at all. A source that publishes no volume drops the term section-wide instead.

    If fewer than two symbols carry a positive volume the whole term is zeroed — a
    cross-section of one is not relative volume, it is a number with nothing relative about it.
    """
    raw: list[float] = []
    for quote in quotes:
        if quote.volume is not None and quote.volume > _ZERO:
            raw.append(float(quote.volume))
        else:
            raw.append(0.0)
    positive = [value for value in raw if value > 0.0]
    if len(positive) < 2:
        return tuple(0.0 for _ in raw)
    mean = sum(positive) / len(positive)
    return tuple(value / mean if value > 0.0 else 0.0 for value in raw)


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """Composite-score weights — see the module docstring for the score's shape.

    All weights are non-negative and at least one must be positive. The spread weight enters
    with a negative sign in the score, so it is stored positive here: ``w_spread=0.5`` penalises
    wide books, it does not reward them.
    """

    w_gap: float = 1.0
    w_turnover: float = 1.0
    w_rvol: float = 1.0
    w_spread: float = 0.5

    def __post_init__(self) -> None:
        for name in ("w_gap", "w_turnover", "w_rvol", "w_spread"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative weight, got {value}")
        if self.w_gap == 0.0 and self.w_turnover == 0.0 and self.w_rvol == 0.0:
            # A score with only the spread penalty would rank the *least liquid* books first
            # whenever spreads are missing (all-zero penalty = tie). Refuse the shape.
            raise ValueError("at least one of w_gap, w_turnover, w_rvol must be positive")


def composite_scores(quotes: Sequence[PreOpenQuote], weights: RankingWeights) -> tuple[float, ...]:
    """The cross-sectional composite score, aligned to ``quotes``. Pure.

    ``S_i = w_gap*Z(|gap|) + w_turnover*Z(log10 turnover) + w_rvol*Z(RVOL) - w_spread*spread_bps``

    Three documented departures from the naive formula:

    * **``|gap|``** — §4.1 is symmetric: the strategy shorts gap-downs on the same confluence
      it buys gap-ups. A signed z would rank the day's biggest faller last; the magnitude is
      the information.
    * **``log10`` turnover before standardisation** — raw turnover cross-sections are dominated
      by one or two mega-caps whose z-scores pin everyone else near zero; the log compresses
      the tail so the rest of the board can compete. Survivors all cleared a positive floor,
      so the log is defined.
    * **spread in raw basis points** — already a naturally scaled penalty; z-scoring it too
      would let one pathological quote stretch the whole term.

    Missing data never penalises and never rewards: an absent turnover cannot occur here
    (screening rejects it), an absent volume zeroes the RVOL term section-wide (see
    :func:`rvol_values`), and an absent spread contributes ``0.0`` to the penalty.
    """
    if not quotes:
        return ()
    gaps = [float(abs(gap_percent(q.previous_close, q.pre_open_price))) for q in quotes]
    turnovers = [
        math.log10(float(q.turnover)) if q.turnover is not None and q.turnover > _ZERO else 0.0
        for q in quotes
    ]
    rvols = list(rvol_values(quotes))
    z_gap = cross_sectional_z(gaps)
    z_turnover = cross_sectional_z(turnovers)
    z_rvol = cross_sectional_z(rvols)

    scores: list[float] = []
    for index, quote in enumerate(quotes):
        spread = float(quote.spread_bps) if quote.spread_bps is not None else 0.0
        score = (
            weights.w_gap * z_gap[index]
            + weights.w_turnover * z_turnover[index]
            + weights.w_rvol * z_rvol[index]
            - weights.w_spread * spread
        )
        scores.append(score)
    return tuple(scores)


@dataclass(frozen=True, slots=True)
class _Ranked:
    """Internal: a quote that survived screening, with its computed gap and score."""

    quote: PreOpenQuote
    gap_pct: Decimal
    margin: Decimal
    score: float = 0.0


def _screen(
    quotes: Iterable[PreOpenQuote],
    *,
    budget: Decimal,
    price_floor: Decimal,
    leverage: Decimal,
    min_turnover: Decimal,
    required_series: str,
) -> tuple[list[_Ranked], list[Rejection]]:
    """Apply the data-validity, series, price, liquidity and affordability filters. Pure.

    Ordered so that the most categorical rejection wins: a trade-to-trade scrip is not a
    marginal call about liquidity, it is an instrument this system must never open a position
    in at all (§1.1), and the tally should say so rather than filing it under turnover. Each
    symbol carries exactly one reason — the first filter it failed.
    """
    survivors: list[_Ranked] = []
    rejected: list[Rejection] = []

    for quote in quotes:
        if quote.pre_open_price <= _ZERO:
            # Dropped rather than gapped from zero. A 0 price against any positive close is a
            # -100% gap, which under abs() ranking would beat every genuine mover on the board.
            rejected.append(
                Rejection(quote.symbol, RejectReason.NO_PRE_OPEN_PRICE, "no price discovered")
            )
            continue
        if quote.previous_close <= _ZERO:
            rejected.append(
                Rejection(quote.symbol, RejectReason.NO_PREVIOUS_CLOSE, "gap denominator is zero")
            )
            continue
        if quote.series.strip().upper() != required_series:
            # Unknown series lands here too, and must: we cannot prove an unlabelled scrip is
            # not trade-to-trade, and a T2T position cannot be flattened at 15:15 at all.
            rejected.append(
                Rejection(
                    quote.symbol,
                    RejectReason.NOT_EQUITY_SERIES,
                    f"series {quote.series or '<unknown>'!r} is not {required_series} — "
                    f"intraday square-off may not be permitted on it",
                )
            )
            continue
        if quote.pre_open_price < price_floor:
            rejected.append(
                Rejection(
                    quote.symbol,
                    RejectReason.BELOW_PRICE_FLOOR,
                    f"Rs.{quote.pre_open_price} < Rs.{price_floor}",
                )
            )
            continue
        if quote.turnover is None or quote.turnover < min_turnover:
            matched = "not published" if quote.turnover is None else f"Rs.{quote.turnover}"
            rejected.append(
                Rejection(
                    quote.symbol,
                    RejectReason.BELOW_TURNOVER_FLOOR,
                    f"pre-open matched {matched} < floor Rs.{min_turnover}; the gap is not "
                    f"evidence of a move at this depth",
                )
            )
            continue

        margin = margin_per_share(quote.pre_open_price, leverage=leverage)
        if margin > budget:
            rejected.append(
                Rejection(
                    quote.symbol,
                    RejectReason.UNAFFORDABLE,
                    f"one share blocks Rs.{margin} > budget Rs.{budget}",
                )
            )
            continue

        survivors.append(
            _Ranked(quote, gap_percent(quote.previous_close, quote.pre_open_price), margin)
        )

    return survivors, rejected


def _rank(survivors: Sequence[_Ranked]) -> list[_Ranked]:
    """Sort by composite score, descending. Ties break on symbol so a scan is reproducible.

    With the default weights on a source that publishes neither volume nor spread and a board
    whose turnovers tie, every term but ``Z(|gap|)`` drops out and this reduces exactly to the
    original absolute-gap order.
    """
    return sorted(survivors, key=lambda item: (-item.score, item.quote.symbol))


# ──────────────────────────────────────────────────────────────────────────────
# Instrument resolution
# ──────────────────────────────────────────────────────────────────────────────


class SymbolResolver(Protocol):
    """Maps a bare NSE symbol to the broker's own instrument row."""

    def resolve(self, symbol: str) -> InstrumentRecord | None: ...


class MasterSymbolResolver:
    """Resolves symbols against Angel One's scrip master, cash equity only.

    Keyed on the master's ``name`` column, which is exactly the field
    :meth:`~tachyon.ingestion.instruments.InstrumentMaster.verify_watchlist` compares against at
    ingestor boot. Keying on anything else would let this module emit a watchlist that the
    ingestor then refuses — a config the system writes for itself must pass the system's own
    verification by construction.
    """

    __slots__ = ("_index",)

    def __init__(self, records: Iterable[InstrumentRecord], *, exchange: str = "NSE") -> None:
        index: dict[str, InstrumentRecord] = {}
        for record in records:
            if record.exchange != exchange:
                continue
            if not record.trading_symbol.upper().endswith(EQUITY_SERIES_SUFFIX):
                continue
            if record.expiry:
                # A dated contract is not cash equity, whatever series it claims.
                continue
            key = record.name.strip().upper()
            if key and key not in index:
                index[key] = record
        self._index = index

    @classmethod
    def from_master(cls, master: InstrumentMaster, *, exchange: str = "NSE") -> Self:
        """Build from the cached master. Caller is responsible for its freshness."""
        return cls(master.load(exchanges=frozenset({exchange})), exchange=exchange)

    def __len__(self) -> int:
        return len(self._index)

    def resolve(self, symbol: str) -> InstrumentRecord | None:
        return self._index.get(symbol.strip().upper())


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────


class PreOpenSource(Protocol):
    """Supplies the whole pre-open board in one call."""

    async def fetch(self) -> tuple[PreOpenQuote, ...]: ...


class PreOpenFetchError(RuntimeError):
    """The pre-open board could not be retrieved or parsed."""


#: NSE's own pre-open JSON. One request, whole board.
NSE_PRE_OPEN_URL: Final[str] = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"

#: Fetched first, purely to collect the cookies the API endpoint requires.
NSE_PRIMING_URL: Final[str] = (
    "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
)

#: NSE rejects non-browser agents outright.
_NSE_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_PRIMING_URL,
}

#: Generous, but bounded. The whole board is a few megabytes and the window is short.
NSE_TIMEOUT_SECONDS: Final[float] = 20.0


def _money(value: object) -> Decimal | None:
    """Parse a published number into ``Decimal``, or ``None`` if it is not usable.

    Returns ``None`` rather than raising or defaulting to zero: a zero would sort to the top of
    an ``abs(gap)`` ranking, and every caller treats ``None`` as "drop this symbol".
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = Decimal(str(value))
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text == "-":
            return None
        try:
            parsed = Decimal(text)
        except InvalidOperation:
            return None
    else:
        return None
    return parsed if parsed.is_finite() else None


def parse_nse_pre_open(payload: object) -> tuple[PreOpenQuote, ...]:
    """Project NSE's pre-open document into quotes. Malformed rows are skipped, not fatal.

    A row we cannot read is one symbol we will not consider; a body we cannot read at all is a
    failed scan. Only the second one raises.

    Raises:
        PreOpenFetchError: the document has no ``data`` array.
    """
    if not isinstance(payload, dict):
        raise PreOpenFetchError(f"expected a JSON object, got {type(payload).__name__}")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise PreOpenFetchError("pre-open document carried no 'data' array")

    quotes: list[PreOpenQuote] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        symbol = str(metadata.get("symbol", "")).strip().upper()
        if not symbol:
            continue

        previous_close = _money(metadata.get("previousClose"))
        # `lastPrice` is the auction's discovered price; `iep` is the same number under the
        # indicative-equilibrium name and is present on some rows when lastPrice is not.
        pre_open = _money(metadata.get("lastPrice")) or _money(metadata.get("iep"))
        if pre_open is None:
            pre_open = _money(_nested(row, "detail", "preOpenMarket", "finalPrice"))
        if previous_close is None or pre_open is None:
            continue

        # Published in rupees — verified against price x finalQuantity across the whole board,
        # which agrees to a ratio of exactly 1.000. Not lakhs, not crores. A units error here
        # would silently make the floor 10^5 times too strict or too loose, so the check is
        # written down rather than assumed. `None` (absent, unparseable) fails the floor.
        turnover = _money(metadata.get("totalTurnover"))
        if turnover is None:
            quantity = _money(metadata.get("finalQuantity"))
            turnover = pre_open * quantity if quantity is not None else None

        quotes.append(
            PreOpenQuote(
                symbol=symbol,
                previous_close=previous_close,
                pre_open_price=pre_open,
                turnover=turnover,
                series=str(metadata.get("series", "")).strip().upper(),
            )
        )
    return tuple(quotes)


def _nested(row: dict[str, Any], *keys: str) -> object:
    """Walk a nested mapping, returning ``None`` the moment the path stops being one."""
    node: object = row
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


class NsePreOpenSource:
    """Fetches the entire pre-open board from NSE in a single request.

    Args:
        url: the JSON endpoint.
        priming_url: fetched first for its cookies; NSE serves the API only to a session that
            has one. A priming failure is tolerated — the API call is tried anyway, because the
            cookie requirement has come and gone over the years and hard-failing on it would
            cost a session for a step that may not have been needed.
        timeout: total budget for the pair of requests.
        client_factory: injected so tests never patch ``httpx`` globally.
    """

    __slots__ = ("_client_factory", "_priming_url", "_timeout", "_url")

    def __init__(
        self,
        *,
        url: str = NSE_PRE_OPEN_URL,
        priming_url: str = NSE_PRIMING_URL,
        timeout: float = NSE_TIMEOUT_SECONDS,
        client_factory: object | None = None,
    ) -> None:
        self._url = url
        self._priming_url = priming_url
        self._timeout = timeout
        self._client_factory = client_factory

    def _build_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            client = self._client_factory()  # type: ignore[operator]
            assert isinstance(client, httpx.AsyncClient)
            return client
        return httpx.AsyncClient(timeout=self._timeout, follow_redirects=True, headers=_NSE_HEADERS)

    async def fetch(self) -> tuple[PreOpenQuote, ...]:
        """Retrieve and parse the board.

        Raises:
            PreOpenFetchError: on any transport, HTTP or decode failure.
        """
        try:
            async with self._build_client() as client:
                try:
                    await client.get(self._priming_url)
                except httpx.HTTPError as exc:
                    _log.warning(
                        "scanner.nse_priming_failed",
                        error=str(exc),
                        note="continuing; the cookie may not be required",
                    )
                response = await client.get(self._url, headers=_NSE_HEADERS)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise PreOpenFetchError(f"NSE pre-open fetch failed: {exc}") from exc
        except ValueError as exc:
            raise PreOpenFetchError(f"NSE pre-open response was not JSON: {exc}") from exc

        quotes = parse_nse_pre_open(payload)
        _log.info("scanner.pre_open_fetched", symbols=len(quotes), url=self._url)
        return quotes


# ──────────────────────────────────────────────────────────────────────────────
# Secondary source — Angel One REST screener
# ──────────────────────────────────────────────────────────────────────────────


def _mac_address() -> str:
    """A real MAC string for the SmartAPI identity headers (no packets are sent)."""
    node = uuid.getnode()
    return ":".join(f"{(node >> shift) & 0xFF:02X}" for shift in range(40, -1, -8))


def angel_auth_headers(
    *, api_key: str, client_id: str, jwt_token: str, feed_token: str = ""
) -> dict[str, str]:
    """The header set SmartAPI's authenticated REST endpoints require on every call.

    The IP values are placeholders by design: Angel One's WAF rejects loopback addresses but
    does not verify the reported addresses are reachable, so a static LAN-shaped value is the
    same choice ``execution.api.ClientIdentity`` makes after its own probe. The JWT is a
    credential — callers must keep it out of logs.
    """
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserKey": api_key,
        "X-ClientLocalIP": "192.168.1.1",
        "X-ClientPublicIP": "192.168.1.1",
        "X-MACAddress": _mac_address(),
        "Authorization": f"Bearer {jwt_token}",
        "x-api-key": api_key,
        "x-client-code": client_id,
        "x-feed-token": feed_token,
    }


def cached_angel_headers(*, api_key: str, client_id: str) -> Callable[[], Mapping[str, str]]:
    """A ``header_provider`` resolving the JWT from the same-day token cache at fetch time.

    Resolved lazily — per fetch, not per construction — because the session cache is written
    by the feed's login, which may not have happened yet when a scanner is built. A missing
    cache raises :class:`PreOpenFetchError` from inside the provider, which the source reports
    as an ordinary fetch failure: the fallback source being unauthenticated is a degraded
    scan, not a crash.
    """

    def provider() -> Mapping[str, str]:
        from tachyon.core import token_cache  # deferred: keeps import graph light

        cached = token_cache.load_session_cache(api_key, client_id)
        if cached is None:
            raise PreOpenFetchError(
                "no cached SmartAPI session — the Angel fallback source cannot authenticate"
            )
        return angel_auth_headers(
            api_key=api_key,
            client_id=client_id,
            jwt_token=cached.jwt_token,
            feed_token=cached.feed_token,
        )

    return provider


def parse_angel_movers(payload: object) -> tuple[PreOpenQuote, ...]:
    """Project Angel One's gainers/losers screener into pre-open-shaped quotes.

    The screener publishes live session prices, not auction results: ``lastTradedPrice`` stands
    in for the pre-open price and the previous close is reconstructed from ``netChange``
    (``prev = ltp - change``), or from ``percentChange`` when that is all the row carries.
    Rows that cannot yield a positive previous close are skipped — the gap denominator is the
    one number this module never guesses.

    Series is stamped ``EQ`` by explicit assumption: the endpoint screens the NSE cash segment,
    but the document does not label series per row. The assumption is contained — the resolver
    still proves every selected symbol is a cash ``-EQ`` row in the scrip master — and it is
    the one way a fallback scan could admit a symbol the primary source would not, so it is
    written down here instead of being silent.

    Raises:
        PreOpenFetchError: the document has no ``data`` mapping — a refusal, not an empty board.
    """
    if not isinstance(payload, dict):
        raise PreOpenFetchError(f"expected a JSON object, got {type(payload).__name__}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise PreOpenFetchError("gainers/losers document carried no 'data' object")

    rows: list[object] = []
    for key in ("gainers", "losers"):
        section = data.get(key)
        if isinstance(section, list):
            rows.extend(section)

    quotes: list[PreOpenQuote] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("tradingSymbol") or "").strip().upper()
        if symbol.endswith(EQUITY_SERIES_SUFFIX):
            symbol = symbol[: -len(EQUITY_SERIES_SUFFIX)]
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)

        ltp = _money(row.get("lastTradedPrice", row.get("ltp")))
        if ltp is None or ltp <= _ZERO:
            continue
        change = _money(row.get("netChange", row.get("change")))
        if change is not None:
            previous_close = ltp - change
        else:
            pct = _money(row.get("percentChange"))
            if pct is None or pct <= -_HUNDRED:
                continue
            previous_close = ltp / (_HUNDRED + pct) * _HUNDRED
        if previous_close <= _ZERO:
            continue

        quotes.append(
            PreOpenQuote(
                symbol=symbol,
                previous_close=previous_close.quantize(_PAISE),
                pre_open_price=ltp,
                turnover=_money(row.get("turnover", row.get("totalTradedValue"))),
                series=EQUITY_SERIES,
                volume=_money(row.get("volume", row.get("totalTradedVolume"))),
            )
        )
    return tuple(quotes)


class AngelMoversSource:
    """Top gainers & losers off Angel One's REST screener — the secondary universe source.

    The endpoint is authenticated, so the source needs SmartAPI headers. Pass them static
    (``headers=``) or — the production shape — a ``header_provider`` that resolves a same-day
    JWT at fetch time (see :func:`cached_angel_headers`).

    Args:
        url: the screener endpoint.
        body: the request payload; defaults to today's NSE cash movers.
        headers: static header mapping. Mutually optional with ``header_provider`` — exactly
            one of the two must be supplied.
        header_provider: zero-argument callable returning headers at fetch time.
        timeout: request budget in seconds.
        client_factory: injected so tests never patch ``httpx`` globally.
    """

    __slots__ = ("_body", "_client_factory", "_header_provider", "_headers", "_timeout", "_url")

    def __init__(
        self,
        *,
        url: str = ANGEL_API_BASE_URL + ANGEL_GAINERS_LOSERS_PATH,
        body: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        header_provider: Callable[[], Mapping[str, str]] | None = None,
        timeout: float = NSE_TIMEOUT_SECONDS,
        client_factory: object | None = None,
    ) -> None:
        if headers is None and header_provider is None:
            raise ValueError(
                "AngelMoversSource requires headers or a header_provider — "
                "the screener is an authenticated endpoint"
            )
        self._url = url
        self._body: Mapping[str, object] = (
            dict(body) if body is not None else {"exchange": "NSE", "duration": "1"}
        )
        self._headers = headers
        self._header_provider = header_provider
        self._timeout = timeout
        self._client_factory = client_factory

    def _build_client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            client = self._client_factory()  # type: ignore[operator]
            assert isinstance(client, httpx.AsyncClient)
            return client
        return httpx.AsyncClient(timeout=self._timeout, follow_redirects=True)

    def _resolve_headers(self) -> Mapping[str, str]:
        if self._headers is not None:
            return self._headers
        assert self._header_provider is not None
        return self._header_provider()

    async def fetch(self) -> tuple[PreOpenQuote, ...]:
        """Retrieve and parse the screener.

        Raises:
            PreOpenFetchError: on any transport, HTTP, authentication or decode failure.
        """
        try:
            headers = self._resolve_headers()
        except PreOpenFetchError:
            raise
        try:
            async with self._build_client() as client:
                response = await client.post(self._url, json=dict(self._body), headers=headers)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise PreOpenFetchError(f"Angel movers fetch failed: {exc}") from exc
        except ValueError as exc:
            raise PreOpenFetchError(f"Angel movers response was not JSON: {exc}") from exc

        quotes = parse_angel_movers(payload)
        _log.info("scanner.angel_movers_fetched", symbols=len(quotes), url=self._url)
        return quotes


class DualSourcePreOpen:
    """Primary universe source with a secondary fallback.

    A fallback is an acknowledged degradation, never a blend: the two sources carry different
    liquidity semantics (auction turnover vs session turnover), and merging their rows into one
    cross-section would corrupt every z-score computed on it. So exactly one board feeds a
    scan — the primary's when it answers, the secondary's only when the primary fails — and
    :attr:`last_used` records which, so the boot report can say so.

    Both sources failing raises :class:`PreOpenFetchError` carrying both causes: the caller's
    contract is to leave the existing watchlist untouched.
    """

    __slots__ = ("_primary", "_secondary", "last_used")

    def __init__(self, primary: PreOpenSource, secondary: PreOpenSource) -> None:
        self._primary = primary
        self._secondary = secondary
        self.last_used = ""

    async def fetch(self) -> tuple[PreOpenQuote, ...]:
        try:
            quotes = await self._primary.fetch()
        except PreOpenFetchError as primary_error:
            _log.warning(
                "scanner.primary_source_failed",
                error=str(primary_error),
                action="falling back to the secondary source",
            )
            try:
                quotes = await self._secondary.fetch()
            except PreOpenFetchError as secondary_error:
                raise PreOpenFetchError(
                    f"all universe sources failed — primary: {primary_error}; "
                    f"secondary: {secondary_error}"
                ) from secondary_error
            self.last_used = "secondary"
            return quotes
        self.last_used = "primary"
        return quotes


# ──────────────────────────────────────────────────────────────────────────────
# Scanner
# ──────────────────────────────────────────────────────────────────────────────


class PreMarketScanner:
    """Screens, ranks and resolves the pre-open board into a session watchlist.

    Args:
        source: supplies the whole board in one call.
        resolver: proves each shortlisted symbol against the broker's instrument master.
        size: how many symbols to select.
        price_floor: minimum share price.
        leverage: assumed intraday margin multiplier for the affordability filter.
        min_turnover: minimum pre-open matched value, in rupees. Lowering it re-admits the
            thin books the ranking is otherwise biased towards — see
            :data:`MIN_PREOPEN_TURNOVER_INR` for what that costs. When ``turnover_percentile``
            is set this becomes the fallback floor (used only when the percentile is
            uncomputable) and the backstop's reference value.
        required_series: the NSE series a symbol must carry. Configurable so a test can drive
            it, not so a session can relax it; see :data:`EQUITY_SERIES`.
        turnover_percentile: derive the liquidity floor from the board itself — the
            nearest-rank percentile of published positive turnovers (``90.0`` = a symbol must
            clear what 90 % of the board sits below, i.e. the top decile). ``None`` keeps the
            static ``min_turnover`` floor. See the module docstring.
        turnover_backstop: absolute lower bound on a percentile-derived floor. Ignored in
            static mode.
        weights: composite ranking weights. ``None`` uses :class:`RankingWeights` defaults.
    """

    __slots__ = (
        "_leverage",
        "_min_turnover",
        "_price_floor",
        "_required_series",
        "_resolver",
        "_size",
        "_source",
        "_turnover_backstop",
        "_turnover_percentile",
        "_weights",
    )

    def __init__(
        self,
        *,
        source: PreOpenSource,
        resolver: SymbolResolver,
        size: int = DEFAULT_SELECTION_SIZE,
        price_floor: Decimal = PENNY_PRICE_FLOOR_INR,
        leverage: Decimal = INTRADAY_LEVERAGE,
        min_turnover: Decimal = MIN_PREOPEN_TURNOVER_INR,
        required_series: str = EQUITY_SERIES,
        turnover_percentile: float | None = None,
        turnover_backstop: Decimal = TURNOVER_BACKSTOP_INR,
        weights: RankingWeights | None = None,
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        if price_floor < _ZERO:
            raise ValueError(f"price_floor must not be negative, got {price_floor}")
        if leverage <= _ZERO:
            raise ValueError(f"leverage must be positive, got {leverage}")
        if min_turnover < _ZERO:
            raise ValueError(f"min_turnover must not be negative, got {min_turnover}")
        if turnover_percentile is not None and not 0.0 < turnover_percentile <= 100.0:
            raise ValueError(
                f"turnover_percentile must be in (0, 100] or None, got {turnover_percentile}"
            )
        if turnover_backstop < _ZERO:
            raise ValueError(f"turnover_backstop must not be negative, got {turnover_backstop}")
        if not required_series.strip():
            # An empty series would match nothing, not everything — every quote would be
            # rejected and the scan would silently select zero. Refuse it at construction
            # rather than let it look like a disabled filter.
            raise ValueError("required_series must not be blank")
        self._source = source
        self._resolver = resolver
        self._size = size
        self._price_floor = price_floor
        self._leverage = leverage
        self._min_turnover = min_turnover
        self._required_series = required_series.strip().upper()
        self._turnover_percentile = turnover_percentile
        self._turnover_backstop = turnover_backstop
        self._weights = weights if weights is not None else RankingWeights()

    @property
    def turnover_percentile(self) -> float | None:
        return self._turnover_percentile

    @property
    def weights(self) -> RankingWeights:
        return self._weights

    def select(self, quotes: Iterable[PreOpenQuote], *, budget: Decimal) -> ScanResult:
        """Screen, rank and resolve. Pure — no I/O, and the unit under test.

        Walks the ranked list resolving as it goes, accepting until ``size`` symbols have passed
        every filter. Slicing the top ``size`` first and resolving afterwards would silently
        return a short watchlist whenever a highly-ranked symbol failed to resolve.
        """
        materialised = list(quotes)
        floor, floor_source = self._resolve_turnover_floor(materialised)
        survivors, rejected = _screen(
            materialised,
            budget=budget,
            price_floor=self._price_floor,
            leverage=self._leverage,
            min_turnover=floor,
            required_series=self._required_series,
        )
        if survivors:
            scores = composite_scores([entry.quote for entry in survivors], self._weights)
            survivors = [
                _Ranked(entry.quote, entry.gap_pct, entry.margin, score)
                for entry, score in zip(survivors, scores, strict=True)
            ]
        ranked = _rank(survivors)

        selected: list[Candidate] = []
        for entry in ranked:
            if len(selected) >= self._size:
                rejected.append(
                    Rejection(
                        entry.quote.symbol,
                        RejectReason.OUTRANKED,
                        f"score {entry.score:.4f} (gap {entry.gap_pct}%) ranked below the "
                        f"top {self._size}",
                    )
                )
                continue

            candidate = self._resolve(entry)
            if isinstance(candidate, Rejection):
                rejected.append(candidate)
                continue
            selected.append(candidate)

        result = ScanResult(
            selected=tuple(selected),
            rejected=tuple(rejected),
            considered=len(materialised),
            budget=budget,
            min_turnover=floor,
            required_series=self._required_series,
            floor_source=floor_source,
        )
        _log.info(
            "scanner.selected",
            symbols=list(result.symbols),
            gaps=[str(candidate.gap_pct) for candidate in result.selected],
            scores=[round(candidate.score, 4) for candidate in result.selected],
            turnovers=[str(candidate.turnover) for candidate in result.selected],
            considered=result.considered,
            rejected=result.rejection_counts(),
            budget_inr=str(budget),
            affordability_ceiling_inr=str(result.affordability_ceiling),
            turnover_floor_inr=str(floor),
            turnover_floor_source=floor_source,
            required_series=self._required_series,
        )
        if len(selected) < self._size:
            _log.warning(
                "scanner.short_selection",
                wanted=self._size,
                got=len(selected),
                note="the watchlist will carry fewer symbols than configured",
            )
        return result

    def _resolve_turnover_floor(self, quotes: Sequence[PreOpenQuote]) -> tuple[Decimal, str]:
        """The liquidity floor this scan applies, and where it came from.

        Static mode (``turnover_percentile is None``) returns the configured rupee floor
        unchanged. Dynamic mode derives the nearest-rank percentile of published positive
        turnovers, bounded below by the backstop; when the percentile is uncomputable (no
        usable sample) it falls back to the static floor — a scan may only ever produce less,
        and a floor it cannot derive must not silently become zero.
        """
        if self._turnover_percentile is None:
            return self._min_turnover, "static"
        derived = percentile_turnover_floor(
            (quote.turnover for quote in quotes if quote.turnover is not None),
            self._turnover_percentile,
        )
        if derived is None:
            _log.warning(
                "scanner.turnover_floor_fallback",
                percentile=self._turnover_percentile,
                fallback_inr=str(self._min_turnover),
                note="too few published turnovers to derive a percentile floor",
            )
            return self._min_turnover, "fallback-static"
        if derived < self._turnover_backstop:
            return self._turnover_backstop, "backstop"
        return derived, "percentile"

    def _resolve(self, entry: _Ranked) -> Candidate | Rejection:
        """Prove one shortlisted symbol against the master, or explain why it cannot be."""
        symbol = entry.quote.symbol
        record = self._resolver.resolve(symbol)
        if record is None:
            return Rejection(
                symbol,
                RejectReason.NOT_IN_MASTER,
                "no NSE cash-equity row carries this name; the ingestor would refuse the token",
            )

        tick_size = Decimal(str(record.tick_size)).quantize(Decimal("0.0001")).normalize()
        if tick_size <= _ZERO:
            # §6.2 rounds every transmitted price to the tick. Without one we cannot construct
            # an order the broker will accept, and guessing ₹0.05 is how a silent rejection
            # loop starts.
            return Rejection(
                symbol,
                RejectReason.UNKNOWN_TICK_SIZE,
                f"master reports tick_size {record.tick_size!r}",
            )

        return Candidate(
            symbol=symbol,
            token=record.token,
            previous_close=entry.quote.previous_close,
            pre_open_price=entry.quote.pre_open_price,
            gap_pct=entry.gap_pct,
            margin_per_share=entry.margin,
            tick_size=tick_size,
            # Cash equity trades in single shares; the master's lotsize column is 1 for every
            # -EQ row and 0 only when the column is absent.
            lot_size=max(record.lot_size, 1),
            # Screened as non-None above; the fallback keeps the type honest without inventing
            # liquidity a rejected symbol never had.
            turnover=entry.quote.turnover if entry.quote.turnover is not None else _ZERO,
            series=entry.quote.series,
            score=entry.score,
        )

    async def scan(self, *, budget: Decimal) -> ScanResult:
        """Fetch the board and select from it.

        Raises:
            PreOpenFetchError: the board could not be retrieved. Callers must leave the
                existing watchlist untouched rather than writing a partial one.
        """
        if budget <= _ZERO:
            raise ValueError(f"budget must be positive to screen for affordability, got {budget}")
        quotes = await self._source.fetch()
        return self.select(quotes, budget=budget)


# ──────────────────────────────────────────────────────────────────────────────
# Intraday rotation
# ──────────────────────────────────────────────────────────────────────────────


class IntradayScanner:
    """A live rescan for intraday rotation.

    The same pipeline as :class:`PreMarketScanner` — screen, rank, resolve — pointed at a live
    movers source instead of the pre-open document, and constructed with the dynamic liquidity
    floor and composite ranking by default: a rescan at 11:30 is ranking *session* turnover,
    and a rigid rupee floor calibrated for the auction would be wrong in both directions on it.

    Args mirror :class:`PreMarketScanner`; ``turnover_percentile`` defaults to the top decile
    rather than ``None``.
    """

    __slots__ = ("_scanner",)

    def __init__(
        self,
        *,
        source: PreOpenSource,
        resolver: SymbolResolver,
        size: int = DEFAULT_SELECTION_SIZE,
        price_floor: Decimal = PENNY_PRICE_FLOOR_INR,
        leverage: Decimal = INTRADAY_LEVERAGE,
        min_turnover: Decimal = MIN_PREOPEN_TURNOVER_INR,
        turnover_percentile: float | None = DEFAULT_TURNOVER_PERCENTILE,
        turnover_backstop: Decimal = TURNOVER_BACKSTOP_INR,
        weights: RankingWeights | None = None,
    ) -> None:
        self._scanner = PreMarketScanner(
            source=source,
            resolver=resolver,
            size=size,
            price_floor=price_floor,
            leverage=leverage,
            min_turnover=min_turnover,
            turnover_percentile=turnover_percentile,
            turnover_backstop=turnover_backstop,
            weights=weights,
        )

    @property
    def scanner(self) -> PreMarketScanner:
        return self._scanner

    async def scan(self, *, budget: Decimal) -> ScanResult:
        """Fetch the live board and select from it. Same failure contract as the pre-open scan:
        a failed scan raises, and the caller keeps the current watchlist."""
        return await self._scanner.scan(budget=budget)


class RotationScheduler:
    """Fires on fixed IST slots — default 09:30, 11:30, 13:30 (:data:`DEFAULT_ROTATION_TIMES`).

    The scheduler is *armed* at construction (or via :meth:`arm`): slots that already passed
    before the arm never fire, so a process booting at 12:00 waits for 13:30 instead of
    immediately catching up on 09:30 and 11:30. Slots missed *after* the arm — the loop was
    blocked through a slot — fire once, catching up on the latest missed slot only; firing
    every missed slot in a burst would rotate the watchlist three times in one second.

    All datetimes are timezone-aware IST; naive inputs raise rather than silently comparing
    across timezones.
    """

    __slots__ = ("_armed_at", "_clock", "_last_fired", "_times")

    def __init__(
        self,
        times: Sequence[time] = DEFAULT_ROTATION_TIMES,
        clock: Clock = SYSTEM_CLOCK,
        *,
        arm_now: bool = True,
    ) -> None:
        if not times:
            raise ValueError("at least one rotation time is required")
        self._times = tuple(sorted(times))
        self._clock = clock
        self._last_fired: datetime | None = None
        self._armed_at: datetime | None = now_ist(clock) if arm_now else None

    @property
    def times(self) -> tuple[time, ...]:
        return self._times

    @property
    def last_fired(self) -> datetime | None:
        return self._last_fired

    def arm(self, at: datetime | None = None) -> None:
        """(Re-)arm: slots at or before ``at`` will not fire."""
        self._armed_at = at if at is not None else now_ist(self._clock)

    def next_fire(self, after: datetime | None = None) -> datetime:
        """The earliest slot strictly after ``after`` — today's, or tomorrow's first."""
        reference = after if after is not None else now_ist(self._clock)
        for slot_time in self._times:
            candidate = ist_at(reference.date(), slot_time)
            if candidate > reference:
                return candidate
        return ist_at(reference.date() + timedelta(days=1), self._times[0])

    def advance(self, now: datetime | None = None) -> datetime | None:
        """Consume the latest due slot, if any. Returns the slot fired, else ``None``.

        A slot is due when it has passed, lies after the arm time, and has not already fired.
        Idempotent within a slot: a second call before the next slot returns ``None``.
        """
        moment = now if now is not None else now_ist(self._clock)
        due: datetime | None = None
        for slot_time in self._times:
            candidate = ist_at(moment.date(), slot_time)
            if candidate > moment:
                continue
            if self._armed_at is not None and candidate <= self._armed_at:
                continue
            if self._last_fired is not None and candidate <= self._last_fired:
                continue
            due = candidate
        if due is not None:
            self._last_fired = due
        return due


@dataclass(frozen=True, slots=True)
class RotationSwap:
    """One substitution in a rotation plan: drop ``out_symbol``, watch ``in_symbol``."""

    out_symbol: str
    out_score: float
    in_symbol: str
    in_score: float


@dataclass(frozen=True, slots=True)
class RotationPlan:
    """The swaps one rescan produced, in the order they should be applied."""

    swaps: tuple[RotationSwap, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.swaps


def plan_rotation(
    *,
    watched_scores: Mapping[str, float],
    rotatable: Set[str],
    candidates: Sequence[tuple[str, float]],
    min_score: float = 0.0,
    max_swaps: int = 1,
) -> RotationPlan:
    """Pair the weakest rotatable watched symbols with the strongest unwatched movers. Pure.

    Args:
        watched_scores: current watchlist with each symbol's score *in the new scan's
            cross-section*. A watched symbol absent from the new scan is no longer a mover —
            the caller scores it ``-inf`` so it is first in line to be rotated out.
        rotatable: the subset of watched symbols eligible to drop — flat, not cooling, nothing
            in flight. The caller proves eligibility; this function only honours it. A symbol
            with an open position is never rotated out, whatever its score: the subscription
            carries the tick stream the exit depends on.
        candidates: ``(symbol, score)`` movers from the rescan, in any order.
        min_score: absolute floor for an incoming symbol — a weak mover does not displace a
            watched symbol even when it technically outranks it.
        max_swaps: most swaps per cycle. Churn is a cost: every rotation restarts a symbol's
            VWAP and sequence state, so one deliberate swap beats three marginal ones.

    A candidate must *strictly* beat the weakest eligible watched symbol — a swap between
    equals is churn, not rotation. Deterministic throughout: candidates best-first with symbol
    tie-break, watched weakest-first with symbol tie-break.
    """
    if max_swaps < 1:
        raise ValueError(f"max_swaps must be at least 1, got {max_swaps}")

    pool = sorted(
        (
            (symbol, score)
            for symbol, score in watched_scores.items()
            if symbol in rotatable
        ),
        key=lambda pair: (pair[1], pair[0]),
    )
    watched = set(watched_scores)
    ordered = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))

    swaps: list[RotationSwap] = []
    for symbol, score in ordered:
        if len(swaps) >= max_swaps:
            break
        if symbol in watched or score < min_score:
            continue
        if not pool:
            break
        out_symbol, out_score = pool[0]
        if score <= out_score:
            # Candidates are sorted descending and the pool's weakest is unchanged by a swap
            # that did not happen — nothing later can clear this bar either.
            break
        pool.pop(0)
        watched.add(symbol)
        swaps.append(RotationSwap(out_symbol, out_score, symbol, score))
    return RotationPlan(tuple(swaps))
