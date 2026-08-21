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

What this module is not
-----------------------
It does not authorise anything. It chooses *which instruments the strategy is allowed to look
at* — §8.1's watchlist gate — and every one of them still faces the full §4 veto checklist on
every signal. A gap is not an edge; it is a reason to be watching.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final, Protocol, Self

import httpx

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


@dataclass(frozen=True, slots=True)
class _Ranked:
    """Internal: a quote that survived screening, with its computed gap."""

    quote: PreOpenQuote
    gap_pct: Decimal
    margin: Decimal


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
    """Sort by absolute gap, descending. Ties break on symbol so a scan is reproducible."""
    return sorted(survivors, key=lambda item: (-abs(item.gap_pct), item.quote.symbol))


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
            :data:`MIN_PREOPEN_TURNOVER_INR` for what that costs.
        required_series: the NSE series a symbol must carry. Configurable so a test can drive
            it, not so a session can relax it; see :data:`EQUITY_SERIES`.
    """

    __slots__ = (
        "_leverage",
        "_min_turnover",
        "_price_floor",
        "_required_series",
        "_resolver",
        "_size",
        "_source",
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
    ) -> None:
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        if price_floor < _ZERO:
            raise ValueError(f"price_floor must not be negative, got {price_floor}")
        if leverage <= _ZERO:
            raise ValueError(f"leverage must be positive, got {leverage}")
        if min_turnover < _ZERO:
            raise ValueError(f"min_turnover must not be negative, got {min_turnover}")
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

    def select(self, quotes: Iterable[PreOpenQuote], *, budget: Decimal) -> ScanResult:
        """Screen, rank and resolve. Pure — no I/O, and the unit under test.

        Walks the ranked list resolving as it goes, accepting until ``size`` symbols have passed
        every filter. Slicing the top ``size`` first and resolving afterwards would silently
        return a short watchlist whenever a highly-ranked symbol failed to resolve.
        """
        materialised = list(quotes)
        survivors, rejected = _screen(
            materialised,
            budget=budget,
            price_floor=self._price_floor,
            leverage=self._leverage,
            min_turnover=self._min_turnover,
            required_series=self._required_series,
        )
        ranked = _rank(survivors)

        selected: list[Candidate] = []
        for entry in ranked:
            if len(selected) >= self._size:
                rejected.append(
                    Rejection(
                        entry.quote.symbol,
                        RejectReason.OUTRANKED,
                        f"gap {entry.gap_pct}% ranked below the top {self._size}",
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
            min_turnover=self._min_turnover,
            required_series=self._required_series,
        )
        _log.info(
            "scanner.selected",
            symbols=list(result.symbols),
            gaps=[str(candidate.gap_pct) for candidate in result.selected],
            turnovers=[str(candidate.turnover) for candidate in result.selected],
            considered=result.considered,
            rejected=result.rejection_counts(),
            budget_inr=str(budget),
            affordability_ceiling_inr=str(result.affordability_ceiling),
            turnover_floor_inr=str(self._min_turnover),
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
        )

    async def scan(self, *, budget: Decimal) -> ScanResult:
        """Fetch the board and select from it.

        Raises:
            PreOpenFetchError: the board could not be retrieved. Callers must leave the
                existing watchlist alone rather than writing a partial one.
        """
        if budget <= _ZERO:
            raise ValueError(f"budget must be positive to screen for affordability, got {budget}")
        quotes = await self._source.fetch()
        return self.select(quotes, budget=budget)
