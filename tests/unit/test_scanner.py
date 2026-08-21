"""Pre-market gap scanner — CLAUDE.md §2.3, §6.3, §8.1.

Every test here runs on static, hand-built data. Nothing in this file opens a socket: the
network-facing part of the scanner is one method (:meth:`NsePreOpenSource.fetch`) and it is
exercised through an injected client factory serving a fixture body, never against NSE.

The tests worth reading twice are the ones about *rejection*. A scanner's job is to throw away
1,996 symbols and keep four, so almost every way it can be wrong is a way of keeping the wrong
one — and the sharpest of those is a missing pre-open price silently becoming a -100 % gap,
which under an absolute-value ranking beats every genuine mover on the board.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx
import pytest

from tachyon.ingestion.instruments import InstrumentRecord
from tachyon.strategy.scanner import (
    DEFAULT_SELECTION_SIZE,
    EQUITY_SERIES,
    INTRADAY_LEVERAGE,
    MIN_PREOPEN_TURNOVER_INR,
    PENNY_PRICE_FLOOR_INR,
    Candidate,
    MasterSymbolResolver,
    NsePreOpenSource,
    PreMarketScanner,
    PreOpenFetchError,
    PreOpenQuote,
    RejectReason,
    gap_percent,
    margin_per_share,
    parse_nse_pre_open,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — static data only
# ──────────────────────────────────────────────────────────────────────────────


#: A turnover comfortably clear of the ₹1 crore floor, so tests about *other* filters are not
#: silently also testing liquidity. Tests that care pass ``turnover=`` explicitly.
LIQUID = Decimal("50000000")  # ₹5 crore


def _quote(
    symbol: str,
    previous_close: str,
    pre_open: str,
    *,
    turnover: Decimal | None = LIQUID,
    series: str = EQUITY_SERIES,
) -> PreOpenQuote:
    return PreOpenQuote(
        symbol, Decimal(previous_close), Decimal(pre_open), turnover=turnover, series=series
    )


def _record(
    name: str,
    token: str,
    *,
    trading_symbol: str | None = None,
    exchange: str = "NSE",
    expiry: str = "",
    lot_size: int = 1,
    tick_size: float = 0.05,
) -> InstrumentRecord:
    return InstrumentRecord(
        token=token,
        trading_symbol=trading_symbol if trading_symbol is not None else f"{name}-EQ",
        name=name,
        exchange=exchange,
        instrument_type="",
        expiry=expiry,
        lot_size=lot_size,
        tick_size=tick_size,
    )


class _StaticResolver:
    """A resolver backed by a dict, so selection can be tested without a scrip master."""

    def __init__(self, records: dict[str, InstrumentRecord]) -> None:
        self._records = records

    def resolve(self, symbol: str) -> InstrumentRecord | None:
        return self._records.get(symbol.strip().upper())


class _StaticSource:
    """A pre-open source that returns a fixed board, or raises."""

    def __init__(
        self, quotes: tuple[PreOpenQuote, ...] = (), error: Exception | None = None
    ) -> None:
        self._quotes = quotes
        self._error = error
        self.calls = 0

    async def fetch(self) -> tuple[PreOpenQuote, ...]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._quotes


#: Four symbols, gaps of +5 %, -8 %, +1 %, -0.5 %. Ranked by |gap|: B, A, C, D.
_BOARD: tuple[PreOpenQuote, ...] = (
    _quote("AAA", "1000", "1050"),  # +5.00 %
    _quote("BBB", "500", "460"),  # -8.00 %
    _quote("CCC", "200", "202"),  # +1.00 %
    _quote("DDD", "400", "398"),  # -0.50 %
)

_MASTER: dict[str, InstrumentRecord] = {
    "AAA": _record("AAA", "111", tick_size=0.10),
    "BBB": _record("BBB", "222", tick_size=0.05),
    "CCC": _record("CCC", "333", tick_size=0.05),
    "DDD": _record("DDD", "444", tick_size=0.01),
}


def _scanner(
    *,
    quotes: tuple[PreOpenQuote, ...] = _BOARD,
    master: dict[str, InstrumentRecord] | None = None,
    size: int = DEFAULT_SELECTION_SIZE,
    **kwargs: Any,
) -> PreMarketScanner:
    return PreMarketScanner(
        source=_StaticSource(quotes),
        resolver=_StaticResolver(_MASTER if master is None else master),
        size=size,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────


class TestGapMath:
    """The one formula the whole module is built on."""

    def test_a_gap_up_is_positive(self) -> None:
        assert gap_percent(Decimal("1000"), Decimal("1050")) == Decimal("5.0000")

    def test_a_gap_down_is_negative(self) -> None:
        assert gap_percent(Decimal("500"), Decimal("460")) == Decimal("-8.0000")

    def test_no_move_is_zero(self) -> None:
        assert gap_percent(Decimal("742.35"), Decimal("742.35")) == Decimal("0.0000")

    def test_it_is_quantised_to_a_basis_point(self) -> None:
        """Not for display — for the sort. Two gaps that differ in the ninth decimal are the
        same gap, and letting them differ makes the tie-break on symbol unreachable."""
        assert gap_percent(Decimal("3"), Decimal("4")).as_tuple().exponent == -4

    @pytest.mark.parametrize("previous_close", ["0", "-1"])
    def test_a_non_positive_close_is_refused(self, previous_close: str) -> None:
        """The screen drops these first; this is the assertion behind that, so a future caller
        that forgets gets a named error rather than a DivisionByZero inside `sorted`."""
        with pytest.raises(ValueError, match="previous_close must be positive"):
            gap_percent(Decimal(previous_close), Decimal("100"))


class TestMarginMath:
    def test_one_share_blocks_a_fifth_at_5x(self) -> None:
        assert margin_per_share(Decimal("1000")) == Decimal("200.00")

    def test_it_quantises_to_the_paisa(self) -> None:
        assert margin_per_share(Decimal("101")) == Decimal("20.20")

    def test_the_leverage_is_injectable(self) -> None:
        assert margin_per_share(Decimal("1000"), leverage=Decimal("1")) == Decimal("1000.00")

    def test_zero_leverage_is_refused(self) -> None:
        with pytest.raises(ValueError, match="leverage must be positive"):
            margin_per_share(Decimal("100"), leverage=Decimal("0"))


class TestScreening:
    """The filters, and specifically what they refuse to let through."""

    def test_a_symbol_with_no_pre_open_price_is_dropped_not_gapped_from_zero(self) -> None:
        """The sharpest failure mode in the module.

        A pre-open price of 0 against a previous close of 1000 computes a -100 % gap. Under the
        abs() ranking that is the single largest number on the board, so the four symbols the
        session trades would be exactly the four with NO pre-open data. Dropping the row is not
        tidiness; it is the difference between trading movers and trading holes in the feed.
        """
        board = (*_BOARD, _quote("ZERO", "1000", "0"))
        result = _scanner(quotes=board, master={**_MASTER, "ZERO": _record("ZERO", "999")}).select(
            board, budget=Decimal("50000")
        )

        assert "ZERO" not in result.symbols
        assert result.symbols[0] == "BBB", "the real -8% mover must still rank first"
        assert any(
            r.symbol == "ZERO" and r.reason is RejectReason.NO_PRE_OPEN_PRICE
            for r in result.rejected
        )

    def test_a_symbol_with_no_previous_close_is_dropped(self) -> None:
        board = (_quote("NEW", "0", "500"), *_BOARD)
        result = _scanner(quotes=board).select(board, budget=Decimal("50000"))

        assert "NEW" not in result.symbols
        assert any(
            r.symbol == "NEW" and r.reason is RejectReason.NO_PREVIOUS_CLOSE
            for r in result.rejected
        )

    def test_a_penny_stock_is_dropped_however_large_its_gap(self) -> None:
        """A 40 % gap on a Rs.7 stock is the largest number on the board and the worst trade
        on it — the tick is 0.7 % of the price and the book is thinner than OBI assumes."""
        penny = _quote("PENNY", "5", "7")  # +40 %
        board = (penny, *_BOARD)
        result = _scanner(
            quotes=board, master={**_MASTER, "PENNY": _record("PENNY", "555")}
        ).select(board, budget=Decimal("50000"))

        assert "PENNY" not in result.symbols
        assert result.symbols[0] == "BBB"
        assert any(
            r.symbol == "PENNY" and r.reason is RejectReason.BELOW_PRICE_FLOOR
            for r in result.rejected
        )

    def test_the_floor_is_inclusive_at_exactly_one_hundred(self) -> None:
        """`< floor` rejects, so a stock priced at exactly the floor is kept."""
        at_floor = _quote("EDGE", "90", str(PENNY_PRICE_FLOOR_INR))
        result = _scanner(
            quotes=(at_floor,), master={"EDGE": _record("EDGE", "666")}, size=1
        ).select((at_floor,), budget=Decimal("50000"))

        assert result.symbols == ("EDGE",)

    def test_a_share_the_budget_cannot_margin_is_dropped(self) -> None:
        """At 5x, a Rs.1050 share blocks Rs.210. A Rs.200 budget cannot take one."""
        result = _scanner().select(_BOARD, budget=Decimal("200"))

        assert "AAA" not in result.symbols
        rejection = next(r for r in result.rejected if r.symbol == "AAA")
        assert rejection.reason is RejectReason.UNAFFORDABLE
        assert "210.00" in rejection.detail

    def test_the_affordability_boundary_is_exact(self) -> None:
        """Margin equal to the budget is affordable; a paisa more is not."""
        quote = _quote("EDGE", "1000", "1000")
        exact = margin_per_share(Decimal("1000"))  # 200.00

        kept = _scanner(quotes=(quote,), master={"EDGE": _record("EDGE", "1")}, size=1).select(
            (quote,), budget=exact
        )
        dropped = _scanner(quotes=(quote,), master={"EDGE": _record("EDGE", "1")}, size=1).select(
            (quote,), budget=exact - Decimal("0.01")
        )

        assert kept.symbols == ("EDGE",)
        assert dropped.symbols == ()

    def test_a_custom_leverage_moves_the_affordability_line(self) -> None:
        result = _scanner(leverage=Decimal("1")).select(_BOARD, budget=Decimal("500"))

        assert "AAA" not in result.symbols  # 1050 > 500 unlevered
        assert "BBB" in result.symbols  # 460 <= 500


class TestLiquidityFloor:
    """The filter that stops the ranking selecting for illiquidity.

    ``|gap %|`` with no depth requirement does not find the biggest movers, it finds the
    thinnest books — a thin book is exactly where a couple of shares move the print double
    digits. These tests use the real numbers off a live NSE board, because the synthetic case
    understates how extreme it gets.
    """

    def test_a_sixteen_percent_gap_on_two_shares_is_rejected(self) -> None:
        """LOYALTEX, observed 2026-08-12: +16.10 % — the largest gap on the entire exchange —
        discovered on a match of 2 shares, Rs.510 of turnover. Without this floor it ranks
        first and becomes the session's headline position."""
        loyaltex = _quote("LOYALTEX", "219.64", "255.00", turnover=Decimal("510"))
        board = (loyaltex, *_BOARD)

        result = _scanner(
            quotes=board, master={**_MASTER, "LOYALTEX": _record("LOYALTEX", "10590")}
        ).select(board, budget=Decimal("50000"))

        assert "LOYALTEX" not in result.symbols
        rejection = next(r for r in result.rejected if r.symbol == "LOYALTEX")
        assert rejection.reason is RejectReason.BELOW_TURNOVER_FLOOR
        assert "510" in rejection.detail

    def test_a_genuinely_traded_gap_passes_cleanly(self) -> None:
        """SHILCTECH, same board: -13.31 % on Rs.1.74 crore of matched value. That is a real
        overnight repricing, and it is exactly what the scanner should be selecting."""
        shilctech = _quote("SHILCTECH", "4581.70", "3972.00", turnover=Decimal("17437080"))

        result = _scanner(
            quotes=(shilctech,), master={"SHILCTECH": _record("SHILCTECH", "759847")}, size=1
        ).select((shilctech,), budget=Decimal("50000"))

        assert result.symbols == ("SHILCTECH",)
        assert result.selected[0].turnover == Decimal("17437080")
        assert result.selected[0].gap_pct == Decimal("-13.3073")

    def test_the_whole_observed_board_reduces_to_the_one_liquid_name(self) -> None:
        """All four symbols the unfiltered scanner picked on 2026-08-12, with their real
        turnovers. Three were discovered on less than Rs.4 lakh between them."""
        observed = (
            _quote("LOYALTEX", "219.64", "255.00", turnover=Decimal("510")),
            _quote("BEEKAY", "397.90", "454.00", turnover=Decimal("350034")),
            _quote("SHILCTECH", "4581.70", "3972.00", turnover=Decimal("17437080")),
            _quote("KRISHIVAL", "402.15", "449.90", turnover=Decimal("226749")),
        )
        master = {
            "LOYALTEX": _record("LOYALTEX", "10590", tick_size=0.01),
            "BEEKAY": _record("BEEKAY", "762573"),
            "SHILCTECH": _record("SHILCTECH", "759847", tick_size=0.10),
            "KRISHIVAL": _record("KRISHIVAL", "756782"),
        }

        result = _scanner(quotes=observed, master=master).select(observed, budget=Decimal("50000"))

        assert result.symbols == ("SHILCTECH",)
        assert result.rejection_counts() == {"BELOW_TURNOVER_FLOOR": 3}

    def test_the_floor_boundary_is_inclusive(self) -> None:
        """Exactly Rs.1 crore passes; a rupee less does not."""
        at_floor = _quote("EDGE", "100", "150", turnover=MIN_PREOPEN_TURNOVER_INR)
        below = _quote("EDGE", "100", "150", turnover=MIN_PREOPEN_TURNOVER_INR - Decimal("1"))
        master = {"EDGE": _record("EDGE", "1")}

        kept = _scanner(quotes=(at_floor,), master=master, size=1).select(
            (at_floor,), budget=Decimal("50000")
        )
        dropped = _scanner(quotes=(below,), master=master, size=1).select(
            (below,), budget=Decimal("50000")
        )

        assert kept.symbols == ("EDGE",)
        assert dropped.symbols == ()

    def test_unpublished_turnover_is_rejected_not_waved_through(self) -> None:
        """A source that does not publish liquidity cannot prove a symbol is liquid. This
        module may only ever produce less, so unknown resolves to *out*."""
        unknown = _quote("MYSTERY", "100", "150", turnover=None)

        result = _scanner(
            quotes=(unknown,), master={"MYSTERY": _record("MYSTERY", "1")}, size=1
        ).select((unknown,), budget=Decimal("50000"))

        assert result.symbols == ()
        assert result.rejected[0].reason is RejectReason.BELOW_TURNOVER_FLOOR
        assert "not published" in result.rejected[0].detail

    def test_zero_turnover_is_rejected(self) -> None:
        nothing = _quote("EMPTY", "100", "150", turnover=Decimal("0"))

        result = _scanner(
            quotes=(nothing,), master={"EMPTY": _record("EMPTY", "1")}, size=1
        ).select((nothing,), budget=Decimal("50000"))

        assert result.rejected[0].reason is RejectReason.BELOW_TURNOVER_FLOOR

    def test_the_floor_is_adjustable(self) -> None:
        thin = _quote("THIN", "100", "150", turnover=Decimal("5000"))

        result = _scanner(
            quotes=(thin,),
            master={"THIN": _record("THIN", "1")},
            size=1,
            min_turnover=Decimal("1000"),
        ).select((thin,), budget=Decimal("50000"))

        assert result.symbols == ("THIN",)

    def test_the_result_reports_the_floor_it_applied(self) -> None:
        result = _scanner().select(_BOARD, budget=Decimal("50000"))

        assert result.min_turnover == MIN_PREOPEN_TURNOVER_INR
        assert result.required_series == EQUITY_SERIES


class TestSeriesFilter:
    """§1.1's prerequisite: a trade-to-trade scrip cannot be squared off intraday at all."""

    @pytest.mark.parametrize("series", ["BE", "BZ"])
    def test_a_trade_to_trade_scrip_is_rejected(self, series: str) -> None:
        """BE/BZ settle by delivery. A position opened in one could not be flattened at 15:15
        — which is the single outcome CLAUDE.md §1.1 exists to make impossible."""
        t2t = _quote("T2T", "100", "150", series=series)

        result = _scanner(quotes=(t2t,), master={"T2T": _record("T2T", "1")}, size=1).select(
            (t2t,), budget=Decimal("50000")
        )

        assert result.symbols == ()
        rejection = result.rejected[0]
        assert rejection.reason is RejectReason.NOT_EQUITY_SERIES
        assert "square-off" in rejection.detail

    @pytest.mark.parametrize("series", ["SM", "ST", "IV"])
    def test_sme_and_other_non_equity_lines_are_rejected(self, series: str) -> None:
        odd = _quote("ODD", "100", "150", series=series)

        result = _scanner(quotes=(odd,), master={"ODD": _record("ODD", "1")}, size=1).select(
            (odd,), budget=Decimal("50000")
        )

        assert result.rejected[0].reason is RejectReason.NOT_EQUITY_SERIES

    def test_an_unknown_series_is_rejected(self) -> None:
        """We cannot prove an unlabelled scrip is not trade-to-trade, so it does not trade."""
        blank = _quote("BLANK", "100", "150", series="")

        result = _scanner(quotes=(blank,), master={"BLANK": _record("BLANK", "1")}, size=1).select(
            (blank,), budget=Decimal("50000")
        )

        assert result.rejected[0].reason is RejectReason.NOT_EQUITY_SERIES
        assert "<unknown>" in result.rejected[0].detail

    @pytest.mark.parametrize("series", ["eq", " EQ ", "Eq"])
    def test_the_comparison_is_case_and_whitespace_insensitive(self, series: str) -> None:
        quote = _quote("OK", "100", "150", series=series)

        result = _scanner(quotes=(quote,), master={"OK": _record("OK", "1")}, size=1).select(
            (quote,), budget=Decimal("50000")
        )

        assert result.symbols == ("OK",)

    def test_series_is_checked_before_liquidity_so_the_tally_is_honest(self) -> None:
        """A T2T scrip that is also thin is reported as T2T. Filing it under turnover would
        suggest a bigger floor could admit it, and no floor ever can."""
        both = _quote("T2T", "100", "150", series="BE", turnover=Decimal("1"))

        result = _scanner(quotes=(both,), master={"T2T": _record("T2T", "1")}, size=1).select(
            (both,), budget=Decimal("50000")
        )

        assert result.rejection_counts() == {"NOT_EQUITY_SERIES": 1}


class TestRanking:
    """Absolute gap, descending — a gap down is as tradable as a gap up (§4.1 is symmetric)."""

    def test_it_ranks_by_absolute_gap_so_a_fall_can_outrank_a_rise(self) -> None:
        result = _scanner().select(_BOARD, budget=Decimal("50000"))

        assert result.symbols == ("BBB", "AAA", "CCC", "DDD")
        assert result.selected[0].gap_pct == Decimal("-8.0000")
        assert result.selected[0].direction == "DOWN"
        assert result.selected[1].direction == "UP"

    def test_it_takes_only_the_top_n(self) -> None:
        result = _scanner(size=2).select(_BOARD, budget=Decimal("50000"))

        assert result.symbols == ("BBB", "AAA")
        assert sum(1 for r in result.rejected if r.reason is RejectReason.OUTRANKED) == 2

    def test_equal_gaps_break_on_symbol_so_a_scan_is_reproducible(self) -> None:
        """Two symbols with the same gap must not select in dict/network order — a watchlist
        that differs between two runs of the same data is not auditable."""
        board = (_quote("ZZZ", "100", "105"), _quote("MMM", "200", "210"))
        master = {"ZZZ": _record("ZZZ", "1"), "MMM": _record("MMM", "2")}

        first = _scanner(quotes=board, master=master, size=1).select(board, budget=Decimal("5000"))
        second = _scanner(quotes=tuple(reversed(board)), master=master, size=1).select(
            tuple(reversed(board)), budget=Decimal("5000")
        )

        assert first.symbols == second.symbols == ("MMM",)


class TestInstrumentResolution:
    """A machine-written watchlist must pass the ingestor's own token verification (§2.3)."""

    def test_a_symbol_absent_from_the_master_is_never_selected(self) -> None:
        """The ingestor treats an unverifiable token as fatal (exit 2, never retried). Writing
        one here would turn a good scan into a session that refuses to boot."""
        board = (_quote("GHOST", "100", "150"), *_BOARD)  # +50 %, would rank first
        result = _scanner(quotes=board).select(board, budget=Decimal("50000"))

        assert "GHOST" not in result.symbols
        assert any(
            r.symbol == "GHOST" and r.reason is RejectReason.NOT_IN_MASTER for r in result.rejected
        )

    def test_a_master_row_with_no_tick_size_is_refused(self) -> None:
        """§6.2 rounds every transmitted price to the tick. Defaulting to Rs.0.05 would produce
        an order the broker rejects, silently, for the whole session."""
        board = (_quote("NOTICK", "100", "150"),)
        result = _scanner(
            quotes=board, master={"NOTICK": _record("NOTICK", "777", tick_size=0.0)}, size=1
        ).select(board, budget=Decimal("50000"))

        assert result.symbols == ()
        assert result.rejected[0].reason is RejectReason.UNKNOWN_TICK_SIZE

    def test_selection_walks_past_an_unresolvable_symbol_to_fill_the_quota(self) -> None:
        """Slicing the top N and resolving afterwards would return three symbols here. The
        scanner ranks, then accepts until N have passed every filter."""
        board = (_quote("GHOST", "100", "200"), *_BOARD)  # ghost ranks first at +100 %
        result = _scanner(quotes=board, size=4).select(board, budget=Decimal("50000"))

        assert len(result.selected) == 4
        assert result.symbols == ("BBB", "AAA", "CCC", "DDD")

    def test_the_master_tick_and_lot_reach_the_candidate(self) -> None:
        result = _scanner(size=4).select(_BOARD, budget=Decimal("50000"))
        by_symbol = {c.symbol: c for c in result.selected}

        assert by_symbol["AAA"].tick_size == Decimal("0.1")
        assert by_symbol["DDD"].tick_size == Decimal("0.01")
        assert by_symbol["AAA"].token == "111"

    def test_a_zero_lot_size_becomes_one_rather_than_an_invalid_item(self) -> None:
        """Cash equity trades in single shares; the master publishes 0 only when the column is
        missing. WatchlistItem requires >= 1, so a raw 0 would fail validation at write time."""
        board = (_quote("EDGE", "100", "150"),)
        result = _scanner(
            quotes=board, master={"EDGE": _record("EDGE", "9", lot_size=0)}, size=1
        ).select(board, budget=Decimal("50000"))

        assert result.selected[0].lot_size == 1


class TestWatchlistConversion:
    """The scanner's output is the config model, so an invalid pick cannot be written."""

    def test_a_candidate_renders_a_valid_watchlist_item(self) -> None:
        candidate = Candidate(
            symbol="AAA",
            token="111",
            previous_close=Decimal("1000"),
            pre_open_price=Decimal("1050"),
            gap_pct=Decimal("5.0000"),
            margin_per_share=Decimal("210.00"),
            tick_size=Decimal("0.10"),
            lot_size=1,
        )
        item = candidate.to_watchlist_item()

        assert (item.symbol, item.token, item.exchange) == ("AAA", "111", "NSE")
        assert item.tick_size == Decimal("0.10")

    def test_the_result_renders_the_whole_watchlist_in_rank_order(self) -> None:
        result = _scanner(size=2).select(_BOARD, budget=Decimal("50000"))

        assert [item.symbol for item in result.watchlist()] == ["BBB", "AAA"]


class TestScanResult:
    def test_an_empty_selection_is_not_usable(self) -> None:
        """The caller's contract: an unusable result leaves the existing watchlist alone."""
        result = _scanner(quotes=()).select((), budget=Decimal("50000"))

        assert not result.is_usable
        assert result.considered == 0

    def test_it_reports_the_affordability_ceiling(self) -> None:
        """Printed at boot so the operator can see the filter is inert at their budget rather
        than assume it is doing work."""
        result = _scanner().select(_BOARD, budget=Decimal("50000"))

        assert result.affordability_ceiling == Decimal("50000") * INTRADAY_LEVERAGE

    def test_rejections_are_counted_by_reason(self) -> None:
        result = _scanner(size=1).select(_BOARD, budget=Decimal("50000"))

        assert result.rejection_counts() == {"OUTRANKED": 3}


class TestScannerConstruction:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"size": 0}, "size must be at least 1"),
            ({"price_floor": Decimal("-1")}, "price_floor must not be negative"),
            ({"leverage": Decimal("0")}, "leverage must be positive"),
            ({"min_turnover": Decimal("-1")}, "min_turnover must not be negative"),
            ({"required_series": "  "}, "required_series must not be blank"),
        ],
    )
    def test_nonsense_parameters_are_refused_at_construction(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            PreMarketScanner(source=_StaticSource(), resolver=_StaticResolver({}), **kwargs)


@pytest.mark.asyncio
class TestScanAsync:
    async def test_scan_fetches_then_selects(self) -> None:
        source = _StaticSource(_BOARD)
        scanner = PreMarketScanner(source=source, resolver=_StaticResolver(_MASTER), size=2)

        result = await scanner.scan(budget=Decimal("50000"))

        assert source.calls == 1
        assert result.symbols == ("BBB", "AAA")

    async def test_a_fetch_failure_propagates_rather_than_returning_nothing(self) -> None:
        """An empty result and a failed scan must be distinguishable: the first is "no symbol
        qualified", the second is "we do not know". Only the exception says the second."""
        scanner = PreMarketScanner(
            source=_StaticSource(error=PreOpenFetchError("NSE unreachable")),
            resolver=_StaticResolver(_MASTER),
        )

        with pytest.raises(PreOpenFetchError):
            await scanner.scan(budget=Decimal("50000"))

    async def test_a_non_positive_budget_is_refused_before_any_io(self) -> None:
        source = _StaticSource(_BOARD)
        scanner = PreMarketScanner(source=source, resolver=_StaticResolver(_MASTER))

        with pytest.raises(ValueError, match="budget must be positive"):
            await scanner.scan(budget=Decimal("0"))
        assert source.calls == 0


# ──────────────────────────────────────────────────────────────────────────────
# The NSE document
# ──────────────────────────────────────────────────────────────────────────────


def _nse_row(symbol: str, previous_close: object, last_price: object, **extra: object) -> Any:
    metadata: dict[str, object] = {
        "symbol": symbol,
        "previousClose": previous_close,
        "lastPrice": last_price,
        "series": "EQ",
        "totalTurnover": 50_000_000.0,
    }
    metadata.update(extra)
    return {"metadata": metadata, "detail": {"preOpenMarket": {"finalPrice": last_price}}}


def _parsed(symbol: str, previous_close: str, pre_open: str) -> PreOpenQuote:
    """What :func:`_nse_row` parses into, given its EQ/liquid defaults."""
    return PreOpenQuote(
        symbol,
        Decimal(previous_close),
        Decimal(pre_open),
        turnover=Decimal("50000000.0"),
        series="EQ",
    )


class TestParseNsePreOpen:
    """The document is unofficial and can change shape. Every projection is defensive."""

    def test_it_projects_the_published_shape(self) -> None:
        payload = {"data": [_nse_row("RELIANCE", 2885.0, 2900.5)], "advances": 1}

        quotes = parse_nse_pre_open(payload)

        assert quotes == (
            PreOpenQuote(
                "RELIANCE",
                Decimal("2885.0"),
                Decimal("2900.5"),
                turnover=Decimal("50000000.0"),
                series="EQ",
            ),
        )

    def test_turnover_is_read_in_rupees(self) -> None:
        """Verified against the live board: metadata.totalTurnover divided by
        lastPrice x finalQuantity is exactly 1.000 for every symbol on it. Not lakhs, not
        crores — a units error here would move the floor by five orders of magnitude."""
        row = _nse_row("SHILCTECH", 4581.70, 3972.0, totalTurnover=17437080.0, finalQuantity=4390)

        quote = parse_nse_pre_open({"data": [row]})[0]

        assert quote.turnover == Decimal("17437080.0")
        assert quote.turnover == quote.pre_open_price * Decimal("4390")

    def test_turnover_falls_back_to_price_times_quantity(self) -> None:
        """If NSE drops the column, the same number is reconstructible from two fields we
        already read. Better than rejecting the whole board on a schema tweak."""
        row = _nse_row("X", 100.0, 110.0, totalTurnover=None, finalQuantity=1000)

        assert parse_nse_pre_open({"data": [row]})[0].turnover == Decimal("110000.0")

    def test_turnover_is_none_when_neither_field_is_usable(self) -> None:
        """None, not zero — and the screen rejects it either way. What must not happen is a
        default that reads as "liquid"."""
        row = _nse_row("X", 100.0, 110.0, totalTurnover=None, finalQuantity=None)

        assert parse_nse_pre_open({"data": [row]})[0].turnover is None

    @pytest.mark.parametrize(("raw", "expected"), [("be", "BE"), (" eq ", "EQ"), (None, "NONE")])
    def test_series_is_normalised(self, raw: object, expected: str) -> None:
        row = _nse_row("X", 100.0, 110.0, series=raw)

        assert parse_nse_pre_open({"data": [row]})[0].series == expected

    def test_a_missing_series_key_reads_as_unknown(self) -> None:
        row = {"metadata": {"symbol": "X", "previousClose": 100, "lastPrice": 110}}

        assert parse_nse_pre_open({"data": [row]})[0].series == ""

    def test_prices_arrive_as_floats_via_str_so_they_stay_exact(self) -> None:
        """Decimal(2900.5) would carry the float's binary error into money."""
        quotes = parse_nse_pre_open({"data": [_nse_row("X", 100.1, 200.2)]})

        assert quotes[0].pre_open_price == Decimal("200.2")

    def test_it_accepts_comma_formatted_strings(self) -> None:
        quotes = parse_nse_pre_open({"data": [_nse_row("X", "2,885.00", "2,900.50")]})

        assert quotes[0].previous_close == Decimal("2885.00")

    def test_it_falls_back_from_last_price_to_iep_then_to_final_price(self) -> None:
        via_iep = parse_nse_pre_open({"data": [_nse_row("X", 100, None, iep=110.0)]})
        assert via_iep[0].pre_open_price == Decimal("110.0")

        via_final = parse_nse_pre_open(
            {
                "data": [
                    {
                        "metadata": {"symbol": "Y", "previousClose": 100},
                        "detail": {"preOpenMarket": {"finalPrice": 120.0}},
                    }
                ]
            }
        )
        assert via_final[0].pre_open_price == Decimal("120.0")

    @pytest.mark.parametrize("value", [None, "-", "", "not-a-number", True])
    def test_an_unusable_price_skips_the_row_rather_than_becoming_zero(self, value: object) -> None:
        """`True` is in this list deliberately: `isinstance(True, int)` is True in Python, so a
        bool would otherwise parse as a price of Rs.1."""
        quotes = parse_nse_pre_open({"data": [_nse_row("X", 100, value)]})

        assert quotes == ()

    def test_a_row_missing_its_symbol_is_skipped(self) -> None:
        quotes = parse_nse_pre_open({"data": [{"metadata": {"previousClose": 1, "lastPrice": 2}}]})

        assert quotes == ()

    def test_malformed_rows_are_skipped_but_good_ones_survive(self) -> None:
        """One bad row is one symbol we will not consider. It must not cost the whole scan."""
        payload = {"data": ["nonsense", {"no": "metadata"}, _nse_row("GOOD", 100, 110)]}

        assert parse_nse_pre_open(payload) == (_parsed("GOOD", "100", "110"),)

    @pytest.mark.parametrize("payload", [[], "text", {"advances": 1}, {"data": "nope"}])
    def test_a_body_we_cannot_read_at_all_is_a_failed_scan(self, payload: object) -> None:
        """The distinction that matters: a body with no `data` array is not an empty board, it
        is a scan that did not happen — and the caller must keep the old watchlist."""
        with pytest.raises(PreOpenFetchError):
            parse_nse_pre_open(payload)


@pytest.mark.asyncio
class TestNsePreOpenSource:
    """Exercised against a mock transport. Never against nseindia.com."""

    @staticmethod
    def _client(handler: Any) -> Any:
        def factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        return factory

    async def test_it_fetches_the_whole_board_in_one_request(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if "api/market-data-pre-open" in str(request.url):
                return httpx.Response(200, json={"data": [_nse_row("AAA", 100, 110)]})
            return httpx.Response(200, text="<html>cookie</html>")

        source = NsePreOpenSource(client_factory=self._client(handler))
        quotes = await source.fetch()

        assert quotes == (_parsed("AAA", "100", "110"),)
        api_calls = [url for url in calls if "api/market-data-pre-open" in url]
        assert len(api_calls) == 1, "the whole universe must cost exactly one API request"

    async def test_a_priming_failure_does_not_abort_the_fetch(self) -> None:
        """The cookie requirement has come and gone over the years. Hard-failing on a step that
        may no longer be needed would cost a session for nothing."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "api/market-data-pre-open" in str(request.url):
                return httpx.Response(200, json={"data": [_nse_row("AAA", 100, 110)]})
            raise httpx.ConnectError("priming refused", request=request)

        source = NsePreOpenSource(client_factory=self._client(handler))

        assert await source.fetch() == (_parsed("AAA", "100", "110"),)

    @pytest.mark.parametrize("status", [403, 429, 500, 503])
    async def test_an_http_error_becomes_a_fetch_error(self, status: int) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "api/market-data-pre-open" in str(request.url):
                return httpx.Response(status, text="denied")
            return httpx.Response(200, text="ok")

        source = NsePreOpenSource(client_factory=self._client(handler))

        with pytest.raises(PreOpenFetchError, match="NSE pre-open fetch failed"):
            await source.fetch()

    async def test_a_non_json_body_becomes_a_fetch_error(self) -> None:
        """NSE serves an HTML block page when it dislikes the request. Parsed as a board that
        would be an empty scan; it is actually a refusal."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Access Denied</html>")

        source = NsePreOpenSource(client_factory=self._client(handler))

        with pytest.raises(PreOpenFetchError, match="not JSON"):
            await source.fetch()

    async def test_a_transport_failure_becomes_a_fetch_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("timed out", request=request)

        source = NsePreOpenSource(client_factory=self._client(handler))

        with pytest.raises(PreOpenFetchError):
            await source.fetch()


class TestMasterSymbolResolver:
    """Keyed on the master's `name`, the same column the ingestor verifies against."""

    def test_it_resolves_a_cash_equity_row_by_name(self) -> None:
        resolver = MasterSymbolResolver([_record("RELIANCE", "2885")])

        record = resolver.resolve("RELIANCE")

        assert record is not None
        assert record.token == "2885"

    def test_lookup_is_case_and_whitespace_insensitive(self) -> None:
        resolver = MasterSymbolResolver([_record("INFY", "1594")])

        assert resolver.resolve("  infy  ") is not None

    def test_it_ignores_derivatives_and_other_series(self) -> None:
        """Only `-EQ` is cash equity. A futures row sharing the name carries a token that would
        subscribe to a completely different instrument."""
        resolver = MasterSymbolResolver(
            [
                _record("NIFTY", "999", trading_symbol="NIFTY28AUG25FUT", expiry="28AUG2025"),
                _record("SBIN", "111", trading_symbol="SBIN-BE"),
            ]
        )

        assert len(resolver) == 0
        assert resolver.resolve("NIFTY") is None
        assert resolver.resolve("SBIN") is None

    def test_it_ignores_a_dated_contract_even_with_an_eq_symbol(self) -> None:
        resolver = MasterSymbolResolver([_record("ODD", "1", expiry="28AUG2025")])

        assert resolver.resolve("ODD") is None

    def test_it_ignores_other_exchanges(self) -> None:
        resolver = MasterSymbolResolver([_record("RELIANCE", "500325", exchange="BSE")])

        assert resolver.resolve("RELIANCE") is None

    def test_the_first_row_for_a_name_wins_deterministically(self) -> None:
        resolver = MasterSymbolResolver([_record("DUP", "first"), _record("DUP", "second")])

        record = resolver.resolve("DUP")
        assert record is not None
        assert record.token == "first"
