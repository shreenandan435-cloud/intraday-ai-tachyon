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

import json
from datetime import datetime, time
from decimal import Decimal
from typing import Any

import httpx
import pytest

from tachyon.core.clock import IST, ManualClock, ist_at
from tachyon.ingestion.instruments import InstrumentRecord
from tachyon.strategy.scanner import (
    DEFAULT_ROTATION_TIMES,
    DEFAULT_SELECTION_SIZE,
    DEFAULT_TURNOVER_PERCENTILE,
    EQUITY_SERIES,
    INTRADAY_LEVERAGE,
    MIN_PREOPEN_TURNOVER_INR,
    MIN_TURNOVER_SAMPLE,
    PENNY_PRICE_FLOOR_INR,
    TURNOVER_BACKSTOP_INR,
    AngelMoversSource,
    Candidate,
    DualSourcePreOpen,
    IntradayScanner,
    MasterSymbolResolver,
    NsePreOpenSource,
    PreMarketScanner,
    PreOpenFetchError,
    PreOpenQuote,
    RankingWeights,
    RejectReason,
    RotationScheduler,
    composite_scores,
    cross_sectional_z,
    gap_percent,
    margin_per_share,
    parse_angel_movers,
    parse_nse_pre_open,
    percentile_turnover_floor,
    plan_rotation,
    rvol_values,
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
    volume: Decimal | None = None,
    spread_bps: Decimal | None = None,
) -> PreOpenQuote:
    return PreOpenQuote(
        symbol,
        Decimal(previous_close),
        Decimal(pre_open),
        turnover=turnover,
        series=series,
        volume=volume,
        spread_bps=spread_bps,
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


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic liquidity floor
# ──────────────────────────────────────────────────────────────────────────────


class TestPercentileTurnoverFloor:
    """The nearest-rank percentile of published positive turnovers."""

    def test_it_returns_an_observed_value_never_an_interpolation(self) -> None:
        values = ("100", "200", "300", "400", "500", "600", "700", "800", "900", "1000")
        turnovers = [Decimal(n) for n in values]
        assert percentile_turnover_floor(turnovers, 90.0) == Decimal("900")

    def test_top_decile_of_a_realistic_board(self) -> None:
        """p90 of 20 symbols: rank ceil(0.9 * 20) = 18 → the third-largest value."""
        turnovers = [Decimal(i * 1_000_000) for i in range(1, 21)]
        assert percentile_turnover_floor(turnovers, 90.0) == Decimal("18000000")

    def test_non_positive_turnovers_are_excluded(self) -> None:
        turnovers = [Decimal("-5"), Decimal("0"), *(Decimal(i) for i in range(1, 11))]
        assert percentile_turnover_floor(turnovers, 100.0) == Decimal("10")

    def test_a_sample_below_the_minimum_is_uncomputable(self) -> None:
        """Fewer than MIN_TURNOVER_SAMPLE rows: a quantile of a handful is noise, and the
        caller falls back to the static floor rather than trusting it."""
        few = [Decimal("1000000")] * (MIN_TURNOVER_SAMPLE - 1)
        assert percentile_turnover_floor(few, 90.0) is None

    def test_an_empty_sample_is_uncomputable(self) -> None:
        assert percentile_turnover_floor((), 90.0) is None

    @pytest.mark.parametrize("percentile", [0.0, -1.0, 100.1])
    def test_an_out_of_range_percentile_is_refused(self, percentile: float) -> None:
        with pytest.raises(ValueError, match="percentile must be in"):
            percentile_turnover_floor([Decimal("1")] * MIN_TURNOVER_SAMPLE, percentile)


class TestDynamicTurnoverFloor:
    """The scanner's liquidity floor, derived from the board instead of a rigid constant."""

    @staticmethod
    def _board_of_ten() -> tuple[PreOpenQuote, ...]:
        """Ten EQ symbols, turnovers 1cr..10cr, gaps all +5% so only liquidity varies."""
        return tuple(
            _quote(f"S{i:02d}", "1000", "1050", turnover=Decimal(i * 10_000_000))
            for i in range(1, 11)
        )

    def test_static_mode_is_unchanged_and_labelled(self) -> None:
        """No percentile → the configured rupee floor, exactly as before."""
        board = self._board_of_ten()
        result = _scanner(quotes=board, size=10).select(board, budget=Decimal("50000"))
        assert result.floor_source == "static"
        assert result.min_turnover == MIN_PREOPEN_TURNOVER_INR

    def test_the_top_decile_floor_is_derived_from_the_board(self) -> None:
        """p90 of 1cr..10cr is 9cr: the eight thinnest symbols drop, two survive."""
        board = self._board_of_ten()
        master = {
            quote.symbol: _record(quote.symbol, str(index)) for index, quote in enumerate(board)
        }
        scanner = _scanner(quotes=board, master=master, size=10, turnover_percentile=90.0)

        result = scanner.select(board, budget=Decimal("50000"))

        assert result.floor_source == "percentile"
        assert result.min_turnover == Decimal("90000000")
        # Gaps all tie at +5 %, so the turnover term orders the survivors: S10 first.
        assert result.symbols == ("S10", "S09")
        assert result.rejection_counts()["BELOW_TURNOVER_FLOOR"] == 8

    def test_the_floor_adapts_down_on_a_thin_day(self) -> None:
        """The rigidity the dynamic floor removes: on a board where everything matches
        ~Rs.20 lakh, the static 1cr floor rejects the entire board and the session starts
        blind. The percentile floor keeps the top of what the day actually offers."""
        thin = tuple(
            _quote(f"T{i:02d}", "1000", "1050", turnover=Decimal(1_500_000 + i * 100_000))
            for i in range(10)
        )
        master = {
            quote.symbol: _record(quote.symbol, str(index)) for index, quote in enumerate(thin)
        }
        static = _scanner(quotes=thin, master=master, size=10).select(
            thin, budget=Decimal("50000")
        )
        dynamic = _scanner(
            quotes=thin, master=master, size=10, turnover_percentile=90.0
        ).select(thin, budget=Decimal("50000"))

        assert static.symbols == ()
        assert dynamic.floor_source == "percentile"
        # Nearest-rank p90 of ten rows is the ninth value; the boundary value itself passes,
        # so the two deepest symbols survive where the static floor admitted none.
        assert dynamic.symbols == ("T09", "T08")

    def test_the_backstop_bounds_the_percentile_from_below(self) -> None:
        """A universally dead board's top decile is still dead. The percentile may adapt,
        but it may not bottom out below the backstop."""
        dead = tuple(
            _quote(f"D{i:02d}", "1000", "1050", turnover=Decimal(10_000 + i * 1_000))
            for i in range(10)
        )
        result = _scanner(quotes=dead, size=10, turnover_percentile=90.0).select(
            dead, budget=Decimal("50000")
        )

        assert result.floor_source == "backstop"
        assert result.min_turnover == TURNOVER_BACKSTOP_INR
        assert result.symbols == ()

    def test_an_uncomputable_percentile_falls_back_to_the_static_floor(self) -> None:
        """Too few published turnovers → the scan may only ever produce less, so the known
        floor stands rather than a quantile of three rows."""
        sparse = tuple(
            _quote(f"P{i}", "1000", "1050", turnover=Decimal(50_000_000) if i < 3 else None)
            for i in range(6)
        )
        result = _scanner(quotes=sparse, size=6, turnover_percentile=90.0).select(
            sparse, budget=Decimal("50000")
        )

        assert result.floor_source == "fallback-static"
        assert result.min_turnover == MIN_PREOPEN_TURNOVER_INR

    def test_the_result_reports_the_floor_it_applied(self) -> None:
        board = self._board_of_ten()
        result = _scanner(quotes=board, size=10, turnover_percentile=90.0).select(
            board, budget=Decimal("50000")
        )
        assert result.min_turnover == Decimal("90000000")
        assert result.floor_source == "percentile"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"turnover_percentile": 0.0}, "turnover_percentile must be in"),
            ({"turnover_percentile": 100.5}, "turnover_percentile must be in"),
            ({"turnover_backstop": Decimal("-1")}, "turnover_backstop must not be negative"),
        ],
    )
    def test_nonsense_dynamic_parameters_are_refused(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            PreMarketScanner(source=_StaticSource(), resolver=_StaticResolver({}), **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Composite ranking score
# ──────────────────────────────────────────────────────────────────────────────


class TestCrossSectionalZ:
    def test_it_standardises_to_mean_zero_unit_variance(self) -> None:
        z = cross_sectional_z([1.0, 2.0, 3.0, 4.0])
        assert sum(z) == pytest.approx(0.0, abs=1e-12)
        assert max(z) == pytest.approx(1.3416407864998738)  # population std, not sample

    def test_a_single_value_is_not_a_cross_section(self) -> None:
        assert cross_sectional_z([42.0]) == (0.0,)

    def test_a_tied_section_contributes_nothing(self) -> None:
        """Every symbol with the same turnover: the term drops out instead of dividing
        by zero. This is what keeps tied boards ranked purely by gap."""
        assert cross_sectional_z([5.0, 5.0, 5.0]) == (0.0, 0.0, 0.0)

    def test_empty_in_empty_out(self) -> None:
        assert cross_sectional_z([]) == ()


class TestRvolValues:
    @staticmethod
    def _q(symbol: str, volume: Decimal | None) -> PreOpenQuote:
        return _quote(symbol, "100", "110", volume=volume)

    def test_it_is_volume_relative_to_the_cross_sectional_mean(self) -> None:
        quotes = (self._q("A", Decimal("100")), self._q("B", Decimal("300")))
        # mean of positives = 200 → 0.5x and 1.5x
        assert rvol_values(quotes) == (0.5, 1.5)

    def test_a_source_without_volumes_drops_the_term_section_wide(self) -> None:
        """No invented proxies: turnover/price would double-count the turnover term while
        smuggling in a price correlation."""
        quotes = (self._q("A", None), self._q("B", None))
        assert rvol_values(quotes) == (0.0, 0.0)

    def test_a_single_published_volume_is_not_relative(self) -> None:
        quotes = (self._q("A", Decimal("100")), self._q("B", None))
        assert rvol_values(quotes) == (0.0, 0.0)


class TestCompositeScores:
    def test_the_spread_penalty_is_subtracted_in_raw_basis_points(self) -> None:
        quotes = (
            _quote("TIGHT", "100", "110", spread_bps=Decimal("2")),
            _quote("WIDE", "100", "110", spread_bps=Decimal("50")),
        )
        # Identical gaps and tied turnovers zero those terms; the spread penalty stands alone.
        weights = RankingWeights(w_gap=1.0, w_turnover=0.0, w_rvol=0.0, w_spread=0.5)
        tight, wide = composite_scores(quotes, weights)
        assert tight == pytest.approx(-1.0)
        assert wide == pytest.approx(-25.0)

    def test_the_gap_term_uses_magnitude_so_a_fall_ranks_with_a_rise(self) -> None:
        """§4.1 is symmetric: a signed z would rank the day's biggest faller last."""
        quotes = (
            _quote("FALLER", "1000", "900"),  # -10 %
            _quote("RISER", "1000", "1050"),  # +5 %
            _quote("QUIET", "1000", "1000"),  # 0 %
        )
        weights = RankingWeights(w_turnover=0.0, w_rvol=0.0, w_spread=0.0)
        faller, riser, quiet = composite_scores(quotes, weights)
        assert faller > riser > quiet

    def test_turnover_is_log_scaled_before_standardisation(self) -> None:
        """One mega-cap must not pin the rest of the board at z≈0."""
        quotes = (
            _quote("MEGA", "100", "110", turnover=Decimal("100000000000")),  # 1e11
            _quote("MID", "100", "110", turnover=Decimal("100000000")),  # 1e8
            _quote("SMALL", "100", "110", turnover=Decimal("10000000")),  # 1e7
        )
        weights = RankingWeights(w_gap=0.0, w_rvol=0.0, w_spread=0.0)
        mega, mid, small = composite_scores(quotes, weights)
        # log10: 11, 8, 7 → z-scores all O(1); raw turnover would skew one mega-cap to dominate.
        assert mega > mid > small
        assert mid - small == pytest.approx(0.588348405414552, rel=1e-9)

    def test_missing_spread_is_neutral_not_a_bonus(self) -> None:
        quotes = (
            _quote("NOSPREAD", "100", "110", spread_bps=None),
            _quote("SOME", "100", "110", spread_bps=Decimal("5")),
        )
        # Identical gaps zero the gap term; the spread term is the only thing that can differ.
        weights = RankingWeights(w_gap=1.0, w_turnover=0.0, w_rvol=0.0, w_spread=1.0)
        none, some = composite_scores(quotes, weights)
        assert none == pytest.approx(0.0)
        assert some == pytest.approx(-5.0)

    def test_empty_board_scores_empty(self) -> None:
        assert composite_scores((), RankingWeights()) == ()

    def test_default_weights_collapse_to_gap_order_when_liquidity_ties(self) -> None:
        """The backward-compatibility guarantee: on the NSE pre-open document (no volume,
        no spread) with tied turnovers, the composite reduces to the original |gap| order."""
        quotes = _BOARD  # four symbols, identical LIQUID turnover
        symbols = (q.symbol for q in quotes)
        scores = dict(zip(symbols, composite_scores(quotes, RankingWeights()), strict=True))
        assert scores["BBB"] > scores["AAA"] > scores["CCC"] > scores["DDD"]


class TestRankingWeights:
    def test_a_negative_weight_is_refused(self) -> None:
        with pytest.raises(ValueError, match="w_spread must be a finite non-negative"):
            RankingWeights(w_spread=-0.5)

    def test_a_score_with_only_the_spread_penalty_is_refused(self) -> None:
        """It would rank the least-liquid books first whenever spreads are missing."""
        with pytest.raises(ValueError, match="at least one"):
            RankingWeights(w_gap=0.0, w_turnover=0.0, w_rvol=0.0, w_spread=1.0)


class TestCompositeSelection:
    """End-to-end: the scanner ranks by composite score, not gap alone."""

    def test_a_high_turnover_symbol_outranks_a_bigger_gap(self) -> None:
        # A third anchor keeps the cross-section non-degenerate: with only two symbols the gap
        # and turnover z-scores are exact mirrors (±1) and always cancel to a tie.
        board = (
            _quote("BIGGAP", "1000", "1100", turnover=Decimal("10000000")),  # +10 %, 1cr
            _quote("BIGLIQ", "1000", "1060", turnover=Decimal("500000000")),  # +6 %, 50cr
            _quote("ANCHOR", "1000", "1010", turnover=Decimal("10000000")),  # +1 %, 1cr
        )
        master = {
            "BIGGAP": _record("BIGGAP", "1"),
            "BIGLIQ": _record("BIGLIQ", "2"),
            "ANCHOR": _record("ANCHOR", "3"),
        }

        result = _scanner(quotes=board, master=master, size=2).select(
            board, budget=Decimal("50000")
        )

        assert result.symbols == ("BIGLIQ", "BIGGAP")
        assert result.selected[0].score > result.selected[1].score

    def test_volume_evidence_moves_the_ranking(self) -> None:
        board = (
            _quote("NOVOL", "1000", "1080", volume=None),  # +8 %, no volume evidence
            _quote("HIVOL", "1000", "1050", volume=Decimal("9000000")),  # +5 %, 9x mean vol
            _quote("LOVOL", "1000", "1050", volume=Decimal("1000000")),
        )
        master = {s: _record(s, str(i)) for i, s in enumerate(("NOVOL", "HIVOL", "LOVOL"))}

        result = _scanner(quotes=board, master=master, size=3).select(
            board, budget=Decimal("50000")
        )

        assert result.symbols[0] == "HIVOL"

    def test_the_candidate_carries_its_score(self) -> None:
        result = _scanner().select(_BOARD, budget=Decimal("50000"))
        assert all(isinstance(candidate.score, float) for candidate in result.selected)
        scores = [candidate.score for candidate in result.selected]
        assert scores == sorted(scores, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Angel One fallback source
# ──────────────────────────────────────────────────────────────────────────────


def _angel_row(
    symbol: str,
    ltp: object,
    *,
    net_change: object = None,
    percent_change: object = None,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {"symbol": symbol, "lastTradedPrice": ltp}
    if net_change is not None:
        row["netChange"] = net_change
    if percent_change is not None:
        row["percentChange"] = percent_change
    row.update(extra)
    return row


def _angel_payload(
    *gainers: dict[str, object], losers: tuple[dict[str, object], ...] = ()
) -> dict[str, object]:
    return {"status": True, "data": {"gainers": list(gainers), "losers": list(losers)}}


class TestParseAngelMovers:
    def test_it_projects_gainers_and_losers_into_quotes(self) -> None:
        payload = _angel_payload(
            _angel_row("RELIANCE", 2900.5, net_change=115.5, volume=123456, turnover=350000000),
            losers=(_angel_row("ZEEL", 120.0, net_change=-6.0),),
        )
        quotes = parse_angel_movers(payload)

        assert tuple(q.symbol for q in quotes) == ("RELIANCE", "ZEEL")
        reliance = quotes[0]
        assert reliance.pre_open_price == Decimal("2900.5")
        assert reliance.previous_close == Decimal("2785.00")
        assert reliance.turnover == Decimal("350000000")
        assert reliance.volume == Decimal("123456")
        assert reliance.series == EQUITY_SERIES
        assert quotes[1].previous_close == Decimal("126.00")

    def test_the_previous_close_falls_back_to_percent_change(self) -> None:
        quotes = parse_angel_movers(_angel_payload(_angel_row("X", 105.0, percent_change=5.0)))
        assert quotes[0].previous_close == Decimal("100.00")

    def test_the_eq_suffix_is_stripped_from_trading_symbols(self) -> None:
        payload = _angel_payload(_angel_row("RELIANCE-EQ", 100.0, net_change=1.0))
        quotes = parse_angel_movers(payload)
        assert quotes[0].symbol == "RELIANCE"

    def test_a_row_with_no_usable_previous_close_is_skipped(self) -> None:
        """The gap denominator is the one number this module never guesses."""
        payload = _angel_payload(
            _angel_row("NOCHANGE", 100.0),  # neither netChange nor percentChange
            _angel_row("GOOD", 200.0, net_change=10.0),
        )
        assert tuple(q.symbol for q in parse_angel_movers(payload)) == ("GOOD",)

    def test_a_change_that_implies_a_non_positive_close_is_skipped(self) -> None:
        # netChange = ltp - prev, so a "gain" larger than the price itself implies a negative
        # previous close — an unusable gap denominator, so the row is dropped.
        payload = _angel_payload(_angel_row("BLOWUP", 100.0, net_change=150.0))
        assert parse_angel_movers(payload) == ()

    def test_duplicate_symbols_are_taken_once(self) -> None:
        payload = _angel_payload(
            _angel_row("DUP", 100.0, net_change=1.0),
            _angel_row("DUP", 101.0, net_change=2.0),
        )
        assert len(parse_angel_movers(payload)) == 1

    @pytest.mark.parametrize("value", [None, "-", "junk", 0, -5])
    def test_an_unusable_price_skips_the_row(self, value: object) -> None:
        assert parse_angel_movers(_angel_payload(_angel_row("X", value, net_change=1.0))) == ()

    @pytest.mark.parametrize("payload", [[], "text", {"status": False}, {"data": []}])
    def test_a_body_we_cannot_read_is_a_failed_scan_not_an_empty_board(
        self, payload: object
    ) -> None:
        with pytest.raises(PreOpenFetchError):
            parse_angel_movers(payload)


@pytest.mark.asyncio
class TestAngelMoversSource:
    """Exercised against a mock transport. Never against apiconnect.angelone.in."""

    _HEADERS: dict[str, str] = {"Authorization": "Bearer jwt", "x-api-key": "key"}

    @staticmethod
    def _client(handler: Any) -> Any:
        def factory() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        return factory

    async def test_it_posts_the_screener_and_projects_quotes(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200, json=_angel_payload(_angel_row("AAA", 110.0, net_change=10.0))
            )

        source = AngelMoversSource(headers=self._HEADERS, client_factory=self._client(handler))
        quotes = await source.fetch()

        assert tuple(q.symbol for q in quotes) == ("AAA",)
        assert len(seen) == 1
        assert seen[0].method == "POST"
        assert seen[0].url.path == "/rest/secure/angelbroking/marketData/v1/gainersLosers"
        assert seen[0].headers["Authorization"] == "Bearer jwt"
        assert json.loads(seen[0].content) == {"exchange": "NSE", "duration": "1"}

    async def test_the_header_provider_resolves_at_fetch_time(self) -> None:
        calls = {"n": 0}

        def provider() -> dict[str, str]:
            calls["n"] += 1
            return {"Authorization": "Bearer late-jwt"}

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer late-jwt"
            return httpx.Response(200, json=_angel_payload())

        source = AngelMoversSource(header_provider=provider, client_factory=self._client(handler))
        assert calls["n"] == 0, "headers must not be resolved at construction"
        await source.fetch()
        assert calls["n"] == 1

    async def test_an_unauthenticated_provider_is_a_fetch_error(self) -> None:
        def provider() -> dict[str, str]:
            raise PreOpenFetchError("no cached session")

        factory = self._client(lambda r: httpx.Response(200))
        source = AngelMoversSource(header_provider=provider, client_factory=factory)
        with pytest.raises(PreOpenFetchError, match="no cached session"):
            await source.fetch()

    @pytest.mark.parametrize("status", [401, 403, 429, 500])
    async def test_an_http_error_becomes_a_fetch_error(self, status: int) -> None:
        source = AngelMoversSource(
            headers=self._HEADERS,
            client_factory=self._client(lambda r: httpx.Response(status, text="denied")),
        )
        with pytest.raises(PreOpenFetchError, match="Angel movers fetch failed"):
            await source.fetch()

    async def test_a_non_json_body_becomes_a_fetch_error(self) -> None:
        source = AngelMoversSource(
            headers=self._HEADERS,
            client_factory=self._client(lambda r: httpx.Response(200, text="<html>nope</html>")),
        )
        with pytest.raises(PreOpenFetchError, match="not JSON"):
            await source.fetch()


def test_angel_movers_construction_without_any_headers_is_refused() -> None:
    with pytest.raises(ValueError, match="authenticated endpoint"):
        AngelMoversSource()


class TestAngelAuthHeaders:
    def test_it_carries_every_header_the_waf_and_api_require(self) -> None:
        from tachyon.strategy.scanner import angel_auth_headers

        headers = angel_auth_headers(
            api_key="key", client_id="client", jwt_token="jwt", feed_token="feed"
        )
        assert headers["Authorization"] == "Bearer jwt"
        assert headers["x-api-key"] == "key"
        assert headers["x-client-code"] == "client"
        assert headers["x-feed-token"] == "feed"
        assert headers["X-UserKey"] == "key"
        # Loopback in the IP headers is a known WAF rejection — never emit it.
        assert not headers["X-ClientLocalIP"].startswith("127.")
        assert not headers["X-ClientPublicIP"].startswith("127.")


@pytest.mark.asyncio
class TestDualSourcePreOpen:
    """Primary first; the secondary is a degradation path, never a blend."""

    async def test_a_healthy_primary_is_used_alone(self) -> None:
        primary = _StaticSource(_BOARD)
        secondary = _StaticSource((_quote("FALLBACK", "100", "150"),))
        dual = DualSourcePreOpen(primary, secondary)

        quotes = await dual.fetch()

        assert quotes == _BOARD
        assert dual.last_used == "primary"
        assert secondary.calls == 0

    async def test_a_failed_primary_falls_back_to_the_secondary(self) -> None:
        fallback = (_quote("FALLBACK", "100", "150"),)
        dual = DualSourcePreOpen(
            _StaticSource(error=PreOpenFetchError("NSE blocked")), _StaticSource(fallback)
        )

        quotes = await dual.fetch()

        assert quotes == fallback
        assert dual.last_used == "secondary"

    async def test_both_sources_failing_reports_both_causes(self) -> None:
        dual = DualSourcePreOpen(
            _StaticSource(error=PreOpenFetchError("NSE blocked")),
            _StaticSource(error=PreOpenFetchError("Angel unauthenticated")),
        )

        with pytest.raises(PreOpenFetchError, match="NSE blocked") as excinfo:
            await dual.fetch()
        assert "Angel unauthenticated" in str(excinfo.value)


# ──────────────────────────────────────────────────────────────────────────────
# Intraday rotation
# ──────────────────────────────────────────────────────────────────────────────


def _clock_at(hour: int, minute: int) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 28, hour, minute, tzinfo=IST))


class TestRotationScheduler:
    def test_the_default_slots_are_the_documented_ones(self) -> None:
        assert DEFAULT_ROTATION_TIMES == (time(9, 30), time(11, 30), time(13, 30))
        assert RotationScheduler(clock=_clock_at(8, 0)).times == DEFAULT_ROTATION_TIMES

    def test_times_are_normalised_to_sorted_order(self) -> None:
        scheduler = RotationScheduler([time(13, 30), time(9, 30)], clock=_clock_at(8, 0))
        assert scheduler.times == (time(9, 30), time(13, 30))

    def test_an_empty_schedule_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one rotation time"):
            RotationScheduler((), clock=_clock_at(8, 0))

    def test_next_fire_rolls_through_the_day(self) -> None:
        scheduler = RotationScheduler(clock=_clock_at(8, 0))
        assert scheduler.next_fire(ist_at(scheduler.next_fire().date(), time(10, 0))) == ist_at(
            datetime(2026, 8, 28, tzinfo=IST).date(), time(11, 30)
        )

    def test_next_fire_rolls_to_tomorrow_after_the_last_slot(self) -> None:
        scheduler = RotationScheduler(clock=_clock_at(8, 0))
        after = datetime(2026, 8, 28, 14, 0, tzinfo=IST)
        assert scheduler.next_fire(after) == datetime(2026, 8, 29, 9, 30, tzinfo=IST)

    def test_a_slot_fires_exactly_once(self) -> None:
        scheduler = RotationScheduler(clock=_clock_at(9, 0))
        assert scheduler.advance(datetime(2026, 8, 28, 9, 31, tzinfo=IST)) == datetime(
            2026, 8, 28, 9, 30, tzinfo=IST
        )
        assert scheduler.advance(datetime(2026, 8, 28, 9, 32, tzinfo=IST)) is None
        assert scheduler.last_fired == datetime(2026, 8, 28, 9, 30, tzinfo=IST)

    def test_slots_before_the_arm_time_never_fire(self) -> None:
        """A process booting at 12:00 waits for 13:30 — it does not burst-catch-up on
        09:30 and 11:30."""
        scheduler = RotationScheduler(clock=_clock_at(12, 0), arm_now=True)
        assert scheduler.advance(datetime(2026, 8, 28, 12, 1, tzinfo=IST)) is None
        assert scheduler.advance(datetime(2026, 8, 28, 13, 31, tzinfo=IST)) == datetime(
            2026, 8, 28, 13, 30, tzinfo=IST
        )

    def test_a_blocked_loop_catches_up_on_the_latest_missed_slot_only(self) -> None:
        scheduler = RotationScheduler(clock=_clock_at(9, 0))
        assert scheduler.advance(datetime(2026, 8, 28, 9, 31, tzinfo=IST)) is not None
        # The loop was blocked through 11:30 and 13:30; only the latest fires.
        assert scheduler.advance(datetime(2026, 8, 28, 14, 0, tzinfo=IST)) == datetime(
            2026, 8, 28, 13, 30, tzinfo=IST
        )
        assert scheduler.advance(datetime(2026, 8, 28, 14, 1, tzinfo=IST)) is None

    def test_nothing_fires_before_the_first_slot(self) -> None:
        scheduler = RotationScheduler(clock=_clock_at(8, 0))
        assert scheduler.advance(datetime(2026, 8, 28, 9, 29, tzinfo=IST)) is None


class TestPlanRotation:
    """The pure swap planner: who may leave, who may enter, and in what order."""

    _WATCHED: dict[str, float] = {"AAA": 2.0, "BBB": -float("inf"), "CCC": 0.5}

    def test_a_symbol_no_longer_a_mover_is_the_first_to_go(self) -> None:
        """BBB scored -inf: it fell out of the rescan entirely."""
        plan = plan_rotation(
            watched_scores=self._WATCHED,
            rotatable={"AAA", "BBB", "CCC"},
            candidates=[("NEW", 3.0)],
        )
        assert [(s.out_symbol, s.in_symbol) for s in plan.swaps] == [("BBB", "NEW")]

    def test_a_symbol_with_an_open_position_is_never_rotated_out(self) -> None:
        """The subscription carries the ticks the exit depends on."""
        plan = plan_rotation(
            watched_scores={"BBB": -float("inf")},
            rotatable=set(),  # BBB is LONG — not rotatable
            candidates=[("NEW", 3.0)],
        )
        assert plan.is_empty

    def test_a_candidate_already_watched_is_not_swapped_in(self) -> None:
        plan = plan_rotation(
            watched_scores=self._WATCHED,
            rotatable={"AAA", "BBB", "CCC"},
            candidates=[("AAA", 5.0), ("NEW", 3.0)],
        )
        assert [(s.out_symbol, s.in_symbol) for s in plan.swaps] == [("BBB", "NEW")]

    def test_the_incoming_symbol_must_clear_min_score(self) -> None:
        plan = plan_rotation(
            watched_scores=self._WATCHED,
            rotatable={"AAA", "BBB", "CCC"},
            candidates=[("WEAK", 0.5)],
            min_score=1.0,
        )
        assert plan.is_empty

    def test_the_incoming_symbol_must_strictly_beat_the_one_it_displaces(self) -> None:
        """A swap between equals is churn, not rotation."""
        plan = plan_rotation(
            watched_scores={"AAA": 3.0},
            rotatable={"AAA"},
            candidates=[("TWIN", 3.0)],
        )
        assert plan.is_empty

    def test_max_swaps_bounds_the_cycle(self) -> None:
        plan = plan_rotation(
            watched_scores={"A": -float("inf"), "B": -float("inf")},
            rotatable={"A", "B"},
            candidates=[("X", 3.0), ("Y", 2.0)],
            max_swaps=1,
        )
        assert [(s.out_symbol, s.in_symbol) for s in plan.swaps] == [("A", "X")]

    def test_swaps_pair_best_candidate_with_weakest_watched(self) -> None:
        plan = plan_rotation(
            watched_scores={"STRONG": 2.0, "WEAK": -1.0},
            rotatable={"STRONG", "WEAK"},
            candidates=[("BEST", 5.0), ("NEXT", 4.0)],
            max_swaps=2,
        )
        assert [(s.out_symbol, s.in_symbol) for s in plan.swaps] == [
            ("WEAK", "BEST"),
            ("STRONG", "NEXT"),
        ]

    def test_ties_break_on_symbol_so_a_plan_is_reproducible(self) -> None:
        plan = plan_rotation(
            watched_scores={"ZZZ": 0.0, "AAA": 0.0},
            rotatable={"ZZZ", "AAA"},
            candidates=[("MMM", 2.0)],
        )
        assert plan.swaps[0].out_symbol == "AAA"

    def test_a_non_positive_swap_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_swaps must be at least 1"):
            plan_rotation(
                watched_scores=self._WATCHED, rotatable={"AAA"}, candidates=[], max_swaps=0
            )


class TestIntradayScanner:
    """The live-rescan wrapper: dynamic floor and composite ranking by construction."""

    @staticmethod
    def _live_board() -> tuple[PreOpenQuote, ...]:
        return tuple(
            PreOpenQuote(
                symbol=f"M{i:02d}",
                previous_close=Decimal("1000"),
                pre_open_price=Decimal(1000 + i * 10),
                turnover=Decimal(i * 2_000_000),
                series=EQUITY_SERIES,
                volume=Decimal(i * 100_000),
            )
            for i in range(1, 13)
        )

    async def test_it_defaults_to_the_dynamic_floor(self) -> None:
        board = self._live_board()
        master = {q.symbol: _record(q.symbol, str(i)) for i, q in enumerate(board)}
        source = _StaticSource(board)
        scanner = IntradayScanner(source=source, resolver=_StaticResolver(master), size=12)

        result = await scanner.scan(budget=Decimal("50000"))

        assert source.calls == 1
        assert scanner.scanner.turnover_percentile == DEFAULT_TURNOVER_PERCENTILE
        assert result.floor_source in {"percentile", "backstop"}
        assert result.selected[0].symbol == "M12"  # strongest mover and deepest liquidity

    async def test_a_failed_scan_raises_so_the_caller_keeps_the_watchlist(self) -> None:
        scanner = IntradayScanner(
            source=_StaticSource(error=PreOpenFetchError("screener down")),
            resolver=_StaticResolver({}),
        )
        with pytest.raises(PreOpenFetchError):
            await scanner.scan(budget=Decimal("50000"))
