"""Tests for the tearsheet math.

The tearsheet is the canonical summary that a portfolio manager reads
off the screen; every metric must reconcile to a single number on a
:class:`ClosedTrade`. The tests below build a tiny trade set by hand
and check each formula against an independent reference.
"""

from __future__ import annotations

import math

import pytest

from tachyon.backtest.dry_run import (
    ClosedTrade,
    DailyLedger,
    LedgerEntry,
    LedgerExit,
)
from tachyon.backtest.tearsheet import (
    Tearsheet,
    aggregate_ledgers,
    aggregate_stats,
    build_tearsheet,
)


def _trade(
    *,
    net_pnl: float,
    fees: float = 0.0,
    symbol: str = "RELIANCE",
    side: str = "LONG",
) -> ClosedTrade:
    """Hand-build a :class:`ClosedTrade` for a known net P&L.

    The fields that affect the tearsheet (entry_price, exit_price,
    fees) are derived from the requested ``net_pnl`` so the test
    can stay focused on the math.
    """
    entry = 100.0
    exit = 100.0 + (net_pnl + fees)
    if side == "SHORT":
        exit = 100.0 - (net_pnl + fees)
    return ClosedTrade(
        trade_id="t-1",
        symbol=symbol,
        side=side,
        quantity=1,
        entry_price=entry,
        entry_time=0.0,
        exit_price=exit,
        exit_time=1.0,
        exit_reason="TIME_STOP_STAGNANT",
        initial_stop=95.0,
        initial_target=110.0,
        initial_risk=5.0,
        is_breakeven_active=False,
        gross_pnl=net_pnl + fees,
        net_pnl=net_pnl,
        fees_paid=fees,
    )


def _ledger_with_trades(*trades: ClosedTrade) -> DailyLedger:
    return DailyLedger(
        symbol="RELIANCE",
        session_date="2026-08-10",
        closed_trades=tuple(trades),
        stats={},
    )


class TestBuildTearsheet:
    def test_empty_input_is_zeroed(self) -> None:
        t = build_tearsheet([])
        assert t.n_trades == 0
        assert t.net_pnl == 0.0
        assert t.win_rate == 0.0
        assert t.profit_factor == 0.0
        assert t.max_drawdown == 0.0

    def test_single_winner(self) -> None:
        t = build_tearsheet([_ledger_with_trades(_trade(net_pnl=10.0, fees=1.0))])
        assert t.n_trades == 1
        assert t.n_wins == 1
        assert t.n_losses == 0
        assert t.win_rate == 1.0
        assert t.gross_profit == 10.0
        assert t.gross_loss == 0.0
        assert t.net_pnl == 10.0
        assert t.total_fees == 1.0
        assert t.expectancy == 10.0
        # Profit factor is infinite when gross_loss == 0.
        assert math.isinf(t.profit_factor)
        # Equity curve is monotonically non-decreasing → no drawdown.
        assert t.max_drawdown == 0.0

    def test_single_loser(self) -> None:
        t = build_tearsheet([_ledger_with_trades(_trade(net_pnl=-5.0, fees=1.0))])
        assert t.n_trades == 1
        assert t.n_wins == 0
        assert t.n_losses == 1
        assert t.win_rate == 0.0
        assert t.gross_profit == 0.0
        assert t.gross_loss == -5.0
        assert t.net_pnl == -5.0
        # No winners → profit factor zero (not inf).
        assert t.profit_factor == 0.0
        assert t.expectancy == -5.0
        # Equity curve: -5.0 throughout, max drawdown = 5.
        assert t.max_drawdown == 5.0

    def test_mixed_trades(self) -> None:
        # 2 winners @ 10 each, 1 loser @ -5 → 3 trades, 67% win rate.
        trades = [
            _trade(net_pnl=10.0, fees=0.5),
            _trade(net_pnl=10.0, fees=0.5),
            _trade(net_pnl=-5.0, fees=0.5),
        ]
        t = build_tearsheet([_ledger_with_trades(*trades)])
        assert t.n_trades == 3
        assert t.n_wins == 2
        assert t.n_losses == 1
        assert t.win_rate == pytest.approx(2 / 3)
        assert t.gross_profit == pytest.approx(20.0)
        assert t.gross_loss == pytest.approx(-5.0)
        assert t.profit_factor == pytest.approx(4.0)  # 20 / 5
        assert t.total_fees == pytest.approx(1.5)
        assert t.net_pnl == pytest.approx(15.0)
        # Expectancy = (2/3) * 10 + (1/3) * (-5) = 5.0
        assert t.expectancy == pytest.approx(5.0)
        # Equity: 10, 20, 15. Peak = 20, drawdown at last = 5.0.
        assert t.max_drawdown == pytest.approx(5.0)

    def test_max_drawdown_tracks_longest_underwater_span(self) -> None:
        # Win, win, big loss, win, win. The biggest peak-to-trough
        # is from peak=20 down to 5 → 15 rupees; the drawdown lasts
        # from the loss to the last bar because the curve never
        # returns to the 20 peak — three bars underwater.
        trades = [
            _trade(net_pnl=10.0),
            _trade(net_pnl=10.0),
            _trade(net_pnl=-15.0),
            _trade(net_pnl=5.0),
            _trade(net_pnl=5.0),
        ]
        t = build_tearsheet([_ledger_with_trades(*trades)])
        assert t.max_drawdown == pytest.approx(15.0)
        assert t.max_drawdown_duration_bars == 3

    def test_drawdown_duration_counts_full_underwater_span(self) -> None:
        # Win, then three losses in a row, then recovery back to the
        # peak. Equity (prepended 0): 0, 10, 5, 0, -5, 5. Peak=10
        # set after the first win. Underwater for 4 bars (5, 0, -5,
        # 5 — recovery is the bar where equity returns to peak).
        # Drawdown = 15.0 (10 → -5).
        trades = [
            _trade(net_pnl=10.0),
            _trade(net_pnl=-5.0),
            _trade(net_pnl=-5.0),
            _trade(net_pnl=-5.0),
            _trade(net_pnl=10.0),  # recovery back to 10
        ]
        t = build_tearsheet([_ledger_with_trades(*trades)])
        assert t.max_drawdown == pytest.approx(15.0)
        assert t.max_drawdown_duration_bars == 4

    def test_unrecovered_drawdown_uses_end_as_recovery(self) -> None:
        # Win, then 4 losses. Equity: 0, 10, 8, 6, 4, 2. Peak=10 at
        # bar 1. Underwater for 4 bars (the four losses).
        trades = [
            _trade(net_pnl=10.0),
            _trade(net_pnl=-2.0),
            _trade(net_pnl=-2.0),
            _trade(net_pnl=-2.0),
            _trade(net_pnl=-2.0),
        ]
        t = build_tearsheet([_ledger_with_trades(*trades)])
        assert t.max_drawdown == pytest.approx(8.0)
        assert t.max_drawdown_duration_bars == 4

    def test_aggregates_across_ledgers(self) -> None:
        ledger_a = _ledger_with_trades(_trade(net_pnl=5.0), _trade(net_pnl=-2.0))
        ledger_b = _ledger_with_trades(_trade(net_pnl=8.0))
        t = build_tearsheet([ledger_a, ledger_b])
        assert t.n_trades == 3
        assert t.net_pnl == pytest.approx(11.0)


class TestAggregateLedgers:
    def test_concatenates_preserving_order(self) -> None:
        ledger_a = _ledger_with_trades(_trade(net_pnl=1.0))
        ledger_b = _ledger_with_trades(_trade(net_pnl=2.0))
        out = aggregate_ledgers([ledger_a, ledger_b])
        assert [t.net_pnl for t in out] == [1.0, 2.0]

    def test_empty_input(self) -> None:
        assert aggregate_ledgers([]) == ()


class TestAggregateStats:
    def test_sums_per_day_counters(self) -> None:
        ledger_a = DailyLedger(
            symbol="A",
            session_date="2026-08-10",
            stats={"ticks_consumed": 100, "entries": 1},
        )
        ledger_b = DailyLedger(
            symbol="B",
            session_date="2026-08-10",
            stats={"ticks_consumed": 200, "entries": 2, "exits": 3},
        )
        merged = aggregate_stats([ledger_a, ledger_b])
        assert merged["ticks_consumed"] == 300
        assert merged["entries"] == 3
        assert merged["exits"] == 3

    def test_empty_input(self) -> None:
        assert aggregate_stats([]) == {}


class TestRender:
    def test_render_contains_every_metric(self) -> None:
        t = build_tearsheet(
            [_ledger_with_trades(_trade(net_pnl=10.0), _trade(net_pnl=-5.0))]
        )
        text = t.render()
        for label in (
            "Trades",
            "Wins / Losses",
            "Win Rate",
            "Avg Win",
            "Avg Loss",
            "Expectancy",
            "Gross Profit",
            "Gross Loss",
            "Profit Factor",
            "Total Fees",
            "Net PnL",
            "Max Drawdown",
        ):
            assert label in text

    def test_to_dict_is_jsonable(self) -> None:
        import json

        t = build_tearsheet([_ledger_with_trades(_trade(net_pnl=1.0))])
        # json.dumps will raise if any field is not serialisable.
        json.dumps(t.to_dict())
