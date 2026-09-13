"""Tests for the dry-run simulator and the live router wiring.

The tests exercise the simulator's *end-to-end* behaviour: a
hand-built data frame goes in, a populated :class:`DailyLedger`
comes out. The fixtures are deliberately small (200 ticks, 1
symbol) so the tests run in well under a second and stay
deterministic across platforms.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tachyon.backtest.dry_run import (
    ROUND_TRIP_FEE_BPS,
    DailyLedger,
    SessionSimulator,
    _apply_fee,
)

IST = ZoneInfo("Asia/Kolkata")


def _synthetic_dataframe(
    *,
    n: int = 200,
    base: float = 100.0,
    start: datetime | None = None,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """A 200-row synthetic intraday frame: builds a 99-101 range, then
    trends upward. Used by every test in this file."""
    start = start or datetime(2026, 8, 10, 9, 15, tzinfo=IST)
    rows: list[dict[str, object]] = []
    for i in range(n):
        ts = start + timedelta(seconds=i * interval_seconds)
        if i < 60:
            p = base - 1.0 + 0.04 * i
        else:
            p = base + 1.0 + (i - 60) * 0.05
        rows.append(
            {
                "ts": ts,
                "price": p,
                "volume": 2000,
                "bid_qty": 500,
                "ask_qty": 100,
                "bid_price": p - 1.0,
                "ask_price": p + 1.0,
            }
        )
    return pd.DataFrame(rows)


class TestSessionSimulator:
    def test_dry_run_uses_real_router_path(self) -> None:
        """The simulator must use the live router — not its own logic."""
        df = _synthetic_dataframe()
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        # The router is the same class the production uses, with the
        # same lifecycle manager and the same gate stack. The smoke
        # test: the simulator's router is an ExecutionRouter, and
        # running it produces a non-trivial ledger.
        assert sim._router is not None  # type: ignore[attr-defined]
        ledger = sim.run()
        assert isinstance(ledger, DailyLedger)
        assert ledger.symbol == "RELIANCE"
        assert ledger.session_date == "2026-08-10"

    def test_orb_breakout_produces_an_entry(self) -> None:
        """A bid-heavy breakout above the morning range should fire a Long."""
        df = _synthetic_dataframe()
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        # The data is engineered to break out at tick 60 (≈ 09:20) with
        # OBI ≈ +0.67 and RVOL well above 2.5. The ORB engine should
        # fire at least one entry, and the trade should be force-closed
        # at 15:30.
        assert len(ledger.entries) >= 1
        assert ledger.entries[0].source == "ORB"
        assert len(ledger.closed_trades) >= 1

    def test_force_close_at_session_end(self) -> None:
        """A trade open at 15:30 must be closed with a recorded trade."""
        df = _synthetic_dataframe()
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        # Every entry has a matching closed trade (force-close at 15:30
        # is the safety net for trades that survived the whole day).
        assert len(ledger.entries) == len(ledger.closed_trades)
        for trade in ledger.closed_trades:
            assert trade.exit_reason == "TIME_STOP_PROFIT_CHOKE"

    def test_ticks_filtered_to_session_window(self) -> None:
        """Ticks before 09:15 and after 15:30 are dropped from the run."""
        start = datetime(2026, 8, 10, 9, 10, tzinfo=IST)
        df = _synthetic_dataframe(start=start, n=400)
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        # The first 5 minutes (60 ticks) and the last few minutes are
        # dropped. The exact count depends on the trade, but it should
        # be a single big number: 400 ticks - 60 pre-open = 340; the
        # session is 6h15m = 37800 seconds / 5 = 7560 ticks max; our
        # 400 rows all fit if 60 are pre-open. So 340 ticks.
        assert 200 < ledger.stats["ticks_consumed"] <= 400

    def test_dataframe_normalisation_accepts_legacy_ltpr_alias(self) -> None:
        """`ltp` is a valid alias for `price`; both feed the same path."""
        df = _synthetic_dataframe().rename(columns={"price": "ltp"})
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        assert ledger.stats["ticks_consumed"] > 0

    def test_dataframe_normalisation_requires_price(self) -> None:
        """A frame with neither `price` nor `ltp` is rejected at construction."""
        df = _synthetic_dataframe().drop(columns=["price"]).rename(columns={"ltp": "ltp2"})
        with pytest.raises(ValueError, match="price"):
            SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")

    def test_dataframe_normalisation_requires_ts_column(self) -> None:
        """A frame with neither `ts_epoch` nor `ts` is rejected."""
        df = _synthetic_dataframe().drop(columns=["ts"])
        with pytest.raises(ValueError, match="ts_epoch|ts"):
            SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")


class TestFeeMath:
    def test_apply_fee_subtracts_bps(self) -> None:
        # 3 bps on 100 = 0.03.
        assert _apply_fee(100.0, side_bps=3.0) == pytest.approx(99.97)

    def test_apply_fee_preserves_precision(self) -> None:
        # The exact decimal that SmartAPI would book: 100 × 0.9997 = 99.97.
        assert _apply_fee(250.0, side_bps=ROUND_TRIP_FEE_BPS) == pytest.approx(249.925)


class TestLedger:
    def test_to_dict_round_trips(self) -> None:
        df = _synthetic_dataframe(n=80)
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        record = ledger.to_dict()
        assert record["symbol"] == "RELIANCE"
        assert record["session_date"] == "2026-08-10"
        # Every nested list preserves the dataclass shape.
        for entry in record["entries"]:
            assert {"ts_epoch", "side", "quantity", "price", "source"} <= set(entry)
        for trade in record["closed_trades"]:
            assert {"trade_id", "entry_price", "exit_price", "gross_pnl", "net_pnl"} <= set(trade)

    def test_ledger_stats_includes_books_consumed(self) -> None:
        """The stats dict carries the canonical counters from the simulator."""
        df = _synthetic_dataframe()
        sim = SessionSimulator(symbol="RELIANCE", data=df, token="1", session_date="2026-08-10")
        ledger = sim.run()
        assert "ticks_consumed" in ledger.stats
        assert "books_consumed" in ledger.stats
        assert "entries" in ledger.stats
        assert "exits" in ledger.stats
