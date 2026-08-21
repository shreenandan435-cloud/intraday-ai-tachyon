"""Trade persistence — tachyon.persistence.trade_logger.

Every test writes into ``tmp_path`` and drives the writer synchronously via
``drain_for_test``. Nothing here touches the network, a socket, or the real ``data/`` tree.
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.persistence.trade_logger import (
    TRADE_COLUMNS,
    TRIGGER_RISK_STOP,
    TRIGGER_UNKNOWN,
    TRIGGER_VWAP_CONFLUENCE,
    VETO_COLUMNS,
    SessionSummary,
    TradeLogger,
    session_summary_from_disk,
)
from tachyon.risk.engine import RiskDecision, VetoReason
from tachyon.ui.postback import OrderUpdate

AT = datetime(2026, 8, 11, 10, 30, tzinfo=IST)


@pytest.fixture
def logger(tmp_path: Path) -> TradeLogger:
    """A logger with no background thread — tests drain it themselves."""
    return TradeLogger(directory=tmp_path / "trades", clock=ManualClock(AT), start=False)


def _fill(
    *,
    order_id: str = "OID-1",
    symbol: str = "RELIANCE",
    side: str = "BUY",
    filled: int = 10,
    price: str = "1401.50",
    order_type: str = "MARKET",
) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id,
        symbol=symbol,
        status="complete",
        side=side,
        quantity=filled,
        filled_quantity=filled,
        average_price=Decimal(price),
        order_type=order_type,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ── the CSV appender ─────────────────────────────────────────────────────────


class TestTradeCsv:
    def test_directory_is_created_automatically(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "trades"
        assert not target.exists()
        TradeLogger(directory=target, clock=ManualClock(AT), start=False)
        assert target.is_dir()

    def test_fill_appends_a_row_with_every_required_column(self, logger: TradeLogger) -> None:
        logger.record_fill(_fill())
        logger.drain_for_test()

        path = logger._directory / "trades_2026-08-11.csv"
        rows = _rows(path)
        assert len(rows) == 1
        assert tuple(rows[0]) == TRADE_COLUMNS
        assert rows[0]["symbol"] == "RELIANCE"
        assert rows[0]["side"] == "BUY"
        assert rows[0]["quantity"] == "10"
        assert rows[0]["fill_price"] == "1401.50"
        assert rows[0]["order_id"] == "OID-1"
        assert rows[0]["timestamp_ist"].startswith("2026-08-11T10:30")

    def test_header_written_once_across_appends(self, logger: TradeLogger) -> None:
        logger.record_fill(_fill(order_id="A"))
        logger.drain_for_test()
        logger.record_fill(_fill(order_id="B"))
        logger.drain_for_test()

        path = logger._directory / "trades_2026-08-11.csv"
        assert path.read_text(encoding="utf-8").count("timestamp_ist") == 1
        assert len(_rows(path)) == 2

    def test_file_is_named_for_the_ist_date(self, tmp_path: Path) -> None:
        """23:30 IST on the 11th is the 11th, though it is still the 11th in UTC too — the
        point is that the date comes from the IST clock, not the host's locale."""
        logger = TradeLogger(
            directory=tmp_path,
            clock=ManualClock(datetime(2026, 8, 11, 23, 30, tzinfo=IST)),
            start=False,
        )
        logger.record_fill(_fill())
        logger.drain_for_test()
        assert (tmp_path / "trades_2026-08-11.csv").exists()

    def test_row_keeps_the_date_it_was_created_on(self, tmp_path: Path) -> None:
        """A row queued at 23:59:59 must not land in tomorrow's file because the writer was
        briefly behind."""
        clock = ManualClock(datetime(2026, 8, 11, 23, 59, 59, tzinfo=IST))
        logger = TradeLogger(directory=tmp_path, clock=clock, start=False)
        logger.record_fill(_fill())

        clock.advance(2.0)  # now 00:00:01 on the 12th
        assert clock.now().date().isoformat() == "2026-08-12"
        logger.drain_for_test()

        assert (tmp_path / "trades_2026-08-11.csv").exists()
        assert not (tmp_path / "trades_2026-08-12.csv").exists()

    def test_realized_pnl_blank_on_an_opening_fill(self, logger: TradeLogger) -> None:
        logger.record_fill(_fill())
        logger.drain_for_test()
        assert _rows(logger._directory / "trades_2026-08-11.csv")[0]["realized_pnl"] == ""

    def test_realized_pnl_present_on_the_closing_fill(self, logger: TradeLogger) -> None:
        logger.record_fill(_fill(order_id="EXIT", side="SELL"), Decimal("87.40"))
        logger.drain_for_test()
        assert _rows(logger._directory / "trades_2026-08-11.csv")[0]["realized_pnl"] == "87.40"

    def test_disabled_logger_writes_nothing(self, tmp_path: Path) -> None:
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT), enabled=False, start=False)
        logger.record_fill(_fill())
        logger.drain_for_test()
        assert list(tmp_path.glob("*.csv")) == []


# ── latency and trigger reason ───────────────────────────────────────────────


class TestLatencyAndTrigger:
    def test_latency_is_measured_from_placement(self, tmp_path: Path) -> None:
        clock = ManualClock(AT)
        logger = TradeLogger(directory=tmp_path, clock=clock, start=False)
        logger.note_order_placed("OID-1", trigger_reason=TRIGGER_VWAP_CONFLUENCE)

        clock.advance(0.325)
        logger.record_fill(_fill())
        logger.drain_for_test()

        row = _rows(tmp_path / "trades_2026-08-11.csv")[0]
        assert row["latency_ms"] == "325.0"
        assert row["trigger_reason"] == TRIGGER_VWAP_CONFLUENCE

    def test_latency_uses_monotonic_not_wall_clock(self, tmp_path: Path) -> None:
        """An NTP correction between placement and fill must not corrupt the measurement."""
        clock = ManualClock(AT)
        logger = TradeLogger(directory=tmp_path, clock=clock, start=False)
        logger.note_order_placed("OID-1")

        clock.advance(0.100)
        clock.jump_wall_clock(3600.0)  # the wall clock leaps an hour; monotonic does not
        logger.record_fill(_fill())
        logger.drain_for_test()

        assert _rows(tmp_path / "trades_2026-08-11.csv")[0]["latency_ms"] == "100.0"

    def test_latency_blank_when_placement_was_never_seen(self, logger: TradeLogger) -> None:
        """A restart, or a bracket leg the broker exits on its own. Blank is honest; 0 would
        read as an instant fill."""
        logger.record_fill(_fill())
        logger.drain_for_test()
        row = _rows(logger._directory / "trades_2026-08-11.csv")[0]
        assert row["latency_ms"] == ""
        assert row["trigger_reason"] == TRIGGER_UNKNOWN

    def test_stop_order_overrides_the_recorded_trigger(self, tmp_path: Path) -> None:
        """A stop-out is identified by the order that closed the position (CLAUDE.md §7.4)."""
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT), start=False)
        logger.note_order_placed("OID-1", trigger_reason=TRIGGER_VWAP_CONFLUENCE)
        logger.record_fill(_fill(order_type="STOPLOSS_MARKET"))
        logger.drain_for_test()
        assert _rows(tmp_path / "trades_2026-08-11.csv")[0]["trigger_reason"] == TRIGGER_RISK_STOP

    def test_intent_registry_is_bounded(self, logger: TradeLogger) -> None:
        from tachyon.persistence.trade_logger import MAX_TRACKED_ORDERS

        for index in range(MAX_TRACKED_ORDERS + 50):
            logger.note_order_placed(f"OID-{index}")
        assert len(logger._intents) == MAX_TRACKED_ORDERS
        assert "OID-0" not in logger._intents, "oldest intents are evicted first"

    def test_blank_order_id_is_ignored(self, logger: TradeLogger) -> None:
        logger.note_order_placed("")
        assert not logger._intents


# ── vetoes ───────────────────────────────────────────────────────────────────


class TestVetoCsv:
    def test_veto_goes_to_its_own_file(self, logger: TradeLogger) -> None:
        logger.record_veto(
            RiskDecision(
                allowed=False,
                symbol="INFY",
                at_ist=AT,
                reason=VetoReason.FEED_STALE,
                detail="no tick for 3.2s",
                failed_check="feed",
            )
        )
        logger.drain_for_test()

        rows = _rows(logger._directory / "vetoes_2026-08-11.csv")
        assert tuple(rows[0]) == VETO_COLUMNS
        assert rows[0]["symbol"] == "INFY"
        assert "FEED_STALE" in rows[0]["reason"]
        assert not (logger._directory / "trades_2026-08-11.csv").exists(), (
            "a veto is not a trade; mixing them corrupts every P&L sum over the trades file"
        )

    def test_allowed_decision_is_not_recorded(self, logger: TradeLogger) -> None:
        logger.record_veto(RiskDecision(allowed=True, symbol="INFY", at_ist=AT))
        logger.drain_for_test()
        assert not (logger._directory / "vetoes_2026-08-11.csv").exists()
        assert logger.summary.vetoes == 0


# ── session summary ──────────────────────────────────────────────────────────


class TestSummary:
    def test_aggregates_wins_losses_and_net(self, logger: TradeLogger) -> None:
        logger.record_close("RELIANCE", Decimal("100"), Decimal("15"), was_stop_out=False)
        logger.record_close("INFY", Decimal("-40"), Decimal("12"), was_stop_out=True)

        summary = logger.summary
        assert summary.total_trades == 2
        assert summary.wins == 1
        assert summary.losses == 1
        assert summary.stop_outs == 1
        assert summary.gross_pnl == Decimal("60")
        assert summary.total_charges == Decimal("27")
        assert summary.net_pnl == Decimal("33")
        assert summary.win_rate == 0.5

    def test_charges_can_turn_a_gross_win_into_a_net_loss(self, logger: TradeLogger) -> None:
        """The reason realized_pnl is net: on a ₹500 budget a ₹5 gross win that cost ₹18 to
        make is a loss, and a summary saying otherwise flatters the session."""
        logger.record_close("RELIANCE", Decimal("5"), Decimal("18"), was_stop_out=False)
        assert logger.summary.wins == 0
        assert logger.summary.losses == 1
        assert logger.summary.net_pnl == Decimal("-13")

    def test_scratch_excluded_from_win_rate_denominator(self, logger: TradeLogger) -> None:
        logger.record_close("A", Decimal("10"), Decimal("0"), was_stop_out=False)
        logger.record_close("B", Decimal("5"), Decimal("5"), was_stop_out=False)
        assert logger.summary.scratches == 1
        assert logger.summary.win_rate == 1.0

    def test_max_drawdown_is_peak_to_trough_on_the_net_curve(self) -> None:
        summary = SessionSummary()
        for net in (Decimal("100"), Decimal("-60"), Decimal("-30"), Decimal("50")):
            summary.book(net, Decimal("0"), was_stop_out=False)
        # equity: 100 → 40 → 10 → 60. Peak 100, trough 10.
        assert summary.net_pnl == Decimal("60")
        assert summary.max_drawdown == Decimal("90")

    def test_drawdown_is_zero_for_a_monotonically_rising_session(self) -> None:
        summary = SessionSummary()
        summary.book(Decimal("10"), Decimal("0"), was_stop_out=False)
        summary.book(Decimal("20"), Decimal("0"), was_stop_out=False)
        assert summary.max_drawdown == Decimal("0")

    def test_best_and_worst_trades_tracked(self, logger: TradeLogger) -> None:
        logger.record_close("A", Decimal("70"), Decimal("0"), was_stop_out=False)
        logger.record_close("B", Decimal("-25"), Decimal("0"), was_stop_out=True)
        assert logger.summary.best_trade == Decimal("70")
        assert logger.summary.worst_trade == Decimal("-25")

    def test_win_rate_of_an_empty_session_is_zero_not_a_crash(self) -> None:
        assert SessionSummary().win_rate == 0.0

    def test_summary_json_is_written(self, logger: TradeLogger) -> None:
        logger.record_close("RELIANCE", Decimal("100"), Decimal("15"), was_stop_out=False)
        path = logger.write_summary()

        assert path is not None
        assert path.name == "summary_2026-08-11.json"
        payload = session_summary_from_disk(path)
        assert payload["total_trades"] == 1
        assert payload["net_pnl"] == "85"
        assert payload["gross_pnl"] == "100"
        assert payload["total_charges"] == "15"
        assert payload["date"] == "2026-08-11"

    def test_summary_money_is_strings_not_floats(self, logger: TradeLogger) -> None:
        """Decimal survives the round trip; a float would reintroduce binary rounding into the
        record of what the ₹500 limit was enforced against."""
        logger.record_close("A", Decimal("0.10"), Decimal("0.20"), was_stop_out=False)
        path = logger.write_summary()
        assert path is not None
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["net_pnl"] == "-0.10"
        assert isinstance(raw["net_pnl"], str)

    def test_disabled_logger_writes_no_summary(self, tmp_path: Path) -> None:
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT), enabled=False, start=False)
        assert logger.write_summary() is None
        assert list(tmp_path.glob("*.json")) == []


# ── non-blocking behaviour ───────────────────────────────────────────────────


class TestNonBlocking:
    def test_record_fill_does_not_touch_the_disk(self, logger: TradeLogger) -> None:
        """The caller enqueues and returns; the file appears only once the writer runs."""
        logger.record_fill(_fill())
        assert not (logger._directory / "trades_2026-08-11.csv").exists()
        logger.drain_for_test()
        assert (logger._directory / "trades_2026-08-11.csv").exists()

    def test_full_queue_drops_rather_than_blocks(self, tmp_path: Path) -> None:
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT), queue_size=2, start=False)
        for index in range(6):
            logger.record_fill(_fill(order_id=f"OID-{index}"))

        assert logger.summary.rows_dropped == 4
        assert logger.drain_for_test() == 2

    def test_background_thread_writes_and_close_joins_it(self, tmp_path: Path) -> None:
        before = threading.active_count()
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT))
        logger.record_fill(_fill())
        logger.close()

        assert threading.active_count() == before, "the writer thread was not joined"
        assert len(_rows(tmp_path / "trades_2026-08-11.csv")) == 1
        assert (tmp_path / "summary_2026-08-11.json").exists()

    def test_writer_thread_is_a_daemon(self, tmp_path: Path) -> None:
        """Non-daemon would hang the process if close() were ever missed, and CPython joins
        non-daemon threads *before* atexit runs, so the usual safety net cannot fire."""
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT))
        try:
            assert logger._thread is not None
            assert logger._thread.daemon is True
        finally:
            logger.close()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        logger = TradeLogger(directory=tmp_path, clock=ManualClock(AT))
        logger.close()
        logger.close()

    def test_a_write_failure_never_reaches_the_caller(self, tmp_path: Path) -> None:
        """A full disk must not become an exception on the path that books P&L."""
        directory = tmp_path / "trades"
        directory.mkdir()
        # A directory where the CSV should be: every open() for append fails.
        (directory / "trades_2026-08-11.csv").mkdir()

        logger = TradeLogger(directory=directory, clock=ManualClock(AT), start=False)
        logger.record_fill(_fill())
        logger.drain_for_test()  # must not raise

        assert logger.summary.write_failures == 1

    def test_a_broken_update_never_reaches_the_caller(self, logger: TradeLogger) -> None:
        class Exploding:
            def __getattr__(self, name: str) -> object:
                raise RuntimeError("boom")

        logger.record_fill(Exploding())  # type: ignore[arg-type]
        assert logger.summary.total_fills == 0
