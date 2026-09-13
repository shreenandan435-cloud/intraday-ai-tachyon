"""Tests for the trade lifecycle manager and its router integration.

The lifecycle math is small but it sizes every stop and decides every
bracket-order exit, so each branch — breakeven promotion, time-stop
stagnation, time-stop profit-choke — is checked against an independent
reference computed in the test. A test that merely re-runs the
implementation proves nothing about whether the formula is right.
"""

from __future__ import annotations

import math
import time

import pytest

from tachyon.execution.trade_manager import (
    ActiveTrade,
    BREAKEVEN_FEE_BPS,
    BREAKEVEN_R,
    ExitIntent,
    ExitKind,
    ExitReason,
    PROFIT_CHOKE_CUSHION_R,
    PROFIT_CHOKE_R,
    Side,
    TIME_STOP_SECONDS,
    TradeLifecycleManager,
    true_breakeven_price,
)


def _long_trade(
    *,
    trade_id: str = "t-1",
    symbol: str = "RELIANCE",
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    entry_time: float | None = None,
    quantity: int = 10,
) -> ActiveTrade:
    return ActiveTrade(
        trade_id=trade_id,
        symbol=symbol,
        side=Side.LONG,
        quantity=quantity,
        entry_price=entry,
        entry_time=entry_time if entry_time is not None else time.time(),
        initial_stop=stop,
        initial_target=target,
        current_stop=stop,
    )


def _short_trade(**kwargs: object) -> ActiveTrade:
    base = _long_trade(**kwargs)
    return ActiveTrade(
        trade_id=str(kwargs.get("trade_id", "t-1")),
        symbol=str(kwargs.get("symbol", "RELIANCE")),
        side=Side.SHORT,
        quantity=int(kwargs.get("quantity", 10)),
        entry_price=float(kwargs.get("entry", 100.0)),
        entry_time=float(kwargs.get("entry_time", kwargs.get("entry_time", time.time()))),
        initial_stop=float(kwargs.get("stop", 105.0)),
        initial_target=float(kwargs.get("target", 90.0)),
        current_stop=float(kwargs.get("stop", 105.0)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Initial Risk and Unrealized R
# ──────────────────────────────────────────────────────────────────────────────


class TestInitialRisk:
    def test_long_risk_is_entry_minus_stop(self) -> None:
        trade = _long_trade(entry=100.0, stop=95.0)
        assert trade.initial_risk == pytest.approx(5.0)

    def test_short_risk_is_stop_minus_entry(self) -> None:
        trade = _short_trade(entry=100.0, stop=105.0)
        assert trade.initial_risk == pytest.approx(5.0)


class TestUnrealizedR:
    def test_long_in_profit_is_positive(self) -> None:
        trade = _long_trade(entry=100.0, stop=95.0)
        # LTP=104, gain=4, risk=5 → 0.8R
        assert trade.unrealized_r(104.0) == pytest.approx(0.8)

    def test_long_at_entry_is_zero(self) -> None:
        trade = _long_trade(entry=100.0, stop=95.0)
        assert trade.unrealized_r(100.0) == 0.0

    def test_long_in_loss_is_negative(self) -> None:
        trade = _long_trade(entry=100.0, stop=95.0)
        # LTP=98, loss=2, risk=5 → -0.4R
        assert trade.unrealized_r(98.0) == pytest.approx(-0.4)

    def test_short_in_profit_is_positive(self) -> None:
        # entry=100, stop=105 (above entry), LTP=96 → fall of 4, risk=5 → 0.8R
        trade = _short_trade(entry=100.0, stop=105.0)
        assert trade.unrealized_r(96.0) == pytest.approx(0.8)

    def test_short_in_loss_is_negative(self) -> None:
        trade = _short_trade(entry=100.0, stop=105.0)
        # LTP=102, rise of 2, risk=5 → -0.4R
        assert trade.unrealized_r(102.0) == pytest.approx(-0.4)

    def test_degenerate_zero_risk_is_nan(self) -> None:
        trade = _long_trade(entry=100.0, stop=100.0)
        assert math.isnan(trade.unrealized_r(100.0))


# ──────────────────────────────────────────────────────────────────────────────
# True breakeven
# ──────────────────────────────────────────────────────────────────────────────


class TestTrueBreakeven:
    def test_long_breakeven_adds_fee_buffer(self) -> None:
        be = true_breakeven_price(100.0, Side.LONG)
        assert be == pytest.approx(100.0 * (1.0 + BREAKEVEN_FEE_BPS / 10_000.0))

    def test_short_breakeven_subtracts_fee_buffer(self) -> None:
        be = true_breakeven_price(100.0, Side.SHORT)
        assert be == pytest.approx(100.0 * (1.0 - BREAKEVEN_FEE_BPS / 10_000.0))

    def test_breakeven_trigger_at_threshold(self) -> None:
        """0.7R long → LTP = 100 + 0.7×5 = 103.5. Just above fires breakeven."""
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0)
        manager.register(trade)
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.6})
        assert len(intents) == 1
        intent = intents[0]
        assert intent.kind is ExitKind.STOP_UPDATE
        assert intent.reason is ExitReason.BREAKEVEN_PROMOTED
        assert intent.new_stop == pytest.approx(100.0 * (1.0 + BREAKEVEN_FEE_BPS / 10_000.0))

    def test_breakeven_below_threshold_no_op(self) -> None:
        """0.69R is below the 0.7R threshold → no breakeven promotion."""
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0)
        manager.register(trade)
        # LTP=103.45 → r = 0.69
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.45})
        # Below 0.7R and below 45 minutes → no intent.
        assert intents == []

    def test_breakeven_never_widens(self) -> None:
        """If the current stop is *already* past breakeven, the new stop must
        not move to a worse (lower) level than the current stop."""
        manager = TradeLifecycleManager()
        # Trade with current_stop already above the breakeven level.
        trade = _long_trade(entry=100.0, stop=95.0)
        # Push the current_stop to 102 (already above the BE of 100.03).
        manager.register(trade.with_stop(102.0))
        # 0.8R → LTP=104 → breakeven would normally fire at 100.03. The
        # tightening rule for a long is max(102, 100.03) = 102, which equals
        # the current stop. No change → no intent.
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 104.0})
        assert intents == []

    def test_short_breakeven_promotion(self) -> None:
        manager = TradeLifecycleManager()
        trade = _short_trade(entry=100.0, stop=105.0)
        manager.register(trade)
        # 0.8R short → LTP = 100 - 0.8×5 = 96
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 96.0})
        assert len(intents) == 1
        expected_be = 100.0 * (1.0 - BREAKEVEN_FEE_BPS / 10_000.0)
        assert intents[0].new_stop == pytest.approx(expected_be)

    def test_breakeven_flip_is_idempotent(self) -> None:
        """Once breakeven is on, a subsequent tick at the same R does not
        emit a redundant intent (the current stop is already there)."""
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0)
        manager.register(trade)
        # First tick: breakeven promoted.
        intents1 = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.6})
        manager.apply(intents1)
        # Second tick at the same LTP — the current stop is already at BE,
        # so tightening it to BE again produces a no-op (intent unchanged).
        intents2 = manager.evaluate_ticks(time.time() + 1, {"RELIANCE": 103.6})
        assert intents2 == []


# ──────────────────────────────────────────────────────────────────────────────
# Time-stop — stagnant branch
# ──────────────────────────────────────────────────────────────────────────────


class TestTimeStopStagnant:
    def test_no_exit_before_45_minutes(self) -> None:
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - 1000)
        manager.register(trade)
        # Even with negative R, the 45-minute window has not elapsed.
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 90.0})
        assert intents == []

    def test_stagnant_exit_at_45_minutes(self) -> None:
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # r = (101-100)/5 = 0.2 < 0.5 → TIME_STOP_STAGNANT
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 101.0})
        assert len(intents) == 1
        assert intents[0].kind is ExitKind.MARKET_EXIT
        assert intents[0].reason is ExitReason.TIME_STOP_STAGNANT

    def test_short_stagnant_exit(self) -> None:
        manager = TradeLifecycleManager()
        trade = _short_trade(entry=100.0, stop=105.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # r_short = (100-99)/5 = 0.2 < 0.5 → TIME_STOP_STAGNANT
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 99.0})
        assert len(intents) == 1
        assert intents[0].kind is ExitKind.MARKET_EXIT
        assert intents[0].reason is ExitReason.TIME_STOP_STAGNANT

    def test_stagnant_threshold_is_half_r(self) -> None:
        """r == 0.5 is *at* the threshold; the time-stop branch only fires
        when r < 0.5. A trade that has produced exactly 0.5R survives."""
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # r = 0.5 → LTP = 100 + 0.5*5 = 102.5
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 102.5})
        # 0.5 is not < 0.5, so the stagnant branch does not fire. The profit-
        # choke branch *will* fire (r >= 0.5).
        assert any(i.reason is ExitReason.TIME_STOP_STAGNANT for i in intents) is False
        assert any(i.reason is ExitReason.TIME_STOP_PROFIT_CHOKE for i in intents) is True


# ──────────────────────────────────────────────────────────────────────────────
# Time-stop — profit-choke branch
# ──────────────────────────────────────────────────────────────────────────────


class TestTimeStopProfitChoke:
    def test_profitable_trade_is_choked_not_killed(self) -> None:
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # r = 0.6 → LTP = 100 + 0.6*5 = 103. Above 0.5R, below 0.7R breakeven
        # threshold → the only intent is the profit-choke.
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.0})
        assert len(intents) == 1
        intent = intents[0]
        assert intent.kind is ExitKind.STOP_UPDATE
        assert intent.reason is ExitReason.TIME_STOP_PROFIT_CHOKE
        # Choke stop = LTP - 0.2R = 103 - 1.0 = 102
        assert intent.new_stop == pytest.approx(102.0)

    def test_short_profit_choke(self) -> None:
        manager = TradeLifecycleManager()
        trade = _short_trade(entry=100.0, stop=105.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # 0.6R short → LTP = 100 - 0.6*5 = 97. Below BE threshold, above 0.5R.
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 97.0})
        assert len(intents) == 1
        assert intents[0].reason is ExitReason.TIME_STOP_PROFIT_CHOKE
        # Choke stop = LTP + 0.2R = 97 + 1.0 = 98
        assert intents[0].new_stop == pytest.approx(98.0)

    def test_choke_never_widens(self) -> None:
        """If the current stop is already past the choke level, the choke
        must not widen the stop away from the entry."""
        manager = TradeLifecycleManager()
        # Trade with the stop already at 99 (well above the choke at 102).
        # r=0.6R → choke = 102. max(99, 102) = 102 → tighten. 
        # Wait, the test should set current_stop to ABOVE the choke level.
        # LTP=103, choke=102, current=104 → max(104, 102) = 104 → no change.
        base = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(base.with_stop(104.0))
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.0})
        # Choke at 102 < current 104 → no tightening → no intent.
        assert intents == []

    def test_high_r_trade_emits_breakeven_then_choke(self) -> None:
        """At r=0.8 with elapsed>45min, both breakeven and profit-choke fire.

        Breakeven is evaluated first; on a subsequent apply the trade's
        is_breakeven_active flips True and a fresh evaluation runs the
        time-stop branch. The spec is consistent: each rule fires when its
        own condition is met, and the order in which they fire is the
        order in the function.
        """
        manager = TradeLifecycleManager()
        trade = _long_trade(entry=100.0, stop=95.0, entry_time=time.time() - TIME_STOP_SECONDS)
        manager.register(trade)
        # r=0.8 → LTP=104. Breakeven fires first (r>=0.7).
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 104.0})
        assert len(intents) >= 1
        assert intents[0].reason is ExitReason.BREAKEVEN_PROMOTED
        assert intents[0].new_stop == pytest.approx(100.03)


# ──────────────────────────────────────────────────────────────────────────────
# Apply / deregister
# ──────────────────────────────────────────────────────────────────────────────


class TestApply:
    def test_stop_update_persists_to_active_set(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade())
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 103.6})
        manager.apply(intents)
        active = manager.get("t-1")
        assert active is not None
        assert active.is_breakeven_active is True
        assert active.current_stop == pytest.approx(100.03)

    def test_market_exit_deregisters(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade(entry_time=time.time() - TIME_STOP_SECONDS))
        intents = manager.evaluate_ticks(time.time(), {"RELIANCE": 101.0})
        assert len(intents) == 1
        manager.apply(intents)
        assert manager.get("t-1") is None
        assert "t-1" not in manager

    def test_apply_ignores_stale_intent(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade())
        # Apply an intent for a trade that does not exist (already closed).
        intent = ExitIntent(
            trade_id="ghost",
            symbol="RELIANCE",
            side=Side.LONG,
            kind=ExitKind.STOP_UPDATE,
            reason=ExitReason.BREAKEVEN_PROMOTED,
            quantity=1,
            ltp_at_decision=103.0,
            new_stop=102.0,
        )
        # No exception raised.
        manager.apply([intent])

    def test_next_trade_id_is_unique(self) -> None:
        manager = TradeLifecycleManager()
        ids = {manager.next_trade_id("RELIANCE") for _ in range(100)}
        assert len(ids) == 100

    def test_get_by_symbol(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade(symbol="INFY", trade_id="i-1"))
        assert manager.get_by_symbol("INFY") is not None
        assert manager.get_by_symbol("RELIANCE") is None

    def test_deregister_returns_removed_trade(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade())
        removed = manager.deregister("t-1")
        assert removed is not None
        assert removed.trade_id == "t-1"
        # Removing twice returns None.
        assert manager.deregister("t-1") is None


# ──────────────────────────────────────────────────────────────────────────────
# Multi-trade evaluation
# ──────────────────────────────────────────────────────────────────────────────


class TestEvaluateTicks:
    def test_missing_symbol_is_skipped(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade())
        # Empty ltp dict → no action.
        assert manager.evaluate_ticks(time.time(), {}) == []

    def test_nan_ltp_is_skipped(self) -> None:
        manager = TradeLifecycleManager()
        manager.register(_long_trade())
        assert manager.evaluate_ticks(time.time(), {"RELIANCE": math.nan}) == []

    def test_multiple_trades_evaluated(self) -> None:
        manager = TradeLifecycleManager()
        # Two trades: one breakeven-ready, one below threshold.
        manager.register(_long_trade(trade_id="r-1", symbol="RELIANCE", entry=100.0, stop=95.0))
        manager.register(_long_trade(trade_id="i-1", symbol="INFY", entry=200.0, stop=190.0))
        intents = manager.evaluate_ticks(
            time.time(),
            {"RELIANCE": 104.0, "INFY": 192.0},
        )
        # Only RELIANCE has r >= 0.7 (0.8R); INFY at LTP=192 is r=−0.8 (loss).
        by_symbol = {i.symbol: i for i in intents}
        assert "RELIANCE" in by_symbol
        assert "INFY" not in by_symbol


# ──────────────────────────────────────────────────────────────────────────────
# Signal integrity
# ──────────────────────────────────────────────────────────────────────────────


class TestSignalIntegrity:
    def test_active_trade_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            ActiveTrade(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                quantity=1,
                entry_price=math.nan,
                entry_time=time.time(),
                initial_stop=95.0,
                initial_target=110.0,
                current_stop=95.0,
            )

    def test_active_trade_rejects_zero_entry(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ActiveTrade(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                quantity=1,
                entry_price=0.0,
                entry_time=time.time(),
                initial_stop=0.0,
                initial_target=110.0,
                current_stop=0.0,
            )

    def test_active_trade_rejects_zero_quantity(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ActiveTrade(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                quantity=0,
                entry_price=100.0,
                entry_time=time.time(),
                initial_stop=95.0,
                initial_target=110.0,
                current_stop=95.0,
            )

    def test_exit_intent_rejects_nan_ltp(self) -> None:
        with pytest.raises(ValueError, match="ltp"):
            ExitIntent(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                kind=ExitKind.MARKET_EXIT,
                reason=ExitReason.TIME_STOP_STAGNANT,
                quantity=1,
                ltp_at_decision=math.nan,
            )

    def test_stop_update_requires_finite_new_stop(self) -> None:
        with pytest.raises(ValueError, match="new_stop"):
            ExitIntent(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                kind=ExitKind.STOP_UPDATE,
                reason=ExitReason.BREAKEVEN_PROMOTED,
                quantity=1,
                ltp_at_decision=100.0,
                new_stop=math.nan,
            )

    def test_exit_intent_rejects_zero_quantity(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            ExitIntent(
                trade_id="t-1",
                symbol="RELIANCE",
                side=Side.LONG,
                kind=ExitKind.MARKET_EXIT,
                reason=ExitReason.TIME_STOP_STAGNANT,
                quantity=0,
                ltp_at_decision=100.0,
            )


class TestModuleConstants:
    def test_constants_match_spec(self) -> None:
        assert BREAKEVEN_R == pytest.approx(0.7)
        assert BREAKEVEN_FEE_BPS == pytest.approx(3.0)
        assert TIME_STOP_SECONDS == 45 * 60
        assert PROFIT_CHOKE_R == pytest.approx(0.5)
        assert PROFIT_CHOKE_CUSHION_R == pytest.approx(0.2)
