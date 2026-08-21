"""Dynamic session budget and the drawdown hard-stop.

These are the numbers that decide how much real money a bad day costs, so the arithmetic is
asserted against values worked out by hand rather than against the implementation's own
expression.

Everything here is in-process and allocation-free of the outside world: no sockets, no
network, no writes outside ``tmp_path``. The watchdog tests drive a :class:`ManualClock` and
never start the real thread except where explicitly noted.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.constants import DAILY_LOSS_LIMIT_INR, PER_TRADE_RISK_INR
from tachyon.core.state import StateMachine, TradingState
from tachyon.execution.builder import OrderBuilder, OrderRejected
from tachyon.risk.budget import CONCENTRATION_GUIDANCE, SessionBudget
from tachyon.risk.watchdog import SquareOffWatchdog

AT = datetime(2026, 8, 11, 11, 0, tzinfo=IST)


def _budget(capital: str, drawdown: str = "2.0", per_trade: str = "1.0") -> SessionBudget:
    return SessionBudget.resolve(
        capital=Decimal(capital),
        drawdown_pct=Decimal(drawdown),
        per_trade_pct=Decimal(per_trade),
    )


# ── derivation ───────────────────────────────────────────────────────────────


class TestDerivation:
    def test_worked_example_from_the_brief(self) -> None:
        """₹50,000 at 2 % daily / 1 % per trade. Hand-computed: ₹1,000 and ₹500."""
        budget = _budget("50000")
        assert budget.daily_loss_limit == Decimal("1000.00")
        assert budget.per_trade_risk == Decimal("500.00")
        assert budget.is_dynamic

    def test_derived_limit_can_exceed_the_constitutional_default(self) -> None:
        """The operator asked for this explicitly. It is honoured, not clamped — a silently
        capped limit is one the operator believes is in force and is not."""
        assert _budget("50000").daily_loss_limit > DAILY_LOSS_LIMIT_INR

    def test_fractional_percentage(self) -> None:
        budget = _budget("250000", drawdown="1.5", per_trade="0.25")
        assert budget.daily_loss_limit == Decimal("3750.00")
        assert budget.per_trade_risk == Decimal("625.00")

    def test_rounded_to_paise(self) -> None:
        budget = _budget("33333", drawdown="1.0", per_trade="0.333")
        assert budget.daily_loss_limit == Decimal("333.33")
        assert budget.per_trade_risk == Decimal("111.00")

    def test_drawdown_pct_round_trips(self) -> None:
        assert _budget("50000").drawdown_pct == Decimal("2.00")

    def test_float_input_does_not_leak_binary_error(self) -> None:
        """2.0 as a float must not become 2.00000000001 of the capital."""
        budget = SessionBudget.resolve(capital=50000, drawdown_pct=2.0, per_trade_pct=1.0)
        assert budget.daily_loss_limit == Decimal("1000.00")


# ── the fail-safe direction ──────────────────────────────────────────────────


class TestFallsBackConservatively:
    """Unknown budget must resolve to the *smaller* limit, never a larger one and never none."""

    def test_unset_capital_uses_the_constants(self) -> None:
        budget = _budget("0")
        assert not budget.is_dynamic
        assert budget.daily_loss_limit == DAILY_LOSS_LIMIT_INR
        assert budget.per_trade_risk == PER_TRADE_RISK_INR

    @pytest.mark.parametrize("capital", ["-1", "-50000"])
    def test_negative_capital_uses_the_constants(self, capital: str) -> None:
        assert _budget(capital).daily_loss_limit == DAILY_LOSS_LIMIT_INR

    @pytest.mark.parametrize(
        ("drawdown", "per_trade"),
        [("0", "1.0"), ("-2", "1.0"), ("2.0", "0"), ("2.0", "-1")],
    )
    def test_non_positive_percentages_use_the_constants(
        self, drawdown: str, per_trade: str
    ) -> None:
        budget = _budget("50000", drawdown=drawdown, per_trade=per_trade)
        assert not budget.is_dynamic
        assert budget.daily_loss_limit == DAILY_LOSS_LIMIT_INR

    @pytest.mark.parametrize("junk", ["", "abc", "NaN", "Infinity", "1,00,000"])
    def test_unparseable_capital_uses_the_constants(self, junk: str) -> None:
        budget = SessionBudget.resolve(capital=junk, drawdown_pct="2.0", per_trade_pct="1.0")
        assert not budget.is_dynamic
        assert budget.daily_loss_limit == DAILY_LOSS_LIMIT_INR

    def test_none_is_not_silently_unlimited(self) -> None:
        budget = SessionBudget.resolve(capital=None, drawdown_pct="2.0", per_trade_pct="1.0")  # type: ignore[arg-type]
        assert budget.daily_loss_limit == DAILY_LOSS_LIMIT_INR

    def test_capital_too_small_to_round_uses_the_constants(self) -> None:
        """₹0.10 at 1 % is ₹0.001 — below a paisa. A limit of zero would veto every trade."""
        budget = _budget("0.10", drawdown="1.0", per_trade="1.0")
        assert not budget.is_dynamic

    def test_static_budget_reports_zero_capital_not_a_fake_one(self) -> None:
        budget = SessionBudget.static()
        assert budget.capital == Decimal("0")
        assert budget.drawdown_pct == Decimal("0")


# ── concentration guidance ───────────────────────────────────────────────────


class TestConcentration:
    def test_brief_defaults_break_the_twenty_percent_shape(self) -> None:
        """1 % of capital per trade against a 2 % day means one loser spends half the budget.
        Honoured, but it is not §6.3's shape and the operator should know."""
        budget = _budget("50000")
        assert budget.concentration == Decimal("0.5")
        assert budget.exceeds_concentration_guidance

    def test_point_four_percent_restores_it(self) -> None:
        budget = _budget("50000", drawdown="2.0", per_trade="0.4")
        assert budget.concentration == Decimal("0.2")
        assert not budget.exceeds_concentration_guidance

    def test_the_constants_satisfy_their_own_guidance(self) -> None:
        """₹100 against ₹500 is exactly 20 % — the shape CLAUDE.md §6.3 describes."""
        budget = SessionBudget.static()
        assert budget.concentration == CONCENTRATION_GUIDANCE
        assert not budget.exceeds_concentration_guidance

    def test_a_zero_daily_limit_reports_zero_rather_than_dividing(self) -> None:
        """Unreachable through ``resolve`` — which is exactly why it is guarded.

        ``concentration`` divides by the daily limit. A future caller constructing a
        ``SessionBudget`` directly must get a number, not a ``DivisionByZero`` propagating out
        of a property the boot log reads.
        """
        budget = SessionBudget(
            capital=Decimal("0"),
            daily_loss_limit=Decimal("0"),
            per_trade_risk=Decimal("100"),
            is_dynamic=False,
            source="hand-built",
        )
        assert budget.concentration == Decimal("0")
        assert not budget.exceeds_concentration_guidance
        assert budget.drawdown_pct == Decimal("0")

    def test_log_summary_never_raises(self) -> None:
        _budget("50000").log_summary()
        SessionBudget.static().log_summary()


# ── sizing ───────────────────────────────────────────────────────────────────


class TestDynamicSizing:
    def test_builder_defaults_to_the_constant(self) -> None:
        assert OrderBuilder(settings=_settings()).per_trade_risk == PER_TRADE_RISK_INR

    def test_budget_uses_the_configured_per_trade_risk(self) -> None:
        builder = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        assert builder.budget(headroom=Decimal("1000")) == Decimal("500")

    def test_headroom_still_binds_below_the_per_trade_risk(self) -> None:
        """The daily limit stays the binding constraint late in a losing day."""
        builder = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        assert builder.budget(headroom=Decimal("70")) == Decimal("70")

    @pytest.mark.parametrize("bad", [Decimal("0"), Decimal("-100")])
    def test_non_positive_per_trade_risk_falls_back_not_to_zero(self, bad: Decimal) -> None:
        """A zero per-trade budget would veto every trade; falling back is the safe read of a
        misconfiguration, and it is the *smaller* of the two live options."""
        assert OrderBuilder(settings=_settings(), per_trade_risk=bad).per_trade_risk == (
            PER_TRADE_RISK_INR
        )

    def test_quantity_risks_the_configured_percentage_at_the_stop(self) -> None:
        """The requirement, stated directly: a stop-out costs ~1 % of ₹50,000 = ₹500."""
        builder = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        result = builder.size(risk_per_share=Decimal("12.50"), budget=Decimal("500"))
        assert result.quantity == 40
        assert result.risk_inr == Decimal("500.00")

    def test_quantity_floors_to_whole_lots(self) -> None:
        """₹500 / ₹12.50 = 40 shares, but a 15-lot instrument only permits 30."""
        builder = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        result = builder.size(risk_per_share=Decimal("12.50"), budget=Decimal("500"), lot_size=15)
        assert result.quantity == 30
        assert result.lots == 2
        assert result.risk_inr == Decimal("375.00"), "flooring risks less, never more"

    def test_never_rounds_up_to_reach_a_lot(self) -> None:
        builder = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        with pytest.raises(OrderRejected, match="QUANTITY_BELOW_ONE|below one lot"):
            builder.size(risk_per_share=Decimal("12.50"), budget=Decimal("500"), lot_size=50)

    def test_bigger_budget_buys_proportionally_more(self) -> None:
        small = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("100"))
        large = OrderBuilder(settings=_settings(), per_trade_risk=Decimal("500"))
        risk = Decimal("10")
        assert small.size(risk_per_share=risk, budget=small.budget(Decimal("1e9"))).quantity == 10
        assert large.size(risk_per_share=risk, budget=large.budget(Decimal("1e9"))).quantity == 50


# ── drawdown hard-stop ───────────────────────────────────────────────────────


class TestDrawdownHardStop:
    def _watchdog(
        self, net: Decimal, limit: Decimal, *, flatten: list[str] | None = None
    ) -> tuple[SquareOffWatchdog, list[Decimal]]:
        breaches: list[Decimal] = []
        machine = StateMachine(TradingState.ACTIVE)
        watchdog = SquareOffWatchdog(
            machine,
            clock=ManualClock(AT),
            on_date=date(2026, 8, 11),
            on_square_off=(lambda: flatten.append("flattened")) if flatten is not None else None,
            drawdown_probe=lambda: net,
            drawdown_limit=limit,
            on_drawdown_breach=breaches.append,
        )
        return watchdog, breaches

    def test_breach_fires_the_flatten_and_the_latch(self) -> None:
        flatten: list[str] = []
        watchdog, breaches = self._watchdog(Decimal("-1000"), Decimal("1000"), flatten=flatten)

        watchdog._check_drawdown()

        assert watchdog.drawdown_tripped
        assert breaches == [Decimal("-1000")]
        assert flatten == ["flattened"], "positions must actually be flattened, not just latched"

    def test_exactly_at_the_limit_breaches(self) -> None:
        """`<= -limit`, not `< -limit`: at ₹1,000 lost against a ₹1,000 limit, we are done."""
        watchdog, breaches = self._watchdog(Decimal("-1000"), Decimal("1000"))
        watchdog._check_drawdown()
        assert breaches

    def test_one_paisa_short_does_not_breach(self) -> None:
        watchdog, breaches = self._watchdog(Decimal("-999.99"), Decimal("1000"))
        watchdog._check_drawdown()
        assert not breaches
        assert not watchdog.drawdown_tripped

    def test_profit_never_breaches(self) -> None:
        watchdog, breaches = self._watchdog(Decimal("2500"), Decimal("1000"))
        watchdog._check_drawdown()
        assert not breaches

    def test_fires_only_once(self) -> None:
        """A second market exit does not close a position twice — it reverses it, creating
        fresh naked risk (CLAUDE.md §6.5). The latch is what prevents that."""
        flatten: list[str] = []
        watchdog, breaches = self._watchdog(Decimal("-5000"), Decimal("1000"), flatten=flatten)

        for _ in range(10):
            watchdog._check_drawdown()

        assert len(breaches) == 1
        assert len(flatten) == 1

    def test_a_breach_without_a_latch_callback_still_flattens(self) -> None:
        """``on_drawdown_breach`` is optional; the flatten is not.

        A watchdog wired for the deadline but not the P&L latch must still get the account
        flat. Being flat is the point (§6.5); the latch is bookkeeping on top of it.
        """
        flatten: list[str] = []
        watchdog = SquareOffWatchdog(
            StateMachine(TradingState.ACTIVE),
            clock=ManualClock(AT),
            on_date=date(2026, 8, 11),
            on_square_off=lambda: flatten.append("flattened"),
            drawdown_probe=lambda: Decimal("-1500"),
            drawdown_limit=Decimal("1000"),
            on_drawdown_breach=None,
        )

        watchdog._check_drawdown()

        assert watchdog.drawdown_tripped
        assert flatten == ["flattened"]

    def test_a_raising_probe_does_not_flatten(self) -> None:
        """Unknown P&L is not a breach. Flattening a healthy session on a probe bug would be
        the fail-safe pointed the wrong way."""
        machine = StateMachine(TradingState.ACTIVE)
        breaches: list[Decimal] = []

        def explode() -> Decimal:
            raise RuntimeError("probe is broken")

        watchdog = SquareOffWatchdog(
            machine,
            clock=ManualClock(AT),
            on_date=date(2026, 8, 11),
            drawdown_probe=explode,
            drawdown_limit=Decimal("1000"),
            on_drawdown_breach=breaches.append,
        )
        watchdog._check_drawdown()

        assert not breaches
        assert not watchdog.drawdown_tripped
        assert watchdog.thread_errors == 1

    def test_nan_net_is_not_treated_as_a_breach_here(self) -> None:
        """§4 check 6 treats undefined P&L as breached on the *gate*. This thread only
        flattens on a number it can actually compare."""
        watchdog, breaches = self._watchdog(Decimal("NaN"), Decimal("1000"))
        watchdog._check_drawdown()
        assert not breaches

    def test_zero_limit_disables_the_check(self) -> None:
        watchdog, breaches = self._watchdog(Decimal("-99999"), Decimal("0"))
        watchdog._check_drawdown()
        assert not breaches
        assert watchdog.drawdown_limit == Decimal("0")

    def test_breach_action_failure_still_flattens(self) -> None:
        """Latch first, flatten second — but a failed latch must not cancel the flatten."""
        flatten: list[str] = []
        machine = StateMachine(TradingState.ACTIVE)

        def boom(_net: Decimal) -> None:
            raise RuntimeError("disk full")

        watchdog = SquareOffWatchdog(
            machine,
            clock=ManualClock(AT),
            on_date=date(2026, 8, 11),
            on_square_off=lambda: flatten.append("flattened"),
            drawdown_probe=lambda: Decimal("-2000"),
            drawdown_limit=Decimal("1000"),
            on_drawdown_breach=boom,
        )
        watchdog._check_drawdown()

        assert flatten == ["flattened"]
        assert watchdog.drawdown_tripped

    def test_default_watchdog_has_no_drawdown_check(self) -> None:
        """Existing callers keep their behaviour exactly."""
        watchdog = SquareOffWatchdog(StateMachine(TradingState.ACTIVE), clock=ManualClock(AT))
        assert watchdog.drawdown_limit == Decimal("0")
        watchdog._check_drawdown()
        assert not watchdog.drawdown_tripped


def _settings() -> object:
    from tachyon.core.config import Settings, WatchlistItem

    return Settings(watchlist=(WatchlistItem(symbol="RELIANCE", token="2885"),))
