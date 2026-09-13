"""Phase 7 execution tests — CLAUDE.md §6, §9.

The bulk of this file is the :class:`~tachyon.execution.builder.OrderBuilder`, because that is
where the arithmetic that decides how much money is at risk actually lives. Two properties get
the most attention:

* **Tick quantisation** — every transmitted price must be a whole multiple of the instrument
  tick. A misaligned price is a guaranteed broker rejection, and the failure is silent until an
  order comes back refused with a position half-open.
* **R-multiple geometry** — the stop is 1.5 × ATR, T1 is 1.5 R and T2 is 2.5 R against that
  *rounded* stop. The ratios are asserted against the realised stop distance, not the raw ATR,
  because the realised distance is what the money is measured in.

The rest covers the parts of the execution layer that can lose money silently: offsets sent
where absolute prices belong, a stop moved the wrong way, a duplicate exit that reverses a
position, and a placement retried when its outcome is unknown.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import socket
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pyotp
import pytest

from tachyon.core import token_cache
from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import (
    ExecutionSettings,
    Settings,
    WatchlistItem,
)
from tachyon.core.constants import (
    PER_TRADE_RISK_INR,
    SL_ATR_MULTIPLIER,
    T1_RR,
    T2_RR,
    TradingMode,
)
from tachyon.core.state import StateMachine, TradingState
from tachyon.execution.api import (
    LOGIN,
    ORDER_BOOK,
    PLACE_ORDER,
    BrokerOrder,
    BrokerPosition,
    ClientIdentity,
    PaperModeError,
    SmartApiAuthError,
    SmartApiClient,
    SmartApiError,
    TokenBucket,
    UnknownOrderOutcomeError,
)
from tachyon.execution.builder import (
    Leg,
    OrderBuilder,
    OrderRejected,
    OrderTagSequencer,
    Side,
    clamp_multiplier,
    floor_div_decimal,
    round_to_tick,
    to_decimal,
    trading_symbol_for,
)
from tachyon.execution.executor import (
    MAX_OFFSET_FRACTION,
    NotFlatError,
    OffsetSanityError,
    OpenBracket,
    RoboExecutor,
    StopWidenedError,
    assert_offsets_sane,
    assert_stop_not_widened,
)
from tachyon.execution.journal import OrderJournal
from tachyon.execution.reconciliation import ReconcileOutcome, StateReconciler
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math_engine import warmup
from tachyon.risk.engine import RiskDecision, RiskEngine
from tachyon.risk.tracker import PnLTracker, PositionRegistry

TICK = Decimal("0.05")


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock(hh: int = 11, mm: int = 0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=1000.0)


def _settings(**overrides: Any) -> Settings:
    """A Settings instance built from init kwargs, bypassing .env and settings.yaml."""
    base: dict[str, Any] = {
        "watchlist": (
            WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),
            WatchlistItem(symbol="HDFCBANK", token="1333", exchange="NSE"),
            WatchlistItem(
                symbol="NIFTY26AUG25000CE",
                token="99926",
                exchange="NFO",
                tick_size=Decimal("0.05"),
                lot_size=75,
                trading_symbol="NIFTY26AUG25000CE",
            ),
        ),
        "execution": ExecutionSettings(),
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def settings() -> Settings:
    return _settings()


@pytest.fixture
def builder(settings: Settings) -> OrderBuilder:
    return OrderBuilder(settings=settings, clock=_clock())


@pytest.fixture
def journal(tmp_path: Path) -> OrderJournal:
    return OrderJournal(tmp_path / "journal", clock=_clock())


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────


class TestToDecimal:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.1, Decimal("0.1")),
            (2500, Decimal("2500")),
            ("8.35", Decimal("8.35")),
            (Decimal("1.005"), Decimal("1.005")),
        ],
    )
    def test_converts_without_binary_artefacts(self, value: Any, expected: Decimal) -> None:
        """Routed through str(), so 0.1 is 0.1 — not 0.1000000000000000055511151231257827."""
        assert to_decimal(value) == expected

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_undefined_floats(self, value: float) -> None:
        """A NaN ATR must never reach an order — every comparison against it is false."""
        with pytest.raises(OrderRejected) as exc:
            to_decimal(value, field="atr")
        assert exc.value.reason == "UNDEFINED_INPUT"

    def test_rejects_non_numeric(self) -> None:
        with pytest.raises(OrderRejected) as exc:
            to_decimal("not a price")
        assert exc.value.reason == "NON_NUMERIC"


class TestRoundToTick:
    @pytest.mark.parametrize(
        ("raw", "tick", "expected"),
        [
            # Exact multiples pass through unchanged.
            ("2500.00", "0.05", "2500.00"),
            ("2500.05", "0.05", "2500.05"),
            # Below the halfway point rounds down.
            ("2500.02", "0.05", "2500.00"),
            ("2500.024999", "0.05", "2500.00"),
            # Exactly halfway rounds up — ROUND_HALF_UP, mandated by CLAUDE.md §6.2.
            ("2500.025", "0.05", "2500.05"),
            ("0.025", "0.05", "0.05"),
            # Above halfway rounds up.
            ("2500.03", "0.05", "2500.05"),
            ("2500.049", "0.05", "2500.05"),
            # A classic float trap: 8.4 * 1.5 = 12.600000000000001 in binary.
            ("12.600000000000001", "0.05", "12.60"),
            # Other tick sizes.
            ("101.006", "0.01", "101.01"),
            ("101.004", "0.01", "101.00"),
            ("1234.4", "0.5", "1234.5"),
            ("1234.24", "0.5", "1234.00"),
            # Small numbers must not collapse to zero when they are above half a tick.
            ("0.03", "0.05", "0.05"),
            ("0.02", "0.05", "0.00"),
        ],
    )
    def test_quantisation(self, raw: str, tick: str, expected: str) -> None:
        assert round_to_tick(Decimal(raw), Decimal(tick)) == Decimal(expected)

    @pytest.mark.parametrize(
        ("raw", "tick"),
        [
            ("2500.02", "0.05"),
            ("2500.033", "0.05"),
            ("99.999", "0.05"),
            ("1.23456789", "0.05"),
            ("17.5", "0.25"),
            ("3.14159", "0.01"),
        ],
    )
    def test_result_is_always_a_whole_number_of_ticks(self, raw: str, tick: str) -> None:
        """The property that actually matters: no residue, at any scale."""
        tick_size = Decimal(tick)
        result = round_to_tick(Decimal(raw), tick_size)
        assert result % tick_size == 0
        assert abs(result - Decimal(raw)) <= tick_size / 2

    def test_preserves_the_tick_exponent_for_transmission(self) -> None:
        """`2500` and `2500.00` are equal Decimals but different strings on the wire."""
        assert str(round_to_tick(Decimal("2500"), TICK)) == "2500.00"

    @pytest.mark.parametrize("tick", ["0", "-0.05"])
    def test_rejects_non_positive_tick(self, tick: str) -> None:
        with pytest.raises(OrderRejected) as exc:
            round_to_tick(Decimal("100"), Decimal(tick))
        assert exc.value.reason == "BAD_TICK_SIZE"


class TestFloorAndClamp:
    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [
            ("100", "12.60", 7),  # 7.936... -> 7, never 8
            ("100", "10", 10),
            ("100", "100", 1),
            ("100", "100.05", 0),
            ("99.99", "10", 9),
        ],
    )
    def test_floor_never_rounds_up(self, numerator: str, denominator: str, expected: int) -> None:
        assert floor_div_decimal(Decimal(numerator), Decimal(denominator)) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.5, Decimal("0.5")),
            (1.0, Decimal("1")),
            (1.5, Decimal("1")),  # clamped down — the Sentinel may never upsize
            (99.0, Decimal("1")),
            (0.0, Decimal("0")),
            (-1.0, Decimal("0")),
            (float("nan"), Decimal("0")),
        ],
    )
    def test_sentinel_multiplier_is_clamped_in_our_code(
        self, raw: float, expected: Decimal
    ) -> None:
        assert clamp_multiplier(raw) == expected

    @pytest.mark.parametrize("denominator", ["0", "-1"])
    def test_floor_rejects_a_non_positive_divisor(self, denominator: str) -> None:
        with pytest.raises(OrderRejected) as exc:
            floor_div_decimal(Decimal("100"), Decimal(denominator))
        assert exc.value.reason == "BAD_DIVISOR"


class TestSideAndGeometryHelpers:
    def test_side_sign_and_opposite(self) -> None:
        assert Side.BUY.sign == Decimal("1")
        assert Side.SELL.sign == Decimal("-1")
        assert Side.BUY.opposite is Side.SELL
        assert Side.SELL.opposite is Side.BUY

    @pytest.mark.parametrize(
        ("side", "price", "target", "expected"),
        [
            (Side.BUY, "2519.00", "2518.90", True),
            (Side.BUY, "2518.90", "2518.90", True),  # exactly at target counts
            (Side.BUY, "2518.85", "2518.90", False),
            (Side.SELL, "2481.00", "2481.10", True),
            (Side.SELL, "2481.15", "2481.10", False),
        ],
    )
    def test_is_favourable_respects_direction(
        self, builder: OrderBuilder, side: Side, price: str, target: str, expected: bool
    ) -> None:
        geometry = builder.geometry(
            side=side, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        assert geometry.is_favourable(Decimal(price), Decimal(target)) is expected


class TestOrderTagSequencer:
    def test_format_and_monotonicity(self) -> None:
        seq = OrderTagSequencer(_clock())
        assert seq.next_tag() == "TCHYN-20260810-0001"
        assert seq.next_tag() == "TCHYN-20260810-0002"
        assert len(seq.next_tag()) <= 20  # Angel One caps ordertag at 20 characters
        assert seq.issued == 3

    def test_resets_across_the_ist_date_boundary(self) -> None:
        clock = _clock(hh=23, mm=59)
        seq = OrderTagSequencer(clock)
        assert seq.next_tag() == "TCHYN-20260810-0001"
        clock.advance(120)
        assert seq.next_tag() == "TCHYN-20260811-0001"

    def test_two_legs_never_share_a_tag(self, builder: OrderBuilder) -> None:
        """The tag is our idempotency handle; duplicates would confuse reconciliation."""
        plan = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("2.00"),
            headroom=Decimal("500"),
        )
        tags = [leg.order_tag for leg in plan.legs]
        assert len(tags) == len(set(tags)) == 2


class TestTradingSymbol:
    @pytest.mark.parametrize(
        ("symbol", "exchange", "explicit", "expected"),
        [
            ("RELIANCE", "NSE", None, "RELIANCE-EQ"),
            ("HDFCBANK", "BSE", None, "HDFCBANK-EQ"),
            ("RELIANCE-EQ", "NSE", None, "RELIANCE-EQ"),
            ("NIFTY26AUG25000CE", "NFO", None, "NIFTY26AUG25000CE"),
            ("RELIANCE", "NSE", "RELIANCE-BE", "RELIANCE-BE"),
        ],
    )
    def test_suffixing(
        self, symbol: str, exchange: str, explicit: str | None, expected: str
    ) -> None:
        item = WatchlistItem(
            symbol=symbol,
            token="1",
            exchange=exchange,  # type: ignore[arg-type]
            trading_symbol=explicit,
        )
        assert trading_symbol_for(item) == expected


# ──────────────────────────────────────────────────────────────────────────────
# Geometry — CLAUDE.md §6.1
# ──────────────────────────────────────────────────────────────────────────────


class TestGeometry:
    @pytest.mark.parametrize(
        ("atr", "expected_r"),
        [
            # R = round_to_tick(1.5 x ATR, 0.05)
            ("8.40", "12.60"),
            ("10.00", "15.00"),
            ("1.00", "1.50"),
            ("0.10", "0.15"),
            ("3.33", "5.00"),  # 4.995 -> 5.00
            ("7.777", "11.65"),  # 11.6655 -> 11.65
            ("0.04", "0.05"),  # 0.06 -> 0.05
        ],
    )
    def test_r_is_one_and_a_half_atr_tick_aligned(
        self, builder: OrderBuilder, atr: str, expected_r: str
    ) -> None:
        geometry = builder.geometry(
            side=Side.BUY, entry_price=Decimal("2500"), atr=Decimal(atr), tick_size=TICK
        )
        assert geometry.risk_per_share == Decimal(expected_r)
        assert geometry.risk_per_share == round_to_tick(SL_ATR_MULTIPLIER * Decimal(atr), TICK)

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    @pytest.mark.parametrize("atr", ["8.40", "2.00", "13.37", "0.55"])
    def test_r_multiples_hold_against_the_realised_stop(
        self, builder: OrderBuilder, side: Side, atr: str
    ) -> None:
        """T1 = 1.5R and T2 = 2.5R measured against the *rounded* stop distance.

        Rounding R first and then taking multiples of it is what keeps the reward/risk honest.
        Rounding ``1.5 x 1.5 x ATR`` independently would let the ratios drift by up to a tick
        in each direction against the distance we are actually filled at.
        """
        entry = Decimal("2500")
        geometry = builder.geometry(side=side, entry_price=entry, atr=Decimal(atr), tick_size=TICK)
        r = geometry.risk_per_share

        assert geometry.t1_offset == round_to_tick(T1_RR * r, TICK)
        assert geometry.t2_offset == round_to_tick(T2_RR * r, TICK)

        # And the same relationship expressed in absolute prices, with the sign convention.
        assert abs(geometry.entry_price - geometry.stop_loss_price) == r
        assert abs(geometry.target_1_price - entry) == geometry.t1_offset
        assert abs(geometry.target_2_price - entry) == geometry.t2_offset

    @pytest.mark.parametrize(
        ("side", "atr", "entry", "stop", "t1", "t2"),
        [
            # R = 12.60 on an 8.40 ATR.
            (Side.BUY, "8.40", "2500.00", "2487.40", "2518.90", "2531.50"),
            (Side.SELL, "8.40", "2500.00", "2512.60", "2481.10", "2468.50"),
            # R = 1.50 on a 1.00 ATR.
            (Side.BUY, "1.00", "1000.00", "998.50", "1002.25", "1003.75"),
            (Side.SELL, "1.00", "1000.00", "1001.50", "997.75", "996.25"),
        ],
    )
    def test_absolute_prices(
        self,
        builder: OrderBuilder,
        side: Side,
        atr: str,
        entry: str,
        stop: str,
        t1: str,
        t2: str,
    ) -> None:
        geometry = builder.geometry(
            side=side, entry_price=Decimal(entry), atr=Decimal(atr), tick_size=TICK
        )
        assert geometry.stop_loss_price == Decimal(stop)
        assert geometry.target_1_price == Decimal(t1)
        assert geometry.target_2_price == Decimal(t2)

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_stop_is_on_the_losing_side_and_targets_on_the_winning_side(
        self, builder: OrderBuilder, side: Side
    ) -> None:
        """A sign error here inverts the entire trade, so it is asserted directionally."""
        geometry = builder.geometry(
            side=side, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        if side is Side.BUY:
            assert geometry.stop_loss_price < geometry.entry_price
            assert geometry.entry_price < geometry.target_1_price < geometry.target_2_price
        else:
            assert geometry.stop_loss_price > geometry.entry_price
            assert geometry.entry_price > geometry.target_1_price > geometry.target_2_price

    @pytest.mark.parametrize("atr", ["8.40", "0.10", "31.416"])
    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_every_price_lands_on_a_tick(self, builder: OrderBuilder, atr: str, side: Side) -> None:
        geometry = builder.geometry(
            side=side, entry_price=Decimal("2499.97"), atr=Decimal(atr), tick_size=TICK
        )
        for price in (
            geometry.entry_price,
            geometry.stop_loss_price,
            geometry.target_1_price,
            geometry.target_2_price,
            geometry.trail_step,
            geometry.breakeven_stop(),
        ):
            assert price % TICK == 0, f"{price} is not a multiple of {TICK}"

    def test_trail_step_is_half_r_with_a_one_tick_floor(self, builder: OrderBuilder) -> None:
        geometry = builder.geometry(
            side=Side.BUY, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        assert geometry.trail_step == round_to_tick(Decimal("0.5") * Decimal("12.60"), TICK)

        tiny = builder.geometry(
            side=Side.BUY, entry_price=Decimal("100"), atr=Decimal("0.04"), tick_size=TICK
        )
        assert tiny.risk_per_share == TICK
        assert tiny.trail_step == TICK  # 0.5 x 0.05 = 0.025 -> floored to one tick, never zero

    def test_trail_step_never_rounds_away_to_zero(self) -> None:
        """A trail step of 0 would be sent to the broker as "no trail" without complaint."""
        narrow = OrderBuilder(
            settings=_settings(execution=ExecutionSettings(trail_step_r=Decimal("0.1"))),
            clock=_clock(),
        )
        geometry = narrow.geometry(
            side=Side.BUY, entry_price=Decimal("100"), atr=Decimal("0.03"), tick_size=TICK
        )
        assert geometry.risk_per_share == TICK
        assert geometry.trail_step == TICK  # 0.1 x 0.05 = 0.005 -> would quantise to 0.00

    @pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
    def test_breakeven_stop_is_one_tick_into_profit(
        self, builder: OrderBuilder, side: Side
    ) -> None:
        geometry = builder.geometry(
            side=side, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        expected = Decimal("2500.05") if side is Side.BUY else Decimal("2499.95")
        assert geometry.breakeven_stop() == expected

    @pytest.mark.parametrize(
        ("atr", "entry", "reason"),
        [
            ("0", "2500", "BAD_ATR"),
            ("-1", "2500", "BAD_ATR"),
            ("8.40", "0", "BAD_ENTRY_PRICE"),
            ("8.40", "-2500", "BAD_ENTRY_PRICE"),
            ("0.01", "2500", "RISK_BELOW_TICK"),  # 1.5 x 0.01 = 0.015 -> rounds to 0.00
        ],
    )
    def test_refuses_degenerate_inputs(
        self, builder: OrderBuilder, atr: str, entry: str, reason: str
    ) -> None:
        with pytest.raises(OrderRejected) as exc:
            builder.geometry(
                side=Side.BUY,
                entry_price=Decimal(entry),
                atr=Decimal(atr),
                tick_size=TICK,
            )
        assert exc.value.reason == reason

    def test_nan_atr_is_refused_rather_than_propagated(self, builder: OrderBuilder) -> None:
        with pytest.raises(OrderRejected) as exc:
            builder.geometry(
                side=Side.BUY, entry_price=Decimal("2500"), atr=float("nan"), tick_size=TICK
            )
        assert exc.value.reason == "UNDEFINED_INPUT"


# ──────────────────────────────────────────────────────────────────────────────
# Sizing — CLAUDE.md §6.3
# ──────────────────────────────────────────────────────────────────────────────


class TestSizing:
    @pytest.mark.parametrize(
        ("budget", "risk_per_share", "expected_qty"),
        [
            ("100", "12.60", 7),  # 7.93 -> 7
            ("100", "10.00", 10),  # exact
            ("100", "1.50", 66),  # 66.67 -> 66
            ("100", "0.05", 2000),
            ("100", "50.00", 2),
            ("100", "99.95", 1),
            ("70", "12.60", 5),  # reduced headroom sizes down
        ],
    )
    def test_quantity_is_floored(
        self, builder: OrderBuilder, budget: str, risk_per_share: str, expected_qty: int
    ) -> None:
        result = builder.size(
            risk_per_share=Decimal(risk_per_share), budget=Decimal(budget), lot_size=1
        )
        assert result.quantity == expected_qty

    @pytest.mark.parametrize("risk_per_share", ["100.05", "150.00", "501.00"])
    def test_no_trade_when_the_budget_buys_less_than_one(
        self, builder: OrderBuilder, risk_per_share: str
    ) -> None:
        """`qty < 1` means no trade. There is no "just one share" branch (CLAUDE.md §6.3)."""
        with pytest.raises(OrderRejected) as exc:
            builder.size(
                risk_per_share=Decimal(risk_per_share),
                budget=PER_TRADE_RISK_INR,
                lot_size=1,
            )
        assert exc.value.reason == "QUANTITY_BELOW_ONE"

    @pytest.mark.parametrize(
        ("budget", "risk_per_share", "lot_size", "expected_qty"),
        [
            ("100", "1.00", 75, 75),  # 100 shares -> 1 lot of 75
            ("200", "1.00", 75, 150),  # 200 -> 2 lots
            ("100", "2.00", 75, 0),  # 50 shares -> 0 lots -> rejected
        ],
    )
    def test_quantity_is_a_whole_number_of_lots(
        self,
        builder: OrderBuilder,
        budget: str,
        risk_per_share: str,
        lot_size: int,
        expected_qty: int,
    ) -> None:
        if expected_qty == 0:
            with pytest.raises(OrderRejected):
                builder.size(
                    risk_per_share=Decimal(risk_per_share),
                    budget=Decimal(budget),
                    lot_size=lot_size,
                )
            return
        result = builder.size(
            risk_per_share=Decimal(risk_per_share),
            budget=Decimal(budget),
            lot_size=lot_size,
        )
        assert result.quantity == expected_qty
        assert result.quantity % lot_size == 0

    @pytest.mark.parametrize(
        ("headroom", "multiplier", "expected"),
        [
            ("500", 1.0, "100"),  # capped by PER_TRADE_RISK_INR
            ("100", 1.0, "100"),
            ("70", 1.0, "70"),  # capped by remaining headroom
            ("500", 0.5, "50.0"),  # Sentinel downsizing
            ("500", 2.0, "100"),  # clamped: the Sentinel may never upsize
            ("500", 0.0, "0"),
            ("0", 1.0, "0"),
            ("-50", 1.0, "0"),
        ],
    )
    def test_budget_is_the_smaller_of_the_per_trade_cap_and_the_headroom(
        self, builder: OrderBuilder, headroom: str, multiplier: float, expected: str
    ) -> None:
        assert builder.budget(Decimal(headroom), multiplier) == Decimal(expected)

    @pytest.mark.parametrize("lot_size", [0, -1])
    def test_rejects_a_nonsensical_lot_size(self, builder: OrderBuilder, lot_size: int) -> None:
        with pytest.raises(OrderRejected) as exc:
            builder.size(risk_per_share=Decimal("10"), budget=Decimal("100"), lot_size=lot_size)
        assert exc.value.reason == "BAD_LOT_SIZE"

    def test_risk_never_exceeds_the_budget(self, builder: OrderBuilder) -> None:
        for r in ("0.05", "1.00", "12.60", "33.33", "99.95"):
            result = builder.size(risk_per_share=Decimal(r), budget=PER_TRADE_RISK_INR, lot_size=1)
            assert result.risk_inr <= PER_TRADE_RISK_INR


# ──────────────────────────────────────────────────────────────────────────────
# The 60/40 split — CLAUDE.md §6.1
# ──────────────────────────────────────────────────────────────────────────────


class TestSplit:
    @pytest.mark.parametrize(
        ("lots", "expected"),
        [
            (1, (1, 0)),  # indivisible: the whole position books at the nearer target
            (2, (1, 1)),
            (3, (2, 1)),
            (4, (2, 2)),
            (5, (3, 2)),
            (7, (4, 3)),
            (10, (6, 4)),
            (100, (60, 40)),
        ],
    )
    def test_lots_split_sixty_forty(
        self, builder: OrderBuilder, lots: int, expected: tuple[int, int]
    ) -> None:
        assert builder.split_lots(lots) == expected

    @pytest.mark.parametrize("lots", [0, -3])
    def test_no_lots_means_no_legs(self, builder: OrderBuilder, lots: int) -> None:
        assert builder.split_lots(lots) == (0, 0)

    @pytest.mark.parametrize("lots", list(range(1, 40)))
    def test_split_always_conserves_and_favours_leg_a(
        self, builder: OrderBuilder, lots: int
    ) -> None:
        a, b = builder.split_lots(lots)
        assert a + b == lots
        assert a >= b
        assert a >= 1

    def test_single_lot_produces_one_leg_at_t1(self, builder: OrderBuilder) -> None:
        plan = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("60.00"),  # R = 90.00 -> only 1 share affordable on a 100 budget
            headroom=Decimal("500"),
        )
        assert plan.total_quantity == 1
        assert len(plan.legs) == 1
        assert plan.legs[0].leg is Leg.A
        assert plan.legs[0].target_price == plan.geometry.target_1_price
        assert plan.leg(Leg.B) is None


# ──────────────────────────────────────────────────────────────────────────────
# Full plans
# ──────────────────────────────────────────────────────────────────────────────


class TestBuild:
    def test_end_to_end_geometry_and_split(self, builder: OrderBuilder) -> None:
        plan = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        # R = 12.60, budget = 100 -> floor(100 / 12.60) = 7 shares -> 4 / 3
        assert plan.total_quantity == 7
        assert [leg.quantity for leg in plan.legs] == [4, 3]

        leg_a, leg_b = plan.legs
        assert leg_a.squareoff_offset == Decimal("18.90")  # 1.5 R
        assert leg_b.squareoff_offset == Decimal("31.50")  # 2.5 R
        assert leg_a.stoploss_offset == leg_b.stoploss_offset == Decimal("12.60")
        assert leg_a.trailing_stop_loss == leg_b.trailing_stop_loss == Decimal("6.30")
        assert plan.total_risk_inr == Decimal("12.60") * 7
        assert plan.total_risk_inr <= PER_TRADE_RISK_INR

    def test_both_legs_carry_the_identical_stop(self, builder: OrderBuilder) -> None:
        """CLAUDE.md §6.1 — one position, one stop, two targets."""
        plan = builder.build(
            symbol="RELIANCE",
            side=Side.SELL,
            entry_price=Decimal("2500"),
            atr=Decimal("4.00"),
            headroom=Decimal("500"),
        )
        stops = {leg.stop_loss_price for leg in plan.legs}
        offsets = {leg.stoploss_offset for leg in plan.legs}
        assert len(stops) == len(offsets) == 1

    def test_derivative_legs_are_whole_lots(self, builder: OrderBuilder) -> None:
        plan = builder.build(
            symbol="NIFTY26AUG25000CE",
            side=Side.BUY,
            entry_price=Decimal("120.00"),
            atr=Decimal("0.40"),  # R = 0.60 -> 166 shares -> 2 lots of 75
            headroom=Decimal("500"),
        )
        assert plan.total_quantity == 150
        for leg in plan.legs:
            assert leg.quantity % 75 == 0
        assert plan.legs[0].trading_symbol == "NIFTY26AUG25000CE"

    def test_refuses_a_symbol_off_the_watchlist(self, builder: OrderBuilder) -> None:
        with pytest.raises(OrderRejected) as exc:
            builder.build(
                symbol="YESBANK",
                side=Side.BUY,
                entry_price=Decimal("20"),
                atr=Decimal("0.40"),
                headroom=Decimal("500"),
            )
        assert exc.value.reason == "SYMBOL_NOT_ALLOWED"

    def test_refuses_when_headroom_is_exhausted(self, builder: OrderBuilder) -> None:
        with pytest.raises(OrderRejected) as exc:
            builder.build(
                symbol="RELIANCE",
                side=Side.BUY,
                entry_price=Decimal("2500"),
                atr=Decimal("8.40"),
                headroom=Decimal("0"),
            )
        assert exc.value.reason == "NO_BUDGET"

    def test_sentinel_downsizing_reduces_quantity_and_never_raises_it(
        self, builder: OrderBuilder
    ) -> None:
        full = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        half = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
            sentinel_multiplier=0.5,
        )
        doubled = builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
            sentinel_multiplier=2.0,
        )
        assert half.total_quantity < full.total_quantity
        assert doubled.total_quantity == full.total_quantity


# ──────────────────────────────────────────────────────────────────────────────
# Executor invariants — CLAUDE.md §6.1, §6.2, §8.1
# ──────────────────────────────────────────────────────────────────────────────


class TestOffsetSanity:
    def test_accepts_ordinary_offsets(self) -> None:
        assert_offsets_sane(
            Decimal("2500"),
            {"stoploss": Decimal("12.60"), "squareoff": Decimal("31.50")},
        )

    @pytest.mark.parametrize("bad", ["0", "-1", "NaN", "Infinity"])
    def test_rejects_non_positive_or_undefined(self, bad: str) -> None:
        with pytest.raises(OffsetSanityError):
            assert_offsets_sane(Decimal("2500"), {"stoploss": Decimal(bad)})

    def test_rejects_an_absolute_price_sent_as_an_offset(self) -> None:
        """The single most expensive mistake available: the broker accepts it silently."""
        with pytest.raises(OffsetSanityError, match="absolute price"):
            assert_offsets_sane(Decimal("2500"), {"stoploss": Decimal("2487.40")})

    def test_boundary_of_the_ceiling(self) -> None:
        entry = Decimal("100")
        assert_offsets_sane(entry, {"stoploss": entry * MAX_OFFSET_FRACTION})
        with pytest.raises(OffsetSanityError):
            assert_offsets_sane(entry, {"stoploss": entry * MAX_OFFSET_FRACTION + Decimal("0.01")})


class TestStopDirection:
    @pytest.mark.parametrize(
        ("side", "current", "new"),
        [
            (Side.BUY, "2487.40", "2500.05"),  # long stop moves up = toward profit
            (Side.BUY, "2487.40", "2487.40"),  # no-op is permitted
            (Side.SELL, "2512.60", "2499.95"),  # short stop moves down
            (Side.SELL, "2512.60", "2512.60"),
        ],
    )
    def test_allows_movement_toward_profit(self, side: Side, current: str, new: str) -> None:
        assert_stop_not_widened(side, Decimal(current), Decimal(new))

    @pytest.mark.parametrize(
        ("side", "current", "new"),
        [
            (Side.BUY, "2487.40", "2480.00"),
            (Side.BUY, "2500.05", "2499.95"),
            (Side.SELL, "2512.60", "2520.00"),
            (Side.SELL, "2499.95", "2500.05"),
        ],
    )
    def test_refuses_any_widening(self, side: Side, current: str, new: str) -> None:
        """Stops move one direction only: toward profit (CLAUDE.md §8.1)."""
        with pytest.raises(StopWidenedError):
            assert_stop_not_widened(side, Decimal(current), Decimal(new))


# ──────────────────────────────────────────────────────────────────────────────
# Token bucket
# ──────────────────────────────────────────────────────────────────────────────


class TestTokenBucket:
    def test_starts_full_and_drains(self) -> None:
        clock = _clock()
        bucket = TokenBucket(5.0, clock=clock)
        assert all(bucket.try_acquire() for _ in range(5))
        assert not bucket.try_acquire()

    def test_refills_on_the_monotonic_clock(self) -> None:
        clock = _clock()
        bucket = TokenBucket(5.0, clock=clock)
        for _ in range(5):
            bucket.try_acquire()
        clock.advance(0.2)  # one token's worth at 5/s
        assert bucket.try_acquire()
        assert not bucket.try_acquire()

    def test_a_wall_clock_jump_hands_out_no_tokens(self) -> None:
        """An NTP correction must not turn into a burst against the broker's limit."""
        clock = _clock()
        bucket = TokenBucket(5.0, clock=clock)
        for _ in range(5):
            bucket.try_acquire()
        clock.jump_wall_clock(3600)
        assert not bucket.try_acquire()

    def test_never_exceeds_capacity(self) -> None:
        clock = _clock()
        bucket = TokenBucket(2.0, clock=clock)
        clock.advance(1000)
        assert bucket.tokens == 2.0

    def test_delay_for_reports_the_wait(self) -> None:
        clock = _clock()
        bucket = TokenBucket(2.0, clock=clock)
        bucket.try_acquire(2.0)
        assert bucket.delay_for(1.0) == pytest.approx(0.5)

    async def test_acquire_waits_then_succeeds(self) -> None:
        bucket = TokenBucket(50.0)
        for _ in range(50):
            bucket.try_acquire()
        waited = await bucket.acquire()
        assert waited > 0
        assert bucket.waits >= 1

    def test_rejects_a_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucket(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# SmartApiClient
# ──────────────────────────────────────────────────────────────────────────────


def _ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json={"status": True, "message": "SUCCESS", "data": data})


def _fail(message: str = "Invalid order", code: str = "AB1010") -> httpx.Response:
    return httpx.Response(200, json={"status": False, "message": message, "errorcode": code})


def _client(
    handler: Any,
    *,
    mode: TradingMode = TradingMode.LIVE,
    journal: OrderJournal | None = None,
    settings: Settings | None = None,
    simulate_paper: bool = True,
) -> SmartApiClient:
    transport = httpx.MockTransport(handler)
    return SmartApiClient(
        settings=settings if settings is not None else _settings(),
        journal=journal,
        client=httpx.AsyncClient(base_url="https://test", transport=transport),
        mode=mode,
        clock=_clock(),
        identity=ClientIdentity(local_ip="1.2.3.4", public_ip="1.2.3.4", mac_address="AA:BB"),
        # Against a mock transport there is no broker to protect, and the real 1-per-second
        # limit on getOrderBook would make this suite spend most of its time asleep. The
        # limiter itself is tested directly in TestTokenBucket.
        rate_limit_scale=10_000.0,
        simulate_paper_orders=simulate_paper,
    )


class TestSmartApiClient:
    async def test_login_captures_the_feed_token(self, journal: OrderJournal) -> None:
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == LOGIN.path
            return _ok(
                {
                    "jwtToken": "Bearer jwt-value",
                    "refreshToken": "refresh-value",
                    "feedToken": "feed-value",
                }
            )

        client = _client(handler, journal=journal, settings=settings)
        session = await client.login()
        assert session.jwt_token == "jwt-value"  # the "Bearer " prefix is stripped
        assert client.feed_token == "feed-value"
        assert client.is_authenticated
        await client.aclose()

    async def test_login_without_credentials_fails_fast(self) -> None:
        client = _client(lambda request: _ok({}))
        with pytest.raises(SmartApiAuthError, match="required to log in"):
            await client.login()
        await client.aclose()

    async def test_login_sends_every_header_the_waf_requires(self) -> None:
        """Angel One's edge firewall 403s a login that is missing any of these."""
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key-value",
        )
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(request.headers))
            return _ok({"jwtToken": "jwt", "refreshToken": "r", "feedToken": "f"})

        client = _client(handler, settings=settings)
        await client.login()
        await client.aclose()

        assert seen["content-type"] == "application/json"
        assert seen["accept"] == "application/json"
        assert seen["x-privatekey"] == "key-value"  # SMARTAPI_API_KEY
        assert seen["x-usertype"] == "USER"
        assert seen["x-sourceid"] == "WEB"
        assert seen["x-clientlocalip"] == "1.2.3.4"
        assert seen["x-clientpublicip"] == "1.2.3.4"
        assert seen["x-macaddress"] == "AA:BB"
        assert "authorization" not in seen  # the login itself is unauthenticated

    def test_client_identity_falls_back_to_a_dummy_when_there_is_no_egress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_socket(*args: Any, **kwargs: Any) -> Any:
            raise OSError("no route to host")

        monkeypatch.setattr(socket, "socket", no_socket)
        identity = ClientIdentity.detect()
        # Loopback would be 403ed by the WAF, so the fallback is a dummy LAN address.
        assert identity.local_ip == "192.168.1.1"
        # ``public_ip`` is the WAF-whitelisted Giganode egress, independent of the
        # local LAN — the broker does not care what the LAN address is, only what
        # address reaches it. Reporting the LAN address here is the failure mode
        # that produced the original HTTP 401/403 from the WAF.
        assert identity.public_ip == "87.76.191.175"
        assert re.fullmatch(r"(?:[0-9A-F]{2}:){5}[0-9A-F]{2}", identity.mac_address)

    def test_client_identity_rejects_a_loopback_egress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class LoopbackProbe:
            def settimeout(self, seconds: float) -> None:
                pass

            def connect(self, address: Any) -> None:
                pass

            def getsockname(self) -> tuple[str, int]:
                return ("127.0.0.1", 0)

            def __enter__(self) -> LoopbackProbe:
                return self

            def __exit__(self, *exc_info: Any) -> None:
                pass

        monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: LoopbackProbe())
        identity = ClientIdentity.detect()
        assert identity.local_ip == "192.168.1.1"
        # The loopback detection still applies to ``local_ip`` (an informational
        # header), but ``public_ip`` is the static egress IP — not derived from
        # the detected interface, never the LAN.
        assert identity.public_ip == "87.76.191.175"

    def test_client_identity_honours_angel_public_ip_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANGEL_PUBLIC_IP", "203.0.113.42")
        try:
            identity = ClientIdentity.detect()
            assert identity.public_ip == "203.0.113.42"
        finally:
            monkeypatch.delenv("ANGEL_PUBLIC_IP", raising=False)

    async def test_totp_survives_whitespace_and_missing_padding(self) -> None:
        """Secrets pasted from authenticator exports lose padding and gain spaces."""
        settings = _settings(smartapi_totp_secret=" jbsw y3dp ehpk 3px ")
        client = _client(lambda request: _ok({}), settings=settings)
        expected = str(pyotp.TOTP("JBSWY3DPEHPK3PX=").now())
        assert client.current_totp() == expected
        await client.aclose()

    async def test_a_waf_block_surfaces_its_rejection_text(self, journal: OrderJournal) -> None:
        html = (
            "<html><head><title>Access Denied</title></head><body>  Access   Denied  </body></html>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text=html, headers={"Content-Type": "text/html"})

        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
        )
        client = _client(handler, journal=journal, settings=settings)
        # A rejected *session* request is an auth failure, not an unknown order outcome —
        # and this is the shape the login backoff inspects for the rate-limit phrase.
        with pytest.raises(SmartApiAuthError, match=r"HTTP 403\): Access Denied"):
            await client.login()
        await client.aclose()

        text = journal.path_for().read_text(encoding="utf-8")
        assert "Access Denied" in text

    async def test_a_non_json_body_on_a_read_is_a_plain_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="<html>Request rejected</html>")

        client = _client(handler)
        with pytest.raises(SmartApiError) as exc_info:
            await client.order_book()
        await client.aclose()
        assert "Request rejected" in str(exc_info.value)
        # A read cannot leave the broker in an unknown state — no escalation needed.
        assert not isinstance(exc_info.value, UnknownOrderOutcomeError)

    async def test_an_empty_non_json_body_is_reported_safely(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="")

        client = _client(handler)
        with pytest.raises(SmartApiError, match="<empty body>"):
            await client.order_book()
        await client.aclose()

    # ── session token cache + login rate-limit backoff ───────────────────────

    async def test_login_adopts_todays_cache_without_touching_the_network(self) -> None:
        token_cache.save_session_cache(
            api_key="key",
            client_code="ABC123",
            jwt_token="cached-jwt",
            refresh_token="cached-refresh",
            feed_token="cached-feed",
            clock=_clock(),  # the client's own clock decides what "today" means
        )
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("a cached session must not hit the network")

        client = _client(handler, settings=settings)
        session = await client.login()
        assert session.jwt_token == "cached-jwt"
        assert session.feed_token == "cached-feed"
        assert client.is_authenticated
        await client.aclose()

    async def test_login_saves_the_cache_for_the_next_start(self) -> None:
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        client = _client(
            lambda request: _ok(
                {"jwtToken": "jwt", "refreshToken": "refresh", "feedToken": "feed"}
            ),
            settings=settings,
        )
        assert token_cache.load_session_cache("key", "ABC123", clock=_clock()) is None
        await client.login()
        await client.aclose()

        saved = token_cache.load_session_cache("key", "ABC123", clock=_clock())
        assert saved is not None  # the next start adopts this instead of logging in
        assert saved.jwt_token == "jwt"
        assert saved.feed_token == "feed"

    async def test_a_stale_cache_falls_back_to_a_real_login(self) -> None:
        yesterday = ManualClock(wall=datetime(2026, 8, 9, 15, 30, tzinfo=IST), mono=900.0)
        token_cache.save_session_cache(
            api_key="key",
            client_code="ABC123",
            jwt_token="stale-jwt",
            refresh_token="stale-refresh",
            feed_token="stale-feed",
            clock=yesterday,
        )
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _ok({"jwtToken": "fresh-jwt", "refreshToken": "r", "feedToken": "fresh-feed"})

        client = _client(handler, settings=settings)
        session = await client.login()
        assert session.jwt_token == "fresh-jwt"
        assert calls == 1
        await client.aclose()

    async def test_login_backs_off_on_the_rate_limit_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(token_cache, "LOGIN_RATE_LIMIT_BACKOFFS", (0.0, 0.0))
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 3:
                return httpx.Response(
                    403,
                    json={
                        "status": False,
                        "message": "Access denied because of exceeding access rate",
                        "errorcode": "AB1016",
                    },
                )
            return _ok({"jwtToken": "jwt", "refreshToken": "r", "feedToken": "feed"})

        client = _client(handler, settings=settings)
        session = await client.login()
        assert session.jwt_token == "jwt"
        assert calls == 3  # two rate-limited refusals, then success
        assert client.stats.rate_limit_waits == 2
        await client.aclose()

    async def test_login_does_not_retry_a_genuine_rejection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Wrong credentials must surface at once — retrying them burns the attempt budget."""
        monkeypatch.setattr(token_cache, "LOGIN_RATE_LIMIT_BACKOFFS", (0.0, 0.0))
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="wrong",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200, json={"status": False, "message": "Invalid password", "errorcode": "AB1010"}
            )

        client = _client(handler, settings=settings)
        with pytest.raises(SmartApiAuthError, match="Invalid password"):
            await client.login()
        assert calls == 1
        await client.aclose()

    async def test_login_gives_up_after_the_backoff_schedule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(token_cache, "LOGIN_RATE_LIMIT_BACKOFFS", (0.0,))
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                403,
                json={
                    "status": False,
                    "message": "Access denied because of exceeding access rate",
                    "errorcode": "AB1016",
                },
            )

        client = _client(handler, settings=settings)
        with pytest.raises(SmartApiAuthError, match="exceeding access rate"):
            await client.login()
        assert calls == 2  # one attempt plus one retry, then the schedule is exhausted
        await client.aclose()

    async def test_a_bare_http_401_counts_as_token_expired(self) -> None:
        """A 401 with an undocumented error code still refreshes — the status is the signal."""
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == ORDER_BOOK.path and paths.count(ORDER_BOOK.path) == 1:
                return httpx.Response(
                    401, json={"status": False, "message": "Unauthorized", "errorcode": "XX9999"}
                )
            if request.url.path.endswith("generateTokens"):
                return _ok({"jwtToken": "new-jwt", "feedToken": "f"})
            return _ok([])

        client = _client(handler)
        client.session.refresh_token = "refresh-value"
        assert await client.order_book() == ()
        assert client.stats.token_refreshes == 1
        await client.aclose()

    async def test_a_dead_refresh_token_triggers_a_full_relogin(self) -> None:
        """Cached session 401s, refresh is refused → invalidate cache, full login, replay."""
        token_cache.save_session_cache(
            api_key="key",
            client_code="ABC123",
            jwt_token="cached-jwt",
            refresh_token="cached-refresh",
            feed_token="cached-feed",
            clock=_clock(),
        )
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == ORDER_BOOK.path and paths.count(ORDER_BOOK.path) == 1:
                return httpx.Response(
                    401, json={"status": False, "message": "Token expired", "errorcode": "AG8001"}
                )
            if request.url.path.endswith("generateTokens"):
                return httpx.Response(
                    200,
                    json={
                        "status": False,
                        "message": "Invalid refresh token",
                        "errorcode": "AB1234",
                    },
                )
            if request.url.path == LOGIN.path:
                return _ok({"jwtToken": "fresh-jwt", "refreshToken": "r", "feedToken": "fresh"})
            return _ok([])

        client = _client(handler, settings=settings)
        await client.login()  # adopts the cache — no network yet
        assert paths == []

        assert await client.order_book() == ()
        assert LOGIN.path in paths  # the dead refresh forced a full re-login
        assert client.session.jwt_token == "fresh-jwt"
        assert client.stats.token_refreshes == 0
        await client.aclose()

    async def test_logout_invalidates_the_cache(self) -> None:
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
            smartapi_api_key="key",
        )
        client = _client(
            lambda request: _ok({"jwtToken": "jwt", "refreshToken": "r", "feedToken": "feed"}),
            settings=settings,
        )
        await client.login()
        assert token_cache.load_session_cache("key", "ABC123", clock=_clock()) is not None

        await client.logout()
        assert token_cache.load_session_cache("key", "ABC123", clock=_clock()) is None
        await client.aclose()

    async def test_secrets_never_reach_the_journal(self, journal: OrderJournal) -> None:
        settings = _settings(
            smartapi_client_code="ABC123",
            smartapi_password="hunter2",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
        )
        client = _client(
            lambda request: _ok({"jwtToken": "jwt", "refreshToken": "r", "feedToken": "f"}),
            journal=journal,
            settings=settings,
        )
        await client.login()
        await client.aclose()

        text = journal.path_for().read_text(encoding="utf-8")
        assert "hunter2" not in text
        assert "JBSWY3DPEHPK3PXP" not in text

    async def test_paper_mode_intercepts_every_mutating_call(self) -> None:
        """Phase 11 changed this from "refuse" to "intercept and simulate".

        The invariant is unchanged and is what the tests below pin: nothing is transmitted.
        Simulating rather than raising means a paper session exercises payload construction,
        journalling and response handling instead of bailing out before any of it.
        """
        sent = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal sent
            sent += 1
            return _ok({})

        client = _client(handler, mode=TradingMode.PAPER)
        placed = await client.place_order({"ordertag": "TCHYN-20260810-0001"})
        cancelled = await client.cancel_order("PAPER-000001")

        assert sent == 0, "PAPER must not transmit a mutating request"
        assert placed["orderid"].startswith("PAPER-")
        assert placed["simulated"] is True
        assert cancelled["simulated"] is True
        assert client.stats.paper_intercepts == 2
        await client.aclose()

    async def test_strict_paper_mode_still_refuses(self) -> None:
        """The old behaviour remains available for a caller that wants a hard stop."""
        client = _client(lambda request: _ok({}), mode=TradingMode.PAPER, simulate_paper=False)
        with pytest.raises(PaperModeError):
            await client.place_order({"ordertag": "TCHYN-20260810-0001"})
        with pytest.raises(PaperModeError):
            await client.cancel_order("1")
        await client.aclose()

    async def test_broker_rejection_becomes_an_error_not_a_silent_none(self) -> None:
        client = _client(lambda request: _fail("Insufficient funds", "AB1001"))
        with pytest.raises(SmartApiError) as exc:
            await client.place_order({"ordertag": "T"})
        assert exc.value.error_code == "AB1001"
        assert client.stats.broker_rejections == 1
        await client.aclose()

    async def test_empty_order_book_is_not_an_error(self) -> None:
        client = _client(lambda request: _ok(None))
        assert await client.order_book() == ()
        await client.aclose()

    async def test_placement_is_never_retried_when_the_outcome_is_unknown(self) -> None:
        """CLAUDE.md §6.4 — a duplicate live order is worse than a missed fill."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("no response", request=request)

        client = _client(handler)
        with pytest.raises(UnknownOrderOutcomeError):
            await client.place_order({"ordertag": "TCHYN-20260810-0001"})
        assert calls == 1, "a post-transmission failure must not be retried"
        assert client.stats.unknown_outcomes == 1
        await client.aclose()

    async def test_placement_is_retried_when_it_provably_never_left(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ConnectError("refused", request=request)
            return _ok({"orderid": "251008000001"})

        client = _client(handler)
        result = await client.place_order({"ordertag": "T"})
        assert result["orderid"] == "251008000001"
        assert calls == 3
        await client.aclose()

    async def test_read_only_endpoints_may_retry_after_transmission(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("slow", request=request)
            return _ok([])

        client = _client(handler)
        assert await client.order_book() == ()
        assert calls == 2
        await client.aclose()

    async def test_expired_token_triggers_one_refresh_then_replays(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path == ORDER_BOOK.path and paths.count(ORDER_BOOK.path) == 1:
                return httpx.Response(
                    200,
                    json={"status": False, "message": "Token expired", "errorcode": "AG8002"},
                )
            if request.url.path.endswith("generateTokens"):
                return _ok({"jwtToken": "new-jwt", "feedToken": "f"})
            return _ok([])

        client = _client(handler)
        client.session.refresh_token = "refresh-value"
        assert await client.order_book() == ()
        assert client.stats.token_refreshes == 1
        assert client.session.jwt_token == "new-jwt"
        await client.aclose()

    async def test_available_margin_is_decimal_and_rejects_garbage(self) -> None:
        client = _client(lambda request: _ok({"availablecash": "25000.55"}))
        assert await client.available_margin() == Decimal("25000.55")
        await client.aclose()

        broken = _client(lambda request: _ok({"availablecash": "n/a"}))
        with pytest.raises(SmartApiError):
            await broken.available_margin()
        await broken.aclose()

    async def test_rate_limiter_is_applied_per_endpoint(self) -> None:
        client = _client(lambda request: _ok({"orderid": "1"}))
        bucket = client._bucket(PLACE_ORDER)  # noqa: SLF001 - asserting the wiring
        assert bucket is client._bucket(PLACE_ORDER)  # noqa: SLF001
        assert bucket is not client._bucket(ORDER_BOOK)  # noqa: SLF001
        await client.aclose()


class TestBrokerRowParsing:
    @pytest.mark.parametrize(
        ("status", "expected_open"),
        [
            ("open", True),
            ("trigger pending", True),
            ("open pending", True),
            ("complete", False),
            ("cancelled", False),
            ("rejected", False),
            ("REJECTED", False),
            ("some-status-angel-invented-last-week", True),  # unknown counts as live
            ("", True),
        ],
    )
    def test_order_openness_fails_safe(self, status: str, expected_open: bool) -> None:
        order = BrokerOrder.from_row({"orderid": "1", "orderstatus": status})
        assert order.is_open is expected_open

    def test_recognises_our_own_tags(self) -> None:
        assert BrokerOrder.from_row({"ordertag": "TCHYN-20260810-0001"}).is_ours
        assert not BrokerOrder.from_row({"ordertag": "manual"}).is_ours
        assert not BrokerOrder.from_row({}).is_ours

    @pytest.mark.parametrize(
        ("netqty", "expected"),
        [("0", 0), ("10", 10), ("-5", -5), (7, 7)],
    )
    def test_position_quantity(self, netqty: Any, expected: int) -> None:
        assert BrokerPosition.from_row({"netqty": netqty}).net_quantity == expected

    def test_unparseable_quantity_counts_as_exposure(self) -> None:
        """ "We could not read the quantity" is not "we are flat"."""
        position = BrokerPosition.from_row({"netqty": "???"})
        assert position.is_open


# ──────────────────────────────────────────────────────────────────────────────
# RoboExecutor
# ──────────────────────────────────────────────────────────────────────────────


class _Stack:
    """A full risk + execution stack over a mock broker."""

    def __init__(
        self,
        tmp_path: Path,
        handler: Any,
        *,
        mode: TradingMode = TradingMode.LIVE,
    ) -> None:
        self.clock = _clock()
        self.settings = _settings()
        self.machine = StateMachine(TradingState.ACTIVE, clock=self.clock)
        self.pnl = PnLTracker(self.machine, clock=self.clock)
        self.monitor = FeedMonitor(clock=self.clock)
        self.monitor.record()
        self.positions = PositionRegistry()
        self.risk = RiskEngine(
            self.machine,
            self.pnl,
            self.monitor,
            self.positions,
            settings=self.settings,
            clock=self.clock,
        )
        self.journal = OrderJournal(tmp_path / "journal", clock=self.clock)
        self.client = (
            _client(handler, mode=mode, journal=self.journal, settings=self.settings)
            if handler is not None
            else None
        )
        self.builder = OrderBuilder(settings=self.settings, clock=self.clock)
        self.executor = RoboExecutor(
            client=self.client,
            builder=self.builder,
            risk=self.risk,
            positions=self.positions,
            journal=self.journal,
            settings=self.settings,
            mode=mode,
            clock=self.clock,
            pnl=self.pnl,
        )

    async def aclose(self) -> None:
        if self.client is not None:
            await self.client.aclose()

    def allow(self, symbol: str = "RELIANCE") -> RiskDecision:
        decision = self.risk.evaluate(symbol)
        assert decision.allowed, decision.detail
        return decision


class TestPayload:
    def test_robo_payload_carries_offsets_not_prices(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        plan = stack.builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        payload = stack.executor.payload_for(plan.legs[0])

        assert payload["variety"] == "ROBO"
        assert payload["producttype"] == "BO"
        assert payload["tradingsymbol"] == "RELIANCE-EQ"
        assert payload["price"] == "2500.00"
        # Offsets, in points — not 2487.40 / 2518.90.
        assert payload["stoploss"] == "12.60"
        assert payload["squareoff"] == "18.90"
        assert payload["trailingStopLoss"] == "6.30"
        assert payload["ordertag"].startswith("TCHYN-")

    def test_a_stop_is_always_attached_in_the_same_call(self, tmp_path: Path) -> None:
        """CLAUDE.md §8.1 forbids an order without a stop-loss attached."""
        stack = _Stack(tmp_path, lambda request: _ok({}))
        plan = stack.builder.build(
            symbol="RELIANCE",
            side=Side.SELL,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        for leg in plan.legs:
            payload = stack.executor.payload_for(leg)
            assert Decimal(payload["stoploss"]) > 0

    def test_exit_payload_reverses_the_position_side_at_market(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        long_position = BrokerPosition.from_row(
            {"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}
        )
        payload = stack.executor.exit_payload_for(long_position)
        assert payload["transactiontype"] == "SELL"
        assert payload["ordertype"] == "MARKET"
        assert payload["quantity"] == "7"

        short_position = BrokerPosition.from_row({"symboltoken": "2885", "netqty": "-4"})
        assert stack.executor.exit_payload_for(short_position)["transactiontype"] == "BUY"


class TestOpenPosition:
    async def test_places_both_legs(self, tmp_path: Path) -> None:
        placed: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            placed.append(_json.loads(request.content))
            return _ok({"orderid": f"ORD{len(placed)}"})

        stack = _Stack(tmp_path, handler)
        report = await stack.executor.open_position(
            stack.allow(),
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert report.placed
        assert len(placed) == 2
        assert [p["quantity"] for p in placed] == ["4", "3"]
        assert stack.positions.is_open("RELIANCE")
        await stack.aclose()

    async def test_refuses_an_unauthorised_decision(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({"orderid": "1"}))
        refused = RiskDecision(allowed=False, symbol="RELIANCE", at_ist=stack.clock.now())
        report = await stack.executor.open_position(
            refused,
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert not report.placed
        assert report.rejected_reason == "NOT_AUTHORISED"
        await stack.aclose()

    async def test_re_runs_the_gate_immediately_before_transmitting(self, tmp_path: Path) -> None:
        """The world moves between a signal and a socket write."""
        stack = _Stack(tmp_path, lambda request: _ok({"orderid": "1"}))
        decision = stack.allow()

        # Everything was fine when the signal fired; the feed dies a moment later.
        stack.clock.advance(30)

        report = await stack.executor.open_position(
            decision,
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert not report.placed
        assert report.rejected_reason == "FEED_STALE"
        await stack.aclose()

    async def test_paper_mode_places_nothing(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, None, mode=TradingMode.PAPER)
        report = await stack.executor.open_position(
            stack.allow(),
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert report.simulated
        assert report.placed
        assert report.quantity_placed == 7
        assert stack.positions.is_open("RELIANCE")

    async def test_unknown_outcome_stops_before_sending_the_second_leg(
        self, tmp_path: Path
    ) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("no response", request=request)

        stack = _Stack(tmp_path, handler)
        report = await stack.executor.open_position(
            stack.allow(),
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert calls == 1
        assert not report.placed
        assert report.needs_reconciliation
        await stack.aclose()

    async def test_a_rejected_leg_does_not_stop_the_other(self, tmp_path: Path) -> None:
        """Leg A is a complete bracket on its own — a failed leg B just means smaller size."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _fail() if calls == 1 else _ok({"orderid": "ORD2"})

        stack = _Stack(tmp_path, handler)
        report = await stack.executor.open_position(
            stack.allow(),
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        assert calls == 2
        assert report.placed
        assert report.quantity_placed == 3
        await stack.aclose()

    async def test_a_rejection_never_raises_into_the_strategy_loop(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({"orderid": "1"}))
        report = await stack.executor.open_position(
            stack.allow(),
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("0"),  # degenerate ATR
            headroom=Decimal("500"),
        )
        assert not report.placed
        assert report.rejected_reason == "BAD_ATR"
        await stack.aclose()


class TestStopManagement:
    async def test_breakeven_move_is_one_tick_into_profit(self, tmp_path: Path) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            sent.append(_json.loads(request.content))
            return _ok({"orderid": "ORD2"})

        stack = _Stack(tmp_path, handler)
        geometry = stack.builder.geometry(
            side=Side.BUY, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        await stack.executor.move_stop_to_breakeven(
            order_id="ORD2",
            geometry=geometry,
            current_stop=geometry.stop_loss_price,
            quantity=3,
            trading_symbol="RELIANCE-EQ",
            token="2885",
            exchange="NSE",
        )
        assert sent[0]["triggerprice"] == "2500.05"
        await stack.aclose()

    async def test_refuses_to_widen(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        geometry = stack.builder.geometry(
            side=Side.BUY, entry_price=Decimal("2500"), atr=Decimal("8.40"), tick_size=TICK
        )
        with pytest.raises(StopWidenedError):
            await stack.executor.move_stop_to_breakeven(
                order_id="ORD2",
                geometry=geometry,
                current_stop=Decimal("2600.00"),  # already better than breakeven
                quantity=3,
                trading_symbol="RELIANCE-EQ",
                token="2885",
                exchange="NSE",
            )
        await stack.aclose()


class TestSquareOff:
    async def test_cancels_orders_exits_positions_and_confirms_flat(self, tmp_path: Path) -> None:
        state = {
            "orders": [{"orderid": "1", "orderstatus": "open", "variety": "ROBO"}],
            "positions": [
                {
                    "tradingsymbol": "RELIANCE-EQ",
                    "symboltoken": "2885",
                    "netqty": "7",
                    "exchange": "NSE",
                }
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("getOrderBook"):
                return _ok(list(state["orders"]))
            if path.endswith("getPosition"):
                return _ok(list(state["positions"]))
            if path.endswith("cancelOrder"):
                state["orders"] = []
                return _ok({"orderid": "1"})
            if path.endswith("placeOrder"):
                state["positions"] = []
                return _ok({"orderid": "2"})
            return _ok({})

        stack = _Stack(tmp_path, handler)
        report = await stack.executor.flatten_everything()
        assert report.is_flat
        assert report.orders_cancelled == 1
        assert report.exits_submitted == 1
        assert stack.positions.open_count == 0
        await stack.aclose()

    async def test_is_not_flat_while_a_position_persists(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("getOrderBook"):
                return _ok([])
            if path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}])
            return _ok({"orderid": "2"})

        stack = _Stack(tmp_path, handler)
        report = await stack.executor.flatten_everything()
        assert not report.is_flat
        assert report.residual_positions == 1
        await stack.aclose()

    async def test_never_sends_a_second_exit_for_the_same_instrument(self, tmp_path: Path) -> None:
        """A duplicate exit does not flatten twice — it reverses the position."""
        exits = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal exits
            path = request.url.path
            if path.endswith("getOrderBook"):
                return _ok([])
            if path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}])
            if path.endswith("placeOrder"):
                exits += 1
                return _ok({"orderid": "2"})
            return _ok({})

        stack = _Stack(tmp_path, handler)
        for _ in range(4):  # the watchdog retries
            await stack.executor.flatten_everything()
        assert exits == 1
        await stack.aclose()

    async def test_a_definitively_rejected_exit_is_retried(self, tmp_path: Path) -> None:
        """Nothing reached the market, so re-sending cannot duplicate anything."""
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            path = request.url.path
            if path.endswith("getOrderBook"):
                return _ok([])
            if path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}])
            if path.endswith("placeOrder"):
                attempts += 1
                return _fail("Market closed", "AB1004")
            return _ok({})

        stack = _Stack(tmp_path, handler)
        await stack.executor.flatten_everything()
        await stack.executor.flatten_everything()
        assert attempts == 2
        await stack.aclose()

    async def test_paper_square_off_touches_no_broker(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, None, mode=TradingMode.PAPER)
        stack.positions.record_entry("RELIANCE", 7)
        report = await stack.executor.flatten_everything()
        assert report.simulated
        assert report.is_flat
        assert stack.positions.open_count == 0

    async def test_watchdog_action_raises_until_actually_flat(self, tmp_path: Path) -> None:
        """The watchdog's success criterion is being flat, not having sent the requests."""

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("getOrderBook"):
                return _ok([{"orderid": "1", "orderstatus": "open"}])
            if path.endswith("getPosition"):
                return _ok([])
            return _fail("cannot cancel")

        stack = _Stack(tmp_path, handler)
        loop = asyncio.get_running_loop()
        action = stack.executor.square_off_action(loop, timeout=5.0)
        with pytest.raises(NotFlatError):
            await asyncio.to_thread(action)
        await stack.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Reconciliation — CLAUDE.md §9
# ──────────────────────────────────────────────────────────────────────────────


def _reconciler(
    tmp_path: Path,
    handler: Any,
    *,
    mode: TradingMode = TradingMode.LIVE,
    known: tuple[str, ...] = (),
) -> tuple[StateReconciler, StateMachine, SmartApiClient | None]:
    clock = _clock()
    settings = _settings()
    machine = StateMachine(TradingState.ACTIVE, clock=clock)
    positions = PositionRegistry()
    for symbol in known:
        positions.record_entry(symbol, 1)
    client = _client(handler, mode=mode, settings=settings) if handler is not None else None
    reconciler = StateReconciler(
        client=client,
        state_machine=machine,
        positions=positions,
        settings=settings,
        journal=OrderJournal(tmp_path / "journal", clock=clock),
        mode=mode,
        clock=clock,
    )
    return reconciler, machine, client


class TestReconciliation:
    async def test_clean_when_the_broker_is_flat(self, tmp_path: Path) -> None:
        reconciler, machine, client = _reconciler(tmp_path, lambda request: _ok([]))
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.CLEAN
        assert report.may_trade
        assert machine.state is TradingState.ACTIVE
        assert client is not None
        await client.aclose()

    async def test_unknown_position_locks_the_system(self, tmp_path: Path) -> None:
        """The hard-crash case: a live bracket we have no memory of."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}])
            return _ok([])

        reconciler, machine, client = _reconciler(tmp_path, handler)
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.MISMATCH
        assert report.unknown_positions == ("RELIANCE-EQ",)
        assert machine.state is TradingState.LOCKED
        assert client is not None
        await client.aclose()

    async def test_unknown_working_order_locks_the_system(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("getOrderBook"):
                return _ok(
                    [
                        {
                            "orderid": "99",
                            "orderstatus": "open",
                            "tradingsymbol": "RELIANCE-EQ",
                            "symboltoken": "2885",
                        }
                    ]
                )
            return _ok([])

        reconciler, machine, client = _reconciler(tmp_path, handler)
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.MISMATCH
        assert machine.state is TradingState.LOCKED
        assert client is not None
        await client.aclose()

    async def test_a_position_we_know_about_is_not_a_mismatch(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "7"}])
            return _ok([])

        reconciler, machine, client = _reconciler(tmp_path, handler, known=("RELIANCE",))
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.CLEAN
        assert machine.state is TradingState.ACTIVE
        assert client is not None
        await client.aclose()

    async def test_a_closed_position_is_not_exposure(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("getPosition"):
                return _ok([{"tradingsymbol": "RELIANCE-EQ", "symboltoken": "2885", "netqty": "0"}])
            return _ok([])

        reconciler, machine, client = _reconciler(tmp_path, handler)
        assert (await reconciler.run_at_boot()).outcome is ReconcileOutcome.CLEAN
        assert client is not None
        await client.aclose()

    async def test_an_unreachable_broker_locks_the_system(self, tmp_path: Path) -> None:
        """ "We could not read the order book" is not "the order book is empty"."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host", request=request)

        reconciler, machine, client = _reconciler(tmp_path, handler)
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.UNAVAILABLE
        assert not report.may_trade
        assert machine.state is TradingState.LOCKED
        assert client is not None
        await client.aclose()

    async def test_paper_without_a_client_skips(self, tmp_path: Path) -> None:
        reconciler, machine, _ = _reconciler(tmp_path, None, mode=TradingMode.PAPER)
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.SKIPPED
        assert report.may_trade
        assert machine.state is TradingState.ACTIVE

    async def test_orphaned_local_state_is_reported_but_not_fatal(self, tmp_path: Path) -> None:
        reconciler, machine, client = _reconciler(
            tmp_path, lambda request: _ok([]), known=("RELIANCE",)
        )
        report = await reconciler.run_at_boot()
        assert report.orphaned_local == ("RELIANCE",)
        assert report.outcome is ReconcileOutcome.CLEAN
        assert machine.state is TradingState.ACTIVE
        assert client is not None
        await client.aclose()

    async def test_reconcile_never_raises(self, tmp_path: Path) -> None:
        class Exploding:
            async def order_book(self) -> Any:
                raise RuntimeError("boom")

            async def positions(self) -> Any:
                raise RuntimeError("boom")

        reconciler, machine, _ = _reconciler(tmp_path, lambda request: _ok([]))
        reconciler._client = Exploding()  # type: ignore[assignment]  # noqa: SLF001
        report = await reconciler.run_at_boot()
        assert report.outcome is ReconcileOutcome.UNAVAILABLE
        assert machine.state is TradingState.LOCKED


# ──────────────────────────────────────────────────────────────────────────────
# Journal
# ──────────────────────────────────────────────────────────────────────────────


class TestOrderJournal:
    def test_appends_and_reads_back(self, journal: OrderJournal) -> None:
        journal.request("place_order", {"quantity": "7"}, order_tag="TCHYN-20260810-0001")
        journal.response("place_order", {"orderid": "1"}, order_tag="TCHYN-20260810-0001")
        records = journal.read()
        assert [r["kind"] for r in records] == ["REQUEST", "RESPONSE"]
        assert records[0]["payload"]["quantity"] == "7"

    def test_scrubs_credentials(self, journal: OrderJournal) -> None:
        journal.request("login", {"clientcode": "ABC", "password": "hunter2", "totp": "123456"})
        text = journal.path_for().read_text(encoding="utf-8")
        assert "hunter2" not in text
        assert "123456" not in text
        assert "ABC" in text

    def test_a_write_failure_never_raises(self, tmp_path: Path) -> None:
        """A full disk must not become "we cannot cancel this order"."""
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        journal = OrderJournal(blocked, clock=_clock())
        journal.request("place_order", {"quantity": "1"})  # must not raise
        assert journal.write_failures == 1

    def test_a_truncated_line_does_not_poison_the_rest(self, journal: OrderJournal) -> None:
        journal.request("place_order", {"quantity": "7"})
        with journal.path_for().open("ab") as handle:
            handle.write(b'{"kind": "REQUEST", trunc\n')
        journal.response("place_order", {"orderid": "1"})
        assert len(journal.read()) == 2

    def test_dated_filenames(self, journal: OrderJournal) -> None:
        assert journal.path_for().name == "orders_2026-08-10.jsonl"


class TestOpenBracket:
    def test_built_from_a_successful_report(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, None, mode=TradingMode.PAPER)
        plan = stack.builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        report = stack.executor._simulate(plan, "now")  # noqa: SLF001
        bracket = OpenBracket.from_report(report)
        assert bracket is not None
        assert bracket.leg_a_order_id and bracket.leg_b_order_id
        assert bracket.current_stop == plan.geometry.stop_loss_price

    def test_none_when_nothing_was_placed(self) -> None:
        from tachyon.execution.executor import ExecutionReport

        empty = ExecutionReport(symbol="RELIANCE", side=Side.BUY, at_ist="now", plan=None)
        assert OpenBracket.from_report(empty) is None


# ──────────────────────────────────────────────────────────────────────────────
# P&L booking — the "+Rs.0.00" square-off fix
# ──────────────────────────────────────────────────────────────────────────────


class TestPnlBooking:
    def test_bracket_entry_records_levels_and_direction(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        plan = stack.builder.build(
            symbol="RELIANCE",
            side=Side.BUY,
            entry_price=Decimal("2500"),
            atr=Decimal("8.40"),
            headroom=Decimal("500"),
        )
        stack.executor._record_bracket_entry(plan)  # noqa: SLF001

        record = stack.positions.get("RELIANCE")
        assert record is not None
        assert record.direction == "LONG"
        assert record.entry_price == plan.legs[0].entry_price
        assert record.stop_loss == plan.legs[0].stop_loss_price
        assert record.target == plan.legs[0].target_price

    def test_short_entry_inverts_direction(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        plan = stack.builder.build(
            symbol="RELIANCE",
            side=Side.SELL,
            entry_price=Decimal("890.70"),
            atr=Decimal("4.15"),
            headroom=Decimal("500"),
        )
        stack.executor._record_bracket_entry(plan)  # noqa: SLF001
        record = stack.positions.get("RELIANCE")
        assert record is not None and record.direction == "SHORT"

    def test_on_fill_books_long_pnl_from_broker_strings(self, tmp_path: Path) -> None:
        """Broker fills arrive as JSON strings — they must book exactly."""
        stack = _Stack(tmp_path, lambda request: _ok({}))
        stack.positions.record_entry(
            "RELIANCE",
            quantity=10,
            entry_price="100.00",
            direction="LONG",
        )
        booked = stack.executor.on_fill("RELIANCE", "105.00", 10, charges="5")

        assert booked
        assert stack.pnl.realised == Decimal("50")  # (105-100)*10, gross
        assert stack.pnl.total == Decimal("45")  # gross − ₹5 charges
        assert not stack.positions.is_open("RELIANCE")

    def test_on_fill_books_short_pnl_inverted(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        stack.positions.record_entry(
            "RELIANCE",
            quantity=10,
            entry_price="100.00",
            direction="SHORT",
        )
        booked = stack.executor.on_fill("RELIANCE", "90.00", 10)

        assert booked
        assert stack.pnl.realised == Decimal("100")  # (100-90)*10 for a short

    def test_unparseable_fill_price_is_rejected_not_zeroed(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        stack.positions.record_entry("RELIANCE", 10, entry_price="100")
        realised_before = stack.pnl.realised

        assert stack.executor.on_fill("RELIANCE", "N/A", 5) is False
        assert stack.pnl.realised == realised_before, "a bad price must never book ₹0.00"

    def test_unknown_symbol_returns_false(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        assert stack.executor.on_fill("UNKNOWN", "100", 1) is False

    def test_missing_tracker_is_critical_not_silent(self, tmp_path: Path) -> None:
        stack = _Stack(tmp_path, lambda request: _ok({}))
        executor_no_pnl = RoboExecutor(
            client=stack.client,
            builder=stack.builder,
            risk=stack.risk,
            positions=stack.positions,
            journal=stack.journal,
            settings=stack.settings,
            mode=TradingMode.LIVE,
            clock=stack.clock,
            pnl=None,
        )
        stack.positions.record_entry("RELIANCE", 5, entry_price="100")
        assert executor_no_pnl.on_fill("RELIANCE", "101", 5) is False

    async def test_square_off_books_pnl_from_position_row_ltp(self, tmp_path: Path) -> None:
        """The 15:15 path: exit accepted → P&L booked from the broker row's own LTP."""
        from tachyon.execution.api import BrokerPosition

        submitted: list[dict[str, str]] = []
        stack = _Stack(
            tmp_path,
            lambda request: (_ok({"orderid": "exit-1"}), submitted.append(request))[0],
        )
        stack.positions.record_entry(
            "RELIANCE",
            quantity=10,
            entry_price="890.70",
            direction="SHORT",
            stop_loss="894.85",
            target="875.00",
        )

        position = BrokerPosition(
            trading_symbol="RELIANCE-EQ",
            token="738561",
            exchange="NSE",
            net_quantity=-10,
            product_type="BO",
            raw={"ltp": "920.35"},
        )
        executor = _StubBrokerQueries(
            client=stack.client,
            builder=stack.builder,
            risk=stack.risk,
            positions=stack.positions,
            journal=stack.journal,
            settings=stack.settings,
            mode=TradingMode.LIVE,
            clock=stack.clock,
            pnl=stack.pnl,
            open_positions=[position],
            working_orders=[],
        )

        report = await executor.flatten_everything()
        assert report.exits_submitted == 1
        assert len(submitted) == 1

        # SHORT from 890.70 exited at 920.35 → −29.65 × 10.
        assert stack.pnl.realised == Decimal("-296.50"), (
            "square-off P&L must be booked from the best known price"
        )

    async def test_square_off_without_any_price_defers_instead_of_booking_zero(
        self,
        tmp_path: Path,
    ) -> None:
        from tachyon.execution.api import BrokerPosition

        stack = _Stack(tmp_path, lambda request: _ok({"orderid": "exit-1"}))
        stack.positions.record_entry(
            "RELIANCE", quantity=10, entry_price="890.70", direction="SHORT"
        )
        position = BrokerPosition(
            trading_symbol="RELIANCE-EQ",
            token="738561",
            exchange="NSE",
            net_quantity=-10,
            product_type="BO",
            raw={},
        )
        executor = _StubBrokerQueries(
            client=stack.client,
            builder=stack.builder,
            risk=stack.risk,
            positions=stack.positions,
            journal=stack.journal,
            settings=stack.settings,
            mode=TradingMode.LIVE,
            clock=stack.clock,
            pnl=stack.pnl,
            open_positions=[position],
            working_orders=[],
        )

        realised_before = stack.pnl.realised
        await executor.flatten_everything()
        assert stack.pnl.realised == realised_before, (
            "no price ⇒ no booking; silence was the ₹0.00 bug"
        )


class _StubBrokerQueries(RoboExecutor):
    """Overrides the two broker queries square-off depends on (slots-safe)."""

    def __init__(
        self,
        *,
        open_positions: list[Any],
        working_orders: list[Any],
        **kwargs: Any,
    ) -> None:
        self._stub_positions = open_positions
        self._stub_orders = working_orders
        super().__init__(**kwargs)

    async def _positions_open(self) -> tuple[BrokerPosition, ...]:
        return tuple(self._stub_positions)

    async def _orders(self) -> tuple[Any, ...]:
        return tuple(self._stub_orders)


async def _aret(value: object) -> object:
    return value
    return value


class TestShmTopOfBookReader:
    """The router's SHM book reader must unpack the full 56-byte slot.

    Regression: ``_SLOT_STRUCT`` is ``<QQ8fI4x`` (timestamp, sequence, 8 floats,
    flags). The reader originally unpacked that into four names, which raised
    ``too many values to unpack (expected 4, got 11)`` the moment a matching slot
    was found — silently denying every book lookup. This writes a tick through the
    real :class:`~tachyon.ingestion.shm_writer.SHMWriter` and reads it back on a
    private segment name, so the live ring is never touched.
    """

    @pytest.fixture()
    def segment_name(self) -> Iterator[str]:
        name = f"tachyon_test_book_{os.getpid()}"
        yield name
        if os.name == "posix":
            with contextlib.suppress(FileNotFoundError):
                (Path("/dev/shm") / name).unlink()

    def test_reads_back_written_tick(self, segment_name: str) -> None:
        from tachyon.execution.router import ShmTopOfBookReader
        from tachyon.ingestion.shm_writer import SHMWriter

        token = 12345
        floats = (100.5, 40.0, 100.4, 15.0, 101.0, 25.0, 101.1, 9.0)
        writer = SHMWriter(name=segment_name)
        try:
            writer.write_tick(token, 1_700_000_000_000_000_000, floats)
        finally:
            writer.close()

        reader = ShmTopOfBookReader(name=segment_name)
        try:
            book = reader.read(token)
        finally:
            reader.close()

        assert book is not None, "a fresh matching slot must produce a book"
        assert book.bid == 100.5
        assert book.bid_qty == 40.0
        assert book.ask == 101.0
        assert book.ask_qty == 25.0
        assert book.usable

    def test_unknown_token_returns_none(self, segment_name: str) -> None:
        from tachyon.execution.router import ShmTopOfBookReader
        from tachyon.ingestion.shm_writer import SHMWriter

        writer = SHMWriter(name=segment_name)
        try:
            writer.write_tick(
                777, 1_700_000_000_000_000_000, (1.0, 1.0, 0.9, 1.0, 1.1, 1.0, 1.2, 1.0)
            )
        finally:
            writer.close()

        reader = ShmTopOfBookReader(name=segment_name)
        try:
            assert reader.read(999) is None
        finally:
            reader.close()

    def test_newest_slot_wins_across_multiple_writes(self, segment_name: str) -> None:
        """Cells beyond the first must pass the seqlock integrity check.

        Regression: the reader once required ``sequence == stamp - 1``, which only
        holds for the very first slot of a fresh ring — every later cell was
        silently rejected and book lookups returned ``None``. The stamp counts a
        cell's writes (2 per completed write) while the payload sequence is the
        global tail, so the invariant must derive the expected sequence from the
        stamp and the cell index.
        """
        from tachyon.execution.router import ShmTopOfBookReader
        from tachyon.ingestion.shm_writer import SHMWriter

        token = 4242
        writer = SHMWriter(name=segment_name)
        try:
            for i in range(1, 6):
                other = 9999 if i % 2 else token
                writer.write_tick(
                    other,
                    1_700_000_000_000_000_000 + i,
                    (100.0 + i, 1.0, 100.0, 1.0, 101.0 + i, 2.0, 102.0, 1.0),
                )
            writer.write_tick(
                token, 1_700_000_000_000_000_999, (200.0, 7.0, 199.0, 3.0, 201.0, 9.0, 202.0, 1.0)
            )
        finally:
            writer.close()

        reader = ShmTopOfBookReader(name=segment_name)
        try:
            book = reader.read(token)
        finally:
            reader.close()

        assert book is not None, "cells after the first must satisfy the integrity check"
        assert book.bid == 200.0, "the newest matching slot must win"
        assert book.ask == 201.0
        assert book.ask_qty == 9.0
