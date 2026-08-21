"""Order geometry, sizing and the two-leg bracket split — CLAUDE.md §6.1, §6.2, §6.3.

This module is **pure arithmetic**. It performs no I/O, holds no state beyond an order-tag
counter, and talks to no broker. That is deliberate: the numbers that decide how much money is
at risk should be verifiable by reading them, and testable without a network.

Everything here is :class:`~decimal.Decimal`. ``float`` is banned at the risk boundary
(CLAUDE.md §8) — binary floating point cannot represent ₹0.05, and a stop that lands on
₹2499.9999999999995 is a broker rejection at best and a wrong stop at worst. The ATR arrives
from the Numba engine as a ``float64``; :func:`to_decimal` is the single conversion point, and
it refuses NaN and infinity rather than propagating them into an order.

The geometry
------------
With ``A`` = ATR(14) on 5-minute candles and ``E`` = the intended entry::

    R  = round_to_tick(1.5 × A)          # risk per share, and the stop distance

    LONG                                  SHORT
      stop = E − R                          stop = E + R
      T1   = E + 1.5 × R                    T1   = E − 1.5 × R
      T2   = E + 2.5 × R                    T2   = E − 2.5 × R

**R is tick-aligned before the targets are derived from it**, not after. The alternative —
computing ``1.5 × 1.5 × ATR`` and rounding that — makes the reward/risk ratio drift away from
1:1.5 against the stop distance we will *actually* be filled at. Since the realised risk is the
rounded stop distance, every multiple must be taken against the rounded number. A pleasant
side effect: an offset that is a whole number of ticks added to a tick-aligned entry lands on a
tick-aligned price by construction, so no price in a payload can be misaligned.

Sizing
------
::

    budget = min(PER_TRADE_RISK_INR, remaining_loss_headroom) × sentinel_multiplier
    qty    = floor(budget / R)

Floored, always. ``qty < 1`` means **no trade** — there is no "just one share" branch, because
one share of a ₹3000 stock with a ₹40 stop is ₹40 of risk that was never budgeted for.

The 60/40 split
---------------
A Robo Order carries exactly one target, so two targets means two orders (CLAUDE.md §6.1):

* **Leg A** — 60 % of quantity, target at 1.5 R
* **Leg B** — the remainder, target at 2.5 R

Both legs carry the identical stop. The split is computed in **lots**, not shares, so an F&O
instrument cannot end up with a leg that is not a whole number of lots — the exchange would
reject it. For a single lot, ``round(1 × 0.60) == 1``: the entire position goes to leg A and
books at T1. Concentrating a minimum-size position on the nearer target is the conservative
resolution, and it is what the arithmetic already does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from enum import StrEnum
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, today_ist
from tachyon.core.config import Settings, WatchlistItem, get_settings
from tachyon.core.constants import (
    PER_TRADE_RISK_INR,
    SL_ATR_MULTIPLIER,
    T1_QTY_FRACTION,
    T1_RR,
    T2_RR,
)
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")
ONE: Final[Decimal] = Decimal("1")

#: Tag prefix for every order this system places (CLAUDE.md §6.4). Angel One caps ``ordertag``
#: at 20 characters; ``TCHYN-20260810-0001`` is 19, leaving the format headroom-free but valid.
ORDER_TAG_PREFIX: Final[str] = "TCHYN"

#: Trading symbols on the NSE/BSE cash segment carry an ``-EQ`` suffix. Sending the bare name
#: is a guaranteed rejection, so it is appended when the watchlist does not spell it out.
_CASH_SEGMENTS: Final[frozenset[str]] = frozenset({"NSE", "BSE"})
_EQ_SUFFIX: Final[str] = "-EQ"


class OrderRejected(ValueError):  # noqa: N818 - see the docstring
    """The requested order cannot be built. Carries the reason for the journal and the UI.

    A rejection is a normal, safe outcome — a missed trade costs nothing (CLAUDE.md §0). It is
    an exception rather than a sentinel so that a caller cannot accidentally place an order
    built from a half-populated result.

    Deliberately not suffixed ``Error``: "rejected" is the domain word, and :attr:`reason` is
    carried verbatim into ``ExecutionReport.rejected_reason`` and the order journal. Renaming
    the class would make the code and the audit record disagree about the same thing.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


class Side(StrEnum):
    """Direction of the *entry* order. Exits are the opposite side."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> Decimal:
        """``+1`` for a long, ``−1`` for a short. Targets are above, stops below, for a long."""
        return ONE if self is Side.BUY else -ONE

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class Leg(StrEnum):
    """Which half of the split a Robo order represents."""

    A = "A"  # 60% of quantity, target T1 = 1.5R
    B = "B"  # the remainder, target T2 = 2.5R


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────


def to_decimal(value: Decimal | float | int | str, *, field: str = "value") -> Decimal:
    """Convert a number to ``Decimal`` at the risk boundary, rejecting the undefined.

    Floats are routed through ``str`` rather than ``Decimal(float)``: the latter faithfully
    reproduces the binary artefact (``Decimal(0.1)`` is ``0.1000000000000000055511...``), which
    then propagates through every subsequent multiplication.

    Raises:
        OrderRejected: the value is NaN, infinite, or not a number. A NaN ATR is exactly the
            case that must never reach an order — every comparison against it is false, so a
            naive implementation would sail through its own sanity checks.
    """
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise OrderRejected("NON_NUMERIC", f"{field}={value!r} ({exc})") from exc
    if not result.is_finite():
        raise OrderRejected("UNDEFINED_INPUT", f"{field} is {result} — refusing to build")
    return result


def round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    """Quantise ``value`` to the nearest whole multiple of ``tick`` (CLAUDE.md §6.2).

    ``ROUND_HALF_UP`` is mandated by the constitution, and the result is normalised to the
    tick's own exponent so that the string sent to the broker has the expected number of
    decimal places — ``2500.00``, never ``2500`` or ``2500.000000``.

    Raises:
        OrderRejected: ``tick`` is not positive. A zero or negative tick size would make every
            price meaningless, so it fails here rather than dividing by zero later.
    """
    if tick <= ZERO:
        raise OrderRejected("BAD_TICK_SIZE", f"tick_size={tick} must be positive")
    steps = (value / tick).quantize(ONE, rounding=ROUND_HALF_UP)
    return (steps * tick).quantize(tick)


def floor_div_decimal(numerator: Decimal, denominator: Decimal) -> int:
    """``floor(numerator / denominator)`` as an int. Never rounds up (CLAUDE.md §6.3)."""
    if denominator <= ZERO:
        raise OrderRejected("BAD_DIVISOR", f"denominator={denominator} must be positive")
    return int((numerator / denominator).to_integral_value(rounding=ROUND_FLOOR))


def clamp_multiplier(value: Decimal | float) -> Decimal:
    """Clamp a Sentinel size multiplier to ``[0, 1]`` (CLAUDE.md §5).

    Applied in our code, never trusted from the model. The Sentinel may only ever make the
    system *more* conservative, so a multiplier above 1 — however it got there — is discarded
    rather than honoured. A non-numeric or undefined multiplier resolves to ``0``: refusing to
    size the trade at all, which is the safe direction.
    """
    try:
        multiplier = to_decimal(value, field="sentinel_multiplier")
    except OrderRejected:
        return ZERO
    if multiplier <= ZERO:
        return ZERO
    return ONE if multiplier > ONE else multiplier


def trading_symbol_for(item: WatchlistItem) -> str:
    """Broker trading symbol for a watchlist entry, e.g. ``RELIANCE`` -> ``RELIANCE-EQ``."""
    if item.trading_symbol:
        return item.trading_symbol
    if item.exchange in _CASH_SEGMENTS and not item.symbol.endswith(_EQ_SUFFIX):
        return f"{item.symbol}{_EQ_SUFFIX}"
    return item.symbol


# ──────────────────────────────────────────────────────────────────────────────
# Order tags — CLAUDE.md §6.4
# ──────────────────────────────────────────────────────────────────────────────


class OrderTagSequencer:
    """Issues ``TCHYN-{yyyymmdd}-{seq}`` client tags.

    The tag is our idempotency handle: it is the only field that lets a reconciliation pass
    recognise an order the broker accepted but whose response we never saw. Each *order* gets
    its own sequence number — the two legs of one bracket are two orders and must be
    distinguishable, otherwise a duplicate-detection pass would treat one as an echo of the
    other and cancel a live leg.

    The counter resets when the IST date rolls over, so a process left running past midnight
    does not emit yesterday's date.
    """

    __slots__ = ("_clock", "_date", "_seq")

    def __init__(self, clock: Clock = SYSTEM_CLOCK, start: int = 0) -> None:
        self._clock = clock
        self._seq = start
        self._date: date | None = None

    def next_tag(self) -> str:
        today = today_ist(self._clock)
        if today != self._date:
            self._date = today
            self._seq = 0
        self._seq += 1
        return f"{ORDER_TAG_PREFIX}-{today:%Y%m%d}-{self._seq:04d}"

    @property
    def issued(self) -> int:
        return self._seq


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OrderGeometry:
    """Every price and offset for one bracket. All tick-aligned, all ``Decimal``.

    The broker takes *offsets in points* (:attr:`stoploss_offset`, :attr:`squareoff_offset`),
    not absolute prices. The absolute prices are carried alongside them because they are what a
    human reads in the journal and the UI, and because an offset with the sign convention
    silently inverted is invisible until it costs money — having both lets a test assert the
    relationship directly.
    """

    side: Side
    atr: Decimal
    tick_size: Decimal

    entry_price: Decimal
    stop_loss_price: Decimal
    target_1_price: Decimal
    target_2_price: Decimal

    risk_per_share: Decimal
    """R — the stop distance in points, tick-aligned. Sizing divides the budget by this."""

    stoploss_offset: Decimal
    """Points from entry to the stop. Equal to :attr:`risk_per_share` by definition."""

    t1_offset: Decimal
    """Points from entry to T1 (1.5 R). Leg A's ``squareoff``."""

    t2_offset: Decimal
    """Points from entry to T2 (2.5 R). Leg B's ``squareoff``."""

    trail_step: Decimal
    """Broker-native trailing step in points (CLAUDE.md §6.2). Never a client-side loop."""

    trail_activation: Decimal
    """Favourable move, in points, before the trail is expected to engage."""

    def breakeven_stop(self) -> Decimal:
        """Breakeven plus one tick, in the direction of profit (CLAUDE.md §6.1).

        Where leg B's stop moves once leg A books at T1. "Plus one tick" rather than exactly
        breakeven so that the trade cannot round back into a loss on the exit fill.
        """
        return self.entry_price + self.side.sign * self.tick_size

    def is_favourable(self, price: Decimal, target: Decimal) -> bool:
        """True if ``price`` is at or beyond ``target`` in the profitable direction."""
        return (price - target) * self.side.sign >= ZERO


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of the position-sizing calculation (CLAUDE.md §6.3)."""

    quantity: int
    lots: int
    lot_size: int
    budget: Decimal
    risk_per_share: Decimal
    risk_inr: Decimal
    """Rupees genuinely at risk: ``quantity × R``. Always ≤ :attr:`budget` by construction."""


@dataclass(frozen=True, slots=True)
class LegPlan:
    """One Robo order. The executor translates this into a broker payload verbatim."""

    leg: Leg
    side: Side
    symbol: str
    trading_symbol: str
    token: str
    exchange: str
    quantity: int
    entry_price: Decimal
    stop_loss_price: Decimal
    target_price: Decimal
    squareoff_offset: Decimal
    stoploss_offset: Decimal
    trailing_stop_loss: Decimal
    order_tag: str

    @property
    def risk_inr(self) -> Decimal:
        """Rupees at risk on this leg alone."""
        return self.stoploss_offset * self.quantity


@dataclass(frozen=True, slots=True)
class BracketPlan:
    """A complete two-leg bracket, ready to place.

    Constructed only by :meth:`OrderBuilder.build`, which itself refuses to produce one unless
    the arithmetic is sound. It is *not* an authorisation to trade — that comes from
    :class:`~tachyon.risk.engine.RiskDecision`, and the executor requires both.
    """

    symbol: str
    side: Side
    geometry: OrderGeometry
    sizing: SizingResult
    legs: tuple[LegPlan, ...]

    @property
    def total_quantity(self) -> int:
        return sum(leg.quantity for leg in self.legs)

    @property
    def total_risk_inr(self) -> Decimal:
        """Total rupees at risk across both legs, if both stops are hit."""
        return sum((leg.risk_inr for leg in self.legs), start=ZERO)

    def leg(self, which: Leg) -> LegPlan | None:
        """The named leg, or ``None`` if the split did not produce it (single-lot positions)."""
        return next((plan for plan in self.legs if plan.leg is which), None)


# ──────────────────────────────────────────────────────────────────────────────
# The builder
# ──────────────────────────────────────────────────────────────────────────────


class OrderBuilder:
    """Turns an intent plus an ATR into a fully-specified two-leg bracket.

    Args:
        settings: resolved config, for the watchlist (tick size, lot size, token).
        clock: injected for testing; drives the order-tag date.
        sequencer: order-tag source. Share one per process so tags stay unique.

    Example::

        plan = builder.build(
            symbol="RELIANCE", side=Side.BUY,
            entry_price=Decimal("2500"), atr=8.40, headroom=Decimal("500"),
        )
        for leg in plan.legs:
            await executor.place(leg)
    """

    __slots__ = ("_clock", "_per_trade_risk", "_sequencer", "_settings")

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        clock: Clock = SYSTEM_CLOCK,
        sequencer: OrderTagSequencer | None = None,
        per_trade_risk: Decimal | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._sequencer = sequencer if sequencer is not None else OrderTagSequencer(clock)
        # Defaults to the §1 constant. A caller with a configured session budget passes the
        # derived amount (tachyon.risk.budget); anything non-positive falls back rather than
        # disabling sizing, because "no budget" must never resolve to "unlimited".
        self._per_trade_risk = (
            per_trade_risk
            if per_trade_risk is not None and per_trade_risk > ZERO
            else PER_TRADE_RISK_INR
        )

    @property
    def sequencer(self) -> OrderTagSequencer:
        return self._sequencer

    @property
    def per_trade_risk(self) -> Decimal:
        """Rupees a single position may risk. The §1 constant unless a budget was configured."""
        return self._per_trade_risk

    # ── instrument lookup ────────────────────────────────────────────────────

    def instrument(self, symbol: str) -> WatchlistItem:
        """Look up a watchlist entry.

        Raises:
            OrderRejected: the symbol is not on the watchlist. Trading an instrument absent
                from it is forbidden (CLAUDE.md §8.1), and this is the last line of defence
                before a payload is constructed.
        """
        item = self._settings.find_symbol(symbol)
        if item is None:
            raise OrderRejected("SYMBOL_NOT_ALLOWED", f"{symbol!r} is not on the watchlist")
        return item

    # ── budget ───────────────────────────────────────────────────────────────

    def budget(
        self,
        headroom: Decimal,
        sentinel_multiplier: Decimal | float = ONE,
    ) -> Decimal:
        """Rupees this trade may risk (CLAUDE.md §6.3).

        ``min(per_trade_risk, headroom)`` — so a day that has already spent most of its budget
        sizes the next trade against what remains, not against the full per-trade allowance.
        Late trades get smaller, which is the intended shape: the limit is a budget, not a
        per-trade allowance that resets.

        ``per_trade_risk`` is the §1 constant by default and the configured percentage of
        session capital when one is set. The ``min`` against headroom is what makes the daily
        limit binding regardless of which it is.
        """
        multiplier = clamp_multiplier(sentinel_multiplier)
        per_trade = self._per_trade_risk
        available = headroom if headroom < per_trade else per_trade
        if available <= ZERO:
            return ZERO
        return available * multiplier

    # ── geometry ─────────────────────────────────────────────────────────────

    def geometry(
        self,
        *,
        side: Side,
        entry_price: Decimal | float,
        atr: Decimal | float,
        tick_size: Decimal,
    ) -> OrderGeometry:
        """Compute every price and offset for the bracket (CLAUDE.md §6.1).

        Raises:
            OrderRejected: the ATR or entry price is undefined or non-positive, or the
                resulting stop distance rounds to less than one tick.
        """
        entry = to_decimal(entry_price, field="entry_price")
        atr_value = to_decimal(atr, field="atr")

        if entry <= ZERO:
            raise OrderRejected("BAD_ENTRY_PRICE", f"entry_price={entry} must be positive")
        if atr_value <= ZERO:
            # A zero ATR means the math engine has not warmed its candle window yet, or the
            # instrument has not moved. Either way there is no risk unit to size against.
            raise OrderRejected("BAD_ATR", f"atr={atr_value} must be positive")

        entry = round_to_tick(entry, tick_size)
        risk = round_to_tick(SL_ATR_MULTIPLIER * atr_value, tick_size)
        if risk < tick_size:
            raise OrderRejected(
                "RISK_BELOW_TICK",
                f"1.5 x ATR({atr_value}) = {risk} rounds below one tick ({tick_size})",
            )

        t1_offset = round_to_tick(T1_RR * risk, tick_size)
        t2_offset = round_to_tick(T2_RR * risk, tick_size)

        execution = self._settings.execution
        trail_step = round_to_tick(execution.trail_step_r * risk, tick_size)
        if trail_step < tick_size:
            trail_step = tick_size
        trail_activation = round_to_tick(execution.trail_activation_r * risk, tick_size)

        sign = side.sign
        return OrderGeometry(
            side=side,
            atr=atr_value,
            tick_size=tick_size,
            entry_price=entry,
            stop_loss_price=entry - sign * risk,
            target_1_price=entry + sign * t1_offset,
            target_2_price=entry + sign * t2_offset,
            risk_per_share=risk,
            stoploss_offset=risk,
            t1_offset=t1_offset,
            t2_offset=t2_offset,
            trail_step=trail_step,
            trail_activation=trail_activation,
        )

    # ── sizing ───────────────────────────────────────────────────────────────

    def size(
        self,
        *,
        risk_per_share: Decimal,
        budget: Decimal,
        lot_size: int = 1,
    ) -> SizingResult:
        """``qty = floor(budget / R)``, floored again to a whole number of lots.

        Raises:
            OrderRejected: the budget buys less than one lot. **This is the correct outcome**,
                not an error to work around — see CLAUDE.md §6.3.
        """
        if lot_size < 1:
            raise OrderRejected("BAD_LOT_SIZE", f"lot_size={lot_size} must be >= 1")
        if budget <= ZERO:
            raise OrderRejected("NO_BUDGET", f"budget={budget} — no risk capacity remaining")

        shares = floor_div_decimal(budget, risk_per_share)
        lots = shares // lot_size
        quantity = lots * lot_size

        if quantity < 1:
            raise OrderRejected(
                "QUANTITY_BELOW_ONE",
                f"budget {budget} / risk-per-share {risk_per_share} = {shares} share(s), "
                f"below one lot of {lot_size} — no trade (never round up)",
            )

        return SizingResult(
            quantity=quantity,
            lots=lots,
            lot_size=lot_size,
            budget=budget,
            risk_per_share=risk_per_share,
            risk_inr=risk_per_share * quantity,
        )

    # ── the 60/40 split ──────────────────────────────────────────────────────

    def split_lots(self, lots: int) -> tuple[int, int]:
        """Divide lots between leg A (T1) and leg B (T2) per :data:`T1_QTY_FRACTION`.

        Split in lots rather than shares so both legs remain whole multiples of the lot size —
        an F&O order for a fractional lot is rejected outright by the exchange.

        A single lot yields ``(1, 0)``: the whole position books at the nearer target. That is
        the conservative resolution of an indivisible position, and it falls out of the
        arithmetic rather than being special-cased.
        """
        if lots < 1:
            return (0, 0)
        lots_a = int((Decimal(lots) * T1_QTY_FRACTION).quantize(ONE, rounding=ROUND_HALF_UP))
        lots_a = min(max(lots_a, 1), lots)
        return (lots_a, lots - lots_a)

    # ── assembly ─────────────────────────────────────────────────────────────

    def build(
        self,
        *,
        symbol: str,
        side: Side,
        entry_price: Decimal | float,
        atr: Decimal | float,
        headroom: Decimal,
        sentinel_multiplier: Decimal | float = ONE,
    ) -> BracketPlan:
        """Build the complete two-leg bracket.

        Args:
            symbol: watchlist symbol. Anything else is rejected (CLAUDE.md §8.1).
            side: entry direction.
            entry_price: intended entry. Rounded to the instrument tick.
            atr: ATR(14) on 5-minute candles, from the math engine.
            headroom: remaining daily loss budget, from :class:`~tachyon.risk.tracker.PnLTracker`.
            sentinel_multiplier: advisory downsizing, clamped to ``[0, 1]``.

        Raises:
            OrderRejected: for any reason the order should not exist. Every path out of this
                method either returns a fully-specified plan or refuses.
        """
        item = self.instrument(symbol)
        geometry = self.geometry(
            side=side,
            entry_price=entry_price,
            atr=atr,
            tick_size=item.tick_size,
        )
        sizing = self.size(
            risk_per_share=geometry.risk_per_share,
            budget=self.budget(headroom, sentinel_multiplier),
            lot_size=item.lot_size,
        )

        lots_a, lots_b = self.split_lots(sizing.lots)
        trading_symbol = trading_symbol_for(item)

        specs = (
            (Leg.A, lots_a * item.lot_size, geometry.t1_offset, geometry.target_1_price),
            (Leg.B, lots_b * item.lot_size, geometry.t2_offset, geometry.target_2_price),
        )
        legs = tuple(
            LegPlan(
                leg=leg,
                side=side,
                symbol=symbol,
                trading_symbol=trading_symbol,
                token=item.token,
                exchange=item.exchange,
                quantity=quantity,
                entry_price=geometry.entry_price,
                stop_loss_price=geometry.stop_loss_price,
                target_price=target_price,
                squareoff_offset=offset,
                stoploss_offset=geometry.stoploss_offset,
                trailing_stop_loss=geometry.trail_step,
                order_tag=self._sequencer.next_tag(),
            )
            for leg, quantity, offset, target_price in specs
            if quantity > 0
        )

        plan = BracketPlan(symbol=symbol, side=side, geometry=geometry, sizing=sizing, legs=legs)

        _log.info(
            "execution.plan_built",
            symbol=symbol,
            side=side,
            quantity=plan.total_quantity,
            legs=len(plan.legs),
            entry=str(geometry.entry_price),
            stop=str(geometry.stop_loss_price),
            t1=str(geometry.target_1_price),
            t2=str(geometry.target_2_price),
            risk_per_share=str(geometry.risk_per_share),
            risk_inr=str(plan.total_risk_inr),
            budget=str(sizing.budget),
        )
        return plan
