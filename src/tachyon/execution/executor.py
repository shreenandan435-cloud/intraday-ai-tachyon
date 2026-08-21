"""Robo (bracket) order execution — CLAUDE.md §6.

Turns a :class:`~tachyon.execution.builder.BracketPlan` into Angel One Robo Order payloads and
sends them. Three rules shape everything below.

**1. No order without an authorisation.** :meth:`RoboExecutor.open_position` requires a passing
:class:`~tachyon.risk.engine.RiskDecision` *and* re-runs the gate itself immediately before
transmitting. Re-running is not paranoia about the caller: between a signal being generated and
an order reaching the broker, the feed can go stale, the loss limit can trip, or 15:00 can pass.
The gate is cheap; the alternative is entering a position the risk engine would now refuse.

**2. Offsets, not prices.** SmartAPI Robo Orders take ``squareoff``, ``stoploss`` and
``trailingStopLoss`` as **point offsets from the entry price**, not absolute levels. Sending
absolute prices produces an order that is accepted and completely wrong — a stop 2500 points
away on a ₹2500 stock. :meth:`RoboExecutor.payload_for` is the single place that conversion
happens, and :func:`assert_offsets_sane` refuses payloads whose numbers cannot be right.

**3. The stop only ever moves toward profit.** Widening a stop after entry is forbidden
(CLAUDE.md §8.1). :func:`assert_stop_not_widened` enforces it in code, so the breakeven trail on
leg B cannot be inverted by a sign error.

Trailing is broker-native (``trailingStopLoss``, LTP-jump based). There is no client-side
trailing loop anywhere in this system: if our process dies, the trail must still be live at the
exchange (CLAUDE.md §6.2).

Square-off
----------
:meth:`RoboExecutor.flatten_everything` is the action the 15:15 watchdog drives. It returns
normally **only when the broker reports zero working orders and zero net positions** — being
flat is the success criterion, not having sent the requests. Anything else raises, and the
watchdog retries every two seconds until it succeeds (CLAUDE.md §1.1).

It will not submit a second market exit for an instrument whose first exit was accepted or
whose outcome is unknown. A duplicate exit does not flatten a position twice; it *reverses* it,
creating brand-new naked risk at the moment the system is trying to have none.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import TradingMode
from tachyon.core.logger import get_logger
from tachyon.execution.api import (
    BrokerOrder,
    BrokerPosition,
    PaperModeError,
    SmartApiClient,
    SmartApiError,
    UnknownOrderOutcomeError,
)
from tachyon.execution.builder import (
    BracketPlan,
    Leg,
    LegPlan,
    OrderBuilder,
    OrderGeometry,
    OrderRejected,
    Side,
    round_to_tick,
)
from tachyon.execution.journal import OrderJournal
from tachyon.risk.engine import RiskDecision, RiskEngine
from tachyon.risk.tracker import PositionRegistry

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")

VARIETY_ROBO: Final[str] = "ROBO"
VARIETY_NORMAL: Final[str] = "NORMAL"
PRODUCT_BO: Final[str] = "BO"
ORDER_TYPE_LIMIT: Final[str] = "LIMIT"
ORDER_TYPE_MARKET: Final[str] = "MARKET"
ORDER_TYPE_SL_LIMIT: Final[str] = "STOPLOSS_LIMIT"
DURATION_DAY: Final[str] = "DAY"

#: Ceiling on a single price offset relative to the entry price. A ``stoploss`` of 40 % of the
#: instrument's price is not a wide stop, it is a units bug — almost certainly an absolute price
#: sent where an offset belongs. Refusing is free; the order would have been nonsense.
MAX_OFFSET_FRACTION: Final[Decimal] = Decimal("0.40")

#: How long the watchdog thread will block waiting for an async flatten to complete. Shorter
#: than the operator would like, but the watchdog retries — a stuck call must not wedge it.
SQUARE_OFF_TIMEOUT_SECONDS: Final[float] = 20.0


class StopWidenedError(RuntimeError):
    """An attempt was made to move a stop away from the entry price (CLAUDE.md §8.1)."""


class OffsetSanityError(RuntimeError):
    """A Robo payload's offsets cannot be point offsets — refusing to transmit."""


class NotFlatError(RuntimeError):
    """Square-off ran but the broker still reports working orders or open positions."""


# ──────────────────────────────────────────────────────────────────────────────
# Invariants
# ──────────────────────────────────────────────────────────────────────────────


def assert_offsets_sane(entry_price: Decimal, offsets: dict[str, Decimal]) -> None:
    """Refuse offsets that cannot be point offsets from ``entry_price``.

    Catches the single most expensive mistake available in this module: passing an absolute
    price where SmartAPI expects a distance. Such an order is *accepted* by the broker, so
    nothing downstream would complain.

    Raises:
        OffsetSanityError: an offset is non-positive, or exceeds
            :data:`MAX_OFFSET_FRACTION` of the entry price.
    """
    ceiling = entry_price * MAX_OFFSET_FRACTION
    for name, value in offsets.items():
        if not value.is_finite() or value <= ZERO:
            raise OffsetSanityError(f"{name}={value} must be a positive, finite point offset")
        if value > ceiling:
            raise OffsetSanityError(
                f"{name}={value} exceeds {MAX_OFFSET_FRACTION:%} of the entry price "
                f"({entry_price}) — this looks like an absolute price, not an offset"
            )


def assert_stop_not_widened(
    side: Side,
    current_stop: Decimal,
    new_stop: Decimal,
) -> None:
    """Refuse a stop that moves away from profit (CLAUDE.md §8.1).

    For a long, a stop may only move up; for a short, only down. Equal is permitted so a
    no-op modification is not an error.

    Raises:
        StopWidenedError: the new stop increases the distance to the entry.
    """
    moved = (new_stop - current_stop) * side.sign
    if moved < ZERO:
        raise StopWidenedError(
            f"{side} stop may only move toward profit: {current_stop} -> {new_stop} widens risk"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LegResult:
    """Outcome of placing one Robo leg."""

    leg: Leg
    order_tag: str
    quantity: int
    accepted: bool
    order_id: str = ""
    error: str = ""
    outcome_unknown: bool = False
    """True when the call failed after transmission. Reconcile; never re-send."""


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """What happened when a bracket was submitted."""

    symbol: str
    side: Side
    at_ist: str
    plan: BracketPlan | None
    results: tuple[LegResult, ...] = ()
    simulated: bool = False
    rejected_reason: str = ""
    rejected_detail: str = ""

    @property
    def placed(self) -> bool:
        """True if at least one leg was accepted (or simulated in PAPER)."""
        return any(result.accepted for result in self.results)

    @property
    def quantity_placed(self) -> int:
        return sum(result.quantity for result in self.results if result.accepted)

    @property
    def needs_reconciliation(self) -> bool:
        """True if any leg's outcome is unknown — the broker's state must be re-read."""
        return any(result.outcome_unknown for result in self.results)


@dataclass(frozen=True, slots=True)
class FlattenReport:
    """Outcome of one square-off attempt."""

    at_ist: str
    orders_cancelled: int = 0
    exits_submitted: int = 0
    residual_orders: int = 0
    residual_positions: int = 0
    errors: tuple[str, ...] = ()
    simulated: bool = False

    @property
    def is_flat(self) -> bool:
        """True only when the broker reports nothing working and nothing open."""
        return self.residual_orders == 0 and self.residual_positions == 0


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────


class RoboExecutor:
    """Places, modifies and flattens Angel One Robo Orders.

    Args:
        client: the rate-limited broker client. ``None`` is permitted only in PAPER.
        builder: order geometry and sizing.
        risk: the veto gate. Re-consulted immediately before transmission.
        positions: local exposure registry, updated on acceptance.
        journal: append-only audit record.
        settings: resolved config.
        mode: PAPER simulates and never touches the broker.

    Example::

        decision = risk.evaluate("RELIANCE")
        report = await executor.open_position(
            decision, side=Side.BUY, entry_price=Decimal("2500"), atr=8.4,
        )
    """

    __slots__ = (
        "_builder",
        "_client",
        "_clock",
        "_exit_attempted",
        "_journal",
        "_mode",
        "_positions",
        "_risk",
        "_settings",
    )

    def __init__(
        self,
        *,
        client: SmartApiClient | None,
        builder: OrderBuilder,
        risk: RiskEngine,
        positions: PositionRegistry,
        journal: OrderJournal | None = None,
        settings: Settings | None = None,
        mode: TradingMode | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._client = client
        self._builder = builder
        self._risk = risk
        self._positions = positions
        self._journal = journal if journal is not None else OrderJournal(clock=clock)
        self._mode = mode if mode is not None else self._settings.trading_mode
        self._clock = clock
        self._exit_attempted: set[str] = set()

        if self._mode is TradingMode.LIVE and client is None:
            raise ValueError("LIVE mode requires a SmartApiClient — refusing to run blind")

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._mode is TradingMode.LIVE

    # ── payload construction ─────────────────────────────────────────────────

    def payload_for(self, leg: LegPlan) -> dict[str, str]:
        """Exact Angel One Robo Order payload for one leg (CLAUDE.md §6.1, §6.2).

        ``squareoff``, ``stoploss`` and ``trailingStopLoss`` are **point offsets from the entry
        price**. The stop rides in the same call as the entry — placing an order without one
        attached is forbidden (CLAUDE.md §8.1) — which is precisely what a Robo order is for.

        Raises:
            OffsetSanityError: the offsets cannot be distances from this entry price.
        """
        assert_offsets_sane(
            leg.entry_price,
            {
                "squareoff": leg.squareoff_offset,
                "stoploss": leg.stoploss_offset,
                "trailingStopLoss": leg.trailing_stop_loss,
            },
        )
        return {
            "variety": VARIETY_ROBO,
            "tradingsymbol": leg.trading_symbol,
            "symboltoken": leg.token,
            "transactiontype": leg.side.value,
            "exchange": leg.exchange,
            "ordertype": ORDER_TYPE_LIMIT,
            "producttype": PRODUCT_BO,
            "duration": DURATION_DAY,
            "price": str(leg.entry_price),
            "squareoff": str(leg.squareoff_offset),
            "stoploss": str(leg.stoploss_offset),
            "trailingStopLoss": str(leg.trailing_stop_loss),
            "quantity": str(leg.quantity),
            "ordertag": leg.order_tag,
        }

    def exit_payload_for(self, position: BrokerPosition) -> dict[str, str]:
        """Market order that closes ``position`` — the square-off instrument.

        MARKET, not LIMIT: at 15:15 the priority is being flat, and a limit that does not fill
        leaves an overnight position. Slippage is a cost; an unflattened intraday position is a
        margin call.
        """
        closing_side = Side.SELL if position.net_quantity > 0 else Side.BUY
        return {
            "variety": VARIETY_NORMAL,
            "tradingsymbol": position.trading_symbol,
            "symboltoken": position.token,
            "transactiontype": closing_side.value,
            "exchange": position.exchange,
            "ordertype": ORDER_TYPE_MARKET,
            "producttype": position.product_type or PRODUCT_BO,
            "duration": DURATION_DAY,
            "price": "0",
            "quantity": str(abs(position.net_quantity)),
            "ordertag": self._builder.sequencer.next_tag(),
        }

    # ── entry ────────────────────────────────────────────────────────────────

    async def open_position(
        self,
        decision: RiskDecision,
        *,
        side: Side,
        entry_price: Decimal | float,
        atr: Decimal | float,
        headroom: Decimal,
        sentinel_multiplier: Decimal | float = Decimal("1"),
    ) -> ExecutionReport:
        """Build and place a two-leg bracket.

        Args:
            decision: an **allowed** :class:`~tachyon.risk.engine.RiskDecision` for this symbol.
            side: entry direction.
            entry_price: intended entry, rounded to the instrument tick.
            atr: ATR(14) on 5-minute candles.
            headroom: remaining daily loss budget.
            sentinel_multiplier: advisory downsizing, clamped to ``[0, 1]``.

        Never raises for an ordinary refusal — a rejected entry comes back as a report with
        :attr:`ExecutionReport.rejected_reason` set, because a missed trade is a normal
        outcome and must not take down the strategy loop.
        """
        at = now_ist(self._clock).isoformat(timespec="milliseconds")
        symbol = decision.symbol

        if not decision.allowed:
            return self._refuse(symbol, side, at, "NOT_AUTHORISED", str(decision.reason))

        # The gate is re-run here, not merely trusted, because the world moves between a signal
        # and a socket write: the feed can go stale, the limit can trip, 15:00 can pass.
        fresh = self._risk.evaluate(symbol)
        if not fresh.allowed:
            return self._refuse(symbol, side, at, str(fresh.reason), fresh.detail)

        try:
            plan = self._builder.build(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                atr=atr,
                headroom=headroom,
                sentinel_multiplier=sentinel_multiplier,
            )
        except OrderRejected as exc:
            return self._refuse(symbol, side, at, exc.reason, exc.detail)

        if not plan.legs:
            return self._refuse(symbol, side, at, "NO_LEGS", "sizing produced no tradeable leg")

        return await self.place_robo_order(plan, fresh)

    async def place_robo_order(self, plan: BracketPlan, decision: RiskDecision) -> ExecutionReport:
        """Transmit a pre-built bracket. PAPER simulates; LIVE sends.

        Args:
            plan: a complete bracket from :class:`~tachyon.execution.builder.OrderBuilder`.
            decision: the authorisation covering ``plan.symbol``. **Required**, and verified —
                this is the last point at which an order can be stopped, and it is reachable
                from outside :meth:`open_position`, so it cannot take the caller's word for it.

        Never raises for an ordinary refusal.
        """
        at = now_ist(self._clock).isoformat(timespec="milliseconds")

        if not decision.allowed:
            return self._refuse(plan.symbol, plan.side, at, "NOT_AUTHORISED", str(decision.reason))
        if decision.symbol != plan.symbol:
            # An authorisation for RELIANCE does not authorise an order in HDFCBANK. This is
            # cheap to check and catches a whole class of wiring bug that would otherwise
            # present as "the risk engine approved it".
            return self._refuse(
                plan.symbol,
                plan.side,
                at,
                "AUTHORISATION_MISMATCH",
                f"decision authorises {decision.symbol!r}, plan is for {plan.symbol!r}",
            )
        if not plan.legs:
            return self._refuse(
                plan.symbol, plan.side, at, "NO_LEGS", "plan carries no tradeable leg"
            )

        if not self.is_live:
            return self._simulate(plan, at)
        return await self._transmit(plan, at)

    def _refuse(
        self, symbol: str, side: Side, at: str, reason: str, detail: str
    ) -> ExecutionReport:
        self._journal.decision("entry_refused", symbol=symbol, reason=reason, detail=detail)
        _log.info("execution.entry_refused", symbol=symbol, reason=reason, detail=detail)
        return ExecutionReport(
            symbol=symbol,
            side=side,
            at_ist=at,
            plan=None,
            rejected_reason=reason,
            rejected_detail=detail,
        )

    def _simulate(self, plan: BracketPlan, at: str) -> ExecutionReport:
        """PAPER mode: record exactly what would have been sent, and send nothing."""
        results = tuple(
            LegResult(
                leg=leg.leg,
                order_tag=leg.order_tag,
                quantity=leg.quantity,
                accepted=True,
                order_id=f"PAPER-{leg.order_tag}",
            )
            for leg in plan.legs
        )
        for leg in plan.legs:
            self._journal.decision(
                "paper_order", order_tag=leg.order_tag, payload=self.payload_for(leg)
            )
        self._positions.record_entry(plan.symbol, plan.total_quantity)
        _log.info(
            "execution.paper_entry",
            symbol=plan.symbol,
            side=plan.side,
            quantity=plan.total_quantity,
            legs=len(plan.legs),
        )
        return ExecutionReport(
            symbol=plan.symbol,
            side=plan.side,
            at_ist=at,
            plan=plan,
            results=results,
            simulated=True,
        )

    async def _transmit(self, plan: BracketPlan, at: str) -> ExecutionReport:
        """Send each leg, recording every outcome. Never raises."""
        assert self._client is not None  # noqa: S101 - guaranteed by the LIVE-mode constructor
        results: list[LegResult] = []

        for leg in plan.legs:
            try:
                payload = self.payload_for(leg)
            except OffsetSanityError as exc:
                _log.critical(
                    "execution.offsets_insane",
                    symbol=plan.symbol,
                    leg=leg.leg,
                    error=str(exc),
                    action="leg not transmitted",
                )
                results.append(
                    LegResult(leg.leg, leg.order_tag, leg.quantity, accepted=False, error=str(exc))
                )
                continue

            try:
                response = await self._client.place_order(payload)
            except UnknownOrderOutcomeError as exc:
                results.append(
                    LegResult(
                        leg.leg,
                        leg.order_tag,
                        leg.quantity,
                        accepted=False,
                        error=str(exc),
                        outcome_unknown=True,
                    )
                )
                _log.critical(
                    "execution.leg_outcome_unknown",
                    symbol=plan.symbol,
                    leg=leg.leg,
                    order_tag=leg.order_tag,
                    action="RECONCILE — this leg may be live at the broker",
                )
                # Stop after an unknown outcome. Sending leg B on top of a leg A that may or may
                # not exist compounds an unknown position into an unknown *and* wrongly-sized one.
                break
            except (SmartApiError, PaperModeError) as exc:
                results.append(
                    LegResult(leg.leg, leg.order_tag, leg.quantity, accepted=False, error=str(exc))
                )
                _log.error(
                    "execution.leg_rejected",
                    symbol=plan.symbol,
                    leg=leg.leg,
                    order_tag=leg.order_tag,
                    error=str(exc),
                )
                continue

            results.append(
                LegResult(
                    leg=leg.leg,
                    order_tag=leg.order_tag,
                    quantity=leg.quantity,
                    accepted=True,
                    order_id=str(response.get("orderid", "")),
                )
            )
            _log.info(
                "execution.leg_placed",
                symbol=plan.symbol,
                leg=leg.leg,
                order_tag=leg.order_tag,
                order_id=str(response.get("orderid", "")),
                quantity=leg.quantity,
            )

        report = ExecutionReport(
            symbol=plan.symbol,
            side=plan.side,
            at_ist=at,
            plan=plan,
            results=tuple(results),
        )
        if report.placed:
            self._positions.record_entry(plan.symbol, report.quantity_placed)
        return report

    # ── stop management ──────────────────────────────────────────────────────

    async def move_stop_to_breakeven(
        self,
        *,
        order_id: str,
        geometry: OrderGeometry,
        current_stop: Decimal,
        quantity: int,
        trading_symbol: str,
        token: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Trail leg B's stop to breakeven + 1 tick after leg A books at T1 (CLAUDE.md §6.1).

        Raises:
            StopWidenedError: the move would increase risk. Never caught and ignored — a
                widened stop is the failure mode this whole rule exists to prevent.
        """
        new_stop = round_to_tick(geometry.breakeven_stop(), geometry.tick_size)
        assert_stop_not_widened(geometry.side, current_stop, new_stop)

        if not self.is_live:
            self._journal.decision(
                "paper_stop_moved",
                order_id=order_id,
                from_stop=str(current_stop),
                to_stop=str(new_stop),
            )
            _log.info(
                "execution.paper_stop_moved",
                order_id=order_id,
                to_stop=str(new_stop),
            )
            return {"orderid": order_id, "simulated": True}

        assert self._client is not None  # noqa: S101 - guaranteed by the LIVE-mode constructor
        payload = {
            "variety": VARIETY_ROBO,
            "orderid": order_id,
            "tradingsymbol": trading_symbol,
            "symboltoken": token,
            "exchange": exchange,
            "ordertype": ORDER_TYPE_SL_LIMIT,
            "producttype": PRODUCT_BO,
            "duration": DURATION_DAY,
            "price": str(new_stop),
            "triggerprice": str(new_stop),
            "quantity": str(quantity),
        }
        response = await self._client.modify_order(payload)
        _log.info(
            "execution.stop_moved_to_breakeven",
            order_id=order_id,
            from_stop=str(current_stop),
            to_stop=str(new_stop),
        )
        return response

    # ── square-off ───────────────────────────────────────────────────────────

    async def flatten_everything(self) -> FlattenReport:
        """Cancel every working order and exit every open position (CLAUDE.md §1.1).

        Returns a report whose :attr:`FlattenReport.is_flat` is True only when the broker
        confirms nothing is left. The watchdog retries on anything else.

        Raises:
            SmartApiError: the broker could not be queried. The watchdog treats this as a
                failed attempt and retries — we cannot prove we are flat, so we are not.
        """
        at = now_ist(self._clock).isoformat(timespec="milliseconds")

        if not self.is_live or self._client is None:
            self._positions.reset_session()
            self._journal.decision("paper_square_off")
            _log.critical("execution.paper_square_off", action="PAPER — nothing to flatten")
            return FlattenReport(at_ist=at, simulated=True)

        errors: list[str] = []
        cancelled = await self._cancel_working_orders(errors)
        exits = await self._exit_open_positions(errors)

        # Re-read the broker rather than trusting what we just sent. "We issued the cancels" is
        # not the same fact as "nothing is working", and only the second one ends the retry.
        residual_orders = sum(1 for order in await self._orders() if order.is_open)
        residual_positions = len(await self._positions_open())

        report = FlattenReport(
            at_ist=at,
            orders_cancelled=cancelled,
            exits_submitted=exits,
            residual_orders=residual_orders,
            residual_positions=residual_positions,
            errors=tuple(errors),
        )
        self._journal.decision(
            "square_off_attempt",
            orders_cancelled=cancelled,
            exits_submitted=exits,
            residual_orders=residual_orders,
            residual_positions=residual_positions,
            errors=list(errors),
        )
        if report.is_flat:
            self._positions.reset_session()
            _log.critical("execution.square_off_flat", orders_cancelled=cancelled, exits=exits)
        else:
            _log.critical(
                "execution.square_off_incomplete",
                residual_orders=residual_orders,
                residual_positions=residual_positions,
                errors=errors,
                action="watchdog will retry",
            )
        return report

    async def _orders(self) -> tuple[BrokerOrder, ...]:
        assert self._client is not None  # noqa: S101 - LIVE-only path
        return tuple(BrokerOrder.from_row(row) for row in await self._client.order_book())

    async def _positions_open(self) -> tuple[BrokerPosition, ...]:
        assert self._client is not None  # noqa: S101 - LIVE-only path
        rows = await self._client.positions()
        return tuple(p for p in (BrokerPosition.from_row(row) for row in rows) if p.is_open)

    async def _cancel_working_orders(self, errors: list[str]) -> int:
        """Cancel every order that can still execute — ours or not (CLAUDE.md §1.1)."""
        assert self._client is not None  # noqa: S101 - LIVE-only path
        cancelled = 0
        for order in await self._orders():
            if not order.is_open:
                continue
            try:
                await self._client.cancel_order(order.order_id, variety=order.variety or "NORMAL")
            except UnknownOrderOutcomeError as exc:
                # A cancel whose outcome is unknown is safe to leave alone: the verification
                # pass below re-reads the book, and a still-open order simply gets cancelled
                # again on the next retry. Re-sending immediately would spam a struggling API.
                errors.append(f"cancel {order.order_id}: {exc}")
            except SmartApiError as exc:
                errors.append(f"cancel {order.order_id}: {exc}")
                _log.error("execution.cancel_failed", order_id=order.order_id, error=str(exc))
            else:
                cancelled += 1
        return cancelled

    async def _exit_open_positions(self, errors: list[str]) -> int:
        """Submit one market exit per open instrument. At most one per square-off.

        A second exit for the same instrument does not close it twice — it opens an equal and
        opposite position. So a token is marked as attempted the moment a placement is accepted
        *or* becomes unknown, and only a definitively rejected placement is retried.
        """
        assert self._client is not None  # noqa: S101 - LIVE-only path
        submitted = 0
        for position in await self._positions_open():
            key = position.token or position.trading_symbol
            if key in self._exit_attempted:
                _log.critical(
                    "execution.exit_still_open",
                    trading_symbol=position.trading_symbol,
                    net_quantity=position.net_quantity,
                    action="exit already submitted — NOT re-sending (a duplicate would reverse it)",
                )
                errors.append(f"{position.trading_symbol}: exit submitted but position persists")
                continue

            payload = self.exit_payload_for(position)
            try:
                await self._client.place_order(payload)
            except UnknownOrderOutcomeError as exc:
                # Unknown means it may well be live. Marking it attempted is the safe choice.
                self._exit_attempted.add(key)
                errors.append(f"exit {position.trading_symbol}: {exc}")
                _log.critical(
                    "execution.exit_outcome_unknown",
                    trading_symbol=position.trading_symbol,
                    action="treating as submitted — reconcile before any further action",
                )
            except SmartApiError as exc:
                # A definitive rejection: nothing reached the market, so a retry is correct.
                errors.append(f"exit {position.trading_symbol}: {exc}")
                _log.error(
                    "execution.exit_rejected",
                    trading_symbol=position.trading_symbol,
                    error=str(exc),
                )
            else:
                self._exit_attempted.add(key)
                submitted += 1
                _log.critical(
                    "execution.exit_submitted",
                    trading_symbol=position.trading_symbol,
                    quantity=abs(position.net_quantity),
                )
        return submitted

    def square_off_action(
        self,
        loop: asyncio.AbstractEventLoop,
        timeout: float = SQUARE_OFF_TIMEOUT_SECONDS,
    ) -> Callable[[], None]:
        """Adapt :meth:`flatten_everything` for the square-off watchdog thread.

        The watchdog is a plain non-daemon thread (CLAUDE.md §1.1, §2.2) and the executor is
        async, so the coroutine is submitted to the Brain's event loop and waited on from the
        thread. Blocking that thread is fine and intended — it has one job.

        The returned callable raises unless the account is confirmed flat, which is exactly
        what :meth:`~tachyon.risk.watchdog.SquareOffWatchdog` needs in order to keep retrying.
        """

        def action() -> None:
            future = asyncio.run_coroutine_threadsafe(self.flatten_everything(), loop)
            report = future.result(timeout=timeout)
            if not report.is_flat:
                raise NotFlatError(
                    f"square-off incomplete: {report.residual_orders} working order(s), "
                    f"{report.residual_positions} open position(s); errors={report.errors}"
                )

        return action


@dataclass(slots=True)
class OpenBracket:
    """Local record of a live two-leg bracket, so leg B can be trailed when leg A books.

    Held in the Brain rather than re-derived from the order book on every tick: the order book
    endpoint is rate-limited to roughly one call a second, which is far too slow to notice a
    T1 fill promptly.
    """

    symbol: str
    plan: BracketPlan
    leg_a_order_id: str = ""
    leg_b_order_id: str = ""
    leg_a_filled: bool = False
    leg_b_stop_moved: bool = False
    current_stop: Decimal = ZERO
    order_ids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_report(cls, report: ExecutionReport) -> OpenBracket | None:
        """Build a record from a successful entry, or ``None`` if nothing was placed."""
        if report.plan is None or not report.placed:
            return None
        ids = {result.leg.value: result.order_id for result in report.results if result.accepted}
        return cls(
            symbol=report.symbol,
            plan=report.plan,
            leg_a_order_id=ids.get(Leg.A.value, ""),
            leg_b_order_id=ids.get(Leg.B.value, ""),
            current_stop=report.plan.geometry.stop_loss_price,
            order_ids=ids,
        )
