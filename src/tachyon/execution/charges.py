"""Intraday equity transaction costs — CLAUDE.md §1.2.

The daily loss limit is enforced against P&L **inclusive of estimated charges**. That is not a
detail: on a ₹500 budget with a ₹100-per-trade risk unit, round-trip costs of ₹40–60 are a
tenth of the day's budget. A limit enforced on gross P&L would let the account bleed past ₹500
in real money while the tracker still read ₹460.

These are *estimates*, and the estimate is deliberately biased upward:

* brokerage is charged at the full cap rather than the percentage where they are close;
* charges are rounded **up** to the paisa.

Erring high trips the kill switch slightly early, which costs a marginal trade. Erring low lets
the real loss exceed ₹500, which is the one outcome §1.2 exists to prevent. The broker's
contract note remains the record of truth; this is what the *live* limit is enforced against
between now and when that note arrives the next morning.

Rates below are Angel One's published intraday-equity structure. **They change.** They are
config-driven for exactly that reason, and a rate that has drifted shows up as a discrepancy
between the journal's estimate and the contract note — which is why both are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Final

ZERO: Final[Decimal] = Decimal("0")
PAISA: Final[Decimal] = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class ChargeSchedule:
    """Cost rates for one instrument class. Defaults: NSE intraday equity.

    Every rate is a fraction of turnover unless named otherwise.
    """

    brokerage_rate: Decimal = Decimal("0.0003")
    """0.03 % of turnover per executed order..."""

    brokerage_cap: Decimal = Decimal("20")
    """...capped at ₹20 per order. Whichever is lower — but see :func:`estimate_charges`."""

    stt_sell_rate: Decimal = Decimal("0.00025")
    """Securities Transaction Tax: 0.025 % on the **sell** side only, intraday."""

    exchange_rate: Decimal = Decimal("0.0000297")
    """NSE transaction charge, both sides."""

    sebi_rate: Decimal = Decimal("0.000001")
    """SEBI turnover fee, ₹10 per crore."""

    stamp_buy_rate: Decimal = Decimal("0.00003")
    """Stamp duty: 0.003 % on the **buy** side only, intraday."""

    gst_rate: Decimal = Decimal("0.18")
    """18 % GST on (brokerage + exchange + SEBI)."""


DEFAULT_SCHEDULE: Final[ChargeSchedule] = ChargeSchedule()


@dataclass(frozen=True, slots=True)
class ChargeBreakdown:
    """Itemised costs, so a discrepancy against the contract note is diagnosable."""

    brokerage: Decimal
    stt: Decimal
    exchange: Decimal
    sebi: Decimal
    stamp: Decimal
    gst: Decimal

    @property
    def total(self) -> Decimal:
        return self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst


def _ceil_paisa(value: Decimal) -> Decimal:
    """Round up to the paisa. Upward, always — see the module docstring."""
    return value.quantize(PAISA, rounding=ROUND_CEILING)


def estimate_charges(
    *,
    buy_turnover: Decimal,
    sell_turnover: Decimal,
    orders: int = 2,
    schedule: ChargeSchedule = DEFAULT_SCHEDULE,
) -> ChargeBreakdown:
    """Estimate the round-trip cost of an intraday equity position.

    Args:
        buy_turnover: total rupee value bought (price × quantity, summed over fills).
        sell_turnover: total rupee value sold.
        orders: executed orders in the round trip. A two-leg bracket that fills entry and exit
            on both legs is four, not two — brokerage is per order, and undercounting it is
            the easiest way to underestimate the total.
        schedule: rates to apply.

    Returns:
        An itemised :class:`ChargeBreakdown`. Negative turnover is treated as zero rather than
        producing a negative charge — a cost that reduces the loss would be a very expensive
        sign error.
    """
    buy = buy_turnover if buy_turnover > ZERO else ZERO
    sell = sell_turnover if sell_turnover > ZERO else ZERO
    turnover = buy + sell
    order_count = max(0, orders)

    per_order = min(schedule.brokerage_cap, _ceil_paisa(turnover / 2 * schedule.brokerage_rate))
    brokerage = _ceil_paisa(per_order * order_count)

    stt = _ceil_paisa(sell * schedule.stt_sell_rate)
    exchange = _ceil_paisa(turnover * schedule.exchange_rate)
    sebi = _ceil_paisa(turnover * schedule.sebi_rate)
    stamp = _ceil_paisa(buy * schedule.stamp_buy_rate)
    gst = _ceil_paisa((brokerage + exchange + sebi) * schedule.gst_rate)

    return ChargeBreakdown(
        brokerage=brokerage, stt=stt, exchange=exchange, sebi=sebi, stamp=stamp, gst=gst
    )
