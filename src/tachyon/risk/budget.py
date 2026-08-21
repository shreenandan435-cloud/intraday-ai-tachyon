"""Session capital budget — the dynamic source of the daily and per-trade risk limits.

CLAUDE.md §1 fixed ``DAILY_LOSS_LIMIT_INR`` at ₹500 and ``PER_TRADE_RISK_INR`` at ₹100. This
module makes both **derived** from a configured account size instead:

```
daily_loss_limit = session_capital_budget × max_daily_drawdown_pct / 100
per_trade_risk   = session_capital_budget × per_trade_risk_pct     / 100
```

The constants are not deleted and are not overridable — nothing may write them, and
``config.py`` still refuses to boot if a config source mentions them by name. They become the
**fallback**: the values used when no budget is configured, and the values used when the
configured budget is unusable.

That direction is the whole safety argument. Everywhere else in this system unknown state
produces a veto (§4); here it produces the *smaller* of the two possible worlds. A budget that
fails to parse, is negative, or is absent falls back to ₹500/₹100 — never to something larger,
and never to "unlimited". If we cannot prove how much may be lost, the answer is the old,
smaller number.

Two things this module deliberately does not do:

* **It does not cap the configured values.** An operator who sets a 5 % drawdown on ₹5,00,000
  gets a ₹25,000 limit. Silently clamping a number the operator explicitly chose would be
  worse than honouring it: they would believe a limit was in force that was not.
* **It does not enforce the 20 % relationship.** §6.3's ₹100-against-₹500 shape means no single
  loser consumes more than a fifth of the day. A configuration that breaks that ratio is
  reported at ``WARNING`` with the arithmetic spelled out, and then honoured. It is a risk
  decision, and risk decisions belong to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final

from tachyon.core.constants import DAILY_LOSS_LIMIT_INR, PER_TRADE_RISK_INR
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

ZERO: Final[Decimal] = Decimal("0")
HUNDRED: Final[Decimal] = Decimal("100")

#: Rupee precision. Sub-paise thresholds are not meaningful and make logs unreadable.
_PAISE: Final[Decimal] = Decimal("0.01")

#: §6.3's design shape: per-trade risk should not exceed this fraction of the daily budget.
#: Exceeding it is a warning, never a refusal — see the module docstring.
CONCENTRATION_GUIDANCE: Final[Decimal] = Decimal("0.20")


@dataclass(frozen=True, slots=True)
class SessionBudget:
    """Resolved risk limits for one session.

    Attributes:
        capital: the configured account size, or zero when unset.
        daily_loss_limit: positive rupee amount at which the kill switch trips.
        per_trade_risk: positive rupee amount any one position may risk.
        is_dynamic: True when the limits came from configuration, False when they are the
            frozen §1 constants.
        source: human-readable provenance, for the boot log and the UI.
    """

    capital: Decimal
    daily_loss_limit: Decimal
    per_trade_risk: Decimal
    is_dynamic: bool
    source: str

    @property
    def concentration(self) -> Decimal:
        """Per-trade risk as a fraction of the daily budget. §6.3 intends ≤ 0.20."""
        if self.daily_loss_limit <= ZERO:
            return ZERO
        return self.per_trade_risk / self.daily_loss_limit

    @property
    def exceeds_concentration_guidance(self) -> bool:
        return self.concentration > CONCENTRATION_GUIDANCE

    @property
    def drawdown_pct(self) -> Decimal:
        """The effective daily drawdown as a percentage of capital. Zero when static."""
        if self.capital <= ZERO:
            return ZERO
        return (self.daily_loss_limit / self.capital) * HUNDRED

    @classmethod
    def static(cls) -> SessionBudget:
        """The CLAUDE.md §1 constants. The fallback, and the default."""
        return cls(
            capital=ZERO,
            daily_loss_limit=DAILY_LOSS_LIMIT_INR,
            per_trade_risk=PER_TRADE_RISK_INR,
            is_dynamic=False,
            source="constants.py (CLAUDE.md §1)",
        )

    @classmethod
    def resolve(
        cls,
        *,
        capital: Decimal | float | int | str,
        drawdown_pct: Decimal | float | int | str,
        per_trade_pct: Decimal | float | int | str,
    ) -> SessionBudget:
        """Derive limits from a configured budget, falling back to the constants.

        Never raises. Every rejection path returns :meth:`static`, because a boot failure here
        would be a config typo taking the whole session down, and the safe response to "we do
        not understand this number" is the smaller limit we already trust.
        """
        try:
            amount = _decimal(capital)
            drawdown = _decimal(drawdown_pct)
            per_trade = _decimal(per_trade_pct)
        except (InvalidOperation, TypeError, ValueError) as exc:
            _log.error(
                "budget.unparseable",
                error=str(exc),
                action="falling back to the CLAUDE.md §1 constants",
            )
            return cls.static()

        if amount <= ZERO:
            # Not configured. Not an error — this is how the system ships.
            return cls.static()

        if drawdown <= ZERO or per_trade <= ZERO:
            _log.error(
                "budget.non_positive_percentage",
                capital=str(amount),
                drawdown_pct=str(drawdown),
                per_trade_pct=str(per_trade),
                action="falling back to the CLAUDE.md §1 constants",
                reason="a zero or negative risk percentage would disable sizing entirely",
            )
            return cls.static()

        daily = _quantize(amount * drawdown / HUNDRED)
        each = _quantize(amount * per_trade / HUNDRED)

        if daily <= ZERO or each <= ZERO:
            _log.error(
                "budget.rounds_to_zero",
                capital=str(amount),
                action="falling back to the CLAUDE.md §1 constants",
                reason="the configured percentages of this capital round below one paisa",
            )
            return cls.static()

        return cls(
            capital=amount,
            daily_loss_limit=daily,
            per_trade_risk=each,
            is_dynamic=True,
            source=f"capital ₹{amount} × {drawdown}% daily / {per_trade}% per trade",
        )

    def log_summary(self) -> None:
        """Announce the limits in force at boot, loudly enough to notice a wrong one.

        The operator must be able to read one log line and know exactly how much this session
        may lose — the number is no longer a constant they can look up in the source.
        """
        _log.info(
            "budget.resolved",
            dynamic=self.is_dynamic,
            capital_inr=str(self.capital),
            daily_loss_limit_inr=str(self.daily_loss_limit),
            per_trade_risk_inr=str(self.per_trade_risk),
            drawdown_pct=str(_quantize(self.drawdown_pct)),
            source=self.source,
            static_reference=f"§1 defaults are {DAILY_LOSS_LIMIT_INR}/{PER_TRADE_RISK_INR}",
        )
        if self.is_dynamic and self.daily_loss_limit > DAILY_LOSS_LIMIT_INR:
            _log.warning(
                "budget.limit_raised_above_constitution",
                configured_inr=str(self.daily_loss_limit),
                constitutional_inr=str(DAILY_LOSS_LIMIT_INR),
                multiple=str(_quantize(self.daily_loss_limit / DAILY_LOSS_LIMIT_INR)),
                meaning="this session may lose more than CLAUDE.md §1's ₹500 before halting",
            )
        if self.exceeds_concentration_guidance:
            trades_to_ruin = _quantize(self.daily_loss_limit / self.per_trade_risk)
            _log.warning(
                "budget.concentration_high",
                per_trade_inr=str(self.per_trade_risk),
                daily_limit_inr=str(self.daily_loss_limit),
                fraction=str(_quantize(self.concentration)),
                guidance=str(CONCENTRATION_GUIDANCE),
                meaning=f"{trades_to_ruin} losing trade(s) exhaust the day's entire budget; "
                f"CLAUDE.md §6.3 intends at least {int(1 / CONCENTRATION_GUIDANCE)}",
            )


def _decimal(value: Decimal | float | int | str) -> Decimal:
    """Parse a configured number. Floats go via ``str`` so 2.0 does not become 2.00000000001."""
    parsed = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    if not parsed.is_finite():
        raise InvalidOperation(f"{value!r} is not a finite number")
    return parsed


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(_PAISE)
