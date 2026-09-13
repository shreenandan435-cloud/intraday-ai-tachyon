"""Paper trading engine — realistic simulation for autonomous cloud deployment.

Features:
- Virtual capital management (default ₹1,00,000)
- Realistic slippage and broker latency simulation
- Position tracking with P&L calculation
- Risk management (daily loss limit, per-trade risk)
- Trade logging to CSV and Parquet
- Human-readable operational logs
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings
from tachyon.core.constants import PROJECT_ROOT
from tachyon.core.logger import get_logger

_log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_VIRTUAL_CAPITAL: Final[Decimal] = Decimal("100000")
DEFAULT_MAX_DAILY_DRAWDOWN_PCT: Final[Decimal] = Decimal("2.0")
DEFAULT_PER_TRADE_RISK_PCT: Final[Decimal] = Decimal("1.0")

# Slippage simulation (basis points)
BASE_SLIPPAGE_BPS: Final[int] = 2  # 2 bps base slippage
MAX_SLIPPAGE_BPS: Final[int] = 10  # Max 10 bps in volatile conditions

# Latency simulation (milliseconds)
MIN_LATENCY_MS: Final[int] = 50
MAX_LATENCY_MS: Final[int] = 300

# Broker charges (Indian markets)
STT_RATE: Final[Decimal] = Decimal("0.00025")  # 0.025% on sell side
EXCHANGE_TXN_CHARGE: Final[Decimal] = Decimal("0.0000345")  # NSE
GST_RATE: Final[Decimal] = Decimal("0.18")
SEBI_RATE: Final[Decimal] = Decimal("0.000001")
STAMP_DUTY_RATE: Final[Decimal] = Decimal("0.00003")  # 0.003% on buy
BROKERAGE_PER_ORDER: Final[Decimal] = Decimal("20")  # Flat ₹20 per order

# Data directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
TRADES_DIR: Final[Path] = DATA_DIR / "trades"
TELEMETRY_DIR: Final[Path] = DATA_DIR / "telemetry"

# Ensure directories exist
TRADES_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)


# ── Enums ────────────────────────────────────────────────────────────────────


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class ExitReason(StrEnum):
    STOP_LOSS = "STOP_LOSS"
    TARGET = "TARGET"
    SQUARE_OFF = "SQUARE_OFF"
    MANUAL = "MANUAL"
    KILL_SWITCH = "KILL_SWITCH"


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PaperOrder:
    """Simulated order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET"
    trigger_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: Decimal = Decimal("0")
    timestamp: datetime = field(default_factory=now_ist)
    order_tag: str = ""
    is_exit: bool = False


@dataclass(slots=True)
class PaperPosition:
    """Open position in paper trading."""

    symbol: str
    quantity: int
    avg_entry_price: Decimal
    side: OrderSide
    stop_loss: Decimal
    target: Decimal
    entry_time: datetime
    order_id: str
    unrealised_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    trailing_stop: Decimal | None = None
    trail_activated: bool = False


@dataclass(slots=True)
class PaperTrade:
    """Completed trade record."""

    trade_id: str
    symbol: str
    side: OrderSide
    quantity: int
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    gross_pnl: Decimal
    charges: Decimal
    net_pnl: Decimal
    exit_reason: ExitReason
    hold_time_seconds: int
    order_id: str


@dataclass(slots=True)
class PaperAccount:
    """Paper trading account state."""

    virtual_capital: Decimal
    available_cash: Decimal
    used_margin: Decimal
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    unrealised_pnl: Decimal = Decimal("0")
    total_charges: Decimal = Decimal("0")
    trades_today: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: Decimal = Decimal("0")
    peak_balance: Decimal = Decimal("0")


# ── Charges Calculator ───────────────────────────────────────────────────────


class ChargesCalculator:
    """Calculate realistic broker charges for Indian markets."""

    @staticmethod
    def calculate(
        side: OrderSide,
        quantity: int,
        price: Decimal,
    ) -> dict[str, Decimal]:
        """Calculate all charges for a trade (intraday — the only engine mode)."""
        turnover = Decimal(quantity) * price

        # Brokerage
        brokerage = BROKERAGE_PER_ORDER

        # STT (only on sell side for intraday)
        stt = Decimal("0")
        if side == OrderSide.SELL:
            stt = (turnover * STT_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Exchange transaction charges
        exchange_charge = (turnover * EXCHANGE_TXN_CHARGE).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # SEBI fees
        sebi_fee = (turnover * SEBI_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Stamp duty (on buy side)
        stamp_duty = Decimal("0")
        if side == OrderSide.BUY:
            stamp_duty = (turnover * STAMP_DUTY_RATE).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # GST on brokerage + exchange charges
        taxable = brokerage + exchange_charge
        gst = (taxable * GST_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        total = brokerage + stt + exchange_charge + sebi_fee + stamp_duty + gst

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange_charge": exchange_charge,
            "sebi_fee": sebi_fee,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "total": total,
        }


# ── Slippage & Latency Simulator ─────────────────────────────────────────────


class MarketSimulator:
    """Simulate realistic market conditions: slippage, latency, partial fills."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._volatility_factor = Decimal("1.0")

    def set_volatility(self, factor: Decimal) -> None:
        """Adjust volatility factor (1.0 = normal, >1.0 = high volatility)."""
        self._volatility_factor = factor

    def simulate_latency(self) -> float:
        """Simulate broker latency in seconds."""
        return self._rng.uniform(MIN_LATENCY_MS / 1000, MAX_LATENCY_MS / 1000)

    def simulate_slippage(
        self,
        side: OrderSide,
        price: Decimal,
        volatility_bps: int = 0,
    ) -> Decimal:
        """Simulate slippage in basis points."""
        base_slippage = BASE_SLIPPAGE_BPS + volatility_bps
        max_slippage = min(MAX_SLIPPAGE_BPS, float(base_slippage * self._volatility_factor))

        # Slippage is adverse: worse price for the trader
        slippage_bps = self._rng.uniform(0, max_slippage)
        slippage_factor = Decimal(str(slippage_bps)) / Decimal("10000")

        if side == OrderSide.BUY:
            # Pay more when buying
            return price * (Decimal("1") + slippage_factor)
        # Receive less when selling
        return price * (Decimal("1") - slippage_factor)

    def simulate_partial_fill(self, quantity: int, probability: float = 0.05) -> int:
        """Simulate partial fill (rare in liquid stocks)."""
        if self._rng.random() < probability and quantity > 1:
            return self._rng.randint(1, quantity - 1)
        return quantity


# ── Paper Trading Engine ─────────────────────────────────────────────────────


class PaperTradingEngine:
    """Main paper trading engine for autonomous operation."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock = SYSTEM_CLOCK,
        virtual_capital: Decimal | None = None,
    ):
        self._settings = settings
        self._clock = clock

        # Initialize account
        capital = virtual_capital or Decimal(
            str(os.environ.get("PAPER_TRADING_BUDGET_INR", "100000"))
        )
        self._account = PaperAccount(
            virtual_capital=capital,
            available_cash=capital,
            used_margin=Decimal("0"),
            peak_balance=capital,
        )

        # Risk limits
        drawdown_pct = Decimal(str(os.environ.get("PAPER_MAX_DAILY_DRAWDOWN_PCT", "2.0")))
        per_trade_pct = Decimal(str(os.environ.get("PAPER_PER_TRADE_RISK_PCT", "1.0")))
        self._max_daily_loss = capital * drawdown_pct / Decimal("100")
        self._per_trade_risk = capital * per_trade_pct / Decimal("100")

        # State
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._trade_history: list[PaperTrade] = []
        self._daily_realised_pnl = Decimal("0")

        # Simulators
        self._market = MarketSimulator()
        self._charges = ChargesCalculator()

        # Locks for thread safety
        self._lock = threading.RLock()

        # Data logging
        self._trade_file: Any = None
        self._trade_file_handle: Any = None
        self._telemetry_buffer: list[dict[str, Any]] = []
        self._telemetry_lock = threading.Lock()
        self._flush_interval = 30  # seconds
        self._last_flush = time.time()

        # Initialize CSV file
        self._rotate_trade_csv()

        # Start background flush task
        self._flush_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start background tasks."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    async def stop(self) -> None:
        """Stop background tasks and flush data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
        self._flush_telemetry()
        if self._trade_file_handle:
            self._trade_file_handle.close()

    def _rotate_trade_csv(self) -> None:
        """Rotate daily trade CSV if date changed."""
        today = now_ist(self._clock).date()
        csv_path = TRADES_DIR / f"trades_{today:%Y%m%d}.csv"

        # Close previous day's file if rotated
        if self._trade_file_handle and self._trade_file_handle.name != str(csv_path):
            self._trade_file_handle.close()

        file_exists = csv_path.exists()
        # Long-lived handle by design — one CSV writer held for the trading day,
        # closed via stop()/_rotate_trade_csv(). See rollout.py for the same pattern.
        self._trade_file_handle = csv_path.open("a", newline="", encoding="utf-8")  # noqa: SIM115
        self._trade_file = csv.writer(self._trade_file_handle)

        if not file_exists:
            self._trade_file.writerow([
                "trade_id", "symbol", "side", "quantity",
                "entry_price", "exit_price", "entry_time", "exit_time",
                "gross_pnl", "charges", "net_pnl", "exit_reason",
                "hold_time_seconds", "order_id"
            ])

    async def _periodic_flush(self) -> None:
        """Periodically flush telemetry to Parquet."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                self._flush_telemetry()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                _log.error("paper.flush_failed", error=str(exc))

    def _flush_telemetry(self) -> None:
        """Flush telemetry buffer to Parquet."""
        with self._telemetry_lock:
            if not self._telemetry_buffer:
                return

            today = now_ist(self._clock).date()
            parquet_path = TELEMETRY_DIR / f"market_telemetry_{today:%Y%m%d}.parquet"

            # Convert buffer to Arrow table
            table = pa.Table.from_pylist(self._telemetry_buffer)

            # Append or create
            if parquet_path.exists():
                existing = pq.read_table(parquet_path)
                table = pa.concat_tables([existing, table])

            pq.write_table(table, parquet_path, compression="zstd")
            self._telemetry_buffer.clear()
            _log.debug("paper.telemetry_flushed", rows=len(table), path=str(parquet_path))

    def _log_telemetry(self, record: dict[str, Any]) -> None:
        """Add telemetry record to buffer."""
        record["timestamp"] = now_ist(self._clock).isoformat()
        with self._telemetry_lock:
            self._telemetry_buffer.append(record)

    # ── Order Management ──────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
        order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET",
        trigger_price: Decimal | None = None,
        stop_loss: Decimal | None = None,
        target: Decimal | None = None,
        order_tag: str = "",
        is_exit: bool = False,
    ) -> PaperOrder:
        """Place a simulated order."""
        with self._lock:
            # Check risk limits
            if not is_exit:
                if self._account.daily_pnl <= -self._max_daily_loss:
                    raise RuntimeError("Daily loss limit exceeded")

                # Check per-trade risk
                notional = Decimal(quantity) * price
                max_qty = int(self._per_trade_risk / price) if price > 0 else 0
                if quantity > max_qty:
                    raise RuntimeError(f"Quantity exceeds per-trade risk limit (max {max_qty})")

                # Check margin
                required_margin = notional * Decimal("0.2")  # 20% margin
                if self._account.available_cash < required_margin:
                    raise RuntimeError("Insufficient margin")

            # Create order
            order = PaperOrder(
                order_id=str(uuid4())[:8],
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
                trigger_price=trigger_price,
                order_tag=order_tag,
                is_exit=is_exit,
            )

            self._orders[order.order_id] = order

            # Simulate execution
            asyncio.create_task(self._execute_order(order, stop_loss, target))

            return order

    async def place_order_async(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: Decimal,
        order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET",
        trigger_price: Decimal | None = None,
        stop_loss: Decimal | None = None,
        target: Decimal | None = None,
        order_tag: str = "",
        is_exit: bool = False,
    ) -> PaperOrder:
        """Async version of place_order for use in async contexts."""
        return self.place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            trigger_price=trigger_price,
            stop_loss=stop_loss,
            target=target,
            order_tag=order_tag,
            is_exit=is_exit,
        )

    async def _execute_order(
        self,
        order: PaperOrder,
        stop_loss: Decimal | None,
        target: Decimal | None,
    ) -> None:
        """Simulate order execution with latency and slippage."""
        # Simulate latency
        latency = self._market.simulate_latency()
        await asyncio.sleep(latency)

        with self._lock:
            # Check if order still valid
            if order.status != OrderStatus.PENDING:
                return

            # Simulate slippage
            fill_price = self._market.simulate_slippage(order.side, order.price)
            fill_qty = self._market.simulate_partial_fill(order.quantity)

            # Calculate charges
            charges = self._charges.calculate(order.side, fill_qty, fill_price)
            total_charges = charges["total"]

            # Update order
            order.status = OrderStatus.FILLED if fill_qty == order.quantity else OrderStatus.PARTIAL
            order.filled_quantity = fill_qty
            order.avg_fill_price = fill_price

            # Update account
            notional = Decimal(fill_qty) * fill_price
            if order.side == OrderSide.BUY:
                self._account.used_margin += notional * Decimal("0.2")
                self._account.available_cash -= total_charges
            else:
                self._account.used_margin -= notional * Decimal("0.2")

            # Handle position
            if order.is_exit:
                await self._close_position(order, fill_price, fill_qty, total_charges)
            else:
                await self._open_position(order, fill_price, fill_qty, stop_loss, target)

            # Log trade
            self._log_telemetry({
                "event": "ORDER_FILLED",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": fill_qty,
                "price": str(fill_price),
                "charges": str(total_charges),
                "latency_ms": latency * 1000,
            })

    async def _open_position(
        self,
        order: PaperOrder,
        fill_price: Decimal,
        fill_qty: int,
        stop_loss: Decimal | None,
        target: Decimal | None,
    ) -> None:
        """Open a new position. Charges accrue at exit, not entry (see _close_position)."""
        position = PaperPosition(
            symbol=order.symbol,
            quantity=fill_qty,
            avg_entry_price=fill_price,
            side=order.side,
            stop_loss=stop_loss or Decimal("0"),
            target=target or Decimal("0"),
            entry_time=now_ist(self._clock),
            order_id=order.order_id,
        )
        self._positions[order.symbol] = position
        self._account.trades_today += 1

    async def _close_position(
        self,
        order: PaperOrder,
        fill_price: Decimal,
        fill_qty: int,
        charges: Decimal,
    ) -> None:
        """Close an existing position."""
        position = self._positions.get(order.symbol)
        if not position:
            return

        # Calculate P&L
        if position.side == OrderSide.BUY:
            gross_pnl = (fill_price - position.avg_entry_price) * Decimal(fill_qty)
        else:
            gross_pnl = (position.avg_entry_price - fill_price) * Decimal(fill_qty)

        net_pnl = gross_pnl - charges

        # Determine exit reason
        exit_reason = ExitReason.MANUAL
        if fill_price <= position.stop_loss and position.side == OrderSide.BUY:
            exit_reason = ExitReason.STOP_LOSS
        elif fill_price >= position.target and position.side == OrderSide.BUY:
            exit_reason = ExitReason.TARGET
        elif fill_price >= position.stop_loss and position.side == OrderSide.SELL:
            exit_reason = ExitReason.STOP_LOSS
        elif fill_price <= position.target and position.side == OrderSide.SELL:
            exit_reason = ExitReason.TARGET

        # Create trade record
        trade = PaperTrade(
            trade_id=str(uuid4())[:8],
            symbol=order.symbol,
            side=position.side,
            quantity=fill_qty,
            entry_price=position.avg_entry_price,
            exit_price=fill_price,
            entry_time=position.entry_time,
            exit_time=now_ist(self._clock),
            gross_pnl=gross_pnl,
            charges=charges,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            hold_time_seconds=int((now_ist(self._clock) - position.entry_time).total_seconds()),
            order_id=order.order_id,
        )

        # Update account
        self._account.total_pnl += net_pnl
        self._account.daily_pnl += net_pnl
        self._account.total_charges += charges
        self._account.unrealised_pnl -= position.unrealised_pnl

        if net_pnl >= 0:
            self._account.winning_trades += 1
        else:
            self._account.losing_trades += 1

        # Update peak/drawdown
        current_balance = self._account.virtual_capital + self._account.total_pnl
        if current_balance > self._account.peak_balance:
            self._account.peak_balance = current_balance
        drawdown = self._account.peak_balance - current_balance
        if drawdown > self._account.max_drawdown:
            self._account.max_drawdown = drawdown

        # Remove position
        del self._positions[order.symbol]

        # Write to CSV
        self._write_trade_csv(trade)

        # Log telemetry
        self._log_telemetry({
            "event": "POSITION_CLOSED",
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side.value,
            "quantity": trade.quantity,
            "entry_price": str(trade.entry_price),
            "exit_price": str(trade.exit_price),
            "gross_pnl": str(trade.gross_pnl),
            "net_pnl": str(trade.net_pnl),
            "exit_reason": trade.exit_reason.value,
            "hold_time_seconds": trade.hold_time_seconds,
        })

    def _write_trade_csv(self, trade: PaperTrade) -> None:
        """Write trade to CSV."""
        if self._trade_file:
            self._trade_file.writerow([
                trade.trade_id, trade.symbol, trade.side.value, trade.quantity,
                str(trade.entry_price), str(trade.exit_price),
                trade.entry_time.isoformat(), trade.exit_time.isoformat(),
                str(trade.gross_pnl), str(trade.charges), str(trade.net_pnl),
                trade.exit_reason.value, trade.hold_time_seconds, trade.order_id
            ])
            self._trade_file_handle.flush()

    # ── Position Management ───────────────────────────────────────────────────

    def update_mark_to_market(self, symbol: str, ltp: Decimal) -> Decimal | None:
        """Update unrealised P&L for a position."""
        with self._lock:
            position = self._positions.get(symbol)
            if not position:
                return None

            if position.side == OrderSide.BUY:
                unrealised = (ltp - position.avg_entry_price) * Decimal(position.quantity)
            else:
                unrealised = (position.avg_entry_price - ltp) * Decimal(position.quantity)

            position.unrealised_pnl = unrealised
            self._account.unrealised_pnl = sum(
                (p.unrealised_pnl for p in self._positions.values()),
                Decimal("0")
            )

            # Check trailing stop
            if position.trail_activated and position.trailing_stop is not None:
                trail_hit = (
                    (position.side == OrderSide.BUY and ltp <= position.trailing_stop)
                    or (position.side == OrderSide.SELL and ltp >= position.trailing_stop)
                )
                if trail_hit:
                    return self._trigger_stop_loss(symbol, ltp)

            return unrealised

    def _trigger_stop_loss(self, symbol: str, ltp: Decimal) -> Decimal:
        """Trigger stop loss for a position."""
        position = self._positions.get(symbol)
        if not position:
            return Decimal("0")

        # Create exit order
        exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        # Schedule the order placement (non-blocking)
        asyncio.create_task(self.place_order_async(
            symbol=symbol,
            side=exit_side,
            quantity=position.quantity,
            price=ltp,
            is_exit=True,
        ))

        return position.unrealised_pnl

    def activate_trailing_stop(self, symbol: str, trail_price: Decimal) -> bool:
        """Activate trailing stop for a position."""
        with self._lock:
            position = self._positions.get(symbol)
            if not position:
                return False
            position.trailing_stop = trail_price
            position.trail_activated = True
            return True

    # ── Account Info ──────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict[str, Any]:
        """Get current account summary."""
        with self._lock:
            current_balance = self._account.virtual_capital + self._account.total_pnl
            return {
                "virtual_capital": str(self._account.virtual_capital),
                "current_balance": str(current_balance),
                "available_cash": str(self._account.available_cash),
                "used_margin": str(self._account.used_margin),
                "total_pnl": str(self._account.total_pnl),
                "daily_pnl": str(self._account.daily_pnl),
                "unrealised_pnl": str(self._account.unrealised_pnl),
                "total_charges": str(self._account.total_charges),
                "trades_today": self._account.trades_today,
                "winning_trades": self._account.winning_trades,
                "losing_trades": self._account.losing_trades,
                "max_drawdown": str(self._account.max_drawdown),
                "peak_balance": str(self._account.peak_balance),
                "open_positions": len(self._positions),
                "daily_loss_limit": str(self._max_daily_loss),
                "per_trade_risk": str(self._per_trade_risk),
            }

    def get_positions(self) -> dict[str, dict[str, Any]]:
        """Get all open positions."""
        with self._lock:
            return {
                symbol: {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "avg_entry_price": str(pos.avg_entry_price),
                    "side": pos.side.value,
                    "stop_loss": str(pos.stop_loss),
                    "target": str(pos.target),
                    "entry_time": pos.entry_time.isoformat(),
                    "unrealised_pnl": str(pos.unrealised_pnl),
                    "trailing_stop": str(pos.trailing_stop) if pos.trailing_stop else None,
                    "trail_activated": pos.trail_activated,
                }
                for symbol, pos in self._positions.items()
            }

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Get today's trade history."""
        with self._lock:
            return [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side.value,
                    "quantity": t.quantity,
                    "entry_price": str(t.entry_price),
                    "exit_price": str(t.exit_price),
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "gross_pnl": str(t.gross_pnl),
                    "charges": str(t.charges),
                    "net_pnl": str(t.net_pnl),
                    "exit_reason": t.exit_reason.value,
                    "hold_time_seconds": t.hold_time_seconds,
                }
                for t in self._trade_history
            ]

    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached."""
        with self._lock:
            return self._account.daily_pnl > -self._max_daily_loss


# ── Factory Function ─────────────────────────────────────────────────────────


def create_paper_engine(settings: Settings, clock: Clock = SYSTEM_CLOCK) -> PaperTradingEngine:
    """Create paper trading engine from settings."""
    return PaperTradingEngine(settings, clock)
