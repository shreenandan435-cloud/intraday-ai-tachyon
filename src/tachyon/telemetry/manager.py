"""Telemetry logging system — RotatingFileHandler, CSV, and Parquet for paper trading.

Provides:
- Human-readable operational logs via RotatingFileHandler
- Tick-by-tick trade CSV logging
- Market telemetry Parquet storage
- Structured JSON logging for machine parsing
"""

from __future__ import annotations

import csv
import json
import logging
import logging.handlers
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from pathlib import Path
from typing import Any, Final, Optional
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.constants import PROJECT_ROOT

# ── Constants ────────────────────────────────────────────────────────────────

LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
TRADES_DIR: Final[Path] = DATA_DIR / "trades"
TELEMETRY_DIR: Final[Path] = DATA_DIR / "telemetry"

# Ensure directories exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)

# Log rotation settings
MAX_LOG_SIZE_MB: Final[int] = 100
BACKUP_COUNT: Final[int] = 10

# Parquet flush settings
PARQUET_FLUSH_INTERVAL_SEC: Final[int] = 30
PARQUET_BUFFER_SIZE: Final[int] = 10000


# ── Log Formatters ───────────────────────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "name", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info"
            }:
                log_obj[key] = value

        return json.dumps(log_obj, default=str)


class HumanReadableFormatter(logging.Formatter):
    """Human-readable formatter for console/rotating file."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = f"{record.levelname:<8}"
        logger = f"{record.name:<30}"
        message = record.getMessage()
        return f"{timestamp} | {level} | {logger} | {message}"


# ── CSV Trade Logger ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class TradeRecord:
    """Trade record for CSV logging."""

    trade_id: str
    symbol: str
    side: str
    quantity: int
    entry_price: Decimal
    exit_price: Decimal
    entry_time: datetime
    exit_time: datetime
    gross_pnl: Decimal
    charges: Decimal
    net_pnl: Decimal
    exit_reason: str
    hold_time_seconds: int
    order_id: str


class CSVTradeLogger:
    """Daily rotating CSV trade logger."""

    def __init__(self, trades_dir: Path = TRADES_DIR):
        self._trades_dir = trades_dir
        self._trades_dir.mkdir(parents=True, exist_ok=True)
        self._current_date: Optional[date] = None
        self._writer: Any = None
        self._file_handle: Any = None
        self._lock = threading.Lock()
        self._fieldnames = [
            "trade_id", "symbol", "side", "quantity",
            "entry_price", "exit_price", "entry_time", "exit_time",
            "gross_pnl", "charges", "net_pnl", "exit_reason",
            "hold_time_seconds", "order_id"
        ]
        self._rotate()

    def _get_path(self, dt: date) -> Path:
        return self._trades_dir / f"trades_{dt:%Y%m%d}.csv"

    def _rotate(self) -> None:
        """Rotate to current day's file."""
        with self._lock:
            today = datetime.now().date()
            if self._current_date == today and self._file_handle:
                return

            # Close existing file if rotating to a new day
            if self._file_handle:
                self._file_handle.close()

            path = self._get_path(today)
            file_exists = path.exists()
            self._file_handle = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file_handle)

            if not file_exists:
                self._writer.writerow(self._fieldnames)

            self._current_date = today

    def log_trade(self, record: TradeRecord) -> None:
        """Log a completed trade."""
        self._rotate()
        if self._writer:
            self._writer.writerow([
                record.trade_id,
                record.symbol,
                record.side,
                record.quantity,
                str(record.entry_price),
                str(record.exit_price),
                record.entry_time.isoformat(),
                record.exit_time.isoformat(),
                str(record.gross_pnl),
                str(record.charges),
                str(record.net_pnl),
                record.exit_reason,
                record.hold_time_seconds,
                record.order_id,
            ])
            self._file_handle.flush()

    def close(self) -> None:
        """Close the current file."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None
                self._writer = None


# ── Parquet Telemetry Logger ─────────────────────────────────────────────────


@dataclass(slots=True)
class TelemetryConfig:
    """Configuration for Parquet telemetry logger."""

    telemetry_dir: Path = TELEMETRY_DIR
    flush_interval_sec: int = PARQUET_FLUSH_INTERVAL_SEC
    buffer_size: int = PARQUET_BUFFER_SIZE
    compression: str = "zstd"


class ParquetTelemetryLogger:
    """Buffered Parquet telemetry logger with periodic flush."""

    def __init__(self, config: Optional[TelemetryConfig] = None):
        self._config = config or TelemetryConfig()
        self._config.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._current_date: Optional[date] = None
        self._flush_task: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start background flush thread."""
        self._running = True
        self._flush_task = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_task.start()

    def stop(self) -> None:
        """Stop background flush and flush remaining."""
        self._running = False
        if self._flush_task:
            self._flush_task.join(timeout=10)
        self.flush()

    def log(self, record: dict[str, Any]) -> None:
        """Add telemetry record to buffer."""
        record["timestamp"] = datetime.now().isoformat()
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._config.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to Parquet (caller must hold lock)."""
        if not self._buffer:
            return

        today = datetime.now().date()
        path = self._config.telemetry_dir / f"market_telemetry_{today:%Y%m%d}.parquet"

        table = pa.Table.from_pylist(self._buffer)

        if path.exists():
            existing = pq.read_table(path)
            table = pa.concat_tables([existing, table])

        pq.write_table(table, path, compression=self._config.compression)
        self._buffer.clear()

    def flush(self) -> None:
        """Flush buffer to disk."""
        with self._lock:
            self._flush_buffer()

    def _flush_loop(self) -> None:
        """Background flush loop."""
        while self._running:
            time.sleep(self._config.flush_interval_sec)
            self.flush()


# ── Main Telemetry Manager ───────────────────────────────────────────────────


class TelemetryManager:
    """Unified telemetry manager for paper trading."""

    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        trades_dir: Path = TRADES_DIR,
        telemetry_dir: Path = TELEMETRY_DIR,
        clock: Clock = SYSTEM_CLOCK,
    ):
        self._clock = clock
        self._trades_dir = trades_dir
        self._telemetry_dir = telemetry_dir

        # Setup loggers
        self._setup_loggers(log_dir)

        # Initialize sub-loggers
        self._csv_logger = CSVTradeLogger(trades_dir)
        self._parquet_logger = ParquetTelemetryLogger(
            TelemetryConfig(telemetry_dir=telemetry_dir)
        )
        self._parquet_logger.start()

        self._logger = logging.getLogger("tachyon.telemetry")

    def _setup_loggers(self, log_dir: Path) -> None:
        """Setup rotating file handlers for operational and JSON logs."""
        log_dir.mkdir(parents=True, exist_ok=True)

        # Human-readable rotating log
        ops_handler = logging.handlers.RotatingFileHandler(
            log_dir / "paper_trading.log",
            maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        ops_handler.setFormatter(HumanReadableFormatter())
        ops_handler.setLevel(logging.INFO)

        # JSON rotating log for machine parsing
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "paper_trading.jsonl",
            maxBytes=MAX_LOG_SIZE_MB * 1024 * 1024,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(logging.INFO)

        # Configure root logger
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        # Remove existing file handlers to avoid duplicates
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)

        root.addHandler(ops_handler)
        root.addHandler(json_handler)

    def log_operation(
        self,
        event: str,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        """Log an operational event with structured data."""
        extra = {"event": event, **kwargs}
        self._logger.log(level, event, extra=extra)

    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: Decimal,
        exit_price: Decimal,
        entry_time: datetime,
        exit_time: datetime,
        gross_pnl: Decimal,
        charges: Decimal,
        net_pnl: Decimal,
        exit_reason: str,
        hold_time_seconds: int,
        order_id: str,
    ) -> None:
        """Log a completed trade to CSV."""
        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            gross_pnl=gross_pnl,
            charges=charges,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            hold_time_seconds=hold_time_seconds,
            order_id=order_id,
        )
        self._csv_logger.log_trade(record)

        # Also log to operational log
        self._logger.info(
            "trade_completed",
            extra={
                "event": "TRADE_COMPLETED",
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": str(entry_price),
                "exit_price": str(exit_price),
                "gross_pnl": str(gross_pnl),
                "charges": str(charges),
                "net_pnl": str(net_pnl),
                "exit_reason": exit_reason,
                "hold_time_seconds": hold_time_seconds,
            }
        )

    def log_telemetry(self, record: dict[str, Any]) -> None:
        """Log telemetry record to Parquet buffer."""
        self._parquet_logger.log(record)

    def log_position_update(
        self,
        symbol: str,
        quantity: int,
        side: str,
        entry_price: Decimal,
        current_price: Decimal,
        unrealised_pnl: Decimal,
        stop_loss: Decimal,
        target: Decimal,
    ) -> None:
        """Log position update for telemetry."""
        self.log_telemetry({
            "event": "POSITION_UPDATE",
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "entry_price": str(entry_price),
            "current_price": str(current_price),
            "unrealised_pnl": str(unrealised_pnl),
            "stop_loss": str(stop_loss),
            "target": str(target),
        })

    def log_account_summary(self, summary: dict[str, Any]) -> None:
        """Log account summary for telemetry."""
        self.log_telemetry({
            "event": "ACCOUNT_SUMMARY",
            **{k: str(v) if isinstance(v, Decimal) else v for k, v in summary.items()},
        })

    def log_order_event(
        self,
        event: str,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: Decimal,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Log order event."""
        self.log_telemetry({
            "event": event,
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": str(price),
            "status": status,
            **kwargs,
        })

    def get_csv_logger(self) -> CSVTradeLogger:
        return self._csv_logger

    def get_parquet_logger(self) -> ParquetTelemetryLogger:
        return self._parquet_logger

    def close(self) -> None:
        """Close all loggers and flush buffers."""
        self._csv_logger.close()
        self._parquet_logger.stop()

    @property
    def logger(self) -> logging.Logger:
        return self._logger


# ── Factory Function ─────────────────────────────────────────────────────────


def create_telemetry_manager(
    log_dir: Path = LOG_DIR,
    trades_dir: Path = TRADES_DIR,
    telemetry_dir: Path = TELEMETRY_DIR,
    clock: Clock = SYSTEM_CLOCK,
) -> TelemetryManager:
    """Create telemetry manager with default paths."""
    return TelemetryManager(log_dir, trades_dir, telemetry_dir, clock)