from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum, auto

class OrderState(Enum):
    NEW = auto()
    SUBMITTED = auto()
    ACKNOWLEDGED = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCEL_REQUESTED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    UNKNOWN = auto()

class MarketRegime(Enum):
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()

@dataclass
class MarketTick:
    token: str
    exchange_timestamp: int
    local_receive_timestamp: float
    sequence_number: int
    ltp: float
    last_traded_quantity: int
    cumulative_volume: int

@dataclass
class Candle:
    token: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    ticks: int
    vwap: float
    atr: float = 0.0
    rvol: float = 1.0
    tick_ratio: float = 1.0

@dataclass
class SignalScore:
    token: str
    total_score: float
    passed_conditions: List[str] = field(default_factory=list)
    failed_conditions: List[str] = field(default_factory=list)
    setup_type: str = "MOMENTUM_BREAKOUT"
    is_valid: bool = False

@dataclass
class OrderExecution:
    internal_id: str
    token: str
    symbol: str
    quantity: int
    limit_price: float
    stop_loss: float
    target: float
    state: OrderState = OrderState.NEW
    broker_order_id: Optional[str] = None
    filled_quantity: int = 0
    average_fill_price: float = 0.0
