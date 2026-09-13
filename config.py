import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BrokerConfig:
    api_key: str = os.getenv("SMARTAPI_API_KEY", "")
    client_code: str = os.getenv("SMARTAPI_CLIENT_CODE", "")
    password: str = os.getenv("SMARTAPI_PASSWORD", "")
    totp_secret: str = os.getenv("SMARTAPI_TOTP_SECRET", "")

@dataclass
class RiskConfig:
    risk_per_trade_rupees: float = 1000.0
    max_daily_loss_rupees: float = 3000.0
    max_consecutive_losses: int = 3
    max_open_positions: int = 2
    slippage_atr_ratio: float = 0.05
    target_risk_reward_ratio: float = 2.0

@dataclass
class StrategyConfig:
    candle_minutes: int = 5
    atr_period: int = 14
    volume_lookback: int = 20
    rvol_threshold: float = 1.8
    tick_ratio_threshold: float = 1.3
    breakout_atr_buffer: float = 0.15
    extrema_order: int = 3
    active_tokens: dict = field(default_factory=lambda: {
        "11536": "TCS-EQ",
        "1594": "INFY-EQ",
        "1333": "HDFCBANK-EQ",
        "3045": "SBIN-EQ",
        "11532": "RELIANCE-EQ"
    })

@dataclass
class SystemConfig:
    trading_mode: str = os.getenv("TRADING_MODE", "PAPER").upper()
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

@dataclass
class TachyonConfig:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
