"""Configuration loading — CLAUDE.md §8.

Two sources, strictly separated by *what may change*:

* ``.env``                → secrets and per-machine wiring (API keys, ports, log level).
* ``config/settings.yaml`` → session tunables (watchlist, buffer sizes, timeouts).

And one source that is **not** a source: :mod:`tachyon.core.constants`. The hard risk limits
live there and are unreachable from here by design.

Silence is the danger. If someone writes ``DAILY_LOSS_LIMIT_INR: 5000`` into ``settings.yaml``,
the naive outcome is that pydantic ignores an unknown key, the operator believes the limit was
raised, and the system trades all day against a limit that never moved. So every source is
scanned for protected names *before* validation and a match is a hard boot failure —
see :class:`ProtectedConstantOverrideError`.

Precedence (highest first): init kwargs → environment → .env → settings.yaml → defaults.

Credential env-var precedence (highest first): SMARTAPI_* (new convention), then legacy
aliases (SMARTAPI_CLIENT_ID / SMARTAPI_PIN, ANGEL_*), then empty.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from tachyon.core.constants import (
    DEFAULT_TRADING_MODE,
    PROJECT_ROOT,
    PROTECTED_CONSTANT_NAMES,
    SETTINGS_YAML,
    TradingMode,
)

ENV_FILE: Final[Path] = PROJECT_ROOT / ".env"

#: Plausible spellings an operator might reach for when trying to "just tweak" a hard limit.
#: Protected alongside the real constant names so near-misses fail loudly too.
_PROTECTED_ALIASES: Final[frozenset[str]] = frozenset(
    {
        "DAILY_LOSS_LIMIT",
        "LOSS_LIMIT",
        "MAX_DAILY_LOSS",
        "PER_TRADE_RISK",
        "PER_TRADE_RISK_FRACTION",
        "RISK_PER_TRADE",
        "AUTO_SQUAREOFF",
        "SQUAREOFF_TIME",
        "SQUARE_OFF_TIME",
        "NO_NEW_ENTRIES",
        "STOP_LOSS_MULTIPLIER",
        "ATR_MULTIPLIER",
        "RISK_REWARD",
        "DAILY_LOCK_FILE",
        "LOCK_FILE",
    }
)

_ALL_PROTECTED: Final[frozenset[str]] = PROTECTED_CONSTANT_NAMES | _PROTECTED_ALIASES


class ProtectedConstantOverrideError(RuntimeError):
    """A config source tried to set a value that CLAUDE.md §1 declares immutable."""


def _walk_keys(node: object, trail: tuple[str, ...] = ()) -> Iterable[tuple[str, tuple[str, ...]]]:
    """Yield ``(key, path)`` for every mapping key anywhere in a nested structure."""
    if isinstance(node, Mapping):
        for raw_key, value in node.items():
            key = str(raw_key)
            path = (*trail, key)
            yield key, path
            yield from _walk_keys(value, path)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk_keys(value, (*trail, f"[{index}]"))


def _reject_protected(payload: object, origin: str) -> None:
    """Raise if ``payload`` mentions a protected constant at any depth."""
    offenders = sorted(
        {
            ".".join(path)
            for key, path in _walk_keys(payload)
            if key.strip().upper() in _ALL_PROTECTED
        }
    )
    if offenders:
        raise ProtectedConstantOverrideError(
            f"{origin} attempts to set hard constant(s): {', '.join(offenders)}. "
            f"These are fixed in tachyon.core.constants per CLAUDE.md §1 and cannot be "
            f"configured. Remove the key; if the value genuinely must change, edit "
            f"constants.py and update its test."
        )


# SmartAPI credential resolution -- SMARTAPI_* (primary) then ANGEL_* (legacy alias).
#
# Precedence, highest first:
#   1. SMARTAPI_*  -- the new convention written into .env.
#   2. Legacy aliases (SMARTAPI_CLIENT_ID / SMARTAPI_PIN, ANGEL_*)  -- accepted as a
#      fallback so an operator with an existing .env is not silently broken.
#   3. Empty default  -- the field still constructs, but validate_live_ready() refuses
#      to let the process trade.
#
# Both conventions resolve into the same four smartapi_* pydantic fields; the only
# observable difference is which env name wins. _apply_legacy_aliases runs in
# mode="before" so pydantic-settings sees the merged value as if it had always been
# written that way.

_LEGACY_ENV_FALLBACKS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "smartapi_api_key": ("SMARTAPI_API_KEY", "ANGEL_API_KEY"),
        "smartapi_client_code": (
            "SMARTAPI_CLIENT_CODE",
            "SMARTAPI_CLIENT_ID",
            "ANGEL_CLIENT_ID",
        ),
        "smartapi_password": ("SMARTAPI_PASSWORD", "SMARTAPI_PIN", "ANGEL_PIN"),
        "smartapi_totp_secret": ("SMARTAPI_TOTP_SECRET", "ANGEL_TOTP_SECRET"),
    }
)


def _resolve_legacy_aliases() -> dict[str, str]:
    """Resolve the four SmartAPI credentials from os.environ, honouring legacy aliases.

    Per-field lookup order is the order declared in _LEGACY_ENV_FALLBACKS: SMARTAPI_*
    first, then legacy aliases (SMARTAPI_CLIENT_ID / SMARTAPI_PIN and ANGEL_*).
    Returns {field_name: first_non_empty_env_value}.
    """
    resolved: dict[str, str] = {}
    for field_name, env_names in _LEGACY_ENV_FALLBACKS.items():
        for env_name in env_names:
            value = os.environ.get(env_name)
            if value:
                resolved[field_name] = value
                break
    return resolved


def _is_blank(value: object) -> bool:
    """True if value is missing, None, or an empty SecretStr / empty string."""
    if value is None:
        return True
    if isinstance(value, SecretStr):
        return not value.get_secret_value()
    if isinstance(value, str):
        return not value
    return False


def _guard_all_sources() -> None:
    """Scan every configuration source for protected names before anything is parsed.

    Checked explicitly rather than relying on pydantic's ``extra='forbid'``, because an
    unknown *environment variable* is not an "extra field" — it is simply never read, which
    is precisely the silent failure this guard exists to prevent.
    """
    _reject_protected(dict(os.environ), origin="Environment")

    if ENV_FILE.is_file():
        _reject_protected(dict(dotenv_values(ENV_FILE)), origin=f"{ENV_FILE.name}")

    if SETTINGS_YAML.is_file():
        raw = yaml.safe_load(SETTINGS_YAML.read_text(encoding="utf-8")) or {}
        _reject_protected(raw, origin=f"{SETTINGS_YAML.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Tunable sections (config/settings.yaml)
# ──────────────────────────────────────────────────────────────────────────────


class _Section(BaseModel):
    """Base for every config section: immutable, and unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionSettings(_Section):
    timezone: Literal["Asia/Kolkata"] = "Asia/Kolkata"
    watchlist_only: bool = True


class CapitalSettings(_Section):
    """Account size, from which the daily and per-trade risk limits are derived.

    Leaving ``session_budget_inr`` at 0 keeps the CLAUDE.md §1 constants (₹500 / ₹100) in
    force — that is the shipped default and the fallback for any value that will not resolve.
    See :mod:`tachyon.risk.budget`; the percentages mean nothing on their own.

    These are deliberately *not* named after the constants they replace. ``config.py`` rejects
    any source mentioning ``DAILY_LOSS_LIMIT_INR`` or its near-misses, and that guard stays —
    what changes is that a limit may now be *derived* from capital, not that the constant
    became writable.
    """

    #: Trading capital for the session. 0 disables dynamic sizing entirely.
    session_budget_inr: Decimal = Field(default=Decimal("0"), ge=0)

    #: Percent of capital that may be lost in a day before the kill switch trips.
    max_daily_drawdown_pct: Decimal = Field(default=Decimal("2.0"), gt=0, le=100)

    #: Percent of capital risked on a single position if its stop is hit.
    per_trade_risk_pct: Decimal = Field(default=Decimal("1.0"), gt=0, le=100)

    @model_validator(mode="after")
    def _per_trade_within_daily(self) -> CapitalSettings:
        """A single trade may not be allowed to risk more than the whole day.

        Rejected rather than warned, because it is not a risk preference — it is arithmetic
        that cannot be honoured. Sizing takes ``min(per_trade, headroom)``, so a per-trade
        budget above the daily one is silently unreachable: the operator would believe each
        trade could risk more than it ever can. Ratios *within* the day (§6.3's 20 % shape) are
        a preference, and those only warn.
        """
        if self.per_trade_risk_pct > self.max_daily_drawdown_pct:
            raise ValueError(
                f"capital.per_trade_risk_pct ({self.per_trade_risk_pct}%) exceeds "
                f"capital.max_daily_drawdown_pct ({self.max_daily_drawdown_pct}%) — one trade "
                f"cannot be permitted to lose more than the whole day's budget"
            )
        return self


class WatchlistItem(_Section):
    symbol: str
    token: str
    exchange: Literal["NSE", "BSE", "NFO"] = "NSE"
    tick_size: Decimal = Decimal("0.05")
    lot_size: int = Field(default=1, ge=1)

    #: Broker trading symbol, when it differs from :attr:`symbol` — e.g. ``RELIANCE-EQ`` on
    #: NSE cash, or a full F&O contract name. Left unset, cash-segment entries get an ``-EQ``
    #: suffix appended by ``execution.builder.trading_symbol_for``; derivatives must spell it
    #: out here, because a contract name cannot be derived from the underlying.
    trading_symbol: str | None = None

    @field_validator("tick_size")
    @classmethod
    def _positive_tick(cls, value: Decimal) -> Decimal:
        if value <= 0:
            raise ValueError("tick_size must be positive — it gates all price rounding")
        return value


class PositionSettings(_Section):
    max_concurrent: int = Field(default=2, ge=1, le=10)
    max_per_symbol: int = Field(default=1, ge=1, le=1)  # pyramiding is banned (CLAUDE.md §8.1)


class MathEngineSettings(_Section):
    tick_buffer_size: int = Field(default=2000, ge=100)
    candle_buffer_size: int = Field(default=500, ge=50)
    obi_levels: tuple[int, ...] = (1, 3, 5)
    ema_periods: tuple[int, ...] = (9, 21, 50)
    vwap_band_sigma: float = Field(default=1.5, gt=0)
    warmup_on_boot: bool = True

    @field_validator("obi_levels")
    @classmethod
    def _depth_within_l2(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(level < 1 or level > 5 for level in value):
            raise ValueError("obi_levels must be within 1..5 — SmartAPI publishes 5 depth levels")
        return value


class FeedSettings(_Section):
    """Live feed transport tuning.

    ``stream_url`` overrides the SmartStream WebSocket endpoint. Empty (the default) keeps
    the client's canonical SmartStream v2 URL ``wss://smartapisocket.angelone.in/smart-stream``;
    a non-empty value must use the ``ws://`` or ``wss://`` scheme.
    """

    stream_url: str = ""
    reconnect_backoff_seconds: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0)
    max_reconnect_attempts: int = Field(default=20, ge=1)


class IpcSettings(_Section):
    hwm: int = Field(default=10_000, ge=1)
    linger_ms: int = Field(default=0, ge=0)
    ui_conflate: bool = True


class RecorderSettings(_Section):
    """Track 1 LOB harvesting sidecar — ``scripts/run_recorder.py``.

    Every value here is a knob on a process that only ever *reads* the tick spine. None of it
    can reach the trading path: the recorder runs in its own interpreter and the publisher
    sends with ``NOBLOCK``, so the worst a misconfiguration here can do is lose data the
    system was going to discard anyway.
    """

    enabled: bool = True

    streams: tuple[Literal["depth", "tick"], ...] = ("depth", "tick")
    """Which spines to persist. ``depth`` is the L2 book, ``tick`` the trade prints —
    ``Tick.seq`` is the only loss signal on the wire, so dropping it costs gap detection."""

    queue_size: int = Field(default=100_000, ge=1_000)
    """Rows buffered between the socket thread and the writer thread. Overflow drops the row
    and is counted; it never blocks the reader."""

    row_group_size: int = Field(default=4_096, ge=64, le=1_048_576)
    """Rows accumulated before a Parquet row group is written. Small groups bloat the file
    with per-group statistics; huge ones lose more data to a hard kill."""

    flush_interval_seconds: float = Field(default=30.0, gt=0, le=600)
    """Force a partial row group after this long, so a thinly traded symbol still reaches
    disk. Bounds the data a crash can lose from an in-memory buffer."""

    rotate_minutes: int = Field(default=60, ge=1, le=1_440)
    """Close the current file and start a new one on this cadence. An open Parquet file has
    no footer and is unreadable, so this bounds what an ungraceful kill destroys."""

    compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"
    """ZSTD wins on both ratio and decompression speed for float/int columns, which is what
    the offline training loader cares about."""


class StrategySettings(_Section):
    """VWAP + OBI confluence tunables — CLAUDE.md §3.1, §8.

    Every value here may be tuned between sessions. None of them can loosen a §1 risk limit:
    the strategy only ever decides *whether* to ask, and the Risk Engine decides the answer.
    """

    obi_threshold: float = Field(default=0.3, gt=0.0, lt=1.0)
    """Minimum Order Book Imbalance magnitude for a directional signal. Bounded below 1.0
    because OBI is normalised to ``[-1, +1]`` and a threshold of 1 can only fire on a
    completely one-sided book, which is a data fault rather than a trade."""

    ema_period: int = Field(default=21, ge=2)
    """Which EMA is the trend filter. Must be one of ``math_engine.ema_periods`` — a period
    that is not being computed would silently disable the filter."""

    require_obi_agreement: bool = True
    """Require the L1 and depth-weighted OBI to agree beyond the threshold. Their disagreeing
    is the classic spoofing shape (CLAUDE.md §3.1): size at the touch pulling one way while
    the ladder pulls the other. Trading it means trading into the spoof."""

    evaluate_every_n_ticks: int = Field(default=20, ge=1)
    """Signals are evaluated on this cadence and on every candle close. Every tick would be
    affordable (~16 µs) but would produce a burst of identical signals on one move."""

    cooldown_on_every_exit: bool = True
    """Apply the re-entry cooldown after any close, not only a stop-out. CLAUDE.md §8.1
    mandates the stop-out case; this extends it to winners too, which is what prevents
    thrashing around a volatile VWAP line. Strictly more conservative either way."""

    max_signals_per_symbol_per_session: int = Field(default=0, ge=0)
    """Hard cap on entries per symbol per day. ``0`` disables the cap."""

    max_vwap_extension_pct: float = Field(default=1.5, gt=0)
    """Overextension guardrail (CLAUDE.md §4, entry check 12): a LONG is vetoed when the
    LTP sits more than this percentage **above** session VWAP — the signature of buying
    the top of a vertical breakout, where mean reversion against the position is immediate.
    Exits are never gated by this; it protects entries only."""


class ExecutionSettings(_Section):
    order_variety: Literal["ROBO"] = "ROBO"
    product_type: Literal["BO"] = "BO"
    trail_activation_r: Decimal = Field(default=Decimal("1.0"), gt=0)
    trail_step_r: Decimal = Field(default=Decimal("0.5"), gt=0)
    move_t2_stop_to_breakeven_on_t1: bool = True
    max_order_retries: int = Field(default=3, ge=0, le=3)  # CLAUDE.md §6.4 caps this at 3
    rate_limit_per_second: int = Field(default=8, ge=1)


class SentinelSettings(_Section):
    premarket_run_ist: str = "08:45"
    news_scan_interval_minutes: int = Field(default=15, ge=1)
    request_timeout_seconds: float = Field(default=8.0, gt=0, le=30)
    news_blacklist_score: float = Field(default=0.7, ge=0, le=1)
    news_blacklist_minutes: int = Field(default=30, ge=0)
    max_session_requests: int = Field(default=120, ge=0)

    risk_off_confidence: int = Field(default=80, ge=0, le=100)
    """Minimum confidence at which a RISK_OFF regime blocks new entries for the session.
    Below it the Sentinel downsizes instead of halting — it may always restrict more, and this
    knob only decides how much evidence is needed before it restricts completely."""


class UiSettings(_Section):
    ws_max_hz: int = Field(default=10, ge=1, le=60)
    stale_badge_after_seconds: float = Field(default=2.0, gt=0)


class RLSettings(_Section):
    """PPO training hyperparameters.

    These control the offline training loop in tachyon.rl.train.PPOTrainer.
    They do NOT affect the live trading path.
    """

    lr: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, gt=0, le=1)
    clip_ratio: float = Field(default=0.2, gt=0, le=1)
    value_clip: float = Field(default=0.2, gt=0, le=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    batch_size: int = Field(default=64, ge=1)
    minibatch_size: int = Field(default=32, ge=1)
    epochs: int = Field(default=4, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"


class LoggingSettings(_Section):
    json_lines: bool = True
    directory: str = "logs"


# ──────────────────────────────────────────────────────────────────────────────
# Settings root
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Resolved runtime configuration. Immutable once constructed."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        yaml_file=SETTINGS_YAML,
        extra="forbid",
        frozen=True,
        case_sensitive=False,
    )

    # ── from .env ────────────────────────────────────────────────────────────
    trading_mode: TradingMode = DEFAULT_TRADING_MODE
    # Override for the 15:15 IST square-off deadline, populated by the --mock CLI flag
    # only (offline rehearsals). Accepts "HH:MM" (24-hour); absent or blank keeps the
    # unconditional 15:15 deadline — there is no other way to move it.
    mock_squareoff_time: str | None = None

    smartapi_api_key: SecretStr = SecretStr("")
    smartapi_client_code: SecretStr = SecretStr("")
    smartapi_password: SecretStr = SecretStr("")
    smartapi_totp_secret: SecretStr = SecretStr("")

    #: Streaming credential issued by ``generateSession`` at login, distinct from the API key.
    #: Supplied here only until Phase 7's SmartApiClient performs the login itself; a token
    #: pasted into .env expires and will start failing the handshake.
    smartapi_feed_token: SecretStr = SecretStr("")

    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash-lite"
    sentinel_enabled: bool = True

    #: Operator alerts (utils/telegram_alerts.py). A bot token is a credential — anyone
    #: holding it can post as the bot — so it lives in .env with the others and never in
    #: settings.yaml, which is tracked (§8). Absent either field, alerts are inert and
    #: trading is unaffected.
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""
    telegram_alerts_enabled: bool = True

    # ── ZMQ port map — single source of truth ───────────────────────────────────
    # Every endpoint has EXACTLY ONE binder; everyone else connects:
    #   5555  tick spine     PUB  binds: ingestor (ingestion/service.py)
    #                        SUB  connects: brain, UI, recorder
    #   5556  state spine    PUB  binds: brain (strategy/telemetry.py)
    #                        SUB  connects: UI
    #   5557  sentinel       reserved — nothing binds it yet
    #   5566  sidecar control REP binds: tachyon_sidecar (EngineSidecar.cpp)
    #                        REQ  connects: rl.export.signal_hot_swap
    #   5567  action spine   PUSH binds: tachyon_sidecar
    #                        PULL connects: execution/router.py
    # A second bind() on any of these is a startup fault (EADDRINUSE), never a
    # silent share — the sidecar once defaulted its control server to 5555 and
    # contested the tick spine; keep new services on connect() instead.
    zmq_tick_endpoint: str = "tcp://127.0.0.1:5555"
    zmq_state_endpoint: str = "tcp://127.0.0.1:5556"
    zmq_sentinel_endpoint: str = "tcp://127.0.0.1:5557"

    #: Sidecar control (REQ/REP). The C++ sidecar binds the REP server; hot-swap
    #: signals connect to it as REQ. ``main.py`` passes this to the sidecar as
    #: ``--endpoint`` and ``rl.export.DEFAULT_CONTROL_ENDPOINT`` mirrors it.
    zmq_sidecar_control_endpoint: str = "tcp://127.0.0.1:5566"

    #: The action spine (Track 2 → Track 1): the C++ sidecar PUSHes discrete
    #: actions here; the ExecutionRouter PULLs them into the risk gate.
    zmq_action_endpoint: str = "tcp://127.0.0.1:5567"

    ui_host: str = "127.0.0.1"
    ui_port: int = Field(default=8787, ge=1024, le=65535)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ── from config/settings.yaml ────────────────────────────────────────────
    session: SessionSettings = SessionSettings()
    capital: CapitalSettings = CapitalSettings()
    watchlist: tuple[WatchlistItem, ...] = ()
    positions: PositionSettings = PositionSettings()
    math_engine: MathEngineSettings = MathEngineSettings()
    feed: FeedSettings = FeedSettings()
    ipc: IpcSettings = IpcSettings()
    recorder: RecorderSettings = RecorderSettings()
    strategy: StrategySettings = StrategySettings()
    execution: ExecutionSettings = ExecutionSettings()
    sentinel: SentinelSettings = SentinelSettings()
    ui: UiSettings = UiSettings()
    rl: RLSettings = RLSettings()
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - rebuilt below
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003 - fixed pydantic hook
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Docker/K8s secret files are deliberately dropped: this system runs as a desktop
        # process and its only secret source is .env (CLAUDE.md §8).
        #
        # Both file sources are resolved here, at load time, rather than baked into
        # model_config at class-definition time — so a test (or a future --config flag) can
        # redirect them. For ``.env`` that is not a convenience: without it, an operator's real
        # credentials leak into every test run on that machine, and a test suite that finds a
        # live API key will go and use it.
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(
                settings_cls, env_file=ENV_FILE, env_file_encoding="utf-8", case_sensitive=False
            ),
            YamlConfigSettingsSource(settings_cls, yaml_file=SETTINGS_YAML),
        )

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_aliases(cls, data: object) -> Mapping[str, Any]:
        """Apply legacy ANGEL_* / SMARTAPI_CLIENT_ID / SMARTAPI_PIN aliases.

        Runs before _guard_constants so guards still see the merged view. For each of
        the four SmartAPI credential fields, fills in a value from
        _resolve_legacy_aliases() only if the incoming data is blank or missing. The
        SMARTAPI_* env names win over the legacy aliases by virtue of
        _LEGACY_ENV_FALLBACKS ordering. Non-mapping data (e.g. an init positional
        value) is returned unchanged so pydantic's own coercion still runs.
        """
        if not isinstance(data, Mapping):
            return cast(Mapping[str, Any], data)
        merged: dict[str, Any] = dict(data)
        for field_name, value in _resolve_legacy_aliases().items():
            if _is_blank(merged.get(field_name)):
                merged[field_name] = SecretStr(value)
        return merged

    @model_validator(mode="before")
    @classmethod
    def _guard_constants(cls, data: Any) -> Any:
        """Fail the boot if any source tries to move a hard limit."""
        _guard_all_sources()
        if isinstance(data, Mapping):
            _reject_protected(dict(data), origin="Merged configuration")
        return data

    @field_validator("trading_mode", mode="before")
    @classmethod
    def _fail_safe_trading_mode(cls, value: Any) -> TradingMode:
        """Resolve the trading mode, failing safe to PAPER.

        Only the exact string ``LIVE`` selects live trading. Blank, missing, misspelled, or
        garbage values all resolve to PAPER — there is no input that accidentally goes live
        (CLAUDE.md §9).
        """
        if isinstance(value, TradingMode):
            return value
        if value is None:
            return DEFAULT_TRADING_MODE
        return TradingMode.LIVE if str(value).strip().upper() == "LIVE" else TradingMode.PAPER

    @model_validator(mode="after")
    def _strategy_ema_is_computed(self) -> Settings:
        """The trend-filter EMA must be one the math engine actually computes.

        Otherwise the filter silently evaluates against a period that is never produced, and
        the signal reduces to "price vs VWAP and OBI" with no trend gate at all — a strictly
        looser strategy than the operator configured, with nothing to indicate it.
        """
        if self.strategy.ema_period not in self.math_engine.ema_periods:
            raise ValueError(
                f"strategy.ema_period={self.strategy.ema_period} is not in "
                f"math_engine.ema_periods={list(self.math_engine.ema_periods)}. Add it there, "
                f"or the trend filter would evaluate against an EMA nothing computes."
            )
        return self

    @field_validator("watchlist")
    @classmethod
    def _unique_symbols(cls, value: tuple[WatchlistItem, ...]) -> tuple[WatchlistItem, ...]:
        symbols = [item.symbol for item in value]
        duplicates = {s for s in symbols if symbols.count(s) > 1}
        if duplicates:
            raise ValueError(f"Duplicate watchlist symbols: {sorted(duplicates)}")
        return value

    # ── derived helpers ──────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        """True only in LIVE mode. Callers must also require interactive confirmation."""
        return self.trading_mode is TradingMode.LIVE

    @property
    def log_dir(self) -> Path:
        return PROJECT_ROOT / self.logging.directory

    def symbol_tokens(self) -> dict[str, str]:
        """``{symbol: exchange token}`` for the configured watchlist."""
        return {item.symbol: item.token for item in self.watchlist}

    def find_symbol(self, symbol: str) -> WatchlistItem | None:
        """Look up a watchlist entry. ``None`` means *do not trade it* (CLAUDE.md §8.1)."""
        return next((item for item in self.watchlist if item.symbol == symbol), None)

    def missing_live_credentials(self) -> tuple[str, ...]:
        """Names of credentials required for LIVE trading that are absent or blank.

        Resolves each credential alias-aware: the resolved smartapi_* field on self
        (already populated by _apply_legacy_aliases) is consulted first; if it is
        still empty, the legacy ANGEL_* env names are consulted directly. Either
        source is enough for the credential to be considered set, so this remains
        backward compatible with operators still using the ANGEL_* convention.
        """
        resolved: dict[str, str] = {
            "smartapi_api_key": self.smartapi_api_key.get_secret_value(),
            "smartapi_client_code": self.smartapi_client_code.get_secret_value(),
            "smartapi_password": self.smartapi_password.get_secret_value(),
            "smartapi_totp_secret": self.smartapi_totp_secret.get_secret_value(),
        }
        for field_name, env_names in _LEGACY_ENV_FALLBACKS.items():
            if resolved[field_name]:
                continue
            for env_name in env_names:
                legacy = os.environ.get(env_name)
                if legacy:
                    resolved[field_name] = legacy
                    break
        return tuple(
            label
            for label, field_name in (
                ("SMARTAPI_API_KEY", "smartapi_api_key"),
                ("SMARTAPI_CLIENT_CODE", "smartapi_client_code"),
                ("SMARTAPI_PASSWORD", "smartapi_password"),
                ("SMARTAPI_TOTP_SECRET", "smartapi_totp_secret"),
            )
            if not resolved[field_name]
        )

    def validate_live_ready(self) -> None:
        """Raise unless the process is genuinely equipped to trade live.

        Called at boot by the LIVE path only, *after* interactive confirmation.
        """
        if not self.is_live:
            return
        missing = self.missing_live_credentials()
        if missing:
            raise ValueError(
                f"TRADING_MODE=LIVE but these credentials are unset: {', '.join(missing)}."
            )
        if not self.watchlist:
            raise ValueError("TRADING_MODE=LIVE with an empty watchlist — nothing may be traded.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call :func:`reload_settings` in tests."""
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache and re-read every source. Intended for tests and boot-time reload."""
    get_settings.cache_clear()
    return get_settings()
