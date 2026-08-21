"""Headless master boot — paper trading mode for autonomous cloud deployment.

Run::

    .venv\\Scripts\\python.exe scripts/boot_tachyon.py

Sequence
--------
1. **Refuse to run over a live session.** If anything is already bound to the ZeroMQ tick or
   state endpoint, this script exits without touching a single file.
2. **Refuse to run under an engaged daily lock.** The lock means the day is over (§1.2).
3. **Load configuration from environment** — no interactive prompts.
4. **Scan the pre-open board** and pick the top movers the budget can trade.
5. **Write config** into ``config/settings.yaml`` atomically.
6. **Launch the orchestrator** in this process.

Configuration
-------------
All configuration via environment variables or .env file:

- ``PAPER_TRADING_BUDGET_INR`` — Session budget (default: 100000)
- ``PAPER_MAX_DAILY_DRAWDOWN_PCT`` — Max daily drawdown % (default: 2.0)
- ``PAPER_PER_TRADE_RISK_PCT`` — Per-trade risk % (default: 1.0)
- ``SCAN_SYMBOLS`` — Number of symbols to select (default: 4)
- ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` — Alert credentials

Flags
-----
``--dry-run``     Scan, render, validate, print — write nothing, launch nothing.
``--no-launch``   Do the writes, stop before the orchestrator.
``--skip-scan``   Set the budget only; leave the watchlist alone.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.core import eventloop  # noqa: E402
from tachyon.core.clock import now_ist  # noqa: E402
from tachyon.core.config import Settings, get_settings, reload_settings  # noqa: E402
from tachyon.core.config_writer import (  # noqa: E402
    ConfigWriteError,
    format_decimal,
    write_settings,
)
from tachyon.core.constants import (  # noqa: E402
    DAILY_LOSS_LIMIT_INR,
    PER_TRADE_RISK_INR,
    SETTINGS_YAML,
)
from tachyon.core.logger import configure_logging, get_logger  # noqa: E402
from tachyon.core.state import DailyLock  # noqa: E402
from tachyon.ingestion.instruments import InstrumentMaster  # noqa: E402
from tachyon.risk.budget import SessionBudget  # noqa: E402
from tachyon.strategy.scanner import (  # noqa: E402
    DEFAULT_SELECTION_SIZE,
    INTRADAY_LEVERAGE,
    NSE_PRE_OPEN_URL,
    PENNY_PRICE_FLOOR_INR,
    MasterSymbolResolver,
    NsePreOpenSource,
    PreMarketScanner,
    PreOpenFetchError,
    ScanResult,
)

RULE = "=" * 78

#: Exit codes. 0 launches; anything else stops before the orchestrator.
EXIT_OK = 0
EXIT_CONFIG_FAULT = 2
EXIT_ABORTED = 3
EXIT_REFUSED = 4


def _say(message: str = "") -> None:
    """Console output for the operator. Deliberately not the log — this is a conversation."""
    print(message, flush=True)


def _get_env_budget(settings: Settings) -> Decimal:
    """Load budget from environment with fallback to settings.yaml."""
    # Priority: env var > settings.yaml > default (100000)
    env_budget = os.environ.get("PAPER_TRADING_BUDGET_INR")
    if env_budget:
        try:
            return Decimal(env_budget.strip().replace(",", ""))
        except InvalidOperation:
            pass

    if settings.capital.session_budget_inr > 0:
        return settings.capital.session_budget_inr

    return Decimal("100000")  # Default ₹1,00,000


def _get_env_scan_symbols() -> int:
    """Load scan symbols count from environment."""
    env_symbols = os.environ.get("SCAN_SYMBOLS")
    if env_symbols:
        try:
            return int(env_symbols)
        except ValueError:
            pass
    return DEFAULT_SELECTION_SIZE


def _endpoint_is_bound(endpoint: str, *, timeout: float = 0.3) -> bool:
    """True if something already accepts connections on a ``tcp://host:port`` endpoint."""
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(timeout)
            return probe.connect_ex((parsed.hostname, parsed.port)) == 0
    except OSError:
        return False


def guard_no_live_session(settings: Settings) -> str | None:
    """Return a refusal reason if a session appears to be running, else ``None``."""
    for name, endpoint in (
        ("tick spine", settings.zmq_tick_endpoint),
        ("state spine", settings.zmq_state_endpoint),
    ):
        if _endpoint_is_bound(endpoint):
            return (
                f"the {name} at {endpoint} is already bound — a Tachyon session is running.\n"
                "  Rewriting config/settings.yaml under a live Brain would change which symbols\n"
                "  CLAUDE.md 8.1 permits it to trade, possibly while it holds a position.\n"
                "  Shut the session down cleanly (Ctrl-C once, let it flatten), then re-run."
            )
    return None


def guard_not_locked(lock: DailyLock | None = None) -> str | None:
    """Return a refusal reason if today's daily lock is engaged, else ``None``."""
    lock = lock if lock is not None else DailyLock()
    if not lock.is_engaged():
        return None
    record = lock.read()
    detail = f" ({record.reason})" if record is not None and record.reason else ""
    return (
        f"the daily lock is engaged for today{detail}.\n"
        "  The session is over (CLAUDE.md 1.2) and a config rewrite cannot reopen it. It would\n"
        "  only destroy the configuration that was in force when the limit tripped.\n"
        f"  Lock file: {lock.path}"
    )


def parse_budget(raw: str) -> Decimal | None:
    """Parse a typed rupee amount. ``None`` means "that was not a usable number"."""
    text = raw.strip().replace(",", "").replace("_", "").lstrip("₹").strip()
    for prefix in ("rs.", "inr", "rs"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break
    if not text:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite() or value <= 0:
        return None
    return value


def describe_budget(settings: Settings, budget: Decimal) -> str:
    """The arithmetic §1.3 will apply, spelled out before it is committed."""
    resolved = SessionBudget.resolve(
        capital=budget,
        drawdown_pct=settings.capital.max_daily_drawdown_pct,
        per_trade_pct=settings.capital.per_trade_risk_pct,
    )
    lines = [
        RULE,
        "  PAPER TRADING BUDGET — HEADLESS MODE",
        "  CLAUDE.md 1.3 derives this session's hard limits from it:",
        "",
        f"    session_budget_inr        Rs.{budget:,f}",
        f"    x max_daily_drawdown_pct  {settings.capital.max_daily_drawdown_pct}%"
        f"  ->  DAILY LOSS LIMIT  Rs.{resolved.daily_loss_limit:,f}",
        f"    x per_trade_risk_pct      {settings.capital.per_trade_risk_pct}%"
        f"  ->  PER-TRADE RISK    Rs.{resolved.per_trade_risk:,f}",
        "",
        f"    CLAUDE.md section 1 defaults are Rs.{DAILY_LOSS_LIMIT_INR} / "
        f"Rs.{PER_TRADE_RISK_INR}.",
    ]

    if not resolved.is_dynamic:
        lines.append("    ** This budget did not resolve; the section-1 constants stay in force.")
    else:
        if resolved.daily_loss_limit > DAILY_LOSS_LIMIT_INR:
            multiple = (resolved.daily_loss_limit / DAILY_LOSS_LIMIT_INR).quantize(Decimal("0.01"))
            lines.append(
                f"    ** This day may lose {multiple}x the section-1 limit before it halts."
            )
        losers = int(resolved.daily_loss_limit / resolved.per_trade_risk)
        lines.append(f"    {losers} losing trade(s) at full risk exhaust the day's budget.")
        if resolved.exceeds_concentration_guidance:
            lines.append(
                "    ** Above section 6.3's 20% concentration shape (it intends at least 5)."
            )

    lines.extend(
        [
            "",
            "  The scanner also reads this number as buying power for its affordability filter,",
            f"  at {INTRADAY_LEVERAGE}x intraday margin: it will consider shares priced up to "
            f"Rs.{budget * INTRADAY_LEVERAGE:,f}.",
            "  On NSE cash that admits every symbol, so expect that filter to reject nothing.",
            RULE,
        ]
    )
    return "\n".join(lines)


async def run_scan(budget: Decimal, *, size: int) -> ScanResult:
    """Refresh the scrip master, fetch the pre-open board, and select.

    Raises:
        PreOpenFetchError: the board was unavailable. Callers must leave the watchlist alone.
    """
    master = InstrumentMaster()
    try:
        await master.ensure_fresh()
    except Exception as exc:  # noqa: BLE001
        _say(f"  [warn] scrip master refresh failed: {exc}")

    resolver = MasterSymbolResolver.from_master(master)
    _say(f"  [ ok ] scrip master: {len(resolver)} NSE cash-equity instruments")

    scanner = PreMarketScanner(source=NsePreOpenSource(), resolver=resolver, size=size)
    return await scanner.scan(budget=budget)


def report_scan(result: ScanResult) -> None:
    """Print the selection and a tally of what it discarded."""
    _say()
    _say(RULE)
    _say(f"  PRE-MARKET SCAN — {result.considered} symbols on the board")
    _say(
        f"  filters: series {result.required_series} | price >= Rs.{PENNY_PRICE_FLOOR_INR} | "
        f"pre-open turnover >= Rs.{result.min_turnover:,f}"
    )
    _say(RULE)
    _say(
        f"  {'SYMBOL':<13}{'TOKEN':>8}{'PREV':>11}{'PRE-OPEN':>11}"
        f"{'GAP %':>9}{'PRE-OPEN Rs':>16}   TICK"
    )
    for candidate in result.selected:
        _say(
            f"  {candidate.symbol:<13}{candidate.token:>8}"
            f"{candidate.previous_close:>11,.2f}{candidate.pre_open_price:>11,.2f}"
            f"{candidate.gap_pct:>+9.2f}{candidate.turnover:>16,.0f}"
            f"   {format_decimal(candidate.tick_size)}"
        )
    counts = result.rejection_counts()
    if counts:
        _say()
        _say("  filtered out: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    _say(RULE)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="boot_tachyon",
        description="Headless paper trading boot — budget from env, scan, then launch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="write nothing, launch nothing")
    parser.add_argument("--no-launch", action="store_true", help="write, but do not boot")
    parser.add_argument("--skip-scan", action="store_true", help="budget only; keep the watchlist")
    return parser.parse_args(argv)


def _configure(settings: Settings) -> None:
    """Set up logging once."""
    configure_logging(
        role="tachyon",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        _say(f"\n  [FATAL] configuration did not load: {exc}\n")
        return EXIT_CONFIG_FAULT

    _configure(settings)
    log = get_logger("boot_tachyon")

    _say()
    _say(RULE)
    _say("  I N T R A D A Y   A I   T A C H Y O N   —   headless paper boot")
    _say(f"  {now_ist().isoformat(timespec='seconds')}   mode={settings.trading_mode.value}")
    _say(RULE)

    for refusal in (guard_no_live_session(settings), guard_not_locked()):
        if refusal is not None:
            _say(f"\n  [REFUSED] {refusal}\n")
            return EXIT_REFUSED

    # Load budget from environment (no interactive prompt)
    budget = _get_env_budget(settings)
    scan_symbols = _get_env_scan_symbols()

    _say()
    _say(describe_budget(settings, budget))

    result: ScanResult | None = None
    if not args.skip_scan:
        _say("\n  [ .. ] scanning the NSE pre-open board (one request, whole universe)")
        try:
            result = eventloop.run(run_scan(budget, size=scan_symbols))
        except (PreOpenFetchError, ValueError) as exc:
            _say(f"  [warn] pre-market scan failed: {exc}")
            _say("         the existing watchlist is left exactly as it is.")
            result = None
        else:
            report_scan(result)
            if not result.is_usable:
                _say("  [warn] the scan selected nothing; the existing watchlist stands.")
                result = None

    provenance = ""
    if result is not None:
        provenance = (
            f"top {len(result.selected)} by |gap%| from {NSE_PRE_OPEN_URL} "
            f"({result.considered} considered, budget Rs.{budget})"
        )

    try:
        report = write_settings(
            path=SETTINGS_YAML,
            session_budget_inr=budget,
            watchlist=result.watchlist() if result is not None else None,
            generated_at=now_ist(),
            provenance=provenance,
            dry_run=args.dry_run,
        )
    except ConfigWriteError as exc:
        _say(f"\n  [FATAL] {exc}\n")
        return EXIT_CONFIG_FAULT

    verb = "would write" if args.dry_run else "wrote"
    _say(f"\n  [ ok ] {verb} session_budget_inr = {report.session_budget_inr}")
    if report.symbols:
        _say(f"  [ ok ] {verb} watchlist = {', '.join(report.symbols)}")
    else:
        _say("  [ ok ] watchlist unchanged")
    if report.backup_path is not None:
        _say(f"  [ ok ] previous settings kept at {report.backup_path.name}")

    if args.dry_run:
        _say("\n  --dry-run: nothing was written and nothing will be launched.\n")
        return EXIT_OK

    reload_settings()
    log.info("boot.configuration_written", next="orchestrator" if not args.no_launch else "none")

    if args.no_launch:
        _say("\n  --no-launch: configuration written. Start trading with:")
        _say("      .venv\\Scripts\\python.exe -m tachyon.main\n")
        return EXIT_OK

    _say("\n  [ .. ] handing off to the orchestrator\n")
    from tachyon.main import main as orchestrator_main  # noqa: PLC0415

    return orchestrator_main()


if __name__ == "__main__":
    raise SystemExit(main())