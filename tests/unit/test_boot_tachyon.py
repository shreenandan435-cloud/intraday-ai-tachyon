"""The interactive boot sequence's decision logic — CLAUDE.md §1.3, §8.1, §9.

Only the parts that decide something are covered here: how a typed budget is parsed, what the
operator is shown before they confirm it, and the two guards that refuse to run at all. The
orchestrator hand-off is not exercised — it boots the real system.

Nothing here runs ``main()``. The two guards are driven against a socket bound to an ephemeral
port and an injected :class:`~tachyon.core.state.DailyLock` under ``tmp_path``, so no test can
reach the ZeroMQ spines at 5555/5556 or the real ``data/journal/``.
"""

from __future__ import annotations

import importlib.util
import socket
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from tachyon.core.config import CapitalSettings, Settings
from tachyon.core.state import DailyLock

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "boot_tachyon.py"


def _load() -> ModuleType:
    """Import the script by path — it lives in ``scripts/``, not in the package."""
    spec = importlib.util.spec_from_file_location("boot_tachyon", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boot = _load()


def _settings(**kwargs: object) -> Settings:
    return Settings(**kwargs)  # type: ignore[arg-type]


@pytest.fixture
def bound_port() -> Iterator[int]:
    """A real listening socket on an ephemeral port — never 5555 or 5556."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        yield int(listener.getsockname()[1])


@pytest.fixture
def free_port() -> int:
    """A port nothing is listening on: bound, read, then closed."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class TestParseBudget:
    """Typed at 09:05 by someone who has had one coffee. It has to be forgiving about form
    and completely unforgiving about meaning."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("50000", "50000"),
            ("  50000  ", "50000"),
            ("50,000", "50000"),
            ("50_000", "50000"),
            ("Rs.50000", "50000"),
            ("rs 50000", "50000"),
            ("INR 50000", "50000"),
            ("₹50000", "50000"),
            ("50000.50", "50000.50"),
        ],
    )
    def test_it_accepts_the_ways_a_rupee_amount_gets_typed(self, raw: str, expected: str) -> None:
        assert boot.parse_budget(raw) == Decimal(expected)

    @pytest.mark.parametrize("raw", ["", "   ", "abc", "50k", "1e", "Rs.", "--", "nan", "inf"])
    def test_an_unusable_amount_is_rejected_rather_than_guessed(self, raw: str) -> None:
        """`nan` and `inf` are in this list on purpose: `Decimal('nan')` parses fine, and a
        non-finite budget would resolve limits of NaN — which every comparison in §4 is false
        against, so the loss limit would never trip."""
        assert boot.parse_budget(raw) is None

    @pytest.mark.parametrize("raw", ["0", "-1", "-50000"])
    def test_a_non_positive_budget_is_rejected(self, raw: str) -> None:
        """Zero is a valid *config* value meaning "use the §1 constants", but as a typed answer
        to "what is today's budget" it is far more likely to be a slip. The operator who wants
        the constants back edits the file or passes --skip-scan."""
        assert boot.parse_budget(raw) is None


class TestDescribeBudget:
    """What the operator reads before they type yes. This is the safety-critical screen —
    ``session_budget_inr`` is a risk base, and someone who reads it as buying power has
    authorised a much larger loss than they think."""

    @staticmethod
    def _with_capital(drawdown: str, per_trade: str) -> Settings:
        return _settings(
            capital=CapitalSettings(
                session_budget_inr=Decimal("0"),
                max_daily_drawdown_pct=Decimal(drawdown),
                per_trade_risk_pct=Decimal(per_trade),
            )
        )

    def test_it_shows_the_derived_rupee_limits_not_just_the_percentages(self) -> None:
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("50000"))

        assert "DAILY LOSS LIMIT  Rs.1,000.00" in text, (
            "50000 x 2% must be shown as rupees, not left as '2%'"
        )
        assert "PER-TRADE RISK    Rs.200.00" in text

    def test_it_says_the_number_is_not_spending_money(self) -> None:
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("50000"))

        assert "RISK BUDGET, NOT SPENDING MONEY" in text

    def test_it_warns_when_the_day_may_lose_more_than_the_constitutional_limit(self) -> None:
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("50000"))

        assert "2.00x the section-1 limit" in text

    def test_it_does_not_warn_when_the_limit_stays_under_the_constant(self) -> None:
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("10000"))

        assert "section-1 limit before it halts" not in text
        assert "200.00" in text  # 10000 x 2%

    def test_it_states_how_many_losers_exhaust_the_day(self) -> None:
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("50000"))

        assert "5 losing trade(s) at full risk exhaust the day's budget." in text

    def test_it_flags_a_concentration_above_the_twenty_percent_shape(self) -> None:
        """§6.3 intends no single loser to cost more than a fifth of the day. Config that
        breaks it is honoured — and said out loud."""
        text = boot.describe_budget(self._with_capital("2.0", "1.0"), Decimal("50000"))

        assert "concentration shape" in text
        assert "2 losing trade(s)" in text

    def test_it_shows_the_affordability_ceiling_the_scanner_will_use(self) -> None:
        """The same number is doing two different jobs — risk base and buying power. The
        operator should see both readings rather than discover the second one later."""
        text = boot.describe_budget(self._with_capital("2.0", "0.4"), Decimal("50000"))

        assert "shares priced up to Rs.250,000" in text  # 50000 x 5
        assert "expect that filter to reject nothing" in text

    def test_a_budget_that_rounds_below_a_paisa_says_the_constants_stay(self) -> None:
        """`SessionBudget.resolve` falls back rather than sizing off a sub-paise limit. If the
        screen still printed the derived numbers it would be describing limits that are not the
        ones in force."""
        text = boot.describe_budget(self._with_capital("0.001", "0.001"), Decimal("0.01"))

        assert "section-1 constants stay in force" in text


class TestLiveSessionGuard:
    """The guard the operator's own warning is about: never rewrite the watchlist under a
    running Brain, which may be holding a position in a symbol the new list does not contain."""

    def test_a_bound_tick_spine_refuses_the_run(self, bound_port: int) -> None:
        settings = _settings(zmq_tick_endpoint=f"tcp://127.0.0.1:{bound_port}")

        refusal = boot.guard_no_live_session(settings)

        assert refusal is not None
        assert "tick spine" in refusal
        assert "a Tachyon session is running" in refusal

    def test_a_bound_state_spine_refuses_the_run(self, bound_port: int, free_port: int) -> None:
        settings = _settings(
            zmq_tick_endpoint=f"tcp://127.0.0.1:{free_port}",
            zmq_state_endpoint=f"tcp://127.0.0.1:{bound_port}",
        )

        refusal = boot.guard_no_live_session(settings)

        assert refusal is not None
        assert "state spine" in refusal

    def test_closed_ports_permit_the_run(self, free_port: int) -> None:
        settings = _settings(
            zmq_tick_endpoint=f"tcp://127.0.0.1:{free_port}",
            zmq_state_endpoint=f"tcp://127.0.0.1:{free_port}",
        )

        assert boot.guard_no_live_session(settings) is None

    @pytest.mark.parametrize(
        "endpoint", ["ipc:///tmp/x", "inproc://x", "tcp://127.0.0.1", "", "nonsense"]
    )
    def test_a_non_tcp_endpoint_is_not_probed(self, endpoint: str) -> None:
        """The guard proves a session is running; it never claims one is not. An endpoint it
        cannot probe returns False, and the write proceeds — which is why this is the *first*
        of several protections, not the only one."""
        assert boot._endpoint_is_bound(endpoint) is False


class TestDailyLockGuard:
    def test_an_absent_lock_permits_the_run(self, tmp_path: Path) -> None:
        assert boot.guard_not_locked(DailyLock(path=tmp_path / "daily_lock.txt")) is None

    def test_an_engaged_lock_refuses_the_run(self, tmp_path: Path) -> None:
        """The day is over (§1.2). A config rewrite cannot reopen it and would destroy the
        configuration that was in force when the limit tripped."""
        lock = DailyLock(path=tmp_path / "daily_lock.txt")
        lock.engage("daily loss limit breached")

        refusal = boot.guard_not_locked(lock)

        assert refusal is not None
        assert "daily lock is engaged" in refusal
        assert "daily loss limit breached" in refusal

    def test_an_unreadable_lock_refuses_the_run(self, tmp_path: Path) -> None:
        """A lock that will not parse counts as engaged everywhere else in this system
        (§1.2); it counts as engaged here too."""
        path = tmp_path / "daily_lock.txt"
        path.write_text("this is not a lock file", encoding="utf-8")

        assert boot.guard_not_locked(DailyLock(path=path)) is not None
