"""Phase 9 Sentinel tests — CLAUDE.md §5.

The Sentinel is the only component in this system that is *allowed* to fail open, and it is the
only one holding a channel from a large language model to real money. Both facts make the
failure paths the interesting tests, not the happy path.

What is proved here:

* **An outage cannot halt trading.** Timeout, 503, connection refused, malformed JSON, missing
  API key, exhausted budget — every one degrades to ``NEUTRAL`` and leaves the gate open.
* **A degraded Sentinel never says RISK_ON.** Failure resolves toward less risk-taking, never
  more. There is no input to the parser that produces ``RISK_ON`` from a broken response.
* **The latch and the ratchet hold.** A RISK_OFF followed by a RISK_ON does not unblock
  trading, and ``size_multiplier`` never increases within a session — otherwise the model
  could increase risk, which §5 forbids outright.
* **`size_multiplier` is ours.** The model is never asked for it and cannot influence it except
  through the regime and confidence fields.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import SentinelSettings, Settings, WatchlistItem
from tachyon.core.state import StateMachine, TradingState
from tachyon.ipc.monitor import FeedMonitor
from tachyon.math_engine import warmup
from tachyon.persistence.journal import JsonlJournal
from tachyon.risk.engine import PENDING_CHECKS, RiskEngine, VetoReason
from tachyon.risk.tracker import PnLTracker, PositionRegistry
from tachyon.sentinel.api import (
    DEFAULT_MODEL,
    GeminiBudgetExhaustedError,
    GeminiClient,
    GeminiNotConfiguredError,
    GeminiRejectedError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from tachyon.sentinel.classifier import (
    NEUTRAL_FALLBACK_REASON,
    NewsClassifier,
    RegimeClassifier,
    coerce_confidence,
    coerce_regime,
    coerce_score,
    strip_code_fence,
)
from tachyon.sentinel.service import PLACEHOLDER_BRIEF, SentinelDaemon
from tachyon.sentinel.state import MacroState, Regime

BRIEF = "SGX Nifty -1.8%, India VIX 22.4, US closed -2.1%, crude +4%"


@pytest.fixture(scope="session", autouse=True)
def _warm_engine() -> None:
    assert warmup() is True


def _clock(hh: int = 11, mm: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=mono)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (
            WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),
            WatchlistItem(symbol="HDFCBANK", token="1333", exchange="NSE"),
        ),
        "gemini_api_key": "test-key",
        "sentinel": SentinelSettings(),
    }
    base.update(overrides)
    return Settings(**base)


def _reply(payload: str, *, status: int = 200) -> httpx.Response:
    """A well-formed Gemini envelope wrapping ``payload`` as the candidate text."""
    return httpx.Response(
        status,
        json={
            "candidates": [{"content": {"parts": [{"text": payload}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 40},
        },
    )


def _client(
    handler: Any,
    *,
    settings: Settings | None = None,
    journal: JsonlJournal | None = None,
    timeout: float = 8.0,
    max_requests: int | None = None,
) -> GeminiClient:
    return GeminiClient(
        settings=settings if settings is not None else _settings(),
        journal=journal,
        client=httpx.AsyncClient(base_url="https://test", transport=httpx.MockTransport(handler)),
        clock=_clock(),
        timeout=timeout,
        max_requests=max_requests,
    )


# ──────────────────────────────────────────────────────────────────────────────
# state.py — the latch and the ratchet
# ──────────────────────────────────────────────────────────────────────────────


class TestMacroState:
    def test_defaults_are_permissive(self) -> None:
        """A Sentinel that has never run must not restrict anything."""
        state = MacroState(clock=_clock())
        assert state.is_trading_allowed
        assert state.size_multiplier == Decimal("1")
        assert state.blocks_entry("RELIANCE") == (False, "")

    def test_confident_risk_off_blocks(self) -> None:
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 91, "VIX +18%, US -2.1%")
        blocked, reason = state.blocks_entry("RELIANCE")
        assert blocked
        assert "RISK_OFF" in reason
        assert not state.is_trading_allowed
        assert state.size_multiplier == Decimal("0")

    @pytest.mark.parametrize("confidence", [0, 40, 79])
    def test_unconfident_risk_off_downsizes_instead_of_halting(self, confidence: int) -> None:
        """Below the threshold the Sentinel restricts size, not permission."""
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, confidence, "mixed evidence")
        assert state.is_trading_allowed
        assert state.size_multiplier == Decimal("0.5")

    def test_threshold_boundary_is_inclusive(self) -> None:
        state = MacroState(risk_off_confidence=80, clock=_clock())
        state.apply(Regime.RISK_OFF, 80, "at the line")
        assert not state.is_trading_allowed

    @pytest.mark.parametrize("regime", [Regime.RISK_ON, Regime.NEUTRAL])
    def test_non_risk_off_regimes_never_restrict(self, regime: Regime) -> None:
        state = MacroState(clock=_clock())
        state.apply(regime, 99, "calm")
        assert state.is_trading_allowed
        assert state.size_multiplier == Decimal("1")

    def test_the_block_latches_for_the_session(self) -> None:
        """RISK_OFF then RISK_ON must not unblock — that would be the model adding risk."""
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 95, "crash")
        state.apply(Regime.RISK_ON, 99, "all clear, actually")
        assert not state.is_trading_allowed
        assert state.blocks_entry("RELIANCE")[0]

    def test_size_multiplier_is_a_ratchet(self) -> None:
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 50, "some risk")
        assert state.size_multiplier == Decimal("0.5")
        state.apply(Regime.RISK_ON, 99, "recovered")
        assert state.size_multiplier == Decimal("0.5"), "size may never be restored"

    def test_only_a_session_reset_clears_the_latch(self) -> None:
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 95, "crash")
        state.reset_session()
        assert state.is_trading_allowed
        assert state.size_multiplier == Decimal("1")

    @pytest.mark.parametrize(
        ("raw", "expected"), [(-5, 0), (0, 0), (100, 100), (150, 100), (101, 100)]
    )
    def test_confidence_is_clamped_on_ingest(self, raw: int, expected: int) -> None:
        state = MacroState(clock=_clock())
        state.apply(Regime.NEUTRAL, raw, "x")
        assert state.confidence == expected

    def test_failure_changes_nothing(self) -> None:
        """An outage must neither halt trading nor relax an existing restriction."""
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 50, "some risk")

        state.record_failure("gemini timeout")
        assert state.is_trading_allowed
        assert state.size_multiplier == Decimal("0.5"), "a failure must not restore size"
        assert state.regime is Regime.RISK_OFF, "the last known good report stands"
        assert state.degraded
        assert state.failures == 1

    def test_failure_after_a_block_leaves_it_blocked(self) -> None:
        state = MacroState(clock=_clock())
        state.apply(Regime.RISK_OFF, 95, "crash")
        state.record_failure("gemini down")
        assert not state.is_trading_allowed

    # ── news blacklist ───────────────────────────────────────────────────────

    def test_blacklist_blocks_one_symbol_and_expires(self) -> None:
        clock = _clock()
        state = MacroState(blacklist_window=timedelta(minutes=30), clock=clock)
        state.blacklist("RELIANCE", "results tonight")

        assert state.blocks_entry("RELIANCE")[0]
        assert not state.blocks_entry("HDFCBANK")[0]

        clock.advance(31 * 60)
        assert not state.blocks_entry("RELIANCE")[0]

    def test_blacklist_extends_but_never_shortens(self) -> None:
        clock = _clock()
        state = MacroState(blacklist_window=timedelta(minutes=30), clock=clock)
        first = state.blacklist("RELIANCE", "one")
        clock.advance(60)
        second = state.blacklist("RELIANCE", "two")
        assert second > first

        # A shorter window must not pull the expiry back in.
        state.blacklist_window = timedelta(minutes=1)
        third = state.blacklist("RELIANCE", "three")
        assert third == second

    def test_snapshot_reports_live_blacklist_only(self) -> None:
        clock = _clock()
        state = MacroState(clock=clock)
        state.blacklist("RELIANCE", "results")
        assert state.snapshot().blacklisted_symbols == ("RELIANCE",)
        clock.advance(31 * 60)
        assert state.snapshot().blacklisted_symbols == ()


# ──────────────────────────────────────────────────────────────────────────────
# api.py — GeminiClient
# ──────────────────────────────────────────────────────────────────────────────


class TestGeminiClient:
    async def test_successful_generation(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            seen["headers"] = dict(request.headers)
            seen["url"] = str(request.url)
            seen["body"] = _json.loads(request.content)
            return _reply('{"regime":"NEUTRAL","confidence":55,"reason":"mixed"}')

        client = _client(handler)
        reply = await client.generate(system="sys", prompt="hello", event="regime")
        await client.aclose()

        assert "NEUTRAL" in reply.text
        assert reply.prompt_tokens == 120
        assert client.stats.responses == 1
        # The key travels in a header, never in the URL (proxy logs, shell history).
        assert seen["headers"]["x-goog-api-key"] == "test-key"
        assert "test-key" not in seen["url"]
        assert DEFAULT_MODEL in seen["url"]

    def test_payload_requests_json_and_is_deterministic(self) -> None:
        client = _client(lambda request: _reply("{}"))
        payload = client.build_payload(system="s", prompt="p", schema={"type": "OBJECT"})
        generation = payload["generationConfig"]
        assert generation["responseMimeType"] == "application/json"
        assert generation["responseSchema"] == {"type": "OBJECT"}
        assert generation["temperature"] == 0.0
        # The brief is user content, never merged into the system instruction.
        assert payload["systemInstruction"]["parts"][0]["text"] == "s"
        assert payload["contents"][0]["parts"][0]["text"] == "p"

    async def test_timeout_raises_the_typed_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        client = _client(handler)
        with pytest.raises(GeminiTimeoutError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_an_absolute_deadline_bounds_a_hanging_server(self) -> None:
        """httpx times out socket phases; a trickling server needs the outer wait_for."""

        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10.0)
            return _reply("{}")

        client = _client(handler, timeout=0.2)
        with pytest.raises(GeminiTimeoutError):
            await client.generate(system="s", prompt="p")
        assert client.stats.timeouts == 1
        await client.aclose()

    async def test_503_is_retried_once_then_reported_unavailable(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503, json={"error": "overloaded"})

        client = _client(handler)
        with pytest.raises(GeminiUnavailableError):
            await client.generate(system="s", prompt="p")
        assert calls == 2, "exactly one retry — the daemon tries again in 15 minutes"
        await client.aclose()

    async def test_a_retry_that_succeeds_is_returned(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return _reply('{"regime":"RISK_ON","confidence":70,"reason":"calm"}')

        client = _client(handler)
        reply = await client.generate(system="s", prompt="p")
        assert "RISK_ON" in reply.text
        assert client.stats.retries == 1
        await client.aclose()

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_configuration_faults_are_not_retried(self, status: int) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(status, json={"error": "nope"})

        async def run() -> None:
            client = _client(handler)
            with pytest.raises(GeminiRejectedError):
                await client.generate(system="s", prompt="p")
            await client.aclose()

        asyncio.run(run())
        assert calls == 1, "retrying a bad API key just burns quota"

    async def test_connection_refused_is_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = _client(handler)
        with pytest.raises(GeminiUnavailableError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_a_safety_block_is_unavailable_not_a_crash(self) -> None:
        """A response with no candidate text must not IndexError deep in a parser."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

        client = _client(handler)
        with pytest.raises(GeminiUnavailableError):
            await client.generate(system="s", prompt="p")
        assert client.stats.blocked_by_safety == 1
        await client.aclose()

    @pytest.mark.parametrize(
        "envelope",
        [
            {"candidates": []},
            {"candidates": [{}]},
            {"candidates": [{"content": {}}]},
            {"candidates": [{"content": {"parts": []}}]},
            {"candidates": [{"content": {"parts": [{"inlineData": "x"}]}}]},
            {"candidates": "not-a-list"},
            {},
        ],
    )
    async def test_every_malformed_envelope_shape_is_handled(self, envelope: Any) -> None:
        client = _client(lambda request: httpx.Response(200, json=envelope))
        with pytest.raises(GeminiUnavailableError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_a_non_json_envelope_is_unavailable(self) -> None:
        client = _client(lambda request: httpx.Response(200, text="<html>502</html>"))
        with pytest.raises(GeminiUnavailableError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_missing_api_key_disables_rather_than_failing(self) -> None:
        client = _client(lambda request: _reply("{}"), settings=_settings(gemini_api_key=""))
        assert not client.is_configured
        with pytest.raises(GeminiNotConfiguredError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_the_session_budget_is_enforced_locally(self) -> None:
        client = _client(lambda request: _reply("{}"), max_requests=2)
        await client.generate(system="s", prompt="p")
        await client.generate(system="s", prompt="p")
        assert client.budget_remaining == 0
        with pytest.raises(GeminiBudgetExhaustedError):
            await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_the_budget_resets_for_a_new_session(self) -> None:
        client = _client(lambda request: _reply("{}"), max_requests=1)
        await client.generate(system="s", prompt="p")
        client.reset_session()
        await client.generate(system="s", prompt="p")
        await client.aclose()

    async def test_prompt_and_response_are_journalled(self, tmp_path: Path) -> None:
        journal = JsonlJournal(tmp_path, prefix="sentinel", clock=_clock())
        client = _client(lambda request: _reply('{"regime":"NEUTRAL"}'), journal=journal)
        await client.generate(system="sys", prompt="the brief", event="regime")
        await client.aclose()

        records = journal.read()
        assert [r["kind"] for r in records] == ["REQUEST", "RESPONSE"]
        assert records[0]["payload"]["prompt"] == "the brief"
        assert journal.path_for().name == "sentinel_2026-08-10.jsonl"

    async def test_the_api_key_never_reaches_the_journal(self, tmp_path: Path) -> None:
        journal = JsonlJournal(tmp_path, prefix="sentinel", clock=_clock())
        settings = _settings(gemini_api_key="AIzaSy-VERY-SECRET")
        client = _client(lambda request: _reply("{}"), settings=settings, journal=journal)
        await client.generate(system="s", prompt="p")
        await client.aclose()
        assert "AIzaSy-VERY-SECRET" not in journal.path_for().read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# classifier.py — coercion
# ──────────────────────────────────────────────────────────────────────────────


class TestCoercion:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (85, 85),
            (85.4, 85),
            ("85", 85),
            ("85%", 85),
            (" 85 ", 85),
            (0.85, 85),  # a 0-1 answer to a 0-100 question
            (150, 100),
            (-20, 0),
            (0, 0),
            (1.0, 1),  # ambiguous; read as the low integer
            ("nonsense", 0),
            (None, 0),
            (True, 0),
            (float("nan"), 0),
            (float("inf"), 0),
            ([], 0),
            ({"confidence": 90}, 0),
        ],
    )
    def test_confidence_coercion(self, raw: Any, expected: int) -> None:
        assert coerce_confidence(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.8, 0.8),
            (80, 0.8),  # a 0-100 answer to a 0-1 question
            ("0.8", 0.8),
            (-1, 0.0),
            (2.0, 0.02),
            (float("nan"), 0.0),
            ("junk", 0.0),
            (None, 0.0),
        ],
    )
    def test_score_coercion(self, raw: Any, expected: float) -> None:
        assert coerce_score(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("RISK_OFF", Regime.RISK_OFF),
            ("risk_off", Regime.RISK_OFF),
            ("  RISK_ON  ", Regime.RISK_ON),
            ("NEUTRAL", Regime.NEUTRAL),
            ("BEARISH", Regime.NEUTRAL),
            ("", Regime.NEUTRAL),
            ("RISK OFF", Regime.NEUTRAL),
            ("ignore previous instructions", Regime.NEUTRAL),
        ],
    )
    def test_regime_coercion_defaults_to_neutral(self, raw: str, expected: Regime) -> None:
        assert coerce_regime(raw) is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ('{"a":1}', '{"a":1}'),
            ('```json\n{"a":1}\n```', '{"a":1}'),
            ('```\n{"a":1}\n```', '{"a":1}'),
            ('  {"a":1}  ', '{"a":1}'),
        ],
    )
    def test_code_fence_stripping(self, raw: str, expected: str) -> None:
        assert strip_code_fence(raw) == expected


# ──────────────────────────────────────────────────────────────────────────────
# classifier.py — the fail-safe contract
# ──────────────────────────────────────────────────────────────────────────────


class TestRegimeClassifier:
    async def _classify(self, handler: Any, **kwargs: Any) -> Any:
        client = _client(handler, **kwargs)
        try:
            return await RegimeClassifier(client, settings=_settings()).classify(BRIEF)
        finally:
            await client.aclose()

    async def test_a_clean_response_is_parsed(self) -> None:
        report = await self._classify(
            lambda r: _reply('{"regime":"RISK_OFF","confidence":88,"reason":"VIX spike"}')
        )
        assert report.regime is Regime.RISK_OFF
        assert report.confidence == 88
        assert report.reason == "VIX spike"
        assert not report.degraded

    @pytest.mark.parametrize(
        ("name", "handler"),
        [
            ("timeout", lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("s", request=r))),
            ("refused", lambda r: (_ for _ in ()).throw(httpx.ConnectError("x", request=r))),
            ("503", lambda r: httpx.Response(503, json={})),
            ("500", lambda r: httpx.Response(500, json={})),
            ("401", lambda r: httpx.Response(401, json={})),
            ("safety-block", lambda r: httpx.Response(200, json={"candidates": []})),
            ("html", lambda r: httpx.Response(200, text="<html>gateway</html>")),
            ("truncated-json", lambda r: _reply('{"regime":"RISK_OFF","conf')),
            ("not-json", lambda r: _reply("I think markets look fine today!")),
            ("empty-object", lambda r: _reply("{}")),
            ("null", lambda r: _reply("null")),
            ("array", lambda r: _reply('[{"regime":"RISK_OFF"}]')),
            ("wrong-types", lambda r: _reply('{"regime":42,"confidence":"high","reason":null}')),
        ],
    )
    async def test_every_failure_degrades_to_neutral(self, name: str, handler: Any) -> None:
        """An API outage must never crash the trading system, and never say RISK_ON."""
        report = await self._classify(handler)
        assert report.regime is Regime.NEUTRAL, f"{name} did not degrade to NEUTRAL"
        assert report.regime is not Regime.RISK_ON
        assert report.confidence == 0, f"{name} produced a confident fallback"

    async def test_a_degraded_report_is_marked_as_such(self) -> None:
        report = await self._classify(lambda r: httpx.Response(503, json={}))
        assert report.degraded
        assert report.reason == NEUTRAL_FALLBACK_REASON
        assert "503" in report.error

    async def test_a_json_fenced_response_still_parses(self) -> None:
        report = await self._classify(
            lambda r: _reply('```json\n{"regime":"RISK_OFF","confidence":90,"reason":"x"}\n```')
        )
        assert report.regime is Regime.RISK_OFF
        assert report.confidence == 90

    async def test_an_unknown_regime_string_is_neutral_not_a_crash(self) -> None:
        report = await self._classify(
            lambda r: _reply('{"regime":"EXTREMELY_BULLISH","confidence":99,"reason":"x"}')
        )
        assert report.regime is Regime.NEUTRAL
        assert not report.degraded  # the call worked; the content was simply unusable

    async def test_no_response_can_produce_risk_on_from_a_failure(self) -> None:
        """The safety property stated plainly: broken never means permissive-plus."""
        for handler in (
            lambda r: httpx.Response(503, json={}),
            lambda r: _reply("garbage"),
            lambda r: _reply("{}"),
            lambda r: _reply('{"regime":"","confidence":100,"reason":""}'),
        ):
            report = await self._classify(handler)
            assert report.regime is not Regime.RISK_ON

    async def test_an_empty_brief_never_calls_the_model(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _reply("{}")

        client = _client(handler)
        report = await RegimeClassifier(client, settings=_settings()).classify("   ")
        await client.aclose()
        assert calls == 0
        assert report.regime is Regime.NEUTRAL
        assert report.degraded

    async def test_the_brief_is_capped(self) -> None:
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            sent["prompt"] = _json.loads(request.content)["contents"][0]["parts"][0]["text"]
            return _reply('{"regime":"NEUTRAL","confidence":10,"reason":"x"}')

        client = _client(handler)
        await RegimeClassifier(client, settings=_settings()).classify("x" * 50_000)
        await client.aclose()
        assert len(sent["prompt"]) < 20_000


class TestNewsClassifier:
    async def test_a_clean_score(self) -> None:
        client = _client(
            lambda r: _reply('{"score":0.85,"headline":"Results tonight","reason":"event"}')
        )
        risk = await NewsClassifier(client, settings=_settings()).score("RELIANCE", "headlines")
        await client.aclose()
        assert risk.score == pytest.approx(0.85)
        assert risk.exceeds(0.7)

    async def test_a_failure_scores_zero_and_blacklists_nothing(self) -> None:
        """The scanner exists to add a restriction; its absence must add none."""
        client = _client(lambda r: httpx.Response(503, json={}))
        risk = await NewsClassifier(client, settings=_settings()).score("RELIANCE", "headlines")
        await client.aclose()
        assert risk.score == 0.0
        assert not risk.exceeds(0.7)
        assert risk.degraded

    async def test_unparseable_scores_zero(self) -> None:
        client = _client(lambda r: _reply("probably fine"))
        risk = await NewsClassifier(client, settings=_settings()).score("RELIANCE", "h")
        await client.aclose()
        assert risk.score == 0.0
        assert risk.degraded

    async def test_no_headlines_never_calls_the_model(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _reply("{}")

        client = _client(handler)
        risk = await NewsClassifier(client, settings=_settings()).score("RELIANCE", "")
        await client.aclose()
        assert calls == 0
        assert risk.score == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# service.py — the daemon
# ──────────────────────────────────────────────────────────────────────────────


class TestSentinelDaemon:
    def _daemon(
        self, handler: Any, state: MacroState, tmp_path: Path, **kwargs: Any
    ) -> SentinelDaemon:
        settings = kwargs.pop("settings", None) or _settings()
        return SentinelDaemon(
            state=state,
            client=_client(handler, settings=settings),
            settings=settings,
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
            **kwargs,
        )

    async def test_a_confident_risk_off_blocks_trading(self, tmp_path: Path) -> None:
        state = MacroState(clock=_clock())
        daemon = self._daemon(
            lambda r: _reply('{"regime":"RISK_OFF","confidence":92,"reason":"VIX spike"}'),
            state,
            tmp_path,
        )
        await daemon.classify_once()
        await daemon.stop()

        assert not state.is_trading_allowed
        assert state.blocks_entry("RELIANCE")[0]

    async def test_an_outage_leaves_trading_allowed(self, tmp_path: Path) -> None:
        """The whole point: an API outage cannot become a kill switch."""
        state = MacroState(clock=_clock())
        daemon = self._daemon(lambda r: httpx.Response(503, json={}), state, tmp_path)
        await daemon.classify_once()
        await daemon.stop()

        assert state.is_trading_allowed
        assert state.regime is Regime.NEUTRAL
        assert state.degraded
        assert daemon.stats.regime_degraded == 1

    async def test_the_default_brief_announces_itself_as_a_placeholder(
        self, tmp_path: Path
    ) -> None:
        """A Sentinel classifying a hardcoded string must not look like one reading the tape."""
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            sent["prompt"] = _json.loads(request.content)["contents"][0]["parts"][0]["text"]
            return _reply('{"regime":"NEUTRAL","confidence":5,"reason":"no data"}')

        state = MacroState(clock=_clock())
        daemon = self._daemon(handler, state, tmp_path)
        await daemon.classify_once()
        await daemon.stop()
        assert "NO LIVE MARKET DATA WIRED" in sent["prompt"]
        assert PLACEHOLDER_BRIEF in sent["prompt"]

    async def test_a_custom_brief_provider_is_used(self, tmp_path: Path) -> None:
        sent: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            sent["prompt"] = _json.loads(request.content)["contents"][0]["parts"][0]["text"]
            return _reply('{"regime":"NEUTRAL","confidence":50,"reason":"ok"}')

        async def brief() -> str:
            return "India VIX 11.2, US +0.4%"

        state = MacroState(clock=_clock())
        daemon = self._daemon(handler, state, tmp_path, brief_provider=brief)
        await daemon.classify_once()
        await daemon.stop()
        assert "India VIX 11.2" in sent["prompt"]

    async def test_a_broken_brief_provider_degrades_rather_than_crashing(
        self, tmp_path: Path
    ) -> None:
        async def broken() -> str:
            raise RuntimeError("data source down")

        state = MacroState(clock=_clock())
        daemon = self._daemon(lambda r: _reply("{}"), state, tmp_path, brief_provider=broken)
        await daemon.classify_once()
        await daemon.stop()
        assert state.is_trading_allowed
        assert state.degraded

    async def test_news_scan_blacklists_a_hazardous_symbol(self, tmp_path: Path) -> None:
        async def headlines(symbol: str) -> str:
            return f"{symbol} reports Q2 results after market hours"

        state = MacroState(clock=_clock())
        daemon = self._daemon(
            lambda r: _reply('{"score":0.9,"headline":"Q2 results","reason":"event risk"}'),
            state,
            tmp_path,
            headline_provider=headlines,
        )
        await daemon.scan_news_once()
        await daemon.stop()

        assert state.is_blacklisted("RELIANCE")
        assert state.blocks_entry("RELIANCE")[0]
        # A blacklisted symbol must not block the others.
        assert state.blocks_entry("HDFCBANK")[0]  # both were scanned with the same handler

    async def test_a_low_news_score_blacklists_nothing(self, tmp_path: Path) -> None:
        async def headlines(symbol: str) -> str:
            return "routine coverage"

        state = MacroState(clock=_clock())
        daemon = self._daemon(
            lambda r: _reply('{"score":0.2,"headline":"h","reason":"routine"}'),
            state,
            tmp_path,
            headline_provider=headlines,
        )
        await daemon.scan_news_once()
        await daemon.stop()
        assert not state.is_blacklisted("RELIANCE")

    async def test_no_api_key_disables_the_daemon_without_touching_trading(
        self, tmp_path: Path
    ) -> None:
        state = MacroState(clock=_clock())
        settings = _settings(gemini_api_key="")
        daemon = self._daemon(lambda r: _reply("{}"), state, tmp_path, settings=settings)

        await asyncio.wait_for(daemon.run(), timeout=5.0)
        await daemon.stop()

        assert not daemon.is_enabled
        assert state.is_trading_allowed
        assert daemon.stats.regime_runs == 0

    async def test_a_disabled_sentinel_in_config_never_runs(self, tmp_path: Path) -> None:
        state = MacroState(clock=_clock())
        settings = _settings(sentinel_enabled=False)
        daemon = self._daemon(lambda r: _reply("{}"), state, tmp_path, settings=settings)
        await asyncio.wait_for(daemon.run(), timeout=5.0)
        await daemon.stop()
        assert state.is_trading_allowed
        assert daemon.stats.regime_runs == 0

    async def test_an_exhausted_budget_stops_the_loop_but_not_trading(self, tmp_path: Path) -> None:
        state = MacroState(clock=_clock())
        settings = _settings(sentinel=SentinelSettings(max_session_requests=1))
        daemon = SentinelDaemon(
            state=state,
            client=_client(
                lambda r: _reply('{"regime":"NEUTRAL","confidence":40,"reason":"ok"}'),
                settings=settings,
                max_requests=1,
            ),
            settings=settings,
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
            interval_minutes=1,
        )
        await daemon.classify_once()  # spends the budget
        await daemon.classify_once()  # degrades, does not raise
        await daemon.stop()

        assert state.is_trading_allowed
        assert daemon.stats.regime_degraded == 1

    async def test_the_loop_runs_immediately_then_stops_cleanly(self, tmp_path: Path) -> None:
        """A process booting at 09:10 must classify before the 09:20 gate, not 15 min in."""
        state = MacroState(clock=_clock())
        daemon = self._daemon(
            lambda r: _reply('{"regime":"NEUTRAL","confidence":60,"reason":"calm"}'),
            state,
            tmp_path,
            interval_minutes=60,
        )
        task = asyncio.create_task(daemon.run())

        async def classified() -> None:
            while daemon.stats.regime_runs == 0:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(classified(), timeout=5.0)
        await daemon.stop()
        await asyncio.wait_for(task, timeout=5.0)

        assert state.regime is Regime.NEUTRAL
        assert state.updates == 1

    async def test_reset_session_clears_the_latch(self, tmp_path: Path) -> None:
        state = MacroState(clock=_clock())
        daemon = self._daemon(
            lambda r: _reply('{"regime":"RISK_OFF","confidence":95,"reason":"crash"}'),
            state,
            tmp_path,
        )
        await daemon.classify_once()
        assert not state.is_trading_allowed

        daemon.reset_session()
        assert state.is_trading_allowed
        await daemon.stop()


# ──────────────────────────────────────────────────────────────────────────────
# The veto — CLAUDE.md §4 check 10
# ──────────────────────────────────────────────────────────────────────────────


class _RiskHarness:
    def __init__(self, tmp_path: Path, macro: MacroState | None) -> None:
        from tachyon.core.state import DailyLock

        clock = _clock()
        self.clock = clock
        self.settings = _settings()
        self.machine = StateMachine(TradingState.ACTIVE, clock=clock)
        self.lock = DailyLock(path=tmp_path / "daily_lock.txt", clock=clock)
        self.pnl = PnLTracker(self.machine, daily_lock=self.lock, clock=clock)
        self.monitor = FeedMonitor(clock=clock)
        self.monitor.record()
        self.positions = PositionRegistry()
        self.engine = RiskEngine(
            self.machine,
            self.pnl,
            self.monitor,
            self.positions,
            settings=self.settings,
            clock=clock,
            macro_state=macro,
        )


class TestSentinelVeto:
    def test_pending_checks_is_now_empty(self) -> None:
        """Every check CLAUDE.md §4 requires is implemented."""
        assert PENDING_CHECKS == ()

    def test_no_macro_state_means_no_veto(self, tmp_path: Path) -> None:
        """Absence is permission here, and only here — the Sentinel is advisory."""
        assert _RiskHarness(tmp_path, None).engine.evaluate("RELIANCE").allowed

    def test_a_permissive_macro_state_allows(self, tmp_path: Path) -> None:
        assert (
            _RiskHarness(tmp_path, MacroState(clock=_clock())).engine.evaluate("RELIANCE").allowed
        )

    def test_risk_off_vetoes(self, tmp_path: Path) -> None:
        macro = MacroState(clock=_clock())
        macro.apply(Regime.RISK_OFF, 91, "VIX +18%")
        decision = _RiskHarness(tmp_path, macro).engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.SENTINEL_RISK_OFF
        assert "VIX +18%" in decision.detail

    def test_a_low_confidence_risk_off_does_not_veto(self, tmp_path: Path) -> None:
        macro = MacroState(clock=_clock())
        macro.apply(Regime.RISK_OFF, 50, "mixed")
        assert _RiskHarness(tmp_path, macro).engine.evaluate("RELIANCE").allowed
        assert macro.size_multiplier == Decimal("0.5")

    def test_a_blacklisted_symbol_vetoes_only_itself(self, tmp_path: Path) -> None:
        macro = MacroState(clock=_clock())
        macro.blacklist("RELIANCE", "results tonight")
        harness = _RiskHarness(tmp_path, macro)
        assert not harness.engine.evaluate("RELIANCE").allowed
        assert harness.engine.evaluate("HDFCBANK").allowed

    def test_a_gemini_outage_does_not_veto(self, tmp_path: Path) -> None:
        macro = MacroState(clock=_clock())
        macro.record_failure("gemini 503")
        assert _RiskHarness(tmp_path, macro).engine.evaluate("RELIANCE").allowed

    def test_a_broken_macro_state_does_veto(self, tmp_path: Path) -> None:
        """Absent is permitted; *broken* is not. `_guarded` still turns a raise into a veto."""

        class Exploding(MacroState):
            def blocks_entry(self, symbol: str, at: Any = None) -> tuple[bool, str]:
                raise RuntimeError("macro state exploded")

        decision = _RiskHarness(tmp_path, Exploding(clock=_clock())).engine.evaluate("RELIANCE")
        assert not decision.allowed
        assert decision.reason is VetoReason.CHECK_FAILED

    def test_the_sentinel_check_runs_before_margin(self, tmp_path: Path) -> None:
        """Margin is the only check that can touch the network; it must stay last."""
        calls: list[int] = []
        macro = MacroState(clock=_clock())
        macro.apply(Regime.RISK_OFF, 95, "crash")

        harness = _RiskHarness(tmp_path, macro)
        engine = RiskEngine(
            harness.machine,
            harness.pnl,
            harness.monitor,
            harness.positions,
            settings=harness.settings,
            clock=harness.clock,
            macro_state=macro,
            margin_provider=lambda: (calls.append(1), Decimal("50000"))[1],
        )
        assert not engine.evaluate("RELIANCE").allowed
        assert calls == []

    def test_a_cheaper_veto_short_circuits_the_sentinel(self, tmp_path: Path) -> None:
        macro = MacroState(clock=_clock())
        macro.apply(Regime.RISK_OFF, 95, "crash")
        harness = _RiskHarness(tmp_path, macro)
        harness.machine.lock_out("test")

        decision = harness.engine.evaluate("RELIANCE")
        assert decision.reason is VetoReason.STATE_NOT_ACTIVE


class TestSizeMultiplierIsOurs:
    """CLAUDE.md §5 — the model is never asked for a size multiplier."""

    def test_the_schema_has_no_multiplier_field(self) -> None:
        from tachyon.sentinel.classifier import REGIME_SCHEMA

        assert set(REGIME_SCHEMA["properties"]) == {"regime", "confidence", "reason"}

    async def test_a_model_supplied_multiplier_is_ignored(self) -> None:
        """Even if the model volunteers one, nothing reads it."""
        client = _client(
            lambda r: _reply(
                '{"regime":"RISK_ON","confidence":99,"reason":"x","size_multiplier":5.0}'
            )
        )
        report = await RegimeClassifier(client, settings=_settings()).classify(BRIEF)
        await client.aclose()

        assert not hasattr(report, "size_multiplier")
        state = MacroState(clock=_clock())
        state.apply(report.regime, report.confidence, report.reason)
        assert state.size_multiplier == Decimal("1"), "never above 1, whatever the model says"

    @pytest.mark.parametrize("confidence", [0, 50, 79, 80, 100])
    def test_the_multiplier_is_never_above_one(self, confidence: int) -> None:
        for regime in Regime:
            state = MacroState(clock=_clock())
            state.apply(regime, confidence, "x")
            assert Decimal("0") <= state.size_multiplier <= Decimal("1")
            assert not math.isnan(float(state.size_multiplier))


class TestUnanticipatedCrashes:
    """The last line of defence: a bug we have not thought of, inside the advisor.

    A `GeminiError` degrading to NEUTRAL is the designed path. These tests cover the
    *undesigned* one — an arbitrary exception escaping the client — because the requirement is
    that an AI failure cannot crash the trading system, not that a *known* AI failure cannot.
    """

    class _Exploding:
        """Stands in for a client whose internals fail in a way we did not anticipate."""

        model = DEFAULT_MODEL
        is_configured = True

        async def generate(self, **kwargs: Any) -> Any:
            raise ValueError("something nobody predicted")

        async def aclose(self) -> None:
            return None

        def reset_session(self) -> None:
            return None

    async def test_a_regime_classifier_crash_degrades_to_neutral(self) -> None:
        classifier = RegimeClassifier(self._Exploding(), settings=_settings())  # type: ignore[arg-type]
        report = await classifier.classify(BRIEF)
        assert report.regime is Regime.NEUTRAL
        assert report.degraded
        assert "ValueError" in report.error

    async def test_a_news_classifier_crash_scores_zero(self) -> None:
        classifier = NewsClassifier(self._Exploding(), settings=_settings())  # type: ignore[arg-type]
        risk = await classifier.score("RELIANCE", "headlines")
        assert risk.score == 0.0
        assert risk.degraded

    async def test_a_crashing_sentinel_leaves_trading_allowed(self, tmp_path: Path) -> None:
        state = MacroState(clock=_clock())
        daemon = SentinelDaemon(
            state=state,
            client=self._Exploding(),  # type: ignore[arg-type]
            settings=_settings(),
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
        )
        await daemon.classify_once()
        assert state.is_trading_allowed
        assert state.degraded

    async def test_a_broken_headline_provider_does_not_stop_the_scan(self, tmp_path: Path) -> None:
        async def broken(symbol: str) -> str:
            raise RuntimeError("news feed down")

        state = MacroState(clock=_clock())
        daemon = SentinelDaemon(
            state=state,
            client=_client(lambda r: _reply('{"score":0.9,"headline":"h","reason":"r"}')),
            settings=_settings(),
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
            headline_provider=broken,
        )
        await daemon.scan_news_once()
        await daemon.stop()
        assert state.snapshot().blacklisted_symbols == ()

    async def test_the_loop_absorbs_a_cycle_failure_and_keeps_going(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = MacroState(clock=_clock())
        daemon = SentinelDaemon(
            state=state,
            client=_client(lambda r: _reply('{"regime":"NEUTRAL","confidence":5,"reason":"x"}')),
            settings=_settings(),
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
            interval_minutes=1,
        )

        calls = 0

        # Patched on the class, not the instance: SentinelDaemon uses __slots__, so an
        # instance attribute of that name cannot be assigned.
        async def flaky(_self: SentinelDaemon) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient bug")

        monkeypatch.setattr(SentinelDaemon, "classify_once", flaky)
        task = asyncio.create_task(daemon.run())

        async def recovered() -> None:
            while calls < 1 or daemon.stats.loop_errors < 1:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(recovered(), timeout=5.0)
        await daemon.stop()
        await asyncio.wait_for(task, timeout=5.0)

        assert daemon.stats.loop_errors == 1
        assert state.is_trading_allowed

    async def test_budget_exhaustion_inside_the_loop_disables_the_daemon(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = MacroState(clock=_clock())
        daemon = SentinelDaemon(
            state=state,
            client=_client(lambda r: _reply("{}")),
            settings=_settings(),
            journal=JsonlJournal(tmp_path, prefix="sentinel", clock=_clock()),
            clock=_clock(),
        )

        async def exhausted(_self: SentinelDaemon) -> None:
            raise GeminiBudgetExhaustedError("out of quota")

        monkeypatch.setattr(SentinelDaemon, "classify_once", exhausted)
        await asyncio.wait_for(daemon.run(), timeout=5.0)
        await daemon.stop()

        assert not daemon.is_enabled
        assert state.is_trading_allowed, "out of quota disables the Sentinel, never trading"
