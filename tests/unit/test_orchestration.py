"""Phase 11 hardening tests — CLAUDE.md §2, §6.4, §9.

The centrepiece is the PAPER guarantee, asserted at the transport rather than at the caller:
**no mutating request reaches the network while ``TRADING_MODE`` is not LIVE.** A test that only
checked the return value would still pass if the order had gone out and a simulated reply were
layered on top, so every one of these counts actual transport invocations.

The rest covers the two things that make a paper session trustworthy: the order-book sweep that
survives a dropped webhook, and a shutdown that leaves nothing running and no order in an
unknown state.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.config import Settings, WatchlistItem
from tachyon.core.constants import TradingMode
from tachyon.execution.api import (
    PAPER_INTERCEPTED,
    PAPER_ORDER_PREFIX,
    ClientIdentity,
    PaperBroker,
    PaperModeError,
    SmartApiClient,
)
from tachyon.execution.reconciliation import (
    DEFAULT_POLL_SECONDS,
    OrderBookPoller,
)
from tachyon.main import IngestorSupervisor, Orchestrator, confirm_live
from tachyon.persistence.journal import JsonlJournal
from tachyon.ui.postback import OrderStatusListener, watchlist_resolver


def _clock(hh: int = 11, mm: int = 0, mono: float = 1000.0) -> ManualClock:
    return ManualClock(wall=datetime(2026, 8, 10, hh, mm, tzinfo=IST), mono=mono)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "watchlist": (WatchlistItem(symbol="RELIANCE", token="2885", exchange="NSE"),),
    }
    base.update(overrides)
    return Settings(**base)


class _CountingTransport(httpx.AsyncBaseTransport):
    """Records every request that reaches the wire.

    The point of the whole file: a mutating call in PAPER must never arrive here. Counting at
    the transport, not at the client, is what makes that assertion mean something.
    """

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._response = response

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._response is not None:
            return self._response
        return httpx.Response(
            200, json={"status": True, "message": "SUCCESS", "errorcode": "", "data": {}}
        )

    @property
    def paths(self) -> list[str]:
        return [str(request.url.path) for request in self.requests]


def _client(
    transport: _CountingTransport,
    *,
    mode: TradingMode = TradingMode.PAPER,
    journal: JsonlJournal | None = None,
    simulate_paper: bool = True,
) -> SmartApiClient:
    return SmartApiClient(
        settings=_settings(smartapi_api_key="k", smartapi_client_code="c"),
        journal=journal,
        client=httpx.AsyncClient(base_url="https://test", transport=transport),
        mode=mode,
        clock=_clock(),
        identity=ClientIdentity(local_ip="1.2.3.4", public_ip="1.2.3.4", mac_address="AA:BB"),
        rate_limit_scale=10_000.0,
        simulate_paper_orders=simulate_paper,
    )


def _payload(tag: str = "TCHYN-20260810-0001") -> dict[str, Any]:
    return {
        "variety": "ROBO",
        "tradingsymbol": "RELIANCE-EQ",
        "symboltoken": "2885",
        "transactiontype": "BUY",
        "exchange": "NSE",
        "ordertype": "LIMIT",
        "producttype": "BO",
        "duration": "DAY",
        "price": "2500.00",
        "squareoff": "18.90",
        "stoploss": "12.60",
        "trailingStopLoss": "6.30",
        "quantity": "4",
        "ordertag": tag,
    }


# ──────────────────────────────────────────────────────────────────────────────
# The PAPER guarantee
# ──────────────────────────────────────────────────────────────────────────────


class TestPaperNeverTransmits:
    async def test_place_order_fires_no_network_request(self) -> None:
        """The assertion this phase exists for."""
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER)

        result = await client.place_order(_payload())

        assert transport.requests == [], "PAPER transmitted a placement"
        assert result["orderid"].startswith(PAPER_ORDER_PREFIX)
        assert result["simulated"] is True
        await client.aclose()

    async def test_modify_and_cancel_fire_no_network_request(self) -> None:
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER)

        placed = await client.place_order(_payload())
        await client.modify_order({"orderid": placed["orderid"], "price": "2501.00"})
        await client.cancel_order(placed["orderid"])

        assert transport.requests == []
        assert client.stats.paper_intercepts == 3
        await client.aclose()

    @pytest.mark.parametrize("endpoint", sorted(PAPER_INTERCEPTED))
    def test_every_intercepted_endpoint_is_a_mutating_one(self, endpoint: str) -> None:
        """Login and refresh are non-idempotent too, and must NOT be intercepted."""
        assert endpoint in {"place_order", "modify_order", "cancel_order"}

    async def test_read_only_endpoints_still_hit_the_network(self) -> None:
        """A paper session that invents its own order book is fiction, not a rehearsal."""
        transport = _CountingTransport(
            httpx.Response(200, json={"status": True, "message": "OK", "data": []})
        )
        client = _client(transport, mode=TradingMode.PAPER)

        await client.order_book()
        await client.positions()
        await client.trade_book()

        assert len(transport.requests) == 3
        assert all("get" in path.lower() for path in transport.paths)
        await client.aclose()

    async def test_margin_still_hits_the_network_in_paper(self) -> None:
        transport = _CountingTransport(
            httpx.Response(
                200, json={"status": True, "message": "OK", "data": {"availablecash": "25000"}}
            )
        )
        client = _client(transport, mode=TradingMode.PAPER)
        assert await client.available_margin() == Decimal("25000")
        assert len(transport.requests) == 1
        await client.aclose()

    async def test_login_still_hits_the_network_in_paper(self) -> None:
        """A paper session needs a real feed token, or its market data is fiction too."""
        transport = _CountingTransport(
            httpx.Response(
                200,
                json={
                    "status": True,
                    "message": "SUCCESS",
                    "data": {"jwtToken": "j", "refreshToken": "r", "feedToken": "f"},
                },
            )
        )
        settings = _settings(
            smartapi_client_code="C1",
            smartapi_password="1234",
            smartapi_totp_secret="JBSWY3DPEHPK3PXP",
        )
        client = SmartApiClient(
            settings=settings,
            client=httpx.AsyncClient(base_url="https://test", transport=transport),
            mode=TradingMode.PAPER,
            clock=_clock(),
            identity=ClientIdentity(local_ip="1", public_ip="1", mac_address="A"),
            rate_limit_scale=10_000.0,
        )
        await client.login()
        assert len(transport.requests) == 1
        assert client.feed_token == "f"
        await client.aclose()

    async def test_live_mode_does_transmit(self) -> None:
        """The interception is conditional, not a permanent muzzle."""
        transport = _CountingTransport(
            httpx.Response(
                200, json={"status": True, "message": "OK", "data": {"orderid": "251008000001"}}
            )
        )
        client = _client(transport, mode=TradingMode.LIVE)

        result = await client.place_order(_payload())

        assert len(transport.requests) == 1
        assert result["orderid"] == "251008000001"
        assert "simulated" not in result
        assert client.stats.paper_intercepts == 0
        await client.aclose()

    async def test_interception_costs_no_rate_limit_quota(self) -> None:
        """It runs before the throttle: a paper burst must not spend the real budget."""
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER)
        for i in range(50):
            await client.place_order(_payload(f"TCHYN-20260810-{i:04d}"))
        assert transport.requests == []
        assert client.stats.requests == 0, "no request counter moved, so no token was spent"
        await client.aclose()

    async def test_strict_mode_refuses_instead_of_simulating(self) -> None:
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER, simulate_paper=False)
        with pytest.raises(PaperModeError):
            await client.place_order(_payload())
        assert transport.requests == []
        await client.aclose()

    async def test_the_interception_is_journalled(self, tmp_path: Path) -> None:
        journal = JsonlJournal(tmp_path, prefix="orders", clock=_clock())
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER, journal=journal)
        await client.place_order(_payload())
        await client.aclose()

        records = journal.read()
        assert any(r["event"] == "paper_place_order" for r in records)
        entry = next(r for r in records if r["event"] == "paper_place_order")
        assert "nothing was transmitted" in entry["note"]

    async def test_simulated_ids_are_unmistakable(self) -> None:
        """Angel One order ids are numeric; a PAPER- prefix can never collide with one."""
        transport = _CountingTransport()
        client = _client(transport, mode=TradingMode.PAPER)
        ids = {(await client.place_order(_payload(f"T{i}")))["orderid"] for i in range(5)}
        assert len(ids) == 5
        assert all(order_id.lstrip(PAPER_ORDER_PREFIX).isdigit() is not False for order_id in ids)
        assert all(order_id.startswith(PAPER_ORDER_PREFIX) for order_id in ids)
        await client.aclose()


class TestPaperBroker:
    def test_place_then_cancel_forgets_the_order(self) -> None:
        from tachyon.execution.api import CANCEL_ORDER, PLACE_ORDER

        broker = PaperBroker()
        placed = broker.simulate(PLACE_ORDER, _payload())
        assert placed["orderid"] in broker.orders

        broker.simulate(CANCEL_ORDER, {"orderid": placed["orderid"]})
        assert placed["orderid"] not in broker.orders

    def test_modify_merges_into_the_recorded_order(self) -> None:
        from tachyon.execution.api import MODIFY_ORDER, PLACE_ORDER

        broker = PaperBroker()
        placed = broker.simulate(PLACE_ORDER, _payload())
        broker.simulate(MODIFY_ORDER, {"orderid": placed["orderid"], "price": "2499.00"})
        assert broker.orders[placed["orderid"]]["price"] == "2499.00"

    def test_reset_session_clears_everything(self) -> None:
        from tachyon.execution.api import PLACE_ORDER

        broker = PaperBroker()
        broker.simulate(PLACE_ORDER, _payload())
        broker.reset_session()
        assert broker.orders == {}
        assert broker.issued == 0


# ──────────────────────────────────────────────────────────────────────────────
# OrderBookPoller
# ──────────────────────────────────────────────────────────────────────────────


class _FakeBroker:
    """Serves canned order-book responses, and can fail on demand."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.calls = 0
        self.fail_with: Exception | None = None

    async def order_book(self) -> tuple[dict[str, Any], ...]:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return tuple(self.rows)


def _row(order_id: str, side: str, price: str, *, order_type: str = "LIMIT") -> dict[str, Any]:
    return {
        "orderid": order_id,
        "symboltoken": "2885",
        "tradingsymbol": "RELIANCE-EQ",
        "transactiontype": side,
        "orderstatus": "complete",
        "quantity": 10,
        "filledshares": 10,
        "averageprice": price,
        "ordertype": order_type,
        # Every order this system places carries a TCHYN- tag (CLAUDE.md §6.4). Without it the
        # listener treats the fill as rogue and locks the day, which is tested separately.
        "ordertag": "TCHYN-20260810-0001",
    }


def _listener(tmp_path: Path) -> tuple[OrderStatusListener, list[tuple[Any, ...]]]:
    closures: list[tuple[Any, ...]] = []
    listener = OrderStatusListener(
        on_closed=lambda *args: closures.append(args),
        symbol_resolver=watchlist_resolver({"2885": "RELIANCE"}, {"RELIANCE-EQ": "RELIANCE"}),
        journal=JsonlJournal(tmp_path, prefix="fills", clock=_clock()),
        clock=_clock(),
    )
    return listener, closures


class TestOrderBookPoller:
    async def test_a_sweep_books_fills_the_webhook_missed(self, tmp_path: Path) -> None:
        """The reason this exists: a dropped POST is invisible to the receiver."""
        listener, closures = _listener(tmp_path)
        broker = _FakeBroker([_row("A1", "BUY", "2500.00"), _row("A2", "SELL", "2510.00")])
        poller = OrderBookPoller(client=broker, listener=listener)  # type: ignore[arg-type]

        assert await poller.poll_once() == 2
        assert len(closures) == 1
        assert closures[0][1] == Decimal("100.00")

    async def test_repeated_sweeps_do_not_double_book(self, tmp_path: Path) -> None:
        """Idempotency is what makes re-offering the whole book every 3 s free."""
        listener, closures = _listener(tmp_path)
        broker = _FakeBroker([_row("B1", "BUY", "2500.00"), _row("B2", "SELL", "2510.00")])
        poller = OrderBookPoller(client=broker, listener=listener)  # type: ignore[arg-type]

        for _ in range(10):
            await poller.poll_once()

        assert broker.calls == 10
        assert len(closures) == 1
        assert poller.stats.updates_accepted == 2

    async def test_a_broker_outage_never_raises(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker()
        broker.fail_with = httpx.ConnectError("no route to host")
        poller = OrderBookPoller(client=broker, listener=listener)  # type: ignore[arg-type]

        assert await poller.poll_once() == 0
        assert poller.stats.failures == 1
        assert poller.stats.consecutive_failures == 1

    async def test_repeated_failure_backs_off(self, tmp_path: Path) -> None:
        """A broker that is down does not recover faster for being asked every 3 seconds."""
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker()
        broker.fail_with = RuntimeError("down")
        poller = OrderBookPoller(client=broker, listener=listener, interval_seconds=3.0)  # type: ignore[arg-type]

        delays = []
        for _ in range(8):
            await poller.poll_once()
            delays.append(poller._delay())  # noqa: SLF001

        assert delays[0] == 3.0, "no backoff while the failure could be a blip"
        assert delays[-1] > delays[0]
        assert delays[-1] <= 30.0, "and it is capped"

    async def test_recovery_resets_the_backoff(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker([_row("C1", "BUY", "2500.00")])
        broker.fail_with = RuntimeError("down")
        poller = OrderBookPoller(client=broker, listener=listener)  # type: ignore[arg-type]

        for _ in range(5):
            await poller.poll_once()
        broker.fail_with = None
        await poller.poll_once()

        assert poller.stats.consecutive_failures == 0
        assert poller._delay() == poller.interval_seconds  # noqa: SLF001

    async def test_the_degraded_callback_fires_once(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker()
        broker.fail_with = RuntimeError("down")
        alerts: list[str] = []
        poller = OrderBookPoller(
            client=broker,  # type: ignore[arg-type]
            listener=listener,
            on_degraded=alerts.append,
        )

        for _ in range(6):
            await poller.poll_once()
        assert len(alerts) == 1, "one alert per outage, not one per failed poll"

    async def test_a_crashing_alert_callback_does_not_stop_the_poller(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker()
        broker.fail_with = RuntimeError("down")

        def explode(_detail: str) -> None:
            raise ValueError("alerting is broken too")

        poller = OrderBookPoller(
            client=broker,  # type: ignore[arg-type]
            listener=listener,
            on_degraded=explode,
        )
        for _ in range(6):
            await poller.poll_once()
        assert poller.stats.failures == 6

    async def test_no_client_disables_the_poller(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        poller = OrderBookPoller(client=None, listener=listener)
        assert not poller.is_enabled
        assert await poller.poll_once() == 0
        await asyncio.wait_for(poller.run(), timeout=5.0)  # returns immediately

    async def test_the_loop_stops_cleanly(self, tmp_path: Path) -> None:
        listener, _closures = _listener(tmp_path)
        broker = _FakeBroker([_row("D1", "BUY", "2500.00")])
        poller = OrderBookPoller(client=broker, listener=listener, interval_seconds=1.0)  # type: ignore[arg-type]

        task = asyncio.create_task(poller.run())

        async def swept() -> None:
            while poller.stats.polls == 0:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(swept(), timeout=5.0)
        await poller.stop()
        await asyncio.wait_for(task, timeout=5.0)
        assert poller.stats.polls >= 1

    def test_the_default_cadence_respects_the_rate_limit(self) -> None:
        """getOrderBook is ~1 req/s; 3 s leaves room for reconciliation and square-off."""
        assert DEFAULT_POLL_SECONDS >= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# The orchestrator
# ──────────────────────────────────────────────────────────────────────────────


class _FakeBrain:
    """Stands in for StrategyBrain. Records the shutdown ordering."""

    def __init__(self, *, tradeable: bool = True, run_forever: bool = True) -> None:
        self.tradeable = tradeable
        self.run_forever = run_forever
        self.events: list[str] = []
        self.state = "ACTIVE"
        self.stats = type("S", (), {"ticks": 0, "entries_placed": 0})()
        self.pnl = type("P", (), {"total": Decimal("0")})()
        self._stop = asyncio.Event()

    async def boot(self) -> bool:
        self.events.append("boot")
        return self.tradeable

    async def run(self) -> None:
        self.events.append("run")
        if self.run_forever:
            await self._stop.wait()

    async def stop(self) -> None:
        self.events.append("stop")
        self._stop.set()

    async def shutdown(self) -> None:
        self.events.append("shutdown")
        self._stop.set()


class TestOrchestrator:
    def _build(self, **kwargs: Any) -> tuple[Orchestrator, _FakeBrain, IngestorSupervisor]:
        brain = _FakeBrain(**kwargs)
        supervisor = IngestorSupervisor(enabled=False)
        orchestrator = Orchestrator(
            settings=_settings(),
            client=None,
            supervisor=supervisor,
            brain=brain,  # type: ignore[arg-type]
        )
        return orchestrator, brain, supervisor

    async def test_a_clean_run_boots_then_runs_then_tears_down(self) -> None:
        orchestrator, brain, _supervisor = self._build(run_forever=False)
        assert await orchestrator.run() == 0
        assert brain.events == ["boot", "run", "stop", "shutdown"]

    async def test_a_locked_session_still_runs_and_reports_it(self) -> None:
        """Read-only still needs telemetry and the 15:15 watchdog armed."""
        orchestrator, brain, _supervisor = self._build(tradeable=False, run_forever=False)
        assert await orchestrator.run() == 3
        assert "run" in brain.events

    async def test_shutdown_stops_the_brain_before_the_ingestor(self) -> None:
        orchestrator, brain, _supervisor = self._build()
        task = asyncio.create_task(orchestrator.run())

        async def running() -> None:
            while "run" not in brain.events:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(running(), timeout=5.0)
        await orchestrator.request_shutdown("test")
        assert await asyncio.wait_for(task, timeout=10.0) == 0

        # The Brain is asked to stop first; its own shutdown awaits in-flight entries.
        assert brain.events.index("stop") < brain.events.index("shutdown")

    async def test_a_second_shutdown_request_is_ignored(self) -> None:
        """Ctrl-C twice, half-way through flattening, is the dangerous moment."""
        orchestrator, brain, _supervisor = self._build()
        task = asyncio.create_task(orchestrator.run())

        async def running() -> None:
            while "run" not in brain.events:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(running(), timeout=5.0)
        await orchestrator.request_shutdown("first")
        await orchestrator.request_shutdown("second")
        await orchestrator.request_shutdown("third")
        await asyncio.wait_for(task, timeout=10.0)

        assert brain.events.count("stop") == 1

    async def test_a_hanging_brain_does_not_hang_shutdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wedged shutdown must not strand the ingestor subprocess forever."""
        import tachyon.main as main_module

        monkeypatch.setattr(main_module, "BRAIN_SHUTDOWN_SECONDS", 0.2)

        class Wedged(_FakeBrain):
            async def shutdown(self) -> None:
                self.events.append("shutdown")
                await asyncio.sleep(30)

        brain = Wedged(run_forever=False)
        orchestrator = Orchestrator(
            settings=_settings(),
            client=None,
            supervisor=IngestorSupervisor(enabled=False),
            brain=brain,  # type: ignore[arg-type]
        )
        await asyncio.wait_for(orchestrator.run(), timeout=10.0)
        assert "shutdown" in brain.events


class TestIngestorSupervisor:
    async def test_a_disabled_supervisor_spawns_nothing(self) -> None:
        supervisor = IngestorSupervisor(enabled=False)
        await asyncio.wait_for(supervisor.run(), timeout=5.0)
        assert supervisor.stats.ingestor_starts == 0
        assert not supervisor.is_running

    async def test_a_missing_script_is_reported_not_raised(self, tmp_path: Path) -> None:
        supervisor = IngestorSupervisor(script=tmp_path / "nope.py")
        await asyncio.wait_for(supervisor.run(), timeout=5.0)
        assert supervisor.stats.ingestor_starts == 0

    async def test_a_child_that_exits_is_restarted_up_to_the_budget(self, tmp_path: Path) -> None:
        """Bounded, because a crash loop against the broker is how a key gets throttled."""
        script = tmp_path / "dies.py"
        script.write_text("raise SystemExit(7)\n", encoding="utf-8")
        supervisor = IngestorSupervisor(script=script, max_restarts=2)

        await asyncio.wait_for(supervisor.run(), timeout=60.0)

        assert supervisor.stats.gave_up_on_ingestor
        assert supervisor.stats.ingestor_starts == 3  # initial + 2 restarts
        assert supervisor.stats.ingestor_exit_code == 7

    async def test_stop_terminates_a_live_child(self, tmp_path: Path) -> None:
        script = tmp_path / "sleeps.py"
        script.write_text("import time\nwhile True:\n    time.sleep(0.2)\n", encoding="utf-8")
        supervisor = IngestorSupervisor(script=script)
        task = asyncio.create_task(supervisor.run())

        async def started() -> None:
            while not supervisor.is_running:
                await asyncio.sleep(0.02)

        await asyncio.wait_for(started(), timeout=15.0)
        await supervisor.stop()
        await asyncio.wait_for(task, timeout=15.0)
        assert not supervisor.is_running


class TestLiveConfirmation:
    def test_paper_needs_no_confirmation(self) -> None:
        assert confirm_live(_settings(trading_mode=TradingMode.PAPER))

    def test_live_without_a_terminal_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No tty means no confirmation, and no confirmation means no live trading."""
        import builtins

        def no_input(_prompt: str = "") -> str:
            raise EOFError

        monkeypatch.setattr(builtins, "input", no_input)
        settings = _settings(
            trading_mode=TradingMode.LIVE,
            smartapi_api_key="k",
            smartapi_client_code="c",
            smartapi_password="p",
            smartapi_totp_secret="t",
        )
        assert not confirm_live(settings)

    @pytest.mark.parametrize("answer", ["", "y", "yes", "live", "LIVE ", "no"])
    def test_only_the_exact_word_confirms(
        self, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        import builtins

        monkeypatch.setattr(builtins, "input", lambda _prompt="": answer)
        settings = _settings(
            trading_mode=TradingMode.LIVE,
            smartapi_api_key="k",
            smartapi_client_code="c",
            smartapi_password="p",
            smartapi_totp_secret="t",
        )
        assert confirm_live(settings) is (answer.strip() == "LIVE")

    async def test_a_configuration_fault_is_not_retried(self, tmp_path: Path) -> None:
        """Exit 2 means bad credentials or an empty watchlist. Retrying cannot fix that."""
        script = tmp_path / "misconfigured.py"
        script.write_text("raise SystemExit(2)\n", encoding="utf-8")
        supervisor = IngestorSupervisor(script=script, max_restarts=5)

        await asyncio.wait_for(supervisor.run(), timeout=30.0)

        assert supervisor.stats.ingestor_starts == 1, "one attempt, not six"
        assert supervisor.stats.ingestor_restarts == 0
        assert supervisor.stats.gave_up_on_ingestor


class TestClientBootstrap:
    async def test_paper_without_credentials_runs_without_a_broker(self) -> None:
        from tachyon.main import build_client

        assert await build_client(_settings(trading_mode=TradingMode.PAPER)) is None

    async def test_live_without_credentials_is_fatal(self) -> None:
        """Without a broker the reconciler cannot prove the account is flat, so it would lock
        the session — and the operator would debug a lockdown whose cause was a blank .env."""
        from tachyon.execution.api import SmartApiError
        from tachyon.main import build_client

        with pytest.raises(SmartApiError, match="SMARTAPI_PASSWORD"):
            await build_client(_settings(trading_mode=TradingMode.LIVE))


class TestSupervisorSurface:
    def test_a_spawn_failure_is_absorbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A supervisor that dies takes the ingestor's only caretaker with it."""

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("cannot fork")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", explode)

        async def run() -> None:
            supervisor = IngestorSupervisor(script=Path(__file__), max_restarts=1)
            await asyncio.wait_for(supervisor.run(), timeout=20.0)
            assert supervisor.stats.gave_up_on_ingestor

        asyncio.run(run())

    async def test_stop_is_safe_before_anything_started(self) -> None:
        supervisor = IngestorSupervisor(enabled=False)
        await supervisor.stop()
        assert supervisor.pid is None
