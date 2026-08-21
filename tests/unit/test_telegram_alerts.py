"""Telegram operator alerts — CLAUDE.md §5.1 (advisory asymmetry), §2.2, §8, §9.2.

Nothing here reaches the network: every alerter is built with ``start=False`` and an injected
``httpx.MockTransport``, then drained synchronously with ``drain_for_test``. A test that could
post to a real chat would be the same defect ``tests/conftest.py`` exists to prevent.

The load-bearing properties, in order of how much they would cost if wrong:

1. **Nothing raises.** An unreachable Telegram must not become an exception on the path that
   books P&L or flattens a position.
2. **Nothing blocks.** :meth:`send` enqueues and returns; it is called from the event loop and
   from the square-off watchdog thread.
3. **The token never reaches a log or an exception.**
"""

from __future__ import annotations

import re
import threading
from datetime import datetime
from decimal import Decimal

import httpx
import pytest

from tachyon.core.clock import IST, ManualClock
from tachyon.core.logger import redact_secrets
from tachyon.utils import telegram_alerts as alerts_module
from tachyon.utils.telegram_alerts import (
    MAX_MESSAGE_CHARS,
    AlertKind,
    TelegramAlerter,
    _endpoint,
    reset_default_alerter,
    send_telegram_alert,
)

TOKEN = "123456:FAKE-TOKEN-FOR-TESTS"
CHAT = "999"

#: Mid-session, so the timestamp in every alert is stable and readable in an assertion.
NOON = datetime(2026, 8, 17, 12, 0, 0, tzinfo=IST)


class _Recorder:
    """Captures every request the alerter makes, and replays a scripted status sequence."""

    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses) or [200]
        self.requests: list[httpx.Request] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            self.requests.append(request)
            status = self.statuses[min(len(self.requests) - 1, len(self.statuses) - 1)]
        return httpx.Response(status, json={"ok": status == 200})

    @property
    def payloads(self) -> list[dict[str, object]]:
        import json

        return [json.loads(r.content) for r in self.requests]

    @property
    def texts(self) -> list[str]:
        return [str(p["text"]) for p in self.payloads]


def _alerter(recorder: _Recorder | None = None, **kwargs: object) -> TelegramAlerter:
    rec = recorder if recorder is not None else _Recorder()
    kwargs.setdefault("start", False)
    return TelegramAlerter(
        bot_token=TOKEN,
        chat_id=CHAT,
        transport=httpx.MockTransport(rec),
        clock=ManualClock(wall=NOON),
        **kwargs,  # type: ignore[arg-type]
    )


class TestDisabledByDefault:
    """An unconfigured alerter must be inert, never an obstacle (CLAUDE.md §5.1)."""

    @pytest.mark.parametrize(
        ("token", "chat"),
        [("", CHAT), (TOKEN, ""), ("", ""), ("   ", CHAT)],
    )
    def test_missing_credentials_disable_delivery(self, token: str, chat: str) -> None:
        alerter = TelegramAlerter(bot_token=token, chat_id=chat, start=False)
        assert alerter.enabled is False
        assert alerter.send("anything") is False
        assert alerter.stats.suppressed == 1

    def test_the_master_switch_wins_over_credentials(self) -> None:
        alerter = _alerter(enabled=False)
        assert alerter.enabled is False
        assert alerter.send("anything") is False

    def test_a_disabled_alerter_closes_cleanly(self) -> None:
        alerter = TelegramAlerter(bot_token="", chat_id="", start=False)
        alerter.close()
        alerter.close()  # idempotent


class TestItNeverRaises:
    """Rule 1. Every one of these would otherwise surface on the P&L or flatten path."""

    def test_a_connection_error_is_swallowed(self) -> None:
        def explode(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        alerter = TelegramAlerter(
            bot_token=TOKEN,
            chat_id=CHAT,
            transport=httpx.MockTransport(explode),
            start=False,
        )
        alerter.send("hello")
        assert alerter.drain_for_test() == 1
        assert alerter.stats.failed == 1
        assert alerter.stats.sent == 0

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
    def test_every_error_status_is_swallowed(self, status: int) -> None:
        recorder = _Recorder(status, status)
        alerter = _alerter(recorder)
        alerter.send("hello")
        alerter.drain_for_test()
        assert alerter.stats.sent == 0
        assert alerter.stats.failed == 1

    def test_a_timeout_is_swallowed(self) -> None:
        def slow(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow")

        alerter = TelegramAlerter(
            bot_token=TOKEN, chat_id=CHAT, transport=httpx.MockTransport(slow), start=False
        )
        alerter.send("hello")
        alerter.drain_for_test()
        assert alerter.stats.failed == 1

    def test_the_convenience_function_never_raises_without_config(self) -> None:
        # conftest blanks TELEGRAM_BOT_TOKEN, so the process-wide alerter is inert.
        reset_default_alerter()
        try:
            assert send_telegram_alert("test") is False
        finally:
            reset_default_alerter()


class TestRetryPolicy:
    """§5's rule, applied here: exactly one retry, and only on 429/5xx."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transient_failures_are_retried_exactly_once(self, status: int) -> None:
        recorder = _Recorder(status, 200)
        alerter = _alerter(recorder)
        alerter.send("hello")
        alerter.drain_for_test()
        assert len(recorder.requests) == 2
        assert alerter.stats.sent == 1

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_configuration_faults_are_not_retried(self, status: int) -> None:
        recorder = _Recorder(status)
        alerter = _alerter(recorder)
        alerter.send("hello")
        alerter.drain_for_test()
        assert len(recorder.requests) == 1, "a bad token is not fixed by asking again"

    def test_a_persistent_5xx_gives_up_after_two_attempts(self) -> None:
        recorder = _Recorder(503, 503, 503)
        alerter = _alerter(recorder)
        alerter.send("hello")
        alerter.drain_for_test()
        assert len(recorder.requests) == 2
        assert alerter.stats.failed == 1


class TestTheTokenNeverLeaks:
    """Rule 3. Telegram forces the token into the URL path, so nothing may echo the URL."""

    def test_the_token_is_in_the_path_not_the_query_string(self) -> None:
        url = _endpoint(TOKEN, "sendMessage")
        assert f"/bot{TOKEN}/sendMessage" in url
        assert "?" not in url, "a query string would reach proxy logs (CLAUDE.md §5)"

    def test_the_bot_token_is_redacted_from_log_events(self) -> None:
        event = redact_secrets(None, "info", {"telegram_bot_token": TOKEN})  # type: ignore[arg-type]
        assert TOKEN not in str(event["telegram_bot_token"])

    @pytest.mark.parametrize("failure", ["status", "transport"])
    def test_no_log_event_from_the_send_path_can_carry_the_url(
        self, failure: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``str(httpx.HTTPError)`` embeds the request URL, and that URL holds the token.

        Asserted against every value the module logs rather than against captured output,
        because structlog renders to stderr and a test reading the stdlib capture would pass
        while leaking.
        """
        events: list[tuple[str, dict[str, object]]] = []

        class _SpyLog:
            def __getattr__(self, level: str) -> object:
                def record(event: str, **kw: object) -> None:
                    events.append((event, kw))

                return record

        monkeypatch.setattr(alerts_module, "_log", _SpyLog())

        if failure == "status":
            alerter = _alerter(_Recorder(401))
        else:

            def explode(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("failed", request=request)

            alerter = TelegramAlerter(
                bot_token=TOKEN,
                chat_id=CHAT,
                transport=httpx.MockTransport(explode),
                start=False,
            )

        alerter.send("hello")
        alerter.drain_for_test()

        assert events, "the failure must be logged, just without the credential"
        for event, kwargs in events:
            blob = f"{event} {kwargs}"
            assert TOKEN not in blob
            assert "api.telegram.org" not in blob


class TestQueueDiscipline:
    """Rule 2, plus §9.2's bounded-queue rule: drop and log, never block."""

    def test_send_returns_before_any_request_is_made(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        assert alerter.send("hello") is True
        assert recorder.requests == [], "send() must not touch the network on the caller's thread"
        alerter.drain_for_test()
        assert len(recorder.requests) == 1

    def test_a_full_queue_drops_rather_than_blocking(self) -> None:
        alerter = _alerter(queue_size=2)
        assert alerter.send("one") is True
        assert alerter.send("two") is True
        assert alerter.send("three") is False, "the third must be dropped, not block the caller"
        assert alerter.stats.dropped == 1

    def test_the_session_ceiling_disables_alerts_not_trading(self) -> None:
        alerter = _alerter(max_sends=2)
        assert alerter.send("one") is True
        assert alerter.send("two") is True
        assert alerter.send("three") is False
        assert alerter.stats.suppressed == 1

    def test_an_empty_message_is_ignored(self) -> None:
        alerter = _alerter()
        assert alerter.send("   ") is False

    def test_an_oversized_message_is_truncated_not_dropped(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.send("x" * (MAX_MESSAGE_CHARS * 2))
        alerter.drain_for_test()
        assert alerter.stats.truncated == 1
        assert len(recorder.texts[0]) <= MAX_MESSAGE_CHARS
        assert alerter.stats.sent == 1, "a clipped alert still tells the operator what happened"


class TestTheThreeEvents:
    """The events CLAUDE.md §1.1, §6.1 and §7.4 say an operator must learn about."""

    def test_an_entry_carries_the_geometry(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.entry_placed(
            symbol="RELIANCE",
            direction="LONG",
            quantity=12,
            entry=Decimal("1400.00"),
            stop=Decimal("1385.50"),
            target=Decimal("1421.75"),
            simulated=False,
        )
        alerter.drain_for_test()
        text = recorder.texts[0]
        assert "RELIANCE" in text
        assert "LONG" in text
        assert "1,385.50" in text, "an entry alert without the stop cannot be acted on"
        assert "1,421.75" in text
        assert AlertKind.ENTRY.value in text

    def test_a_paper_entry_says_so(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.entry_placed(
            symbol="INFY",
            direction="SHORT",
            quantity=5,
            entry=Decimal("1500"),
            stop=Decimal("1510"),
            target=Decimal("1485"),
            simulated=True,
        )
        alerter.drain_for_test()
        assert "[PAPER]" in recorder.texts[0], "a simulated fill must never read as a real one"

    def test_a_stop_out_is_labelled_a_stop_out(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.position_closed(
            symbol="INFY",
            realised=Decimal("-100.00"),
            charges=Decimal("22.50"),
            was_stop_out=True,
            session_total=Decimal("-122.50"),
            headroom=Decimal("377.50"),
        )
        alerter.drain_for_test()
        text = recorder.texts[0]
        assert AlertKind.STOP_LOSS.value in text
        assert "-Rs.122.50" in text, "net must be realised minus charges (CLAUDE.md §9.2)"
        assert "377.50" in text

    def test_a_profitable_close_is_a_target_not_a_stop_out(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.position_closed(
            symbol="TCS",
            realised=Decimal("150.00"),
            charges=Decimal("20.00"),
            was_stop_out=False,
            session_total=Decimal("130.00"),
            headroom=Decimal("630.00"),
        )
        alerter.drain_for_test()
        assert AlertKind.TARGET.value in recorder.texts[0]
        assert "+Rs.130.00" in recorder.texts[0]

    def test_a_losing_trade_closed_at_target_is_still_not_a_stop_out(self) -> None:
        """CLAUDE.md §7.4: the label comes from the closing order, never from the P&L sign."""
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.position_closed(
            symbol="TCS",
            realised=Decimal("5.00"),
            charges=Decimal("25.00"),  # net negative on charges alone
            was_stop_out=False,
            session_total=Decimal("-20.00"),
            headroom=Decimal("480.00"),
        )
        alerter.drain_for_test()
        assert AlertKind.TARGET.value in recorder.texts[0]
        assert AlertKind.STOP_LOSS.value not in recorder.texts[0]

    def test_the_square_off_trio(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.square_off_started(open_positions=2)
        alerter.square_off_failed(detail="NotFlatError: 1 working order")
        alerter.square_off_completed(session_total=Decimal("-40.00"))
        alerter.drain_for_test()
        assert "15:15" in recorder.texts[0]
        assert "Check the broker terminal" in recorder.texts[1]
        assert "flat" in recorder.texts[2]


class TestLifecycle:
    def test_close_is_idempotent_and_never_raises(self) -> None:
        alerter = _alerter()
        alerter.close()
        alerter.close()

    def test_a_started_alerter_uses_a_daemon_thread(self) -> None:
        """§9.2: non-daemon is for the square-off watchdog, which must outlive shutdown."""
        alerter = _alerter(start=True)
        try:
            thread = next(t for t in threading.enumerate() if t.name == "telegram-alerts")
            assert thread.daemon is True
        finally:
            alerter.close()

    def test_start_is_idempotent(self) -> None:
        alerter = _alerter(start=False)
        alerter.start()
        alerter.start()
        try:
            names = [t.name for t in threading.enumerate() if t.name == "telegram-alerts"]
            assert len(names) == 1
        finally:
            alerter.close()


class TestSetupDiagnostics:
    """The two failures every first-time setup hits, and the hint that resolves each."""

    def test_chat_not_found_explains_that_a_bot_cannot_open_a_chat(self) -> None:
        """The dominant setup failure. It reads as a bad id and usually is not."""

        def refuse(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"ok": False, "error_code": 400, "description": "Bad Request: chat not found"},
            )

        response = httpx.Client(transport=httpx.MockTransport(refuse)).post("https://x/")
        assert "chat not found" in alerts_module._describe(response)
        hint = alerts_module._hint_for(response)
        assert "Start" in hint and "cannot" in hint

    def test_a_revoked_token_says_to_reissue_it(self) -> None:
        def refuse(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"ok": False, "description": "Unauthorized"})

        response = httpx.Client(transport=httpx.MockTransport(refuse)).post("https://x/")
        assert "BotFather" in alerts_module._hint_for(response)

    def test_a_non_json_body_does_not_break_the_diagnosis(self) -> None:
        def refuse(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="<html>bad gateway</html>")

        response = httpx.Client(transport=httpx.MockTransport(refuse)).post("https://x/")
        assert alerts_module._describe(response) == response.reason_phrase

    def test_the_diagnosis_never_echoes_the_token(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"description": f"Bad Request for {request.url}"})

        recorder = _Recorder()
        alerter = _alerter(recorder)
        assert TOKEN not in alerts_module._hint_for(
            httpx.Client(transport=httpx.MockTransport(refuse)).post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            )
        )
        alerter.close()


class TestKillSwitchAndRogueFill:
    """CLAUDE.md §1.2 and §7.4 — the two alerts an operator cannot afford to miss."""

    def test_the_kill_switch_carries_pnl_and_the_limit(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.kill_switch(
            reason="DAILY_LOSS_LIMIT_TOTAL",
            detail="realised=-480 total=-502 limit=500",
            realised=Decimal("-480.00"),
            total=Decimal("-502.00"),
            limit=Decimal("500"),
            lock_durable=True,
        )
        alerter.drain_for_test()
        text = recorder.texts[0]
        assert "TRADING HALTED" in text
        assert "-Rs.502.00" in text
        assert "Rs.500.00" in text
        assert "restart will not reopen it" in text
        assert "RESTART WOULD NOT BE" not in text, "the lock was written; no warning is due"

    def test_an_unwritten_lock_file_warns_not_to_restart(self) -> None:
        """The one fact the operator cannot infer from the P&L number (CLAUDE.md §1.2)."""
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.kill_switch(
            reason="DAILY_LOSS_LIMIT_REALISED",
            detail="disk full",
            realised=Decimal("-500.00"),
            total=Decimal("-500.00"),
            limit=Decimal("500"),
            lock_durable=False,
        )
        alerter.drain_for_test()
        assert "RESTART WOULD NOT BE" in recorder.texts[0]
        assert "Do not restart today" in recorder.texts[0]

    @pytest.mark.parametrize(
        ("reason", "needle"),
        [
            ("DAILY_LOSS_LIMIT_REALISED", "No action needed"),
            ("DAILY_LOSS_LIMIT_TOTAL", "No action needed"),
            ("DAILY_DRAWDOWN_BREACHED", "No action needed"),
            ("ROGUE_FILL", "CHECK THE BROKER TERMINAL NOW"),
            ("PNL_UNDEFINED", "reconcile before restarting"),
            ("PNL_UNCOMPARABLE", "reconcile before restarting"),
        ],
    )
    def test_each_reason_carries_its_own_instruction(self, reason: str, needle: str) -> None:
        """A buzz saying only "locked" makes the operator open a laptop to learn which one."""
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.kill_switch(
            reason=reason,
            detail="d",
            realised=Decimal("-500"),
            total=Decimal("-500"),
            limit=Decimal("500"),
            lock_durable=True,
        )
        alerter.drain_for_test()
        assert needle in recorder.texts[0]

    def test_an_unknown_reason_still_gets_safe_guidance(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.kill_switch(
            reason="SOMETHING_NEW",
            detail="d",
            realised=Decimal("0"),
            total=Decimal("0"),
            limit=Decimal("500"),
            lock_durable=True,
        )
        alerter.drain_for_test()
        assert "Check the broker terminal before restarting" in recorder.texts[0]

    def test_a_rogue_fill_says_it_was_not_booked(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.rogue_fill(
            symbol="RELIANCE",
            order_id="99999",
            quantity=50,
            side="BUY",
            price=Decimal("1402.50"),
        )
        alerter.drain_for_test()
        text = recorder.texts[0]
        assert "RELIANCE" in text
        assert "99999" in text
        assert "NOT booked" in text, "booking it would corrupt the number the limit uses"
        assert "OPEN THE BROKER TERMINAL NOW" in text
        assert AlertKind.ROGUE_FILL.value in text

    def test_a_non_pnl_lockdown_is_labelled_as_such(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.session_locked(reason="ENTRY_OUTCOME_UNKNOWN")
        alerter.drain_for_test()
        text = recorder.texts[0]
        assert "ENTRY_OUTCOME_UNKNOWN" in text
        # Escaped, because the message is sent with parse_mode=HTML — a bare "&" would make
        # Telegram reject the whole message with a 400.
        assert "not a P&amp;L breach" in text
        assert AlertKind.LOCKED.value in text


class TestHtmlSafety:
    """Every message goes out with ``parse_mode=HTML``, and Telegram is unforgiving.

    A bare ``&``, ``<`` or ``>`` does not garble the alert — it makes Telegram reject the
    **whole message** with a 400. So an unescaped broker symbol deletes the alert entirely,
    which is worst exactly where it matters most: the rogue-fill and kill-switch messages
    are the ones that carry broker-supplied strings.
    """

    ALLOWED_TAGS = ("<b>", "</b>", "<i>", "</i>", "<code>", "</code>")

    def _assert_parses(self, text: str) -> None:
        stripped = text
        for tag in self.ALLOWED_TAGS:
            stripped = stripped.replace(tag, "")
        assert "<" not in stripped, f"stray '<' would 400 the message: {stripped!r}"
        assert ">" not in stripped, f"stray '>' would 400 the message: {stripped!r}"
        # Every '&' must open a character entity.
        for i, ch in enumerate(stripped):
            if ch == "&":
                tail = stripped[i : i + 8]
                assert re.match(r"&(amp|lt|gt|quot|#\d+);", tail), (
                    f"bare '&' would 400 the message: ...{tail!r}"
                )

    def test_every_alert_type_produces_parseable_html(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.entry_placed(
            symbol="RELIANCE",
            direction="LONG",
            quantity=1,
            entry=Decimal("1"),
            stop=Decimal("1"),
            target=Decimal("1"),
            simulated=True,
        )
        alerter.position_closed(
            symbol="INFY",
            realised=Decimal("-1"),
            charges=Decimal("1"),
            was_stop_out=True,
            session_total=Decimal("-2"),
            headroom=Decimal("1"),
        )
        alerter.square_off_started(open_positions=1)
        alerter.square_off_completed(session_total=Decimal("-2"))
        alerter.square_off_failed(detail="NotFlatError: 1 order")
        alerter.rogue_fill(symbol="TCS", order_id="1", quantity=1, side="BUY", price=Decimal("1"))
        alerter.session_locked(reason="ENTRY_OUTCOME_UNKNOWN")
        for reason in alerts_module._KILL_SWITCH_GUIDANCE:
            alerter.kill_switch(
                reason=reason,
                detail="d",
                realised=Decimal("-1"),
                total=Decimal("-1"),
                limit=Decimal("500"),
                lock_durable=False,
            )
        alerter.drain_for_test()

        assert len(recorder.texts) == 7 + len(alerts_module._KILL_SWITCH_GUIDANCE)
        for text in recorder.texts:
            self._assert_parses(text)

    @pytest.mark.parametrize("hostile", ["A&B", "<script>", "TATA>MOT", 'X"Y'])
    def test_broker_supplied_strings_cannot_break_the_message(self, hostile: str) -> None:
        """Symbols and order ids come off the wire — they are not ours to trust."""
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.rogue_fill(
            symbol=hostile,
            order_id=hostile,
            quantity=1,
            side=hostile,
            price=Decimal("1"),
        )
        alerter.drain_for_test()
        self._assert_parses(recorder.texts[0])
        assert alerter.stats.sent == 1

    def test_a_hostile_symbol_cannot_inject_a_bold_tag(self) -> None:
        recorder = _Recorder()
        alerter = _alerter(recorder)
        alerter.position_closed(
            symbol="<b>FAKE</b>",
            realised=Decimal("0"),
            charges=Decimal("0"),
            was_stop_out=False,
            session_total=Decimal("0"),
            headroom=Decimal("0"),
        )
        alerter.drain_for_test()
        assert "&lt;b&gt;FAKE&lt;/b&gt;" in recorder.texts[0]
