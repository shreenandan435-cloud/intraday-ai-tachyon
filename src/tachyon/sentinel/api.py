"""Async Gemini REST client — CLAUDE.md §5.

``httpx.AsyncClient`` against the Generative Language API directly. The ``google-genai`` SDK is
deliberately not used: it carries its own transport and retry behaviour that we would have to
work around, and CLAUDE.md §2.2 requires that nothing on the Brain's event loop can block. One
small client we fully understand beats a large one we mostly do.

The whole module exists to answer one question — *what is the macro regime* — under a hard
constraint: **it must never sit in the order path.** Everything below follows from that.

Timeouts
--------
Two of them, on purpose. ``httpx`` gets a per-phase timeout, and the whole call is additionally
wrapped in :func:`asyncio.wait_for`. The belt-and-braces matters because an httpx timeout
governs socket phases, not total elapsed time: a server trickling one byte per second resets
the read timer forever and the call never returns. The outer deadline is absolute.

Failure is normal, not exceptional
----------------------------------
A timeout, a 503, a rate limit, a truncated body — each raises a specific exception here, and
:class:`~tachyon.sentinel.classifier.RegimeClassifier` turns every one of them into
``NEUTRAL``. Nothing propagates into the trading path. The one thing that must never happen is
a degraded Gemini producing ``RISK_ON``: failure resolves toward *less* risk-taking, never more.

Budget
------
A session request budget (``sentinel.max_session_requests``) is enforced locally. Exceeding it
disables the Sentinel, **not** trading — an advisory component that has run out of quota is a
component we stop consulting, not a reason to stop working.

Secrets
-------
The API key travels in the ``x-goog-api-key`` header, never in the URL. Query strings end up in
proxy logs, crash reports and shell history; headers mostly do not. It is never logged and
never journaled (CLAUDE.md §5, §8).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Final, Self

import httpx
import msgspec

from tachyon.core.clock import SYSTEM_CLOCK, Clock
from tachyon.core.config import Settings, get_settings
from tachyon.core.logger import get_logger
from tachyon.persistence.journal import JsonlJournal

_log = get_logger(__name__)

BASE_URL: Final[str] = "https://generativelanguage.googleapis.com"

#: Path template for a single-turn generation call.
GENERATE_PATH: Final[str] = "/v1beta/models/{model}:generateContent"

#: The model the Sentinel classifies with. Overridable via ``GEMINI_MODEL`` in ``.env``; this
#: is the default that ``core.config`` uses, restated here because this is the module that
#: sends it and a reader of the request path should not have to go looking.
#:
#: A wrong or retired model id is **not** a hazard: the API answers 404, which is classified as
#: :class:`GeminiRejectedError` (a configuration fault, never retried) and the classifier
#: degrades to ``NEUTRAL``. The Sentinel goes quiet; trading is unaffected (CLAUDE.md §5.1).
#: Confirm an id against the live account with::
#:
#:     GET https://generativelanguage.googleapis.com/v1beta/models
#:     x-goog-api-key: <key>
DEFAULT_MODEL: Final[str] = "gemini-3.5-flash-lite"

#: Absolute ceiling on one classification, enforced by ``asyncio.wait_for``. CLAUDE.md §5 sets
#: 8 s; ``sentinel.request_timeout_seconds`` may tighten it further but this is the fallback.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 8.0

#: HTTP statuses worth one retry: the service is up but momentarily unwilling. Everything else
#: (401, 400, 404) is a configuration fault that retrying cannot fix.
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 504})

_ENCODER: Final[msgspec.json.Encoder] = msgspec.json.Encoder()


class GeminiError(RuntimeError):
    """Base for every Sentinel transport or protocol failure. Always degrades to NEUTRAL."""


class GeminiTimeoutError(GeminiError):
    """The model did not answer within the deadline."""


class GeminiUnavailableError(GeminiError):
    """The service returned a retryable status, or the transport failed."""


class GeminiRejectedError(GeminiError):
    """The service refused the request — bad key, bad model, malformed body. Not retryable."""


class GeminiBudgetExhaustedError(GeminiError):
    """The session request budget is spent. Disables the Sentinel, never trading."""


class GeminiNotConfiguredError(GeminiError):
    """No ``GEMINI_API_KEY``. The Sentinel simply does not run."""


@dataclass(slots=True)
class GeminiStats:
    """Counters for observability. Never used for a trading decision."""

    requests: int = 0
    responses: int = 0
    timeouts: int = 0
    retries: int = 0
    rejections: int = 0
    blocked_by_safety: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class GeminiReply:
    """One successful generation."""

    text: str
    """The model's raw output. JSON when a response schema was supplied, but *unvalidated* —
    parsing is the classifier's job, and it treats a parse failure as NEUTRAL."""

    latency_ms: float
    prompt_tokens: int = 0
    response_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class GeminiClient:
    """Minimal async client for ``models/{model}:generateContent``.

    Args:
        settings: resolved config. Supplies the API key, the model and the timeout.
        journal: every prompt and raw response is written here (CLAUDE.md §5).
        client: injected ``httpx.AsyncClient``, for tests.
        transport: injected ``httpx`` transport, for tests.
        max_requests: session budget. ``0`` disables the limit.

    Example::

        async with GeminiClient(settings=settings) as gemini:
            reply = await gemini.generate(system="...", prompt="...", schema=SCHEMA)
    """

    __slots__ = (
        "_budget",
        "_client",
        "_clock",
        "_journal",
        "_model",
        "_owns_client",
        "_settings",
        "_timeout",
        "stats",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        journal: JsonlJournal | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        base_url: str = BASE_URL,
        clock: Clock = SYSTEM_CLOCK,
        timeout: float | None = None,
        max_requests: int | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        # A blank GEMINI_MODEL in .env would otherwise build "/v1beta/models/:generateContent"
        # and spend the session's budget on 404s. It still fails safe, but silently.
        configured = self._settings.gemini_model.strip()
        if not configured:
            _log.warning(
                "sentinel.model_unset",
                fallback=DEFAULT_MODEL,
                reason="GEMINI_MODEL is blank in the environment",
            )
        self._model = configured or DEFAULT_MODEL
        self._timeout = (
            timeout if timeout is not None else self._settings.sentinel.request_timeout_seconds
        )
        self._budget = (
            max_requests
            if max_requests is not None
            else self._settings.sentinel.max_session_requests
        )
        self._journal = (
            journal if journal is not None else JsonlJournal(prefix="sentinel", clock=clock)
        )

        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(base_url=base_url, timeout=self._timeout, transport=transport)
        )

        self.stats = GeminiStats()

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def model(self) -> str:
        return self._model

    @property
    def is_configured(self) -> bool:
        """True if an API key is present. False means the Sentinel is simply absent."""
        return bool(self._settings.gemini_api_key.get_secret_value().strip())

    @property
    def budget_remaining(self) -> int:
        """Requests left this session. ``-1`` when unlimited."""
        if self._budget <= 0:
            return -1
        return max(0, self._budget - self.stats.requests)

    def reset_session(self) -> None:
        """Restore the request budget for a new trading day."""
        self.stats = GeminiStats()

    # ── the call ─────────────────────────────────────────────────────────────

    def build_payload(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Assemble the request body.

        ``temperature=0`` by default: this is a classification, not a creative task, and a
        deterministic classifier is one whose disagreement with yesterday means something.

        ``responseMimeType``/``responseSchema`` ask the service to constrain its own output to
        the schema. That is a strong hint, not a guarantee — the classifier still parses
        defensively, because "the API usually returns valid JSON" is not a property anyone
        should bet ₹500 on.
        """
        generation: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = schema

        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation,
        }

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        event: str = "generate",
    ) -> GeminiReply:
        """Run one generation.

        Raises:
            GeminiNotConfiguredError: no API key.
            GeminiBudgetExhaustedError: the session budget is spent.
            GeminiTimeoutError: the deadline elapsed.
            GeminiUnavailableError: transport failure or a retryable status.
            GeminiRejectedError: the service refused the request.

        Every one of these is caught by the classifier and becomes ``NEUTRAL``.
        """
        if not self.is_configured:
            raise GeminiNotConfiguredError("GEMINI_API_KEY is not set — Sentinel disabled")
        if self._budget > 0 and self.stats.requests >= self._budget:
            raise GeminiBudgetExhaustedError(
                f"session budget of {self._budget} Gemini requests is spent; "
                f"the Sentinel is disabled for the rest of the day (trading is not)"
            )

        payload = self.build_payload(
            system=system, prompt=prompt, schema=schema, temperature=temperature
        )
        # Journalled before the call, so a crash mid-flight still records what we asked.
        self._journal.request(event, {"model": self._model, "prompt": prompt, "system": system})

        started = self._clock.monotonic()
        try:
            # asyncio.wait_for on top of httpx's own timeout: httpx times out socket *phases*,
            # so a server trickling bytes can hold a connection open indefinitely. This makes
            # the deadline absolute.
            response = await asyncio.wait_for(self._post(payload), timeout=self._timeout + 1.0)
        except (TimeoutError, httpx.TimeoutException) as exc:
            # Both spellings of the same fact. `TimeoutError` is the outer asyncio deadline;
            # `httpx.TimeoutException` is a socket phase giving up first. They differ only in
            # which guard noticed, and reporting them as one keeps the counter meaningful.
            self.stats.timeouts += 1
            self.stats.last_error = "deadline exceeded"
            self._journal.error(event, f"timeout after {self._timeout}s")
            raise GeminiTimeoutError(
                f"Gemini did not answer within {self._timeout}s — degrading to NEUTRAL"
            ) from exc
        except httpx.HTTPError as exc:
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            self._journal.error(event, self.stats.last_error)
            raise GeminiUnavailableError(f"Gemini transport failed: {exc}") from exc

        latency_ms = (self._clock.monotonic() - started) * 1000.0
        return self._parse_response(response, latency_ms, event)

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """POST with one retry on a retryable status.

        Exactly one retry. Generation is idempotent in the only sense that matters — it places
        no orders and moves no money — so a retry is safe, but the Sentinel runs on a 15-minute
        cadence and there is no value in fighting a struggling service inside one cycle.
        """
        path = GENERATE_PATH.format(model=self._model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._settings.gemini_api_key.get_secret_value(),
        }
        body = _ENCODER.encode(payload)

        for attempt in (0, 1):
            self.stats.requests += 1
            response = await self._client.post(
                path, content=body, headers=headers, timeout=self._timeout
            )
            if response.status_code not in _RETRYABLE_STATUSES:
                return response
            if attempt == 0:
                self.stats.retries += 1
                _log.warning(
                    "sentinel.gemini_retry",
                    status=response.status_code,
                    model=self._model,
                )
                await asyncio.sleep(0.5)
        return response

    def _parse_response(
        self, response: httpx.Response, latency_ms: float, event: str
    ) -> GeminiReply:
        """Turn an HTTP response into a :class:`GeminiReply`, or raise."""
        if response.status_code in _RETRYABLE_STATUSES:
            self.stats.last_error = f"HTTP {response.status_code}"
            self._journal.error(event, self.stats.last_error)
            raise GeminiUnavailableError(
                f"Gemini returned HTTP {response.status_code} after a retry"
            )
        if response.status_code >= 400:
            self.stats.rejections += 1
            self.stats.last_error = f"HTTP {response.status_code}"
            self._journal.error(event, self.stats.last_error, body=response.text[:500])
            raise GeminiRejectedError(
                f"Gemini rejected the request: HTTP {response.status_code} {response.text[:200]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            self.stats.last_error = "non-JSON response envelope"
            self._journal.error(event, self.stats.last_error)
            raise GeminiUnavailableError("Gemini returned a non-JSON envelope") from exc

        envelope = body if isinstance(body, dict) else {}
        self.stats.responses += 1
        self._journal.response(event, envelope, latency_ms=latency_ms)

        text = _extract_text(envelope)
        if text is None:
            # A response with no candidate text is almost always a safety block. It is not a
            # crash and it is not a regime — the classifier degrades to NEUTRAL.
            self.stats.blocked_by_safety += 1
            self.stats.last_error = "no candidate text (safety block or empty response)"
            raise GeminiUnavailableError(
                "Gemini returned no candidate text — treating as unavailable"
            )

        usage = envelope.get("usageMetadata", {})
        prompt_tokens = _int_or_zero(usage, "promptTokenCount")
        response_tokens = _int_or_zero(usage, "candidatesTokenCount")
        self.stats.prompt_tokens += prompt_tokens
        self.stats.response_tokens += response_tokens

        _log.info(
            "sentinel.gemini_ok",
            model=self._model,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            budget_remaining=self.budget_remaining,
        )
        return GeminiReply(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            raw=envelope,
        )


def _extract_text(envelope: dict[str, Any]) -> str | None:
    """Pull the first candidate's concatenated text, or ``None``.

    Written defensively on purpose: every level of this structure is optional in some error
    or safety-block response, and an ``IndexError`` deep inside a parser is a worse failure
    than a clean "no text".
    """
    candidates = envelope.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    chunks = [str(part["text"]) for part in parts if isinstance(part, dict) and "text" in part]
    joined = "".join(chunks).strip()
    return joined or None


def _int_or_zero(payload: dict[str, Any], key: str) -> int:
    try:
        return int(payload.get(key, 0) or 0)
    except TypeError, ValueError:
        return 0
