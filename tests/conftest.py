"""Test-suite isolation from the operator's real credentials.

A developer machine has a real ``.env`` with real keys in it. Without this file, every
``Settings()`` in the suite reads it — which means:

* an unrelated key in ``.env`` fails the whole suite on ``extra="forbid"``;
* and, far worse, a correctly-named ``GEMINI_API_KEY`` makes
  :class:`~tachyon.sentinel.api.GeminiClient` report itself configured, so any test that runs
  the Sentinel daemon calls **the live Google API** — spending real quota, at 8 seconds a
  timeout, on a machine that thought it was running unit tests.

That is not a hypothetical: it is how this file came to exist. Tests must be hermetic, and a
test suite that can reach the internet on someone's credentials is a defect in the suite.

Both sources are neutralised: ``ENV_FILE`` is pointed at a path that does not exist (resolved
at load time by ``settings_customise_sources``), and the credential environment variables are
blanked. Environment beats ``.env`` in the precedence chain, so the blanks win either way.

``config/settings.yaml`` is neutralised the same way and for a related reason — see
:func:`_isolate_from_operator_settings`. A suite whose verdict depends on a file the operator
edits between sessions is not a suite.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tachyon.core import config as config_module
from tachyon.core import token_cache as token_cache_module
from tachyon.persistence import tick_recorder as tick_recorder_module
from tachyon.persistence import trade_logger as trade_logger_module

#: Every variable that could make a test reach a real API.
_CREDENTIAL_VARS = (
    "SMARTAPI_API_KEY",
    "SMARTAPI_CLIENT_CODE",
    "SMARTAPI_PASSWORD",
    "SMARTAPI_TOTP_SECRET",
    "SMARTAPI_FEED_TOKEN",
    "GEMINI_API_KEY",
    # Exactly the Gemini hazard in the docstring, with a louder failure mode: a real token
    # here does not merely spend quota, it posts test-run noise into the operator's own
    # Telegram chat — including alerts claiming positions were opened and stopped out.
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
)


@pytest.fixture(scope="session", autouse=True)
def _isolate_from_real_credentials() -> Iterator[None]:
    """Cut the suite off from ``.env`` and from any credential in the environment."""
    original_env_file = config_module.ENV_FILE
    config_module.ENV_FILE = original_env_file.with_name(".env.absent-in-tests")

    saved = {name: os.environ.get(name) for name in _CREDENTIAL_VARS}
    for name in _CREDENTIAL_VARS:
        os.environ[name] = ""

    config_module.get_settings.cache_clear()
    try:
        yield
    finally:
        config_module.ENV_FILE = original_env_file
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        config_module.get_settings.cache_clear()


#: The real, shipped ``config/settings.yaml``. Exposed so the tests whose *subject* is that
#: file can opt back into it explicitly (see ``TestConfig`` in ``tests/unit/test_core.py``).
REAL_SETTINGS_YAML = config_module.SETTINGS_YAML

#: The pinned stand-in every other test sees instead.
FIXTURE_SETTINGS_YAML = Path(__file__).parent / "fixtures" / "settings.test.yaml"


@pytest.fixture(scope="session", autouse=True)
def _isolate_from_operator_settings() -> Iterator[None]:
    """Cut the suite off from the operator's ``config/settings.yaml``.

    ``settings_customise_sources`` merges that file into **every** ``Settings()`` in the suite,
    including ones built from explicit kwargs — a test that passes ``watchlist=`` still inherits
    ``capital``, ``positions``, ``sentinel`` and the rest from whatever the operator has on disk.

    That is not hypothetical either. Activating the dynamic session budget (``session_budget_inr:
    50000``) turned three passing tests red without a line of source changing, because they
    assert against the §1 ₹500 limit and the Brain had silently resolved a ₹1000 one from the
    operator's YAML. Tuning a live config must not be able to change what the tests mean, in
    either direction — the dangerous version of this is the one that turns a failure *green*.

    The file is *replaced* rather than removed: plenty of tests call a bare ``Settings()`` and
    assert on RELIANCE, so an empty watchlist would fail them for the wrong reason.
    ``tests/fixtures/settings.test.yaml`` is a pinned stand-in that only a test may change.
    """
    original = config_module.SETTINGS_YAML
    config_module.SETTINGS_YAML = FIXTURE_SETTINGS_YAML
    config_module.get_settings.cache_clear()
    try:
        yield
    finally:
        config_module.SETTINGS_YAML = original
        config_module.get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _isolate_trade_output(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep test trade CSVs out of the operator's real ``data/trades``.

    Any test that constructs a :class:`~tachyon.strategy.brain.StrategyBrain` gets a real
    :class:`~tachyon.persistence.trade_logger.TradeLogger` unless it injects one, and a veto in
    a test would otherwise append to the same dated CSV a live session is writing. The logger
    resolves ``TRADES_DIR`` at construction, so redirecting the module constant is enough.
    """
    original = trade_logger_module.TRADES_DIR
    trade_logger_module.TRADES_DIR = tmp_path_factory.mktemp("trades")
    try:
        yield
    finally:
        trade_logger_module.TRADES_DIR = original


@pytest.fixture(scope="session", autouse=True)
def _isolate_tick_output(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep test Parquet out of the operator's real ``data/ticks``.

    Same mechanism as :func:`_isolate_trade_output`, and a sharper reason to want it: the tick
    dataset is the training set for Phase 2. A test that appended a few synthetic books into a
    real ``date=`` partition would not fail anything — it would quietly poison the corpus the
    RL agent learns from, and nothing downstream could tell the rows apart.
    """
    original = tick_recorder_module.TICKS_DIR
    tick_recorder_module.TICKS_DIR = tmp_path_factory.mktemp("ticks")
    try:
        yield
    finally:
        tick_recorder_module.TICKS_DIR = original


@pytest.fixture(autouse=True)
def _isolate_token_cache(tmp_path: Path) -> Iterator[None]:
    """Keep test logins away from the operator's real ``data/cache/session_token.json``.

    Two hazards, one in each direction: a test that *wrote* a synthetic session into the real
    cache would hand the next live boot a dead token, and a test that *read* the operator's
    real cached session would skip its mock transport entirely and assert against whatever
    the broker last issued. Function-scoped on purpose: a session-scoped directory would let
    one test's saved cache leak into the next test's login. The module resolves the path at
    call time, so redirecting the constant is enough.
    """
    original = token_cache_module.TOKEN_CACHE_PATH
    token_cache_module.TOKEN_CACHE_PATH = tmp_path / "session_token.json"
    try:
        yield
    finally:
        token_cache_module.TOKEN_CACHE_PATH = original
