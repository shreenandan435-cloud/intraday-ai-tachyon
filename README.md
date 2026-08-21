# intraday-ai-tachyon

Single-operator NSE intraday trading system. Two processes over ZeroMQ, a Numba JIT math
engine, a hard non-negotiable risk gate, an advisory Gemini sentinel, and a dark glassmorphic
terminal UI.

> **[`CLAUDE.md`](CLAUDE.md) is the project constitution and the source of truth.**
> Read it before touching any module. If code and that document disagree, the code is wrong.

## Status

**All 12 phases complete.**

- **`core/`** — frozen constants, guarded config, IST clock with monotonic deadlines,
  structured logging, state machine with on-disk daily lock.
- **`ipc/`** — frozen msgspec schemas, ZeroMQ PUB/SUB wrappers, conflation policy, FeedMonitor.
- **`math_engine/`** — Numba VWAP / EMA / Wilder ATR / OBI kernels, mirrored zero-copy ring
  buffers, `TickAggregator` with sequence-gap detection, boot warmup with self-test.
- **`ingestion/`** — SmartAPI v2 binary decoder (Mode 3 / L2 depth), per-token sequencing,
  flap-resistant reconnecting WebSocket client, publisher daemon.
- **`risk/`** — eleven-step fail-safe veto gate, latching ₹500 kill switch with on-disk lock,
  immortal 15:15 square-off watchdog that retries until actually flat.
- **`execution/`** — rate-limited async SmartAPI client with TOTP login, Robo order geometry
  and 60/40 two-leg split, broker-native trailing stop, boot-time reconciliation, append-only
  order journal.
- **`strategy/`** — VWAP + EMA + OBI confluence signals, 30-minute re-entry cooldown, and the
  async Brain that wires every layer together without blocking on any of them.
- **`sentinel/`** — async Gemini macro-regime classifier over `httpx`, defensive parsing, and a
  latching `RISK_OFF` veto that can only ever restrict.
- **`ui/`** — read-only FastAPI + WebSocket glass terminal, a conflating telemetry bridge that
  cannot block the Brain, and the broker fill listener that closes positions and books P&L.
- **`main.py`** — the orchestrator: launches the ingestor as a child process, runs the Brain,
  supervises both, and shuts them down in an order that strands nothing.
- **`tests/chaos/`** — the Crucible: kill the feed mid-session, storm the broker with 503s,
  forge a webhook fill, yank the clock. Real sockets, real threads, asserted teardown.

994 tests, `mypy --strict` and `ruff` clean. Order geometry and the risk gate are both at
**100 % branch coverage**, as CLAUDE.md §8 requires.

Measured — tick spine: 69 B per tick, 0.18 µs encode / 0.24 µs decode, ~52 µs p50 one-way over
loopback TCP, zero loss and strictly in order at 10× realistic tick rates.
Math engine: **16.1 µs p50** full indicator refresh at capacity, against a 50 µs budget;
JIT warmup 136 ms from cache.

Build order is load-bearing: the Risk Engine (Phase 6) shipped before Execution (Phase 7).
Nothing in `execution/` may place an order on its own authority — `RoboExecutor.open_position`
requires a passing `RiskDecision` *and* re-runs the gate immediately before transmitting.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q             # 994 passed
.\.venv\Scripts\python.exe scripts\bench_math_engine.py   # gates the 50 us budget
.\.venv\Scripts\python.exe scripts\run_tachyon.py         # ingestor + brain + sentinel
.\.venv\Scripts\python.exe scripts\run_ui.py              # terminal at http://127.0.0.1:8787

.\.venv\Scripts\python.exe scripts\run_ingestor.py        # or start the halves separately
.\.venv\Scripts\python.exe scripts\run_brain.py
.\.venv\Scripts\mypy.exe                                  # strict, src/tachyon
.\.venv\Scripts\ruff.exe check src tests scripts
.\.venv\Scripts\python.exe -m pytest tests\chaos -q   # the Crucible, on its own
```

> **`winloop` is required on Windows, not optional.** The default Proactor event loop does not
> implement `add_reader`, so a `zmq.asyncio` socket on it receives nothing at all — the Brain
> would look alive and consume zero ticks. `tachyon.core.eventloop.run` verifies this at
> startup and refuses to run on an unsuitable loop; both entrypoints go through it.

> The binary decoder is written against Angel One's published protocol spec. Its tests prove
> the unpacking matches that spec byte for byte; they cannot prove the spec matches the live
> feed. **Validate a real session in PAPER mode before trusting prices** — especially the
> timestamp unit and the price scale on any non-NSE-cash segment.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
Copy-Item .env.example .env      # then fill in credentials
```

`TRADING_MODE` defaults to `PAPER`. `LIVE` requires the env var set explicitly *and* an
interactive confirmation at boot.

## Layout

```
CLAUDE.md              project constitution — architecture, math, risk, styling
config/settings.yaml   tunables (hard risk constants deliberately NOT here)
scripts/               process entrypoints: run_ingestor.py, run_brain.py, run_ui.py
src/tachyon/
  core/                constants, config, IST clock, structlog, state machine
  ipc/                 ZeroMQ PUB/SUB wrappers + versioned msgspec schemas
  ingestion/           SmartAPI WebSocket v2 → ZMQ publisher (Process A)
  math_engine/         Numba kernels: VWAP, EMA, ATR(Wilder), OBI + ring buffers
  strategy/            confluence signals + the async Brain; intents only, never authorises
  risk/                the only component that may authorise an order
  execution/           SmartAPI Robo/Bracket orders, two-leg T1/T2, native trailing SL
  sentinel/            async Gemini regime + news risk (advisory, downside-only)
  ui/                  FastAPI + WS terminal, telemetry bridge, broker fill listener
  persistence/         append-only JSONL journal shared by orders, fills and the sentinel
tests/                 pytest; risk + order geometry require 100% branch coverage
data/journal/          the post-mortem record of record (git-ignored)
```

## The rules that never bend

- **15:15 IST** — cancel all, exit all, latch `SQUARING_OFF` → `SQUARED_OFF`. Immortal
  non-daemon watchdog thread on the monotonic clock; fires even if the data feed is dead, and
  retries the flatten until it actually succeeds. No flag skips it.
- **₹500 daily loss** — latching kill switch, persisted to disk; a restart boots read-only.
- **Anything ambiguous is a veto.** A risk check that raises refuses the trade; an undefined
  P&L counts as breached; an unreadable lock file counts as locked.
- **Stop-loss = 1.5 × ATR(5m)**, **T1 = 1:1.5 R/R**, **T2 = 1:2.5 R/R**, broker-native trailing.
  Stops move one direction only — toward profit — and that is enforced in code, not by review.
- **A placement whose outcome is unknown is never retried.** A duplicate live order is worse
  than a missed fill; the answer is to reconcile against the order book.
- **Boot never assumes flat.** If the broker holds something we do not know about, or cannot be
  asked, the session locks.
- **A fill we never placed bricks the day.** Provenance is the registered order id or our
  `TCHYN-` client tag; anything else writes the day lock and is not booked.
- **A signal is a request, not permission.** Three-way confluence produces an intent; the Risk
  Engine alone decides, and re-decides immediately before the order is transmitted.
- **The UI can watch but not trade.** Its one control engages the daily lock, which blocks new
  entries and does not flatten — and the button says exactly that. A stale number is amber with
  its age; a number that never arrived is a dash, never a zero.
- **PAPER cannot transmit an order.** The interception sits at the transport, below every
  caller: outside LIVE, a placement, modification or cancellation is simulated and the request
  never leaves the machine. Read-only calls still hit the live API, because a paper session
  reading an invented order book is a fiction rather than a rehearsal.
- **The AI Sentinel may only ever make the system more conservative.** Its block is a latch and
  its size multiplier a ratchet, so a cheerier second opinion can never undo a first. Every
  failure — timeout, outage, malformed JSON, missing key — resolves to `NEUTRAL`, never
  `RISK_ON`, and never halts trading: a broken advisor must not become a kill switch.
