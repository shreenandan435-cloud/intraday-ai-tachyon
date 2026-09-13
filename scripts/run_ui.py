"""UI entrypoint -- the read-only observability terminal (CLAUDE.md Section 7).

Run::

    .venv\\Scripts\\python.exe scripts/run_ui.py

Then open http://127.0.0.1:8787.

This process subscribes to both ZeroMQ spines and serves a browser. It holds no broker client
and cannot place an order. It is safe to start, stop and restart at any point during a session,
including mid-trade -- nothing downstream of it depends on it being up.

Binds to ``127.0.0.1`` by default. The terminal shows live P&L and a button that halts trading;
it has no authentication, so it must not be reachable from anywhere but this machine. Changing
``UI_HOST`` to ``0.0.0.0`` exposes both to your network.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402

from tachyon.core.config import get_settings  # noqa: E402
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging  # noqa: E402
from tachyon.ui.app import create_app  # noqa: E402


def main() -> int:
    settings = get_settings()
    configure_logging(
        role="ui",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )
    log = get_logger("run_ui")

    if settings.ui_host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning(
            "ui.exposed",
            host=settings.ui_host,
            impact="the terminal shows live P&L and can halt trading, and has no auth",
        )

    log.info("ui.serving", url=f"http://{settings.ui_host}:{settings.ui_port}")
    try:
        try:
            # uvicorn owns its own event loop. The UI never touches a zmq.asyncio socket -- the
            # bridge polls the blocking Subscriber with NOBLOCK -- so the Proactor-loop constraint
            # in tachyon.core.eventloop does not apply here.
            uvicorn.run(
                create_app(settings=settings),
                host=settings.ui_host,
                port=settings.ui_port,
                log_config=None,
                access_log=False,
            )
        except KeyboardInterrupt:
            log.info("ui.interrupted")
            return 0
        except SystemExit:
            # Preserve explicit exit codes (sys.exit(N), uvicorn clean shutdown, etc.).
            raise
        except Exception:
            # Hard failure: emit full traceback to BOTH the structured log and stderr.
            # The structured log persists to disk (so it survives a closed console window);
            # stderr is the fallback the operator can read in the same window that just died.
            tb_text = traceback.format_exc()
            log.exception("ui.startup_failed", exc_info=True)
            print(tb_text, file=sys.stderr, flush=True)
            try:
                shutdown_logging()
            except Exception:
                pass
            return 1
    finally:
        shutdown_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
