"""Start the whole system — ingestor, brain, sentinel (CLAUDE.md §2, §9).

Run::

    .venv\\Scripts\\python.exe scripts/run_tachyon.py

Equivalent to ``python -m tachyon.main``; this wrapper exists so every entrypoint lives in one
directory and works without the package being installed.

Starts the ingestor as a **child process** (CLAUDE.md §2 — separate OS processes, never threads
in one interpreter) and runs the Brain in this one. Ctrl-C shuts both down in order.

The UI is deliberately *not* started here. Run ``scripts/run_ui.py`` separately: it is a
read-only observer, and keeping it in its own process is what guarantees a UI fault cannot
reach the trading path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
