# TACHYON — Current Architecture and Change Rules

## Source of truth

[`CLAUDE.md`](CLAUDE.md) is mandatory. Its risk, execution, safety, and operational constraints override all other guidance. Do not weaken, bypass, or reinterpret them.

`src/tachyon/` is the primary maintained architecture.

Root-level scripts and modules (for example `main.py`, `orchestrator.py`, `train_ppo.py`, and `export_onnx.py`) are legacy or experimental unless a task explicitly selects them. Do not use them as a basis for production changes by default.

## Current architecture

Tachyon currently has two coexisting decision paths:

- The conventional live architecture: Angel One ingestion, ZeroMQ IPC, Numba market math, rule-based strategy logic, Risk Engine, execution, reconciliation, watchdogs, and UI.
- The RL path: Transformer/GTrXL PPO training and a TensorRT inference sidecar using normalized LOB state.

RL/TensorRT infrastructure exists, but RL is **not yet proven to be the production decision authority**. Do not replace, bypass, or alter conventional trading decisions with model actions unless explicitly authorized.

## Data collection and offline training

The passive L2 recorder is a separate ZeroMQ sidecar process. It must never back-pressure, block, or degrade the market-data, Brain, risk, or execution paths.

- L2 market recording uses Parquet.
- Operational journals and telemetry use JSONL.
- Offline model training must remain isolated from live trading processes and data paths.

## Inference and performance

The current production-oriented inference pipeline is:

`PyTorch → ONNX → TensorRT`

FP16 TensorRT export/build support is implemented. INT8 quantization is not yet implemented; do not claim, assume, or introduce it without explicit calibration, accuracy validation, and authorization.

Sub-1 ms inference latency is a target, not an already-proven production acceptance criterion. Measure realistic end-to-end latency before making production-performance claims.

## Non-negotiable trading protections

Preserve and test all protections defined in `CLAUDE.md`, including:

- The Risk Engine remains the sole authority that can authorize an order.
- PAPER and LIVE modes remain strictly separated; PAPER must not transmit orders.
- Broker reconciliation must occur where required; unknown outcomes must not be retried as new orders.
- Daily-loss protection and its persistent/latching behavior must remain intact.
- Mandatory square-off behavior must remain intact and cannot be bypassed.
- Model actions, if integrated, must be risk-gated and must never directly place orders.
- Never enable LIVE trading, change production trading-mode defaults, or push a build that changes live-order behavior unless explicitly authorized for that specific change.

Do not change live trading behavior, order routing, risk limits, execution behavior, or production mode semantics unless the task explicitly authorizes it.

## Engineering rules

- Prefer small, isolated changes with focused tests.
- Preserve non-blocking behavior on all live market and execution paths.
- Keep model, recorder, and experimental work separated from the critical trading path unless explicitly integrating them.
- Verify changes with tests appropriate to their risk; risk, execution, IPC, and model-action changes require targeted coverage.
- Avoid broad refactors and unrelated cleanup.

## Remaining goals

The current Phase 2 goals are:

1. Integrate live L2 data into the model action path.
2. Route every model action through the Risk Engine and existing execution safeguards.
3. Implement model/version management and safe rollback.
4. Add PyTorch/ONNX/TensorRT parity testing.
5. Measure realistic end-to-end latency under representative load.
6. Complete PAPER-mode and production validation before granting the RL path any production decision authority.
