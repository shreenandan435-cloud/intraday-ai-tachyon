"""Track 2 — the offline Transformer-PPO pipeline.

Nothing in this package is imported by the live trading path. The Ingestor, the Brain, the risk
gate and the UI never load ``torch``; this package exists to turn what
:mod:`tachyon.persistence.tick_recorder` harvested into a trained policy, and the policy reaches
production as a compiled ONNX/TensorRT artefact rather than as Python.

That separation is deliberate and load-bearing. ``import torch`` alone costs seconds and hundreds
of megabytes of RSS, and a CUDA context in the orchestrator process would put a driver-level
allocator behind the <1ms inference budget. Keeping the boundary at "the trading path never
imports this package" is what makes the training stack free to be as heavy as it needs to be.
"""

from __future__ import annotations

__all__ = ["dataset", "embedding", "transformer", "ppo", "export"]
