# TACHYON: Phase 2 Quant Architecture

**WHAT:** Transitioning a Python/ZeroMQ Limit Order Book (LOB) trading bot on Angel One from rule-based to a SOTA Deep Reinforcement Learning (Transformer-PPO) agent.
**HOW:** Parallel execution. Track 1 utilizes a ZeroMQ Sidecar to asynchronously record Level-2 order book data to Parquet/JSONL without degrading the main orchestrator's tick path. Track 2 focuses on training the offline PyTorch RL model.
**WHY (Strict Constraints):** 
- The ultimate inference target is <1ms execution latency. Speed matters, but not more than system stability.
- All Deep Learning models will be quantized to INT8/FP16 and compiled via ONNX Runtime to NVIDIA TensorRT.
- I/O disk writes must NEVER block the main Python thread.
- Adhere to progressive disclosure: build small, isolated, high-performance scripts to avoid context bloat.