import os
import time
import torch
import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO

print("[*] Loading trained policy into memory...")
model = PPO.load("models/tachyon_brain_generalized", device="cpu")
model.policy.eval()

dummy_input = torch.randn(1, 6, dtype=torch.float32)

class FastONNXPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        features = self.extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        return torch.argmax(action_logits, dim=1)

fast_net = FastONNXPolicy(model.policy)
fast_net.eval()

onnx_path = "models/tachyon_brain.onnx"
torch.onnx.export(
    fast_net,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch_size"}, "action": {0: "batch_size"}}
)
print(f"[✓] Clean ONNX binary compiled to: {onnx_path}")

# LATENCY BENCHMARK: PyTorch vs ONNX Runtime
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

test_obs_np = np.random.randn(1, 6).astype(np.float32)
test_obs_th = torch.from_numpy(test_obs_np)

# Warmup
for _ in range(50):
    _ = model.predict(test_obs_np[0], deterministic=True)
    _ = session.run(None, {"obs": test_obs_np})

# PyTorch timing (1000 ticks)
t0 = time.perf_counter()
for _ in range(1000):
    _ = model.predict(test_obs_np[0], deterministic=True)
pytorch_latency_us = ((time.perf_counter() - t0) / 1000) * 1_000_000.0

# ONNX Runtime timing (1000 ticks)
t0 = time.perf_counter()
for _ in range(1000):
    _ = session.run(None, {"obs": test_obs_np})
onnx_latency_us = ((time.perf_counter() - t0) / 1000) * 1_000_000.0

print(f"\n⚡ INFERENCE LATENCY BENCHMARK (1,000 runs):")
print(f"   • Baseline PyTorch CPU: {pytorch_latency_us:.1f} µs")
print(f"   • ONNX Native C++:      {onnx_latency_us:.1f} µs")
print(f"   • Speedup:              {pytorch_latency_us / max(0.1, onnx_latency_us):.1f}x faster\n")
