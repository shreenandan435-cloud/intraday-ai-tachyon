import torch
import numpy as np
from stable_baselines3 import PPO

model = PPO.load("models/tachyon_brain_generalized", device="cpu")

# Synthetic states: [spread_bps, micro_delta_bps, obi_l1, obi_deep, position, unrealized_bps]
test_states = np.array([
    [10.0,  0.0,  0.0,  0.0, 0.0,  0.0],  # Neutral Tape (Flat)
    [ 6.0,  5.0,  0.9,  0.8, 0.0,  0.0],  # Heavy Bid Pressure (Alpha Long Setup)
    [ 6.0, -5.0, -0.9, -0.8, 1.0, -8.0],  # Heavy Ask Pressure while in position (Exit Setup)
], dtype=np.float32)

obs_tensor = torch.as_tensor(test_states).to("cpu")
with torch.no_grad():
    dist = model.policy.get_distribution(obs_tensor)
    probs = dist.distribution.probs.numpy()

print("\n=== BINARY POLICY PROBABILITY DISTRIBUTION ===")
labels = ["Neutral Tape", "High Alpha Buy Setup", "Drawdown Exit Setup"]
for label, p in zip(labels, probs):
    print(f"\nState: {label}")
    print(f"  Target Flat (0): {p[0]*100:5.1f}% | Target Long (1): {p[1]*100:5.1f}%")
    print(f"  Action Decision: {'TARGET LONG' if np.argmax(p) == 1 else 'TARGET FLAT'}")
