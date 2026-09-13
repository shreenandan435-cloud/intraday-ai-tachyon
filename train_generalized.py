import glob
import pyarrow.parquet as pq
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from multi_tape_env import MultiTapeL2Env

def get_verified_parquet_files(pattern: str) -> list[str]:
    raw_files = glob.glob(pattern)
    verified = []
    print(f"[*] Auditing {len(raw_files)} depth files for Parquet integrity...")
    for f in raw_files:
        try:
            pq.ParquetFile(f)
            verified.append(f)
        except Exception:
            continue
    print(f"[✓] {len(verified)} valid files verified.")
    return verified

if __name__ == "__main__":
    valid_files = get_verified_parquet_files(r"data\ticks\depth\*\*\*.parquet")
    split = int(len(valid_files) * 0.8)
    train_files = valid_files[:split]

    env = DummyVecEnv([lambda: MultiTapeL2Env(train_files, max_episode_steps=2048)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.04,  # Active exploration to break paralysis
        gamma=0.98,
        device="cpu",
        verbose=1,
        tensorboard_log="./tensorboard_multi/"
    )

    print("[*] Commencing MtM dense-reward multi-asset training (200,000 steps)...")
    model.learn(total_timesteps=200_000)
    
    model.save("models/tachyon_brain_generalized")
    print("[✓] Generalized model weights saved to models/tachyon_brain_generalized.zip")
