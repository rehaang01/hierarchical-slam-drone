# ============================================================
# PILLAR 3: PPO Training Script (v2)
#
# Changes from v1:
#   - Uses SB3's built-in CnnPolicy with our CNNFeatureExtractor
#   - Discrete action space (4 directions)
#   - 20x20 grid for fast early training
#   - Tuned hyperparameters for discrete grid-world tasks
# ============================================================

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.vec_env import VecMonitor
from exploration_env import ExplorationEnv
from policy_network import CNNFeatureExtractor


# ============================================================
# PROGRESS CALLBACK
# ============================================================
class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=5000):
        super().__init__()
        self.print_freq = print_freq
        self.ep_rewards = []
        self.ep_coverages = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
            if "coverage" in info:
                self.ep_coverages.append(info["coverage"])

        if self.num_timesteps % self.print_freq == 0 and self.ep_rewards:
            mean_r = np.mean(self.ep_rewards[-100:])
            mean_c = np.mean(self.ep_coverages[-100:]) if self.ep_coverages else 0.0
            print(
                f"Step: {self.num_timesteps:>8} | "
                f"Mean Reward: {mean_r:>8.2f} | "
                f"Mean Coverage: {mean_c:.1%}"
            )
        return True


# ============================================================
# TRAIN
# ============================================================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("=" * 60)
    print("PILLAR 3: PPO Training (v2 - Discrete Actions)")
    print("=" * 60)

    n_envs = 8  # More parallel envs → faster data collection

    print(f"Creating {n_envs} parallel environments (20x20 grid)...")

    vec_env = make_vec_env(
        ExplorationEnv, n_envs=n_envs,
        env_kwargs={"grid_size": 50, "max_steps": 1000}
    )
    vec_env = VecMonitor(vec_env)

    eval_env = make_vec_env(
        ExplorationEnv, n_envs=1,
        env_kwargs={"grid_size": 50, "max_steps": 1000}
    )
    eval_env = VecMonitor(eval_env)

    print("Environments ready.")

    # ── HYPERPARAMETERS ───────────────────────────────────
    # Tuned for discrete grid-world navigation.
    # learning_rate: 3e-4 is standard for PPO discrete tasks
    # n_steps: 512 per env × 8 envs = 4096 per update (good batch)
    # ent_coef: 0.01 encourages action diversity early on
    # gamma: 0.99 standard
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=0,
        tensorboard_log="logs/",
        device="auto",
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "features_extractor_class": CNNFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[128], vf=[128]),
        },
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model built. Parameters: {total_params:,}")
    print(f"Device: {model.device}")

    # ── CALLBACKS ─────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/",
        log_path="logs/",
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50000,
        save_path="checkpoints/",
        name_prefix="ppo_v2",
    )
    prog_cb = ProgressCallback(print_freq=5000)

    # ── TRAINING ─────────────────────────────────────────
    total_timesteps = 1_000_000

    print(f"\nTraining for {total_timesteps:,} steps...")
    print(f"Print every 5,000 steps. Eval every 10,000 steps.")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, ckpt_cb, prog_cb],
        reset_num_timesteps=True,
    )

    model.save("checkpoints/final_model")
    print("-" * 60)
    print("Training complete! Final model: checkpoints/final_model.zip")

    # ── FINAL EVAL ────────────────────────────────────────
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Final Mean Reward: {mean_r:.2f} +/- {std_r:.2f}")


if __name__ == "__main__":
    train()