
# ============================================================
# PILLAR 3: RecurrentPPO Training Script (Phase 3.5)
#
# Changes from train.py (PPO):
#   - Uses RecurrentPPO from sb3_contrib
#   - Uses CnnLstmPolicy instead of CnnPolicy
#   - LSTM gives agent memory of past positions
#   - Expected improvement: coverage 20-35% → 50-70%
#
# What stays identical to train.py:
#   - ExplorationEnv (no changes needed)
#   - CNNFeatureExtractor (no changes needed)
#   - All PPO hyperparameters
#   - 8 parallel environments
#   - 50x50 grid
# ============================================================

import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)

from exploration_env import ExplorationEnv
from policy_network import CNNFeatureExtractor

# ============================================================
# PROGRESS CALLBACK (same as train.py)
# ============================================================
class ProgressCallback(BaseCallback):
    def __init__(self, print_freq=5000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_coverages = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
            if "coverage" in info:
                self.episode_coverages.append(info["coverage"])

        if self.num_timesteps % self.print_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
            mean_coverage = np.mean(self.episode_coverages[-50:]) * 100 if self.episode_coverages else 0
            print(
                f"Step: {self.num_timesteps:>8,} | "
                f"Mean Reward: {mean_reward:>8.2f} | "
                f"Mean Coverage: {mean_coverage:.1f}%"
            )
        return True


# ============================================================
# MAIN TRAINING
# ============================================================
def main():
    print("=" * 60)
    print("PILLAR 3: RecurrentPPO Training (Phase 3.5 - LSTM)")
    print("=" * 60)

    # ── DIRECTORIES ─────────────────────────────────────────
    os.makedirs("checkpoints_lstm", exist_ok=True)
    # logs disabled — disk space

    # ── ENVIRONMENTS ────────────────────────────────────────
    # Same as train.py — 8 parallel envs, 50x50 grid
    n_envs = 8
    print(f"Creating {n_envs} parallel environments (50x50 grid)...")

    env = make_vec_env(ExplorationEnv, n_envs=n_envs, env_kwargs={"grid_size": 50, "max_steps": 1000})
    env = VecMonitor(env)

    eval_env = make_vec_env(ExplorationEnv, n_envs=1, env_kwargs={"grid_size": 50, "max_steps": 1000})
    eval_env = VecMonitor(eval_env)

    print("Environments ready.")

    # ── POLICY KWARGS ────────────────────────────────────────
    # Pass our CNN extractor to RecurrentPPO
    # LSTM size 256 matches our CNN output dimension
    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,      # LSTM hidden state size
        n_lstm_layers=1,           # Single LSTM layer (stable)
        shared_lstm=False,         # Separate LSTM for actor/critic
        enable_critic_lstm=True,   # Critic also uses LSTM
    )

    # ── MODEL ────────────────────────────────────────────────
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        # Same hyperparameters as train.py
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=0,
        device="auto",
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model built. Parameters: {total_params:,}")
    print(f"Device: {model.device}")

    # ── CALLBACKS ────────────────────────────────────────────
    progress_cb = ProgressCallback(print_freq=5000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints_lstm/",
        log_path=None,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=False,   # Keep False — no looping risk with LSTM
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="checkpoints_lstm/",
        name_prefix="lstm_checkpoint",
    )

    # ── TRAIN ────────────────────────────────────────────────
    total_timesteps = 1_000_000
    print(f"Training for {total_timesteps:,} steps...")
    print(f"Print every 5,000 steps. Eval every 10,000 steps.")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_cb, eval_cb, checkpoint_cb],
        reset_num_timesteps=True,
    )

    # ── SAVE FINAL ───────────────────────────────────────────
    model.save("checkpoints_lstm/final_lstm_model")
    print("=" * 60)
    print("Training complete!")
    print("Best model: checkpoints_lstm/best_model.zip")
    print("Final model: checkpoints_lstm/final_lstm_model.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
EOF