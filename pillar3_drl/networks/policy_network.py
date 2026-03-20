# ============================================================
# PILLAR 3: CNN Feature Extractor (v2)
#
# Architecture change from v1:
#   v1: Custom CNN + custom LSTM + custom ActorCriticPolicy
#       Problem: LSTM state not reset at episode boundaries
#       in VecEnv → corrupted gradients → policy collapse
#
#   v2: Custom CNN extractor + SB3's built-in CnnPolicy
#       SB3 handles all episode boundary logic correctly.
#       Once training is verified healthy, we add LSTM via
#       sb3-contrib's RecurrentPPO (Step 3.5).
#
# The CNN architecture is unchanged — same 3 conv layers.
# Only the policy wrapper changes.
# ============================================================

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    3-layer CNN that processes the 2D exploration grid.
    Input:  (batch, 1, H, W) normalized float tensor
    Output: (batch, 256) feature vector

    Registered with SB3 via policy_kwargs — SB3 plugs this
    directly into its Actor and Critic heads automatically.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # = 1

        self.cnn = nn.Sequential(
            # Layer 1: coarse spatial structure
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # H/2, W/2

            # Layer 2: room shapes, corridor widths
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # H/4, W/4

            # Layer 3: fine obstacle boundaries
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Always → (64, 4, 4) = 1024

            nn.Flatten(),
        )

        # Compute output size dynamically (safe for any grid_size)
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_out = self.cnn(sample).shape[1]

        self.head = nn.Sequential(
            nn.Linear(cnn_out, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(obs))