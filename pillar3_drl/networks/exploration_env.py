# ============================================================
# PILLAR 3: Gymnasium Exploration Environment (v2)
#
# Key fixes from v1:
#   1. DISCRETE action space (UP/DOWN/LEFT/RIGHT)
#      Continuous (dx,dy) caused int() truncation to zero —
#      the agent was frozen in place most of the time.
#   2. Smaller grid (20x20) for faster early learning.
#      We scale back to 50x50 after training is verified.
#   3. Simplified reward: exploration only + small collision.
#   4. Removed revisit penalty (was dominating reward signal).
#
# Grid cell values (matching Pillar 2 exactly):
#   -1  = unknown
#    0  = free (explored)
#   50  = drone position
#  100  = obstacle
# ============================================================

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExplorationEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Discrete action map: index → (dx, dy)
    ACTIONS = {
        0: (0,  1),   # UP
        1: (0, -1),   # DOWN
        2: (-1, 0),   # LEFT
        3: (1,  0),   # RIGHT
    }

    def __init__(self, grid_size=20, max_steps=300,
                 obstacle_density=0.10, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.render_mode = render_mode
        self.sensor_radius = 3    # Sees 3 cells around itself

        # ── ACTION SPACE ──────────────────────────────────
        # 4 discrete directions. Simple, unambiguous, easy to learn.
        # The agent always moves exactly 1 cell per step.
        self.action_space = spaces.Discrete(4)

        # ── OBSERVATION SPACE ─────────────────────────────
        # Single-channel normalized grid (1, H, W)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(1, self.grid_size, self.grid_size),
            dtype=np.float32
        )

        # Internal state (initialized in reset)
        self.grid = None
        self.observed = None
        self.drone_x = None
        self.drone_y = None
        self.step_count = 0
        self.total_free_cells = 0

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self._generate_world()
        self.observed = np.full(
            (self.grid_size, self.grid_size), -1, dtype=np.int8
        )

        # Spawn drone at center, guaranteed free
        self.drone_x = self.grid_size // 2
        self.drone_y = self.grid_size // 2
        self.grid[self.drone_y, self.drone_x] = 0

        self.step_count = 0
        self.total_free_cells = max(int(np.sum(self.grid == 0)), 1)

        self._update_observed()

        return self._get_observation(), {}

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):
        self.step_count += 1

        dx, dy = self.ACTIONS[int(action)]
        new_x = int(np.clip(self.drone_x + dx, 1, self.grid_size - 2))
        new_y = int(np.clip(self.drone_y + dy, 1, self.grid_size - 2))

        collision = (self.grid[new_y, new_x] == 100)

        # Count unknown before move
        unknown_before = int(np.sum(self.observed == -1))

        if not collision:
            self.drone_x = new_x
            self.drone_y = new_y

        self._update_observed()

        unknown_after = int(np.sum(self.observed == -1))

        # ── REWARD ──────────────────────────────────────
        new_cells = unknown_before - unknown_after
        reward = float(new_cells)           # +1 per newly explored cell
        if collision:
            reward -= 0.5                   # Small collision penalty

        # ── TERMINATION ─────────────────────────────────
        revealed = int(np.sum(self.observed == 0))
        coverage = revealed / self.total_free_cells

        terminated = bool(coverage >= 0.90)
        truncated  = bool(self.step_count >= self.max_steps)

        info = {"coverage": coverage, "step": self.step_count}
        return self._get_observation(), reward, terminated, truncated, info

    # ============================================================
    # SENSOR: Reveal cells within radius using line-of-sight
    # ============================================================
    def _update_observed(self):
        r = self.sensor_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy > r*r:
                    continue
                cx = self.drone_x + dx
                cy = self.drone_y + dy
                if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                    if self._has_los(self.drone_x, self.drone_y, cx, cy):
                        self.observed[cy, cx] = self.grid[cy, cx]
        self.observed[self.drone_y, self.drone_x] = 0

    def _has_los(self, x0, y0, x1, y1):
        dx, dy = abs(x1-x0), abs(y1-y0)
        x, y = x0, y0
        n = 1 + dx + dy
        xi = 1 if x1 > x0 else -1
        yi = 1 if y1 > y0 else -1
        err = dx - dy
        dx *= 2
        dy *= 2
        for _ in range(n - 1):
            if self.grid[y, x] == 100:
                return False
            if err > 0:
                x += xi; err -= dy
            else:
                y += yi; err += dx
        return True

    # ============================================================
    # OBSERVATION: Normalized float tensor (1, H, W)
    # ============================================================
    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obs[self.observed == 0]   = 0.5    # free
        obs[self.observed == 100] = 1.0    # obstacle
        # unknown stays 0.0
        obs[self.drone_y, self.drone_x] = 0.75  # drone
        return obs[np.newaxis, :, :]       # (1, H, W)

    # ============================================================
    # WORLD GENERATOR
    # ============================================================
    def _generate_world(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Boundary walls
        grid[0, :] = 100; grid[-1, :] = 100
        grid[:, 0] = 100; grid[:, -1] = 100

        # Random interior obstacles
        rng = np.random.default_rng(self.np_random.integers(0, 2**31))
        n_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
        for _ in range(n_obstacles):
            ox = rng.integers(2, self.grid_size - 2)
            oy = rng.integers(2, self.grid_size - 2)
            size = rng.integers(1, 3)
            grid[oy:oy+size, ox:ox+size] = 100

        # Clear 3x3 spawn area at center
        cx, cy = self.grid_size // 2, self.grid_size // 2
        grid[cy-1:cy+2, cx-1:cx+2] = 0

        return grid

    # ============================================================
    # RENDER
    # ============================================================
    def render(self):
        if self.render_mode is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        img[self.observed == -1] = [0.3, 0.3, 0.3]
        img[self.observed == 0]  = [1.0, 1.0, 1.0]
        img[self.observed == 100] = [0.0, 0.0, 0.0]
        img[self.drone_y, self.drone_x] = [1.0, 0.0, 0.0]

        if self.render_mode == "rgb_array":
            return (img * 255).astype(np.uint8)
        if self.render_mode == "human":
            plt.figure(1); plt.clf()
            plt.imshow(img, origin="upper")
            revealed = int(np.sum(self.observed == 0))
            cov = revealed / self.total_free_cells
            plt.title(f"Step: {self.step_count} | Coverage: {cov:.1%}")
            plt.pause(0.01)

    def close(self):
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass