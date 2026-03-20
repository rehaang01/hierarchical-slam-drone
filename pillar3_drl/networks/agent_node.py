
# ============================================================
# PILLAR 3: ROS2 Agent Deployment Node (Phase 3.5 - LSTM)
#
# Updated from PPO → RecurrentPPO (LSTM)
# Key changes from v1:
#   - Uses RecurrentPPO instead of PPO
#   - Maintains LSTM hidden state between steps
#   - Resets hidden state on new episodes
#   - Loads from checkpoints_lstm/best_model.zip
#
# Data flow:
#   Pillar 2 → /pillar2/sliced_map (OccupancyGrid)
#       ↓
#   agent_node.py (THIS FILE)
#       ↓ CNN extracts features → LSTM remembers history
#   /pillar3/waypoint (PoseStamped) → Pillar 5 (Nav2)
#   /pillar3/status  (String)       → Monitoring
#
# Run (inside Docker, after simulation is running):
#   source /opt/ros/humble/setup.bash
#   cd /root/pillar3_ws
#   python3 agent_node.py
# ============================================================

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import numpy as np
import torch
import os

from sb3_contrib import RecurrentPPO
from policy_network import CNNFeatureExtractor


# ============================================================
# ACTION → MOVEMENT MAPPING
# Must match exploration_env.py exactly
# ============================================================
ACTION_TO_DELTA = {
    0: (0,  1),   # UP    → +Y
    1: (0, -1),   # DOWN  → -Y
    2: (-1, 0),   # LEFT  → -X
    3: (1,  0),   # RIGHT → +X
}

GRID_RESOLUTION = 0.2   # 0.2m per cell — matches slicer_node.cpp
GRID_SIZE = 50          # 50x50 cells — matches slicer_node.cpp


class AgentNode(Node):
    """
    ROS2 node that runs the trained RecurrentPPO (LSTM) agent.
    Subscribes to Pillar 2's sliced map and publishes waypoints.
    """

    def __init__(self):
        super().__init__("pillar3_agent")
        self.get_logger().info("=== Pillar 3: Agent Node Starting (LSTM) ===")

        # ── LOAD MODEL ──────────────────────────────────────
        model_path = self._find_model()
        if model_path is None:
            self.get_logger().error(
                "No model found! Expected "
                "checkpoints_lstm/best_model.zip"
            )
            raise FileNotFoundError("No trained model found.")

        self.get_logger().info(f"Loading LSTM model from: {model_path}")
        self.model = RecurrentPPO.load(model_path, device="cpu")
        self.get_logger().info("LSTM Model loaded successfully.")

        # ── LSTM STATE ───────────────────────────────────────
        # This is the key addition over the PPO version.
        # lstm_states carries memory between timesteps.
        # None = let RecurrentPPO initialize on first call.
        self.lstm_states = None

        # episode_start tells LSTM to reset hidden state.
        # Shape (1,) = single environment (not vectorized).
        # Start as True so LSTM initializes cleanly on step 1.
        self.episode_start = np.ones((1,), dtype=bool)

        # ── REGULAR STATE ────────────────────────────────────
        self.latest_grid = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.has_odom = False
        self.step_count = 0

        # ── SUBSCRIBERS ──────────────────────────────────────
        self.grid_sub = self.create_subscription(
            OccupancyGrid,
            "/pillar2/sliced_map",
            self.grid_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/rtabmap/odom",
            self.odom_callback,
            10
        )

        # ── PUBLISHERS ───────────────────────────────────────
        self.waypoint_pub = self.create_publisher(
            PoseStamped,
            "/pillar3/waypoint",
            10
        )

        self.status_pub = self.create_publisher(
            String,
            "/pillar3/status",
            10
        )

        # ── TIMER: Run inference at 1 Hz ─────────────────────
        self.timer = self.create_timer(1.0, self.inference_callback)

        self.get_logger().info(
            "Agent Node ready. Waiting for /pillar2/sliced_map..."
        )

    # ============================================================
    # CALLBACK: Store latest grid from Pillar 2
    # ============================================================
    def grid_callback(self, msg: OccupancyGrid):
        self.latest_grid = msg

    # ============================================================
    # CALLBACK: Store drone position from odometry
    # ============================================================
    def odom_callback(self, msg: Odometry):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        self.has_odom = True

    # ============================================================
    # MAIN INFERENCE LOOP: Called every 1 second
    # ============================================================
    def inference_callback(self):
        if self.latest_grid is None:
            self.get_logger().info(
                "Waiting for /pillar2/sliced_map...",
                throttle_duration_sec=5
            )
            return

        if not self.has_odom:
            self.get_logger().info(
                "Waiting for /rtabmap/odom...",
                throttle_duration_sec=5
            )
            return

        # ── STEP 1: Convert OccupancyGrid → tensor ──────────
        obs_tensor = self._grid_to_tensor(self.latest_grid)
        if obs_tensor is None:
            return

        # ── STEP 2: Run LSTM inference ───────────────────────
        # Pass lstm_states so the model remembers past steps.
        # Pass episode_start so LSTM knows when to reset memory.
        # The returned new_lstm_states become input for next step.
        with torch.no_grad():
            action, self.lstm_states = self.model.predict(
                obs_tensor,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=False,
            )

        # After first step, episode is no longer starting
        self.episode_start = np.zeros((1,), dtype=bool)

        action_idx = int(action)
        dx_cells, dy_cells = ACTION_TO_DELTA[action_idx]

        # ── STEP 3: Convert action → world waypoint ──────────
        step_size_m = 1.0
        target_x = self.drone_x + (dx_cells * step_size_m)
        target_y = self.drone_y + (dy_cells * step_size_m)
        target_z = self.drone_z

        self.step_count += 1

        # ── STEP 4: Publish waypoint ─────────────────────────
        waypoint = PoseStamped()
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.header.frame_id = "map"
        waypoint.pose.position.x = target_x
        waypoint.pose.position.y = target_y
        waypoint.pose.position.z = target_z
        waypoint.pose.orientation.w = 1.0
        self.waypoint_pub.publish(waypoint)

        # ── STEP 5: Publish status ───────────────────────────
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        status_msg = String()
        status_msg.data = (
            f"Step:{self.step_count} | "
            f"Action:{action_names[action_idx]} | "
            f"Target:({target_x:.2f},{target_y:.2f},{target_z:.2f})"
        )
        self.status_pub.publish(status_msg)

        self.get_logger().info(
            f"Step {self.step_count}: "
            f"Action={action_names[action_idx]} | "
            f"Waypoint=({target_x:.2f}, {target_y:.2f}, {target_z:.2f})"
        )

    # ============================================================
    # HELPER: Reset LSTM memory (call when drone starts new mission)
    # ============================================================
    def reset_lstm(self):
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.step_count = 0
        self.get_logger().info("LSTM memory reset.")

    # ============================================================
    # HELPER: Convert OccupancyGrid → numpy tensor
    # ============================================================
    def _grid_to_tensor(self, msg: OccupancyGrid):
        width  = msg.info.width
        height = msg.info.height

        if width != GRID_SIZE or height != GRID_SIZE:
            self.get_logger().warn(
                f"Unexpected grid size: {width}x{height}, "
                f"expected {GRID_SIZE}x{GRID_SIZE}. Skipping."
            )
            return None

        grid_flat = np.array(msg.data, dtype=np.int8)
        grid_2d   = grid_flat.reshape((height, width))

        obs = np.zeros((height, width), dtype=np.float32)
        obs[grid_2d == 0]   = 0.5    # free
        obs[grid_2d == 100] = 1.0    # obstacle
        obs[grid_2d == 50]  = 0.75   # drone position
        # unknown (-1) stays 0.0

        # Shape: (1, 50, 50) — channel first, no batch dim
        obs_tensor = obs[np.newaxis, :, :]
        return obs_tensor

    # ============================================================
    # HELPER: Find trained model file
    # ============================================================
    def _find_model(self):
        search_paths = [
            "/root/pillar3_ws/checkpoints_lstm/best_model.zip",
            "/root/pillar3_ws/checkpoints_lstm/final_lstm_model.zip",
            "checkpoints_lstm/best_model.zip",
            "checkpoints_lstm/final_lstm_model.zip",
        ]
        for path in search_paths:
            if os.path.exists(path):
                return path
        return None


# ============================================================
# MAIN
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    try:
        node = AgentNode()
        rclpy.spin(node)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
EOF