"""
Phase 3 Smoke Test
==================
Run this FIRST before any training.

Tests:
  1. Can Python receive /pillar2/sliced_map?
  2. Is the observation tensor the right shape?
  3. Are the values sensible (not all garbage)?

Run inside Docker container:
    source /opt/ros/humble/setup.bash
    python3 /root/pillar3_drl/smoke_test.py
    
You should see the simulation running in another terminal.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
import time


class ObsChecker(Node):
    def __init__(self):
        super().__init__("obs_checker")
        self.received = 0
        self.create_subscription(
            OccupancyGrid,
            "/pillar2/sliced_map",
            self.callback,
            10,
        )
        self.get_logger().info("Waiting for /pillar2/sliced_map...")

    def callback(self, msg: OccupancyGrid):
        self.received += 1

        # Convert to numpy array
        raw = np.array(msg.data, dtype=np.int8)

        # Count each type of cell
        n_unknown  = np.sum(raw == -1)
        n_free     = np.sum(raw == 0)
        n_occupied = np.sum(raw == 100)
        n_drone    = np.sum(raw == 50)
        total      = len(raw)

        print(f"\n{'='*50}")
        print(f"  Message #{self.received}")
        print(f"{'='*50}")
        print(f"  Grid size   : {msg.info.width} x {msg.info.height} = {total} cells")
        print(f"  Resolution  : {msg.info.resolution:.2f}m per cell")
        print(f"  Grid covers : {msg.info.width * msg.info.resolution:.1f}m x "
              f"{msg.info.height * msg.info.resolution:.1f}m")
        print(f"  ---")
        print(f"  Unknown (-1): {n_unknown:4d}  ({100*n_unknown/total:.1f}%)")
        print(f"  Free    ( 0): {n_free:4d}  ({100*n_free/total:.1f}%)")
        print(f"  Occupied(100): {n_occupied:4d}  ({100*n_occupied/total:.1f}%)")
        print(f"  Drone   ( 50): {n_drone:4d}  ← should always be 1")
        print(f"  ---")
        print(f"  Grid origin : ({msg.info.origin.position.x:.2f}, "
              f"{msg.info.origin.position.y:.2f})")

        # Shape check
        assert raw.shape == (2500,), f"Expected 2500 cells, got {raw.shape}"
        assert n_drone == 1, f"Expected exactly 1 drone cell, got {n_drone}"
        print(f"  Shape check : PASS")
        print(f"  Drone check : PASS")

        # Convert to 3-channel tensor (what the CNN sees)
        obs = np.full((3, 50, 50), -1.0, dtype=np.float32)
        grid_2d = raw.reshape(50, 50)
        obs[0][grid_2d == 100] = 1.0    # obstacles
        obs[1][grid_2d == 0]   = 0.0    # free space
        obs[2][grid_2d == 50]  = 0.5    # drone position

        print(f"  CNN input shape: {obs.shape}  (3 channels × 50 × 50)")
        print(f"  Min: {obs.min():.1f}  Max: {obs.max():.1f}  ← should be -1.0 to 1.0")
        print(f"{'='*50}")

        if self.received >= 3:
            print("\n✅  PHASE 3 OBSERVATION PIPELINE IS WORKING!")
            print("    The CNN will receive tensors in this exact format.")
            print("    You can now proceed to training.\n")
            raise SystemExit(0)


def main():
    rclpy.init()
    node = ObsChecker()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
