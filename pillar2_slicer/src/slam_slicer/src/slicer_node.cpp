// ============================================================
// PILLAR 2: Dynamic Z-Slicer + Z-Supervisor Node
// 
// What this node does every 1 second:
//   1. Reads drone's current altitude (Z) from /rtabmap/odom
//   2. Slices the OctoMap at altitude Z ± 0.5m
//   3. Builds a 3-channel grid:
//        Channel 1 (obstacles): 100 = wall/object
//        Channel 2 (free space): 50 = explored free area
//        Channel 3 (unknown):    0  = never seen
//   4. Counts frontiers (boundary between free and unknown)
//   5. If frontiers == 0: Z-Supervisor triggers altitude shift
//   6. Publishes the 2D grid for Pillar 3
// ============================================================

#include <rclcpp/rclcpp.hpp>
#include <octomap/ColorOcTree.h>
// Message types we subscribe to
#include <nav_msgs/msg/odometry.hpp>           // Drone pose (position + orientation)
#include <nav_msgs/msg/occupancy_grid.hpp>     // 2D grid map format
#include <octomap_msgs/msg/octomap.hpp>        // 3D OctoMap data
#include <geometry_msgs/msg/point.hpp>         // Simple x,y,z point
#include <std_msgs/msg/float32.hpp>            // For publishing altitude command

// OctoMap C++ library for actually reading the 3D tree
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>

// Standard C++ libraries
#include <memory>    // For smart pointers (shared_ptr)
#include <string>
#include <vector>
#include <cmath>     // For math functions like fabs()

// ============================================================
// NODE CLASS DEFINITION
// In ROS2, every node is a C++ class that inherits from rclcpp::Node
// ============================================================
class SlicerNode : public rclcpp::Node
{
public:
  // Constructor: called once when the node starts
  SlicerNode() : Node("slicer_node")
  {
    RCLCPP_INFO(this->get_logger(), "=== Pillar 2: Slicer Node Starting ===");

    // ── PARAMETERS ──
    // These are settings you can change without recompiling
    // Slice thickness: how many meters above/below drone to include
    this->declare_parameter("slice_thickness", 0.5);
    // Grid resolution: size of each cell in meters
    this->declare_parameter("grid_resolution", 0.2);
    // Grid size: how many cells wide/tall (50 = 10m x 10m at 0.2m resolution)
    this->declare_parameter("grid_size", 50);
    // Frontier threshold: minimum frontiers before Z-Supervisor activates
    this->declare_parameter("frontier_threshold", 5);
    // How many meters to shift altitude when floor is complete
    this->declare_parameter("altitude_shift", 2.0);

    // Load the parameter values into member variables
    slice_thickness_ = this->get_parameter("slice_thickness").as_double();
    grid_resolution_ = this->get_parameter("grid_resolution").as_double();
    grid_size_       = this->get_parameter("grid_size").as_int();
    frontier_threshold_ = this->get_parameter("frontier_threshold").as_int();
    altitude_shift_  = this->get_parameter("altitude_shift").as_double();

    // ── SUBSCRIBERS ──
    // Subscribe to the drone's odometry (position + velocity)
    // The "10" means: keep up to 10 messages in queue if processing is slow
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/rtabmap/odom", 10,
      std::bind(&SlicerNode::odomCallback, this, std::placeholders::_1));

    // Subscribe to the full 3D OctoMap
    octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
      "/rtabmap/octomap_full", 10,
      std::bind(&SlicerNode::octomapCallback, this, std::placeholders::_1));

    // ── PUBLISHERS ──
    // Publish the 2D sliced grid for Pillar 3 to consume
    grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/pillar2/sliced_map", 10);

    // Publish altitude command when Z-Supervisor activates
    altitude_cmd_pub_ = this->create_publisher<std_msgs::msg::Float32>(
      "/pillar2/altitude_command", 10);

    // ── TIMER ──
    // Process and publish at 1 Hz (once per second)
    // Pillar 3 (DRL) runs at 1-5 Hz so this matches
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&SlicerNode::timerCallback, this));

    // Initialize drone altitude to a safe default
    current_altitude_ = 1.5;
    has_octomap_ = false;
    has_odom_ = false;

    RCLCPP_INFO(this->get_logger(), "Slicer Node ready. Waiting for OctoMap and Odometry...");
  }

private:
  // ============================================================
  // CALLBACK 1: Odometry — called every time drone pose updates
  // We just store the current altitude for use in timerCallback
  // ============================================================
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // msg->pose.pose.position.z is the drone's height above ground
    current_altitude_ = msg->pose.pose.position.z;
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;
    has_odom_ = true;
  }

  // ============================================================
  // CALLBACK 2: OctoMap — called when 3D map updates
  // We store the latest map for slicing
  // ============================================================
  void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Received Octomap message. id=%s",
              msg->id.c_str());

  octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);

  if (!tree) {
    RCLCPP_WARN(this->get_logger(), "msgToMap returned NULL");
    return;
  }

  // Case 1: normal OcTree
  if (auto oc = dynamic_cast<octomap::OcTree*>(tree)) {
    octomap_.reset(oc);
    has_octomap_ = true;
    RCLCPP_INFO(this->get_logger(), "Stored OcTree directly");
    return;
  }

  // Case 2: ColorOcTree → convert
  if (auto color_tree = dynamic_cast<octomap::ColorOcTree*>(tree)) {

    RCLCPP_INFO(this->get_logger(), "Converting ColorOcTree → OcTree");

    double res = color_tree->getResolution();

    auto converted = std::make_shared<octomap::OcTree>(res);

    for (auto it = color_tree->begin_leafs();
         it != color_tree->end_leafs(); ++it)
    {
      bool occupied = it->getOccupancy() > 0.5;

      converted->updateNode(
        octomap::point3d(it.getX(), it.getY(), it.getZ()),
        occupied);
    }

    octomap_ = converted;
    has_octomap_ = true;

    delete color_tree;

    RCLCPP_INFO(this->get_logger(), "Conversion complete");
    return;
  }

  RCLCPP_WARN(this->get_logger(),
              "Unknown Octomap type: %s",
              typeid(*tree).name());

  delete tree;
}

  // ============================================================
  // MAIN FUNCTION: Called every 1 second by the timer
  // This is where all the slicing logic happens
  // ============================================================
  void timerCallback()
  {
    // Safety check: don't process if we don't have data yet
    if (!has_octomap_ || !has_odom_) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "Waiting for data... OctoMap: %s, Odometry: %s",
        has_octomap_ ? "YES" : "NO",
        has_odom_ ? "YES" : "NO");
      return;
    }

    // ── STEP A: Define the altitude slice bounds ──
    double z_min = current_altitude_ - slice_thickness_;
    double z_max = current_altitude_ + slice_thickness_;

    RCLCPP_DEBUG(this->get_logger(), 
      "Slicing at Z=%.2f (range: %.2f to %.2f)", 
      current_altitude_, z_min, z_max);

    // ── STEP B: Create the output 2D grid ──
    // OccupancyGrid is a standard ROS2 message for 2D maps
    // Values: -1=unknown, 0=free, 1-100=occupied
    auto grid_msg = std::make_shared<nav_msgs::msg::OccupancyGrid>();
    
    // Fill in the metadata
    grid_msg->header.stamp = this->now();
    grid_msg->header.frame_id = "map";  // Same frame as RTAB-Map
    grid_msg->info.resolution = grid_resolution_;  // 0.2m per cell
    grid_msg->info.width  = grid_size_;   // 50 cells wide
    grid_msg->info.height = grid_size_;   // 50 cells tall
    
    // Center the grid on the drone's current position
    grid_msg->info.origin.position.x = current_x_ - (grid_size_ * grid_resolution_) / 2.0;
    grid_msg->info.origin.position.y = current_y_ - (grid_size_ * grid_resolution_) / 2.0;
    grid_msg->info.origin.position.z = 0.0;
    
    // Initialize all cells as unknown (-1)
    int total_cells = grid_size_ * grid_size_;
    grid_msg->data.assign(total_cells, -1);  // -1 = unknown

    // ── STEP C: Iterate through OctoMap and fill the grid ──
    // This is the core slicing operation
    int occupied_count = 0;
    int free_count = 0;

    // Loop through every leaf node in the OctoMap
    // A "leaf" is the smallest voxel (no children)
    for (auto it = octomap_->begin_leafs(); it != octomap_->end_leafs(); ++it)
    {
      // Get this voxel's 3D position
      double vox_x = it.getX();
      double vox_y = it.getY();
      double vox_z = it.getZ();

      // Only process voxels within our altitude slice
      if (vox_z < z_min || vox_z > z_max) continue;

      // Convert 3D world position to 2D grid cell index
      // Formula: cell = (world_pos - grid_origin) / resolution
      int cell_x = static_cast<int>(
        (vox_x - grid_msg->info.origin.position.x) / grid_resolution_);
      int cell_y = static_cast<int>(
        (vox_y - grid_msg->info.origin.position.y) / grid_resolution_);

      // Skip if outside our grid bounds
      if (cell_x < 0 || cell_x >= grid_size_ || 
          cell_y < 0 || cell_y >= grid_size_) continue;

      // Convert 2D (x,y) to 1D array index
      // Row-major order: index = y * width + x
      int index = cell_y * grid_size_ + cell_x;

      // Check occupancy probability
      // getOccupancy() returns probability 0.0 to 1.0
      // > 0.5 means "probably occupied" (wall/obstacle)
      // < 0.5 means "probably free"
      if (it->getOccupancy() > 0.5) {
        grid_msg->data[index] = 100;  // OCCUPIED (Channel 1: obstacle)
        occupied_count++;
      } else {
        // Only mark as free if not already marked occupied
        // (occupied takes priority)
        if (grid_msg->data[index] != 100) {
          grid_msg->data[index] = 0;  // FREE (Channel 2: explored)
          free_count++;
        }
      }
    }

    // ── STEP D: Mark drone's current position on the grid ──
    // This is "Channel 3" — trajectory context
    int drone_cell_x = grid_size_ / 2;  // Center of grid (we centered on drone)
    int drone_cell_y = grid_size_ / 2;
    int drone_index = drone_cell_y * grid_size_ + drone_cell_x;
    grid_msg->data[drone_index] = 50;  // 50 = drone position marker

    // ── STEP E: Count frontiers ──
    // A frontier = free cell that has at least one unknown (-1) neighbor
    int frontier_count = countFrontiers(grid_msg->data);

    // Log the current state
    RCLCPP_INFO(this->get_logger(),
      "Slice stats | Alt: %.1fm | Occupied: %d | Free: %d | Frontiers: %d",
      current_altitude_, occupied_count, free_count, frontier_count);

    // ── STEP F: Z-Supervisor Logic ──
    // If frontiers are below threshold, current floor is complete
    if (frontier_count < frontier_threshold_) {
      RCLCPP_WARN(this->get_logger(),
        "FRONTIER STARVATION DETECTED! Only %d frontiers. Triggering Z-Supervisor...",
        frontier_count);
      
      triggerAltitudeShift();
    }

    // ── STEP G: Publish the grid ──
    grid_pub_->publish(*grid_msg);
  }

  // ============================================================
  // HELPER: Count frontier cells in the grid
  // Frontier = free cell (0) adjacent to unknown cell (-1)
  // ============================================================
  int countFrontiers(const std::vector<int8_t>& grid)
  {
    int frontier_count = 0;

    // Check every cell in the grid
    for (int y = 1; y < grid_size_ - 1; y++) {
      for (int x = 1; x < grid_size_ - 1; x++) {
        
        int idx = y * grid_size_ + x;
        
        // Only free cells can be frontiers
        if (grid[idx] != 0) continue;

        // Check all 4 neighbors (up, down, left, right)
        // If any neighbor is unknown (-1), this cell is a frontier
        bool is_frontier = false;
        
        // Up neighbor
        if (grid[(y+1) * grid_size_ + x] == -1) is_frontier = true;
        // Down neighbor  
        if (grid[(y-1) * grid_size_ + x] == -1) is_frontier = true;
        // Right neighbor
        if (grid[y * grid_size_ + (x+1)] == -1) is_frontier = true;
        // Left neighbor
        if (grid[y * grid_size_ + (x-1)] == -1) is_frontier = true;

        if (is_frontier) frontier_count++;
      }
    }

    return frontier_count;
  }

  // ============================================================
  // Z-SUPERVISOR: Commands drone to shift altitude
  // Called when frontier count drops to zero
  // ============================================================
  void triggerAltitudeShift()
  {
    // Check if there's unexplored space above the drone
    double probe_z_above = current_altitude_ + altitude_shift_;
    double probe_z_below = current_altitude_ - altitude_shift_;
    
    // Count unknown voxels above
    int unknown_above = countUnknownVoxelsAtAltitude(probe_z_above);
    int unknown_below = countUnknownVoxelsAtAltitude(probe_z_below);

    RCLCPP_INFO(this->get_logger(),
      "Bidirectional Probe | Unknown above: %d | Unknown below: %d",
      unknown_above, unknown_below);

    // Choose direction with more unknown space
    double target_altitude;
    if (unknown_above >= unknown_below && probe_z_above > 0.3) {
      target_altitude = probe_z_above;
      RCLCPP_WARN(this->get_logger(), 
        "Z-Supervisor: Commanding ASCENT to %.1fm", target_altitude);
    } else if (unknown_below > 0 && probe_z_below > 0.3) {
      target_altitude = probe_z_below;
      RCLCPP_WARN(this->get_logger(),
        "Z-Supervisor: Commanding DESCENT to %.1fm", target_altitude);
    } else {
      RCLCPP_INFO(this->get_logger(), 
        "Z-Supervisor: No unexplored layers found. Environment fully mapped!");
      return;
    }

    // Publish the altitude command
    // Pillar 5 (Nav2) will listen to this and execute the movement
    auto alt_msg = std_msgs::msg::Float32();
    alt_msg.data = static_cast<float>(target_altitude);
    altitude_cmd_pub_->publish(alt_msg);
  }

  // ============================================================
  // HELPER: Count unknown voxels at a given altitude
  // Used by Z-Supervisor to decide which direction to go
  // ============================================================
  int countUnknownVoxelsAtAltitude(double target_z)
  {
    // We check a thin band at the target altitude
    double z_min = target_z - 0.3;
    double z_max = target_z + 0.3;
    
    int unknown_count = 0;
    double resolution = octomap_->getResolution();

    // Sample a grid of points at the target altitude
    // and check which ones have no OctoMap data (= unknown)
    for (double x = current_x_ - 5.0; x < current_x_ + 5.0; x += resolution) {
      for (double y = current_y_ - 5.0; y < current_y_ + 5.0; y += resolution) {
        
        octomap::OcTreeNode* node = octomap_->search(x, y, target_z);
        
        // If search returns nullptr, the voxel is unknown (never observed)
        if (node == nullptr) {
          unknown_count++;
        }
      }
    }

    return unknown_count;
  }

  // ============================================================
  // MEMBER VARIABLES
  // These store state between callbacks and timer ticks
  // ============================================================
  
  // ROS2 communication objects
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr altitude_cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Stored data
  std::shared_ptr<octomap::OcTree> octomap_;  // The 3D map
  double current_altitude_;                    // Drone's current Z
  double current_x_, current_y_;              // Drone's X, Y position
  bool has_octomap_;                          // Flag: do we have map data?
  bool has_odom_;                             // Flag: do we have pose data?

  // Parameters
  double slice_thickness_;
  double grid_resolution_;
  int    grid_size_;
  int    frontier_threshold_;
  double altitude_shift_;
};

// ============================================================
// MAIN: Entry point — creates and spins the node
// ============================================================
int main(int argc, char* argv[])
{
  // Initialize ROS2
  rclcpp::init(argc, argv);

  // Create our node and keep it running
  // spin() blocks here and processes callbacks until Ctrl+C
  rclcpp::spin(std::make_shared<SlicerNode>());

  // Clean shutdown
  rclcpp::shutdown();
  return 0;
}
