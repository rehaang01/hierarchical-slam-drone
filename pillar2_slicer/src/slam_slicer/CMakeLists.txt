cmake_minimum_required(VERSION 3.8)
project(slam_slicer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap_ros REQUIRED)

# System octomap C++ library (from liboctomap-dev)
find_package(octomap REQUIRED)

# Build executable
add_executable(slicer_node src/slicer_node.cpp)

ament_target_dependencies(slicer_node
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  octomap_msgs
  octomap_ros
)

if (TARGET octomap::octomap)
  # preferred modern target
  target_link_libraries(slicer_node octomap::octomap)
elseif (DEFINED OCTOMAP_LIBRARIES)
  # if FindOctomap provided variables (older style)
  target_link_libraries(slicer_node ${OCTOMAP_LIBRARIES})
else()
  # fallback: try to locate the library manually
  find_library(OCTOMAP_LIB NAMES octomap PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib NO_DEFAULT_PATH)
  if (OCTOMAP_LIB)
    message(STATUS "Found octomap library at ${OCTOMAP_LIB}")
    target_link_libraries(slicer_node ${OCTOMAP_LIB})
  else()
    message(FATAL_ERROR "Could not find octomap library. Ensure liboctomap-dev is installed.")
  endif()
endif()

install(TARGETS
  slicer_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()