cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gz)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)

# ── pick_and_place C++ demo ──
add_executable(pick_and_place src/pick_and_place.cpp)
ament_target_dependencies(pick_and_place
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  shape_msgs
  moveit_msgs
  ament_index_cpp
)
target_compile_features(pick_and_place PUBLIC cxx_std_17)
install(TARGETS pick_and_place
  DESTINATION lib/${PROJECT_NAME})

# Default to off as starting gzsim doesn't shut down correctly at the moment
option(
  UR_SIM_INTEGRATION_TESTS
  "Run ur_simulation_gz integration tests"
  OFF
)

install(DIRECTORY config launch urdf worlds rviz
  DESTINATION share/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake)
  find_package(ament_cmake_pytest REQUIRED)

  if(${UR_SIM_INTEGRATION_TESTS})
    add_launch_test(test/test_gz.py
      TIMEOUT
        180
    )
  endif()
endif()

ament_package()
