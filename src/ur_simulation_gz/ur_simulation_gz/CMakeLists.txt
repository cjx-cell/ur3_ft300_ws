cmake_minimum_required(VERSION 3.5)
project(ur_simulation_gz)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(trajectory_msgs REQUIRED)

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

# ── peg_in_hole C++ demo ──
add_executable(peg_in_hole src/peg_in_hole.cpp)
ament_target_dependencies(peg_in_hole
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  shape_msgs
  moveit_msgs
  ament_index_cpp
)
target_compile_features(peg_in_hole PUBLIC cxx_std_17)
install(TARGETS peg_in_hole
  DESTINATION lib/${PROJECT_NAME})

# ── PAP-MoE peg_in_hole C++ demo ──
add_executable(pap_moe_peg_in_hole src/pap_moe_peg_in_hole.cpp)
ament_target_dependencies(pap_moe_peg_in_hole
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  std_srvs
  trajectory_msgs
  shape_msgs
  moveit_msgs
  ament_index_cpp
)
target_compile_features(pap_moe_peg_in_hole PUBLIC cxx_std_17)
install(TARGETS pap_moe_peg_in_hole
  DESTINATION lib/${PROJECT_NAME})

# Report the exact link pair responsible for MoveIt Servo self-collision scaling.
add_executable(pap_moe_self_collision_diagnostic src/pap_moe_self_collision_diagnostic.cpp)
ament_target_dependencies(pap_moe_self_collision_diagnostic
  rclcpp
  moveit_ros_planning_interface
)
target_compile_features(pap_moe_self_collision_diagnostic PUBLIC cxx_std_17)
install(TARGETS pap_moe_self_collision_diagnostic
  DESTINATION lib/${PROJECT_NAME})

# Default to off as starting gzsim doesn't shut down correctly at the moment
option(
  UR_SIM_INTEGRATION_TESTS
  "Run ur_simulation_gz integration tests"
  OFF
)

install(DIRECTORY config launch urdf worlds rviz models
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/pap_moe_keyboard_teleop.py
  scripts/pap_moe_keyboard_gui_teleop.py
  scripts/pap_moe_keyboard_mouse_teleop_v2.py
  scripts/pap_moe_teleop_motion.py
  DESTINATION lib/${PROJECT_NAME}
)
if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_pytest_test(
    test_pap_moe_teleop_motion
    test/test_pap_moe_teleop_motion.py
  )

  if(${UR_SIM_INTEGRATION_TESTS})
    add_launch_test(test/test_gz.py
      TIMEOUT
        180
    )
  endif()
endif()

ament_package()
