/**
 * UR3 + FT300 + Robotiq 2F85 pick-and-place demo (ROS 2 Humble / MoveIt2)
 *
 * Cartesian moves use computeCartesianPath() for true straight-line motion.
 * Pattern: getCurrentPose("tool0") → borrow orientation → computeCartesianPath
 *
 * Build & run:
 *   colcon build --symlink-install --packages-select ur_simulation_gz
 *   ros2 launch ur3_ft300_moveit_config move_group.launch.py   # Terminal 1
 *   ros2 run ur_simulation_gz pick_and_place                   # Terminal 2
 */

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace std::chrono_literals;

// ── Planning groups (match ur3_ft300_robotiq_2f85.srdf) ──

static const std::string ARM_GROUP     = "ur_manipulator";
static const std::string GRIPPER_GROUP = "gripper";

// Cartesian IK target link.  tool0 is the wrist flange — it's on the MAIN
// kinematic chain (base_link → 6 revolute joints → tool0).  The SRDF chain
// matches this: <chain base_link="base_link" tip_link="tool0"/>
static const std::string IK_LINK = "tool0";

// Standard UR named states for the gripper (defined in SRDF)
static const std::string GRIP_OPEN  = "open";   // knuckle = 0.1
static const std::string GRIP_CLOSE = "close";  // knuckle = 0.55 (ROS 1 value)

// ROS 1 home: arm bent at elbow, wrist folded, gripper pointing DOWN.
static const std::vector<double> READY_JOINTS = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};

// ── Target in WORLD coordinates ──
// base_link is at world z ≈ 0.76.  Shoulder at world z ≈ 0.91.
// Table surface at z = 0.775, block centre at z = 0.795.
// FT300 (0.0415m) + gripper base→fingertip (0.098m) ≈ 0.14m tool0→fingertip.
static constexpr double BLOCK_WORLD_Z    = 0.795;
static constexpr double TABLE_WORLD_Z    = 0.775;
static constexpr double FINGERTIP_OFFSET = 0.14;

static constexpr double ABOVE_DZ = 0.18;
static constexpr double GRASP_DZ = 0.06;

// tool0 world-z targets (fingertip = tool0_z - 0.14)
static constexpr double TOOL0_ABOVE_Z = BLOCK_WORLD_Z + ABOVE_DZ + FINGERTIP_OFFSET;  // 1.115
static constexpr double TOOL0_GRASP_Z = BLOCK_WORLD_Z + GRASP_DZ + FINGERTIP_OFFSET;  // 0.995
static constexpr double TOOL0_PLACE_Z = 1.03;  // user-specified
static constexpr double TOOL0_SAFE_Z  = 1.20;  // well above table, safe for joint moves

// Bowl position set from runtime parameters (bowl_x, bowl_y)
// Default: x=-0.15, y=0.35

// ── Helpers ──

static void log_info(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_INFO(l, "%s", msg.c_str()); }

static void log_error(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_ERROR(l, "%s", msg.c_str()); }

// Joint-space move (for READY, gripper commands)
static bool try_move(moveit::planning_interface::MoveGroupInterface& mgi,
                     const rclcpp::Logger& logger,
                     const std::string& desc)
{
  auto r = mgi.move();
  if (r) return true;
  log_error(logger, "  FAILED: " + desc + " (code " +
            std::to_string(static_cast<int>(r.val)) + ")");
  return false;
}

// Cartesian straight-line move: interpolates waypoints along the line
// from current pose to target, computes IK at each step, and executes.
static bool try_cartesian(moveit::planning_interface::MoveGroupInterface& mgi,
                          const rclcpp::Logger& logger,
                          const geometry_msgs::msg::Pose& target,
                          const std::string& ik_link,
                          const std::string& desc)
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  auto current = mgi.getCurrentPose(ik_link);
  waypoints.push_back(current.pose);
  waypoints.push_back(target);

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double eef_step = 0.01;       // 1 cm resolution
  const double jump_threshold = 0.0;  // no jumps allowed
  double fraction = mgi.computeCartesianPath(waypoints, eef_step,
                                             jump_threshold, trajectory);

  if (fraction < 0.99) {
    log_error(logger, "  " + desc + " Cartesian path only " +
              std::to_string(static_cast<int>(fraction * 100)) + "%");
    return false;
  }
  auto r = mgi.execute(trajectory);
  if (r) return true;
  log_error(logger, "  FAILED: " + desc + " (code " +
            std::to_string(static_cast<int>(r.val)) + ")");
  return false;
}

// ── Main ──

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "pick_and_place",
      rclcpp::NodeOptions().parameter_overrides(
          std::vector{rclcpp::Parameter("use_sim_time", true)}));
  auto logger = node->get_logger();

  // Block & bowl positions from parameters (overridable via --ros-args -p ...)
  node->declare_parameter<double>("block_x", 0.20);
  node->declare_parameter<double>("block_y", 0.35);
  node->declare_parameter<double>("bowl_x",  -0.15);
  node->declare_parameter<double>("bowl_y",   0.35);
  double block_world_x = node->get_parameter("block_x").as_double();
  double block_world_y = node->get_parameter("block_y").as_double();
  double bowl_world_x  = node->get_parameter("bowl_x").as_double();
  double bowl_world_y  = node->get_parameter("bowl_y").as_double();
  RCLCPP_INFO(logger, "Block: [%.3f %.3f]  Bowl: [%.3f %.3f]",
              block_world_x, block_world_y, bowl_world_x, bowl_world_y);

  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  executor->add_node(node);
  auto spin = std::thread([executor] { executor->spin(); });

  // ── Load robot model (URDF + SRDF) from MoveIt config ──
  {
    std::string pkg = ament_index_cpp::get_package_share_directory("ur3_ft300_moveit_config");
    std::string cfg = pkg + "/config/";

    // SRDF
    std::ifstream srf(cfg + "ur3_ft300_robotiq_2f85.srdf");
    if (!srf.is_open()) { RCLCPP_ERROR(logger, "Cannot open SRDF"); return 1; }
    std::stringstream ss; ss << srf.rdbuf();
    try { node->declare_parameter<std::string>("robot_description_semantic", ""); } catch (...) {}
    node->set_parameter(rclcpp::Parameter("robot_description_semantic", ss.str()));
    RCLCPP_INFO(logger, "Loaded SRDF");

    // URDF (xacro → plain XML)
    std::string xacro_path = cfg + "ur3_ft300_robotiq_2f85.urdf.xacro";
    std::string cmd = "xacro " + xacro_path + " 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) { RCLCPP_ERROR(logger, "xacro failed"); return 1; }
    std::string urdf;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe.get())) urdf += buf;
    if (urdf.empty() || urdf.find("<?xml") == std::string::npos) {
      RCLCPP_ERROR(logger, "xacro output invalid (%zu B)", urdf.size()); return 1;
    }
    try { node->declare_parameter<std::string>("robot_description", ""); } catch (...) {}
    node->set_parameter(rclcpp::Parameter("robot_description", urdf));
    RCLCPP_INFO(logger, "Loaded URDF (%zu bytes)", urdf.size());
  }

  std::this_thread::sleep_for(2s);

  // ── MoveGroup interfaces ──
  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface arm(node, ARM_GROUP);
  MoveGroupInterface gripper(node, GRIPPER_GROUP);

  arm.setMaxVelocityScalingFactor(1.0);
  arm.setMaxAccelerationScalingFactor(1.0);
  arm.setPlanningTime(10.0);
  gripper.setMaxVelocityScalingFactor(0.5);
  gripper.setMaxAccelerationScalingFactor(0.5);
  gripper.setPlanningTime(2.0);

  log_info(logger, "UR3 Pick & Place ready.  IK link: " + IK_LINK);

  try {
    // ── 1. Move to home position (joint-space) ──
    log_info(logger, "=== 1. READY ===");
    arm.setJointValueTarget(READY_JOINTS);
    try_move(arm, logger, "READY");
    RCLCPP_INFO(logger, "RECORD_START");  // signal Python to start recording

    // ── 2. Move TCP above block (Cartesian straight line) ──
    log_info(logger, "=== 2. ABOVE block ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = block_world_x;
      tgt.position.y = block_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "ABOVE");
    }

    // ── 3. Open gripper ──
    log_info(logger, "=== 3. Open gripper ===");
    gripper.setNamedTarget(GRIP_OPEN);
    try_move(gripper, logger, "open");

    // ── 4. Move TCP down to grasp (Cartesian straight line) ──
    log_info(logger, "=== 4. GRASP ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = block_world_x;
      tgt.position.y = block_world_y;
      tgt.position.z = TOOL0_GRASP_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "GRASP");
    }

    // ── 5. Close gripper ──
    log_info(logger, "=== 5. Close gripper ===");
    gripper.setNamedTarget(GRIP_CLOSE);
    try_move(gripper, logger, "close");

    // ── 6. LIFT = back to ABOVE block (Cartesian) ──
    log_info(logger, "=== 6. LIFT ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = block_world_x;
      tgt.position.y = block_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "LIFT");
    }

    // ── 7. Move TCP above bowl (Cartesian) ──
    log_info(logger, "=== 7. ABOVE bowl ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = bowl_world_x;
      tgt.position.y = bowl_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "ABOVE bowl");
    }

    // ── 8. Move TCP down to place (Cartesian straight line) ──
    log_info(logger, "=== 8. PLACE ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = bowl_world_x;
      tgt.position.y = bowl_world_y;
      tgt.position.z = TOOL0_PLACE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "PLACE");
    }

    // ── 9. Open gripper ──
    log_info(logger, "=== 9. Open gripper ===");
    gripper.setNamedTarget(GRIP_OPEN);
    try_move(gripper, logger, "open");

    // ── 10. Retract ABOVE bowl (Cartesian, MUST succeed) ──
    log_info(logger, "=== 10. Retract ABOVE bowl ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = bowl_world_x;
      tgt.position.y = bowl_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      if (!try_cartesian(arm, logger, tgt, IK_LINK, "Retract ABOVE bowl"))
        return 1;
    }

    // ── 11. Return HOME (joint-space, safe at z=1.115 above bowl) ──
    // Skip when collecting task-only trajectories (skip_home:=true)
    node->declare_parameter<bool>("skip_home", false);
    if (!node->get_parameter("skip_home").as_bool()) {
      log_info(logger, "=== 11. HOME ===");
      arm.setJointValueTarget(READY_JOINTS);
      try_move(arm, logger, "HOME");
    }

    log_info(logger, "Done.");
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception: %s", e.what());
  }

  executor->cancel();
  spin.join();
  rclcpp::shutdown();
  return 0;
}
