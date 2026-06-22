/**
 * UR3 + FT300 + Robotiq 2F85 peg-in-hole assembly demo (ROS 2 Humble / MoveIt2)
 *
 * 4-stage sequence: approach → align → insert → tighten
 * Cartesian moves use computeCartesianPath() for true straight-line motion.
 *
 * Build & run:
 *   colcon build --symlink-install --packages-select ur_simulation_gz
 *   ros2 launch ur_simulation_gz ur3_ft300_robotiq.launch.py   # Terminal 1
 *   ros2 run ur_simulation_gz peg_in_hole                        # Terminal 2
 */

#include <chrono>
#include <cmath>
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

// Cartesian IK target link
static const std::string IK_LINK = "tool0";

// Standard UR named states for the gripper (defined in SRDF)
static const std::string GRIP_OPEN  = "open";   // knuckle ≈ 0.1
static const std::string GRIP_CLOSE = "close";  // knuckle ≈ 0.55

// ROS 1 home: arm bent at elbow, wrist folded, gripper pointing DOWN.
static const std::vector<double> READY_JOINTS = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};

// ── Geometry constants ──
// base_link at world z ≈ 0.76.  Shoulder at world z ≈ 0.91.
// Table surface at z = 0.775.
// Peg: cylinder 0.025m radius × 0.10m high, center at z = 0.825 (bottom 0.775, top 0.875)
// Hole socket: ring mesh (STL) R_outer=0.042 R_inner=0.027 H=0.05
//   sits on base plate at z=0.795→0.845, round inner hole d=0.054m
//
// PEG_EXTEND is measured FROM FORCE DATA: first base-plate contact at tool0_z≈1.055.
//   PEG_EXTEND = 1.055 - 0.795 (base plate top) = 0.260
//   (Larger than geometric 0.23 because fingers grip peg ~3cm below fingertip.)
//
// peg_bottom = tool0_z - PEG_EXTEND  (PEG_EXTEND = 0.260)
//   ring top=0.845  ring bottom=base_top=0.795  peg_top=0.875  table=0.775

static constexpr double PEG_WORLD_Z      = 0.825;  // peg centre (0.10m tall)
static constexpr double FINGERTIP_OFFSET = 0.14;   // tool0 → fingertip
static constexpr double PEG_EXTEND       = 0.260;  // MEASURED: tool0 → peg_bottom

// tool0 world-z targets (all computed from peg_bottom = tool0_z - 0.260)
//
// ABOVE: peg_bottom=0.880 → clears ring (0.845) by 35mm
static constexpr double TOOL0_ABOVE_Z    = 1.140;
// GRASP: peg_bottom=0.775 → just touching table (peg on table for pickup)
static constexpr double TOOL0_GRASP_Z    = 1.035;
// APPROACH: peg_bottom=0.865 → 20mm above ring top; fingers clear ring body
static constexpr double TOOL0_APPROACH_Z = 0.865 + PEG_EXTEND;  // = 1.125
// INSERT: peg_bottom=0.805 → 10mm above base plate; peg in hole, not touching bottom
static constexpr double TOOL0_INSERT_Z   = 0.805 + PEG_EXTEND;  // = 1.065
// SEAT: peg_bottom=0.793 → light press onto base plate + touch inner walls
static constexpr double TOOL0_SEAT_Z     = 0.793 + PEG_EXTEND;  // = 1.053
// Gripper partial-close: firm grip on 0.05m peg (open=0.1, full=0.55)
static constexpr double GRIPPER_GRASP    = 0.40;

// ── Helpers ──

static void log_info(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_INFO(l, "%s", msg.c_str()); }

static void log_error(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_ERROR(l, "%s", msg.c_str()); }

// Joint-space move
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

// Cartesian straight-line move
static bool try_cartesian(moveit::planning_interface::MoveGroupInterface& mgi,
                          const rclcpp::Logger& logger,
                          const geometry_msgs::msg::Pose& target,
                          const std::string& ik_link,
                          const std::string& desc,
                          double eef_step = 0.01)
{
  std::vector<geometry_msgs::msg::Pose> waypoints;
  auto current = mgi.getCurrentPose(ik_link);
  waypoints.push_back(current.pose);
  waypoints.push_back(target);

  moveit_msgs::msg::RobotTrajectory trajectory;
  const double jump_threshold = 0.0;
  double fraction = mgi.computeCartesianPath(waypoints, eef_step,
                                             jump_threshold, trajectory);

  if (fraction < 0.90) {
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
      "peg_in_hole",
      rclcpp::NodeOptions().parameter_overrides(
          std::vector{rclcpp::Parameter("use_sim_time", true)}));
  auto logger = node->get_logger();

  // Peg & hole positions from parameters (overridable via --ros-args -p ...)
  node->declare_parameter<double>("peg_x",  0.20);
  node->declare_parameter<double>("peg_y",  0.35);
  node->declare_parameter<double>("hole_x", -0.15);
  node->declare_parameter<double>("hole_y",  0.35);
  double peg_world_x  = node->get_parameter("peg_x").as_double();
  double peg_world_y  = node->get_parameter("peg_y").as_double();
  double hole_world_x = node->get_parameter("hole_x").as_double();
  double hole_world_y = node->get_parameter("hole_y").as_double();
  RCLCPP_INFO(logger, "Peg:  [%.3f %.3f]  Hole: [%.3f %.3f]",
              peg_world_x, peg_world_y, hole_world_x, hole_world_y);

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

  log_info(logger, "UR3 Peg-in-Hole ready.  IK link: " + IK_LINK);

  try {
    // ── 1. Move to home position (joint-space) ──
    log_info(logger, "=== 1. HOME ===");
    arm.setJointValueTarget(READY_JOINTS);
    try_move(arm, logger, "HOME");
    RCLCPP_INFO(logger, "RECORD_START");  // signal Python to start recording
    RCLCPP_INFO(logger, "STAGE:0");  // 0=approach: move above peg

    // ── 2. Move TCP above peg (Cartesian) ──
    log_info(logger, "=== 2. ABOVE peg ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = peg_world_x;
      tgt.position.y = peg_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "ABOVE peg");
    }

    // ── 3. Open gripper ──
    log_info(logger, "=== 3. Open gripper ===");
    RCLCPP_INFO(logger, "STAGE:1");  // 1=align: open gripper, approach peg
    gripper.setNamedTarget(GRIP_OPEN);
    try_move(gripper, logger, "open");

    // ── 4. Move TCP down to grasp peg (Cartesian) ──
    log_info(logger, "=== 4. GRASP peg ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = peg_world_x;
      tgt.position.y = peg_world_y;
      tgt.position.z = TOOL0_GRASP_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "GRASP peg");
    }

    // ── 5. Close gripper (partial close — grip 0.05m peg without timeout) ──
    log_info(logger, "=== 5. Close gripper (grip peg) ===");
    gripper.setJointValueTarget("robotiq_85_left_knuckle_joint", GRIPPER_GRASP);
    try_move(gripper, logger, "close");
    RCLCPP_INFO(logger, "STAGE:2");  // 2=grasp: grip peg, lift

    // ── 6. LIFT peg (Cartesian up) ──
    log_info(logger, "=== 6. LIFT ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = peg_world_x;
      tgt.position.y = peg_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "LIFT");
    }

    // ── 7. Move TCP above hole (Cartesian) ──
    log_info(logger, "=== 7. ABOVE hole ===");
    RCLCPP_INFO(logger, "STAGE:0");  // 0=approach: transport to hole
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = hole_world_x;
      tgt.position.y = hole_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;  // safe height above hole
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "ABOVE hole");
    }

    // ── 8. APPROACH hole (slow Cartesian descent) ──
    RCLCPP_INFO(logger, "STAGE:1");  // 1=align: descend to hole entrance
    log_info(logger, "=== 8. APPROACH hole ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = hole_world_x;
      tgt.position.y = hole_world_y;
      tgt.position.z = TOOL0_APPROACH_Z;  // peg tip just above hole
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "APPROACH", 0.005);
    }

    // ── 9. INSERT peg into hole (very slow, small lateral oscillations for alignment) ──
    log_info(logger, "=== 9. INSERT ===");
    RCLCPP_INFO(logger, "STAGE:3");  // 3=insert: push peg into hole
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = hole_world_x;
      tgt.position.y = hole_world_y;
      tgt.position.z = TOOL0_INSERT_Z;  // peg seated in hole
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "INSERT", 0.003);
    }

    // ── 10. SEAT (light press + touch inner walls to confirm insertion) ──
    log_info(logger, "=== 10. SEAT ===");
    RCLCPP_INFO(logger, "STAGE:4");  // 4=confirm: press bottom + touch walls
    {
      // Light downward press: 2mm below INSERT, peg touches base plate
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = hole_world_x;
      tgt.position.y = hole_world_y;
      tgt.position.z = TOOL0_SEAT_Z;
      RCLCPP_INFO(logger, "  press → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      try_cartesian(arm, logger, tgt, IK_LINK, "SEAT press", 0.002);

      // Touch inner walls in x ± direction
      for (int wall = 0; wall < 2; wall++) {
        double dx = (wall % 2 == 0) ? 0.004 : -0.004;
        auto cur = arm.getCurrentPose(IK_LINK);
        geometry_msgs::msg::Pose wtgt;
        wtgt.orientation = cur.pose.orientation;
        wtgt.position.x = hole_world_x + dx;
        wtgt.position.y = hole_world_y;
        wtgt.position.z = TOOL0_SEAT_Z;
        try_cartesian(arm, logger, wtgt, IK_LINK,
                      "SEAT wall_x " + std::to_string(wall), 0.005);
      }

      // Touch inner walls in y ± direction
      for (int wall = 0; wall < 2; wall++) {
        double dy = (wall % 2 == 0) ? 0.004 : -0.004;
        auto cur = arm.getCurrentPose(IK_LINK);
        geometry_msgs::msg::Pose wtgt;
        wtgt.orientation = cur.pose.orientation;
        wtgt.position.x = hole_world_x;
        wtgt.position.y = hole_world_y + dy;
        wtgt.position.z = TOOL0_SEAT_Z;
        try_cartesian(arm, logger, wtgt, IK_LINK,
                      "SEAT wall_y " + std::to_string(wall), 0.005);
      }

      // Return to center
      auto cur2 = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose ctgt;
      ctgt.orientation = cur2.pose.orientation;
      ctgt.position.x = hole_world_x;
      ctgt.position.y = hole_world_y;
      ctgt.position.z = TOOL0_SEAT_Z;
      try_cartesian(arm, logger, ctgt, IK_LINK, "SEAT center", 0.005);
    }

    // ── 11. Open gripper ──
    log_info(logger, "=== 11. Open gripper ===");
    gripper.setNamedTarget(GRIP_OPEN);
    try_move(gripper, logger, "open");

    // ── 12. Retract ABOVE hole (Cartesian up) ──
    log_info(logger, "=== 12. RETRACT ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = hole_world_x;
      tgt.position.y = hole_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      RCLCPP_INFO(logger, "  Cartesian → [%.3f %.3f %.3f]",
                  tgt.position.x, tgt.position.y, tgt.position.z);
      if (!try_cartesian(arm, logger, tgt, IK_LINK, "RETRACT"))
        return 1;
    }

    // ── 13. Return HOME (joint-space, optional) ──
    node->declare_parameter<bool>("skip_home", false);
    if (!node->get_parameter("skip_home").as_bool()) {
      log_info(logger, "=== 13. HOME ===");
      arm.setJointValueTarget(READY_JOINTS);
      try_move(arm, logger, "HOME");
    }

    log_info(logger, "Peg-in-hole complete.");
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception: %s", e.what());
  }

  executor->cancel();
  spin.join();
  rclcpp::shutdown();
  return 0;
}
