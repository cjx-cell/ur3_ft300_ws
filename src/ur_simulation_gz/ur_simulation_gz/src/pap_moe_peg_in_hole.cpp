/**
 * PAP-MoE Peg-in-Hole Assembly Controller (ROS 2 Humble / MoveIt2)
 *
 * Implements mixed direct insertion, force-gradient contact recovery, and a
 * bounded spiral fallback for training the PAP-MoE framework.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <sstream>
#include <thread>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter_client.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

using namespace std::chrono_literals;

static const std::string ARM_GROUP     = "ur_manipulator";
static const std::string GRIPPER_GROUP = "gripper";
static const std::string IK_LINK = "tool0";
static const std::string GRIP_OPEN  = "open";
static const std::string GRIP_CLOSE = "close";

// Shared image-left, downward-facing collection HOME. Keep synchronized with
// the Gazebo initial pose and teleop return-home contract.
static const std::vector<double> READY_JOINTS = {
    -1.254, -1.5707, 1.5707, -1.5707, -1.5707, 0.0};
// World geometry contract shared with pap_moe_peg_in_hole_record.py.
// Real lab fixture: 180 mm peg (80 mm handle + 20 mm collar + 80 mm
// frustum) and a 100 mm tapered socket.  With the handle grasped near its
// middle, the peg tip is approximately 340 mm below tool0.
static constexpr double PEG_TIP_OFFSET = 0.340;
static constexpr double PEG_CENTER_Z = 0.865;
static constexpr double RING_TOP_Z = 0.875;
static constexpr double RING_BOTTOM_Z = 0.795;  // 20 mm solid socket bottom

// Expanded left/right task boxes require more horizontal reach.  At the
// outer sampled corner (hole x=0.313, y=0.175), 1.215 m forced a 2 rad IK
// branch jump during transport.  Lowering the vertical tool by 5 mm keeps
// about 8 mm peg-tip clearance above the socket rim while moving the pose
// away from that UR3 reach singularity.
static constexpr double TOOL0_ABOVE_Z    = 1.210;
// Place the fingertip pads around the middle of the 80 mm handle instead of
// pinching only its upper edge.
// At the measured Robotiq contact angle (~0.627 rad), tool0=1.090 m places
// the explicit fingertip pads across the middle of the 80 mm handle.  The
// previous 1.115 m target pinched its upper edge and allowed the long peg to
// rotate about a small contact patch during horizontal transport.
static constexpr double TOOL0_GRASP_Z    = 1.090;
// Lower another 2 mm before compensating physical-grasp slip.  This retains
// about 6 mm rim clearance and adds lateral IK margin for exact-axis
// calibration at the same outer corner.
static constexpr double TOOL0_APPROACH_Z = 1.208;
static constexpr double TOOL0_ENTRY_CONFIRM_MAX_Z =
    RING_TOP_Z + PEG_TIP_OFFSET - 0.002;  // at least 2 mm guide-in
static constexpr double TOOL0_DESCENT_MIN_Z =
    RING_TOP_Z + PEG_TIP_OFFSET - 0.007;  // bounded 7 mm guide-in
static constexpr double TOOL0_INSERT_Z =
    RING_TOP_Z + PEG_TIP_OFFSET - 0.075;  // near-full taper insertion
static constexpr double TOOL0_SEAT_Z =
    RING_BOTTOM_Z + PEG_TIP_OFFSET;       // ring bottom
static constexpr double INSERT_COARSE_STEP_Z = 0.001;
static constexpr double INSERT_FINE_STEP_Z = 0.00025;
// Cruise through the geometrically free portion at a human-demonstration-like
// 10 mm/s, then use the configurable fine step only for the last 3 mm near
// the rigid socket floor.  Force remains sampled and guarded at 100 Hz in
// both zones; safety does not require making the whole 89 mm descent creep.
static constexpr double INSERT_FREE_CRUISE_STEP = 0.000100;
static constexpr double INSERT_CONTACT_APPROACH_DISTANCE = 0.0030;
// After measured XY calibration the peg tip is still in free space. Descend
// at 10 mm/s while checking FT300 every 10 ms; the last 3 mm keeps the
// separately configurable 1 mm/s contact approach below.
static constexpr double GUIDED_DESCENT_STEP = 0.000100;
static constexpr double TOOL0_FINE_INSERT_START_Z = 1.145;
// The real socket has a rigid 20 mm bottom.  Stop at its predicted contact
// height; the force guard and model-pose service independently verify seating.
static constexpr double TOOL0_FORCE_PROBE_MIN_Z = 1.1380;
static constexpr double TOOL0_SEATED_MAX_Z = 1.1375;
static constexpr double INSERT_CONTACT_FORCE_MIN = 0.8;
// The physical gripper and position servo produce brief 1--2 N axial
// disturbances in the geometrically free part of the taper. Only a stronger
// sustained seating contact may stop final insertion; 12 N remains the hard
// safety ceiling below.
static constexpr double INSERT_SEAT_CONTACT_FORCE_MIN = 5.0;
// After the 0.8 N contact trigger stops the controller, allow a small
// steady-state relaxation while still requiring a sustained non-zero load.
static constexpr double INSERT_VERIFY_FORCE_MIN = 0.75;
static constexpr double INSERT_AXIAL_FORCE_MAX = 12.0;
static constexpr double INSERT_LATERAL_CONTACT_MIN = 10.0;
// Exact-fit human demonstrations can sustain appreciable lateral fitting
// force.  Ten newtons triggers a controlled stop; 40 N remains the hard
// bounded-contact limit. Geometry and no-gravity release gates remain strict.
// Gazebo's rigid, zero-clearance socket can produce short lateral impulses
// above 50 N while the measured peg remains centered and is still seating.
// Keep a bounded guard, but allow the controller to finish that valid motion.
static constexpr double INSERT_LATERAL_FORCE_MAX = 60.0;
static constexpr double INSERT_VERIFY_LATERAL_MEAN_MAX = 58.0;
static constexpr double INSERT_VERIFY_FORCE_STD_MAX = 3.0;
static constexpr int INSERT_VERIFY_FORCE_SAMPLES = 25;
// Universal Robotiq endpoint commands.  A measured intermediate angle while
// holding an object is a physical contact result, never an object-specific
// target angle.
static constexpr double GRIPPER_FULLY_OPEN = 0.0;
static constexpr double GRIPPER_FULLY_CLOSED = 0.8;

// Global force-torque data structure
using WrenchVector = std::array<double, 6>;
geometry_msgs::msg::Wrench current_wrench;
std::mutex wrench_mutex;
std::deque<WrenchVector> wrench_filter_window;
std::mutex joint_state_mutex;
std::vector<std::string> latest_joint_names;
std::vector<double> latest_joint_positions;

static void wrench_callback(const geometry_msgs::msg::WrenchStamped::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(wrench_mutex);
  const WrenchVector raw = {
      msg->wrench.force.x, msg->wrench.force.y, msg->wrench.force.z,
      msg->wrench.torque.x, msg->wrench.torque.y, msg->wrench.torque.z};
  wrench_filter_window.push_back(raw);
  if (wrench_filter_window.size() > 5) {
    wrench_filter_window.pop_front();
  }
  WrenchVector filtered{};
  for (std::size_t axis = 0; axis < filtered.size(); ++axis) {
    std::vector<double> values;
    values.reserve(wrench_filter_window.size());
    for (const auto& sample : wrench_filter_window) {
      values.push_back(sample[axis]);
    }
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    filtered[axis] = *middle;
  }
  current_wrench.force.x = filtered[0];
  current_wrench.force.y = filtered[1];
  current_wrench.force.z = filtered[2];
  current_wrench.torque.x = filtered[3];
  current_wrench.torque.y = filtered[4];
  current_wrench.torque.z = filtered[5];
}

static void joint_state_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(joint_state_mutex);
  latest_joint_names = msg->name;
  latest_joint_positions = msg->position;
}

static void log_info(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_INFO(l, "%s", msg.c_str()); }

static void log_error(const rclcpp::Logger& l, const std::string& msg)
{ RCLCPP_ERROR(l, "%s", msg.c_str()); }

using TriggerClient = rclcpp::Client<std_srvs::srv::Trigger>;
using ParameterClient = rclcpp::AsyncParametersClient;

static bool set_sim_position_gain(
    const std::shared_ptr<ParameterClient>& client,
    const rclcpp::Logger& logger,
    double gain)
{
  if (!client->wait_for_service(5s)) {
    log_error(logger, "controller_manager parameter service is unavailable");
    return false;
  }
  auto future = client->set_parameters(
      {rclcpp::Parameter("position_proportional_gain", gain)});
  if (future.wait_for(5s) != std::future_status::ready) {
    log_error(logger, "Timed out while changing simulation position gain");
    return false;
  }
  const auto results = future.get();
  if (results.empty() || !results.front().successful) {
    log_error(logger, "Simulation position gain change was rejected");
    return false;
  }
  log_info(logger, "SIM_POSITION_GAIN:" + std::to_string(gain));
  return true;
}

static bool call_trigger_service(
    const TriggerClient::SharedPtr& client,
    const rclcpp::Logger& logger,
    const std::string& service_name)
{
  if (!client->wait_for_service(5s)) {
    log_error(logger, service_name + " is unavailable.");
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future = client->async_send_request(request);
  if (future.wait_for(15s) != std::future_status::ready) {
    log_error(logger, service_name + " timed out.");
    return false;
  }
  const auto response = future.get();
  log_info(
      logger,
      service_name + ":" +
      (response->success ? "success," : "failure,") +
      response->message);
  return response->success;
}

static bool query_peg_hole_alignment(
    const TriggerClient::SharedPtr& client,
    const rclcpp::Logger& logger,
    double& dx,
    double& dy,
    double& distance,
    double& peg_center_z,
    double& peg_tilt_rad)
{
  const std::string service_name = "/pap_moe/query_peg_hole_alignment";
  if (!client->wait_for_service(5s)) {
    log_error(logger, service_name + " is unavailable.");
    return false;
  }
  auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto future = client->async_send_request(request);
  if (future.wait_for(15s) != std::future_status::ready) {
    log_error(logger, service_name + " timed out.");
    return false;
  }
  const auto response = future.get();
  log_info(
      logger,
      service_name + ":" +
      (response->success ? "success," : "failure,") +
      response->message);
  if (!response->success) {
    return false;
  }
  return std::sscanf(
      response->message.c_str(),
      "dx=%lf,dy=%lf,hole_dist=%lf,peg_z=%lf,tilt_rad=%lf",
      &dx, &dy, &distance, &peg_center_z, &peg_tilt_rad) == 5;
}

static geometry_msgs::msg::Quaternion quaternion_multiply(
    const geometry_msgs::msg::Quaternion& a,
    const geometry_msgs::msg::Quaternion& b)
{
  geometry_msgs::msg::Quaternion out;
  out.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
  out.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
  out.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
  out.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
  return out;
}

static geometry_msgs::msg::Quaternion quaternion_inverse(
    const geometry_msgs::msg::Quaternion& q)
{
  const double norm_sq =
      q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
  geometry_msgs::msg::Quaternion out;
  out.w = q.w / norm_sq;
  out.x = -q.x / norm_sq;
  out.y = -q.y / norm_sq;
  out.z = -q.z / norm_sq;
  return out;
}

static geometry_msgs::msg::Quaternion quaternion_from_roll_pitch(
    double roll, double pitch)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  geometry_msgs::msg::Quaternion out;
  out.w = cr * cp;
  out.x = sr * cp;
  out.y = cr * sp;
  out.z = -sr * sp;
  return out;
}

static WrenchVector read_wrench_vector()
{
  std::lock_guard<std::mutex> lock(wrench_mutex);
  return {
    current_wrench.force.x, current_wrench.force.y, current_wrench.force.z,
    current_wrench.torque.x, current_wrench.torque.y, current_wrench.torque.z
  };
}

static WrenchVector subtract_wrench(const WrenchVector& value,
                                    const WrenchVector& bias)
{
  WrenchVector corrected{};
  for (std::size_t index = 0; index < corrected.size(); ++index) {
    corrected[index] = value[index] - bias[index];
  }
  return corrected;
}

static double contact_cost(const WrenchVector& wrench)
{
  const double lateral_force = std::hypot(wrench[0], wrench[1]);
  const double normal_force = std::abs(wrench[2]);
  const double tilt_torque = std::hypot(wrench[3], wrench[4]);
  return lateral_force + 0.35 * normal_force + 2.0 * tilt_torque;
}

static std::array<double, 3> rotate_vector(
    const geometry_msgs::msg::Quaternion& quaternion,
    const std::array<double, 3>& vector)
{
  const double qx = quaternion.x;
  const double qy = quaternion.y;
  const double qz = quaternion.z;
  const double qw = quaternion.w;
  const double tx = 2.0 * (qy * vector[2] - qz * vector[1]);
  const double ty = 2.0 * (qz * vector[0] - qx * vector[2]);
  const double tz = 2.0 * (qx * vector[1] - qy * vector[0]);
  return {
    vector[0] + qw * tx + (qy * tz - qz * ty),
    vector[1] + qw * ty + (qz * tx - qx * tz),
    vector[2] + qw * tz + (qx * ty - qy * tx)
  };
}

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

static bool try_cartesian(moveit::planning_interface::MoveGroupInterface& mgi,
                           const rclcpp::Logger& logger,
                           const geometry_msgs::msg::Pose& target,
                           const std::string& ik_link,
                           const std::string& desc,
                           double eef_step = 0.01,
                           double time_scale = 1.0)
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
  if (time_scale > 1.0) {
    for (auto& point : trajectory.joint_trajectory.points) {
      const std::uint64_t original_ns =
          static_cast<std::uint64_t>(point.time_from_start.sec) *
              1000000000ULL +
          static_cast<std::uint64_t>(point.time_from_start.nanosec);
      const std::uint64_t scaled_ns = static_cast<std::uint64_t>(
          std::llround(static_cast<double>(original_ns) * time_scale));
      point.time_from_start.sec =
          static_cast<std::int32_t>(scaled_ns / 1000000000ULL);
      point.time_from_start.nanosec =
          static_cast<std::uint32_t>(scaled_ns % 1000000000ULL);
      for (double& velocity : point.velocities) {
        velocity /= time_scale;
      }
      for (double& acceleration : point.accelerations) {
        acceleration /= time_scale * time_scale;
      }
    }
    log_info(
        logger,
        "CARTESIAN_TIME_SCALE:" + desc + "," +
        std::to_string(time_scale));
  }
  auto r = mgi.execute(trajectory);
  if (r) return true;
  log_error(logger, "  FAILED: " + desc + " (code " +
            std::to_string(static_cast<int>(r.val)) + ")");
  return false;
}

static bool try_cartesian_arc(
    moveit::planning_interface::MoveGroupInterface& mgi,
    const rclcpp::Logger& logger,
    const geometry_msgs::msg::Pose& target,
    const std::string& ik_link,
    const std::string& desc,
    double bulge_x,
    double bulge_y,
    double bulge_z,
    double eef_step = 0.005)
{
  const auto start = mgi.getCurrentPose(ik_link).pose;
  std::vector<geometry_msgs::msg::Pose> waypoints;
  constexpr int ARC_SAMPLES = 12;
  waypoints.reserve(ARC_SAMPLES + 1);
  for (int index = 0; index <= ARC_SAMPLES; ++index) {
    const double alpha =
        static_cast<double>(index) / static_cast<double>(ARC_SAMPLES);
    const double envelope = 4.0 * alpha * (1.0 - alpha);
    geometry_msgs::msg::Pose point = target;
    point.position.x =
        (1.0 - alpha) * start.position.x + alpha * target.position.x +
        envelope * bulge_x;
    point.position.y =
        (1.0 - alpha) * start.position.y + alpha * target.position.y +
        envelope * bulge_y;
    point.position.z =
        (1.0 - alpha) * start.position.z + alpha * target.position.z +
        envelope * bulge_z;
    // All diversified free-space segments keep one tool orientation. The
    // separate lift trajectory remains responsible for peg-axis rotation.
    point.orientation = target.orientation;
    waypoints.push_back(point);
  }
  moveit_msgs::msg::RobotTrajectory trajectory;
  const double fraction = mgi.computeCartesianPath(
      waypoints, eef_step, 0.0, trajectory);
  if (fraction < 0.90) {
    log_error(
        logger,
        desc + " diversified Cartesian arc only " +
        std::to_string(static_cast<int>(fraction * 100)) + "%");
    return false;
  }
  log_info(
      logger,
      "TRAJECTORY_ARC:" + desc + "," + std::to_string(bulge_x) + "," +
      std::to_string(bulge_y) + "," + std::to_string(bulge_z));
  const auto result = mgi.execute(trajectory);
  if (result) {
    return true;
  }
  log_error(logger, "FAILED diversified arc: " + desc);
  return false;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
      "pap_moe_peg_in_hole",
      rclcpp::NodeOptions().parameter_overrides(
          std::vector{rclcpp::Parameter("use_sim_time", true)}));
  auto logger = node->get_logger();

  node->declare_parameter<double>("peg_x",  0.20);
  node->declare_parameter<double>("peg_y",  0.35);
  node->declare_parameter<double>("peg_roll", 0.0);
  node->declare_parameter<double>("peg_pitch", 0.0);
  node->declare_parameter<double>("hole_x", -0.15);
  node->declare_parameter<double>("hole_y",  0.35);
  node->declare_parameter<double>("velocity_scaling", 0.05);
  node->declare_parameter<double>("transport_velocity_scaling", 1.0);
  node->declare_parameter<double>("fine_insertion_step", 0.000010);
  node->declare_parameter<double>("recovery_offset_max", 0.006);
  node->declare_parameter<double>("grasp_recovery_offset_x", 0.0);
  node->declare_parameter<double>("grasp_recovery_offset_y", 0.0);
  node->declare_parameter<int>("trajectory_style_id", 0);
  node->declare_parameter<double>("peg_arc_x", 0.0);
  node->declare_parameter<double>("peg_arc_y", 0.0);
  node->declare_parameter<double>("peg_arc_z", 0.0);
  node->declare_parameter<double>("transport_arc_x", 0.0);
  node->declare_parameter<double>("transport_arc_y", 0.0);
  node->declare_parameter<double>("transport_arc_z", 0.0);
  node->declare_parameter<double>("hole_arc_x", 0.0);
  node->declare_parameter<double>("hole_arc_y", 0.0);
  node->declare_parameter<bool>("grasp_recovery_only", false);
  node->declare_parameter<std::string>("search_mode", "mixed");
  double peg_world_x  = node->get_parameter("peg_x").as_double();
  double peg_world_y  = node->get_parameter("peg_y").as_double();
  const double peg_roll = node->get_parameter("peg_roll").as_double();
  const double peg_pitch = node->get_parameter("peg_pitch").as_double();
  double hole_world_x = node->get_parameter("hole_x").as_double();
  double hole_world_y = node->get_parameter("hole_y").as_double();
  double vel_scale = node->get_parameter("velocity_scaling").as_double();
  double transport_vel_scale = node->get_parameter("transport_velocity_scaling").as_double();
  const double fine_insertion_step =
      node->get_parameter("fine_insertion_step").as_double();
  const double recovery_offset_max =
      node->get_parameter("recovery_offset_max").as_double();
  const double grasp_recovery_offset_x =
      node->get_parameter("grasp_recovery_offset_x").as_double();
  const double grasp_recovery_offset_y =
      node->get_parameter("grasp_recovery_offset_y").as_double();
  const int trajectory_style_id =
      node->get_parameter("trajectory_style_id").as_int();
  const double peg_arc_x = node->get_parameter("peg_arc_x").as_double();
  const double peg_arc_y = node->get_parameter("peg_arc_y").as_double();
  const double peg_arc_z = node->get_parameter("peg_arc_z").as_double();
  const double transport_arc_x =
      node->get_parameter("transport_arc_x").as_double();
  const double transport_arc_y =
      node->get_parameter("transport_arc_y").as_double();
  const double transport_arc_z =
      node->get_parameter("transport_arc_z").as_double();
  const double hole_arc_x = node->get_parameter("hole_arc_x").as_double();
  const double hole_arc_y = node->get_parameter("hole_arc_y").as_double();
  const double grasp_recovery_offset_norm = std::hypot(
      grasp_recovery_offset_x, grasp_recovery_offset_y);
  const bool grasp_recovery_only =
      node->get_parameter("grasp_recovery_only").as_bool();
  const std::string requested_search_mode =
      node->get_parameter("search_mode").as_string();
  if (recovery_offset_max < 0.003 || recovery_offset_max > 0.012) {
    log_error(
        logger,
        "recovery_offset_max must be in [0.003, 0.012] m, got " +
        std::to_string(recovery_offset_max));
    rclcpp::shutdown();
    return 2;
  }
  if (grasp_recovery_offset_norm > 0.12) {
    log_error(
        logger,
        "grasp recovery offset must be <= 0.12 m, got " +
        std::to_string(grasp_recovery_offset_norm));
    rclcpp::shutdown();
    return 2;
  }
  if (grasp_recovery_only && grasp_recovery_offset_norm <= 1e-6) {
    log_error(logger, "grasp_recovery_only requires a non-zero recovery offset");
    rclcpp::shutdown();
    return 2;
  }
  // The force loop runs at 100 Hz.  Keep the commanded advance bounded to
  // 5--25 um per cycle (0.5--2.5 mm/s) so collection can be faster without
  // losing the one-cycle contact-stop guarantee.
  if (fine_insertion_step < 0.000005 ||
      fine_insertion_step > 0.000025) {
    log_error(
        logger,
        "fine_insertion_step must be in [0.000005, 0.000025] m, got " +
        std::to_string(fine_insertion_step));
    rclcpp::shutdown();
    return 2;
  }

  // Subscribe to wrench topic
  auto wrench_sub = node->create_subscription<geometry_msgs::msg::WrenchStamped>(
      "/force_torque_sensor_broadcaster/wrench", 10, wrench_callback);
  auto joint_state_sub = node->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, joint_state_callback);

  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  executor->add_node(node);
  auto spin = std::thread([executor] { executor->spin(); });
  auto detach_client =
      node->create_client<std_srvs::srv::Trigger>("/pap_moe/detach_peg");
  auto confirm_grasp_client =
      node->create_client<std_srvs::srv::Trigger>(
          "/pap_moe/confirm_grasp_attachment");
  auto verify_release_client =
      node->create_client<std_srvs::srv::Trigger>(
          "/pap_moe/verify_peg_release");
  auto confirm_insertion_client =
      node->create_client<std_srvs::srv::Trigger>(
          "/pap_moe/confirm_insertion_geometry");
  auto query_alignment_client =
      node->create_client<std_srvs::srv::Trigger>(
          "/pap_moe/query_peg_hole_alignment");
  auto controller_parameter_client =
      std::make_shared<ParameterClient>(node, "/controller_manager");
  // Load robot semantic description parameters
  {
    std::string pkg = ament_index_cpp::get_package_share_directory("ur3_ft300_moveit_config");
    std::string cfg = pkg + "/config/";

    std::ifstream srf(cfg + "ur3_ft300_robotiq_2f85.srdf");
    if (!srf.is_open()) { RCLCPP_ERROR(logger, "Cannot open SRDF"); return 1; }
    std::stringstream ss; ss << srf.rdbuf();
    try { node->declare_parameter<std::string>("robot_description_semantic", ""); } catch (...) {}
    node->set_parameter(rclcpp::Parameter("robot_description_semantic", ss.str()));

    std::string xacro_path = cfg + "ur3_ft300_robotiq_2f85.urdf.xacro";
    std::string cmd = "xacro " + xacro_path + " 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) { RCLCPP_ERROR(logger, "xacro failed"); return 1; }
    std::string urdf;
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe.get())) urdf += buf;
    try { node->declare_parameter<std::string>("robot_description", ""); } catch (...) {}
    node->set_parameter(rclcpp::Parameter("robot_description", urdf));
  }

  using moveit::planning_interface::MoveGroupInterface;
  MoveGroupInterface arm(node, ARM_GROUP);
  MoveGroupInterface gripper(node, GRIPPER_GROUP);

  // Force lazy initialization of current state monitor on the client side
  try {
    arm.getCurrentPose(IK_LINK);
    gripper.getCurrentState();
  } catch (...) {}

  // Sleep AFTER constructing MoveGroupInterface to let current_state_monitor initialize and receive joint states
  std::this_thread::sleep_for(8s);

  arm.setMaxVelocityScalingFactor(transport_vel_scale);
  arm.setMaxAccelerationScalingFactor(transport_vel_scale);
  arm.setPlanningTime(10.0);
  // Named gripper motions only cover free opening and the non-contact
  // pre-close.  The actual force-closure pickup remains on its separately
  // timed synchronized trajectory below.
  gripper.setMaxVelocityScalingFactor(1.0);
  gripper.setMaxAccelerationScalingFactor(1.0);
  gripper.setPlanningTime(2.0);

  log_info(logger, "PAP-MoE C++ Controller Initialized.");
  log_info(
      logger,
      "TRAJECTORY_STYLE:" + std::to_string(trajectory_style_id));

  try {
    // 1. Move Home
    log_info(logger, "=== 1. HOME ===");
    arm.setJointValueTarget(READY_JOINTS);
    if (!try_move(arm, logger, "HOME")) {
      throw std::runtime_error("HOME planning/execution failed");
    }
    const bool collect_grasp_recovery = grasp_recovery_offset_norm > 1e-6;
    if (collect_grasp_recovery) {
      // Roll into a policy-like high hover error without recording the
      // artificial outward motion. Recording begins at the displaced state,
      // so every saved action supervises recovery toward the peg.
      gripper.setNamedTarget(GRIP_OPEN);
      if (!try_move(gripper, logger, "open before grasp recovery roll-in")) {
        throw std::runtime_error("Grasp recovery roll-in gripper open failed");
      }
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose rollout_target;
      rollout_target.orientation = current.pose.orientation;
      rollout_target.position.x = peg_world_x + grasp_recovery_offset_x;
      rollout_target.position.y = peg_world_y + grasp_recovery_offset_y;
      rollout_target.position.z = TOOL0_ABOVE_Z;
      if (!try_cartesian(
              arm, logger, rollout_target, IK_LINK,
              "grasp recovery roll-in", 0.005)) {
        throw std::runtime_error("Grasp recovery roll-in failed");
      }
      std::this_thread::sleep_for(250ms);
      log_info(
          logger,
          "GRASP_RECOVERY_ROLLIN:" +
          std::to_string(grasp_recovery_offset_x) + "," +
          std::to_string(grasp_recovery_offset_y));
    }
    RCLCPP_INFO(logger, "RECORD_START");
    RCLCPP_INFO(logger, "STAGE:0"); // s=0: Free Motion
    log_info(logger, "SUBTASK:grasp the peg");

    // 2. Move above peg
    log_info(logger, "=== 2. ABOVE peg ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = peg_world_x;
      tgt.position.y = peg_world_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      if (!try_cartesian_arc(
              arm, logger, tgt, IK_LINK, "ABOVE peg",
              peg_arc_x, peg_arc_y, peg_arc_z)) {
        throw std::runtime_error("ABOVE peg planning/execution failed");
      }
    }

    // 3. Open gripper
    log_info(logger, "=== 3. Open gripper ===");
    if (!collect_grasp_recovery) {
      gripper.setNamedTarget(GRIP_OPEN);
      if (!try_move(gripper, logger, "open")) {
        throw std::runtime_error("Open gripper planning/execution failed");
      }
    }

    // 4. Move down to peg
    log_info(logger, "=== 4. GRASP peg ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      tgt.orientation = current.pose.orientation;
      tgt.position.x = peg_world_x;
      tgt.position.y = peg_world_y;
      tgt.position.z = TOOL0_GRASP_Z;
      if (!try_cartesian(arm, logger, tgt, IK_LINK, "GRASP peg")) {
        throw std::runtime_error("GRASP peg planning/execution failed");
      }
    }

    // Establish the simulation grasp while the free peg is still centred and
    // at rest. Closing first can strike the peg and then preserve a large,
    // incorrect transform when DetachableJoint attaches.
    const auto grasp_tool_pose = arm.getCurrentPose(IK_LINK).pose;
    const auto grasp_tool_orientation = grasp_tool_pose.orientation;
    const auto peg_tilt =
        quaternion_from_roll_pitch(peg_roll, peg_pitch);
    const auto aligned_tool_orientation = quaternion_multiply(
        quaternion_inverse(peg_tilt), grasp_tool_orientation);
    // DetachableJoint preserves the complete grasp transform.  Rotating tool0
    // about the wrist therefore sweeps the peg centre laterally.  Compute that
    // offset and compensate tool XY so the corrected peg axis, not tool0,
    // remains over the requested world point.
    const std::array<double, 3> peg_from_tool_world = {
        peg_world_x - grasp_tool_pose.position.x,
        peg_world_y - grasp_tool_pose.position.y,
        PEG_CENTER_Z - grasp_tool_pose.position.z,
    };
    const auto peg_from_tool_local = rotate_vector(
        quaternion_inverse(grasp_tool_orientation),
        peg_from_tool_world);
    const auto aligned_peg_offset = rotate_vector(
        aligned_tool_orientation, peg_from_tool_local);
    const double corrected_peg_tool_x =
        peg_world_x - aligned_peg_offset[0];
    const double corrected_peg_tool_y =
        peg_world_y - aligned_peg_offset[1];
    const double corrected_hole_tool_x =
        hole_world_x - aligned_peg_offset[0];
    const double corrected_hole_tool_y =
        hole_world_y - aligned_peg_offset[1];
    log_info(logger, "C++ PHYSICAL GRASP PEG");
    // 5. Close gripper using physical fingertip contact. A runtime
    // DetachableJoint created large non-physical constraint loads in DART;
    // the lifted peg pose is verified explicitly below instead.
    log_info(logger, "=== 5. Close gripper ===");
    // Close while beginning a small upward pickup. This avoids building a
    // sustained internal load against the table. The generic full-close
    // endpoint remains in the action trajectory; physical contact plus the
    // bottom-layer stall latch determine the achieved aperture.
    const auto pickup_start = arm.getCurrentPose(IK_LINK).pose;
    geometry_msgs::msg::Pose pickup_target = pickup_start;
    pickup_target.position.z += 0.015;
    std::vector<geometry_msgs::msg::Pose> pickup_waypoints{
        pickup_start, pickup_target};
    moveit_msgs::msg::RobotTrajectory pickup_trajectory;
    arm.setMaxVelocityScalingFactor(0.01);
    arm.setMaxAccelerationScalingFactor(0.005);
    const double pickup_fraction = arm.computeCartesianPath(
        pickup_waypoints, 0.0005, 0.0, pickup_trajectory);
    if (
        pickup_fraction < 0.99 ||
        pickup_trajectory.joint_trajectory.points.size() < 2) {
      throw std::runtime_error("Dynamic pickup trajectory planning failed");
    }
    auto& pickup_joint_trajectory = pickup_trajectory.joint_trajectory;
    pickup_joint_trajectory.joint_names.push_back(
        "robotiq_85_left_knuckle_joint");
    const std::size_t pickup_point_count =
        pickup_joint_trajectory.points.size();
    // The previous five-second close/lift profile was substantially slower
    // than the simulated 2F-85 limit and over-represented near-static grasp
    // frames in the policy dataset. Three seconds still stays below the
    // configured 0.5 rad/s knuckle limit while preserving contact latching.
    constexpr double PICKUP_TRAJECTORY_DURATION_S = 3.0;
    log_info(
        logger,
        "GRASP_PICKUP_PROFILE:duration_s=" +
        std::to_string(PICKUP_TRAJECTORY_DURATION_S) +
        ",gripper=ease_out_sqrt");
    for (std::size_t index = 0; index < pickup_point_count; ++index) {
      const double alpha =
          static_cast<double>(index) /
          static_cast<double>(pickup_point_count - 1);
      auto& point = pickup_joint_trajectory.points[index];
      // Approach the object quickly while contact-free, then reduce finger
      // speed around the expected physical stall without encoding any object
      // aperture. Contact still determines the achieved angle.
      const double gripper_alpha = std::sqrt(alpha);
      point.positions.push_back(
          GRIPPER_FULLY_OPEN +
          gripper_alpha *
              (GRIPPER_FULLY_CLOSED - GRIPPER_FULLY_OPEN));
      const std::uint64_t elapsed_ns = static_cast<std::uint64_t>(
          std::llround(
              alpha * PICKUP_TRAJECTORY_DURATION_S * 1.0e9));
      point.time_from_start.sec =
          static_cast<std::int32_t>(elapsed_ns / 1000000000ULL);
      point.time_from_start.nanosec =
          static_cast<std::uint32_t>(elapsed_ns % 1000000000ULL);
      point.velocities.clear();
      point.accelerations.clear();
      point.effort.clear();
    }
    if (!arm.execute(pickup_trajectory)) {
      throw std::runtime_error(
          "Synchronized arm/gripper pickup execution failed");
    }
    const auto pickup_achieved = arm.getCurrentPose(IK_LINK).pose;
    if (pickup_achieved.position.z < pickup_start.position.z + 0.010) {
      throw std::runtime_error("Dynamic pickup did not achieve 10 mm lift");
    }
    arm.setMaxVelocityScalingFactor(transport_vel_scale);
    arm.setMaxAccelerationScalingFactor(transport_vel_scale);
    std::this_thread::sleep_for(250ms); // Settle grasp contact forces
    log_info(
        logger,
        "ORIENTATION_COMPENSATION:" + std::to_string(peg_roll) + "," +
        std::to_string(peg_pitch) + "," +
        std::to_string(aligned_peg_offset[0]) + "," +
        std::to_string(aligned_peg_offset[1]));

    // 6. Lift peg
    log_info(logger, "=== 6. LIFT ===");
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt;
      // Rotate the tool during lift so the grasped peg axis becomes vertical.
      // This produces learnable wrist-action supervision for tilted pegs.
      tgt.orientation = aligned_tool_orientation;
      tgt.position.x = corrected_peg_tool_x;
      tgt.position.y = corrected_peg_tool_y;
      tgt.position.z = TOOL0_ABOVE_Z;
      if (!try_cartesian(arm, logger, tgt, IK_LINK, "LIFT")) {
        throw std::runtime_error("LIFT planning/execution failed");
      }
    }
    if (!call_trigger_service(
            confirm_grasp_client, logger,
            "/pap_moe/confirm_grasp_attachment")) {
      throw std::runtime_error("Physical peg grasp was not confirmed");
    }
    if (grasp_recovery_only) {
      log_info(logger, "TASK_RESULT:success");
      log_info(logger, "Grasp recovery prefix finished successfully.");
      executor->cancel();
      spin.join();
      rclcpp::shutdown();
      return 0;
    }
    log_info(logger, "SUBTASK:transport to the hole");

    // 7. Transport above hole
    double force_bias_z = 0.0;
    log_info(logger, "=== 7. ABOVE hole ===");
    {
      const auto start = arm.getCurrentPose(IK_LINK).pose;
      geometry_msgs::msg::Pose target = start;
      target.orientation = aligned_tool_orientation;
      target.position.x = corrected_hole_tool_x;
      target.position.y = corrected_hole_tool_y;
      target.position.z = TOOL0_ABOVE_Z;
      // One Cartesian trajectory avoids both the former 12 stop/start
      // executions and the trajectory-controller timeout seen with a long
      // OMPL point-to-point plan. The 10 mm peg-tip clearance keeps the
      // straight segment clear of both fixtures.
      if (!try_cartesian_arc(
              arm, logger, target, IK_LINK,
              "continuous ABOVE hole transport",
              transport_arc_x, transport_arc_y, transport_arc_z, 0.005)) {
        // At the outer edge of the expanded tabletop domain, a valid endpoint
        // IK can exist even when a single fixed-orientation Cartesian
        // interpolation loses its final few percent.  Preserve the same high,
        // collision-clear target and vertical tool orientation, but let OMPL
        // choose the joint-space approach as a bounded fallback.
        log_info(
            logger,
            "TRANSPORT_FALLBACK:joint-space pose plan to the same high target");
        arm.setPoseTarget(target, IK_LINK);
        const bool fallback_ok = try_move(
            arm, logger, "joint-space ABOVE hole transport fallback");
        arm.clearPoseTargets();
        if (!fallback_ok) {
          throw std::runtime_error(
              "Continuous and joint-space transport above hole both failed");
        }
      }
    }

    // Capture 6D force-torque bias vector
    geometry_msgs::msg::Wrench force_bias;
    {
      std::lock_guard<std::mutex> lock(wrench_mutex);
      force_bias = current_wrench;
    }
    force_bias_z = force_bias.force.z;
    const WrenchVector force_bias_vector = {
      force_bias.force.x, force_bias.force.y, force_bias.force.z,
      force_bias.torque.x, force_bias.torque.y, force_bias.torque.z
    };
    log_info(logger, "C++ Force bias vector: " + 
             std::to_string(force_bias.force.x) + "," +
             std::to_string(force_bias.force.y) + "," +
             std::to_string(force_bias.force.z) + "," +
             std::to_string(force_bias.torque.x) + "," +
             std::to_string(force_bias.torque.y) + "," +
             std::to_string(force_bias.torque.z));

    // 8. APPROACH & Align with random offsets
    RCLCPP_INFO(logger, "STAGE:1"); // s=1: Optical Blind Zone / Alignment
    log_info(logger, "SUBTASK:approach and align with the hole");
    log_info(logger, "=== 8. APPROACH with offset ===");
    
    // Mixed collection policy.  The Python recorder normally expands "mixed"
    // into an explicit per-episode schedule; this fallback remains useful for
    // direct controller invocation.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_mode(0.0, 1.0);
    std::uniform_real_distribution<> dis_angle(0.0, 2.0 * M_PI);
    std::string search_mode;
    double offset_min = 0.0;
    double offset_max = 0.0;
    const double mode_draw = dis_mode(gen);
    if (requested_search_mode == "direct" ||
        (requested_search_mode == "mixed" && mode_draw < 0.25)) {
      search_mode = "direct";
      offset_min = 0.0;
      // Direct episodes are the clean success reference.  The laboratory
      // taper pair has zero designed radial clearance, so do not inject an
      // artificial offset into this branch.
      offset_max = 0.0;
    } else if (requested_search_mode == "force_gradient" ||
               (requested_search_mode == "mixed" && mode_draw < 0.55)) {
      search_mode = "force_gradient";
      // Recovery branches intentionally inject a measurable contact offset.
      offset_max = std::min(0.0020, recovery_offset_max);
      offset_min = std::min(0.0015, 0.80 * offset_max);
    } else if (requested_search_mode == "admittance" ||
               (requested_search_mode == "mixed" && mode_draw < 0.90)) {
      search_mode = "admittance";
      offset_max = std::min(0.0012, recovery_offset_max);
      offset_min = std::min(0.0009, 0.80 * offset_max);
    } else if (requested_search_mode == "spiral" ||
               requested_search_mode == "mixed") {
      search_mode = "spiral";
      offset_min = 0.0030;
      offset_max = std::min(0.018, 1.5 * recovery_offset_max);
    } else {
      log_error(logger, "Invalid search_mode='" + requested_search_mode +
                        "'; expected mixed/direct/force_gradient/admittance/"
                        "spiral.");
      executor->cancel();
      spin.join();
      rclcpp::shutdown();
      return 2;
    }
    std::uniform_real_distribution<> dis_offset(offset_min, offset_max);
    double r_offset = dis_offset(gen);
    double a_offset = dis_angle(gen);
    double offset_x = r_offset * std::cos(a_offset);
    double offset_y = r_offset * std::sin(a_offset);
    
    double target_x = corrected_hole_tool_x + offset_x;
    double target_y = corrected_hole_tool_y + offset_y;
    
    log_info(logger, "SEARCH_MODE:" + search_mode);
    log_info(
        logger,
        "RECOVERY_OFFSET_RANGE:" + std::to_string(offset_min) + "," +
        std::to_string(offset_max));
    log_info(logger, "Injected offset: dx=" + std::to_string(offset_x) +
             ", dy=" + std::to_string(offset_y));

    // Move laterally at a conservative clearance before descending.  A large
    // injected offset can otherwise sweep the peg tip into the socket rim.
    {
      auto current = arm.getCurrentPose(IK_LINK);
      geometry_msgs::msg::Pose tgt = current.pose;
      tgt.position.x = target_x;
      tgt.position.y = target_y;
      tgt.position.z = TOOL0_APPROACH_Z;
      if (!try_cartesian_arc(
              arm, logger, tgt, IK_LINK, "APPROACH with offset",
              hole_arc_x, hole_arc_y, 0.0, 0.002)) {
        // A zero-offset direct episode is already at this pose after the
        // transport segment.  Near the outer workspace boundary MoveIt's
        // Cartesian interpolator can nevertheless report a low fraction for
        // the remaining millimetre-scale tracking error.  The endpoint is
        // known reachable (the preceding transport reached it), so use a
        // pose-plan fallback at the same collision-clear height.  This also
        // supports nonzero recovery offsets without accepting a partial
        // Cartesian execution.
        log_info(
            logger,
            "APPROACH_FALLBACK:joint-space pose plan to the same high target");
        arm.setPoseTarget(tgt, IK_LINK);
        const bool fallback_ok = try_move(
            arm, logger, "joint-space APPROACH with offset fallback");
        arm.clearPoseTargets();
        if (!fallback_ok) {
          throw std::runtime_error(
              "Cartesian and joint-space APPROACH with offset both failed");
        }
      }
    }

    // The simulator exposes one position gain for every controlled joint,
    // including the Robotiq knuckle. Lowering it here also removes most of
    // the physical grip preload and lets the peg slip before insertion.
    // Preserve the validated 0.5 grasp gain; contact safety comes from the
    // monitored low-speed trajectory below, not a global gain change.
    if (!set_sim_position_gain(controller_parameter_client, logger, 0.5)) {
      throw std::runtime_error("Could not preserve physical-grasp gain");
    }

    // MoveIt executes each guarded Cartesian increment synchronously, so the
    // wrench cannot stop a trajectory mid-segment. Use a dedicated low-speed,
    // low-acceleration contact regime instead of the broader randomized
    // search speed; otherwise even a 0.1 mm final increment can create a large
    // inertial rim-contact impulse in Gazebo.
    const double contact_velocity_scale = std::min(vel_scale, 0.01);
    const double contact_acceleration_scale = std::min(vel_scale, 0.005);
    arm.setMaxVelocityScalingFactor(contact_velocity_scale);
    arm.setMaxAccelerationScalingFactor(contact_acceleration_scale);
    log_info(
        logger,
        "CONTACT_MOTION_SCALING:" +
        std::to_string(contact_velocity_scale) + "," +
        std::to_string(contact_acceleration_scale));

    // A MoveIt action can report success while the low-level position loop
    // still has about 1 mm of XY residual.  That is larger than the radial
    // lead-in clearance of the exact-fit tapers after a 7 mm descent. Close
    // the free-space loop against measured tool0 pose before entering the
    // socket; otherwise a nominally direct demonstration strikes the wall.
    constexpr double APPROACH_XY_TOLERANCE = 0.0002;
    bool approach_xy_converged = false;
    for (int correction = 0; correction < 8; ++correction) {
      const auto measured_pose = arm.getCurrentPose(IK_LINK).pose;
      const double xy_error = std::hypot(
          measured_pose.position.x - target_x,
          measured_pose.position.y - target_y);
      log_info(
          logger,
          "APPROACH_XY_TRACKING:attempt=" +
          std::to_string(correction) + ",error=" +
          std::to_string(xy_error));
      if (xy_error <= APPROACH_XY_TOLERANCE) {
        approach_xy_converged = true;
        break;
      }
      geometry_msgs::msg::Pose correction_target = measured_pose;
      correction_target.position.x = target_x;
      correction_target.position.y = target_y;
      correction_target.position.z = TOOL0_APPROACH_Z;
      if (!try_cartesian(
              arm, logger, correction_target, IK_LINK,
              "Free-space approach XY correction", 0.0001, 3.0)) {
        break;
      }
      std::this_thread::sleep_for(100ms);
    }
    if (!approach_xy_converged) {
      throw std::runtime_error(
          "Approach XY did not converge to exact-fit tolerance");
    }

    // A physical grasp does not preserve the theoretical tool-to-peg
    // transform used by the old attached-object implementation. Calibrate
    // the scripted demonstrator against the actual simulated peg axis before
    // descent. The correction is recorded only as an ordinary arm action;
    // peg/hole truth is never added to policy observations or dataset fields.
    // Exact-fit descent needs a substantially tighter pre-contact axis than
    // the final task acceptance window.  The former 0.3 mm tolerance caused
    // a repeatable ~38 N lateral rim impulse and left the peg 1.5 mm high.
    constexpr double PEG_AXIS_ALIGNMENT_TOLERANCE = 0.00008;
    constexpr double PEG_AXIS_DITHER_TRIGGER = 0.00025;
    constexpr double PEG_AXIS_DITHER_DISTANCE = 0.0010;
    bool peg_axis_aligned = false;
    double observed_peg_center_z = 0.0;
    double observed_peg_tilt_rad = 0.0;
    for (int correction = 0; correction < 10; ++correction) {
      double peg_hole_dx = 0.0;
      double peg_hole_dy = 0.0;
      double peg_hole_distance = 0.0;
      if (!query_peg_hole_alignment(
              query_alignment_client, logger, peg_hole_dx, peg_hole_dy,
              peg_hole_distance, observed_peg_center_z,
              observed_peg_tilt_rad)) {
        break;
      }
      if (peg_hole_distance <= PEG_AXIS_ALIGNMENT_TOLERANCE) {
        peg_axis_aligned = true;
        break;
      }
      // On the positive-world-x IK branch, sub-0.2 mm corrections can fall
      // below the simulated controller/contact resolution and alternate on
      // either side of the exact axis.  Move once to a safely observable
      // high-hover offset, then let the ordinary damped calibration converge
      // from that offset.  This is a smooth recorded arm action; insertion
      // force and final geometry gates remain unchanged.
      if (correction == 0 &&
          peg_hole_distance <= PEG_AXIS_DITHER_TRIGGER) {
        target_x += PEG_AXIS_DITHER_DISTANCE *
                    peg_hole_dx / peg_hole_distance;
        target_y += PEG_AXIS_DITHER_DISTANCE *
                    peg_hole_dy / peg_hole_distance;
        geometry_msgs::msg::Pose dither_target =
            arm.getCurrentPose(IK_LINK).pose;
        dither_target.position.x = target_x;
        dither_target.position.y = target_y;
        dither_target.position.z = TOOL0_APPROACH_Z;
        log_info(
            logger,
            "PEG_AXIS_CALIBRATION_DITHER:distance=" +
            std::to_string(PEG_AXIS_DITHER_DISTANCE));
        if (!try_cartesian(
                arm, logger, dither_target, IK_LINK,
                "Physical-grasp peg-axis calibration dither", 0.0001, 2.0)) {
          break;
        }
        std::this_thread::sleep_for(150ms);
        continue;
      }
      // Physical fingertip compliance can make a full 1.0 correction toggle
      // across the hole axis after a curved transport. Use a damped update.
      target_x -= 0.5 * peg_hole_dx;
      target_y -= 0.5 * peg_hole_dy;
      geometry_msgs::msg::Pose correction_target =
          arm.getCurrentPose(IK_LINK).pose;
      correction_target.position.x = target_x;
      correction_target.position.y = target_y;
      correction_target.position.z = TOOL0_APPROACH_Z;
      if (!try_cartesian(
              arm, logger, correction_target, IK_LINK,
              "Physical-grasp peg-axis calibration", 0.0001, 2.0)) {
        break;
      }
      std::this_thread::sleep_for(150ms);
    }
    if (!peg_axis_aligned) {
      throw std::runtime_error(
          "Physical-grasp peg axis did not converge over the hole");
    }

    // Replacing the rigid attachment with a physical grasp makes the
    // tool-to-peg transform episode dependent. Derive all axial targets from
    // the observed peg centre instead of retaining the old 340 mm constant.
    constexpr double PEG_HALF_HEIGHT = 0.090;
    const double calibration_tool_z =
        arm.getCurrentPose(IK_LINK).pose.position.z;
    const double tool_to_peg_center_z =
        calibration_tool_z - observed_peg_center_z;
    const double calibrated_guide_z =
        (RING_TOP_Z - 0.007 + PEG_HALF_HEIGHT) +
        tool_to_peg_center_z;
    const double calibrated_seat_z =
        (RING_BOTTOM_Z + PEG_HALF_HEIGHT) + tool_to_peg_center_z;
    const double calibrated_fine_start_z = calibrated_seat_z + 0.010;
    // Never command the held peg through the rigid blind-hole floor.  The old
    // 1 mm overtravel created solver penetration before the median wrench
    // guard could stop the 100 Hz stream.  End at the calibrated physical
    // seat; normal controller tracking and the carried load provide contact
    // evidence without an object-interpenetration command.
    const double calibrated_force_probe_min_z = calibrated_seat_z;
    const double calibrated_seated_max_z = calibrated_seat_z + 0.0025;
    log_info(
        logger,
        "PHYSICAL_GRASP_AXIAL_CALIBRATION:tool_to_center=" +
        std::to_string(tool_to_peg_center_z) + ",guide_z=" +
        std::to_string(calibrated_guide_z) + ",fine_start_z=" +
        std::to_string(calibrated_fine_start_z) + ",probe_z=" +
        std::to_string(calibrated_force_probe_min_z) + ",seat_z=" +
        std::to_string(calibrated_seat_z) + ",tilt_rad=" +
        std::to_string(observed_peg_tilt_rad));

    // Descend until physical contact is detected.  A coaxial direct episode
    // has substantial transient clearance before the matching tapers seat, so
    // use one smooth segment instead of hundreds of stop/start moves.
    geometry_msgs::msg::Pose current_pose = arm.getCurrentPose(IK_LINK).pose;
    bool contact_detected = false;
    double current_z = TOOL0_APPROACH_Z;

    // With the grasped peg/tool geometry, top contact is expected near
    // tool0 z=1.215.  Continue a bounded 7 mm into the guide region when the
    // peg is centred and therefore generates no top-surface contact.
    if (search_mode == "direct") {
      geometry_msgs::msg::Pose direct_descent = current_pose;
      direct_descent.position.x = target_x;
      direct_descent.position.y = target_y;
      direct_descent.position.z = calibrated_guide_z;
      if (!try_cartesian(
              arm, logger, direct_descent, IK_LINK,
              "Continuous direct guide-in", 0.0005, 2.0)) {
        throw std::runtime_error("Continuous direct guide-in failed");
      }
      std::this_thread::sleep_for(50ms);
      current_z = arm.getCurrentPose(IK_LINK).pose.position.z;
      const double corrected_fz = read_wrench_vector()[2] - force_bias_z;
      // At this depth the thin peg tip has about 9 mm of radial clearance;
      // the direct curriculum offset is bounded to 0.2 mm.  A force-only
      // trigger here is therefore a servo/payload transient, not geometrically
      // possible socket contact.  Keep it in the recorded FT stream but do
      // not turn a centred success demonstration into contact recovery.
      contact_detected = false;
      if (std::abs(corrected_fz) > INSERT_CONTACT_FORCE_MIN) {
        log_info(
            logger,
            "DIRECT_GUIDE_TRANSIENT_IGNORED:corrected_fz=" +
            std::to_string(corrected_fz));
      }
    }
    if (search_mode != "direct") {
      // Generate one smooth descent and monitor it at 100 Hz. The former
      // blocking 0.02--0.1 mm MoveIt calls introduced hundreds of complete
      // stops, making recovery demonstrations slow and visibly jerky.
      // Keep the shared arm/gripper gain unchanged during guarded descent.
      if (!set_sim_position_gain(
              controller_parameter_client, logger, 0.5)) {
        throw std::runtime_error("Could not preserve guarded-descent gain");
      }
      geometry_msgs::msg::Pose descent_target = current_pose;
      descent_target.position.x = target_x;
      descent_target.position.y = target_y;
      // The peg enters the socket thin-end first; a 7 mm guide-in cannot
      // touch the matching taper even with a 2 mm XY offset. Continue toward
      // the coarse-insertion boundary and stop on the first measured force.
      descent_target.position.z = TOOL0_FINE_INSERT_START_Z;
      moveit_msgs::msg::RobotTrajectory descent_seed;
      const double descent_fraction = arm.computeCartesianPath(
          {current_pose, descent_target}, GUIDED_DESCENT_STEP, 0.0,
          descent_seed);
      const auto& seed_points = descent_seed.joint_trajectory.points;
      const auto& seed_names = descent_seed.joint_trajectory.joint_names;
      if (
          descent_fraction < 0.99 || seed_points.empty() ||
          seed_names.empty()) {
        throw std::runtime_error("Guarded continuous descent path failed");
      }
      const auto& start_positions = seed_points.front().positions;
      const auto& end_positions = seed_points.back().positions;
      const std::size_t descent_cycles = std::max<std::size_t>(
          2, static_cast<std::size_t>(std::ceil(
                 (current_pose.position.z - TOOL0_FINE_INSERT_START_Z) /
                 GUIDED_DESCENT_STEP)));
      moveit_msgs::msg::RobotTrajectory guarded_descent;
      guarded_descent.joint_trajectory.joint_names = seed_names;
      for (std::size_t cycle = 0; cycle < descent_cycles; ++cycle) {
        const double alpha =
            static_cast<double>(cycle + 1) /
            static_cast<double>(descent_cycles);
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions.resize(seed_names.size());
        for (std::size_t joint = 0; joint < seed_names.size(); ++joint) {
          point.positions[joint] =
              start_positions[joint] +
              alpha * (end_positions[joint] - start_positions[joint]);
        }
        const std::uint64_t elapsed_ns =
            static_cast<std::uint64_t>(cycle + 1) * 10000000ULL;
        point.time_from_start.sec =
            static_cast<std::int32_t>(elapsed_ns / 1000000000ULL);
        point.time_from_start.nanosec =
            static_cast<std::uint32_t>(elapsed_ns % 1000000000ULL);
        guarded_descent.joint_trajectory.points.push_back(std::move(point));
      }
      if (!arm.asyncExecute(guarded_descent)) {
        throw std::runtime_error("Guarded continuous descent was rejected");
      }
      bool descent_target_reached = false;
      int descent_target_stable_checks = 0;
      for (std::size_t cycle = 0;
           cycle < descent_cycles * 3 + 100; ++cycle) {
        const WrenchVector measured = subtract_wrench(
            read_wrench_vector(), force_bias_vector);
        const double lateral = std::hypot(measured[0], measured[1]);
        if (
            std::abs(measured[2]) > INSERT_CONTACT_FORCE_MIN ||
            lateral > INSERT_CONTACT_FORCE_MIN) {
          contact_detected = true;
          log_info(
              logger,
              "CONTINUOUS_DESCENT_CONTACT:fz=" +
              std::to_string(measured[2]) + ",fxy=" +
              std::to_string(lateral) + ",cycle=" +
              std::to_string(cycle));
          arm.stop();
          break;
        }
        // Observe completion through the independent joint-state subscriber;
        // querying MoveGroupInterface while asyncExecute owns its action
        // client is unsafe. Three consecutive checks prevent a transient
        // tracking crossing from ending force monitoring early.
        if (cycle % 5 == 0) {
          double max_joint_error = 0.0;
          bool all_targets_observed = true;
          {
            std::lock_guard<std::mutex> lock(joint_state_mutex);
            for (std::size_t target_index = 0;
                 target_index < seed_names.size(); ++target_index) {
              const auto found = std::find(
                  latest_joint_names.begin(), latest_joint_names.end(),
                  seed_names[target_index]);
              if (found == latest_joint_names.end()) {
                all_targets_observed = false;
                break;
              }
              const auto state_index = static_cast<std::size_t>(
                  std::distance(latest_joint_names.begin(), found));
              if (state_index >= latest_joint_positions.size()) {
                all_targets_observed = false;
                break;
              }
              max_joint_error = std::max(
                  max_joint_error,
                  std::abs(
                      latest_joint_positions[state_index] -
                      end_positions[target_index]));
            }
          }
          if (all_targets_observed && max_joint_error < 0.003) {
            ++descent_target_stable_checks;
          } else {
            descent_target_stable_checks = 0;
          }
          if (descent_target_stable_checks >= 3) {
            descent_target_reached = true;
            log_info(
                logger,
                "CONTINUOUS_DESCENT_TARGET_REACHED:max_joint_error=" +
                std::to_string(max_joint_error) + ",cycle=" +
                std::to_string(cycle));
            break;
          }
        }
        std::this_thread::sleep_for(10ms);
      }
      if (!contact_detected && !descent_target_reached) {
        arm.stop();
      }
      // asyncExecute still owns the MoveGroup action client briefly after
      // stop(). Querying pose or starting parameter-mediated motion in that
      // window caused a reproducible race/segmentation fault.
      // The gain was already set to 0.5 immediately before this descent.
      // Re-sending the same parameter here blocked for several simulated
      // seconds under exact-fit DART load and recorded a long stationary
      // action run. Wait only for the action client to release before the
      // first post-stop pose query.
      std::this_thread::sleep_for(500ms);
      current_z = arm.getCurrentPose(IK_LINK).pose.position.z;
    }

    if (!contact_detected) {
      log_info(logger, "No surface contact detected during descent. Peg must have entered the hole directly!");
    }

    // 9. Mixed local search to find the hole entrance.
    // With an exact-fit pair, the pre-descent axis calibration can remove the
    // injected XY offset accurately enough that the peg enters without first
    // striking the rim.  Lack of rim contact is therefore not evidence of a
    // failed alignment, even when the requested search mode was non-direct.
    // The later guarded insertion, active seating, model-geometry query and
    // force/overload checks remain mandatory before release, so this does not
    // turn a free fall or an unverified pose into a successful demonstration.
    bool aligned = !contact_detected;
    double aligned_x = target_x;
    double aligned_y = target_y;
    WrenchVector insertion_force_reference = force_bias_vector;
    auto corrected_wrench = [&insertion_force_reference]() {
      return subtract_wrench(
          read_wrench_vector(), insertion_force_reference);
    };
    auto confirm_hole_entry = [&]() {
      const geometry_msgs::msg::Pose start =
          arm.getCurrentPose(IK_LINK).pose;
      // Verify against an absolute insertion depth, not only a relative
      // displacement.  A prior emergency relief can raise the local search
      // plane, where a free 3 mm descent is still above the hole entrance.
      // Descend in guarded 0.5 mm increments so a rim/wall contact aborts
      // before the controller can push through the workpiece.
      constexpr double VERIFY_STEP = 0.0005;
      constexpr double VERIFY_TARGET_Z =
          TOOL0_ENTRY_CONFIRM_MAX_Z - 0.0005;
      constexpr double VERIFY_MAX_FZ = 8.0;
      constexpr double VERIFY_MAX_FXY = 10.0;
      bool motion_ok = true;
      bool force_safe = true;
      WrenchVector measured = corrected_wrench();
      bool verification_depth_reached = false;
      for (int step = 0; step < 64; ++step) {
        const geometry_msgs::msg::Pose before =
            arm.getCurrentPose(IK_LINK).pose;
        if (before.position.z <= VERIFY_TARGET_Z + 0.0002) {
          verification_depth_reached = true;
          break;
        }
        geometry_msgs::msg::Pose verify = before;
        verify.position.z =
            std::max(VERIFY_TARGET_Z, before.position.z - VERIFY_STEP);
        motion_ok = try_cartesian(
            arm, logger, verify, IK_LINK,
            "Guarded hole-entry verification", 0.0003);
        if (!motion_ok) {
          break;
        }
        std::this_thread::sleep_for(35ms);
        measured = corrected_wrench();
        const double lateral_force =
            std::hypot(measured[0], measured[1]);
        if (std::abs(measured[2]) >= VERIFY_MAX_FZ ||
            lateral_force >= VERIFY_MAX_FXY) {
          force_safe = false;
          log_info(
              logger,
              "ALIGNMENT_VERIFY_GUARD:contact," +
              std::to_string(measured[2]) + "," +
              std::to_string(lateral_force));
          break;
        }
      }
      const geometry_msgs::msg::Pose achieved =
          arm.getCurrentPose(IK_LINK).pose;
      verification_depth_reached =
          verification_depth_reached ||
          achieved.position.z <= VERIFY_TARGET_Z + 0.0002;
      measured = corrected_wrench();
      const double z_drop = start.position.z - achieved.position.z;
      const double lateral_force = std::hypot(measured[0], measured[1]);
      const bool confirmed =
          motion_ok && force_safe && verification_depth_reached &&
          achieved.position.z <= TOOL0_ENTRY_CONFIRM_MAX_Z + 0.0002 &&
          std::abs(measured[2]) < VERIFY_MAX_FZ &&
          lateral_force < VERIFY_MAX_FXY;
      log_info(
          logger,
          std::string("ALIGNMENT_VERIFY:") +
          (confirmed ? "success," : "rejected,") +
          std::to_string(z_drop) + "," +
          std::to_string(achieved.position.z) + "," +
          std::to_string(measured[2]) + "," +
          std::to_string(lateral_force) + ",depth=" +
          (verification_depth_reached ? "1" : "0"));
      if (confirmed) {
        // The guarded descent proves that the peg is inside the guide.  Use
        // the calibrated hole centre for the remaining insertion so a small
        // compliance offset is not carried all the way to the socket bottom.
        aligned_x = corrected_hole_tool_x;
        aligned_y = corrected_hole_tool_y;
        current_z = achieved.position.z;
      } else {
        try_cartesian(
            arm, logger, start, IK_LINK,
            "Return after rejected hole-entry verification", 0.0005);
      }
      return confirmed;
    };
    if (aligned) {
      const auto direct_pose = arm.getCurrentPose(IK_LINK).pose;
      aligned_x = direct_pose.position.x;
      aligned_y = direct_pose.position.y;
      log_info(logger, "ALIGNMENT_VERIFY:success,direct_descent");
    }
    if (contact_detected && search_mode == "admittance") {
      RCLCPP_INFO(logger, "STAGE:2");
      log_info(logger, "=== 9. CARTESIAN ADMITTANCE RECOVERY ===");
      log_info(logger, "RECOVERY_EVENT:contact");
      log_info(logger, "SUBTASK:recover contact and relocate the hole");

      // Position-controlled outer-loop admittance:
      //   M e_ddot + D e_dot + K e = F_ext
      // The slowly moving reference is the visual hole estimate; e is a
      // bounded compliance displacement driven by the filtered wrist wrench.
      // MoveIt remains the inner position loop, so all virtual dynamics and
      // per-cycle motion are deliberately conservative.
      constexpr int MAX_ADMITTANCE_ATTEMPTS = 24;
      constexpr double VIRTUAL_MASS_XY = 1.5;       // kg
      constexpr double VIRTUAL_STIFFNESS_XY = 600.0; // N/m
      constexpr double VIRTUAL_DAMPING_XY = 60.0;  // N*s/m, zeta ~= 1
      constexpr double WRENCH_FILTER_ALPHA = 0.35;
      constexpr double REFERENCE_STEP = 0.0006;
      constexpr double MAX_COMPLIANCE = 0.0030;
      constexpr double MAX_LATERAL_SPEED = 0.0030;
      constexpr double MAX_CARTESIAN_STEP = 0.0008;
      constexpr double TARGET_NORMAL_FORCE = 3.0;
      constexpr double MAX_SEARCH_RADIUS = 0.014;
      constexpr double MAX_SAFE_FORCE = 15.0;

      log_info(
          logger,
          "ADMITTANCE_PARAMS:Mxy=1.5,Dxy=60,Kxy=600,zeta=1.0,"
          "Fz_ref=3.0,dx_max=0.0008");

      auto emergency_relief = [&]() {
        geometry_msgs::msg::Pose relief = arm.getCurrentPose(IK_LINK).pose;
        relief.position.z += 0.003;
        log_info(logger, "RECOVERY_SAFETY:high_force_relief");
        return try_cartesian(
            arm, logger, relief, IK_LINK,
            "Admittance emergency relief", 0.0005);
      };

      WrenchVector filtered = corrected_wrench();
      double compliance_x = 0.0;
      double compliance_y = 0.0;
      double velocity_x = 0.0;
      double velocity_y = 0.0;
      double reference_x = target_x;
      double reference_y = target_y;
      const double contact_z =
          arm.getCurrentPose(IK_LINK).pose.position.z;
      auto last_update = std::chrono::steady_clock::now();
      int recovery_attempts = 0;

      for (int attempt = 1;
           attempt <= MAX_ADMITTANCE_ATTEMPTS && !aligned;
           ++attempt) {
        recovery_attempts = attempt;
        const WrenchVector measured = corrected_wrench();
        for (std::size_t axis = 0; axis < filtered.size(); ++axis) {
          filtered[axis] =
              WRENCH_FILTER_ALPHA * measured[axis] +
              (1.0 - WRENCH_FILTER_ALPHA) * filtered[axis];
        }

        const double lateral_sensor =
            std::hypot(filtered[0], filtered[1]);
        if (std::abs(filtered[2]) > MAX_SAFE_FORCE ||
            lateral_sensor > MAX_SAFE_FORCE) {
          if (!emergency_relief()) {
            break;
          }
          // Do not let the pre-relief force impulse remain in the filter and
          // drive the virtual mass after contact has already been unloaded.
          filtered = corrected_wrench();
          velocity_x = 0.0;
          velocity_y = 0.0;
        }

        const auto now = std::chrono::steady_clock::now();
        const double dt = std::clamp(
            std::chrono::duration<double>(now - last_update).count(),
            0.08, 0.35);
        last_update = now;

        const geometry_msgs::msg::Pose current =
            arm.getCurrentPose(IK_LINK).pose;
        const auto force_world = rotate_vector(
            current.orientation, {filtered[0], filtered[1], 0.0});

        const double ref_error_x = corrected_hole_tool_x - reference_x;
        const double ref_error_y = corrected_hole_tool_y - reference_y;
        const double ref_distance =
            std::hypot(ref_error_x, ref_error_y);
        if (ref_distance > 1e-9) {
          const double ref_step = std::min(REFERENCE_STEP, ref_distance);
          reference_x += ref_step * ref_error_x / ref_distance;
          reference_y += ref_step * ref_error_y / ref_distance;
        }
        // A fixed 3 mm compliance allowance creates a steady-state offset at
        // the hole centre: the visual reference arrives at the opening while
        // the virtual spring still holds the tool outside it. Taper the
        // allowable deflection as the reference converges, preserving wall
        // compliance during search but permitting final centring.
        const double compliance_bound = std::min(
            MAX_COMPLIANCE, std::max(0.0002, 0.5 * ref_distance));

        // MoveIt calls make the outer-loop period irregular (roughly
        // 0.1--0.35 s). Integrate the virtual dynamics with bounded 10 ms
        // substeps; a single explicit-Euler step at the measured wall-clock
        // period is unstable for the selected M/D/K.
        double integration_time = dt;
        while (integration_time > 1e-9) {
          const double h = std::min(0.01, integration_time);
          const double acceleration_x =
              (force_world[0] -
               VIRTUAL_DAMPING_XY * velocity_x -
               VIRTUAL_STIFFNESS_XY * compliance_x) /
              VIRTUAL_MASS_XY;
          const double acceleration_y =
              (force_world[1] -
               VIRTUAL_DAMPING_XY * velocity_y -
               VIRTUAL_STIFFNESS_XY * compliance_y) /
              VIRTUAL_MASS_XY;
          velocity_x = std::clamp(
              velocity_x + h * acceleration_x,
              -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED);
          velocity_y = std::clamp(
              velocity_y + h * acceleration_y,
              -MAX_LATERAL_SPEED, MAX_LATERAL_SPEED);
          compliance_x = std::clamp(
              compliance_x + h * velocity_x,
              -compliance_bound, compliance_bound);
          compliance_y = std::clamp(
              compliance_y + h * velocity_y,
              -compliance_bound, compliance_bound);
          integration_time -= h;
        }

        const double unconstrained_x = reference_x + compliance_x;
        const double unconstrained_y = reference_y + compliance_y;
        double delta_x = unconstrained_x - current.position.x;
        double delta_y = unconstrained_y - current.position.y;
        const double cartesian_delta = std::hypot(delta_x, delta_y);
        if (cartesian_delta > MAX_CARTESIAN_STEP) {
          delta_x *= MAX_CARTESIAN_STEP / cartesian_delta;
          delta_y *= MAX_CARTESIAN_STEP / cartesian_delta;
        }

        const double force_magnitude_z = std::abs(filtered[2]);
        const double z_correction = std::clamp(
            (force_magnitude_z - TARGET_NORMAL_FORCE) * 0.00008,
            -0.0002, 0.0008);
        geometry_msgs::msg::Pose target = current;
        target.position.x += delta_x;
        target.position.y += delta_y;
        target.position.z = std::clamp(
            current.position.z + z_correction,
            contact_z - 0.002, contact_z + 0.004);

        const double radius = std::hypot(
            target.position.x - target_x,
            target.position.y - target_y);
        if (radius > MAX_SEARCH_RADIUS) {
          log_info(logger, "Admittance search-radius limit reached.");
          break;
        }
        if (!try_cartesian(
                arm, logger, target, IK_LINK,
                "Admittance recovery step", 0.0004)) {
          break;
        }
        std::this_thread::sleep_for(50ms);

        const WrenchVector after = corrected_wrench();
        const double step_cost = contact_cost(after);
        log_info(
            logger,
            "ADMITTANCE_STATE:" + std::to_string(attempt) + "," +
            std::to_string(dt) + "," +
            std::to_string(filtered[0]) + "," +
            std::to_string(filtered[1]) + "," +
            std::to_string(filtered[2]) + "," +
            std::to_string(compliance_x) + "," +
            std::to_string(compliance_y) + "," +
            std::to_string(delta_x) + "," +
            std::to_string(delta_y));
        log_info(
            logger,
            "RECOVERY_TRIAL:" + std::to_string(attempt) + "," +
            std::to_string(delta_x) + "," +
            std::to_string(delta_y) + "," +
            std::to_string(step_cost) + ",nan,nan");

        const auto verify_pose = arm.getCurrentPose(IK_LINK).pose;
        const double nominal_xy_error = std::hypot(
            verify_pose.position.x - corrected_hole_tool_x,
            verify_pose.position.y - corrected_hole_tool_y);
        if (nominal_xy_error <= 0.0015 &&
            std::abs(after[2]) < 4.0 &&
            std::hypot(after[0], after[1]) < 4.0) {
          aligned = confirm_hole_entry();
          if (aligned) {
            arm.setMaxVelocityScalingFactor(0.15);
            arm.setMaxAccelerationScalingFactor(0.15);
          }
        }
      }

      log_info(
          logger,
          std::string("RECOVERY_RESULT:") +
          (aligned ? "success," : "failure,") +
          std::to_string(recovery_attempts));
    } else if (contact_detected && search_mode == "force_gradient") {
      RCLCPP_INFO(logger, "STAGE:2");
      log_info(logger, "=== 9. HYBRID CONTACT-OPPOSITION RECOVERY ===");
      log_info(logger, "RECOVERY_EVENT:contact");
      log_info(logger, "SUBTASK:recover contact and relocate the hole");

      // Once the injected offset produces rim contact, unload vertically and make one
      // smooth move opposite the wall, toward the visual hole estimate.
      // This produces the desired failure->correction demonstration without
      // dozens of stop/start symmetric MoveIt probes.
      const geometry_msgs::msg::Pose contact_pose =
          arm.getCurrentPose(IK_LINK).pose;
      const WrenchVector contact_wrench = corrected_wrench();
      // The shared gain must remain at the physical-grasp value throughout
      // recovery; arm-only compliance is supplied by bounded motion steps.
      bool recovery_motion_ok = set_sim_position_gain(
          controller_parameter_client, logger, 0.5);
      geometry_msgs::msg::Pose relief_pose = contact_pose;
      relief_pose.position.z += 0.0010;
      if (recovery_motion_ok) {
        recovery_motion_ok = try_cartesian(
            arm, logger, relief_pose, IK_LINK,
            "Contact-opposition vertical unload", 0.0002);
      }

      geometry_msgs::msg::Pose centred_pose = relief_pose;
      centred_pose.position.x = corrected_hole_tool_x;
      centred_pose.position.y = corrected_hole_tool_y;
      const double recovery_dx =
          centred_pose.position.x - contact_pose.position.x;
      const double recovery_dy =
          centred_pose.position.y - contact_pose.position.y;
      if (recovery_motion_ok) {
        recovery_motion_ok = try_cartesian(
            arm, logger, centred_pose, IK_LINK,
            "Continuous opposite-wall correction", 0.0001);
      }
      // Preserve the gripper preload before descending into the guide.
      recovery_motion_ok = set_sim_position_gain(
          controller_parameter_client, logger, 0.5) &&
          recovery_motion_ok;
      log_info(
          logger,
          "RECOVERY_TRIAL:1," + std::to_string(recovery_dx) + "," +
          std::to_string(recovery_dy) + "," +
          std::to_string(contact_cost(contact_wrench)) + ",nan,nan");

      if (recovery_motion_ok) {
        aligned = confirm_hole_entry();
      }
      if (aligned) {
        arm.setMaxVelocityScalingFactor(0.15);
        arm.setMaxAccelerationScalingFactor(0.15);
      }
      log_info(
          logger,
          std::string("RECOVERY_RESULT:") +
          (aligned ? "success," : "failure,") + "1");
    } else if (
        contact_detected && search_mode != "spiral" &&
        search_mode != "force_gradient") {
      RCLCPP_INFO(logger, "STAGE:2"); // s=2: Rigid Constraint / Recovery
      log_info(logger, "=== 9. FORCE-GRADIENT CONTACT RECOVERY ===");
      log_info(logger, "RECOVERY_EVENT:contact");
      log_info(logger, "SUBTASK:recover contact and relocate the hole");

      constexpr int MAX_RECOVERY_ATTEMPTS = 24;
      constexpr double PROBE_STEP = 0.0006;
      constexpr double MAX_SEARCH_RADIUS = 0.012;
      double search_x = target_x;
      double search_y = target_y;
      // Unload the first contact slightly before lateral probing.  Pressing
      // another 0.1 mm into the rim amplified the DART contact impulse and
      // made recovery abort before it could compare the two directions.
      double slide_z = current_z + 0.0001;
      int recovery_attempts = 0;

      auto emergency_relief = [&]() {
        geometry_msgs::msg::Pose relief = arm.getCurrentPose(IK_LINK).pose;
        relief.position.z += 0.003;
        log_info(logger, "RECOVERY_SAFETY:high_force_relief");
        return try_cartesian(
            arm, logger, relief, IK_LINK, "Emergency force relief", 0.0005);
      };

      auto move_and_measure =
          [&](double x, double y, const std::string& description,
              WrenchVector& measured) {
            geometry_msgs::msg::Pose probe = arm.getCurrentPose(IK_LINK).pose;
            probe.position.x = x;
            probe.position.y = y;
            probe.position.z = slide_z;
            if (!try_cartesian(
                    arm, logger, probe, IK_LINK, description, 0.0005)) {
              return false;
            }
            std::this_thread::sleep_for(40ms);
            measured = corrected_wrench();
            const double lateral_force =
                std::hypot(measured[0], measured[1]);
            if (std::abs(measured[2]) > 15.0 || lateral_force > 15.0) {
              emergency_relief();
              return false;
            }
            return true;
          };

      for (int attempt = 1;
           attempt <= MAX_RECOVERY_ATTEMPTS && !aligned;
           ++attempt) {
        recovery_attempts = attempt;
        const WrenchVector before = corrected_wrench();
        const double before_cost = contact_cost(before);

        // Use lateral force as the candidate axis. Probe both signs, so the
        // algorithm remains correct even if the sensor reaction sign differs.
        const auto current_pose_for_axis = arm.getCurrentPose(IK_LINK).pose;
        auto force_world = rotate_vector(
            current_pose_for_axis.orientation,
            {before[0], before[1], 0.0});
        double axis_x = force_world[0];
        double axis_y = force_world[1];
        const double lateral_world = std::hypot(axis_x, axis_y);
        if (lateral_world > 0.8) {
          axis_x /= lateral_world;
          axis_y /= lateral_world;
        } else {
          // A flat rim can produce axial contact with almost no trustworthy
          // lateral gradient.  In that case use the visual hole estimate as
          // the probe axis; the symmetric force-cost comparison still gets
          // the final say between motion toward and away from that estimate.
          axis_x = corrected_hole_tool_x - search_x;
          axis_y = corrected_hole_tool_y - search_y;
          const double visual_distance = std::hypot(axis_x, axis_y);
          if (visual_distance > 1e-6) {
            axis_x /= visual_distance;
            axis_y /= visual_distance;
          } else {
            axis_x = 1.0;
            axis_y = 0.0;
          }
        }

        if (std::abs(before[2]) > 15.0 ||
            std::hypot(before[0], before[1]) > 15.0) {
          // Preserve the contact direction before unloading.  Move a small
          // step along the measured reaction direction after lifting, rather
          // than aborting at the first wall contact.  Subsequent symmetric
          // probes remain sign-robust if the sensor convention is inverted.
          if (!emergency_relief()) {
            break;
          }
          slide_z = arm.getCurrentPose(IK_LINK).pose.position.z;
          search_x += PROBE_STEP * axis_x;
          search_y += PROBE_STEP * axis_y;
          geometry_msgs::msg::Pose escape =
              arm.getCurrentPose(IK_LINK).pose;
          escape.position.x = search_x;
          escape.position.y = search_y;
          escape.position.z = slide_z;
          if (!try_cartesian(
                  arm, logger, escape, IK_LINK,
                  "Recovery post-relief lateral step", 0.0005)) {
            break;
          }
          log_info(
              logger,
              "RECOVERY_TRIAL:" + std::to_string(attempt) + "," +
              std::to_string(PROBE_STEP * axis_x) + "," +
              std::to_string(PROBE_STEP * axis_y) + "," +
              std::to_string(before_cost) + ",nan,nan");
          continue;
        }

        const double plus_x = search_x + PROBE_STEP * axis_x;
        const double plus_y = search_y + PROBE_STEP * axis_y;
        const double minus_x = search_x - PROBE_STEP * axis_x;
        const double minus_y = search_y - PROBE_STEP * axis_y;

        WrenchVector plus_wrench{};
        WrenchVector minus_wrench{};
        if (!move_and_measure(
                plus_x, plus_y, "Recovery probe +", plus_wrench)) {
          break;
        }
        // Return to the same origin before evaluating the opposite direction.
        WrenchVector origin_wrench{};
        if (!move_and_measure(
                search_x, search_y, "Recovery probe return", origin_wrench)) {
          break;
        }
        if (!move_and_measure(
                minus_x, minus_y, "Recovery probe -", minus_wrench)) {
          break;
        }

        const double plus_cost = contact_cost(plus_wrench);
        const double minus_cost = contact_cost(minus_wrench);
        // Resolve nearly equal force costs with the visual hole estimate.
        // A clearly safer force direction still dominates, while flat-rim
        // contacts no longer random-walk away from the visible socket.
        constexpr double VISUAL_DISTANCE_COST = 40.0;
        const double plus_score =
            plus_cost + VISUAL_DISTANCE_COST * std::hypot(
                plus_x - corrected_hole_tool_x,
                plus_y - corrected_hole_tool_y);
        const double minus_score =
            minus_cost + VISUAL_DISTANCE_COST * std::hypot(
                minus_x - corrected_hole_tool_x,
                minus_y - corrected_hole_tool_y);
        const bool choose_plus = plus_score <= minus_score;
        if (choose_plus) {
          search_x = plus_x;
          search_y = plus_y;
        } else {
          search_x = minus_x;
          search_y = minus_y;
        }

        WrenchVector chosen_wrench{};
        if (!move_and_measure(
                search_x, search_y, "Recovery chosen step", chosen_wrench)) {
          break;
        }

        const double chosen_dx =
            (choose_plus ? axis_x : -axis_x) * PROBE_STEP;
        const double chosen_dy =
            (choose_plus ? axis_y : -axis_y) * PROBE_STEP;
        log_info(
            logger,
            "RECOVERY_TRIAL:" + std::to_string(attempt) + "," +
            std::to_string(chosen_dx) + "," + std::to_string(chosen_dy) + "," +
            std::to_string(before_cost) + "," + std::to_string(plus_cost) + "," +
            std::to_string(minus_cost));

        const double distance_from_start =
            std::hypot(search_x - target_x, search_y - target_y);
        if (distance_from_start > MAX_SEARCH_RADIUS) {
          log_info(logger, "Recovery search-radius limit reached.");
          break;
        }

        // Force relief is only a candidate. Confirm it by moving 3 mm farther
        // down at the discovered XY without excessive contact force.
        const double nominal_xy_error = std::hypot(
            search_x - corrected_hole_tool_x,
            search_y - corrected_hole_tool_y);
        if (
            nominal_xy_error <= 0.0015 &&
            std::abs(chosen_wrench[2]) < 4.0 &&
            std::hypot(chosen_wrench[0], chosen_wrench[1]) < 4.0) {
          aligned = confirm_hole_entry();
          if (aligned) {
            arm.setMaxVelocityScalingFactor(0.15);
            arm.setMaxAccelerationScalingFactor(0.15);
          }
        }
      }

      log_info(
          logger,
          std::string("RECOVERY_RESULT:") +
          (aligned ? "success," : "failure,") +
          std::to_string(recovery_attempts));
    } else if (contact_detected) {
      RCLCPP_INFO(logger, "STAGE:2"); // s=2: Rigid Constraint (Spiral Search)
      log_info(logger, "=== 9. ACTIVE SPIRAL SEARCH ===");
      log_info(logger, "RECOVERY_EVENT:contact");
      log_info(logger, "SUBTASK:recover contact and relocate the hole");
      double theta = 0.0;
      double r = 0.0;
      int spiral_attempts = 0;
      double previous_sx = target_x;
      double previous_sy = target_y;
      
      // Slide at the detected contact height minus 0.1mm to maintain light physical contact
      double slide_z = current_z - 0.0001;
      
      while (!aligned && r < 0.008) {
        ++spiral_attempts;
        theta += 0.3;         // spiral angle step
        r += 0.00015;         // expanding radius
        
        // Search opposite of offset direction
        double sx = target_x - r * std::cos(theta + a_offset);
        double sy = target_y - r * std::sin(theta + a_offset);
        log_info(
            logger,
            "RECOVERY_TRIAL:" + std::to_string(spiral_attempts) + "," +
            std::to_string(sx - previous_sx) + "," +
            std::to_string(sy - previous_sy) + ",nan,nan,nan");
        previous_sx = sx;
        previous_sy = sy;
        
        // Read vertical force before moving
        double fz = 0.0;
        {
          std::lock_guard<std::mutex> lock(wrench_mutex);
          fz = current_wrench.force.z;
        }
        double fz_corrected = fz - force_bias_z;
        
        // If contact force is already too large (> 10.0N corrected), lift up dynamically, shift XY, and descend again
        if (std::abs(fz_corrected) > 10.0) {
          log_info(logger, "Force too high (" + std::to_string(fz_corrected) + "N). Lifting dynamically to relieve pressure...");
          
          double relief_z = slide_z;
          bool relieved = false;
          while (relief_z < slide_z + 0.008 && !relieved) {
            relief_z += 0.0005; // 0.5mm step
            geometry_msgs::msg::Pose lift_tgt = arm.getCurrentPose(IK_LINK).pose;
            lift_tgt.position.z = relief_z;
            if (!try_cartesian(arm, logger, lift_tgt, IK_LINK, "Relief lift step", 0.001)) {
              break;
            }
            
            double rfz = 0.0;
            {
              std::lock_guard<std::mutex> lock(wrench_mutex);
              rfz = current_wrench.force.z;
            }
            double rfz_corrected = rfz - force_bias_z;
            if (std::abs(rfz_corrected) < 2.0) {
              relieved = true;
              log_info(logger, "Pressure relieved. Stopped lifting at delta: " + std::to_string(relief_z - slide_z) + "m");
            }
            std::this_thread::sleep_for(15ms);
          }
          
          if (!relieved) {
            log_info(logger, "Fatal: Gripper lost peg during contact (force not relieved)! Aborting trajectory to prevent infinite lift.");
            break; // Abort spiral search
          }
          
          slide_z = relief_z; // Update slide_z to the relieved height
          
          // 2. Move to new XY in the air
          geometry_msgs::msg::Pose air_tgt = arm.getCurrentPose(IK_LINK).pose;
          air_tgt.position.x = sx;
          air_tgt.position.y = sy;
          try_cartesian(arm, logger, air_tgt, IK_LINK, "XY shift", 0.002);
          
          // 3. Descend until contact
          double desc_z = slide_z;
          bool desc_contact = false;
          while (desc_z > slide_z - 0.003 && !desc_contact) {
            desc_z -= 0.0004; // 0.4mm steps
            geometry_msgs::msg::Pose desc_tgt = air_tgt;
            desc_tgt.position.z = desc_z;
            if (!try_cartesian(arm, logger, desc_tgt, IK_LINK, "Relief descent", 0.001)) {
              break;
            }
            
            double dfz = 0.0;
            {
              std::lock_guard<std::mutex> lock(wrench_mutex);
              dfz = current_wrench.force.z;
            }
            double dfz_corrected = dfz - force_bias_z;
            if (std::abs(dfz_corrected) > 4.0) {
              desc_contact = true;
              slide_z = desc_z; // Update slide_z to the new contact Z!
              log_info(logger, "Contact re-established at Z: " + std::to_string(slide_z));
            }
            std::this_thread::sleep_for(15ms);
          }
        } 
        else {
          // Force is normal, just slide horizontally to next XY
          geometry_msgs::msg::Pose search_tgt = arm.getCurrentPose(IK_LINK).pose;
          search_tgt.position.x = sx;
          search_tgt.position.y = sy;
          search_tgt.position.z = slide_z;
          
          if (!try_cartesian(arm, logger, search_tgt, IK_LINK, "Searching", 0.001)) {
            break;
          }
        }

        // Monitor vertical force for drop (alignment detection)
        double test_fz = 0.0;
        {
          std::lock_guard<std::mutex> lock(wrench_mutex);
          test_fz = current_wrench.force.z;
        }
        double test_fz_corrected = test_fz - force_bias_z;
        
        // Treat a force drop as a candidate only; require a 3 mm guide-in
        // verification at this exact XY before declaring alignment.
        if (std::abs(test_fz_corrected) < 4.0) {
          aligned = confirm_hole_entry();
          if (aligned) {
            log_info(
                logger,
                "Alignment confirmed after spiral. Corrected Fz: " +
                std::to_string(test_fz_corrected));
            arm.setMaxVelocityScalingFactor(0.15);
            arm.setMaxAccelerationScalingFactor(0.15);
          }
        }
        std::this_thread::sleep_for(30ms);
      }
      log_info(
          logger,
          std::string("RECOVERY_RESULT:") +
          (aligned ? "success," : "failure,") +
          std::to_string(spiral_attempts));
    }

    bool assembly_confirmed = false;
    bool insertion_geometry_confirmed = false;
    bool insertion_model_geometry_confirmed = false;
    bool insertion_contact_confirmed = false;
    bool geometric_seating_without_force = false;
    bool insertion_depth_reached = false;
    bool insertion_force_plateau_confirmed = false;
    bool insertion_force_relaxation_confirmed = false;
    bool active_seating_confirmed = false;
    bool verify_subtask_announced = false;
    if (aligned) {
      // The physically held payload can settle inside the fingers while the
      // direct guide-in changes controller gain.  That changes the static
      // wrist load even though the tool orientation is unchanged.  For the
      // geometrically coaxial direct branch the peg tip is still in the
      // wide, contact-free part of the taper here, so establish a local
      // controller guard reference after it has settled.  This reference is
      // used only by the scripted safety controller: the dataset continues
      // to store raw FT300 plus the common episode payload calibration.
      if (search_mode == "direct") {
        std::this_thread::sleep_for(250ms);
        insertion_force_reference = read_wrench_vector();
        log_info(
            logger,
            "INSERT_GUARD_REFERENCE:" +
            std::to_string(insertion_force_reference[0]) + "," +
            std::to_string(insertion_force_reference[1]) + "," +
            std::to_string(insertion_force_reference[2]) + "," +
            std::to_string(insertion_force_reference[3]) + "," +
            std::to_string(insertion_force_reference[4]) + "," +
            std::to_string(insertion_force_reference[5]));
      }
      // 10. INSERT peg (constrained slide-in)
      log_info(logger, "=== 10. INSERT ===");
      log_info(logger, "SUBTASK:insert the peg into the hole");
      RCLCPP_INFO(logger, "STAGE:3");
      bool insert_motion_ok = true;
      bool fine_insert_announced = false;
      bool insertion_overload = false;
      int insertion_steps = 0;
      int stagnant_insertion_steps = 0;
      double contact_axial_force = 0.0;
      double contact_lateral_force = 0.0;
      {
        auto current = arm.getCurrentPose(IK_LINK).pose;
        while (current.position.z >
               calibrated_force_probe_min_z + 0.0001 && insert_motion_ok) {
          if (++insertion_steps > 100) {
            log_info(
                logger,
                "INSERT_PROGRESS_GUARD:max_steps,z=" +
                std::to_string(current.position.z));
            insert_motion_ok = false;
            break;
          }
          // Never execute an unmonitored blocking descent in a zero-clearance
          // pair.  The streaming branch cruises through the open taper and
          // automatically changes to 5 um/cycle for the final 15 mm.
          const bool fine_insert = true;
          if (fine_insert && !fine_insert_announced) {
            log_info(logger, "=== 11. FORCE-LIMITED INSERTION ===");
            fine_insert_announced = true;
          }
          if (fine_insert) {
            // MoveIt execute() is blocking: at the previous 0.25 mm step the
            // contact impulse had already reached 200 N before wrench was
            // checked. Stream partial joint targets at 100 Hz instead. Each
            // The target is seeded from the measured state and advances by a
            // bounded configurable increment. Wrench is still checked every
            // 10 ms, preserving the one-controller-cycle contact stop.
            log_info(
                logger,
                "INSERT_STREAMING_ADMITTANCE:start,rate=100Hz,step_m=" +
                std::to_string(fine_insertion_step));
            const auto joint_model_group =
                arm.getRobotModel()->getJointModelGroup(ARM_GROUP);
            if (joint_model_group == nullptr) {
              log_error(logger, "Missing arm joint model group");
              insert_motion_ok = false;
              break;
            }
            geometry_msgs::msg::Pose servo_target = current;
            servo_target.position.x = aligned_x;
            servo_target.position.y = aligned_y;
            servo_target.position.z = calibrated_force_probe_min_z;
            std::vector<geometry_msgs::msg::Pose> servo_waypoints{
                current, servo_target};
            moveit_msgs::msg::RobotTrajectory servo_path;
            const double servo_fraction = arm.computeCartesianPath(
                servo_waypoints, fine_insertion_step, 0.0, servo_path);
            const auto& servo_points =
                servo_path.joint_trajectory.points;
            const auto& servo_joint_names =
                servo_path.joint_trajectory.joint_names;
            if (
                servo_fraction < 0.99 || servo_points.empty() ||
                servo_joint_names.empty()) {
              log_error(
                  logger,
                  "INSERT_STREAMING_ADMITTANCE:path generation failed");
              insert_motion_ok = false;
              break;
            }
            const auto& start_positions =
                servo_points.front().positions;
            const auto& end_positions =
                servo_points.back().positions;
            if (
                start_positions.size() != servo_joint_names.size() ||
                end_positions.size() != servo_joint_names.size()) {
              log_error(
                  logger,
                  "INSERT_STREAMING_ADMITTANCE:path dimension mismatch");
              insert_motion_ok = false;
              break;
            }
            const double insertion_distance =
                current.position.z - calibrated_force_probe_min_z;
            const double cruise_distance = std::max(
                0.0,
                insertion_distance - INSERT_CONTACT_APPROACH_DISTANCE);
            const double contact_approach_distance =
                insertion_distance - cruise_distance;
            const std::size_t cruise_cycle_count =
                static_cast<std::size_t>(
                    std::ceil(cruise_distance / INSERT_FREE_CRUISE_STEP));
            const std::size_t contact_cycle_count =
                static_cast<std::size_t>(std::ceil(
                    contact_approach_distance /
                    fine_insertion_step));
            const std::size_t servo_cycle_count = std::max<std::size_t>(
                2, cruise_cycle_count + contact_cycle_count);
            log_info(
                logger,
                "INSERT_STREAMING_ADMITTANCE:path_points=" +
                std::to_string(servo_points.size()) +
                ",stream_cycles=" +
                std::to_string(servo_cycle_count) +
                ",cruise_cycles=" +
                std::to_string(cruise_cycle_count) +
                ",contact_cycles=" +
                std::to_string(contact_cycle_count));
            moveit_msgs::msg::RobotTrajectory guarded_path;
            guarded_path.joint_trajectory.joint_names =
                servo_joint_names;
            for (std::size_t cycle = 0;
                 cycle < servo_cycle_count; ++cycle) {
              double commanded_distance = 0.0;
              if (cycle < cruise_cycle_count) {
                commanded_distance = std::min(
                    cruise_distance,
                    static_cast<double>(cycle + 1) *
                        INSERT_FREE_CRUISE_STEP);
              } else {
                const std::size_t contact_cycle =
                    cycle - cruise_cycle_count;
                commanded_distance = cruise_distance + std::min(
                    contact_approach_distance,
                    static_cast<double>(contact_cycle + 1) *
                        fine_insertion_step);
              }
              const double alpha = std::min(
                  1.0, commanded_distance / insertion_distance);
              // Preserve the Cartesian path computed by MoveIt.  Interpolating
              // only between its first and last joint vectors makes a straight
              // line in joint space, which produced millimetres of tool XY
              // drift over the long insertion stroke.  Interpolate locally
              // between neighbouring Cartesian-path samples instead.
              const double path_position =
                  alpha * static_cast<double>(servo_points.size() - 1);
              const std::size_t path_index = std::min<std::size_t>(
                  static_cast<std::size_t>(std::floor(path_position)),
                  servo_points.size() - 1);
              const std::size_t next_path_index = std::min<std::size_t>(
                  path_index + 1, servo_points.size() - 1);
              const double local_alpha =
                  path_position - static_cast<double>(path_index);
              trajectory_msgs::msg::JointTrajectoryPoint point;
              point.positions.resize(servo_joint_names.size());
              for (std::size_t joint = 0;
                   joint < point.positions.size(); ++joint) {
                point.positions[joint] =
                    servo_points[path_index].positions[joint] +
                    local_alpha *
                        (servo_points[next_path_index].positions[joint] -
                         servo_points[path_index].positions[joint]);
              }
              const std::uint64_t elapsed_ns =
                  static_cast<std::uint64_t>(cycle + 1) * 10000000ULL;
              point.time_from_start.sec =
                  static_cast<std::int32_t>(elapsed_ns / 1000000000ULL);
              point.time_from_start.nanosec =
                  static_cast<std::uint32_t>(elapsed_ns % 1000000000ULL);
              guarded_path.joint_trajectory.points.push_back(
                  std::move(point));
            }
            if (!arm.asyncExecute(guarded_path)) {
              log_error(
                  logger,
                  "INSERT_STREAMING_ADMITTANCE:execution rejected");
              insert_motion_ok = false;
              break;
            }
            bool servo_stopped = false;
            bool contact_stop_announced = false;
            int insertion_target_stable_checks = 0;
            std::vector<double> streaming_axial_samples;
            std::vector<double> streaming_lateral_samples;
            streaming_axial_samples.reserve(INSERT_VERIFY_FORCE_SAMPLES);
            streaming_lateral_samples.reserve(INSERT_VERIFY_FORCE_SAMPLES);
            for (std::size_t cycle = 0;
                 // Trajectory timestamps use simulation time while this
                 // watchdog sleeps on wall time. Headless camera rendering
                 // can run below real-time, so allow 3x wall-time margin.
                 cycle < servo_cycle_count * 3 + 100; ++cycle) {
              const WrenchVector measured = corrected_wrench();
              const double axial_force = std::abs(measured[2]);
              const double lateral_force =
                  std::hypot(measured[0], measured[1]);
              insertion_overload =
                  axial_force > INSERT_AXIAL_FORCE_MAX ||
                  lateral_force > INSERT_LATERAL_FORCE_MAX;
              if (insertion_overload) {
                contact_axial_force = axial_force;
                contact_lateral_force = lateral_force;
                log_info(
                    logger,
                    std::string("INSERT_SERVO_FORCE_GUARD:overload,") +
                    "fz=" + std::to_string(measured[2]) +
                    ",fxy=" + std::to_string(lateral_force) +
                    ",cycle=" + std::to_string(cycle));
                arm.stop();
                servo_stopped = true;
                break;
              }

              if (
                  axial_force >= INSERT_SEAT_CONTACT_FORCE_MIN ||
                  lateral_force >= INSERT_LATERAL_CONTACT_MIN) {
                insertion_contact_confirmed = true;
                contact_axial_force = axial_force;
                contact_lateral_force = lateral_force;
                if (!contact_stop_announced) {
                  log_info(
                      logger,
                      "INSERT_SERVO_CONTACT:detected,fz=" +
                      std::to_string(measured[2]) + ",fxy=" +
                      std::to_string(lateral_force) + ",cycle=" +
                      std::to_string(cycle));
                  contact_stop_announced = true;
                }
                // Stop at first median-confirmed contact.  The subsequent
                // stationary hold collects the 25-sample force plateau;
                // continuing to advance after contact only adds penetration.
                arm.stop();
                servo_stopped = true;
                break;
              }

              if (
                  insertion_contact_confirmed &&
                  axial_force >= INSERT_VERIFY_FORCE_MIN) {
                streaming_axial_samples.push_back(axial_force);
                streaming_lateral_samples.push_back(lateral_force);
                if (
                    streaming_axial_samples.size() >
                    static_cast<std::size_t>(INSERT_VERIFY_FORCE_SAMPLES)) {
                  streaming_axial_samples.erase(
                      streaming_axial_samples.begin());
                  streaming_lateral_samples.erase(
                      streaming_lateral_samples.begin());
                }
              } else if (insertion_contact_confirmed) {
                streaming_axial_samples.clear();
                streaming_lateral_samples.clear();
              }

              if (
                  streaming_axial_samples.size() ==
                  static_cast<std::size_t>(INSERT_VERIFY_FORCE_SAMPLES)) {
                double axial_sum = 0.0;
                double lateral_sum = 0.0;
                for (std::size_t index = 0;
                     index < streaming_axial_samples.size(); ++index) {
                  axial_sum += streaming_axial_samples[index];
                  lateral_sum += streaming_lateral_samples[index];
                }
                const double axial_mean =
                    axial_sum / streaming_axial_samples.size();
                const double lateral_mean =
                    lateral_sum / streaming_lateral_samples.size();
                double variance = 0.0;
                for (const double value : streaming_axial_samples) {
                  const double error = value - axial_mean;
                  variance += error * error;
                }
                const double axial_std = std::sqrt(
                    variance / streaming_axial_samples.size());
                insertion_force_plateau_confirmed =
                    axial_mean <= INSERT_AXIAL_FORCE_MAX &&
                    axial_std <= INSERT_VERIFY_FORCE_STD_MAX &&
                    lateral_mean <= INSERT_VERIFY_LATERAL_MEAN_MAX;
                if (insertion_force_plateau_confirmed) {
                  log_info(
                      logger,
                      "INSERT_STREAMING_FORCE_PLATEAU:success,fz_mean=" +
                      std::to_string(axial_mean) + ",fz_std=" +
                      std::to_string(axial_std) + ",fxy_mean=" +
                      std::to_string(lateral_mean));
                  arm.stop();
                  servo_stopped = true;
                  break;
                }
              }

              // MoveGroupInterface is not safe for concurrent state queries
              // while asyncExecute owns its action client.  The independent
              // /joint_states subscriber is safe, however, and lets the force
              // watchdog stop as soon as the precomputed endpoint has really
              // been reached.  Without this check a no-contact exact-fit
              // insertion sat still until the conservative 3x wall-time
              // timeout, adding about 27 repeated 10 Hz policy frames.
              if (cycle % 5 == 0) {
                double max_joint_error = 0.0;
                bool all_targets_observed = true;
                {
                  std::lock_guard<std::mutex> lock(joint_state_mutex);
                  for (std::size_t target_index = 0;
                       target_index < servo_joint_names.size();
                       ++target_index) {
                    const auto found = std::find(
                        latest_joint_names.begin(), latest_joint_names.end(),
                        servo_joint_names[target_index]);
                    if (found == latest_joint_names.end()) {
                      all_targets_observed = false;
                      break;
                    }
                    const auto state_index = static_cast<std::size_t>(
                        std::distance(latest_joint_names.begin(), found));
                    if (state_index >= latest_joint_positions.size()) {
                      all_targets_observed = false;
                      break;
                    }
                    max_joint_error = std::max(
                        max_joint_error,
                        std::abs(
                            latest_joint_positions[state_index] -
                            end_positions[target_index]));
                  }
                }
                if (all_targets_observed && max_joint_error < 0.0002) {
                  ++insertion_target_stable_checks;
                } else {
                  insertion_target_stable_checks = 0;
                }
                if (insertion_target_stable_checks >= 3) {
                  log_info(
                      logger,
                      "INSERT_STREAMING_TARGET_REACHED:max_joint_error=" +
                      std::to_string(max_joint_error) + ",cycle=" +
                      std::to_string(cycle));
                  break;
                }
              }
              std::this_thread::sleep_for(10ms);
            }
            if (!servo_stopped) {
              arm.stop();
            }
            // asyncExecute can retain the action client briefly after stop.
            // Let it settle before querying the current state.
            std::this_thread::sleep_for(250ms);
            current = arm.getCurrentPose(IK_LINK).pose;
            break;
          }
          const double step_z =
              fine_insert ? INSERT_FINE_STEP_Z : INSERT_COARSE_STEP_Z;
          const double previous_z = current.position.z;
          geometry_msgs::msg::Pose tgt = current;
          tgt.position.x = aligned_x;
          tgt.position.y = aligned_y;
          tgt.position.z =
              std::max(
                  calibrated_force_probe_min_z,
                  current.position.z - step_z);
          if (!try_cartesian(
                  arm, logger, tgt, IK_LINK,
                  fine_insert ? "SEAT micro-step" : "INSERT step",
                  fine_insert ? 0.0002 : 0.0005)) {
            insert_motion_ok = false;
            break;
          }

          std::this_thread::sleep_for(50ms);
          current = arm.getCurrentPose(IK_LINK).pose;
          const double downward_progress = previous_z - current.position.z;
          if (downward_progress < 0.00002) {
            ++stagnant_insertion_steps;
          } else {
            stagnant_insertion_steps = 0;
          }
          if (stagnant_insertion_steps >= 5) {
            if (current.position.z <=
                calibrated_force_probe_min_z + 0.0003) {
              log_info(
                  logger,
                  "INSERT_PROGRESS_GUARD:depth_reached,z=" +
                  std::to_string(current.position.z));
            } else {
              log_info(
                  logger,
                  "INSERT_PROGRESS_GUARD:stagnant,z=" +
                  std::to_string(current.position.z) + ",progress=" +
                  std::to_string(downward_progress));
              insert_motion_ok = false;
            }
            break;
          }
          const WrenchVector step_wrench = corrected_wrench();
          const double axial_force = std::abs(step_wrench[2]);
          const double lateral_force =
              std::hypot(step_wrench[0], step_wrench[1]);

          // Stop at the first gentle axial contact. Excessive axial/lateral
          // contact is rejected; a free insertion can instead terminate at
          // the safe pre-bottom depth and pass the geometry+wrench gates.
          if (axial_force >= INSERT_SEAT_CONTACT_FORCE_MIN ||
              lateral_force >= INSERT_LATERAL_CONTACT_MIN) {
            insertion_overload =
                axial_force > INSERT_AXIAL_FORCE_MAX ||
                lateral_force > INSERT_LATERAL_FORCE_MAX;
            insertion_contact_confirmed =
                (axial_force >= INSERT_SEAT_CONTACT_FORCE_MIN ||
                 lateral_force >= INSERT_LATERAL_CONTACT_MIN) &&
                !insertion_overload;
            contact_axial_force = axial_force;
            contact_lateral_force = lateral_force;
            log_info(
                logger,
                std::string(
                    insertion_overload ?
                    "INSERT_FORCE_GUARD:overload,z=" :
                    "INSERT_CONTACT_STOP:success,z=") +
                std::to_string(current.position.z) + ",fz=" +
                std::to_string(step_wrench[2]) + ",fxy=" +
                std::to_string(lateral_force));
            break;
          }
        }
      }
      // DART's ideal rigid constraint can report a short 100+ N solver
      // impulse at first seating contact even though the stopped system
      // immediately settles to a safe load.  The force guard has already
      // stopped motion.  Accept only if a delayed 25-sample stationary hold
      // is bounded; sustained overload and rim contact still fail here or at
      // the independent geometry service below.
      if (insert_motion_ok && insertion_overload) {
        // Force relief is still part of insertion.  The following stationary
        // samples are the actual verification evidence and must carry the
        // verification subtask label for long enough to supervise the
        // subtask head.
        if (insert_motion_ok && fine_insert_announced) {
          log_info(logger, "SUBTASK:verify insertion success");
          verify_subtask_announced = true;
        }
        std::this_thread::sleep_for(150ms);
        constexpr double RELIEF_TARGET_FORCE = 8.0;
        constexpr double RELIEF_TARGET_LATERAL_FORCE = 8.0;
        constexpr double RELIEF_STEP_Z = 0.00005;
        constexpr double RELIEF_XY_TOLERANCE = 0.00008;
        constexpr double RELIEF_XY_STEP_MAX = 0.00010;
        // A bottom-contact impulse can leave the exactly fitted peg carrying
        // little axial load but a large lateral constraint load.  Lifting in
        // Z alone cannot release that jam.  Use the same scripted alignment
        // truth that generated the successful demonstration to recenter the
        // held peg while backing off.  It remains controller-only privileged
        // information and is never exposed in a policy observation.
        // The zero-clearance taper can retain lateral elastic load after the
        // peg itself is already seated.  Allow up to 2.4 mm of tool/gripper
        // back-off (48 x 0.05 mm); the peg pose is queried every step and the
        // loop exits immediately once force and alignment are safe.
        for (int relief_step = 0; relief_step < 48; ++relief_step) {
          const WrenchVector before_relief = corrected_wrench();
          const double axial_before = std::abs(before_relief[2]);
          const double lateral_before = std::hypot(
              before_relief[0], before_relief[1]);
          double peg_hole_dx = 0.0;
          double peg_hole_dy = 0.0;
          double peg_hole_distance = 0.0;
          double peg_center_z = 0.0;
          double peg_tilt_rad = 0.0;
          const bool alignment_available = query_peg_hole_alignment(
              query_alignment_client, logger, peg_hole_dx, peg_hole_dy,
              peg_hole_distance, peg_center_z, peg_tilt_rad);
          if (
              axial_before <= RELIEF_TARGET_FORCE &&
              lateral_before <= RELIEF_TARGET_LATERAL_FORCE &&
              alignment_available &&
              peg_hole_distance <= RELIEF_XY_TOLERANCE) {
            break;
          }
          geometry_msgs::msg::Pose relief_pose =
              arm.getCurrentPose(IK_LINK).pose;
          if (
              axial_before > RELIEF_TARGET_FORCE ||
              lateral_before > RELIEF_TARGET_LATERAL_FORCE) {
            relief_pose.position.z += RELIEF_STEP_Z;
          }
          if (alignment_available) {
            relief_pose.position.x -= std::clamp(
                peg_hole_dx, -RELIEF_XY_STEP_MAX, RELIEF_XY_STEP_MAX);
            relief_pose.position.y -= std::clamp(
                peg_hole_dy, -RELIEF_XY_STEP_MAX, RELIEF_XY_STEP_MAX);
          }
          if (!try_cartesian(
                  arm, logger, relief_pose, IK_LINK,
                  "INSERT force/axis relief", 0.00002)) {
            insert_motion_ok = false;
            break;
          }
          std::this_thread::sleep_for(40ms);
          log_info(
              logger,
              "INSERT_FORCE_RELIEF:step=" +
              std::to_string(relief_step + 1) + ",fz_before=" +
              std::to_string(before_relief[2]) + ",fxy_before=" +
              std::to_string(lateral_before) + ",dx=" +
              std::to_string(peg_hole_dx) + ",dy=" +
              std::to_string(peg_hole_dy));
        }
        // A rigid, non-compliant socket has two legitimate force responses:
        // a bounded seating plateau, or a clear contact followed by safe
        // relaxation after the stop command. Both are force-time evidence;
        // neither can be replaced by geometry alone.
        std::this_thread::sleep_for(150ms);
        std::deque<double> settled_axial;
        std::deque<double> settled_lateral;
        constexpr int SETTLE_MAX_SAMPLES = 75;
        bool settled_window_found = false;
        bool settled_plateau_found = false;
        bool settled_relaxation_found = false;
        bool settled_bounded_found = false;
        int settle_samples_used = 0;
        const auto average = [](const auto& values) {
          double sum = 0.0;
          for (const double value : values) {
            sum += value;
          }
          return sum / static_cast<double>(values.size());
        };
        for (int sample = 0; sample < SETTLE_MAX_SAMPLES; ++sample) {
          const WrenchVector measured = corrected_wrench();
          const double measured_axial = std::abs(measured[2]);
          const double measured_lateral = std::hypot(
              measured[0], measured[1]);
          settle_samples_used = sample + 1;
          settled_axial.push_back(measured_axial);
          settled_lateral.push_back(measured_lateral);
          if (
              settled_axial.size() >
              static_cast<std::size_t>(INSERT_VERIFY_FORCE_SAMPLES)) {
            settled_axial.pop_front();
            settled_lateral.pop_front();
          }
          if (
              settled_axial.size() ==
              static_cast<std::size_t>(INSERT_VERIFY_FORCE_SAMPLES)) {
            const double axial_mean = average(settled_axial);
            double variance = 0.0;
            for (const double value : settled_axial) {
              const double error = value - axial_mean;
              variance += error * error;
            }
            const double axial_std = std::sqrt(
                variance / static_cast<double>(settled_axial.size()));
            const double axial_peak = *std::max_element(
                settled_axial.begin(), settled_axial.end());
            const double axial_floor = *std::min_element(
                settled_axial.begin(), settled_axial.end());
            const double lateral_mean = average(settled_lateral);
            const double lateral_peak = *std::max_element(
                settled_lateral.begin(), settled_lateral.end());
            settled_plateau_found =
                axial_mean >= INSERT_VERIFY_FORCE_MIN &&
                axial_mean <= INSERT_AXIAL_FORCE_MAX &&
                axial_floor >= INSERT_VERIFY_FORCE_MIN &&
                axial_peak <= INSERT_AXIAL_FORCE_MAX &&
                axial_std <= INSERT_VERIFY_FORCE_STD_MAX &&
                lateral_mean <= INSERT_VERIFY_LATERAL_MEAN_MAX &&
                lateral_peak <= INSERT_LATERAL_FORCE_MAX;
            settled_relaxation_found =
                axial_peak < INSERT_VERIFY_FORCE_MIN &&
                lateral_peak <= INSERT_VERIFY_LATERAL_MEAN_MAX;
            settled_bounded_found =
                axial_peak <= INSERT_AXIAL_FORCE_MAX &&
                axial_std <= INSERT_VERIFY_FORCE_STD_MAX &&
                lateral_mean <= INSERT_VERIFY_LATERAL_MEAN_MAX &&
                lateral_peak <= INSERT_LATERAL_FORCE_MAX;
            settled_window_found =
                settled_plateau_found || settled_relaxation_found ||
                settled_bounded_found;
            if (settled_window_found) {
              break;
            }
          }
          std::this_thread::sleep_for(20ms);
        }
        if (settled_axial.empty()) {
          settled_axial.push_back(0.0);
          settled_lateral.push_back(0.0);
        }
        const double settled_axial_mean = average(settled_axial);
        const double settled_lateral_mean = average(settled_lateral);
        double settled_variance = 0.0;
        for (const double value : settled_axial) {
          const double error = value - settled_axial_mean;
          settled_variance += error * error;
        }
        const double settled_axial_std = std::sqrt(
            settled_variance /
            static_cast<double>(settled_axial.size()));
        const double settled_axial_peak =
            *std::max_element(settled_axial.begin(), settled_axial.end());
        const double settled_axial_floor =
            *std::min_element(settled_axial.begin(), settled_axial.end());
        const double settled_lateral_peak =
            *std::max_element(settled_lateral.begin(), settled_lateral.end());
        const bool settled_safe = settled_window_found;
        log_info(
            logger,
            std::string("INSERT_SOLVER_IMPULSE_RECOVERY:") +
            (settled_safe ? "success," : "failure,") +
            "fz_mean=" + std::to_string(settled_axial_mean) +
            ",fz_std=" + std::to_string(settled_axial_std) +
            ",fz_min=" + std::to_string(settled_axial_floor) +
            ",fz_peak=" + std::to_string(settled_axial_peak) +
            ",fxy_mean=" + std::to_string(settled_lateral_mean) +
            ",fxy_peak=" + std::to_string(settled_lateral_peak) +
            ",samples_used=" + std::to_string(settle_samples_used) +
            ",signature=" +
            (settled_plateau_found ? "plateau" :
             (settled_relaxation_found ? "contact_relaxation" :
              (settled_bounded_found ? "bounded_stable" : "none"))));
        if (settled_safe) {
          insertion_overload = false;
          insertion_contact_confirmed = true;
          // This stationary, bounded 25-sample window is already the force
          // plateau required for insertion verification. Do not invalidate
          // it if the ideal rigid constraint later relaxes toward zero while
          // the independent model-geometry check is running.
          insertion_force_plateau_confirmed = settled_plateau_found;
          insertion_force_relaxation_confirmed =
              settled_relaxation_found || settled_bounded_found;
          if (settled_plateau_found) {
            contact_axial_force = settled_axial_mean;
          }
          contact_lateral_force = settled_lateral_mean;
        }
      }
      // Exact tangency at the blind-hole floor can carry essentially zero
      // preload in DART: no force threshold crossing occurs even though the
      // closed gripper actively delivered the peg to the complete seat.
      // Admit that second evidence path only under much tighter geometry
      // tolerances than final acceptance. A later 25-sample bounded-force
      // hold and the pre/post-release no-gravity checks remain mandatory.
      constexpr double FINAL_PEG_CENTER_Z =
          RING_BOTTOM_Z + PEG_HALF_HEIGHT;
      if (
          insert_motion_ok && !insertion_contact_confirmed &&
          !insertion_overload) {
        // Motion has reached its commanded insertion endpoint.  With an
        // exact-fit blind hole DART may report essentially zero preload, so
        // the following simulator-body query is verification evidence, not
        // additional insertion motion.  Announce the boundary before the
        // synchronous query; otherwise its stationary service latency is
        // incorrectly recorded as an ``insert`` stall.
        if (fine_insert_announced && !verify_subtask_announced) {
          log_info(logger, "SUBTASK:verify insertion success");
          verify_subtask_announced = true;
        }
        double peg_hole_dx = 0.0;
        double peg_hole_dy = 0.0;
        double peg_hole_distance = 0.0;
        double peg_center_z = 0.0;
        double peg_tilt_rad = 0.0;
        if (query_peg_hole_alignment(
                query_alignment_client, logger, peg_hole_dx, peg_hole_dy,
                peg_hole_distance, peg_center_z, peg_tilt_rad)) {
          geometric_seating_without_force =
              peg_hole_distance <= 0.00010 &&
              std::abs(peg_center_z - FINAL_PEG_CENTER_Z) <= 0.00005 &&
              peg_tilt_rad <= 0.002;
          log_info(
              logger,
              std::string("GEOMETRIC_ZERO_PRELOAD_SEAT:") +
              (geometric_seating_without_force ? "candidate," : "rejected,") +
              "hole_dist=" + std::to_string(peg_hole_distance) +
              ",peg_z=" + std::to_string(peg_center_z) +
              ",tilt_rad=" + std::to_string(peg_tilt_rad));
        }
      }
      const bool seating_event_confirmed =
          insertion_contact_confirmed || geometric_seating_without_force;

      // A bounded contact transient, an exact zero-preload seat, or a relaxed
      // force is not task success by itself:
      // the closed gripper must actively place the peg on the blind-hole
      // floor before opening.  Iterate against the scripted simulator-truth
      // alignment service, but record only the resulting ordinary robot
      // actions.  This truth is never a policy observation.
      if (
          insert_motion_ok && seating_event_confirmed &&
          !insertion_overload) {
        constexpr double ACTIVE_SEAT_XY_TOLERANCE = 0.0003;
        constexpr double ACTIVE_SEAT_AXIS_TOLERANCE = 0.00008;
        // Allow only a 0.05 mm extension over the former 0.3 mm solver
        // relaxation.  This is enough to avoid oscillating against the rigid
        // floor while still preventing release/gravity from completing a
        // visibly unfinished insertion.
        constexpr double ACTIVE_SEAT_Z_TOLERANCE = 0.00035;
        constexpr double ACTIVE_SEAT_XY_STEP_MAX = 0.00010;
        constexpr double ACTIVE_SEAT_Z_STEP_MAX = 0.00010;
        constexpr int ACTIVE_SEAT_MAX_STEPS = 30;
        log_info(
            logger,
            "ACTIVE_SEATING:start,target_peg_z=" +
            std::to_string(FINAL_PEG_CENTER_Z));
        geometry_msgs::msg::Pose active_command_pose =
            arm.getCurrentPose(IK_LINK).pose;
        for (int step = 0; step < ACTIVE_SEAT_MAX_STEPS; ++step) {
          double peg_hole_dx = 0.0;
          double peg_hole_dy = 0.0;
          double peg_hole_distance = 0.0;
          double peg_center_z = 0.0;
          double peg_tilt_rad = 0.0;
          if (!query_peg_hole_alignment(
                  query_alignment_client, logger, peg_hole_dx, peg_hole_dy,
                  peg_hole_distance, peg_center_z, peg_tilt_rad)) {
            break;
          }
          const double z_error = FINAL_PEG_CENTER_Z - peg_center_z;
          log_info(
              logger,
              "ACTIVE_SEATING:step=" + std::to_string(step) +
              ",dx=" + std::to_string(peg_hole_dx) +
              ",dy=" + std::to_string(peg_hole_dy) +
              ",peg_z=" + std::to_string(peg_center_z) +
              ",z_error=" + std::to_string(z_error));
          if (
              peg_hole_distance <= ACTIVE_SEAT_XY_TOLERANCE &&
              std::abs(z_error) <= ACTIVE_SEAT_Z_TOLERANCE) {
            active_seating_confirmed = true;
            break;
          }

          // Centre the exact-fit axes more tightly than the final acceptance
          // window before descending.  This prevents a 0.2 mm rim error from
          // being converted into a large lateral DART constraint impulse.
          // Accumulate the target: individual 0.1 mm Cartesian moves can be
          // swallowed by the trajectory controller's reached-goal tolerance.
          active_command_pose.position.x -= std::clamp(
              peg_hole_dx,
              -ACTIVE_SEAT_XY_STEP_MAX,
              ACTIVE_SEAT_XY_STEP_MAX);
          active_command_pose.position.y -= std::clamp(
              peg_hole_dy,
              -ACTIVE_SEAT_XY_STEP_MAX,
              ACTIVE_SEAT_XY_STEP_MAX);
          if (peg_hole_distance <= ACTIVE_SEAT_AXIS_TOLERANCE) {
            active_command_pose.position.z += std::clamp(
                z_error,
                -ACTIVE_SEAT_Z_STEP_MAX,
                ACTIVE_SEAT_Z_STEP_MAX);
          }
          if (!try_cartesian(
                  arm, logger, active_command_pose, IK_LINK,
                  "Closed-gripper active seating", 0.00002, 3.0)) {
            break;
          }
          std::this_thread::sleep_for(40ms);
          // Check the achieved geometry before rejecting the just-completed
          // action on its brief contact transient.  The old ordering could
          // place the peg within tolerance and then label that same action a
          // failure because DART momentarily reported ~45 N at the rigid
          // exact-fit floor.
          double achieved_dx = 0.0;
          double achieved_dy = 0.0;
          double achieved_distance = 0.0;
          double achieved_z = 0.0;
          double achieved_tilt = 0.0;
          if (query_peg_hole_alignment(
                  query_alignment_client, logger, achieved_dx, achieved_dy,
                  achieved_distance, achieved_z, achieved_tilt) &&
              achieved_distance <= ACTIVE_SEAT_XY_TOLERANCE &&
              std::abs(achieved_z - FINAL_PEG_CENTER_Z) <=
                  ACTIVE_SEAT_Z_TOLERANCE) {
            active_seating_confirmed = true;
            log_info(
                logger,
                "ACTIVE_SEATING:post_motion_success,dist=" +
                std::to_string(achieved_distance) + ",peg_z=" +
                std::to_string(achieved_z));
            break;
          }
          const WrenchVector active_wrench = corrected_wrench();
          const double active_axial = std::abs(active_wrench[2]);
          const double active_lateral = std::hypot(
              active_wrench[0], active_wrench[1]);
          if (
              active_axial > INSERT_AXIAL_FORCE_MAX ||
              active_lateral > INSERT_LATERAL_FORCE_MAX) {
            log_info(
                logger,
                "ACTIVE_SEATING:force_guard,fz=" +
                std::to_string(active_wrench[2]) + ",fxy=" +
                std::to_string(active_lateral));
            break;
          }
        }
        log_info(
            logger,
            std::string("ACTIVE_SEATING:") +
            (active_seating_confirmed ? "success" : "failure"));
        if (!active_seating_confirmed) {
          insert_motion_ok = false;
        }
      }
      // The peg can settle a few tenths of a millimetre higher or lower in
      // the fingertips across payload/friction randomization.  Therefore
      // tool0 z is not a reliable insertion-depth measurement.  Query the
      // simulated peg and hole bodies directly after controlled contact.
      if (
          insert_motion_ok && active_seating_confirmed &&
          seating_event_confirmed &&
          !insertion_overload) {
        // Contact transients and any controlled force relief still belong to
        // insertion.  Verification begins only after a safe stationary force
        // window has been established.
        if (fine_insert_announced && !verify_subtask_announced) {
          log_info(logger, "SUBTASK:verify insertion success");
          verify_subtask_announced = true;
        }
        insertion_model_geometry_confirmed = call_trigger_service(
            confirm_insertion_client, logger,
            "/pap_moe/confirm_insertion_geometry");
      }
      insertion_depth_reached = insertion_model_geometry_confirmed;
      if (
          insert_motion_ok && seating_event_confirmed &&
          insertion_depth_reached && !insertion_overload) {
        log_info(logger, "INSERT_FORCE_HOLD:start");
        {
        std::vector<double> axial_samples;
        std::vector<double> lateral_samples;
        axial_samples.reserve(INSERT_VERIFY_FORCE_SAMPLES);
        lateral_samples.reserve(INSERT_VERIFY_FORCE_SAMPLES);
        for (int i = 0; i < INSERT_VERIFY_FORCE_SAMPLES; ++i) {
          const WrenchVector sample = corrected_wrench();
          axial_samples.push_back(std::abs(sample[2]));
          lateral_samples.push_back(std::hypot(sample[0], sample[1]));
          std::this_thread::sleep_for(40ms);
        }
        const auto mean = [](const std::vector<double>& values) {
          double total = 0.0;
          for (const double value : values) {
            total += value;
          }
          return total / static_cast<double>(values.size());
        };
        const double axial_mean = mean(axial_samples);
        const double lateral_mean = mean(lateral_samples);
        double axial_variance = 0.0;
        for (const double value : axial_samples) {
          const double error = value - axial_mean;
          axial_variance += error * error;
        }
        axial_variance /= static_cast<double>(axial_samples.size());
        const double axial_std = std::sqrt(axial_variance);
        const double axial_peak =
            *std::max_element(axial_samples.begin(), axial_samples.end());
        const double lateral_peak =
            *std::max_element(lateral_samples.begin(), lateral_samples.end());
        // At the exact geometric seat a rigid solver may settle at zero
        // preload. Require a bounded stable window after active seating;
        // either an earlier measured contact or the strict zero-preload
        // geometry candidate proves the insertion event.
        insertion_force_plateau_confirmed =
            axial_mean <= INSERT_AXIAL_FORCE_MAX &&
            axial_peak <= INSERT_AXIAL_FORCE_MAX &&
            axial_std <= INSERT_VERIFY_FORCE_STD_MAX &&
            lateral_mean <= INSERT_VERIFY_LATERAL_MEAN_MAX &&
            lateral_peak <= INSERT_LATERAL_FORCE_MAX;
        log_info(
            logger,
            std::string("INSERT_FORCE_HOLD:") +
            (insertion_force_plateau_confirmed ? "success," : "failure,") +
            "fz_mean=" + std::to_string(axial_mean) +
            ",fz_std=" + std::to_string(axial_std) +
            ",fz_peak=" + std::to_string(axial_peak) +
            ",fxy_mean=" + std::to_string(lateral_mean) +
            ",fxy_peak=" + std::to_string(lateral_peak) +
            ",samples=" + std::to_string(INSERT_VERIFY_FORCE_SAMPLES));
        }
      }
      std::this_thread::sleep_for(50ms);
      const auto insert_pose = arm.getCurrentPose(IK_LINK).pose;
      const WrenchVector insert_wrench = corrected_wrench();
      const double tool_xy_error = std::hypot(
          insert_pose.position.x - corrected_hole_tool_x,
          insert_pose.position.y - corrected_hole_tool_y);
      const bool tool_tracking_within_nominal_tolerance =
          tool_xy_error <= 0.0008;
      // The task-space success criterion is the measured peg/hole geometry,
      // not the nominal tool-center tracking error.  With a physical grasp,
      // fingertip compliance and a small peg tilt can offset tool0 by a few
      // millimetres even though the peg itself is centered and fully seated.
      // Keep tool tracking as a diagnostic, but do not reject an insertion
      // that the active-seating and body-geometry checks both verified.
      insertion_geometry_confirmed =
          active_seating_confirmed &&
          insertion_model_geometry_confirmed;
      log_info(
          logger,
          std::string("INSERT_GEOMETRY:") +
          (insertion_geometry_confirmed ? "success," : "failure,") +
          "tool_xy_error=" + std::to_string(tool_xy_error) +
          ",tool_tracking=" +
          (tool_tracking_within_nominal_tolerance ? "nominal" : "offset") +
          ",tool0_z=" + std::to_string(insert_pose.position.z));
      assembly_confirmed =
          insert_motion_ok &&
          insertion_geometry_confirmed &&
          active_seating_confirmed &&
          !insertion_overload &&
          seating_event_confirmed &&
          insertion_depth_reached &&
          insertion_force_plateau_confirmed &&
          std::abs(insert_wrench[2]) <= INSERT_AXIAL_FORCE_MAX &&
          std::hypot(insert_wrench[0], insert_wrench[1]) <=
              INSERT_LATERAL_FORCE_MAX;
      log_info(
          logger,
          std::string("INSERT_VERIFY:") +
          (assembly_confirmed ? "success," : "failure,") +
          std::to_string(insert_pose.position.z) + "," +
          std::to_string(insert_wrench[2]) + ",contact_fz=" +
          std::to_string(contact_axial_force) + ",contact_fxy=" +
          std::to_string(contact_lateral_force) + ",contact=" +
          (insertion_contact_confirmed ? "1" : "0") +
          ",geometric_zero_preload=" +
          (geometric_seating_without_force ? "1" : "0") + ",depth=" +
          (insertion_depth_reached ? "1" : "0") + ",geometry=" +
          (insertion_geometry_confirmed ? "1" : "0") + ",plateau=" +
          (insertion_force_plateau_confirmed ? "1" : "0") +
          ",relaxation=" +
          (insertion_force_relaxation_confirmed ? "1" : "0"));
    } else {
      log_info(logger, "INSERT_ABORTED:alignment_not_confirmed");
    }

    // Geometry and commanded depth are deliberately insufficient: while the
    // peg remains attached, require a bounded force-time window plus either
    // measured contact or the strict zero-preload seating geometry. The
    // later release gate still prevents gravity-completed demonstrations.
    if (aligned && !assembly_confirmed) {
      log_info(
          logger,
          "INSERT_REJECTED: controlled_contact=" +
          std::string(insertion_contact_confirmed ? "1" : "0") +
          ",geometric_zero_preload=" +
          std::string(geometric_seating_without_force ? "1" : "0") +
          ",depth=" +
          std::string(insertion_depth_reached ? "1" : "0") +
          ",geometry=" +
          std::string(insertion_geometry_confirmed ? "1" : "0") +
          ",plateau=" +
          std::string(insertion_force_plateau_confirmed ? "1" : "0") +
          ",relaxation=" +
          std::string(insertion_force_relaxation_confirmed ? "1" : "0"));
    }

    bool release_confirmed = false;
    if (assembly_confirmed) {
      log_info(logger, "=== 12. Open gripper ===");
      log_info(logger, "SUBTASK:release the peg after verification");
      log_info(logger, "C++ DETACH PEG");
      const bool detach_requested = call_trigger_service(
          detach_client, logger, "/pap_moe/detach_peg");

      bool gripper_open = false;
      if (detach_requested) {
        for (int attempt = 1; attempt <= 3 && !gripper_open; ++attempt) {
          gripper.setNamedTarget(GRIP_OPEN);
          const bool motion_ok = try_move(gripper, logger, "open");
          std::this_thread::sleep_for(250ms);
          const auto joints = gripper.getCurrentJointValues();
          const double opening_joint =
              joints.empty() ? 1.0 : std::abs(joints.front());
          gripper_open = motion_ok && opening_joint <= 0.12;
          log_info(
              logger,
              "GRIPPER_OPEN_VERIFY:" +
              std::string(gripper_open ? "success," : "failure,") +
              std::to_string(opening_joint) + ",attempt=" +
              std::to_string(attempt));
        }
      }

      bool micro_retract_ok = false;
      if (detach_requested && gripper_open) {
        auto release_pose = arm.getCurrentPose(IK_LINK).pose;
        release_pose.position.z += 0.008;
        micro_retract_ok = try_cartesian(
            arm, logger, release_pose, IK_LINK,
            "RELEASE verification micro-retract", 0.0005);
      }

      if (detach_requested && gripper_open && micro_retract_ok) {
        release_confirmed = call_trigger_service(
            verify_release_client, logger,
            "/pap_moe/verify_peg_release");
      }

      if (release_confirmed) {
        log_info(logger, "RELEASE_VERIFY:success");
        log_info(logger, "TASK_RESULT:success");
      } else {
        log_info(logger, "RELEASE_VERIFY:failure");
        log_info(logger, "TASK_RESULT:failure");
        log_info(
            logger,
            "RELEASE_ABORTED: holding position; full retract/home forbidden.");
      }
    } else {
      log_info(logger, "TASK_RESULT:failure");
      if (aligned) {
        log_info(
            logger,
            "INSERTION_UNCERTAIN: holding position; closed-gripper "
            "retract/home forbidden.");
      } else {
        log_info(
            logger,
            "Keeping gripper closed because the peg never aligned with the hole.");
      }
    }

    // Full retract/home is permitted only after verified release.  Any failure
    // in the hole workspace remains local so a grasped or partly inserted peg
    // can never be carried back to HOME.
    if (release_confirmed) {
      if (!set_sim_position_gain(controller_parameter_client, logger, 0.5)) {
        throw std::runtime_error("Could not restore free-space position gain");
      }
      // Insertion deliberately leaves the arm at the very conservative
      // contact-motion limits (typically 0.01 / 0.005).  The peg has now been
      // physically released and its pose verified, so carrying those limits
      // into the reset retract only wastes wall time and does not add policy
      // samples (recording ends at TASK_RESULT).  Restore a bounded
      // free-space profile before retracting; HOME is still allowed to use
      // the full configured joint limits below.
      constexpr double RESET_RETRACT_VELOCITY_SCALE = 0.30;
      constexpr double RESET_RETRACT_ACCELERATION_SCALE = 0.20;
      arm.setMaxVelocityScalingFactor(RESET_RETRACT_VELOCITY_SCALE);
      arm.setMaxAccelerationScalingFactor(RESET_RETRACT_ACCELERATION_SCALE);
      log_info(
          logger,
          "RESET_RETRACT_SPEED:" +
          std::to_string(RESET_RETRACT_VELOCITY_SCALE) + "," +
          std::to_string(RESET_RETRACT_ACCELERATION_SCALE));
      // 13. Retract
      log_info(logger, "=== 13. RETRACT ===");
      log_info(logger, "SUBTASK:retract and go back to home");
      {
        auto current = arm.getCurrentPose(IK_LINK);
        geometry_msgs::msg::Pose tgt = current.pose;
        tgt.position.z = TOOL0_ABOVE_Z;
        try_cartesian(arm, logger, tgt, IK_LINK, "RETRACT");
      }

      // Restore fast speed for returning HOME
      arm.setMaxVelocityScalingFactor(1.0);
      arm.setMaxAccelerationScalingFactor(1.0);

      // 14. Return HOME
      log_info(logger, "=== 14. HOME ===");
      arm.setJointValueTarget(READY_JOINTS);
      try_move(arm, logger, "HOME");
    } else if (contact_detected && !aligned) {
      // Recovery exhausted on the plate/rim: unload only a few millimetres,
      // then stop.  The recorder will delete/reset the episode objects before
      // the next run.
      log_info(logger, "=== 13. LOCAL CONTACT UNLOAD ===");
      // Keep the established recovery label so the dataset schema remains
      // closed; FAILURE_HOLD below carries the terminal safety semantics.
      log_info(logger, "SUBTASK:recover contact and relocate the hole");
      auto unload = arm.getCurrentPose(IK_LINK).pose;
      unload.position.z += 0.005;
      try_cartesian(
          arm, logger, unload, IK_LINK,
          "LOCAL CONTACT UNLOAD", 0.0005);
      log_info(
          logger,
          "FAILURE_HOLD: local unload complete; retract/home forbidden.");
    } else {
      log_info(
          logger,
          "FAILURE_HOLD: holding position; retract/home forbidden.");
    }

    log_info(
        logger,
        (assembly_confirmed && release_confirmed) ?
        "Assembly episode finished successfully." :
        "Assembly episode finished with failure.");
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(logger, "Exception in controller: %s", e.what());
  }

  executor->cancel();
  spin.join();
  rclcpp::shutdown();
  return 0;
}
