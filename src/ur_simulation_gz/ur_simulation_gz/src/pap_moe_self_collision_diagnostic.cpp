#include <chrono>
#include <iomanip>
#include <memory>
#include <string>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("pap_moe_self_collision_diagnostic");
  auto monitor = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(
      node, "robot_description", "pap_moe_self_collision_monitor");
  if (!monitor->getPlanningScene())
  {
    RCLCPP_ERROR(node->get_logger(), "Unable to construct the MoveIt planning scene");
    rclcpp::shutdown();
    return 1;
  }

  monitor->startStateMonitor("/joint_states");
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline)
  {
    rclcpp::spin_some(node);
    if (monitor->getStateMonitor() && monitor->getStateMonitor()->haveCompleteState())
      break;
    rclcpp::sleep_for(50ms);
  }
  monitor->updateSceneWithCurrentState();

  planning_scene_monitor::LockedPlanningSceneRO scene(monitor);
  collision_detection::DistanceRequest request;
  collision_detection::DistanceResult result;
  request.type = collision_detection::DistanceRequestType::SINGLE;
  request.enable_nearest_points = true;
  request.enable_signed_distance = true;
  request.distance_threshold = 0.05;
  request.acm = &scene->getAllowedCollisionMatrix();
  scene->getCollisionEnvUnpadded()->distanceSelf(request, result, scene->getCurrentState());

  const auto& nearest = result.minimum_distance;
  RCLCPP_INFO(node->get_logger(),
              "nearest self-collision pair: %s <-> %s, signed distance %.9f m",
              nearest.link_names[0].c_str(), nearest.link_names[1].c_str(), nearest.distance);
  RCLCPP_INFO(node->get_logger(), "nearest points: [%.6f %.6f %.6f] <-> [%.6f %.6f %.6f]",
              nearest.nearest_points[0].x(), nearest.nearest_points[0].y(), nearest.nearest_points[0].z(),
              nearest.nearest_points[1].x(), nearest.nearest_points[1].y(), nearest.nearest_points[1].z());

  rclcpp::shutdown();
  return 0;
}
