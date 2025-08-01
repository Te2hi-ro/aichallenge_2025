#include "simple_pure_pursuit/simple_pure_pursuit.hpp"

#include <motion_utils/motion_utils.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath> // std::hypot, std::clamp のために追加
#include <limits> // std::numeric_limits のために追加

namespace simple_pure_pursuit
{

using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcYawDeviation;

SimplePurePursuit::SimplePurePursuit()
: Node("simple_pure_pursuit"),
  // initialize parameters
  wheel_base_(declare_parameter<float>("wheel_base", 2.14)),
  lookahead_gain_(declare_parameter<float>("lookahead_gain", 1.0)),
  lookahead_min_distance_(declare_parameter<float>("lookahead_min_distance", 1.0)),
  speed_proportional_gain_(declare_parameter<float>("speed_proportional_gain", 1.0)),
  use_external_target_vel_(declare_parameter<bool>("use_external_target_vel", false)),
  external_target_vel_(declare_parameter<float>("external_target_vel", 0.0)),
  steering_tire_angle_gain_(declare_parameter<float>("steering_tire_angle_gain", 1.0)),
  cte_proportional_gain_(declare_parameter<double>("cte_proportional_gain", 1.0)),
  k_softening(declare_parameter<double>("k_softening", 1e-4))
{
  pub_cmd_ = create_publisher<AckermannControlCommand>("output/control_cmd", 1);
  pub_raw_cmd_ = create_publisher<AckermannControlCommand>("output/raw_control_cmd", 1);
  pub_lookahead_point_ = create_publisher<PointStamped>("/control/debug/lookahead_point", 1);

  const auto bv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  sub_kinematics_ = create_subscription<Odometry>(
    "input/kinematics", bv_qos, [this](const Odometry::SharedPtr msg) { odometry_ = msg; });
  sub_trajectory_ = create_subscription<Trajectory>(
    "input/trajectory", bv_qos, [this](const Trajectory::SharedPtr msg) { trajectory_ = msg; });

  using namespace std::literals::chrono_literals;
  timer_ =
    rclcpp::create_timer(this, get_clock(), 10ms, std::bind(&SimplePurePursuit::onTimer, this));
}

AckermannControlCommand zeroAckermannControlCommand(rclcpp::Time stamp)
{
  AckermannControlCommand cmd;
  cmd.stamp = stamp;
  cmd.longitudinal.stamp = stamp;
  cmd.longitudinal.speed = 0.0;
  cmd.longitudinal.acceleration = 0.0;
  cmd.lateral.stamp = stamp;
  cmd.lateral.steering_tire_angle = 0.0;
  return cmd;
}

void SimplePurePursuit::onTimer()
{
  // check data
  if (!subscribeMessageAvailable()) {
    return;
  }

  // 1. 最近傍点の探索（縦方向制御用）
  size_t closet_traj_point_idx =
    findNearestIndex(trajectory_->points, odometry_->pose.pose.position);

  AckermannControlCommand cmd = zeroAckermannControlCommand(get_clock()->now());

  // 2. 終了判定
  if (
    (closet_traj_point_idx >= trajectory_->points.size() - 1) ||
    (trajectory_->points.size() <= 2)) {
    cmd.longitudinal.speed = 0.0;
    cmd.longitudinal.acceleration = -10.0;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "reached to the goal");
  } else {
    // 3. 縦方向制御（Pure Pursuitベース）
    TrajectoryPoint closet_traj_point = trajectory_->points.at(closet_traj_point_idx);
    double target_longitudinal_vel =
      use_external_target_vel_? external_target_vel_ : closet_traj_point.longitudinal_velocity_mps;
    double current_longitudinal_vel = odometry_->twist.twist.linear.x;

    cmd.longitudinal.speed = target_longitudinal_vel;
    cmd.longitudinal.acceleration =
      speed_proportional_gain_ * (target_longitudinal_vel - current_longitudinal_vel);

    // -----------------------------------------------------------------
    // 4. 横方向制御（修正されたStanley法）
    // -----------------------------------------------------------------

    // 4.1. 前輪軸中心の座標を計算
    const double current_yaw = tf2::getYaw(odometry_->pose.pose.orientation);
    const double front_x = odometry_->pose.pose.position.x + wheel_base_ / 2.0 * std::cos(current_yaw);
    const double front_y = odometry_->pose.pose.position.y + wheel_base_ / 2.0 * std::sin(current_yaw);

    // 4.2. 前輪軸中心に最も近い経路上の点を探索
    geometry_msgs::msg::Point front_axle_pos;
    front_axle_pos.x = front_x;
    front_axle_pos.y = front_y;
    front_axle_pos.z = odometry_->pose.pose.position.z;
    size_t front_closest_idx = findNearestIndex(trajectory_->points, front_axle_pos);
    const TrajectoryPoint& target_traj_point = trajectory_->points.at(front_closest_idx);

    // 4.3. 横方向偏差（CTE）を頑健な方法で計算
    // 経路の接線方向ベクトルを計算 (境界チェックを含む)
    size_t next_idx = (front_closest_idx + 1 < trajectory_->points.size())? front_closest_idx + 1 : front_closest_idx;
    const double path_dx = trajectory_->points[next_idx].pose.position.x - target_traj_point.pose.position.x;
    const double path_dy = trajectory_->points[next_idx].pose.position.y - target_traj_point.pose.position.y;
    const double path_heading = std::atan2(path_dy, path_dx);

    // 前輪軸中心から最近傍点への誤差ベクトル
    const double error_vec_x = front_x - target_traj_point.pose.position.x;
    const double error_vec_y = front_y - target_traj_point.pose.position.y;

    // 外積を用いて符号付きCTEを計算 (経路の右側が正、左側が負)
    const double cross_product = error_vec_x * path_dy - error_vec_y * path_dx;
    const double cte = std::hypot(error_vec_x, error_vec_y) * ((cross_product > 0.0)? 1.0 : -1.0);

    // 4.4. 方位角偏差（Heading Error）を計算
    double heading_error = path_heading - current_yaw;
    // 角度を-PIからPIの範囲に正規化
    heading_error = std::atan2(std::sin(heading_error), std::cos(heading_error));

    // 4.5. Stanley制御則を適用（ソフトニング定数を追加）
    double steering_angle_crosstrack = std::atan2(cte_proportional_gain_ * cte, current_longitudinal_vel + k_softening);
    
    double steering_angle = heading_error + steering_angle_crosstrack;

    // 4.6. 操舵角を制限し、コマンドにセット
    const double max_steering_angle_ = 0.64; // 最大操舵角
    cmd.lateral.steering_tire_angle = std::clamp(steering_angle, -max_steering_angle_, max_steering_angle_);
  }

  // 5. コマンドをパブリッシュ
  pub_cmd_->publish(cmd);

  // raw_cmdのロジックは必要に応じて調整してください
  // cmd.lateral.steering_tire_angle /=  steering_tire_angle_gain_;
  pub_raw_cmd_->publish(cmd);
}

bool SimplePurePursuit::subscribeMessageAvailable()
{
  if (!odometry_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "odometry is not available");
    return false;
  }
  if (!trajectory_) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/, "trajectory is not available");
    return false;
  }
  if (trajectory_->points.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000 /*ms*/,  "trajectory points is empty");
      return false;
    }
  return true;
}
}  // namespace simple_pure_pursuit

// 修正点: main関数の引数の型をC++の標準である `char * argv` に修正
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<simple_pure_pursuit::SimplePurePursuit>());
  rclcpp::shutdown();
  return 0;
}
