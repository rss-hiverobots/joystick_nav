#!/usr/bin/env python3
# g1_manipulation/mission_orchestrator_node.py

from __future__ import annotations

import math
import time
from threading import Event

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from std_msgs.msg import Bool


def yaw_to_quat(yaw: float) -> Quaternion:
    # z-yaw only
    half = yaw * 0.5
    q = Quaternion()
    q.w = math.cos(half)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(half)
    return q


class MissionOrchestrator(Node):
    """
    Sequence:
      1) Nav to A -> wait /goal_reached
      2) /rest True, wait 3.0 s
      3) drive +0.75 m forward on /grasp/cmd_vel, wait 5.0 s
      4) /grasp True, wait 15.0 s
      5) drive -0.75 m back, wait 5.0 s
      6) Nav to B -> wait /goal_reached
      7) drive +0.75 m forward, wait 5.0 s
      8) /drop True, wait 15.0 s
      9) drive -0.75 m back, wait 5.0 s
    """

    def __init__(self):
        super().__init__('mission_orchestrator')

        qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Parameters (override via YAML or CLI) ---
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('point_a_xytheta', [1.0, 0.0, 0.0])  # [x,y,theta(rad)]
        self.declare_parameter('point_b_xytheta', [2.5, 0.0, 3.14159])

        self.declare_parameter('forward_distance_m', 0.75)
        self.declare_parameter('backward_distance_m', 0.75)
        self.declare_parameter('cmd_linear_speed_mps', 0.15)

        self.declare_parameter('rest_wait_s', 3.0)
        self.declare_parameter('drive_wait_s', 5.0)
        self.declare_parameter('grasp_wait_s', 15.0)
        self.declare_parameter('drop_wait_s', 15.0)

        # --- Publishers ---
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.rest_pub = self.create_publisher(Bool, '/rest', qos_latched)
        self.grasp_pub = self.create_publisher(Bool, '/grasp', qos_latched)
        self.drop_pub = self.create_publisher(Bool, '/drop', qos_latched)
        self.cmd_pub = self.create_publisher(Twist, '/grasp/cmd_vel', 10)

        # --- Subscribers ---
        self.goal_reached_evt = Event()
        self.goal_reached_sub = self.create_subscription(
            Bool, '/goal_reached', self._on_goal_reached, 10
        )

        # Kick off main flow once everything is ready
        self.create_timer(0.3, self._start_once)

        self._started = False

    # -------------------- Callbacks --------------------

    def _on_goal_reached(self, msg: Bool):
        if msg.data:
            self.get_logger().info('✅ Received /goal_reached True')
            self.goal_reached_evt.set()

    # -------------------- Helpers ---------------------

    def _start_once(self):
        if self._started:
            return
        self._started = True
        self.get_logger().info('🚀 Mission orchestrator starting sequence...')
        try:
            self._run_sequence()
            self.get_logger().info('🎯 Mission complete.')
        except Exception as e:
            self.get_logger().error(f'Mission aborted with error: {e}')

    def _publish_bool(self, pub, name: str):
        pub.publish(Bool(data=True))
        self.get_logger().info(f'→ Sent {name}=True')

    def _send_goal(self, x: float, y: float, yaw: float, frame_id: str):
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0
        msg.pose.orientation = yaw_to_quat(yaw)
        self.goal_reached_evt.clear()
        self.goal_pub.publish(msg)
        self.get_logger().info(
            f'→ Sent /goal_pose: frame={frame_id} pos=({x:.3f},{y:.3f}) yaw={yaw:.3f} rad'
        )

    def _wait_goal(self, timeout_s: float | None = None):
        self.get_logger().info('… waiting for /goal_reached True')
        ok = self.goal_reached_evt.wait(timeout=timeout_s)
        if not ok:
            raise TimeoutError('Timed out waiting for /goal_reached')
        # small debounce
        rclpy.sleep(Duration(seconds=0.2))
        self.goal_reached_evt.clear()

    def _drive_distance(self, distance_m: float, speed_mps: float):
        """
        Publish Twist at constant speed for the computed duration, then stop.
        Positive distance -> forward (+x); negative -> reverse.
        """
        speed = abs(speed_mps)
        direction = 1.0 if distance_m >= 0.0 else -1.0
        duration = abs(distance_m) / (speed if speed > 1e-6 else 1e-6)

        twist = Twist()
        twist.linear.x = direction * speed
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.z = 0.0

        self.get_logger().info(
            f'→ Driving {"forward" if direction>0 else "backward"}: '
            f'{abs(distance_m):.2f} m @ {speed:.2f} m/s (~{duration:.1f} s)'
        )

        start = self.get_clock().now()
        rate = self.create_rate(20.0, self.get_clock())  # 20 Hz
        while (self.get_clock().now() - start) < Duration(seconds=duration):
            self.cmd_pub.publish(twist)
            rate.sleep()

        # Stop
        self.cmd_pub.publish(Twist())
        self.get_logger().info('→ Drive stop (zero Twist published)')

    # -------------------- Main Sequence ---------------------

    def _run_sequence(self):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        ax, ay, atheta = self.get_parameter('point_a_xytheta').get_parameter_value().double_array_value
        bx, by, btheta = self.get_parameter('point_b_xytheta').get_parameter_value().double_array_value

        fwd = float(self.get_parameter('forward_distance_m').value)
        back = float(self.get_parameter('backward_distance_m').value)
        v = float(self.get_parameter('cmd_linear_speed_mps').value)

        rest_wait = float(self.get_parameter('rest_wait_s').value)
        drive_wait = float(self.get_parameter('drive_wait_s').value)
        grasp_wait = float(self.get_parameter('grasp_wait_s').value)
        drop_wait = float(self.get_parameter('drop_wait_s').value)

        # ---------- To A ----------
        self._send_goal(ax, ay, atheta, frame_id)
        self._wait_goal()

        # ---------- Rest ----------
        self._publish_bool(self.rest_pub, '/rest')
        self.get_logger().info(f'… waiting {rest_wait:.1f}s after /rest')
        time.sleep(rest_wait)

        # ---------- Approach table at A ----------
        self._drive_distance(+fwd, v)
        self.get_logger().info(f'… waiting {drive_wait:.1f}s after forward drive')
        time.sleep(drive_wait)

        # ---------- Grasp ----------
        self._publish_bool(self.grasp_pub, '/grasp')
        self.get_logger().info(f'… waiting {grasp_wait:.1f}s for grasping motion')
        time.sleep(grasp_wait)

        # ---------- Back away ----------
        self._drive_distance(-back, v)
        self.get_logger().info(f'… waiting {drive_wait:.1f}s after backward drive')
        time.sleep(drive_wait)

        # ---------- To B ----------
        self._send_goal(bx, by, btheta, frame_id)
        self._wait_goal()

        # ---------- Approach table at B ----------
        self._drive_distance(+fwd, v)
        self.get_logger().info(f'… waiting {drive_wait:.1f}s after forward drive')
        time.sleep(drive_wait)

        # ---------- Drop ----------
        self._publish_bool(self.drop_pub, '/drop')
        self.get_logger().info(f'… waiting {drop_wait:.1f}s for drop motion')
        time.sleep(drop_wait)

        # ---------- Back away ----------
        self._drive_distance(-back, v)
        self.get_logger().info(f'… waiting {drive_wait:.1f}s after backward drive')
        time.sleep(drive_wait)


def main():
    rclpy.init()
    node = MissionOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
