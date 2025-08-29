#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

import tf2_ros
from geometry_msgs.msg import TransformStamped


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """
    Minimal, dependency-free quaternion->yaw.
    Assumes ROS quaternion ordering (x,y,z,w).
    """
    # yaw (z-axis rotation)
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def compose_map_pose_from_odom(
    ox: float, oy: float, oyaw: float,
    tmo_x: float, tmo_y: float, tmo_yaw: float
) -> Tuple[float, float, float]:
    """
    Compose map<-odom transform with robot pose in odom to get map pose.
    map_p = T_map_odom * odom_p
    Where T_map_odom is (tmo_x, tmo_y, tmo_yaw).
    """
    cos_t = math.cos(tmo_yaw)
    sin_t = math.sin(tmo_yaw)
    mx = tmo_x + cos_t * ox - sin_t * oy
    my = tmo_y + sin_t * ox + cos_t * oy
    myaw = wrap_to_pi(oyaw + tmo_yaw)
    return mx, my, myaw


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class CollisionAvoid(Node):
    def __init__(self):
        super().__init__('costmap_joystick')

        # Parameters
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('cmd_vel_in', '/cmd_vel_joy')
        self.declare_parameter('cmd_vel_out', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('occupied_threshold', 70)     # occupancy 0..100
        self.declare_parameter('lookahead', 1.0)             # [m]
        self.declare_parameter('corridor_width', 0.3)        # [m] half-width each side
        self.declare_parameter('sampling_step', 0.05)        # [m]
        self.declare_parameter('max_angular_speed', 1.2)     # [rad/s]
        self.declare_parameter('slowdown_margin', 0.4)       # [m]
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')

        # Resolve parameters
        self.costmap_topic = self.get_parameter('costmap_topic').value
        self.cmd_vel_in_topic = self.get_parameter('cmd_vel_in').value
        self.cmd_vel_out_topic = self.get_parameter('cmd_vel_out').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.corridor_width = float(self.get_parameter('corridor_width').value)
        self.sampling_step = float(self.get_parameter('sampling_step').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.slowdown_margin = float(self.get_parameter('slowdown_margin').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value

        # State
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_cmd_in: Optional[Twist] = None

        # TF buffer/listener (no external deps)
        self.tf_buffer = tf2_ros.Buffer()  # default cache time
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        default_qos = QoSProfile(depth=10)

        # IO
        self.sub_costmap = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self.on_costmap, costmap_qos
        )
        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.on_odom, default_qos
        )
        self.sub_cmd = self.create_subscription(
            Twist, self.cmd_vel_in_topic, self.on_cmd_in, default_qos
        )
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)

        # 20 Hz control loop
        self.timer = self.create_timer(0.05, self.control_step)

        self.get_logger().info('collision_avoid_cmd_mux node ready (no tf_transformations needed).')

    # ---- Callbacks ----
    def on_costmap(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def on_cmd_in(self, msg: Twist):
        self.latest_cmd_in = msg

    # ---- Core logic ----
    def control_step(self):
        if self.latest_cmd_in is None:
            return
        if self.latest_costmap is None or self.latest_odom is None:
            self.pub_cmd.publish(self.latest_cmd_in)
            return

        pose_map = self.get_robot_pose_in_map()
        if pose_map is None:
            self.pub_cmd.publish(self.latest_cmd_in)
            return

        rx, ry, r_yaw = pose_map

        min_clearance, left_density, right_density = self.scan_corridor(rx, ry, r_yaw)

        cmd_out = Twist()
        cmd_out.linear.x = self.latest_cmd_in.linear.x
        cmd_out.angular.z = self.latest_cmd_in.angular.z

        if min_clearance is None:
            cmd_out.linear.x = min(cmd_out.linear.x, 0.4)
            self.pub_cmd.publish(cmd_out)
            return

        if min_clearance > (self.slowdown_margin + abs(cmd_out.linear.x) * 0.8):
            self.pub_cmd.publish(cmd_out)
            return

        safe_lin = max(0.0, min(cmd_out.linear.x, max(0.0, min_clearance - self.slowdown_margin)))
        steer = 0.0
        if left_density is not None and right_density is not None:
            steer_dir = -1.0 if left_density > right_density else 1.0
            magnitude = min(1.0, abs(left_density - right_density))
            steer = steer_dir * self.max_angular_speed * magnitude
        elif left_density is not None:
            steer = -0.5 * self.max_angular_speed
        elif right_density is not None:
            steer = 0.5 * self.max_angular_speed

        if min_clearance < 0.15:
            safe_lin = 0.0

        cmd_out.linear.x = safe_lin
        cmd_out.angular.z = max(-self.max_angular_speed,
                                min(self.max_angular_speed,
                                    cmd_out.angular.z + steer))
        self.pub_cmd.publish(cmd_out)

    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, yaw) of robot in the costmap (map) frame without tf_transformations."""
        try:
            # robot pose in odom
            o = self.latest_odom.pose.pose
            ox, oy = o.position.x, o.position.y
            oyaw = yaw_from_quat_xyzw(o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)

            # map <- odom transform (translation + yaw)
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.odom_frame, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            tmo_yaw = yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)

            mx, my, myaw = compose_map_pose_from_odom(ox, oy, oyaw, t.x, t.y, tmo_yaw)
            return mx, my, myaw
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    def scan_corridor(self, rx: float, ry: float, r_yaw: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Sample a rectangular corridor ahead of the robot and return:
        - min_clearance (meters to nearest obstacle along center/nearby rays)
        - left_density (0..1) occupancy fraction in left band
        - right_density (0..1) occupancy fraction in right band
        """
        grid = self.latest_costmap
        res = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        width = grid.info.width
        height = grid.info.height
        data = grid.data  # flat list: row-major

        def world_to_idx(wx, wy):
            mx = int((wx - origin_x) / res)
            my = int((wy - origin_y) / res)
            if mx < 0 or my < 0 or mx >= width or my >= height:
                return None
            return my * width + mx

        # Unit vectors for forward and left in map frame
        fx, fy = math.cos(r_yaw), math.sin(r_yaw)
        lx, ly = -math.sin(r_yaw), math.cos(r_yaw)

        min_clear = None
        left_hits = right_hits = 0
        left_total = right_total = 0

        lateral_samples = max(1, int(self.corridor_width / max(res, 1e-3)))
        for i in range(-lateral_samples, lateral_samples + 1):
            offset = (i / max(1, lateral_samples)) * self.corridor_width
            band = 'left' if offset > 0 else ('right' if offset < 0 else 'center')

            dist = 0.0
            step = max(self.sampling_step, res)
            while dist <= self.lookahead:
                wx = rx + fx * dist + lx * offset
                wy = ry + fy * dist + ly * offset
                idx = world_to_idx(wx, wy)
                if idx is None:
                    break
                occ = data[idx]
                if occ >= self.occupied_threshold:
                    if min_clear is None or dist < min_clear:
                        min_clear = dist
                    if band == 'left':
                        left_hits += 1
                    elif band == 'right':
                        right_hits += 1
                    break  # blocked along this ray

                if band == 'left':
                    left_total += 1
                elif band == 'right':
                    right_total += 1

                dist += step

        left_density = (left_hits / left_total) if left_total > 0 else None
        right_density = (right_hits / right_total) if right_total > 0 else None
        return min_clear, left_density, right_density


def main(args=None):
    rclpy.init(args=args)
    node = CollisionAvoid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
