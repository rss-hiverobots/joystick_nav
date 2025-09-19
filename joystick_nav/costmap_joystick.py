#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### TODO: REPLACE /costmap BY THE ACTUAL /cloud_registered TO DETECT OBSTACLES
### WITHOUT RELYING ON MAP

"""
Collision-avoidance command mux that supports forward, lateral, and backward motion.
- Considers the commanded body-frame (vx, vy) direction to scan a rectangular
  corridor aligned with the intended motion in the map frame.
- Scales linear speed (both x and y) by free clearance along that direction.
- Steers away from higher obstacle density (angular.z for diff drive).
- Supports a /joy override: if designated buttons (4 and 5) are pressed, bypass safety and forward raw cmd_vel.

ROS 2 Humble, Python.
"""

import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

import tf2_ros
from geometry_msgs.msg import TransformStamped


# --------- small math helpers ---------

def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Minimal quaternion->yaw (ROS ordering x,y,z,w)."""
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def compose_map_pose_from_odom(
    ox: float, oy: float, oyaw: float,
    tmo_x: float, tmo_y: float, tmo_yaw: float
) -> Tuple[float, float, float]:
    cos_t = math.cos(tmo_yaw)
    sin_t = math.sin(tmo_yaw)
    mx = tmo_x + cos_t * ox - sin_t * oy
    my = tmo_y + sin_t * ox + cos_t * oy
    myaw = wrap_to_pi(oyaw + tmo_yaw)
    return mx, my, myaw


# --------------- Node -----------------

class CollisionAvoid(Node):
    def __init__(self):
        """
        ROS 2 node that acts as a safety layer between teleop commands and the robot.


        - Subscribes to a costmap, odometry, joystick input, and raw velocity commands.
        - Computes the robot pose in the map frame and scans a rectangular corridor
        aligned with the commanded motion (forward, backward, or lateral).
        - Adjusts or overrides the incoming velocity to avoid collisions:
        * Scales linear velocity based on free clearance ahead/behind.
        * Adds steering or lateral bias away from denser obstacle regions.
        * Stops completely if an obstacle is too close.
        - When both override buttons on the joystick are pressed, bypasses the safety
        layer and forwards commands unchanged.
        - Publishes the resulting safe Twist to the configured output topic.
        """
        super().__init__('costmap_joystick')

        # Parameters
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('cmd_vel_in', '/cmd_vel_joy')
        self.declare_parameter('cmd_vel_out', '/cmd_vel')
        self.declare_parameter('odom_topic', '/aft_mapped_to_init')
        self.declare_parameter('occupied_threshold', 90)
        self.declare_parameter('lookahead', 1.5)
        self.declare_parameter('corridor_width', 0.4)
        self.declare_parameter('sampling_step', 0.05)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('max_linear_speed', 0.4)
        self.declare_parameter('max_lateral_speed', 0.2)
        self.declare_parameter('slowdown_margin', 0.7)
        self.declare_parameter('stop_margin', 0.3)      # increase for earlier stop
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('lateral_avoid_gain', 0.6)
        self.declare_parameter('rot_safety_radius', 0.35)

        # Safety override buttons (index list)
        self.override_buttons = [6, 7]

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
        self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
        self.max_lateral_speed = float(self.get_parameter('max_lateral_speed').value)
        self.slowdown_margin = float(self.get_parameter('slowdown_margin').value)
        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.lateral_avoid_gain = float(self.get_parameter('lateral_avoid_gain').value)
        self.rot_safety_radius = float(self.get_parameter('rot_safety_radius').value)
        self.stop_margin = float(self.get_parameter('stop_margin').value)


        # State
        self.latest_costmap: Optional[OccupancyGrid] = None
        self.latest_odom: Optional[Odometry] = None
        self.latest_cmd_in: Optional[Twist] = None
        self.joy_override_active: bool = False

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
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
        self.sub_joy = self.create_subscription(
            Joy, '/joy', self.on_joy, default_qos
        )
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_out_topic, 10)

        # 20 Hz control loop
        self.timer = self.create_timer(0.05, self.control_step)

        self.get_logger().info('collision_avoid_cmd_mux (holonomic) with joy override ready.')

    # ---- Callbacks ----
    def on_costmap(self, msg: OccupancyGrid):
        self.latest_costmap = msg

    def on_odom(self, msg: Odometry):
        self.latest_odom = msg

    def on_cmd_in(self, msg: Twist):
        self.latest_cmd_in = msg

    def on_joy(self, msg: Joy):
        active = all((i < len(msg.buttons) and msg.buttons[i] == 1) for i in self.override_buttons)
        self.joy_override_active = active

    # ---- Core logic ----
    def control_step(self):
        cmd_in = self.latest_cmd_in
        if cmd_in is None:
            return

        if self.joy_override_active:
            self.get_logger().info("Joy override active: forwarding raw input")
            self.pub_cmd.publish(cmd_in)
            return

        if self.latest_costmap is None or self.latest_odom is None:
            self.get_logger().info("Costmap or odom not available, forwarding input")
            self.pub_cmd.publish(cmd_in)
            return

        pose_map = self.get_robot_pose_in_map()
        if pose_map is None:
            self.get_logger().info("Could not get robot pose in map, forwarding input")
            self.pub_cmd.publish(cmd_in)
            return

        rx, ry, r_yaw = pose_map

        # Body-frame commanded linear velocity
        vx_b = float(cmd_in.linear.x)
        vy_b = float(cmd_in.linear.y)
        v_mag = math.hypot(vx_b, vy_b)

        cmd_out = Twist()
        cmd_out.linear.x = vx_b
        cmd_out.linear.y = vy_b
        cmd_out.angular.z = cmd_in.angular.z

        # If essentially rotation-only, do a simple radial safety check and pass through
        if v_mag < 1e-3:
            if self.is_obstacle_within_radius(rx, ry, r_yaw, self.rot_safety_radius):
                # discourage spinning in tight space
                cmd_out.angular.z = max(-self.max_angular_speed,
                                        min(self.max_angular_speed,
                                            0.5 * cmd_in.angular.z))
            self.pub_cmd.publish(cmd_out)
            return

        # Map-frame direction of motion = R(yaw) * [vx_b, vy_b]
        cos_y = math.cos(r_yaw)
        sin_y = math.sin(r_yaw)
        dx = cos_y * vx_b - sin_y * vy_b
        dy = sin_y * vx_b + cos_y * vy_b
        # Normalize to unit direction
        inv = 1.0 / max(1e-6, math.hypot(dx, dy))
        ux, uy = dx * inv, dy * inv
        # Lateral unit (left of motion)
        lx, ly = -uy, ux

        min_clear, left_density, right_density = self.scan_corridor(rx, ry, ux, uy, lx, ly)

        if min_clear is None:
            # unknown ahead/behind: be cautious
            max_lin = self.max_linear_speed
            scale = min(1.0, max_lin / max(1e-6, v_mag))
            cmd_out.linear.x *= scale
            cmd_out.linear.y *= scale
            self.pub_cmd.publish(cmd_out)
            return

        # Compute allowed magnitude given slowdown margin
        if min_clear > (self.slowdown_margin + 0.8 * v_mag):
            # publish as-is
            self.pub_cmd.publish(cmd_out)
            return

        # Hard stop if too close
        if min_clear < self.stop_margin:
            cmd_out.linear.x = 0.0
            cmd_out.linear.y = 0.0
            self.pub_cmd.publish(cmd_out)
            self.get_logger().info(f"Emergency stop: obstacle at {min_clear:.2f} m")
            return

        if min_clear < self.slowdown_margin:
            # Scale down linearly between stop_margin and slowdown_margin
            scale = (min_clear - self.stop_margin) / max(1e-3, self.slowdown_margin - self.stop_margin)
            scale = max(0.0, min(1.0, scale))
            cmd_out.linear.x *= scale
            cmd_out.linear.y *= scale

        self.pub_cmd.publish(cmd_out)

        # Steering away from denser side
        steer = 0.0
        if (left_density is not None) and (right_density is not None):
            diff = left_density - right_density
            steer_dir = -1.0 if diff > 0.0 else 1.0  # steer away from denser side
            magnitude = min(1.0, abs(diff))
            steer = steer_dir * self.max_angular_speed * magnitude
        elif left_density is not None:
            steer = -0.5 * self.max_angular_speed
        elif right_density is not None:
            steer = 0.5 * self.max_angular_speed

        # Apply angular steer (diff drive) and a small lateral bias if robot supports vy
        cmd_out.angular.z = max(-self.max_angular_speed,
                                min(self.max_angular_speed,
                                    cmd_out.angular.z + steer))

        # Lateral nudge away from denser side for holonomic bases
        if self.lateral_avoid_gain > 0.0 and v_mag > 1e-6:
            # desired lateral change in map frame (per unit time)
            vy_nudge_map = 0.0
            if (left_density is not None) and (right_density is not None):
                vy_nudge_map = (-1.0 if left_density > right_density else 1.0) * self.lateral_avoid_gain * self.max_lateral_speed
            elif left_density is not None:
                vy_nudge_map = -0.5 * self.lateral_avoid_gain * self.max_lateral_speed
            elif right_density is not None:
                vy_nudge_map = 0.5 * self.lateral_avoid_gain * self.max_lateral_speed

            # map-frame lateral unit is (lx, ly); project nudge onto body-frame y
            # body-frame y axis in map frame is R(yaw) * [0,1] = [-sin(yaw), cos(yaw)]
            byx, byy = -sin_y, cos_y
            # magnitude along body y is dot(nudge_vec_map, body_y_dir)
            nudge_body_y = vy_nudge_map * (lx * byx + ly * byy)
            cmd_out.linear.y += nudge_body_y

        # Hard stop if too close
        if (min_clear < self.stop_margin):
            cmd_out.linear.x = 0.0
            cmd_out.linear.y = 0.0
            self.pub_cmd.publish(cmd_out)
            self.get_logger().info(f"Emergency stop: obstacle at {min_clear:.2f} m")
            return


        self.pub_cmd.publish(cmd_out)

    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        """Return (x, y, yaw) of robot in the map frame without tf_transformations."""
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

    # ---- Scanning helpers ----
    def is_obstacle_within_radius(self, rx: float, ry: float, r_yaw: float, radius: float) -> bool:
        grid = self.latest_costmap
        if grid is None:
            return False
        res = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        width = grid.info.width
        height = grid.info.height
        data = grid.data

        def world_to_idx(wx, wy):
            mx = int((wx - origin_x) / res)
            my = int((wy - origin_y) / res)
            if mx < 0 or my < 0 or mx >= width or my >= height:
                return None
            return my * width + mx

        step = max(res, 0.05)
        r2 = radius * radius
        x = rx - radius
        while x <= rx + radius:
            y = ry - radius
            while y <= ry + radius:
                if (x - rx) ** 2 + (y - ry) ** 2 <= r2:
                    idx = world_to_idx(x, y)
                    if idx is not None and data[idx] >= self.occupied_threshold:
                        return True
                y += step
            x += step
        return False

    def scan_corridor(self, rx: float, ry: float, ux: float, uy: float, lx: float, ly: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Scan a rectangular corridor aligned with the unit motion direction (ux,uy)
        and its left lateral unit (lx,ly). Returns:
        - min_clearance (meters along +/− the motion direction)
        - left_density (0..1) occupancy fraction in left band
        - right_density (0..1) occupancy fraction in right band
        Note: corridor extends only in the +direction of (ux,uy); if the commanded
        velocity is negative along body x, (ux,uy) already points backwards in map.
        """
        grid = self.latest_costmap
        res = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        width = grid.info.width
        height = grid.info.height
        data = grid.data  # flat row-major

        def world_to_idx(wx, wy):
            mx = int((wx - origin_x) / res)
            my = int((wy - origin_y) / res)
            if mx < 0 or my < 0 or mx >= width or my >= height:
                return None
            return my * width + mx

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
                wx = rx + ux * dist + lx * offset
                wy = ry + uy * dist + ly * offset
                idx = world_to_idx(wx, wy)
                if idx is None:
                    break
                occ = data[idx]
                if occ >= self.occupied_threshold:
                    if (min_clear is None) or (dist < min_clear):
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
