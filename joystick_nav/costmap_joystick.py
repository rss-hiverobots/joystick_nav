#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Safety-gating velocity node for Livox + joystick control (ROS 2 Humble, Python).

- Sub:  /livox/lidar      (livox_ros_driver2/msg/CustomMsg)
- Sub:  /cmd_vel_joy      (geometry_msgs/msg/Twist)   <-- raw joystick cmd
- Sub:  /joy              (sensor_msgs/msg/Joy)       <-- for override buttons
- Pub:  /cmd_vel_joy_safe (geometry_msgs/msg/Twist)   <-- safety-filtered cmd

Logic:
- From latest Livox CustomMsg, discard points considered floor (z <= floor_z_max)
  and points closer than min_ignore_range (sensor clutter/hood).
- When a /cmd_vel_joy arrives:
    * If override is active (both buttons pressed), forward it as-is.
    * Else, compute heading from (vx, vy). Measure the minimum distance to any
      non-floor point inside an angular sector around that heading. Scale linear
      velocity by:
          scale = clamp((d - stop_distance) / (slow_distance - stop_distance), 0..1)
      so it slows down and stops when d <= stop_distance.
    * If |v| ~ 0, publish the same (near-zero) Twist (no need to compute scale).
- If we haven't yet received /cmd_vel_joy, do not publish anything.

Notes:
- Assumes Livox frame z-axis is up (or roughly aligned to base frame z). Adjust
  floor_z_max if your sensor sits below base_link or is tilted.
- For multi-directional motion (vx, vy), the sector points in the instant
  intended motion direction.
"""

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

# Livox messages
from livox_ros_driver2.msg import CustomMsg  # points are CustomPoint[] within

class CmdVelSafety(Node):
    def __init__(self):
        super().__init__('cmd_vel_joy_safety')

        # ---------------- Parameters (declare & defaults) ----------------
        # Safety distances (meters)
        self.declare_parameter('slow_distance', 1.0)        # start slowing
        self.declare_parameter('stop_distance', 0.2)        # full stop
        self.declare_parameter('min_ignore_range', 0.15)    # ignore ultra-near speckle
        self.declare_parameter('floor_z_max', -1.0)         # (m) z <= this is considered floor
        self.declare_parameter('sector_half_angle_deg', 35.0) # half-angle of sector
        self.declare_parameter('lidar_timeout_sec', 0.25)   # consider lidar stale after this
        self.declare_parameter('override_buttons', [6, 7])  # both must be pressed
        self.declare_parameter('zero_hold_sec', 2.0)  # keep publishing zero for this long

        # Topics (allow remapping via params if you like)
        self.declare_parameter('lidar_topic', '/livox/lidar')
        self.declare_parameter('cmd_in_topic', '/cmd_vel_joy')
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('cmd_out_topic', '/cmd_vel_joy_safe')

        # Read params
        self.slow_distance = float(self.get_parameter('slow_distance').value)
        self.stop_distance = float(self.get_parameter('stop_distance').value)
        self.min_ignore_range = float(self.get_parameter('min_ignore_range').value)
        self.floor_z_max = float(self.get_parameter('floor_z_max').value)
        self.sector_half_angle = math.radians(
            float(self.get_parameter('sector_half_angle_deg').value)
        )
        self.lidar_timeout_sec = float(self.get_parameter('lidar_timeout_sec').value)
        self.override_buttons: List[int] = list(self.get_parameter('override_buttons').value)
        self.zero_hold_sec = float(self.get_parameter('zero_hold_sec').value)

        lidar_topic = self.get_parameter('lidar_topic').value
        cmd_in_topic = self.get_parameter('cmd_in_topic').value
        joy_topic = self.get_parameter('joy_topic').value
        cmd_out_topic = self.get_parameter('cmd_out_topic').value

        # ---------------- State ----------------
        self.latest_lidar_msg: Optional[CustomMsg] = None
        self.latest_lidar_stamp: Optional[rclpy.time.Time] = None
        self.latest_joy_msg: Optional[Joy] = None
        self.have_cmd_in: bool = False  # gate publishing until first /cmd_vel_joy arrives
        # Zero-cmd suppression state
        self._zero_since: Optional[rclpy.time.Time] = None  # when zero stream started


        
        # --- Debug state ---
        self._z_stats_logged: bool = False   # print z-range once at startup
        self.last_closest: Optional[tuple] = None  # (x, y, z, d) of closest in sector


        # ---------------- QoS ----------------
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------- Subscriptions ----------------
        self.sub_lidar = self.create_subscription(
            CustomMsg, lidar_topic, self.on_lidar, qos_sensor
        )
        self.sub_cmd = self.create_subscription(
            Twist, cmd_in_topic, self.on_cmd, qos_cmd
        )
        self.sub_joy = self.create_subscription(
            Joy, joy_topic, self.on_joy, qos_sensor
        )

        # ---------------- Publisher ----------------
        self.pub_cmd = self.create_publisher(Twist, cmd_out_topic, qos_cmd)

        self.get_logger().info(
            f"cmd_vel safety active | slow={self.slow_distance:.2f} m, stop={self.stop_distance:.2f} m, "
            f"floor_z_max={self.floor_z_max:.2f} m, sector={math.degrees(self.sector_half_angle):.1f}°"
        )

    # ---------- Callbacks ----------
    def on_lidar(self, msg: CustomMsg):
        self.latest_lidar_msg = msg
        # Debug: print z-range once at beginning
        if not self._z_stats_logged and msg.points:
            z_min = min(p.z for p in msg.points)
            z_max = max(p.z for p in msg.points)
            self.get_logger().info(f"[DBG] Livox Z range (first packet): min={z_min:.3f} m, max={z_max:.3f} m")
            self._z_stats_logged = True

        # Convert msg time to ROS time (header.stamp if available; CustomMsg has header)
        self.latest_lidar_stamp = self.get_clock().now()

    def on_joy(self, msg: Joy):
        self.latest_joy_msg = msg

    def on_cmd(self, msg: Twist):
        # We only publish **in reaction** to an incoming cmd (no periodic 0s).
        self.have_cmd_in = True

        # If override buttons are BOTH pressed, bypass safety
        if self._override_active():
            self._zero_since = None  # reset zero timer
            self.pub_cmd.publish(msg)
            return

        # If no significant motion commanded, forward as-is (it’s safe)
        vx, vy = msg.linear.x, msg.linear.y
        vmag = math.hypot(vx, vy)
        if vmag < 1e-3 and abs(msg.angular.z) < 1e-3:
            # Zero linear & zero angular: publish zeros only for zero_hold_sec, then stop publishing.
            now = self.get_clock().now()
            if self._zero_since is None:
                self._zero_since = now
            elapsed = (now - self._zero_since).nanoseconds * 1e-9
            if elapsed <= self.zero_hold_sec:
                self.pub_cmd.publish(msg)  # pass through zeros briefly (e.g., to stop smoothly)
            # else: do not publish (go silent)
            return
        else:
            # Non-zero command (either linear or angular) — reset the zero timer
            self._zero_since = None


        # Compute safety scale based on closest obstacle in heading sector
        heading = math.atan2(vy, vx)  # radians in base frame (assuming lidar frame ~= base)
        min_d = self._min_distance_in_sector(heading)

        if min_d is None:
            # No fresh lidar or no valid points -> be permissive: pass through
            self.pub_cmd.publish(msg)
            return
        
        # Debug: print closest object detected in the commanded sector
        if self.last_closest is not None:
            cx, cy, cz, cd = self.last_closest
            self.get_logger().info(
                f"[DBG] Closest obj in sector: d={cd:.3f} m at (x={cx:.3f}, y={cy:.3f}, z={cz:.3f})"
            )


        # Scale linear speed based on distance
        scale = self._speed_scale(min_d)
        safe = Twist()
        safe.linear.x = msg.linear.x * scale
        safe.linear.y = msg.linear.y * scale
        # Keep angular as commanded; optionally reduce if very close:
        if scale <= 0.25:
            safe.angular.z = msg.angular.z * max(scale, 0.25)
        else:
            safe.angular.z = msg.angular.z

        # If we had to stop (scale==0), ensure truly zero
        if scale <= 0.0:
            safe.linear.x = 0.0
            safe.linear.y = 0.0
            safe.angular.z = 0.0

        self.pub_cmd.publish(safe)

    # ---------- Helpers ----------
    def _override_active(self) -> bool:
        if self.latest_joy_msg is None or not self.latest_joy_msg.buttons:
            return False
        # Both buttons pressed (value == 1)
        try:
            return all(
                0 <= b < len(self.latest_joy_msg.buttons) and self.latest_joy_msg.buttons[b] == 1
                for b in self.override_buttons
            )
        except Exception:
            return False

    def _lidar_is_fresh(self) -> bool:
        if self.latest_lidar_stamp is None:
            return False
        age = (self.get_clock().now() - self.latest_lidar_stamp).nanoseconds * 1e-9
        return age <= self.lidar_timeout_sec

    def _min_distance_in_sector(self, heading_rad: float) -> Optional[float]:
        """
        Returns the minimum planar (xy) distance to any non-floor, non-ignored point
        within the angular sector centered on heading_rad. None if lidar stale or no points.
        """
        if not self._lidar_is_fresh() or self.latest_lidar_msg is None:
            return None

        pts = self.latest_lidar_msg.points
        if not pts:
            return None

        half = self.sector_half_angle
        min_d = None
        closest_tuple = None  # (x, y, z, d)


        # Precompute bounds to avoid repeated function calls
        stop2 = self.stop_distance * self.stop_distance
        ignore2 = self.min_ignore_range * self.min_ignore_range

        # Iterate raw Livox points (CustomPoint: x,y,z, etc.)
        for p in pts:
            z = p.z
            if z <= self.floor_z_max:
                continue  # ignore floor
            x, y = p.x, p.y
            d2 = x * x + y * y
            if d2 < ignore2:
                continue  # ignore ultra-near speckle
            # Angle of point in the plane
            ang = math.atan2(y, x)
            # Wrap smallest angular difference
            dif = self._ang_diff(ang, heading_rad)
            if abs(dif) <= half:
                # candidate in sector
                if min_d is None or d2 < min_d * min_d:
                    d = math.sqrt(d2)
                    closest_tuple = (x, y, z, d)
                    # Fast path: if already within stop radius, we can short-circuit
                    if d2 <= stop2:
                        self.last_closest = closest_tuple
                        return d
                    min_d = d

        # Save for debug printing in on_cmd
        if min_d is not None and closest_tuple is not None:
            self.last_closest = closest_tuple
        else:
            self.last_closest = None
        return min_d


    @staticmethod
    def _ang_diff(a: float, b: float) -> float:
        """Smallest signed angle a-b in [-pi, pi]."""
        d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
        return d

    def _speed_scale(self, d: float) -> float:
        """
        Piecewise-linear speed scaling:
            d <= stop_distance      -> 0
            d >= slow_distance      -> 1
            otherwise               -> (d - stop) / (slow - stop)
        """
        if d <= self.stop_distance:
            return 0.0
        if d >= self.slow_distance:
            return 1.0
        # Avoid division by zero if params misconfigured
        denom = max(self.slow_distance - self.stop_distance, 1e-6)
        return max(0.0, min(1.0, (d - self.stop_distance) / denom))


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelSafety()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
