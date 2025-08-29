#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Quaternion
from builtin_interfaces.msg import Time
from math import sin, cos

# 👇 Edit this dictionary to match your map locations.
# Each entry: name -> (x, y, z, yaw_radians)
ROOMS = {
    "office": (18.72, -9.39, 0.0, 0.0),
    "conference_room": (25.55, -8.08, 0.0, 0.0),
    "lobby": (24.20, 4.72, 0.0, 0.0),
    "kitchen": (20.23, 3.54, 0.0, 0.0),
    "living_room": (6.37, 3.56, 0.0, 0.0),
    "dinning_room": (-4.04, -9.66, 0.0, 0.0),
}


DEFAULT_FRAME = "map"
NAV2_ACTION = "/navigate_to_pose"

def yaw_to_quat(yaw: float) -> Quaternion:
    # roll = pitch = 0, yaw variable
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = sin(yaw / 2.0)
    q.w = cos(yaw / 2.0)
    return q

class RoomNavigator(Node):
    def __init__(self):
        super().__init__("room_navigator")
        self.cli = ActionClient(self, NavigateToPose, NAV2_ACTION)
        self._last_logged_dist = None   # track last distance we logged

    def send_goal(self, room_name: str, frame_id: str = DEFAULT_FRAME) -> bool:
        if room_name not in ROOMS:
            self.get_logger().error(
                f"Unknown room '{room_name}'. Known: {', '.join(sorted(ROOMS.keys()))}"
            )
            return False

        x, y, z, yaw = ROOMS[room_name]
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation = yaw_to_quat(float(yaw))

        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.get_logger().info(f"Waiting for Nav2 action server at '{NAV2_ACTION}'...")
        if not self.cli.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Nav2 action server not available.")
            return False

        self.get_logger().info(f"Sending goal to room '{room_name}' ({x:.3f}, {y:.3f}, yaw {yaw:.2f} rad)")

        send_future = self.cli.send_goal_async(
            goal, feedback_callback=self._feedback_cb
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal was rejected by Nav2.")
            return False

        self.get_logger().info("Goal accepted. Navigating...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("No result returned from Nav2.")
            return False

        status = result.status
        if status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info("Navigation succeeded ✅")
            return True
        else:
            self.get_logger().warn(f"Navigation finished with status code: {status}")
            return False

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist = getattr(fb, "distance_remaining", None)
        if dist is not None:
            if self._last_logged_dist is None or abs(dist - self._last_logged_dist) >= 1.0:
                self.get_logger().info(f"Distance remaining: {dist:.2f} m")
                self._last_logged_dist = dist

def main():
    rclpy.init()
    node = RoomNavigator()

    try:
        # Simple CLI prompt
        print("Available rooms:", ", ".join(sorted(ROOMS.keys())))
        room = input("Which room do you want to go to? ").strip()
        if not room:
            print("No room provided. Exiting.")
            rclpy.shutdown()
            return

        ok = node.send_goal(room)
        sys.exit(0 if ok else 2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
