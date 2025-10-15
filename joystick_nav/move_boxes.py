#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist

POINT_A = {"x": -6.651529312133789, "y": -17.346385955810547, "oz": -0.8349573785198705, "ow": 0.5503146155202728}
POINT_B = {
    "x": -1.714616298675537,
    "y": -18.474031448364258,
    "oz": 0.527993528947017,
    "ow": 0.8492483932219569,
}

class GoToPose(Node):
    def __init__(self):
        super().__init__('go_to_pose')
        self.client     = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.rest_pub   = self.create_publisher(Bool, '/rest', 10)
        self.cmd_pub    = self.create_publisher(Twist, '/grasp/cmd_vel', 10)
        self.grasp_pub  = self.create_publisher(Bool, '/grasp', 10)
        self.drop_pub   = self.create_publisher(Bool, '/drop', 10)

        print("⏳ Waiting for Nav2 action server...")
        self.client.wait_for_server()
        print("✅ Nav2 server available.")

        input("🔸 Press [Enter] to send Point A...")
        self._send_nav_goal(POINT_A, tag="Point A", done_cb=self._after_point_a)

    # ---------- helpers ----------
    def _send_nav_goal(self, pose_dict, tag, done_cb):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(pose_dict["x"])
        goal.pose.pose.position.y = float(pose_dict["y"])
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = float(pose_dict["oz"])
        goal.pose.pose.orientation.w = float(pose_dict["ow"])

        print(f"🧭 Sending {tag}...")
        send_future = self.client.send_goal_async(goal)

        def _on_goal_response(fut):
            try:
                goal_handle = fut.result()
            except Exception as e:
                print(f"❌ {tag} send failed: {e}")
                return
            if not goal_handle.accepted:
                print(f"❌ {tag} was REJECTED by server.")
                return
            print(f"✅ {tag} accepted; navigating...")
            goal_handle.get_result_async().add_done_callback(lambda r: done_cb(tag, r))

        send_future.add_done_callback(_on_goal_response)

    def _wait_for_subscribers(self, pub, name, timeout=5.0):
        t0 = time.time()
        while pub.get_subscription_count() == 0 and (time.time() - t0) < timeout:
            print(f"⏳ Waiting for a subscriber on {name}...")
            time.sleep(0.1)
        print(f"🔌 {name} subscriber count = {pub.get_subscription_count()}")
        return pub.get_subscription_count() > 0

    def _drive_for(self, vx, duration_s):
        if not self._wait_for_subscribers(self.cmd_pub, "/grasp/cmd_vel", timeout=5.0):
            print("⚠️ No subscribers on /grasp/cmd_vel. Skipping motion.")
            return
        self.cmd_pub.publish(Twist())  # wake
        time.sleep(0.05)
        twist = Twist()
        twist.linear.x = float(vx)
        t0 = time.time()
        while time.time() - t0 < duration_s and rclpy.ok():
            self.cmd_pub.publish(twist)
            time.sleep(0.1)
        self.cmd_pub.publish(Twist())

    # ---------- sequence ----------
    def _after_point_a(self, tag, result_fut):
        result = result_fut.result()
        print(f"📬 {tag} result received (status={getattr(result, 'status', 'unknown')}).")

        print("📢 Publishing /rest: True")
        self.rest_pub.publish(Bool(data=True))

        input("🔸 Press [Enter] to move forward...")
        print("➡️  Forward 2.2s @ 0.2 m/s")
        self._drive_for(+0.2, 2.2)
        print("🛑 Forward movement complete")

        input("🔸 Press [Enter] to GRASP...")
        self.grasp_pub.publish(Bool(data=True))
        print("🤝 Published /grasp: True")

        input("🔸 Press [Enter] to move backward...")
        print("⬅️  Backward 2.5s @ 0.2 m/s")
        self._drive_for(-0.2, 2.5)
        print("🛑 Backward movement complete")

        print("🧭 Navigating to Point B...")
        self._send_nav_goal(POINT_B, tag="Point B", done_cb=self._after_point_b)

    def _after_point_b(self, tag, result_fut):
        result = result_fut.result()
        print(f"📬 {tag} result received (status={getattr(result, 'status', 'unknown')}).")

        # Move forward 2.5s
        print("➡️  Forward 2.5s @ 0.2 m/s")
        self._drive_for(+0.2, 2.5)
        print("🛑 Forward movement complete")

        # Wait for Enter, then drop
        input("🔸 Press [Enter] to DROP (publish /drop True once)... ")
        self.drop_pub.publish(Bool(data=True))
        print("📦 Published /drop: True")

        # Wait for Enter, then back 2.2s and /rest
        input("🔸 Press [Enter] to move backward and REST... ")
        print("⬅️  Backward 2.2s @ 0.2 m/s")
        self._drive_for(-0.2, 2.2)
        print("🛑 Backward movement complete")

        print("😌 Publishing /rest: True")
        self.rest_pub.publish(Bool(data=True))

        print("✅ Sequence complete. Shutting down.")
        rclpy.shutdown()

def main():
    rclpy.init()
    node = GoToPose()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == "__main__":
    main()
