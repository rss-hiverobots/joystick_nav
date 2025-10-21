#!/usr/bin/env python3
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
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

# POINT_A = {"x": -4.093898296356201, "y": -10.793622970581055, "oz": -0.7973068595125722, "ow": 0.6035741642699756}
# POINT_B = {
#     "x": -0.7935936450958252,
#     "y": -4.847171783447266,
#     "oz": -0.19851819594062212,
#     "ow": 0.9800972022613271,
# }


class GoToPose(Node):
    def __init__(self):
        super().__init__('go_to_pose')
        self.client        = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.rest_pub      = self.create_publisher(Bool, '/rest', 10)
        self.cmd_pub       = self.create_publisher(Twist, '/grasp/cmd_vel', 10)
        self.grasp_pub     = self.create_publisher(Bool, '/grasp', 10)
        self.drop_pub      = self.create_publisher(Bool, '/drop', 10)
        self.approach_pub  = self.create_publisher(Bool, '/approach', 10)

        # Edge-triggered /movement_complete logic
        self._movement_done_evt   = threading.Event()
        self._awaiting_phase      = None          # None | 'approach' | 'drop' | 'rest'
        self._awaiting_gen_min    = -1
        self._mc_gen              = 0
        self._lock                = threading.Lock()
        self.create_subscription(Bool, '/movement_complete', self._on_movement_complete, 10)

        print("⏳ Waiting for Nav2 action server...")
        self.client.wait_for_server()
        print("✅ Nav2 server available.")
        self._send_nav_goal(POINT_A, tag="Point A", done_cb=self._after_point_a)

    # ---------- movement complete handling ----------
    def _on_movement_complete(self, msg: Bool):
        with self._lock:
            self._mc_gen += 1
            gen_now = self._mc_gen
            phase = self._awaiting_phase
            gen_min = self._awaiting_gen_min

        if phase is not None and msg.data and gen_now > gen_min:
            print(f"✅ /movement_complete(True) accepted for phase '{phase}' (gen {gen_now} > {gen_min})")
            self._movement_done_evt.set()
        else:
            # print(f"↩️ Ignored stale /movement_complete (phase={phase}, data={msg.data}, gen_now={gen_now}, gen_min={gen_min})")
            pass

    def _wait_for_mc(self, phase: str, timeout_s: float = 60.0) -> bool:
        """Wait for a NEW /movement_complete=True for a given phase."""
        with self._lock:
            self._awaiting_phase = phase
            self._awaiting_gen_min = self._mc_gen
            self._movement_done_evt.clear()
            gen_min = self._awaiting_gen_min
        print(f"⏳ Waiting for NEW /movement_complete=True (phase: {phase}, starting_gen={gen_min}) ...")

        ok = self._movement_done_evt.wait(timeout=timeout_s)

        with self._lock:
            self._awaiting_phase = None
            self._awaiting_gen_min = -1

        if not ok:
            print(f"⚠️ Timeout waiting for /movement_complete (phase: {phase}); continuing anyway.")
        return ok

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
        while pub.get_subscription_count() == 0 and (time.time() - t0) < timeout and rclpy.ok():
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

    # ========== Main sequence steps ==========
    def _after_point_a(self, tag, result_fut):
        result = result_fut.result()
        print(f"📬 {tag} result received (status={getattr(result, 'status', 'unknown')}).")

        # 1️⃣ REST
        print("📢 Publishing /rest: True")
        self.rest_pub.publish(Bool(data=True))
        self._wait_for_mc('rest', timeout_s=20.0)

        # 2️⃣ MOVE FORWARD
        # print("➡️  Forward 2.2s @ 0.2 m/s")
        # self._drive_for(+0.2, 2.0)
        # print("🛑 Forward movement complete")

        # 3️⃣ APPROACH
        self._wait_for_subscribers(self.approach_pub, "/approach", timeout=5.0)
        self.approach_pub.publish(Bool(data=True))
        print("🚶 Published /approach: True")

        threading.Thread(target=self._continue_after_approach, daemon=True).start()

    def _continue_after_approach(self):
        self._wait_for_mc('approach', timeout_s=60.0)

        print("⬅️  Backward 2.5s @ 0.2 m/s")
        self._drive_for(-0.2, 2.5)
        print("🛑 Backward movement complete")

        print("🧭 Navigating to Point B...")
        self._send_nav_goal(POINT_B, tag="Point B", done_cb=self._after_point_b)

    def _after_point_b(self, tag, result_fut):
        result = result_fut.result()
        print(f"📬 {tag} result received (status={getattr(result, 'status', 'unknown')}).")

        # 4️⃣ MOVE FORWARD
        print("➡️  Forward 2.5s @ 0.2 m/s")
        self._drive_for(+0.2, 2.0)
        print("🛑 Forward movement complete")

        # 5️⃣ DROP
        self.drop_pub.publish(Bool(data=True))
        print("📦 Published /drop: True")
        self._wait_for_mc('drop', timeout_s=20.0)

        # 6️⃣ REST AGAIN
        print("😌 Publishing /rest: True")
        self.rest_pub.publish(Bool(data=True))

        # 7️⃣ MOVE BACKWARD
        print("⬅️  Backward 2.2s @ 0.2 m/s")
        self._drive_for(-0.2, 2.2)
        print("🛑 Backward movement complete")

        print("✅ Sequence complete. Shutting down.")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = GoToPose()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
