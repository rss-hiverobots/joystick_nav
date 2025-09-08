import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time

from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException

import numpy as np
import math
import networkx as nx

"""
1. Create a partial map: ros2 launch my_turtlebot turtlebot_simulation.launch.py slam:=True
2. Start the exploration node: ros2 run my_turtlebot 05_Exploration
"""

class Exploration(Node):
    """
    Subscribes to /map and uses TF map->base_link for localization.
    Samples exploration nodes and builds a PRM.
    Publishes sampled nodes and edges as MarkerArray for RViz.
    Computes info gain for each node and navigates to the best node.
    Loops after reaching each goal.
    """
    def __init__(self):
        super().__init__('exploration')

        self.map = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.05
        self.map_origin = None

        self.exploration_nodes = []
        self.navigation_active = False

        self.robot_x = 0.0
        self.robot_y = 0.0

        # --- Subscriptions / Publishers / Actions ---
        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        # (Optional) keep odom sub if you want, but localization now comes from TF:
        # self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)

        self.map_publisher = self.create_publisher(MarkerArray, '/prm_markers', 10)
        self.edge_publisher = self.create_publisher(MarkerArray, '/prm_edges', 10)
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # --- TF localization: map -> base_link ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_timer = self.create_timer(0.1, self.update_robot_pose)  # 10 Hz

    # Kept for reference; no longer used for localization
    def pose_callback(self, msg: Odometry):
        pass

    def update_robot_pose(self):
        """
        Query TF for transform map->base_link and update self.robot_x/self.robot_y.
        """
        try:
            # Latest available transform
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time())
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
        except TransformException as ex:
            # Use debug to avoid log spam; elevate to warn if you prefer
            self.get_logger().debug(f'No TF map->base_link yet: {ex}')

    def map_callback(self, msg):
        self.map = msg.data
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin.position

        if not self.navigation_active:
            self.get_logger().info('Sampling exploration nodes...')
            self.exploration_nodes.clear()

            marker_array = self.sample_random_nodes(self.map)
            if marker_array:
                for marker in marker_array.markers:
                    x = marker.pose.position.x
                    y = marker.pose.position.y
                    self.exploration_nodes.append((x, y))
                self.map_publisher.publish(marker_array)

                prm_graph = self.build_prm_graph(self.exploration_nodes)
                edge_markers = self.create_edge_markers(prm_graph)
                self.edge_publisher.publish(edge_markers)

                gains, ordered_nodes = self.evaluate_prm_gain(prm_graph)
                for idx, gain in enumerate(gains):
                    self.get_logger().info(f'Node {ordered_nodes[idx]}: Gain = {gain:.2f}')

                if gains:
                    best_idx = int(np.argmax(gains))
                    best_node = ordered_nodes[best_idx]
                    self.navigate_to_node(best_node)

    def sample_random_nodes(self, map_data, N_nodes=100):
        empty_cells = [(i, j) for i in range(self.map_height) for j in range(self.map_width)
                       if map_data[i * self.map_width + j] == 0]
        if len(empty_cells) < N_nodes:
            self.get_logger().warn(f'Not enough empty cells to sample {N_nodes} nodes')
            N_nodes = len(empty_cells)

        sampled_indices = np.random.choice(len(empty_cells), N_nodes, replace=False)
        marker_array = MarkerArray()
        self.exploration_nodes.clear()

        for idx, node in enumerate(sampled_indices):
            i, j = empty_cells[node]
            world_x = j * self.map_resolution + self.map_origin.x
            world_y = i * self.map_resolution + self.map_origin.y

            self.exploration_nodes.append((world_x, world_y))

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "exploration_nodes"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        return marker_array

    def build_prm_graph(self, nodes, connection_radius=2.0):
        graph = nx.Graph()
        for i, node_i in enumerate(nodes):
            graph.add_node(i, pos=node_i)
            for j in range(i + 1, len(nodes)):
                node_j = nodes[j]
                distance = math.hypot(node_i[0] - node_j[0], node_i[1] - node_j[1])
                if distance < connection_radius and self.is_path_free(node_i, node_j):
                    graph.add_edge(i, j, weight=distance)
        return graph

    def is_path_free(self, p1, p2):
        steps = int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]) / self.map_resolution)
        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            x = p1[0] * (1 - t) + p2[0] * t
            y = p1[1] * (1 - t) + p2[1] * t
            col = int((x - self.map_origin.x) / self.map_resolution)
            row = int((y - self.map_origin.y) / self.map_resolution)
            if not (0 <= row < self.map_height and 0 <= col < self.map_width):
                return False
            val = self.map[row * self.map_width + col]
            if val == 100:
                return False
        return True

    def create_edge_markers(self, graph):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "prm_edges"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        for u, v in graph.edges:
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            p1 = self.create_point(pos_u)
            p2 = self.create_point(pos_v)
            marker.points.append(p1)
            marker.points.append(p2)

        marker_array.markers.append(marker)
        return marker_array

    def create_point(self, pos):
        p = Point()
        p.x = pos[0]
        p.y = pos[1]
        p.z = 0.1
        return p

    def evaluate_prm_gain(self, graph, lam=1.0):
        if graph.number_of_nodes() == 0:
            return [], []

        robot_pos = (self.robot_x, self.robot_y)
        graph.add_node("robot", pos=robot_pos)

        for node_id, data in graph.nodes(data=True):
            if node_id == "robot":
                continue
            if self.is_path_free(robot_pos, data['pos']):
                dist = math.hypot(robot_pos[0] - data['pos'][0], robot_pos[1] - data['pos'][1])
                graph.add_edge("robot", node_id, weight=dist)

        best_gain = -float('inf')
        best_node = None
        all_gains = []
        all_targets = []

        for target in graph.nodes:
            if target == "robot":
                continue
            try:
                path = nx.shortest_path(graph, source="robot", target=target, weight='weight')
            except nx.NetworkXNoPath:
                continue

            gain = 0.0
            for i in range(1, len(path)):
                prev = path[i - 1]
                curr = path[i]
                p_prev = graph.nodes[prev]['pos']
                p_curr = graph.nodes[curr]['pos']
                d = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                visible = self.compute_visible_unknown(p_curr)
                gain += visible * math.exp(-lam * d)

            all_gains.append(gain)
            all_targets.append(graph.nodes[path[-1]]['pos'])

            if gain > best_gain:
                best_gain = gain
                best_node = graph.nodes[path[-1]]['pos']

        return all_gains, all_targets

    def compute_visible_unknown(self, node, max_distance=3):
        directions = 60
        max_cells = int(max_distance / self.map_resolution)
        total_unknown = 0
        for d in range(directions):
            angle = 2 * math.pi * d / directions
            total_unknown += self.ray_cast_unknown(node, angle, max_cells)
        return total_unknown

    def ray_cast_unknown(self, node, angle, max_cells):
        origin_x, origin_y = node
        count = 0
        for step in range(1, max_cells + 1):
            rx = origin_x + step * self.map_resolution * math.cos(angle)
            ry = origin_y + step * self.map_resolution * math.sin(angle)
            col = int((rx - self.map_origin.x) / self.map_resolution)
            row = int((ry - self.map_origin.y) / self.map_resolution)
            if not (0 <= row < self.map_height and 0 <= col < self.map_width):
                break
            val = self.map[row * self.map_width + col]
            if val == 100:
                break
            elif val == -1:
                count += 1
        return count

    def navigate_to_node(self, node):
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = node[0]
        goal_msg.pose.pose.position.y = node[1]
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info(f"Sending goal to navigation: {node}")
        self.navigation_active = True
        self.nav_to_pose_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Navigation goal was rejected.')
            self.navigation_active = False
            return

        self.get_logger().info('Navigation goal accepted.')
        goal_handle.get_result_async().add_done_callback(self.navigation_result_callback)

    def navigation_result_callback(self, future):
        self.get_logger().info('Navigation completed')
        self.navigation_active = False
        self.get_logger().info('Waiting 3 seconds before resampling...')
        self.timer = self.create_timer(3.0, lambda: (self.trigger_resample(), self.timer.cancel()))

    def trigger_resample(self):
        self.get_logger().info('Triggering new exploration cycle.')

def main(args=None):
    rclpy.init(args=args)
    node = Exploration()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
