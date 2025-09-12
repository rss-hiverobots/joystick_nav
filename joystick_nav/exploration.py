#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PoseStamped, Quaternion

import numpy as np
import math
import networkx as nx

from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose

def make_quaternion_from_yaw(yaw):
    """Creates a geometry_msgs/Quaternion from a yaw angle in radians."""
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q

class PRMExplorer(Node):
    def __init__(self):
        super().__init__('prm_explorer')

        self.N = 100
        self.map = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None

        self.prm_nodes = []
        self.prm_edges = []
        self.selected_node_idx = None
        self.navigation_active = False
        self.goal_sent = False

        # Robot pose will be read from /aft_mapped_to_init
        self.robot_pose = None
        self.create_subscription(PoseStamped, '/aft_mapped_to_init', self.robot_pose_callback, 10)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.map_callback, 1)
        self.markers_pub = self.create_publisher(MarkerArray, '/prm_markers', 1)
        self.edges_pub = self.create_publisher(MarkerArray, '/prm_edges', 1)
        self.timer = self.create_timer(1.0, self.main_loop)

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.initialized = False  # Flag to start first sampling

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg

    def map_callback(self, msg: OccupancyGrid):
        self.map = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        # self.get_logger().info("Received updated costmap.")
        # Trigger initialization if this is the first map received
        if not self.initialized:
            self.initialized = True
            self.start_new_exploration()

    def main_loop(self):
        # Main loop only handles navigation status
        if not self.initialized or self.map is None:
            return

        if self.navigation_active and hasattr(self, 'nav_future'):
            # Wait for navigation to finish
            pass
        elif not self.navigation_active and self.goal_sent:
            # After reaching/canceling a goal, sample/build/select again
            self.start_new_exploration()
            self.goal_sent = False

    def start_new_exploration(self):
        """Sample, build PRM, select node, and navigate."""
        self.sample_nodes()
        self.build_prm()
        self.select_best_node()
        self.publish_markers()
        if self.selected_node_idx is not None:
            wx, wy = self.prm_nodes[self.selected_node_idx]
            self.navigation_active = True
            self.goal_sent = True
            future = self.send_nav_goal(wx, wy)
            if future is not None:
                self.nav_future = future
            else:
                self.navigation_active = False
                self.goal_sent = False
                self.get_logger().warn("Failed to send goal. Skipping navigation.")

    def nav_done_cb(self, future):
        try:
            result = future.result().result
            code = future.result().status
            self.get_logger().info(f"Navigation completed with result: {result} (status code: {code})")
        except Exception as e:
            self.get_logger().warn(f"Error in navigation callback: {e}")
        self.navigation_active = False
        self.selected_node_idx = None

    def send_nav_goal(self, wx, wy):
        """
        Sends a goal to Nav2 and sets up callbacks for acceptance/result.
        """
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 action server not available!")
            return None

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = wx
        goal_msg.pose.pose.position.y = wy
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = make_quaternion_from_yaw(0.0)
        self.get_logger().info(f"Sending goal to Nav2: ({wx:.2f}, {wy:.2f}) in frame map")
        send_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_cb)
        return send_future

    def goal_response_cb(self, future):
        """
        Called when Nav2 responds to the goal (accept/reject).
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by Nav2!')
            self.navigation_active = False
            return
        self.get_logger().info('Goal accepted by Nav2!')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.nav_done_cb)

    def sample_nodes(self):
        """
        Samples N free nodes from the map.
        """
        self.prm_nodes = []
        free_indices = np.argwhere(self.map == 0)
        self.get_logger().info(f"Found {len(free_indices)} free indices for sampling.")
        if len(free_indices) == 0:
            self.get_logger().warn('No free space in map!')
            return

        sampled_indices = free_indices[np.random.choice(len(free_indices), min(self.N, len(free_indices)), replace=False)]
        for idx in sampled_indices:
            y, x = idx
            wx = self.map_origin[0] + x * self.map_resolution
            wy = self.map_origin[1] + y * self.map_resolution
            self.prm_nodes.append((wx, wy))
        self.get_logger().info(f"Sampled {len(self.prm_nodes)} PRM nodes.")

    def build_prm(self):
        """
        Builds the PRM graph by connecting nodes that are mutually reachable.
        """
        self.get_logger().info("Building PRM graph...")
        self.prm_graph = nx.Graph()
        for i, node in enumerate(self.prm_nodes):
            self.prm_graph.add_node(i, pos=node)

        # Connect nodes within a radius (R)
        R = 3.0  # meters
        self.prm_edges = []
        for i, n1 in enumerate(self.prm_nodes):
            for j, n2 in enumerate(self.prm_nodes):
                if i >= j:
                    continue
                dist = math.hypot(n1[0] - n2[0], n1[1] - n2[1])
                if dist < R and self.is_path_free(n1, n2):
                    self.prm_graph.add_edge(i, j)
                    self.prm_edges.append((i, j))
        self.get_logger().info(f"Built PRM with {self.prm_graph.number_of_nodes()} nodes and {self.prm_graph.number_of_edges()} edges.")

    def is_path_free(self, n1, n2):
        """
        Returns True if the straight line between n1 and n2 does not collide with obstacles.
        """
        x0 = int((n1[0] - self.map_origin[0]) / self.map_resolution)
        y0 = int((n1[1] - self.map_origin[1]) / self.map_resolution)
        x1 = int((n2[0] - self.map_origin[0]) / self.map_resolution)
        y1 = int((n2[1] - self.map_origin[1]) / self.map_resolution)
        for x, y in self.bresenham(x0, y0, x1, y1):
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                return False
            if self.map[y, x] != 0:
                return False
        return True

    def bresenham(self, x0, y0, x1, y1):
        """
        Bresenham's line algorithm for grid traversal.
        """
        points = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        return points

    def select_best_node(self, radius_m=0.5, connectivity_weight=3.0):
        """
        Selects the best node for exploration from the PRM.

        The gain for each node is computed as:
            gain = info_gain + connectivity_weight * connectivity

        - info_gain: Number of unknown (-1) cells in a radius around the node.
        - connectivity: Number of edges (neighbors) this node has in the PRM graph.
        - Only nodes in the largest connected PRM component are considered.
        - Disconnected nodes are ignored.

        Args:
            radius_m (float): Radius (in meters) to consider for info_gain around each node.
            connectivity_weight (float): Weight multiplier for the node's connectivity in the gain formula.

        Sets:
            self.selected_node_idx (int): Index of the selected node in self.prm_nodes.
        """
        if self.prm_graph.number_of_nodes() == 0:
            self.selected_node_idx = None
            self.get_logger().warn("No PRM nodes to select from.")
            return

        # Find the largest connected component in the PRM
        components = list(nx.connected_components(self.prm_graph))
        if not components:
            self.selected_node_idx = None
            self.get_logger().warn("No connected components in PRM.")
            return
        main_component = max(components, key=len)
        node_idxs = list(main_component)

        # Parameters
        radius = int(radius_m / self.map_resolution)
        best_score = -np.inf
        best_idx = None

        for i in node_idxs:
            wx, wy = self.prm_nodes[i]
            x = int((wx - self.map_origin[0]) / self.map_resolution)
            y = int((wy - self.map_origin[1]) / self.map_resolution)
            x0 = max(0, x - radius)
            y0 = max(0, y - radius)
            x1 = min(self.map_width, x + radius)
            y1 = min(self.map_height, y + radius)
            local_map = self.map[y0:y1, x0:x1]
            info_gain = np.count_nonzero(local_map == -1)
            connectivity = self.prm_graph.degree[i]
            score = info_gain + connectivity_weight * connectivity
            self.get_logger().debug(f"Node {i}: info_gain={info_gain}, connectivity={connectivity}, score={score}")
            if score > best_score:
                best_score = score
                best_idx = i
        self.selected_node_idx = best_idx
        if best_idx is not None:
            self.get_logger().info(f"Selected node idx: {best_idx} with score {best_score}.")
        else:
            self.get_logger().warn("No node selected after gain calculation.")

    def publish_markers(self):
        """
        Publishes MarkerArrays for PRM nodes and edges to RViz.
        """
        if not self.prm_nodes:
            self.get_logger().debug("No PRM nodes to publish as markers.")
            return

        marker_array = MarkerArray()
        for i, (wx, wy) in enumerate(self.prm_nodes):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "prm_nodes"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.12
            marker.pose.position.x = wx
            marker.pose.position.y = wy
            marker.pose.position.z = 0.1
            if i == self.selected_node_idx:
                marker.color.r = 0.1
                marker.color.g = 0.3
                marker.color.b = 1.0
                marker.color.a = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.6
                marker.color.b = 0.1
                marker.color.a = 0.9
            marker_array.markers.append(marker)
        self.markers_pub.publish(marker_array)

        edge_array = MarkerArray()
        eid = 0
        for i, j in self.prm_edges:
            n1 = self.prm_nodes[i]
            n2 = self.prm_nodes[j]
            edge_marker = Marker()
            edge_marker.header.frame_id = "map"
            edge_marker.ns = "prm_edges"
            edge_marker.id = eid
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD
            edge_marker.scale.x = 0.01
            edge_marker.color.r = 0.1
            edge_marker.color.g = 0.7
            edge_marker.color.b = 0.1
            edge_marker.color.a = 0.8
            edge_marker.points = [Point(x=n1[0], y=n1[1], z=0.1), Point(x=n2[0], y=n2[1], z=0.1)]
            edge_array.markers.append(edge_marker)
            eid += 1
        self.edges_pub.publish(edge_array)

def main(args=None):
    rclpy.init(args=args)
    node = PRMExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
