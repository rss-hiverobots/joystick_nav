#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
import random
import math
import networkx as nx

class PRMExplorer(Node):
    def __init__(self):
        super().__init__('prm_explorer')

        self.N = 100
        self.resample_interval = 20.0  # seconds
        self.last_resample_time = self.get_clock().now().seconds_nanoseconds()[0]

        self.map = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None

        self.prm_nodes = []
        self.prm_edges = []
        self.selected_node_idx = None

        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.map_callback, 1)
        self.markers_pub = self.create_publisher(MarkerArray, '/prm_markers', 1)
        self.edges_pub = self.create_publisher(MarkerArray, '/prm_edges', 1)

        self.timer = self.create_timer(1.0, self.main_loop)

    def map_callback(self, msg: OccupancyGrid):
        self.map = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_width = msg.info.width
        self.map_height = msg.info.height

    def main_loop(self):
        now = self.get_clock().now().seconds_nanoseconds()[0]
        if self.map is None:
            return

        if now - self.last_resample_time >= self.resample_interval:
            self.sample_nodes()
            self.build_prm()
            self.select_best_node()
            self.publish_markers()
            self.last_resample_time = now

    def sample_nodes(self):
        self.prm_nodes = []
        free_indices = np.argwhere(self.map == 0)
        if len(free_indices) == 0:
            self.get_logger().warn('No free space in map!')
            return

        sampled_indices = free_indices[np.random.choice(len(free_indices), min(self.N, len(free_indices)), replace=False)]
        for idx in sampled_indices:
            y, x = idx
            wx = self.map_origin[0] + x * self.map_resolution
            wy = self.map_origin[1] + y * self.map_resolution
            self.prm_nodes.append((wx, wy))

    def build_prm(self):
        # Create PRM graph
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
                dist = math.hypot(n1[0]-n2[0], n1[1]-n2[1])
                if dist < R and self.is_path_free(n1, n2):
                    self.prm_graph.add_edge(i, j)
                    self.prm_edges.append((i, j))

    def is_path_free(self, n1, n2):
        # Bresenham's line for collision checking
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
        # Bresenham's line algorithm
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
            return

        # Find the largest connected component in the PRM
        components = list(nx.connected_components(self.prm_graph))
        if not components:
            self.selected_node_idx = None
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
            if score > best_score:
                best_score = score
                best_idx = i
        self.selected_node_idx = best_idx


    def publish_markers(self):
        if not self.prm_nodes:
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
