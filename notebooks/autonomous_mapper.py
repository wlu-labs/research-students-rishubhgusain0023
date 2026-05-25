#!/usr/bin/env python3
# coding=utf-8
"""
autonomous_mapper.py
--------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

Autonomous room mapping system.
Drives the robot around the room while RTAB-Map builds a 3D map.
Uses LiDAR for obstacle avoidance and Astra depth camera for 3D mapping.

Behaviour:
  1. Rotate 360 degrees to get initial room scan
  2. Drive forward until obstacle detected within SAFE_DISTANCE
  3. Stop, find clearest direction using LiDAR scan
  4. Turn toward clearest direction
  5. Repeat until EXPLORE_DURATION seconds elapsed or coverage target met
  6. Stop and save map

Run order (all terminals must be running before starting this):
  Terminal 1: ros2 launch yahboomcar_description display_X3.launch.py
  Terminal 2: ros2 run yahboomcar_bringup Mcnamu_driver_X3
  Terminal 3: ros2 run yahboomcar_base_node base_node_X3
  Terminal 4: ros2 launch astra_camera astro_pro_plus.launch.xml
  Terminal 5: ros2 launch ~/x3_rtabmap_depth.launch.py
  Terminal 6: ros2 launch ydlidar_ros2_driver ydlidar_launch.py
  Terminal 7: python3 autonomous_mapper.py

Topics:
  Subscribed : /scan          (YDLiDAR — obstacle avoidance)
               /odom_raw      (odometry — stuck detection)
               /odom          (odometry fallback)
               /map           (RTAB-Map occupancy grid — coverage tracking)
  Published  : /cmd_vel       (motor commands via Mcnamu_driver_X3)
               /mapper_status (String status updates)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
import numpy as np

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String


# ──────────────────────────── CONFIG ──────────────────────────
SAFE_DISTANCE       = 0.45   # metres — stop if obstacle closer than this
SIDE_SAFE_DISTANCE  = 0.30   # metres — side clearance check
FORWARD_SPEED       = 0.15   # metres/second — driving speed
TURN_SPEED          = 0.4    # radians/second — turning speed
EXPLORE_DURATION    = 300    # seconds — total mapping time (5 minutes)
INITIAL_SPIN_TIME   = 12.0   # seconds — time for initial 360 spin
MAP_COVERAGE_TARGET = 0.60   # stop early if 60% of map is explored
SCAN_WAIT_TIME      = 0.5    # seconds — wait after turning before driving
STUCK_THRESHOLD     = 3.0    # seconds — if no movement for this long, reverse
REVERSE_TIME        = 1.0    # seconds — reverse duration when stuck
# ──────────────────────────────────────────────────────────────


class MapperState:
    INIT       = "init"
    SPINNING   = "spinning"
    DRIVING    = "driving"
    OBSTACLE   = "obstacle"
    TURNING    = "turning"
    REVERSING  = "reversing"
    DONE       = "done"


class AutonomousMapperNode(Node):
    """
    Autonomous room mapping node.
    Combines LiDAR obstacle avoidance with RTAB-Map 3D mapping.
    """

    def __init__(self):
        super().__init__("autonomous_mapper")

        # ── State ─────────────────────────────────────────────
        self.state          = MapperState.INIT
        self.start_time     = time.time()
        self.state_start    = time.time()
        self.explored_ratio = 0.0
        self.last_pos       = None
        self.last_move_time = time.time()

        # FIX 1: initialise _turn_duration to avoid AttributeError
        self._turn_duration = 0.0

        # FIX 2: flag to ensure OBSTACLE state only triggers turn once
        self._obstacle_handled = False

        # ── Sensor cache ──────────────────────────────────────
        self.latest_scan = None
        self.latest_odom = None
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.robot_yaw   = 0.0

        # ── QoS ──────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 10,
            durability  = QoSDurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            LaserScan,     '/scan',     self._scan_cb, sensor_qos)
        self.create_subscription(
            Odometry,      '/odom_raw', self._odom_cb, sensor_qos)
        self.create_subscription(
            Odometry,      '/odom',     self._odom_cb, sensor_qos)
        self.create_subscription(
            OccupancyGrid, '/map',      self._map_cb,  10)

        # ── Publishers ────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',       10)
        self.status_pub = self.create_publisher(String, '/mapper_status', 10)

        # ── Main control loop at 10Hz ─────────────────────────
        self.create_timer(0.1, self._control_loop)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  Autonomous Mapper ready")
        self.get_logger().info(f"  Safe distance  : {SAFE_DISTANCE} m")
        self.get_logger().info(f"  Forward speed  : {FORWARD_SPEED} m/s")
        self.get_logger().info(f"  Explore time   : {EXPLORE_DURATION} s")
        self.get_logger().info(f"  Coverage target: {MAP_COVERAGE_TARGET*100:.0f}%")
        self.get_logger().info("  Waiting for LiDAR scan on /scan ...")
        self.get_logger().info("=" * 55)

    # ══════════════════════════════════════════════════════════
    #  Sensor callbacks
    # ══════════════════════════════════════════════════════════

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _map_cb(self, msg: OccupancyGrid):
        """Track explored ratio from RTAB-Map occupancy grid."""
        data  = np.array(msg.data)
        total = len(data)
        if total == 0:
            return
        known = np.sum(data >= 0)
        self.explored_ratio = float(known) / float(total)

    # ══════════════════════════════════════════════════════════
    #  LiDAR analysis helpers
    # ══════════════════════════════════════════════════════════

    def _get_range_at_angle(self, angle_deg, window_deg=10):
        """
        Get minimum range reading within a window around the given angle.
        angle_deg: 0 = forward, 90 = left, -90 = right, 180 = backward.
        Returns range in metres or None if no valid reading.
        """
        if self.latest_scan is None:
            return None

        scan       = self.latest_scan
        angle_rad  = math.radians(angle_deg)
        window_rad = math.radians(window_deg)
        lo = angle_rad - window_rad
        hi = angle_rad + window_rad

        valid = []
        angle = scan.angle_min
        for r in scan.ranges:
            if lo <= angle <= hi:
                if (scan.range_min < r < scan.range_max
                        and not math.isnan(r) and not math.isinf(r)):
                    valid.append(r)
            angle += scan.angle_increment

        return float(np.min(valid)) if valid else None

    def _get_forward_distance(self):
        """Return minimum distance directly ahead (±20 degrees)."""
        d = self._get_range_at_angle(0, window_deg=20)
        return d if d is not None else 999.0

    def _get_side_distances(self):
        """Return (left_dist, right_dist)."""
        left  = self._get_range_at_angle( 90, window_deg=15)
        right = self._get_range_at_angle(-90, window_deg=15)
        return (left  if left  is not None else 999.0,
                right if right is not None else 999.0)

    def _find_clearest_direction(self):
        """
        Scan all directions in 15 degree steps.
        Returns angle in degrees relative to robot forward (-180 to 180)
        pointing toward the most open space.
        """
        if self.latest_scan is None:
            return 90.0  # default: turn left

        best_angle = 90.0
        best_dist  = 0.0

        for angle_deg in range(-180, 181, 15):
            d = self._get_range_at_angle(angle_deg, window_deg=10)
            if d is None:
                d = 999.0
            if d > best_dist:
                best_dist  = d
                best_angle = float(angle_deg)

        self.get_logger().info(
            f"[MAPPER] Clearest direction: {best_angle:.0f} deg "
            f"({best_dist:.2f} m)")
        return best_angle

    def _is_obstacle_ahead(self):
        """Return True if obstacle within SAFE_DISTANCE ahead."""
        return self._get_forward_distance() < SAFE_DISTANCE

    def _is_stuck(self):
        """Return True if robot hasn't moved in STUCK_THRESHOLD seconds."""
        if self.latest_odom is None:
            return False
        pos = (self.robot_x, self.robot_y)
        if self.last_pos is None:
            self.last_pos = pos
            return False
        dist = math.hypot(pos[0] - self.last_pos[0],
                          pos[1] - self.last_pos[1])
        if dist > 0.05:
            self.last_pos       = pos
            self.last_move_time = time.time()
            return False
        return (time.time() - self.last_move_time) > STUCK_THRESHOLD

    # ══════════════════════════════════════════════════════════
    #  Motion helpers
    # ══════════════════════════════════════════════════════════

    def _drive_forward(self):
        twist = Twist()
        twist.linear.x  = FORWARD_SPEED
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _turn(self, angular_z):
        twist = Twist()
        twist.linear.x  = 0.0
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)

    def _reverse(self):
        twist = Twist()
        twist.linear.x  = -FORWARD_SPEED
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    # ══════════════════════════════════════════════════════════
    #  Main control loop — 10Hz
    # ══════════════════════════════════════════════════════════

    def _control_loop(self):
        elapsed = time.time() - self.start_time

        # ── Check termination conditions ──────────────────────
        if self.state != MapperState.DONE:
            if elapsed > EXPLORE_DURATION:
                self._transition(MapperState.DONE, "Time limit reached")
                return
            if self.explored_ratio >= MAP_COVERAGE_TARGET:
                self._transition(
                    MapperState.DONE,
                    f"Coverage target reached: {self.explored_ratio*100:.1f}%")
                return

        # ── INIT — wait for first LiDAR scan ─────────────────
        if self.state == MapperState.INIT:
            if self.latest_scan is not None:
                self._transition(MapperState.SPINNING,
                                 "LiDAR ready — starting initial 360 scan")
            return

        # ── SPINNING — initial 360 to build first map frame ───
        if self.state == MapperState.SPINNING:
            self._turn(TURN_SPEED)
            if time.time() - self.state_start > INITIAL_SPIN_TIME:
                self._stop()
                time.sleep(0.5)
                self._transition(MapperState.DRIVING,
                                 "Initial scan complete — starting exploration")
            return

        # ── DRIVING — move forward until obstacle ─────────────
        if self.state == MapperState.DRIVING:
            if self._is_stuck():
                self._transition(MapperState.REVERSING,
                                 "Stuck detected — reversing")
                return
            if self._is_obstacle_ahead():
                self._stop()
                self._obstacle_handled = False   # reset for OBSTACLE state
                self._transition(
                    MapperState.OBSTACLE,
                    f"Obstacle at {self._get_forward_distance():.2f} m")
                return
            self._drive_forward()

            if self._scan_count_log(elapsed):
                self.get_logger().info(
                    f"[MAPPER] Elapsed: {elapsed:.0f}s | "
                    f"Coverage: {self.explored_ratio*100:.1f}% | "
                    f"Forward clear: {self._get_forward_distance():.2f}m")
            return

        # ── OBSTACLE — find clearest direction, trigger turn ──
        # FIX 2: only compute and start the turn ONCE per obstacle event
        if self.state == MapperState.OBSTACLE:
            if not self._obstacle_handled:
                self._obstacle_handled = True
                best_angle = self._find_clearest_direction()

                # Turn direction based on which side is clearer
                turn_dir = TURN_SPEED if best_angle >= 0 else -TURN_SPEED

                # Turn duration = angle / angular_speed (in radians)
                turn_duration = abs(math.radians(best_angle)) / TURN_SPEED
                # Clamp to reasonable range
                turn_duration = max(0.5, min(turn_duration, 4.0))

                self._turn_for(turn_dir, turn_duration)
            return

        # ── TURNING — execute timed turn ──────────────────────
        if self.state == MapperState.TURNING:
            self._turn(self._current_turn_dir)
            if time.time() - self.state_start > self._turn_duration:
                self._stop()
                time.sleep(SCAN_WAIT_TIME)
                self._transition(MapperState.DRIVING,
                                 "Turn complete — resuming drive")
            return

        # ── REVERSING — back up when stuck ────────────────────
        if self.state == MapperState.REVERSING:
            self._reverse()
            if time.time() - self.state_start > REVERSE_TIME:
                self._stop()
                # Turn left 90 degrees to find new direction
                self._turn_for(TURN_SPEED, math.pi / 2.0 / TURN_SPEED)
            return

        # ── DONE — stop robot and save map ────────────────────
        if self.state == MapperState.DONE:
            self._stop()
            self._publish_status("Mapping complete!")
            self.get_logger().info(
                f"\n{'='*55}\n"
                f"  MAPPING COMPLETE\n"
                f"  Total time : {elapsed:.1f}s\n"
                f"  Coverage   : {self.explored_ratio*100:.1f}%\n"
                f"{'='*55}"
            )
            self._save_map()
            rclpy.shutdown()

    # ══════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════

    def _turn_for(self, angular_z, duration):
        """
        Start a timed turn.
        FIX 1: both _turn_duration and _current_turn_dir are always
        set here before TURNING state uses them.
        """
        self._turn_duration    = duration
        self._current_turn_dir = angular_z
        self._transition(
            MapperState.TURNING,
            f"Turning {'left' if angular_z > 0 else 'right'} "
            f"for {duration:.1f}s")

    def _transition(self, new_state, reason=""):
        self.state       = new_state
        self.state_start = time.time()
        msg = f"[MAPPER] {new_state.upper()} — {reason}"
        self.get_logger().info(msg)
        self._publish_status(msg)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _scan_count_log(self, elapsed) -> bool:
        """Return True every 10 seconds for progress logging."""
        return int(elapsed) % 10 == 0 and int(elapsed) > 0

    def _save_map(self):
        """
        FIX 3: removed unused subprocess import.
        RTAB-Map auto-saves to ~/.ros/rtabmap.db when running.
        Instructions for exporting as 2D occupancy grid are logged.
        """
        self.get_logger().info(
            "Map auto-saved by RTAB-Map to ~/.ros/rtabmap.db")
        self.get_logger().info(
            "To export as a 2D PNG + YAML map run:")
        self.get_logger().info(
            "  ros2 run nav2_map_server map_saver_cli "
            "-f ~/map --ros-args -r map:=/map")
        self.get_logger().info(
            "To view the 3D map relaunch RTAB-Map — "
            "it will load the saved database automatically.")

    def destroy_node(self):
        self._stop()
        super().destroy_node()


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Mapping interrupted. "
            f"Coverage: {node.explored_ratio*100:.1f}%")
        node._stop()
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()