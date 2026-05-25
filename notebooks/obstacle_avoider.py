#!/usr/bin/env python3
# coding=utf-8
"""
obstacle_avoider.py
-------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

Reactive obstacle-avoidance safety layer.

Runs continuously alongside the main planner (main.py).
Monitors /scan in real time. If an obstacle is detected within
the danger zone in front of the robot, it INTERCEPTS the
/cmd_vel stream and replaces it with a safe avoidance command:
  - SLOW : within WARN_DIST  → reduce forward speed
  - VEER : within VEER_DIST  → steer toward the freer side
  - STOP : within STOP_DIST  → hard stop, rotate in place

It does NOT replace the global planner — it sits on top of it.
The planner can keep publishing /cmd_vel_intent (or any topic),
this node consumes that and republishes a safe version on /cmd_vel.

Wiring:
  main.py / virtual_mover.py publishes /cmd_vel_intent
  obstacle_avoider.py consumes /cmd_vel_intent and /scan
  obstacle_avoider.py publishes /cmd_vel  (the real motor command)
  Mcnamu_driver_X3 subscribes to /cmd_vel

If you don't want to retopic, just change INPUT_CMD_TOPIC to
'/virtual_cmd_vel' and OUTPUT_CMD_TOPIC to '/cmd_vel'.

Run:
  python3 obstacle_avoider.py
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String


# ───────────────────────── CONFIG ─────────────────────────────
# Distance bands (metres)
STOP_DIST   = 0.30   # closer than this → hard stop
VEER_DIST   = 0.60   # closer than this → steer aside
WARN_DIST   = 1.00   # closer than this → reduce speed

# Front cone (degrees, ± from straight ahead)
FRONT_CONE_DEG = 40.0

# Side cones used to decide which direction to veer
SIDE_CONE_DEG  = 60.0   # how wide to look left vs right

# Speed limits
MAX_LINEAR  = 0.30
MIN_LINEAR  = 0.05
VEER_ANG    = 0.6    # rad/s when veering
ROTATE_ANG  = 0.8    # rad/s when stopped-and-rotating

# Topics
INPUT_CMD_TOPIC  = '/cmd_vel_intent'   # from planner
OUTPUT_CMD_TOPIC = '/cmd_vel'          # to motor driver
STATUS_TOPIC     = '/avoider_status'

# Loop rate
PUBLISH_HZ = 20.0
# ──────────────────────────────────────────────────────────────


class ObstacleAvoiderNode(Node):

    def __init__(self):
        super().__init__("obstacle_avoider")

        # Latest sensor / command state
        self._latest_scan  = None
        self._latest_intent = Twist()   # default: zero
        self._intent_age   = 0.0
        self._last_intent_time = self.get_clock().now()

        # State machine
        self.state = "CLEAR"   # CLEAR / SLOW / VEER / STOP

        # QoS — LiDAR is BEST_EFFORT
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subs
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(
            Twist, INPUT_CMD_TOPIC, self._intent_cb, 10)

        # Pubs
        self.cmd_pub    = self.create_publisher(Twist,  OUTPUT_CMD_TOPIC, 10)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC,     10)

        # Control loop
        self.create_timer(1.0 / PUBLISH_HZ, self._loop)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  ObstacleAvoiderNode ready")
        self.get_logger().info(f"  Input  : {INPUT_CMD_TOPIC}")
        self.get_logger().info(f"  Output : {OUTPUT_CMD_TOPIC}")
        self.get_logger().info(
            f"  Bands  : stop<{STOP_DIST}m  veer<{VEER_DIST}m  warn<{WARN_DIST}m")
        self.get_logger().info("=" * 50)

    # ── callbacks ──────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        self._latest_scan = msg

    def _intent_cb(self, msg: Twist):
        self._latest_intent     = msg
        self._last_intent_time  = self.get_clock().now()

    # ── core loop ──────────────────────────────────────────────

    def _loop(self):
        if self._latest_scan is None:
            # No scan yet — pass intent straight through
            self.cmd_pub.publish(self._latest_intent)
            return

        # Compute distances in front cone + side cones
        front_min, left_min, right_min = self._scan_distances(self._latest_scan)

        # Decide state based on closest front-cone distance
        if front_min < STOP_DIST:
            new_state = "STOP"
        elif front_min < VEER_DIST:
            new_state = "VEER"
        elif front_min < WARN_DIST:
            new_state = "SLOW"
        else:
            new_state = "CLEAR"

        if new_state != self.state:
            self.get_logger().info(
                f"[AVOIDER] {self.state} → {new_state}  "
                f"(front={front_min:.2f}m  L={left_min:.2f}m  R={right_min:.2f}m)")
            self._publish_status(new_state, front_min)
            self.state = new_state

        # Build output command based on state
        out = Twist()
        intent = self._latest_intent

        if self.state == "CLEAR":
            # Pass intent through unchanged
            out.linear.x  = intent.linear.x
            out.angular.z = intent.angular.z

        elif self.state == "SLOW":
            # Reduce forward speed, keep heading
            scale = (front_min - VEER_DIST) / max(WARN_DIST - VEER_DIST, 1e-6)
            scale = max(0.3, min(1.0, scale))
            out.linear.x  = max(MIN_LINEAR, intent.linear.x * scale)
            out.angular.z = intent.angular.z

        elif self.state == "VEER":
            # Slow + steer toward freer side
            out.linear.x = max(MIN_LINEAR, intent.linear.x * 0.4)
            if left_min > right_min:
                out.angular.z = +VEER_ANG   # turn left
            else:
                out.angular.z = -VEER_ANG   # turn right

        elif self.state == "STOP":
            # Hard stop forward, rotate to find an opening
            out.linear.x = 0.0
            if left_min > right_min:
                out.angular.z = +ROTATE_ANG
            else:
                out.angular.z = -ROTATE_ANG

        # Clamp
        out.linear.x  = max(-MAX_LINEAR, min(MAX_LINEAR, out.linear.x))
        self.cmd_pub.publish(out)

    # ── helpers ────────────────────────────────────────────────

    def _scan_distances(self, scan: LaserScan):
        """
        Returns (front_min, left_min, right_min) in metres.
        front_min:  closest range in ± FRONT_CONE_DEG  about 0°
        left_min:   closest range in +SIDE_CONE_DEG / 2 sector
        right_min:  closest range in -SIDE_CONE_DEG / 2 sector
        """
        front_lim_rad = math.radians(FRONT_CONE_DEG)
        side_lim_rad  = math.radians(SIDE_CONE_DEG)

        front_min = float('inf')
        left_min  = float('inf')
        right_min = float('inf')

        angle = scan.angle_min
        for r in scan.ranges:
            angle_now = angle
            angle    += scan.angle_increment

            if (r < scan.range_min or r > scan.range_max
                    or math.isnan(r) or math.isinf(r)):
                continue

            # Normalise angle to (-pi, pi]
            a = angle_now
            while a >  math.pi: a -= 2 * math.pi
            while a < -math.pi: a += 2 * math.pi

            if abs(a) <= front_lim_rad:
                if r < front_min:
                    front_min = r

            if 0 < a <= side_lim_rad:
                if r < left_min:
                    left_min = r
            elif -side_lim_rad <= a < 0:
                if r < right_min:
                    right_min = r

        # If a sector saw nothing valid, treat as wide open
        if front_min == float('inf'): front_min = 99.0
        if left_min  == float('inf'): left_min  = 99.0
        if right_min == float('inf'): right_min = 99.0

        return front_min, left_min, right_min

    def _publish_status(self, state: str, dist: float):
        msg = String()
        msg.data = f"{state} front_dist={dist:.2f}m"
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoiderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Safety: publish zero stop on shutdown
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()