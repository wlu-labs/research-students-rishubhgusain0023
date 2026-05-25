#!/usr/bin/env python3
# coding=utf-8
"""
navgpt_planner.py
-----------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

NavGPT-style high-level navigation guidance.

Based on:
  Zhou et al., "NavGPT: Explicit Reasoning in Vision-and-Language Navigation
  with Large Language Models", AAAI 2024.
  https://arxiv.org/abs/2305.16986

Core idea (adapted to this hardware):
  At each decision step, build a textual prompt containing:
    (1) the user's natural-language instruction,
    (2) a textual scene description (from YOLOv8 + arm-camera sweep),
    (3) the explorable directions (extracted from LiDAR free space),
    (4) navigation history so far.
  Send that prompt to GPT-4 and parse out:
    - LLM Thought : reasoning trace
    - LLM Action  : one of {move_to(x, y), explore(direction), stop}
  Publish the chosen action as a /goal Point that main.py's planner
  (RRT* / A*) executes on the real robot.

Wiring:
  navgpt_planner.py subscribes to:
     /task      (std_msgs/String)   — natural-language instruction
     /scan      (LaserScan)         — LiDAR for free-direction extraction
     /odom      (Odometry)          — current robot pose
     /yolo_obs  (std_msgs/String)   — JSON list of detections (label,bearing,dist)
  navgpt_planner.py publishes to:
     /goal           (geometry_msgs/Point)
     /navgpt_status  (std_msgs/String)
     /navgpt_thought (std_msgs/String)

Run order:
  Terminal 1: source robot bringup + main.py
  Terminal 2: python3 obstacle_avoider.py
  Terminal 3: export OPENAI_API_KEY="sk-..."
              python3 navgpt_planner.py
  Terminal 4: ros2 topic pub /task std_msgs/String "{data: 'find the chair near the wall'}" --once

Requires:
  pip install openai
"""

import os
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import String

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ───────────────────────── CONFIG ─────────────────────────────
LLM_MODEL          = "gpt-4o"     # gpt-4 / gpt-4o / gpt-4o-mini
MAX_HISTORY        = 6            # navigation history length (steps)
N_DIRECTIONS       = 8            # how many discrete bearings to report
FREE_DIST_MIN      = 1.0          # metres — direction must be at least this clear
STEP_DIST_DEFAULT  = 1.5          # metres — default move distance when LLM says "go N"

STATUS_TOPIC       = '/navgpt_status'
THOUGHT_TOPIC      = '/navgpt_thought'
GOAL_TOPIC         = '/goal'
TASK_TOPIC         = '/task'
YOLO_OBS_TOPIC     = '/yolo_obs'
# ──────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are NavGPT, an embodied navigation agent inside a small wheeled robot in an indoor environment. The robot has a LiDAR and a forward camera with YOLOv8 detection.

At every step you receive:
  - the user's instruction
  - a list of objects currently visible (label, bearing in degrees relative to robot front, estimated distance in metres)
  - the explorable directions: bearings in degrees where LiDAR is clear for at least 1m
  - the navigation history (your past thoughts and actions)
  - the robot's current world pose (x, y, yaw)

You MUST reason briefly, then choose exactly ONE action and respond ONLY in the following JSON format:

{
  "thought": "<one to three sentences of reasoning>",
  "action": "<one of: move_to, explore, stop>",
  "args":   { ... action-specific arguments ... }
}

Action specs:
  move_to: { "x": <world_x>, "y": <world_y> }              # navigate to a world coordinate
  explore: { "bearing_deg": <-180..180>, "dist_m": <m> }    # move along a free direction
  stop   : {}                                               # task complete or impossible

Rules:
  - Never invent objects not in the visible list.
  - Prefer "move_to" when a target object is visible (use its distance + bearing to compute x,y).
  - Prefer "explore" when the target is not yet visible — pick the most promising free bearing.
  - Use "stop" only when the task is done OR clearly impossible.
  - Respond with JSON ONLY. No prose outside the JSON.
"""


class NavGPTPlannerNode(Node):

    def __init__(self):
        super().__init__("navgpt_planner")

        # LLM client
        self.client = None
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI()
            self.get_logger().info(f"OpenAI client ready (model={LLM_MODEL})")
        else:
            self.get_logger().warn(
                "OpenAI unavailable — set OPENAI_API_KEY and `pip install openai`. "
                "Running in DRY-RUN mode (no LLM calls).")

        # State
        self.instruction      = None
        self.latest_scan      = None
        self.robot_x          = 0.0
        self.robot_y          = 0.0
        self.robot_yaw        = 0.0
        self.visible_objects  = []   # list of {"label","bearing_deg","dist_m"}
        self.history          = []   # list of {"step","thought","action","args"}
        self.step_idx         = 0
        self.task_active      = False

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subs
        self.create_subscription(String,    TASK_TOPIC,     self._task_cb,  10)
        self.create_subscription(LaserScan, '/scan',        self._scan_cb,  sensor_qos)
        self.create_subscription(Odometry,  '/odom',        self._odom_cb,  sensor_qos)
        self.create_subscription(Odometry,  '/odom_raw',    self._odom_cb,  sensor_qos)
        self.create_subscription(String,    YOLO_OBS_TOPIC, self._yolo_cb,  10)

        # Pubs
        self.goal_pub    = self.create_publisher(Point,  GOAL_TOPIC,    10)
        self.status_pub  = self.create_publisher(String, STATUS_TOPIC,  10)
        self.thought_pub = self.create_publisher(String, THOUGHT_TOPIC, 10)

        # Decision timer — slow, LLM is expensive
        self.create_timer(5.0, self._maybe_step)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  NavGPTPlannerNode ready")
        self.get_logger().info(f"  Model        : {LLM_MODEL}")
        self.get_logger().info(f"  Step period  : 5.0s")
        self.get_logger().info(f"  Free-dir min : {FREE_DIST_MIN}m")
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            "Send instruction: ros2 topic pub /task std_msgs/String "
            "\"{data: 'find a chair'}\" --once")

    # ══════════════════════════════════════════════════════════
    #  Callbacks
    # ══════════════════════════════════════════════════════════

    def _task_cb(self, msg: String):
        self.instruction = msg.data.strip()
        self.history     = []
        self.step_idx    = 0
        self.task_active = True
        self.get_logger().info(f"[NAVGPT] New task: '{self.instruction}'")
        self._publish_status(f"Task started: {self.instruction}")

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.robot_x = p.x
        self.robot_y = p.y
        self.robot_yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z)
        )

    def _yolo_cb(self, msg: String):
        """
        Expects JSON list:
        [{"label": "chair", "bearing_deg": -10.0, "dist_m": 1.8}, ...]
        """
        try:
            self.visible_objects = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f"[NAVGPT] Bad YOLO obs JSON: {e}")
            self.visible_objects = []

    # ══════════════════════════════════════════════════════════
    #  Main decision step
    # ══════════════════════════════════════════════════════════

    def _maybe_step(self):
        if not self.task_active or self.instruction is None:
            return
        if self.latest_scan is None:
            self.get_logger().warn("[NAVGPT] Waiting for /scan...")
            return

        self.step_idx += 1

        # Build context
        free_dirs = self._extract_free_directions(self.latest_scan)
        prompt    = self._build_user_prompt(free_dirs)

        self.get_logger().info(
            f"[NAVGPT] Step {self.step_idx} — calling LLM "
            f"(visible={len(self.visible_objects)}, free_dirs={len(free_dirs)})")

        # Query LLM
        decision = self._query_llm(prompt)
        if decision is None:
            self.get_logger().warn("[NAVGPT] LLM returned no usable decision.")
            return

        thought = decision.get("thought", "")
        action  = decision.get("action", "stop")
        args    = decision.get("args", {})

        self.get_logger().info(f"[NAVGPT] Thought: {thought}")
        self.get_logger().info(f"[NAVGPT] Action : {action} {args}")
        self._publish_thought(thought)

        # Record in history
        self.history.append({
            "step":    self.step_idx,
            "thought": thought,
            "action":  action,
            "args":    args,
        })
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # Execute
        if action == "stop":
            self.task_active = False
            self._publish_status(f"Task ended: {thought[:80]}")
            self.get_logger().info("[NAVGPT] Task complete (stop).")
            return

        if action == "move_to":
            self._publish_goal(args.get("x"), args.get("y"))
        elif action == "explore":
            bearing = args.get("bearing_deg", 0.0)
            dist    = args.get("dist_m", STEP_DIST_DEFAULT)
            self._explore_bearing(bearing, dist)
        else:
            self.get_logger().warn(f"[NAVGPT] Unknown action: {action}")

    # ══════════════════════════════════════════════════════════
    #  Prompt construction
    # ══════════════════════════════════════════════════════════

    def _build_user_prompt(self, free_dirs):
        # Objects block
        if self.visible_objects:
            obj_lines = []
            for o in self.visible_objects:
                obj_lines.append(
                    f"  - {o.get('label','?')}: bearing={o.get('bearing_deg',0):.0f}°, "
                    f"dist={o.get('dist_m',0):.2f}m")
            obj_block = "\n".join(obj_lines)
        else:
            obj_block = "  (none visible)"

        # Free-direction block
        if free_dirs:
            dir_lines = [f"  - bearing={d['bearing_deg']:.0f}°, clear={d['dist_m']:.2f}m"
                         for d in free_dirs]
            dir_block = "\n".join(dir_lines)
        else:
            dir_block = "  (none — robot is boxed in)"

        # History block
        if self.history:
            hist_lines = []
            for h in self.history[-MAX_HISTORY:]:
                hist_lines.append(
                    f"  Step {h['step']}: {h['action']}({h['args']}) — {h['thought'][:80]}")
            hist_block = "\n".join(hist_lines)
        else:
            hist_block = "  (no prior steps)"

        return f"""User instruction:
  {self.instruction}

Current robot pose:
  x={self.robot_x:.2f}, y={self.robot_y:.2f}, yaw={math.degrees(self.robot_yaw):.0f}°

Visible objects (from YOLOv8):
{obj_block}

Explorable directions (LiDAR free space):
{dir_block}

Navigation history:
{hist_block}

Choose your next action. Respond with JSON only.
"""

    # ══════════════════════════════════════════════════════════
    #  LLM call
    # ══════════════════════════════════════════════════════════

    def _query_llm(self, user_prompt):
        if self.client is None:
            # Dry-run stub — pick the most-clear bearing
            self.get_logger().info("[NAVGPT] DRY-RUN — no API key, returning stub action.")
            return {
                "thought": "(dry-run stub) no LLM call",
                "action":  "stop",
                "args":    {},
            }

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            return json.loads(text)
        except Exception as e:
            self.get_logger().error(f"[NAVGPT] LLM call failed: {e}")
            return None

    # ══════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════

    def _extract_free_directions(self, scan: LaserScan):
        """
        Bin LiDAR ranges into N_DIRECTIONS sectors, return list of
        {bearing_deg, dist_m} where the minimum range exceeds FREE_DIST_MIN.
        Bearings are in robot frame: 0° = front, +90° = left, -90° = right.
        """
        n = N_DIRECTIONS
        sector_min = [float('inf')] * n
        sector_width = 2 * math.pi / n

        angle = scan.angle_min
        for r in scan.ranges:
            angle_now = angle
            angle    += scan.angle_increment
            if (r < scan.range_min or r > scan.range_max
                    or math.isnan(r) or math.isinf(r)):
                continue
            # Normalise to [0, 2π)
            a = angle_now
            while a < 0:        a += 2 * math.pi
            while a >= 2*math.pi: a -= 2 * math.pi
            idx = int(a / sector_width) % n
            if r < sector_min[idx]:
                sector_min[idx] = r

        free = []
        for i, d in enumerate(sector_min):
            if d >= FREE_DIST_MIN and d != float('inf'):
                center = i * sector_width + sector_width / 2.0  # robot-frame [0,2π)
                # Convert to [-180, 180]
                bearing_deg = math.degrees(center)
                if bearing_deg > 180:
                    bearing_deg -= 360
                free.append({"bearing_deg": bearing_deg, "dist_m": d})
        # Sort by clearance descending
        free.sort(key=lambda x: -x["dist_m"])
        return free

    def _publish_goal(self, x, y):
        if x is None or y is None:
            self.get_logger().warn("[NAVGPT] move_to missing x/y")
            return
        msg = Point()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = 0.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f"[NAVGPT] → /goal ({x:.2f}, {y:.2f})")

    def _explore_bearing(self, bearing_deg, dist_m):
        """Convert a robot-frame bearing into a world-frame /goal point."""
        bearing_world = math.radians(bearing_deg) + self.robot_yaw
        x = self.robot_x + dist_m * math.cos(bearing_world)
        y = self.robot_y + dist_m * math.sin(bearing_world)
        self._publish_goal(x, y)

    def _publish_status(self, text):
        msg = String(); msg.data = text
        self.status_pub.publish(msg)

    def _publish_thought(self, text):
        msg = String(); msg.data = text
        self.thought_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NavGPTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()