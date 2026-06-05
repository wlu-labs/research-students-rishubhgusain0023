#!/usr/bin/env python3
# coding=utf-8
"""
arm_grab_place.py
-----------------
Grab and place sequence with IK + preset fallback.
Run: python3 arm_grab_place.py --dry-run --no-drive
"""
import argparse
import math
import time
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

try:
    from Rosmaster_Lib import Rosmaster
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False

# ───────────────────── CONFIG ─────────────────────────────────
ARM_BASE_HEIGHT = 0.05
L1              = 0.105
L2              = 0.098
L3              = 0.095
SERIAL_PORT     = '/dev/ttyUSB0'
MOVE_DELAY      = 1.5
GRIP_DELAY      = 1.0
GRIP_OPEN       = 90
GRIP_CLOSE      = 30
PICK_XYZ        = (0.18, 0.00, 0.05)
PLACE_XYZ       = (0.18, 0.00, 0.05)
HOVER_DZ        = 0.08
POSE_HOME       = [90, 135, 90, 90, 90, GRIP_OPEN]
POSE_TRAVEL     = [90, 150, 30, 90, 90, GRIP_CLOSE]
POSE_PRE_GRAB   = [90, 80, 60, 100, 90, GRIP_OPEN]
POSE_GRAB       = [90, 55, 50, 110, 90, GRIP_OPEN]
POSE_PRE_PLACE  = [90, 80, 60, 100, 90, GRIP_CLOSE]
POSE_PLACE      = [90, 55, 50, 110, 90, GRIP_CLOSE]
BASE_A_XY       = (0.0, 0.0)
BASE_B_XY       = (1.5, 0.0)
BASE_ARRIVAL_TOL     = 0.35
BASE_ARRIVAL_TIMEOUT = 60.0
# ──────────────────────────────────────────────────────────────


def simple_ik(x, y, z):
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return None
    yaw_rad = math.atan2(y, x)
    s1 = 90.0 + math.degrees(yaw_rad)
    if not (0 <= s1 <= 180):
        return None
    r  = math.hypot(x, y)
    h  = z - ARM_BASE_HEIGHT
    rw = r - L3
    if rw < 0:
        return None
    d2 = rw*rw + h*h
    d  = math.sqrt(d2)
    if d > (L1+L2) or d < abs(L1-L2):
        return None
    cos_elbow = (L1*L1 + L2*L2 - d2) / (2*L1*L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_int = math.acos(cos_elbow)
    angle_to_target = math.atan2(h, rw)
    cos_off = (L1*L1 + d2 - L2*L2) / (2*L1*d)
    cos_off = max(-1.0, min(1.0, cos_off))
    shoulder_off = math.acos(cos_off)
    shoulder_rad = angle_to_target + shoulder_off
    elbow_rad    = math.pi - elbow_int
    s2 = 90.0 - math.degrees(shoulder_rad)
    s3 = 90.0 - math.degrees(elbow_rad)
    s4 = 90.0 + math.degrees(shoulder_rad - elbow_rad)
    s5 = 90.0
    for ang in (s2, s3, s4, s5):
        if not (0 <= ang <= 180):
            return None
    return [s1, s2, s3, s4, s5]


def ik_or_preset(xyz, preset, grip_angle):
    sol = simple_ik(*xyz)
    if sol is None:
        print(f"  [IK] Out of reach — using preset fallback")
        return list(preset)
    print(f"  [IK] Solution found: {[round(a,1) for a in sol]}")
    return sol + [grip_angle]


class ArmController:
    def __init__(self, dry_run=False, logger=None):
        self.dry_run = dry_run
        self.logger  = logger
        self.bot     = None
        if dry_run or not ARM_AVAILABLE:
            self._log("DRY-RUN mode — no hardware commands sent")
            return
        try:
            self.bot = Rosmaster(com=SERIAL_PORT)
            self.bot.create_receive_threading()
            time.sleep(0.5)
            self.bot.set_uart_servo_torque(True)
        except Exception as e:
            self._log(f"Arm init failed: {e} — falling back to DRY-RUN")
            self.bot = None

    def _log(self, msg):
        if self.logger:
            self.logger.info(f"[ARM] {msg}")
        else:
            print(f"[ARM] {msg}")

    def set_pose(self, angles, label=""):
        ang_int = [int(round(a)) for a in angles]
        self._log(f"{label} → {ang_int}")
        if self.bot:
            self.bot.set_uart_servo_angle_array(ang_int)
        time.sleep(MOVE_DELAY)

    def grip(self, angle, label=""):
        self._log(f"GRIPPER {label} → {angle}")
        if self.bot:
            self.bot.set_uart_servo_angle(6, int(angle))
        time.sleep(GRIP_DELAY)

    def shutdown(self):
        if self.bot:
            try:
                self.set_pose(POSE_HOME, "HOME (shutdown)")
                self.bot.set_uart_servo_torque(False)
                del self.bot
            except Exception:
                pass


class BaseMover(Node):
    def __init__(self):
        super().__init__("arm_grab_place_base_mover")
        self.goal_pub = self.create_publisher(Point, '/goal', 10)
        self.create_subscription(Odometry, '/odom',     self._odom_cb, 10)
        self.create_subscription(Odometry, '/odom_raw', self._odom_cb, 10)
        self.x = 0.0
        self.y = 0.0

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    def drive_to(self, target_xy, timeout=BASE_ARRIVAL_TIMEOUT):
        msg = Point()
        msg.x, msg.y = float(target_xy[0]), float(target_xy[1])
        msg.z = 0.0
        self.goal_pub.publish(msg)
        self.get_logger().info(f"[BASE] /goal → ({msg.x:.2f}, {msg.y:.2f})")
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if math.hypot(target_xy[0]-self.x, target_xy[1]-self.y) < BASE_ARRIVAL_TOL:
                self.get_logger().info(f"[BASE] Arrived at ({self.x:.2f}, {self.y:.2f})")
                return True
        self.get_logger().warn(f"[BASE] Timeout after {timeout:.0f}s")
        return False


def run_sequence(arm, base, do_drive, logger):
    def log(msg):
        if logger: logger.info(msg)
        else: print(msg)

    log("=" * 55)
    log("  ARM GRAB-AND-PLACE SEQUENCE")
    log("=" * 55)

    log("\n[1] HOME pose")
    arm.set_pose(POSE_HOME, "HOME")

    log("[2] Open gripper")
    arm.grip(GRIP_OPEN, "OPEN")

    log(f"[3] PRE_GRAB above pick {PICK_XYZ}")
    px, py, pz = PICK_XYZ
    pre_grab = ik_or_preset((px, py, pz+HOVER_DZ), POSE_PRE_GRAB, GRIP_OPEN)
    arm.set_pose(pre_grab, "PRE_GRAB")

    log(f"[4] GRAB at pick {PICK_XYZ}")
    grab = ik_or_preset(PICK_XYZ, POSE_GRAB, GRIP_OPEN)
    arm.set_pose(grab, "GRAB")

    log("[5] Close gripper")
    arm.grip(GRIP_CLOSE, "CLOSE")

    log("[6] Lift back to PRE_GRAB")
    pre_grab_held = list(pre_grab); pre_grab_held[-1] = GRIP_CLOSE
    arm.set_pose(pre_grab_held, "LIFT")

    log("[7] TRAVEL pose")
    arm.set_pose(POSE_TRAVEL, "TRAVEL")

    if do_drive and base is not None:
        log(f"[8] Driving base {BASE_A_XY} → {BASE_B_XY}")
        base.drive_to(BASE_B_XY)
    else:
        log("[8] Base drive SKIPPED (--no-drive)")

    log(f"[10] PRE_PLACE above place {PLACE_XYZ}")
    qx, qy, qz = PLACE_XYZ
    pre_place = ik_or_preset((qx, qy, qz+HOVER_DZ), POSE_PRE_PLACE, GRIP_CLOSE)
    arm.set_pose(pre_place, "PRE_PLACE")

    log(f"[11] PLACE at {PLACE_XYZ}")
    place = ik_or_preset(PLACE_XYZ, POSE_PLACE, GRIP_CLOSE)
    arm.set_pose(place, "PLACE")

    log("[12] Release gripper")
    arm.grip(GRIP_OPEN, "OPEN")

    log("[13] Lift to PRE_PLACE")
    pre_place_empty = list(pre_place); pre_place_empty[-1] = GRIP_OPEN
    arm.set_pose(pre_place_empty, "LIFT")

    log("[14] Return to HOME")
    arm.set_pose(POSE_HOME, "HOME")

    log("\n" + "=" * 55)
    log("  SEQUENCE COMPLETE")
    log("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    do_drive  = not args.no_drive
    base_node = None
    logger    = None

    if do_drive:
        rclpy.init()
        base_node = BaseMover()
        logger    = base_node.get_logger()

    arm = ArmController(dry_run=args.dry_run, logger=logger)

    try:
        run_sequence(arm, base_node, do_drive, logger)
    except KeyboardInterrupt:
        print("Interrupted — returning arm to HOME")
    finally:
        arm.shutdown()
        if base_node is not None:
            base_node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()