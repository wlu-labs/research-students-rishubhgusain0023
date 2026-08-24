"""
synthetic_world.py
-------------------
Ground-truth test environments + a synthetic LiDAR ray-caster.

Each "world" is a simple list of axis-aligned rectangular obstacles in
world metres, plus a start pose and a goal point. The ray-caster takes
the ground-truth world and a robot pose and produces a LaserScan-shaped
object (angle_min/angle_max/angle_increment/range_min/range_max/ranges)
identical in spec to the YDLiDAR TG30 scans the real planners expect
(see slam_module.py / rrt_virtual_mover*.py — all three subscribe to
/scan with this same message shape).

We feed this synthetic scan through each variant's own OccupancyMap.update()
so each planner builds its own log-odds occupancy grid exactly as it would
from real LiDAR hardware — the benchmark never hands a planner a precomputed
obstacle mask directly.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Minimal LaserScan-shaped object (duck-types against what
#    OccupancyMap.update() reads: angle_min, angle_increment,
#    range_min, range_max, ranges) ─────────────────────────────
class FakeLaserScan:
    def __init__(self, angle_min, angle_max, angle_increment,
                 range_min, range_max, ranges):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = ranges


# YDLiDAR TG30 nominal spec (matches eda.qmd hardware description):
# 360 deg field of view. We use 720 beams (0.5 deg resolution) which
# is representative of the TG30's real angular resolution.
LIDAR_N_BEAMS     = 720
LIDAR_RANGE_MIN   = 0.10
LIDAR_RANGE_MAX   = 12.0
LIDAR_ANGLE_MIN   = -math.pi
LIDAR_ANGLE_MAX   = math.pi


@dataclass
class World:
    name: str
    width_m: float
    height_m: float
    obstacles: List[Tuple[float, float, float, float]]  # (xmin,ymin,xmax,ymax)
    start: Tuple[float, float]
    goal: Tuple[float, float]
    description: str = ""


def _ray_hits_rect(px, py, dx, dy, rect, max_range):
    """
    Ray (px,py)+t*(dx,dy), t in [0,max_range], vs axis-aligned rect.
    Returns smallest t>0 hit distance, or None.
    """
    xmin, ymin, xmax, ymax = rect
    tmin, tmax = 0.0, max_range

    for (o, d, lo, hi) in ((px, dx, xmin, xmax), (py, dy, ymin, ymax)):
        if abs(d) < 1e-12:
            if o < lo or o > hi:
                return None
        else:
            t1 = (lo - o) / d
            t2 = (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin if tmin > 1e-9 else None


def cast_scan(world: World, robot_x: float, robot_y: float, robot_yaw: float,
              n_beams: int = LIDAR_N_BEAMS,
              range_max: float = LIDAR_RANGE_MAX,
              noise_std_m: float = 0.0,
              rng: random.Random = None) -> FakeLaserScan:
    """
    Ray-cast a synthetic 360-degree LiDAR scan against `world`'s
    ground-truth rectangular obstacles plus the world boundary walls.
    Optional Gaussian range noise (metres) for realism.
    """
    rng = rng or random
    angle_increment = (LIDAR_ANGLE_MAX - LIDAR_ANGLE_MIN) / n_beams
    ranges = []

    # Treat the world boundary as four bounding walls so rays that
    # escape through open space still terminate at a finite range,
    # matching how a real LiDAR in a room always hits a wall eventually.
    half_w, half_h = world.width_m / 2.0, world.height_m / 2.0
    boundary_rects = [
        (-half_w - 0.05, -half_h - 0.05, half_w + 0.05, -half_h),       # bottom wall
        (-half_w - 0.05, half_h, half_w + 0.05, half_h + 0.05),         # top wall
        (-half_w - 0.05, -half_h - 0.05, -half_w, half_h + 0.05),       # left wall
        (half_w, -half_h - 0.05, half_w + 0.05, half_h + 0.05),         # right wall
    ]
    all_rects = world.obstacles + boundary_rects

    angle = LIDAR_ANGLE_MIN
    for _ in range(n_beams):
        beam_world_angle = robot_yaw + angle
        dx, dy = math.cos(beam_world_angle), math.sin(beam_world_angle)

        best_t = range_max
        for rect in all_rects:
            t = _ray_hits_rect(robot_x, robot_y, dx, dy, rect, range_max)
            if t is not None and t < best_t:
                best_t = t

        if noise_std_m > 0.0 and best_t < range_max:
            best_t = max(LIDAR_RANGE_MIN, best_t + rng.gauss(0.0, noise_std_m))

        ranges.append(best_t)
        angle += angle_increment

    return FakeLaserScan(
        angle_min=LIDAR_ANGLE_MIN, angle_max=LIDAR_ANGLE_MAX,
        angle_increment=angle_increment,
        range_min=LIDAR_RANGE_MIN, range_max=range_max,
        ranges=ranges,
    )


# ════════════════════════════════════════════════════════════════
#  Hand-built scenarios — chosen for interpretability
# ════════════════════════════════════════════════════════════════

def world_open_room() -> World:
    """Large empty room — sanity check / best-case path (near-straight line)."""
    return World(
        name="open_room",
        width_m=10.0, height_m=10.0,
        obstacles=[],
        start=(-4.0, -4.0), goal=(4.0, 4.0),
        description="Empty 10x10m room, no obstacles — straight-line baseline.",
    )


def world_single_obstacle() -> World:
    """One block directly between start and goal — forces a single detour."""
    return World(
        name="single_obstacle",
        width_m=10.0, height_m=10.0,
        obstacles=[(-1.0, -1.0, 1.0, 1.0)],
        start=(-4.0, 0.0), goal=(4.0, 0.0),
        description="Single 2x2m block centred on the direct path.",
    )


def world_narrow_corridor() -> World:
    """Tight corridor (~0.6m clear width) — stresses inflation + collision logic."""
    return World(
        name="narrow_corridor",
        width_m=10.0, height_m=6.0,
        obstacles=[
            (-5.0, 0.4, 5.0, 3.0),
            (-5.0, -3.0, 5.0, -0.4),
        ],
        start=(-4.0, 0.0), goal=(4.0, 0.0),
        description="Corridor ~0.8m wide (walls at y=+-0.4m) running the full length.",
    )


def world_cluttered() -> World:
    """Several scattered blocks of varying size — realistic lab-room clutter."""
    return World(
        name="cluttered",
        width_m=12.0, height_m=12.0,
        obstacles=[
            (-3.5, -3.5, -2.0, -2.0),
            (0.5,  -4.0, 2.0, -2.5),
            (-1.0,  0.5, 1.0, 2.0),
            (2.5,   1.0, 4.0, 3.0),
            (-4.5,  2.0, -3.0, 4.0),
            (1.0,  -1.0, 1.8, 0.5),
        ],
        start=(-5.0, -5.0), goal=(5.0, 5.0),
        description="Six scattered rectangular obstacles of varying size — lab clutter analogue.",
    )


def world_maze() -> World:
    """Maze-like nested walls forcing several direction reversals."""
    return World(
        name="maze",
        width_m=10.0, height_m=10.0,
        obstacles=[
            (-5.0, -1.0, 2.0, 1.0),     # long wall, gap on the right
            (1.0, -5.0, 3.0, -1.0),     # vertical wall up from bottom
            (-3.0, 1.0, -1.0, 5.0),     # vertical wall down from top
            (-1.0, 2.5, 3.5, 4.5),      # upper horizontal wall
        ],
        start=(-4.5, -4.5), goal=(4.5, 4.5),
        description="Four interlocking walls forcing a zig-zag route.",
    )


def hand_built_worlds() -> List[World]:
    return [
        world_open_room(),
        world_single_obstacle(),
        world_narrow_corridor(),
        world_cluttered(),
        world_maze(),
    ]


# ════════════════════════════════════════════════════════════════
#  Randomized obstacle fields — for statistical robustness
# ════════════════════════════════════════════════════════════════

def random_world(seed: int, density: str = "medium",
                  size_m: float = 12.0) -> World:
    """
    Randomly scatter rectangular obstacles across a size_m x size_m room.
    density: 'sparse' | 'medium' | 'dense' controls obstacle count.
    Start/goal are pinned at opposite corners (with margin) and re-rolled
    if they land inside an obstacle.
    """
    rng = random.Random(seed)
    n_obstacles = {"sparse": 4, "medium": 9, "dense": 16}[density]

    half = size_m / 2.0
    margin = 0.6
    obstacles = []
    for _ in range(n_obstacles):
        w = rng.uniform(0.4, 1.8)
        h = rng.uniform(0.4, 1.8)
        cx = rng.uniform(-half + 1.0, half - 1.0)
        cy = rng.uniform(-half + 1.0, half - 1.0)
        obstacles.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))

    boundary_clearance = 1.2  # keep start/goal well clear of the room's outer wall

    def free(pt):
        x, y = pt
        if abs(x) > half - boundary_clearance or abs(y) > half - boundary_clearance:
            return False
        for (xmin, ymin, xmax, ymax) in obstacles:
            if xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin:
                return False
        return True

    def nudge_toward_clear(pt, away_from_sign):
        """
        Nudge pt inward (toward room centre, away from the boundary it's
        pinned near) until free(). away_from_sign is (+1,+1) for the
        start corner (bottom-left, nudge toward +x,+y) or (-1,-1) for
        the goal corner (top-right, nudge toward -x,-y) — this guarantees
        every nudge step moves strictly toward the room's interior and
        can never push the point further out toward / past the boundary.
        """
        x, y = pt
        attempts = 0
        while not free((x, y)) and attempts < 200:
            x += away_from_sign[0] * 0.05
            y += away_from_sign[1] * 0.05
            attempts += 1
        return (x, y)

    corner = half - boundary_clearance
    start = nudge_toward_clear((-corner, -corner), away_from_sign=(+1, +1))
    goal  = nudge_toward_clear(( corner,  corner), away_from_sign=(-1, -1))

    return World(
        name=f"random_{density}_seed{seed}",
        width_m=size_m, height_m=size_m,
        obstacles=obstacles,
        start=start, goal=goal,
        description=f"Randomized {density} field, {n_obstacles} obstacles, seed={seed}.",
    )


def random_worlds(densities=("sparse", "medium", "dense"),
                   seeds=(1, 2, 3, 4, 5)) -> List[World]:
    worlds = []
    for density in densities:
        for seed in seeds:
            worlds.append(random_world(seed=seed, density=density))
    return worlds
