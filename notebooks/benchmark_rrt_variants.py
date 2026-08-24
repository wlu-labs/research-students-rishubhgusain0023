#!/usr/bin/env python3
# coding=utf-8
"""
benchmark_rrt_variants.py
--------------------------
Benchmarks the three RRT* implementations currently in the repo:

  rrt_virtual_mover.py            (baseline: brute-force O(n) RRT*)
  rrt_virtual_mover_quadtree.py   (+ QuadTree NN/radius search, side-retreat escape)
  rrt_virtual_mover_convex.py     (+ convex-corridor QP path smoothing)

against each other on the SAME synthetic LiDAR-built occupancy maps, so the
comparison reflects what each planner would actually do given identical
sensor input — not a precomputed obstacle mask handed to all three.

For each (world, trial) this script:
  1. Builds a fresh OccupancyMap for each variant (each variant has its own
     OccupancyMap class — identical fields/behaviour, but kept separate
     because that's how they'd run as separate ROS2 nodes on the robot).
  2. Ray-casts a synthetic 360-deg LiDAR scan from the world's ground-truth
     obstacles and feeds it through map.update(scan, x, y, yaw) — exactly
     the call each real node makes from its /scan callback.
  3. Calls planner.plan(start, goal) and times it.
  4. Records: success, waypoint count, path length, planning time,
     and (for variants exposing it) whether RRT* rewiring ran.

Run:
  PYTHONPATH=ros_stubs python3 benchmark_rrt_variants.py
"""

import os
import sys
import math
import time
import json
import random
import statistics
import importlib
import contextlib
import io

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros_stubs"))

from synthetic_world import (
    World, cast_scan, hand_built_worlds, random_worlds,
)

VARIANT_MODULES = {
    "baseline":  "rrt_virtual_mover",
    "quadtree":  "rrt_virtual_mover_quadtree",
    "convex":    "rrt_virtual_mover_convex",
}

# How many independent scans to feed each map before planning, and from
# how many vantage points — mimics the robot having driven around a bit
# (multiple /scan callbacks) before a /goal arrives, rather than a single
# scan from the start pose only seeing what's in direct line of sight.
SCAN_VANTAGE_POINTS_FRAC = [0.0, 0.25, 0.5, 0.75, 1.0]  # fraction of start->goal line
RANDOM_SEED_BASE = 12345
NOISE_STD_M = 0.01   # 1cm range noise, realistic for YDLiDAR TG30


def _load_variant(module_name):
    mod = importlib.import_module(module_name)
    return mod


def _build_map_from_world(mod, world: World, rng: random.Random):
    """
    Build this variant's OccupancyMap by feeding it synthetic LiDAR scans
    from several vantage points along the start->goal line (simulating the
    robot having partially explored before the goal was issued).
    """
    occ_map = mod.OccupancyMap(
        width_m=world.width_m, height_m=world.height_m, resolution=0.05
    )
    sx, sy = world.start
    gx, gy = world.goal
    for frac in SCAN_VANTAGE_POINTS_FRAC:
        vx = sx + frac * (gx - sx)
        vy = sy + frac * (gy - sy)
        yaw = math.atan2(gy - sy, gx - sx)
        scan = cast_scan(world, vx, vy, yaw, noise_std_m=NOISE_STD_M, rng=rng)
        occ_map.update(scan, vx, vy, yaw)
    return occ_map


def _path_length(path):
    if len(path) < 2:
        return 0.0
    pts = np.array(path, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _path_max_curvature_proxy(path):
    """Mean squared second-difference — a smoothness proxy (lower = smoother)."""
    if len(path) < 3:
        return 0.0
    pts = np.array(path, dtype=np.float64)
    accel = pts[:-2] - 2 * pts[1:-1] + pts[2:]
    return float(np.mean(np.sum(accel ** 2, axis=1)))


def run_single_trial(variant_key: str, mod, world: World, trial_seed: int):
    rng = random.Random(trial_seed)
    # Reseed the variant module's own `random` usage (goal-bias sampling,
    # random.uniform for tree exploration) so each variant sees the SAME
    # random stream for a fair side-by-side comparison on this trial.
    random.seed(trial_seed)

    occ_map = _build_map_from_world(mod, world, rng)
    planner = mod.RRTPlanner(occ_map)

    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            path = planner.plan(world.start, world.goal)
    except Exception as e:
        return {
            "variant": variant_key, "world": world.name, "trial_seed": trial_seed,
            "success": False, "error": str(e),
            "waypoints": 0, "path_length_m": None,
            "planning_time_s": time.perf_counter() - t0,
            "smoothness": None,
        }
    dt = time.perf_counter() - t0

    success = bool(path)
    return {
        "variant": variant_key,
        "world": world.name,
        "trial_seed": trial_seed,
        "success": success,
        "error": None,
        "waypoints": len(path) if success else 0,
        "path_length_m": _path_length(path) if success else None,
        "planning_time_s": dt,
        "smoothness": _path_max_curvature_proxy(path) if success else None,
    }


def run_benchmark(worlds, trials_per_world=5, out_path="results.json"):
    variants = {k: _load_variant(v) for k, v in VARIANT_MODULES.items()}

    results = []
    total = len(worlds) * trials_per_world * len(variants)
    done = 0

    for world in worlds:
        for trial_i in range(trials_per_world):
            trial_seed = RANDOM_SEED_BASE + hash((world.name, trial_i)) % 100000
            for variant_key, mod in variants.items():
                r = run_single_trial(variant_key, mod, world, trial_seed)
                r["world_description"] = world.description
                results.append(r)
                done += 1
                status = "OK" if r["success"] else "FAIL"
                print(f"[{done}/{total}] {world.name:18s} trial{trial_i} "
                      f"{variant_key:10s} {status:5s} "
                      f"t={r['planning_time_s']:.3f}s "
                      f"wp={r['waypoints']:3d} "
                      f"len={r['path_length_m'] if r['path_length_m'] else 0:.2f}m")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} trial records to {out_path}")
    return results


def summarize(results, out_path="summary.json"):
    """Aggregate per (world, variant): success rate, mean/median metrics."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[(r["world"], r["variant"])].append(r)

    summary = []
    for (world, variant), rs in groups.items():
        n = len(rs)
        n_success = sum(1 for r in rs if r["success"])
        succ_rs = [r for r in rs if r["success"]]

        def stat(key, fn):
            vals = [r[key] for r in succ_rs if r[key] is not None]
            return fn(vals) if vals else None

        summary.append({
            "world": world,
            "variant": variant,
            "n_trials": n,
            "success_rate": n_success / n,
            "mean_planning_time_s": stat("planning_time_s", statistics.mean),
            "median_planning_time_s": stat("planning_time_s", statistics.median),
            "mean_waypoints": stat("waypoints", statistics.mean),
            "mean_path_length_m": stat("path_length_m", statistics.mean),
            "mean_smoothness": stat("smoothness", statistics.mean),
        })

    summary.sort(key=lambda s: (s["world"], s["variant"]))
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {out_path}")
    return summary


def print_summary_table(summary):
    worlds = sorted(set(s["world"] for s in summary))
    variants = sorted(set(s["variant"] for s in summary))

    header = f"{'world':18s} {'variant':10s} {'succ%':>6s} {'time(ms)':>9s} {'wpts':>6s} {'len(m)':>7s} {'smooth':>9s}"
    print(header)
    print("-" * len(header))
    for w in worlds:
        for v in variants:
            match = [s for s in summary if s["world"] == w and s["variant"] == v]
            if not match:
                continue
            s = match[0]
            t_ms = s["mean_planning_time_s"] * 1000 if s["mean_planning_time_s"] else 0.0
            print(f"{w:18s} {v:10s} {s['success_rate']*100:6.1f} {t_ms:9.1f} "
                  f"{(s['mean_waypoints'] or 0):6.1f} {(s['mean_path_length_m'] or 0):7.2f} "
                  f"{(s['mean_smoothness'] or 0):9.4f}")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5,
                         help="Trials per (world, variant) — different random seed each time")
    parser.add_argument("--skip-random", action="store_true",
                         help="Only run hand-built worlds (faster)")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    worlds = hand_built_worlds()
    if not args.skip_random:
        worlds += random_worlds()

    print(f"Running {len(worlds)} worlds x {args.trials} trials x 3 variants "
          f"= {len(worlds)*args.trials*3} planner calls\n")

    results = run_benchmark(
        worlds, trials_per_world=args.trials,
        out_path=os.path.join(args.out_dir, "results.json"))
    summary = summarize(results, out_path=os.path.join(args.out_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print_summary_table(summary)
