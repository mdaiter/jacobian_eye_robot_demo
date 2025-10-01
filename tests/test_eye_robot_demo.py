"""Quick behavioral checks for `eye_robot_demo`'s documented functionality."""

import math
import os
import random
import sys
from pathlib import Path

import pytest


sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ["CACHELEVEL"] = "0"
os.environ["DISABLE_DISK_CACHE"] = "1"
os.environ["DISABLE_COMPILER_CACHE"] = "1"

pytest.importorskip("tinygrad")

from tinygrad.tensor import Tensor

import eye_robot_demo as eye


def test_wrap_angle_clamps_within_range():
    angles = [math.pi + 0.25, -math.pi - 1.5, 3 * math.pi]
    wrapped = [eye.wrap_angle(a) for a in angles]
    assert all(-math.pi < w <= math.pi for w in wrapped)


def test_fk_xy_matches_expected_origin_pose():
    zero = Tensor([[0.0, 0.0, 0.0]])
    ee = eye.tensor_row_to_list(eye.fk_xy(zero)[0])
    assert pytest.approx(ee[0], rel=1e-6, abs=1e-6) == sum(eye.LINK)
    assert pytest.approx(ee[1], rel=1e-6, abs=1e-6) == 0.0


def test_simple_ik_hit_goal_with_forward_kinematics():
    goal = [0.65, 0.1]
    guess = eye.simple_ik(goal)
    reached = eye.tensor_row_to_list(eye.fk_xy(Tensor([guess]))[0])
    assert pytest.approx(reached[0], rel=1e-3, abs=1e-3) == goal[0]
    assert pytest.approx(reached[1], rel=1e-3, abs=1e-3) == goal[1]


def test_chunked_planner_step_reduces_energy():
    random.seed(0)
    try:
        Tensor.manual_seed(0)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    planner = eye.ChunkedPlanner(horizon=6, candidates=8, refine_top=3, lr=0.2)
    planner.reset()
    q = [0.0, 0.0, 0.0]
    goal = [0.55, 0.0]
    energies = []
    for _ in range(5):
        e_before, e_after, _, q, *_ = eye.plan_tick(planner, q, goal)
        energies.append(e_after)
    assert min(energies) <= energies[0] + 1e-4
