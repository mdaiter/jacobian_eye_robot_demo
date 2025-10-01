# Jacobian Eye Robot Demo

This repository contains a compact demonstration of a Jacobian-preconditioned energy-based motion planner built with [tinygrad](https://github.com/tinygrad/tinygrad). The planner controls a three-degree-of-freedom "eye" robot arm that tracks targets either in simulation or directly from video frames. The code is intentionally small and fully differentiable, making it a great starting point for students and practitioners who want to understand modern planning loops that blend sampling, gradient refinement, and learned priors.

## What's Inside

- **`eye_robot_demo.py`** – the main script. It layers explanatory docstrings throughout the planner so you can read the source and learn how sampling, Jacobian-based updates, and the tinygrad energy objective fit together.
- **`tests/test_eye_robot_demo.py`** – lightweight pytest checks that confirm the helper utilities and planner loop behave as documented.

## Why This Repo Is Useful

1. **Teaches Differentiable Planning** – The example is small enough to understand in a single sitting yet rich enough to show how automatic differentiation, forward kinematics, and Jacobian preconditioning interact in an energy-based method.
2. **Bridges Simulation and Vision** – You can swap between a pure simulation goal and a live video target by changing a single CLI flag, illustrating how the same planner feeds both setups.
3. **tinygrad Ecosystem Example** – If you're exploring tinygrad, this project shows a practical, end-to-end use case beyond neural network training.
4. **Extensible Scaffold** – Modify the energy terms, plug in a new prior, or retarget the robot's geometry to prototype alternative controllers quickly.

## Getting Started

```bash
# Optional: create a fresh virtual environment first
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

pip install tinygrad opencv-python pytest
```

## Running the Demo

### Simulation Mode

```bash
python eye_robot_demo.py --mode sim
```

The script prints per-tick energy before/after refinement, the commanded joint state, and wall-clock timing broken into CPU vs. accelerator time (if supported by your backend).

### Video Mode

```bash
python eye_robot_demo.py --mode video --camera_index 0 --show
```

- Click anywhere in the OpenCV window to set a goal. A red dot marks the chosen pixel, and the overlay tracks the end effector in workspace coordinates.
- Use `--video_path demo.mp4` to run against a recorded clip instead of a webcam.
- Pass `--no-show` for a headless run that still processes frames and prints planner diagnostics.

> **Tip:** The planner defaults to float32 math. You can experiment with other tinygrad compute dtypes via `EYE_COMPUTE_DTYPE=float16` (or similar) before launching the script. Set `EYE_TINYJIT=1` to enable JIT compilation once you're comfortable with the flow.

## Exploring the Notebook

Launch Jupyter and open the tutorial to step through the planner interactively:

```bash
jupyter lab eye_robot_demo_tutorial.ipynb
```

Each section introduces a new concept—forward kinematics, inverse kinematics seeding, energy evaluation, refinement loop—and includes code cells you can tweak on the fly.

## Running Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

The tests disable tinygrad's caches so they run cleanly in environments with stricter filesystem permissions.

## Next Steps

- Tweak `H`, `K`, `M`, and `lr` in `ChunkedPlanner` to see how the planner trades off compute for better convergence.
- Swap the `PriorMLP` with your own network or disable it entirely to observe the effect on trajectory smoothness.
- Instrument additional metrics (e.g., end-effector velocity) to deepen your understanding of the optimization dynamics.
