#!/usr/bin/env python3
"""Chunked eye robot planner with Jacobian-preconditioned energy minimization.

The planning loop relies on TinyGrad for automatic differentiation and can run
either in a text-based simulator or an OpenCV-powered video interface that maps
image clicks to workspace goals.
"""

import argparse, math, os, random, sys, time
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("DISABLE_DISK_CACHE", "1")
os.environ.setdefault("DISABLE_COMPILER_CACHE", "1")
os.environ.setdefault("CACHELEVEL", "0")

try:
    from tinygrad.tensor import Tensor
    from tinygrad.nn import Linear
    from tinygrad.dtype import dtypes
    from tinygrad import TinyJit
except ImportError:
    print("tinygrad is required. Install via `pip install tinygrad`.")
    sys.exit(1)

LINK = (0.4, 0.3, 0.2)
DELTA = 2
WORK_X = (0.2, 0.9)
WORK_Y = (-0.3, 0.5)
TAU = math.tau


def banner() -> None:
    """Print a quick description and CLI usage examples for the demo."""
    print(
        "eye_robot_demo.py | chunked planner with Jacobian-preconditioned EBM\n"
        "Examples:\n"
        "  METAL=1 DEFAULT_FLOAT=FLOAT32 python eye_robot_demo.py --mode sim\n"
        "  METAL=1 DEFAULT_FLOAT=FLOAT32 python eye_robot_demo.py --mode video --camera_index 0\n"
        "  METAL=1 DEFAULT_FLOAT=FLOAT32 python eye_robot_demo.py --mode video --video_path demo.mp4\n"
        "-"
    )


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar into the inclusive range [`lo`, `hi`]."""
    return lo if x < lo else hi if x > hi else x


def wrap_angle(a: float) -> float:
    """Wrap an angle in radians into the interval (-pi, pi]."""
    a = (a + math.pi) % TAU - math.pi
    return a + TAU if a <= -math.pi else a - TAU if a > math.pi else a


def format_vec(vec: Sequence[float], prec: int = 3) -> str:
    """Format a sequence of floats for pretty logging."""
    return "[" + ", ".join(f"{v:.{prec}f}" for v in vec) + "]"


def tensor_row_to_list(t: Tensor) -> List[float]:
    """Convert a 1D TinyGrad tensor to a Python float list in float32."""
    t_cast = t if t.dtype == dtypes.float32 else t.cast(dtypes.float32)
    return [float(x) for x in t_cast.tolist()]


def wrap_tensor(theta: Tensor) -> Tensor:
    """Elementwise angle wrap for tensors, keeping grads intact."""
    two_pi = TAU
    shifted = theta + math.pi
    wrapped = shifted - (shifted / two_pi).floor() * two_pi - math.pi
    wrapped = Tensor.where(wrapped <= -math.pi, wrapped + two_pi, wrapped)
    wrapped = Tensor.where(wrapped > math.pi, wrapped - two_pi, wrapped)
    return wrapped


def make_energy_jit(prior, horizon: int, alpha: float, beta: float, compute_dtype):
    """Build a TinyJit-ed energy function that matches `ChunkedPlanner.energy`."""
    @TinyJit
    def energy_jit(plan: Tensor, goal: Tensor) -> Tensor:
        plan_c = plan if plan.dtype == compute_dtype else plan.cast(compute_dtype)
        goal_c = goal if goal.dtype == compute_dtype else goal.cast(compute_dtype)
        ee = fk_xy(plan_c[-1])
        goal_term = (ee - goal_c).pow(2).sum()
        if horizon > 1:
            smooth = (plan_c[1:] - plan_c[:-1]).pow(2).sum(axis=1).mean()
        else:
            smooth = plan_c.sum() * 0
        return alpha * goal_term + beta * smooth + prior(plan_c)

    return energy_jit


def fk_xy(q: Tensor) -> Tensor:
    """Planar forward kinematics that maps joint angles to `(x, y)` positions."""
    theta1 = q[..., 0]
    theta12 = theta1 + q[..., 1]
    theta123 = theta12 + q[..., 2]
    x = theta1.cos() * LINK[0] + theta12.cos() * LINK[1] + theta123.cos() * LINK[2]
    y = theta1.sin() * LINK[0] + theta12.sin() * LINK[1] + theta123.sin() * LINK[2]
    return x.stack(y, dim=-1 if len(x.shape) > 0 else 0)


def jacobian_diag(q_vals: Sequence[float]) -> List[float]:
    """Return diagonal terms of (J^T J) for scaling joint-space gradients."""
    q1, q2, q3 = q_vals
    th1, th12, th123 = q1, q1 + q2, q1 + q2 + q3
    s = math.sin
    c = math.cos
    dx1 = -LINK[0] * s(th1) - LINK[1] * s(th12) - LINK[2] * s(th123)
    dx2 = -LINK[1] * s(th12) - LINK[2] * s(th123)
    dx3 = -LINK[2] * s(th123)
    dy1 = LINK[0] * c(th1) + LINK[1] * c(th12) + LINK[2] * c(th123)
    dy2 = LINK[1] * c(th12) + LINK[2] * c(th123)
    dy3 = LINK[2] * c(th123)
    return [dx1 * dx1 + dy1 * dy1, dx2 * dx2 + dy2 * dy2, dx3 * dx3 + dy3 * dy3]


def simple_ik(goal_xy: Sequence[float]) -> List[float]:
    """Analytic IK seed that reaches a goal using a simple wrist-off strategy."""
    x, y = goal_xy
    l1, l2, l3 = LINK
    r = clamp(math.hypot(x, y), 1e-6, l1 + l2 + l3 - 1e-6)
    wx, wy = x - l3 * x / r, y - l3 * y / r
    rw = math.hypot(wx, wy)
    cos_q2 = clamp((rw * rw - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    q2 = math.acos(cos_q2)
    q1 = math.atan2(wy, wx) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    th12 = q1 + q2
    x2 = l1 * math.cos(q1) + l2 * math.cos(th12)
    y2 = l1 * math.sin(q1) + l2 * math.sin(th12)
    q3 = math.atan2(y - y2, x - x2) - th12
    return [wrap_angle(q1), wrap_angle(q2), wrap_angle(q3)]


class PriorMLP:
    """Small frozen MLP that penalizes implausible joint sequences."""

    def __init__(self, horizon: int) -> None:
        hidden = max(16, horizon * 3)
        self.h = horizon
        self.fc1 = Linear(horizon * 3, hidden)
        self.fc2 = Linear(hidden, hidden // 2)
        self.fc3 = Linear(hidden // 2, 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            layer.weight.requires_grad = False
            if layer.bias is not None:
                layer.bias.requires_grad = False

    def __call__(self, plan: Tensor) -> Tensor:
        """Compute the prior cost for a single plan or batch of plans."""
        if len(plan.shape) == 2:
            flat = plan.reshape((1, self.h * 3))
        else:
            batch = plan.shape[0]
            flat = plan.reshape((batch, self.h * 3))
        x = self.fc1(flat).relu()
        x = self.fc2(x).relu()
        out = self.fc3(x).relu()
        return out.reshape((out.shape[0],)) if len(plan.shape) == 3 else out.reshape(())


class ChunkedPlanner:
    """Sampling-based planner that refines top trajectories with Jacobian steps."""

    def __init__(
        self,
        horizon: int,
        candidates: int,
        refine_top: int,
        lr: float,
        alpha: float = 1.0,
        beta: float = 1e-2,
        lam: float = 1e-3,
        alpha_j: float = 0.3,
        beta_ema: float = 0.9,
        noise_scale: float = 0.05,
    ) -> None:
        """Configure planning horizon, sampling counts, and optimizer weights."""
        self.H, self.K, self.M, self.lr = horizon, candidates, refine_top, lr
        self.alpha, self.beta, self.lam = alpha, beta, lam
        self.alpha_j, self.beta_ema, self.noise = alpha_j, beta_ema, noise_scale
        self.prior = PriorMLP(horizon)
        self.ema = [[[0.0] * 3 for _ in range(horizon)] for _ in range(refine_top)]
        compute_name = os.environ.get("EYE_COMPUTE_DTYPE", "float32").lower()
        self.compute_dtype = getattr(dtypes, compute_name, dtypes.float32)
        tau_vals = [(t + 1) / (horizon + 1) for t in range(horizon)]
        self.tau = Tensor(tau_vals).reshape(1, horizon, 1).cast(self.compute_dtype)
        self.use_jit = bool(int(os.environ.get("EYE_TINYJIT", "0")))
        if self.use_jit:
            self._energy_eval = make_energy_jit(self.prior, self.H, self.alpha, self.beta, self.compute_dtype)
        else:
            self._energy_eval = None
        self.proposals: Optional[List[List[List[float]]]] = None
        self.proposal_energy: Optional[List[float]] = None
        self.last_goal: Optional[List[float]] = None
        self.resample_count = max(1, candidates // 4)
        self.goal_reset_thresh = float(os.environ.get("EYE_GOAL_RESET", "0.08"))

    def reset(self) -> None:
        """Clear EMA history and candidate plans so planning restarts fresh."""
        for buf in self.ema:
            for row in buf:
                for j in range(3):
                    row[j] = 0.0
        self.proposals = None
        self.proposal_energy = None
        self.last_goal = None

    def energy(self, plan: Tensor, goal: Tensor) -> Tensor:
        """Evaluate the scalar energy for one plan or a batch of plans."""
        if self._energy_eval is not None:
            return self._energy_eval(plan, goal)
        plan_c = plan if plan.dtype == self.compute_dtype else plan.cast(self.compute_dtype)
        goal_c = goal if goal.dtype == self.compute_dtype else goal.cast(self.compute_dtype)
        if len(plan_c.shape) == 3:
            ee = fk_xy(plan_c[:, -1, :])
            if len(goal_c.shape) == 2:
                goal_expand = goal_c
            else:
                goal_list = goal_c.tolist()
                goal_expand = Tensor([goal_list for _ in range(plan_c.shape[0])], dtype=self.compute_dtype)
            goal_term = (ee - goal_expand).pow(2).sum(axis=1)
            if self.H > 1:
                smooth = (plan_c[:, 1:, :] - plan_c[:, :-1, :]).pow(2).sum(axis=2).mean(axis=1)
            else:
                smooth = Tensor.zeros((plan_c.shape[0],), dtype=self.compute_dtype)
            prior_term = self.prior(plan_c)
            return self.alpha * goal_term + self.beta * smooth + prior_term
        ee = fk_xy(plan_c[-1])
        goal_term = (ee - goal_c).pow(2).sum()
        smooth = (plan_c[1:] - plan_c[:-1]).pow(2).sum(axis=1).mean() if self.H > 1 else Tensor([0.0]).sum()
        return self.alpha * goal_term + self.beta * smooth + self.prior(plan_c)

    def _generate_plan(self, q0: Sequence[float], target: Sequence[float]) -> List[List[float]]:
        """Interpolate from the current joint state to an IK target with noise."""
        plan_rows: List[List[float]] = []
        for t in range(self.H):
            tau = float(t + 1) / float(self.H + 1)
            row = []
            for j in range(3):
                interp = q0[j] + (target[j] - q0[j]) * tau
                noise = random.gauss(0.0, self.noise)
                row.append(wrap_angle(interp + noise))
            plan_rows.append(row)
        return plan_rows

    def _generate_plans(self, q0: Sequence[float], target: Sequence[float], count: int) -> List[List[List[float]]]:
        """Sample `count` noisy plans around the same target seed."""
        return [self._generate_plan(q0, target) for _ in range(count)]

    def refine(self, plan: Tensor, goal: Tensor, ema: List[List[float]]) -> Tuple[Tensor, float, float]:
        """Run one gradient-descent refinement step on a single candidate."""
        storage_dtype = plan.dtype
        W = plan if plan.dtype == self.compute_dtype else plan.cast(self.compute_dtype)
        W.requires_grad = True
        goal_c = goal if goal.dtype == self.compute_dtype else goal.cast(self.compute_dtype)
        e_before = self.energy(W, goal_c)
        e_before.backward()
        grad = W.grad
        if grad is None:
            raise RuntimeError("Gradient is None; ensure candidate tensor has requires_grad and TinyJIT path preserves autograd")
        g_vals = grad.tolist()
        beta = self.beta_ema
        one_minus = 1.0 - beta
        for i in range(self.H):
            for j in range(3):
                ema[i][j] = beta * ema[i][j] + one_minus * (g_vals[i][j] ** 2)
        diag_rows = []
        jt = jacobian_diag(tensor_row_to_list(W[-1]))
        for i in range(self.H):
            if i == self.H - 1:
                diag_rows.append([self.alpha_j * jt[c] + beta * ema[i][c] + self.lam for c in range(3)])
            else:
                diag_rows.append([ema[i][c] + self.lam for c in range(3)])
        D = Tensor(diag_rows)
        step = (grad / D) * self.lr
        W_new = wrap_tensor(W - step)
        if storage_dtype != self.compute_dtype:
            W_new = W_new.cast(storage_dtype)
        e_after = self.energy(W_new, goal_c)
        W.grad = None
        return W_new, float(e_before.item()), float(e_after.item())

    def plan_once(self, q0: Sequence[float], goal_xy: Sequence[float]) -> Tuple[float, float, float, List[float]]:
        """Run one planning tick and return diagnostics plus the next joint move."""
        target = simple_ik(goal_xy)
        reset_required = self.proposals is None
        if self.last_goal is not None and not reset_required:
            dx = goal_xy[0] - self.last_goal[0]
            dy = goal_xy[1] - self.last_goal[1]
            if (dx * dx + dy * dy) ** 0.5 > self.goal_reset_thresh:
                reset_required = True
        if reset_required:
            self.proposals = self._generate_plans(q0, target, self.K)
            self.proposal_energy = None
        elif self.proposal_energy is not None:
            worst = sorted(range(self.K), key=lambda i: self.proposal_energy[i], reverse=True)[: self.resample_count]
            for idx, new_plan in zip(worst, self._generate_plans(q0, target, len(worst))):
                self.proposals[idx] = new_plan

        proposals_tensor = Tensor(self.proposals, dtype=self.compute_dtype)
        goal_tensor = Tensor(goal_xy).cast(self.compute_dtype)
        energy_tensor = self.energy(proposals_tensor, goal_tensor)
        energies = [float(x) for x in energy_tensor.tolist()]
        self.proposal_energy = energies[:]
        top = sorted(range(self.K), key=energies.__getitem__)[: self.M]
        if not top:
            best_idx = int(min(range(self.K), key=lambda i: energies[i]))
            best_plan_vals = self.proposals[best_idx]
            delta_row = best_plan_vals[min(DELTA, self.H - 1)]
            q_next = [wrap_angle(v) for v in delta_row]
            best_energy = energies[best_idx]
            self.last_goal = list(goal_xy)
            return best_energy, best_energy, 0.0, q_next

        batch_tensor = Tensor([self.proposals[idx] for idx in top], dtype=self.compute_dtype)
        batch_tensor.requires_grad = True
        goal_batch = Tensor([goal_xy for _ in top], dtype=self.compute_dtype)
        energies_batch = self.energy(batch_tensor, goal_batch)
        total_energy = energies_batch.sum()
        total_energy.backward()
        energy_before_list = [float(x) for x in energies_batch.detach().tolist()]
        grad_vals = batch_tensor.grad.tolist()
        batch_tensor.grad = None
        batch_detach = batch_tensor.detach().tolist()

        updated_plans: List[List[List[float]]] = []
        after_list: List[float] = []
        start = time.time()
        for local_rank, global_idx in enumerate(top):
            plan_vals = [[float(angle) for angle in row] for row in batch_detach[local_rank]]
            grad_plan = grad_vals[local_rank]
            ema_rank = self.ema[local_rank]
            last_angles = plan_vals[-1]
            jtj_diag = jacobian_diag(last_angles)
            denom_rows: List[List[float]] = []
            for i in range(self.H):
                denom_row: List[float] = []
                for j in range(3):
                    ema_rank[i][j] = self.beta_ema * ema_rank[i][j] + (1.0 - self.beta_ema) * (grad_plan[i][j] ** 2)
                    if i == self.H - 1:
                        denom = self.alpha_j * jtj_diag[j] + self.beta_ema * ema_rank[i][j] + self.lam
                    else:
                        denom = ema_rank[i][j] + self.lam
                    denom_row.append(denom)
                denom_rows.append(denom_row)
            for i in range(self.H):
                for j in range(3):
                    step = (grad_plan[i][j] / denom_rows[i][j]) * self.lr
                    plan_vals[i][j] = wrap_angle(plan_vals[i][j] - step)
            updated_plans.append(plan_vals)
            after_energy = float(self.energy(Tensor(plan_vals).cast(self.compute_dtype), goal_tensor).item())
            after_list.append(after_energy)
            self.proposals[global_idx] = plan_vals
            self.proposal_energy[global_idx] = after_energy

        refine_time = time.time() - start
        best_local = int(min(range(len(top)), key=lambda r: after_list[r]))
        best_global = top[best_local]
        best_before = energy_before_list[best_local]
        best_after = after_list[best_local]
        best_plan_vals = updated_plans[best_local]
        delta_row = best_plan_vals[min(DELTA, self.H - 1)]
        q_next = [wrap_angle(v) for v in delta_row]

        self.last_goal = list(goal_xy)

        return best_before, best_after, refine_time, q_next

def plan_tick(planner: ChunkedPlanner, q: List[float], goal: List[float]) -> Tuple[float, float, float, List[float], float, float, int]:
    """Convenience wrapper that runs one planner step and logs TinyGrad counters."""
    from tinygrad.helpers import GlobalCounters

    GlobalCounters.reset()
    tick_start = time.time()
    e_before, e_after, dt, q_next = planner.plan_once(q, goal)
    elapsed_ms = (time.time() - tick_start) * 1e3
    metal_ms = GlobalCounters.time_sum_s * 1e3
    kernels = GlobalCounters.kernel_count
    cpu_ms = max(elapsed_ms - metal_ms, 0.0)
    return e_before, e_after, dt, q_next, cpu_ms, metal_ms, kernels


def run_sim(planner: ChunkedPlanner) -> None:
    """Drive the planner toward a fixed workspace goal and print tick metrics."""
    planner.reset()
    q = [0.0, 0.0, 0.0]
    goal = [0.6, 0.1]
    total_before = total_after = total_refine = 0.0
    improved = 0
    ticks = 50
    for tick in range(1, ticks + 1):
        e_before, e_after, dt, q_next, cpu_ms, metal_ms, kernels = plan_tick(planner, q, goal)
        q = q_next
        ee = tensor_row_to_list(fk_xy(Tensor(q)))
        total_before += e_before
        total_after += e_after
        total_refine += dt
        if e_after < e_before - 1e-6:
            improved += 1
        print(
            f"tick {tick:02d} | E {e_before:.4f} -> {e_after:.4f} | q0={format_vec(q)} | ee={format_vec(ee)}"
            f" | cpu={cpu_ms:.1f}ms metal={metal_ms:.1f}ms ({kernels} kernels)"
        )
    avg_before = total_before / ticks
    avg_after = total_after / ticks
    avg_refine = (total_refine / ticks) * 1000.0
    print("-\nAverage energy before refine: %.4f\nAverage energy after refine:  %.4f\nAverage refine time per tick (ms): %.2f\nFraction of ticks improved: %.2f" % (avg_before, avg_after, avg_refine, improved / ticks))


def pixel_to_workspace(px: int, py: int, w: int, h: int) -> List[float]:
    """Map pixel coordinates from the video frame into the robot workspace."""
    u, v = px / max(w - 1, 1), py / max(h - 1, 1)
    return [WORK_X[0] + u * (WORK_X[1] - WORK_X[0]), WORK_Y[0] + (1.0 - v) * (WORK_Y[1] - WORK_Y[0])]


def draw_overlay(frame, goal_px: Optional[Tuple[int, int]], ee: Sequence[float]) -> None:
    """Annotate the OpenCV frame with the goal marker and end-effector status."""
    import cv2; h, w = frame.shape[:2]
    cv2.circle(frame, (w // 2, h // 2), 18, (0, 255, 255), 1)
    if goal_px is not None:
        cv2.circle(frame, goal_px, 6, (0, 0, 255), -1)
    cv2.putText(frame, f"EE ({ee[0]:.2f}, {ee[1]:.2f})", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run_video(planner: ChunkedPlanner, args: argparse.Namespace) -> None:
    """Main event loop for webcam/video tracking with optional GUI overlays."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not available. Install via `pip install opencv-python` for video mode.")
        sys.exit(1)

    planner.reset()
    cap = cv2.VideoCapture(args.video_path if args.video_path else args.camera_index)
    if not cap.isOpened():
        print("Failed to open video source."); return
    state = {'goal_px': None, 'goal_xy': None, 'width': 1, 'height': 1, 'goal_updated': False}
    if args.show:
        cv2.namedWindow("eye_robot_demo")
        cv2.setMouseCallback("eye_robot_demo", on_mouse, state)

    q = [0.0, 0.0, 0.0]
    total_before = total_after = total_refine = 0.0
    improved = ticks = 0
    waiting_notice = False
    frame_interval = 1.0 / max(args.fps, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or failed.")
                break
            h, w = frame.shape[:2]
            state['width'], state['height'] = w, h
            if not args.show and state['goal_xy'] is None:
                center = (w // 2, h // 2); state['goal_px'] = center; state['goal_xy'] = pixel_to_workspace(center[0], center[1], w, h); state['goal_updated'] = True
            goal = state['goal_xy']
            if state.get('goal_updated') and goal is not None:
                print(f"New image goal -> workspace {format_vec(goal, prec=3)}")
                state['goal_updated'] = False
            if goal is None:
                if args.show:
                    cv2.putText(frame, "Click to set goal", (int(w * 0.25), int(h * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA); draw_overlay(frame, state['goal_px'], (0.0, 0.0)); cv2.imshow("eye_robot_demo", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        break
                else:
                    if not waiting_notice:
                        print("Waiting for goal... (enable --show to select visually)"); waiting_notice = True
                    time.sleep(frame_interval)
                continue

            ticks += 1
            tick_start = time.time()
            e_before, e_after, dt, q_next, cpu_ms, metal_ms, kernels = plan_tick(planner, q, goal)
            q = q_next
            total_before += e_before
            total_after += e_after
            total_refine += dt
            if e_after < e_before - 1e-6:
                improved += 1
            ee = tensor_row_to_list(fk_xy(Tensor(q)))
            print(
                f"tick {ticks:02d} | E {e_before:.4f} -> {e_after:.4f} | q0={format_vec(q)} | ee={format_vec(ee)}"
                f" | cpu={cpu_ms:.1f}ms metal={metal_ms:.1f}ms ({kernels} kernels)"
            )
            if args.show:
                draw_overlay(frame, state['goal_px'], ee); cv2.imshow("eye_robot_demo", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            elapsed = time.time() - tick_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    if ticks == 0:
        print("No planning ticks executed.")
        return

    avg_before = total_before / ticks
    avg_after = total_after / ticks
    avg_refine = (total_refine / ticks) * 1000.0
    print("-\nAverage energy before refine: %.4f\nAverage energy after refine:  %.4f\nAverage refine time per tick (ms): %.2f\nFraction of ticks improved: %.2f" % (avg_before, avg_after, avg_refine, improved / ticks))


def on_mouse(event, x, y, flags, state):  # pragma: no cover
    """Mouse callback that captures a new video-mode goal when the user clicks."""
    import cv2
    if state and event == cv2.EVENT_LBUTTONDOWN:
        state['goal_px'] = (x, y)
        state['goal_xy'] = pixel_to_workspace(x, y, state.get('width', 1), state.get('height', 1))
        state['goal_updated'] = True


def build_parser() -> argparse.ArgumentParser:
    """Argument parser covering both simulation and video execution modes."""
    parser = argparse.ArgumentParser(description="Chunked planner demo with tinygrad")
    parser.add_argument('--mode', choices=['sim', 'video'], default='sim', help='Run in simulation or video mode')
    parser.add_argument('--video_path', type=str, default=None, help='Video file path for --mode video')
    parser.add_argument('--camera_index', type=int, default=0, help='Webcam index for video mode')
    parser.add_argument('--fps', type=int, default=30, help='Planner refresh rate in video mode')
    parser.add_argument('--H', type=int, default=8, help='Planning horizon')
    parser.add_argument('--K', type=int, default=12, help='Number of candidates per tick')
    parser.add_argument('--M', type=int, default=3, help='Top candidates refined with Jacobian step')
    parser.add_argument('--lr', type=float, default=0.2, help='Jacobian-preconditioned step size')
    parser.add_argument('--show', dest='show', action='store_true', default=None, help='Display OpenCV overlays (video mode)')
    parser.add_argument('--no-show', dest='show', action='store_false', help='Disable OpenCV overlays')
    return parser


def main() -> None:
    """Entry point that seeds randomness, parses flags, and dispatches modes."""
    random.seed(0)
    try:
        Tensor.manual_seed(0)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    parser = build_parser()
    args = parser.parse_args()
    if args.show is None:
        args.show = args.mode == 'video'
    banner()
    planner = ChunkedPlanner(args.H, args.K, args.M, args.lr)
    if args.mode == 'sim':
        run_sim(planner)
    else:
        run_video(planner, args)


if __name__ == '__main__':
    main()
