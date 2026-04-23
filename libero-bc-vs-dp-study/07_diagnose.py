"""
07_diagnose.py — Diagnostic analysis of MLP vs Diffusion Policy.

Produces 4 plots in outputs/diagnostics/:
  1. action_distribution.png  -- per-dim histograms (MLP / DP / GT)
  2. gripper_timeline.png     -- one-episode gripper trace (MLP / DP / GT)
  3. action_smoothness.png    -- ||a_t - a_{t-1}||_2 distribution
  4. dp_open_loop_gripper.png -- DP's 16-step open-loop gripper trajectories

Run:
    python 07_diagnose.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUT = Path("outputs/diagnostics")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load eval module to reuse load_policy / preprocess_image
spec = importlib.util.spec_from_file_location("eval_mod", "04_eval.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

print("Loading models...")
mlp_pred, _, _ = eval_mod.load_policy("outputs/model_mlp_baseline.pt", DEVICE)
dp_pred, _, dp_horizon = eval_mod.load_policy(
    "outputs/model_diffusion_policy.pt", DEVICE
)

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
DEMO_FILE = sorted(
    Path(get_libero_path("datasets"), "libero_spatial").glob("*.hdf5")
)[0]
print(f"Using demo file: {DEMO_FILE.name}")


# ---------------------------------------------------------------------------
# Collect predictions on real env observations
# ---------------------------------------------------------------------------

def collect_predictions(num_init_states: int = 20):
    """For each init state, take 1 obs and record MLP and DP predictions."""
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bm.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states = bm.get_task_init_states(0)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128
    )

    mlp_actions = []  # (N, 7)
    dp_first_actions = []  # (N, 7)   first step of DP horizon
    dp_full_horizons = []  # (N, 16, 7)

    try:
        n = min(num_init_states, len(init_states))
        for i in tqdm(range(n), desc="Collecting predictions"):
            env.reset()
            obs = env.set_init_state(init_states[i])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            img_t = eval_mod.preprocess_image(obs["agentview_image"])
            a_mlp = mlp_pred(img_t)
            a_dp = dp_pred(img_t)  # (16, 7)

            mlp_actions.append(a_mlp)
            dp_first_actions.append(a_dp[0])
            dp_full_horizons.append(a_dp)
    finally:
        env.close()

    return (
        np.stack(mlp_actions),
        np.stack(dp_first_actions),
        np.stack(dp_full_horizons),
    )


def collect_gt_actions():
    """Aggregate ground-truth actions across all demos in the file."""
    all_actions = []
    with h5py.File(DEMO_FILE, "r") as f:
        for k in f["data"].keys():
            all_actions.append(f[f"data/{k}/actions"][:])
    return np.concatenate(all_actions, axis=0)


def get_one_episode_gt(demo_key: str = "demo_0"):
    """Return (T, 7) action sequence of one demo for timeline plotting."""
    with h5py.File(DEMO_FILE, "r") as f:
        return f[f"data/{demo_key}/actions"][:]


# ---------------------------------------------------------------------------
# Plot 1: per-dim action distribution
# ---------------------------------------------------------------------------

def plot_action_distribution(mlp_a, dp_a, gt_a):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for d in range(7):
        ax = axes[d]
        ax.hist(gt_a[:, d], bins=40, alpha=0.4, label="GT", color="gray", density=True)
        ax.hist(mlp_a[:, d], bins=20, alpha=0.6, label="MLP", color="C0", density=True)
        ax.hist(dp_a[:, d], bins=20, alpha=0.6, label="DP", color="C3", density=True)
        ax.set_title(f"action[{d}] = {ACTION_NAMES[d]}")
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend()
    axes[7].axis("off")
    fig.suptitle(
        "Action distribution: predictions vs ground-truth\n"
        "(MLP/DP collected on 20 init-state observations; GT from all demos)",
        y=1.02,
    )
    fig.tight_layout()
    p = OUT / "action_distribution.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p}")


# ---------------------------------------------------------------------------
# Plot 2: gripper timeline along one demo
# ---------------------------------------------------------------------------

def plot_gripper_timeline():
    """Replay demo_0 frame by frame; query MLP/DP at each step on the env image."""
    bm = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bm.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states = bm.get_task_init_states(0)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128
    )

    gt = get_one_episode_gt("demo_0")  # (T, 7)
    T = min(len(gt), 150)

    mlp_g, dp_g = [], []
    try:
        env.reset()
        obs = env.set_init_state(init_states[0])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        for t in tqdm(range(T), desc="Replay GT actions"):
            img_t = eval_mod.preprocess_image(obs["agentview_image"])
            mlp_g.append(mlp_pred(img_t)[6])
            dp_g.append(dp_pred(img_t)[0, 6])  # DP first-step gripper

            # Step env using GT action so we follow expert trajectory
            obs, _, done, _ = env.step(gt[t].astype(np.float32))
            if done:
                break
    finally:
        env.close()

    T_actual = len(mlp_g)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt[:T_actual, 6], label="GT (expert)", color="gray", linewidth=2)
    ax.plot(mlp_g, label="MLP", color="C0")
    ax.plot(dp_g, label="DP (1st step)", color="C3")
    ax.axhline(0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("gripper action  (-1 = open, +1 = close)")
    ax.set_title("Gripper prediction along demo_0 trajectory (env replays GT actions)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = OUT / "gripper_timeline.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p}")


# ---------------------------------------------------------------------------
# Plot 3: action smoothness  (||a_t - a_{t-1}||_2 distribution)
# ---------------------------------------------------------------------------

def plot_action_smoothness(dp_full_horizons, gt_a_full):
    """
    Compare adjacent-step action diffs:
      - DP open-loop within its 16-step horizon  (across init states, average)
      - GT consecutive demo actions
    MLP isn't applicable here (it's per-step independent on changing obs).
    """
    # DP: diff within horizon, then flatten across init states
    dp_diffs = np.linalg.norm(np.diff(dp_full_horizons, axis=1), axis=2).flatten()

    # GT: per-demo consecutive diffs
    gt_diffs = []
    with h5py.File(DEMO_FILE, "r") as f:
        for k in f["data"].keys():
            a = f[f"data/{k}/actions"][:]
            gt_diffs.append(np.linalg.norm(np.diff(a, axis=0), axis=1))
    gt_diffs = np.concatenate(gt_diffs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gt_diffs, bins=60, alpha=0.5, label="GT (expert)", color="gray", density=True)
    ax.hist(dp_diffs, bins=60, alpha=0.5, label="DP (open-loop horizon)", color="C3", density=True)
    ax.set_xlabel(r"$\|a_t - a_{t-1}\|_2$")
    ax.set_ylabel("density")
    ax.set_title("Action smoothness: per-step delta magnitude")
    ax.legend()
    ax.grid(alpha=0.3)
    p = OUT / "action_smoothness.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p}")
    print(f"    GT mean step-delta:  {gt_diffs.mean():.3f}")
    print(f"    DP mean step-delta:  {dp_diffs.mean():.3f}  ({dp_diffs.mean()/gt_diffs.mean():.1f}x)")


# ---------------------------------------------------------------------------
# Plot 4: DP's 16-step open-loop gripper trajectories
# ---------------------------------------------------------------------------

def plot_dp_open_loop_gripper(dp_full_horizons):
    """Visualize the 16-step gripper rollout from each init state."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for traj in dp_full_horizons:
        ax.plot(traj[:, 6], color="C3", alpha=0.3)
    mean = dp_full_horizons[:, :, 6].mean(axis=0)
    ax.plot(mean, color="black", linewidth=2, label="mean across init states")
    ax.axhline(-1, color="green", linestyle="--", alpha=0.5, label="GT open (-1)")
    ax.axhline(+1, color="orange", linestyle="--", alpha=0.5, label="GT close (+1)")
    ax.set_xlabel("step within DP open-loop horizon")
    ax.set_ylabel("predicted gripper")
    ax.set_title(
        "DP's 16-step open-loop gripper prediction\n"
        "(each red line = one init state; ground-truth gripper is binary ±1)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    p = OUT / "dp_open_loop_gripper.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n[1/4] Collecting model predictions on real env observations...")
    mlp_a, dp_a, dp_horizons = collect_predictions(num_init_states=20)
    gt_a = collect_gt_actions()
    print(f"  MLP actions: {mlp_a.shape}")
    print(f"  DP first-step actions: {dp_a.shape}")
    print(f"  DP full horizons: {dp_horizons.shape}")
    print(f"  GT actions (all demos): {gt_a.shape}")

    print("\n[2/4] Plotting action distribution...")
    plot_action_distribution(mlp_a, dp_a, gt_a)

    print("\n[3/4] Plotting gripper timeline (replays demo_0)...")
    plot_gripper_timeline()

    print("\n[4/4] Plotting action smoothness + DP open-loop gripper...")
    plot_action_smoothness(dp_horizons, gt_a)
    plot_dp_open_loop_gripper(dp_horizons)

    print(f"\nAll diagnostics saved to {OUT}/")


if __name__ == "__main__":
    main()
