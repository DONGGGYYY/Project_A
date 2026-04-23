"""
08_task_generalization.py — Evaluate both models across all 10 LIBERO-Spatial tasks.

Both models were trained on task 0 only. This tests cross-task generalization.

Output:
  outputs/task_generalization.png   -- 10 x 2 success-rate heatmap
  outputs/task_generalization.json  -- raw numbers
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_EPISODES_PER_TASK = 10
MAX_STEPS = 300
CHUNK_STEPS = 1   # for DP: per-step re-sense (gives DP its best shot)
SUITE = "libero_spatial"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("outputs")

# ---------------------------------------------------------------------------
# Load eval helpers
# ---------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location("eval_mod", "04_eval.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)


def evaluate_one(predict_fn, kind, horizon, task_id, num_episodes):
    bm = benchmark.get_benchmark_dict()[SUITE]()
    task = bm.get_task(task_id)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states = bm.get_task_init_states(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128
    )
    env.seed(42)

    successes = 0
    try:
        for ep in range(num_episodes):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            res = eval_mod.rollout_episode(
                env, obs, predict_fn, kind, horizon,
                max_steps=MAX_STEPS,
                chunk_steps=CHUNK_STEPS,
                save_video=False,
            )
            successes += int(res["success"])
    finally:
        env.close()

    return successes, task.name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global CHUNK_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp", type=str, default="outputs/model_mlp_baseline.pt")
    parser.add_argument("--dp", type=str, default="outputs/model_diffusion_policy.pt")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for output files, e.g. 'multitask'")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_PER_TASK)
    parser.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS,
                        help="DP receding-horizon stride (1 = per-step resample, 8 = paper default)")
    args = parser.parse_args()

    CHUNK_STEPS = args.chunk_steps

    n_eps = args.episodes
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"Cross-task generalization: {SUITE}")
    print(f"  MLP: {args.mlp}")
    print(f"  DP : {args.dp}")
    print(f"{n_eps} episodes/task, chunk_steps={CHUNK_STEPS}\n")

    # Load both policies once
    print("Loading models...")
    mlp_pred, mlp_kind, _ = eval_mod.load_policy(args.mlp, DEVICE)
    dp_pred, _, dp_h = eval_mod.load_policy(args.dp, DEVICE)
    print(f"  MLP kind: {mlp_kind}")

    bm = benchmark.get_benchmark_dict()[SUITE]()
    n_tasks = bm.n_tasks
    print(f"Number of tasks in suite: {n_tasks}\n")

    results = {"mlp": [], "dp": [], "task_names": []}

    for tid in tqdm(range(n_tasks), desc="Tasks"):
        s_mlp, name = evaluate_one(mlp_pred, mlp_kind, 1, tid, n_eps)
        s_dp, _ = evaluate_one(dp_pred, "diffusion", dp_h, tid, n_eps)
        results["mlp"].append(s_mlp)
        results["dp"].append(s_dp)
        results["task_names"].append(name)
        print(f"  task {tid}: MLP {s_mlp}/{n_eps}  DP {s_dp}/{n_eps}  ({name[:60]})")

    # Save raw
    json_path = OUT_DIR / f"task_generalization{suffix}.json"
    with open(json_path, "w") as f:
        json.dump({
            "num_episodes_per_task": n_eps,
            "chunk_steps": CHUNK_STEPS,
            "mlp_ckpt": args.mlp,
            "dp_ckpt": args.dp,
            "tasks": [
                {
                    "task_id": i,
                    "task_name": results["task_names"][i],
                    "mlp_successes": results["mlp"][i],
                    "dp_successes": results["dp"][i],
                    "mlp_rate": results["mlp"][i] / n_eps,
                    "dp_rate": results["dp"][i] / n_eps,
                }
                for i in range(n_tasks)
            ],
        }, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Plot
    mlp_rates = np.array(results["mlp"]) / n_eps
    dp_rates = np.array(results["dp"]) / n_eps
    matrix = np.stack([mlp_rates, dp_rates], axis=0)  # (2, n_tasks)

    fig, ax = plt.subplots(figsize=(max(8, n_tasks * 0.9), 3.2))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MLP\n(BC)", "Diffusion\nPolicy"])
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)])
    ax.set_xlabel("Task ID")

    # Annotate cells
    for i in range(2):
        for j in range(n_tasks):
            v = matrix[i, j]
            ax.text(j, i, f"{int(v*100)}%", ha="center", va="center",
                    color="black" if v > 0.4 else "white", fontsize=10)

    title_setting = f"multi-task ({args.tag})" if args.tag else "trained on T0 only"
    ax.set_title(
        f"Cross-task evaluation on {SUITE}  ·  {title_setting}\n"
        f"{n_eps} episodes/task"
    )
    plt.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    p = OUT_DIR / f"task_generalization{suffix}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"All-task avg:    MLP {mlp_rates.mean()*100:.1f}%   DP {dp_rates.mean()*100:.1f}%")
    print(f"T0:              MLP {mlp_rates[0]*100:.0f}%       DP {dp_rates[0]*100:.0f}%")
    print(f"T1..T9 avg:      MLP {mlp_rates[1:].mean()*100:.1f}%   DP {dp_rates[1:].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
