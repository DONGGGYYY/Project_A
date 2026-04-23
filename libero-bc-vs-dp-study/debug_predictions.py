"""Quick sanity check: compare MLP vs DP action predictions on real LIBERO obs."""
import os
import importlib.util
from pathlib import Path
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import sys
sys.path.insert(0, str(Path.cwd()))
spec = importlib.util.spec_from_file_location('eval_mod', '04_eval.py')
eval_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(eval_mod)

device = torch.device('cuda')

# Load both
mlp_pred,  _, _ = eval_mod.load_policy('outputs/model_mlp_baseline.pt',  device)
dp_pred,   _, _ = eval_mod.load_policy('outputs/model_diffusion_policy.pt', device)

# Set up env, take a single obs
bm = benchmark.get_benchmark_dict()['libero_spatial']()
task = bm.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
init_states = bm.get_task_init_states(0)

print("Comparing predictions on 5 different init states\n")
print(f"{'state':<6} {'model':<5} {'gripper':<10} {'first 6 dims (xyz + euler)':<60}")
print("-" * 90)
for i in range(5):
    env.reset()
    obs = env.set_init_state(init_states[i])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    img_t = eval_mod.preprocess_image(obs['agentview_image'])

    a_mlp = mlp_pred(img_t)              # (7,)
    a_dp  = dp_pred(img_t)               # (16, 7)  -- first action
    a_dp0 = a_dp[0]

    print(f"{i:<6} MLP   {a_mlp[6]:>+8.3f}   {np.array2string(a_mlp[:6], precision=3, suppress_small=True)}")
    print(f"{i:<6} DP[0] {a_dp0[6]:>+8.3f}   {np.array2string(a_dp0[:6], precision=3, suppress_small=True)}")
    # Show gripper trajectory across the 16-step DP rollout
    print(f"       DP gripper traj (16 steps): {np.array2string(a_dp[:, 6], precision=2, suppress_small=True)}")
    print()

env.close()

# Also compare to ground truth actions in the dataset
print("\n=== Ground truth gripper distribution from training data ===")
import h5py
demo_files = sorted(Path(get_libero_path('datasets'), 'libero_spatial').glob('*.hdf5'))
with h5py.File(demo_files[0], 'r') as f:
    demo_keys = list(f['data'].keys())[:3]
    for k in demo_keys:
        actions = f[f'data/{k}/actions'][:]
        print(f"  {k}: action[:,6] range [{actions[:,6].min():.2f}, {actions[:,6].max():.2f}]  "
              f"unique~{np.unique(np.round(actions[:,6])).tolist()}  "
              f"mean={actions[:,6].mean():.3f}")
