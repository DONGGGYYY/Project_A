"""
Unified evaluation script: 在 LIBERO 仿真环境中评估 MLP baseline 和 Diffusion Policy。

用法 (在项目根目录):
    # 单个模型
    python 04_eval.py --model outputs/model_mlp_baseline.pt --num-episodes 20
    python 04_eval.py --model outputs/model_diffusion_policy.pt --num-episodes 20

    # 对比两个模型 (推荐)
    python 04_eval.py --compare --num-episodes 20 --save-videos
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ---------------------------------------------------------------------------
# Image preprocessing (must match 02_dataset.py exactly)
# ---------------------------------------------------------------------------

def preprocess_image(rgb_uint8: np.ndarray) -> torch.Tensor:
    """env (H,W,3) uint8 -> (3,H,W) float32, vertical flip + /255."""
    img = rgb_uint8[::-1].copy()
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


# ---------------------------------------------------------------------------
# Model loading: auto-detect MLP vs Diffusion Policy
# ---------------------------------------------------------------------------

def _load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MLPModel(nn.Module):
    """Same architecture as in 03_train_mlp.ipynb. Optionally 6-ch input for wrist ablation."""

    def __init__(self, action_dim: int = 7, hidden_dim: int = 256, in_channels: int = 3):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        if in_channels != 3:
            old = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None,
            )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images):
        return self.head(self.backbone(images))


def load_policy(model_path: str, device: torch.device, inference_steps: Optional[int] = None):
    """
    Returns (predict_fn, kind, horizon).
    Auto-detects type via presence of 'ema_state_dict' / 'config.horizon' in ckpt.
    inference_steps: override DDIM steps at sampling time (default = ckpt config).
    """
    ckpt = torch.load(model_path, map_location=device)
    is_diffusion = 'ema_state_dict' in ckpt or (
        'config' in ckpt and 'horizon' in ckpt.get('config', {})
    )

    if is_diffusion:
        dp_mod = _load_local_module(
            'diff_policy', Path(__file__).parent / '05_diffusion_policy.py'
        )
        cfg = ckpt.get('config', {})
        policy = dp_mod.DiffusionPolicy(
            action_dim=cfg.get('action_dim', 7),
            horizon=cfg.get('horizon', 16),
            num_train_timesteps=cfg.get('num_train_timesteps', 100),
            num_inference_steps=cfg.get('num_inference_steps', 10),
        ).to(device)
        policy.load_state_dict(ckpt.get('ema_state_dict', ckpt['model_state_dict']))
        policy.eval()

        eff_steps = inference_steps or cfg.get('num_inference_steps', 10)
        print(f"DDIM inference steps: {eff_steps}")

        @torch.no_grad()
        def predict(img_tensor: torch.Tensor) -> np.ndarray:
            batch = img_tensor.unsqueeze(0).to(device)
            actions = policy.sample(batch, num_inference_steps=eff_steps)
            return actions[0].cpu().numpy()

        return predict, "diffusion", cfg.get('horizon', 16)

    # MLP fallback
    cfg = ckpt.get('config', {})
    in_channels = cfg.get('in_channels', 3)
    use_wrist = cfg.get('use_wrist', False)
    model = MLPModel(action_dim=7, hidden_dim=256, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    @torch.no_grad()
    def predict(img_tensor: torch.Tensor) -> np.ndarray:
        batch = img_tensor.unsqueeze(0).to(device)
        return model(batch)[0].cpu().numpy()

    kind = "mlp_wrist" if use_wrist else "mlp"
    return predict, kind, 1


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------

def rollout_episode(
    env,
    obs: Dict,
    predict_fn,
    kind: str,
    horizon: int,
    max_steps: int = 300,
    chunk_steps: int = 8,
    save_video: bool = False,
) -> Dict:
    """
    chunk_steps: diffusion only - sample horizon-step sequence but only execute
                 first chunk_steps before re-sensing (receding horizon control).
    """
    success = False
    frames: List[np.ndarray] = []
    pending: List[np.ndarray] = []
    step = 0

    while step < max_steps:
        img = obs["agentview_image"]
        if save_video:
            frames.append(img[::-1].copy())  # flip for correct orientation

        if kind == "mlp":
            action = predict_fn(preprocess_image(img))
        elif kind == "mlp_wrist":
            wrist = obs["robot0_eye_in_hand_image"]
            # preprocess each cam (flip + CHW + /255), concat on channel dim -> (6,H,W)
            t_agent = preprocess_image(img)
            t_wrist = preprocess_image(wrist)
            action = predict_fn(torch.cat([t_agent, t_wrist], dim=0))
        else:
            if not pending:
                seq = predict_fn(preprocess_image(img))  # (horizon, 7)
                pending = list(seq[:chunk_steps])
            action = pending.pop(0)

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, reward, done, info = env.step(action)
        step += 1

        try:
            done_success = env.check_success()
        except AttributeError:
            done_success = bool(done)

        if done_success or done:
            success = bool(done_success)
            if save_video:
                frames.append(obs["agentview_image"][::-1].copy())
            break

    return {
        "success": bool(success),
        "steps": step,
        "frames": frames if save_video else None,
    }


# ---------------------------------------------------------------------------
# Evaluate one model on a task
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    num_episodes: int = 20,
    max_steps: int = 300,
    chunk_steps: int = 8,
    save_videos: bool = False,
    video_dir: Optional[Path] = None,
    device: str = "cuda",
    inference_steps: Optional[int] = None,
) -> Dict:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\nEvaluating: {model_path}\n{'='*60}")

    predict_fn, kind, horizon = load_policy(model_path, device, inference_steps=inference_steps)
    print(f"Model kind: {kind}, horizon: {horizon}")

    benchmark_dict = benchmark.get_benchmark_dict()
    bm = benchmark_dict[task_suite_name]()
    task = bm.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    init_states = bm.get_task_init_states(task_id)
    print(f"Task: {task.name}")
    print(f"Available init states: {len(init_states)}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(42)

    if save_videos:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    per_episode = []

    try:
        for ep in tqdm(range(num_episodes), desc="Episodes"):
            env.reset()
            init_idx = ep % len(init_states)
            obs = env.set_init_state(init_states[init_idx])
            # Let physics settle (LIBERO recommends a few zero-action steps)
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

            res = rollout_episode(
                env, obs, predict_fn, kind, horizon,
                max_steps=max_steps,
                chunk_steps=chunk_steps,
                save_video=save_videos,
            )
            successes += int(res["success"])
            per_episode.append({
                "init_state_id": init_idx,
                "success": res["success"],
                "steps": res["steps"],
            })

            if save_videos and res["frames"]:
                vid_path = video_dir / f"ep{ep:03d}_{'OK' if res['success'] else 'FAIL'}.mp4"
                imageio.mimsave(str(vid_path), res["frames"], fps=20)
    finally:
        env.close()

    success_rate = successes / num_episodes
    print(f"\n[{Path(model_path).name}] Success: {successes}/{num_episodes}  "
          f"= {success_rate*100:.1f}%")

    return {
        "model_path": str(model_path),
        "model_kind": kind,
        "task_suite": task_suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "episodes": per_episode,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="single model checkpoint path")
    parser.add_argument("--compare", action="store_true",
                        help="evaluate both MLP baseline and Diffusion Policy")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--chunk-steps", type=int, default=8,
                        help="diffusion: steps to execute per sample (receding horizon)")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="diffusion: override DDIM denoising steps (default from ckpt = 10)")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    if args.compare:
        models = {
            "mlp_baseline": out_dir / "model_mlp_baseline.pt",
            "diffusion_policy": out_dir / "model_diffusion_policy.pt",
        }
        all_results = {}
        for name, path in models.items():
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            video_dir = out_dir / "videos" / name if args.save_videos else None
            all_results[name] = evaluate_model(
                model_path=str(path),
                task_suite_name=args.task_suite,
                task_id=args.task_id,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                chunk_steps=args.chunk_steps,
                save_videos=args.save_videos,
                video_dir=video_dir,
                device=args.device,
                inference_steps=args.inference_steps,
            )

        print(f"\n{'='*60}\n  COMPARISON\n{'='*60}")
        print(f"{'Model':<25} {'Success Rate':<15} {'Successes':<12}")
        for name, res in all_results.items():
            print(f"{name:<25} {res['success_rate']*100:>6.1f}%        "
                  f"{res['successes']}/{res['num_episodes']}")

        with open(out_dir / "eval_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {out_dir/'eval_results.json'}")

    elif args.model:
        video_dir = out_dir / "videos" / Path(args.model).stem if args.save_videos else None
        result = evaluate_model(
            model_path=args.model,
            task_suite_name=args.task_suite,
            task_id=args.task_id,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            chunk_steps=args.chunk_steps,
            save_videos=args.save_videos,
            video_dir=video_dir,
            device=args.device,
            inference_steps=args.inference_steps,
        )
        with open(out_dir / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2)
    else:
        parser.error("specify --model or --compare")


if __name__ == "__main__":
    main()
