"""
10_train_diffusion_multitask.py — Diffusion Policy on ALL 10 LIBERO-Spatial tasks.

Same model + EMA setup as 06_train_diffusion.ipynb.
Output: outputs/model_diffusion_multitask.pt
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from libero.libero import get_libero_path


def _load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ds_mod = _load_local("dataset02", Path(__file__).parent / "02_dataset.py")
make_multitask_dataset = ds_mod.make_multitask_dataset

dp_mod = _load_local("diff_policy", Path(__file__).parent / "05_diffusion_policy.py")
DiffusionPolicy = dp_mod.DiffusionPolicy


class EMAModel:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.shadow.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for eb, b in zip(self.shadow.buffers(), model.buffers()):
            eb.data.copy_(b.data)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    HORIZON = 16
    BATCH = 16
    NUM_EPOCHS = 15

    # --- data: all 10 tasks ---
    dataset_dir = Path(get_libero_path("datasets")) / "libero_spatial"
    hdf5_paths = sorted(str(p) for p in dataset_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_paths)} hdf5 files")

    full = make_multitask_dataset(hdf5_paths, seq_len=HORIZON, use_state=False)
    train_size = int(0.9 * len(full))
    test_size = len(full) - train_size
    train_ds, test_ds = random_split(
        full, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # --- model ---
    policy = DiffusionPolicy(
        action_dim=7,
        horizon=HORIZON,
        cond_dim=256,
        num_train_timesteps=100,
        num_inference_steps=10,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"DP params: {n_params/1e6:.2f}M")

    optimizer = optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    ema = EMAModel(policy, decay=0.995)

    history = {"train_loss": [], "test_loss": [], "lr": []}

    for epoch in range(NUM_EPOCHS):
        policy.train()
        running = 0.0
        n = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            imgs = batch["images"].to(device)
            acts = batch["actions"].to(device)
            loss = policy.compute_loss(imgs, acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(policy)
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = running / n
        history["train_loss"].append(train_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        scheduler.step()

        # eval (EMA model)
        ema.shadow.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["images"].to(device)
                acts = batch["actions"].to(device)
                loss = ema.shadow.compute_loss(imgs, acts)
                running += loss.item() * imgs.size(0)
                n += imgs.size(0)
        test_loss = running / n
        history["test_loss"].append(test_loss)
        print(f"epoch {epoch+1:02d}  train {train_loss:.5f}  test {test_loss:.5f}  lr {history['lr'][-1]:.2e}")

    # --- save ---
    out = Path("outputs") / "model_diffusion_multitask.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "ema_state_dict": ema.shadow.state_dict(),
        "history": history,
        "config": {
            "action_dim": 7,
            "horizon": HORIZON,
            "num_train_timesteps": 100,
            "num_inference_steps": 10,
            "num_tasks": len(hdf5_paths),
        },
    }, out)
    print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
