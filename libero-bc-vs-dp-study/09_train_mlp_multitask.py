"""
09_train_mlp_multitask.py — MLP baseline trained on ALL 10 LIBERO-Spatial tasks.

Same architecture as 03_train_mlp.ipynb, only the dataset changes:
combine all 10 hdf5 files (~50 demos x 10 tasks) into one big training set.

Output: outputs/model_mlp_multitask.pt
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from tqdm import tqdm

from libero.libero import get_libero_path


# --- load local dataset module ----------------------------------------------
def _load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ds_mod = _load_local("dataset02", Path(__file__).parent / "02_dataset.py")
make_multitask_dataset = ds_mod.make_multitask_dataset


# --- model (same as 03_train_mlp) ------------------------------------------
class ImitationLearningModel(nn.Module):
    def __init__(self, action_dim=7, hidden_dim=256):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images):
        return self.head(self.backbone(images))


# --- main -------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- data: ALL 10 tasks ---
    dataset_dir = Path(get_libero_path("datasets")) / "libero_spatial"
    hdf5_paths = sorted(str(p) for p in dataset_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_paths)} hdf5 files")

    full = make_multitask_dataset(hdf5_paths, seq_len=1, use_state=False)

    train_size = int(0.9 * len(full))
    test_size = len(full) - train_size
    train_ds, test_ds = random_split(
        full, [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    BATCH = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # --- model ---
    model = ImitationLearningModel(action_dim=7, hidden_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP params: {n_params/1e6:.1f}M")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    NUM_EPOCHS = 30  # multi-task has ~10x data, fewer epochs needed
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "test_loss": [], "lr": []}

    # --- train ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            imgs = batch["images"].to(device)
            acts = batch["actions"].to(device)
            pred = model(imgs)
            loss = criterion(pred, acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_ds)
        history["train_loss"].append(train_loss)
        history["lr"].append(scheduler.get_last_lr()[0])
        scheduler.step()

        # --- eval ---
        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["images"].to(device)
                acts = batch["actions"].to(device)
                pred = model(imgs)
                running += criterion(pred, acts).item() * imgs.size(0)
        test_loss = running / len(test_ds)
        history["test_loss"].append(test_loss)
        print(f"epoch {epoch+1:02d}  train {train_loss:.5f}  test {test_loss:.5f}  lr {history['lr'][-1]:.2e}")

    # --- save ---
    out = Path("outputs") / "model_mlp_multitask.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {"action_dim": 7, "hidden_dim": 256, "num_tasks": len(hdf5_paths)},
    }, out)
    print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
