"""
09b_train_mlp_multitask_wrist.py — MLP baseline with TWO cameras (agentview + wrist).

Same as 09_train_mlp_multitask.py but uses 6-channel input
(agentview_rgb + eye_in_hand_rgb concatenated on channel dim).
ResNet18 first conv adapted from 3->6 channels (re-init).

Output: outputs/model_mlp_multitask_wrist.pt
Used to ablate "is observation info the bottleneck (vs DP architecture)".
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


def _load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ds_mod = _load_local("dataset02", Path(__file__).parent / "02_dataset.py")
make_multitask_dataset = ds_mod.make_multitask_dataset


# --- model: ResNet18 with 6-ch first conv ----------------------------------
class ImitationLearningModelWrist(nn.Module):
    def __init__(self, action_dim=7, hidden_dim=256, in_channels=6):
        super().__init__()
        self.backbone = resnet18(weights=None)
        # Replace first conv with 6-channel version (same kernel/stride/padding)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images):
        return self.head(self.backbone(images))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_dir = Path(get_libero_path("datasets")) / "libero_spatial"
    hdf5_paths = sorted(str(p) for p in dataset_dir.glob("*.hdf5"))
    print(f"Found {len(hdf5_paths)} hdf5 files")

    # KEY DIFFERENCE: two cameras concatenated on channel dim
    image_keys = ["agentview_rgb", "eye_in_hand_rgb"]
    full = make_multitask_dataset(hdf5_paths, seq_len=1,
                                  image_key=image_keys, use_state=False)

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

    model = ImitationLearningModelWrist(action_dim=7, hidden_dim=256, in_channels=6).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MLP+wrist params: {n_params/1e6:.1f}M")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    NUM_EPOCHS = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "test_loss": [], "lr": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            imgs = batch["images"].to(device)  # (B, 6, H, W)
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

    out = Path("outputs") / "model_mlp_multitask_wrist.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "action_dim": 7, "hidden_dim": 256,
            "num_tasks": len(hdf5_paths),
            "use_wrist": True,
            "in_channels": 6,
            "image_keys": image_keys,
        },
    }, out)
    print(f"\nSaved: {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
