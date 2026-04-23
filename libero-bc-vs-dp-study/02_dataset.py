"""
PyTorch Dataset for LIBERO manipulation tasks.

Loads trajectories from HDF5 files and supports sequence sampling for imitation learning.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json


class LIBERODataset(Dataset):
    """
    LIBERO dataset loader with sequence sampling support.
    
    Each sample is (image_obs, state_obs, action) for a single timestep,
    or (image_seq, state_seq, action_seq) for sequence sampling.
    
    Args:
        hdf5_path: Path to a single HDF5 file
        seq_len: If > 1, sample consecutive seq_len frames
        image_key: Which image to use ('agentview_rgb' or 'eye_in_hand_rgb')
        use_state: Whether to include robot state (ee_pos, ee_ori, joint_states, gripper)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        seq_len: int = 1,
        image_key="agentview_rgb",
        use_state: bool = True,
        image_size: Tuple[int, int] = (128, 128),
        cache_in_ram: bool = True,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.seq_len = seq_len
        # Normalize image_key to a list internally
        if isinstance(image_key, str):
            self.image_keys = [image_key]
        else:
            self.image_keys = list(image_key)
        self.image_key = self.image_keys[0]  # back-compat attr
        self.use_state = use_state
        self.image_size = image_size
        self.cache_in_ram = cache_in_ram
        self._cache: Dict[int, Dict] = {}
        
        # Load HDF5 structure to get demo names and lengths
        with h5py.File(self.hdf5_path, 'r') as f:
            self.demo_names = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            self.demo_lengths = {}
            self.episode_indices = []
            
            for i, demo_name in enumerate(self.demo_names):
                traj_len = len(f['data'][demo_name]['actions'])
                self.demo_lengths[demo_name] = traj_len
                
                # For each demo, record all valid (demo_idx, frame_idx) pairs
                # Valid means: frame_idx in [0, traj_len - seq_len)
                for frame_idx in range(traj_len - self.seq_len + 1):
                    self.episode_indices.append((i, frame_idx))
        
        self.n_samples = len(self.episode_indices)
        print(f"Loaded {len(self.demo_names)} demos from {self.hdf5_path.name}")
        print(f"  Total valid samples (seq_len={seq_len}): {self.n_samples}")
        
        if self.cache_in_ram:
            print(f"  Pre-loading all demos into RAM...")
            for i in range(len(self.demo_names)):
                self._cache[i] = self._load_demo_data_from_disk(i)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def _load_demo_data_from_disk(self, demo_idx: int) -> Dict:
        """Load a single demo from HDF5 (no caching)."""
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_name = self.demo_names[demo_idx]
            group = f['data'][demo_name]

            # Stack all requested cameras along channel dim later in __getitem__.
            # Here we just keep them as a dict per camera.
            data = {
                'images_per_cam': {k: group['obs'][k][:] for k in self.image_keys},
                'actions': group['actions'][:],
            }
            
            if self.use_state:
                obs_group = group['obs']
                data['ee_pos'] = obs_group['ee_pos'][:] if 'ee_pos' in obs_group else None
                data['ee_ori'] = obs_group['ee_ori'][:] if 'ee_ori' in obs_group else None
                data['joint_states'] = obs_group['joint_states'][:] if 'joint_states' in obs_group else None
                data['gripper'] = obs_group['gripper'][:] if 'gripper' in obs_group else None
        
        return data
    
    def _load_demo_data(self, demo_idx: int) -> Dict:
        if self.cache_in_ram and demo_idx in self._cache:
            return self._cache[demo_idx]
        return self._load_demo_data_from_disk(demo_idx)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample. Returns dict with keys:
        - 'images': (seq_len, 3, H, W) or (3, H, W) if seq_len=1
        - 'actions': (seq_len, 7) or (7,) if seq_len=1
        - 'state': concatenated state features, optional
        """
        demo_idx, frame_idx = self.episode_indices[idx]
        data = self._load_demo_data(demo_idx)

        # Extract sequence
        frame_indices = np.arange(frame_idx, frame_idx + self.seq_len)

        # Images: per-camera, flip vertically (MuJoCo convention), then concat on channel dim.
        cam_tensors = []
        for k in self.image_keys:
            imgs_k = data['images_per_cam'][k][frame_indices]  # (seq_len, H, W, 3)
            imgs_k = imgs_k[:, ::-1, :, :].copy()
            t = torch.from_numpy(imgs_k).permute(0, 3, 1, 2).float() / 255.0  # (seq_len, 3, H, W)
            cam_tensors.append(t)
        images = torch.cat(cam_tensors, dim=1)  # (seq_len, 3*K, H, W)

        # Actions
        actions = data['actions'][frame_indices]  # (seq_len, 7)
        actions = torch.from_numpy(actions).float()
        
        result = {'images': images, 'actions': actions}
        
        # Optional: state
        if self.use_state:
            state_parts = []
            if data['ee_pos'] is not None:
                state_parts.append(data['ee_pos'][frame_indices])
            if data['ee_ori'] is not None:
                state_parts.append(data['ee_ori'][frame_indices])
            if data['joint_states'] is not None:
                state_parts.append(data['joint_states'][frame_indices])
            if data['gripper'] is not None:
                state_parts.append(data['gripper'][frame_indices])
            
            if state_parts:
                state = np.concatenate(state_parts, axis=-1)  # (seq_len, state_dim)
                result['state'] = torch.from_numpy(state).float()
        
        # Squeeze sequence dimension if seq_len=1
        if self.seq_len == 1:
            result['images'] = result['images'].squeeze(0)
            result['actions'] = result['actions'].squeeze(0)
            if 'state' in result:
                result['state'] = result['state'].squeeze(0)
        
        return result


def make_multitask_dataset(
    hdf5_paths: List[str],
    seq_len: int = 1,
    image_key="agentview_rgb",
    use_state: bool = False,
    image_size: Tuple[int, int] = (128, 128),
    cache_in_ram: bool = True,
) -> "torch.utils.data.ConcatDataset":
    """Concatenate per-file LIBERODataset instances into one multi-task dataset.
    image_key may be a str or a list of camera keys (channels concatenated)."""
    from torch.utils.data import ConcatDataset
    subsets = [
        LIBERODataset(p, seq_len=seq_len, image_key=image_key,
                      use_state=use_state, image_size=image_size,
                      cache_in_ram=cache_in_ram)
        for p in hdf5_paths
    ]
    total = sum(len(s) for s in subsets)
    print(f"[multi-task] {len(subsets)} tasks combined, total samples: {total}")
    return ConcatDataset(subsets)


def create_dataloader(
    hdf5_paths: List[str],
    batch_size: int = 32,
    seq_len: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader combining multiple HDF5 files.
    
    Args:
        hdf5_paths: List of paths to HDF5 files
        batch_size: Batch size
        seq_len: Sequence length
        shuffle: Whether to shuffle
        num_workers: Number of workers (set to 0 for Windows WSL)
        **dataset_kwargs: Additional args for LIBERODataset
    
    Returns:
        DataLoader
    """
    # For simplicity, use first HDF5 file; can be extended to combine multiple
    if isinstance(hdf5_paths, str):
        hdf5_paths = [hdf5_paths]
    
    dataset = LIBERODataset(
        hdf5_paths[0],
        seq_len=seq_len,
        **dataset_kwargs,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
    return loader


if __name__ == "__main__":
    # Quick test: load one demo file
    from libero.libero import get_libero_path
    
    # Get path to a demo file
    dataset_dir = Path(get_libero_path("datasets")) / "libero_spatial"
    demo_files = sorted(dataset_dir.glob("*.hdf5"))
    
    if demo_files:
        print(f"Found {len(demo_files)} demo files")
        
        # Test single-frame sampling
        print("\n=== Testing seq_len=1 ===")
        ds1 = LIBERODataset(str(demo_files[0]), seq_len=1)
        sample1 = ds1[0]
        print(f"Images shape: {sample1['images'].shape}")  # (3, 128, 128)
        print(f"Actions shape: {sample1['actions'].shape}")  # (7,)
        print(f"Action values: {sample1['actions']}")
        
        # Test sequence sampling
        print("\n=== Testing seq_len=5 ===")
        ds5 = LIBERODataset(str(demo_files[0]), seq_len=5)
        sample5 = ds5[0]
        print(f"Images shape: {sample5['images'].shape}")  # (5, 3, 128, 128)
        print(f"Actions shape: {sample5['actions'].shape}")  # (5, 7)
        
        # Test DataLoader
        print("\n=== Testing DataLoader ===")
        loader = create_dataloader(str(demo_files[0]), batch_size=4, seq_len=5)
        batch = next(iter(loader))
        print(f"Batch images shape: {batch['images'].shape}")  # (4, 5, 3, 128, 128)
        print(f"Batch actions shape: {batch['actions'].shape}")  # (4, 5, 7)
    else:
        print("No HDF5 files found. Check dataset path.")
