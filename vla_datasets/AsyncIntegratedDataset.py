"""
Async Integrated Dataset for VLA Training with Asynchronous VLM and Action Expert

Key Features:
- VLM runs at ~3.33 Hz (300ms period)
- Action Expert runs at 10 Hz (100ms period)
- VL features are reused 3x before updating
- Sensor windows are 65 samples (100ms @ 650Hz)

This dataset structure enables training the model to handle asynchronous
VLM and sensor inputs, which is critical for real-time inference.
"""

import os
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader


class AsyncInsertionMeca500DatasetWithSensor(Dataset):
    """
    Async version of insertionMeca500DatasetWithSensor

    Generates samples with asynchronous VLM and sensor windows:
    - Each VLM feature is paired with multiple sensor windows
    - Sensor windows are 65 timesteps (100ms @ 650Hz)
    - Action expert operates at 10Hz

    Args:
        trajectory_dir: Path to trajectory directory
        horizon: Action horizon (default: 8)
        vlm_reuse_count: How many times to reuse each VL feature (default: 3)
        sensor_window_size: Sensor window size in timesteps (default: 65)
        action_expert_hz: Action expert frequency (default: 10)
    """

    def __init__(
        self,
        trajectory_dir,
        horizon=8,
        vlm_reuse_count=3,
        sensor_window_size=65,
        action_expert_hz=10,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.horizon = horizon
        self.vlm_reuse_count = vlm_reuse_count
        self.sensor_window_size = sensor_window_size
        self.action_expert_hz = action_expert_hz

        # Calculate temporal parameters
        self.sensor_hz = 650  # Sensor sampling rate
        self.vlm_period_ms = 1000.0 / (action_expert_hz / vlm_reuse_count)  # ~300ms
        self.action_period_ms = 1000.0 / action_expert_hz  # 100ms

        # Load trajectory data
        self.data_file = self.trajectory_dir / "data.pkl"
        if not self.data_file.exists():
            raise FileNotFoundError(f"data.pkl not found in {self.trajectory_dir}")

        with open(self.data_file, 'rb') as f:
            self.data = pickle.load(f)

        # Extract components
        self.actions = np.array(self.data["action"], dtype=np.float32)  # (T, 7)
        self.images = self.data.get("image", {})

        # Check if sensor data exists
        self.has_sensor = "sensor_data" in self.data and self.data["sensor_data"] is not None

        if self.has_sensor:
            sensor_raw = self.data["sensor_data"]
            if isinstance(sensor_raw, dict):
                # Extract FPI and force data
                fpi_data = sensor_raw.get("fpi", np.zeros((len(self.actions), 1025)))
                force_data = sensor_raw.get("force", np.zeros((len(self.actions), 1)))

                # Ensure correct shapes
                if fpi_data.shape[0] != len(self.actions):
                    fpi_data = np.zeros((len(self.actions), 1025))
                if force_data.shape[0] != len(self.actions):
                    force_data = np.zeros((len(self.actions), 1))

                # Concatenate: (T, 1026) = (T, 1 force + 1025 FPI)
                self.sensor_data = np.concatenate([force_data, fpi_data], axis=-1).astype(np.float32)
            else:
                self.sensor_data = np.array(sensor_raw, dtype=np.float32)
                if self.sensor_data.shape[-1] != 1026:
                    # Pad or create dummy sensor data
                    self.sensor_data = np.zeros((len(self.actions), 1026), dtype=np.float32)
        else:
            # Create dummy sensor data for datasets without sensor
            self.sensor_data = np.zeros((len(self.actions), 1026), dtype=np.float32)

        # Build async sample indices
        self.samples = self._build_async_samples()

        # Instructions
        self.instruction = self.data.get("instruction", "Perform needle insertion task.")

        print(f"📦 AsyncDataset: {self.trajectory_dir.name}")
        print(f"   Total timesteps: {len(self.actions)}")
        print(f"   Async samples: {len(self.samples)}")
        print(f"   Has sensor: {self.has_sensor}")
        print(f"   VLM reuse: {self.vlm_reuse_count}x")
        print(f"   Sensor window: {self.sensor_window_size} samples")

    def _build_async_samples(self):
        """
        Build async sample indices

        Each sample contains:
        - vlm_idx: Index for VLM feature extraction (image timestamp)
        - sensor_start_idx: Start index for sensor window
        - action_start_idx: Start index for action chunk
        - reuse_step: Which reuse step (0, 1, or 2)
        """
        samples = []

        total_timesteps = len(self.actions)

        # Calculate how many action expert steps we can fit
        action_step_size = self.horizon  # Each action chunk has 'horizon' actions
        max_action_steps = (total_timesteps - action_step_size) // action_step_size

        # For each VLM update position
        vlm_step = 0
        action_step = 0

        while action_step < max_action_steps:
            # VLM index (image at this timestep)
            vlm_idx = min(action_step * action_step_size, total_timesteps - 1)

            # Generate samples for each reuse of this VLM feature
            for reuse_step in range(self.vlm_reuse_count):
                if action_step >= max_action_steps:
                    break

                # Calculate sensor window indices
                sensor_start_idx = action_step * action_step_size
                sensor_end_idx = sensor_start_idx + self.sensor_window_size

                # Calculate action chunk indices
                action_start_idx = action_step * action_step_size
                action_end_idx = action_start_idx + self.horizon

                # Check if we have enough data
                if sensor_end_idx > len(self.sensor_data):
                    break
                if action_end_idx > len(self.actions):
                    break

                samples.append({
                    'vlm_idx': vlm_idx,
                    'sensor_start_idx': sensor_start_idx,
                    'sensor_end_idx': sensor_end_idx,
                    'action_start_idx': action_start_idx,
                    'action_end_idx': action_end_idx,
                    'reuse_step': reuse_step,
                })

                action_step += 1

            vlm_step += 1

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get VLM image paths (multi-view)
        vlm_idx = sample['vlm_idx']
        image_paths = []
        for view_name in sorted(self.images.keys()):
            view_images = self.images[view_name]
            if vlm_idx < len(view_images):
                image_paths.append(view_images[vlm_idx])
            else:
                image_paths.append(view_images[-1])  # Use last image if out of range

        # Get sensor window
        sensor_window = self.sensor_data[
            sample['sensor_start_idx']:sample['sensor_end_idx']
        ]  # (sensor_window_size, 1026)

        # Pad if needed (for edge cases)
        if len(sensor_window) < self.sensor_window_size:
            padding = np.zeros((self.sensor_window_size - len(sensor_window), 1026), dtype=np.float32)
            sensor_window = np.concatenate([sensor_window, padding], axis=0)

        # Get action chunk
        actions = self.actions[
            sample['action_start_idx']:sample['action_end_idx']
        ]  # (horizon, 7)

        # Pad if needed
        if len(actions) < self.horizon:
            padding = np.zeros((self.horizon - len(actions), 7), dtype=np.float32)
            actions = np.concatenate([actions, padding], axis=0)

        return {
            'instruction': self.instruction,
            'images': image_paths,
            'sensor_data': torch.from_numpy(sensor_window),  # (sensor_window_size, 1026)
            'actions': torch.from_numpy(actions),  # (horizon, 7)
            'has_sensor': self.has_sensor,
            'cache_key': f"{self.trajectory_dir.name}_vlm{vlm_idx}_step{idx}",
            'vlm_idx': vlm_idx,
            'reuse_step': sample['reuse_step'],
        }


def async_collate_fn_with_sensor(batch):
    """
    Collate function for async training
    """
    instructions = [item['instruction'] for item in batch]
    image_lists = [item['images'] for item in batch]

    sensor_data = torch.stack([item['sensor_data'] for item in batch])  # (B, sensor_window_size, 1026)
    actions = torch.stack([item['actions'] for item in batch])  # (B, horizon, 7)

    has_sensor_mask = torch.tensor([item['has_sensor'] for item in batch], dtype=torch.bool)
    cache_keys = [item['cache_key'] for item in batch]
    vlm_indices = [item['vlm_idx'] for item in batch]
    reuse_steps = [item['reuse_step'] for item in batch]

    # Confidence: lower confidence for datasets without sensor
    confidence = [1.0 if item['has_sensor'] else 0.5 for item in batch]

    return {
        'instruction': instructions,
        'images': image_lists,
        'sensor_data': sensor_data,
        'actions': actions,
        'has_sensor_mask': has_sensor_mask,
        'cache_keys': cache_keys,
        'vlm_indices': vlm_indices,
        'reuse_steps': reuse_steps,
        'confidence': confidence,
    }


def create_async_integrated_dataloader(
    dataset_patterns,
    batch_size=1,
    num_workers=4,
    shuffle=True,
    horizon=8,
    vlm_reuse_count=3,
    sensor_window_size=65,
):
    """
    Create async integrated dataloader from multiple dataset patterns

    Args:
        dataset_patterns: List of dataset directory patterns
        batch_size: Batch size
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle
        horizon: Action horizon
        vlm_reuse_count: VLM feature reuse count
        sensor_window_size: Sensor window size in timesteps
    """
    import glob

    datasets = []

    for pattern in dataset_patterns:
        expanded_paths = glob.glob(pattern)
        for traj_dir in expanded_paths:
            try:
                ds = AsyncInsertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=horizon,
                    vlm_reuse_count=vlm_reuse_count,
                    sensor_window_size=sensor_window_size,
                )
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Failed to load {traj_dir}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    # Concatenate all datasets
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    print(f"\n📊 Total async dataset size: {len(full_dataset)} samples")

    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=async_collate_fn_with_sensor,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False,
    )

    return dataloader


if __name__ == "__main__":
    # Test async dataset
    print("🧪 Testing AsyncIntegratedDataset...")

    test_dir = "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308"

    if Path(test_dir).exists():
        dataset = AsyncInsertionMeca500DatasetWithSensor(
            trajectory_dir=test_dir,
            horizon=8,
            vlm_reuse_count=3,
            sensor_window_size=65,
        )

        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")

        # Test first sample
        sample = dataset[0]
        print(f"\n📦 First sample:")
        print(f"   Instruction: {sample['instruction']}")
        print(f"   Images: {len(sample['images'])} views")
        print(f"   Sensor shape: {sample['sensor_data'].shape}")
        print(f"   Actions shape: {sample['actions'].shape}")
        print(f"   Has sensor: {sample['has_sensor']}")
        print(f"   VLM idx: {sample['vlm_idx']}")
        print(f"   Reuse step: {sample['reuse_step']}")

        # Test dataloader
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=async_collate_fn_with_sensor,
        )

        batch = next(iter(loader))
        print(f"\n📦 First batch:")
        print(f"   Batch sensor shape: {batch['sensor_data'].shape}")
        print(f"   Batch actions shape: {batch['actions'].shape}")
        print(f"   VLM indices: {batch['vlm_indices']}")
        print(f"   Reuse steps: {batch['reuse_steps']}")
    else:
        print(f"⚠️ Test directory not found: {test_dir}")
