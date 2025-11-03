"""
Quick test script to verify dataset loading before training

This script tests:
1. Dataset paths are valid
2. Datasets can be loaded
3. Sample data can be accessed
"""

import sys
from pathlib import Path
import glob

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor
from vla_datasets.NewAsyncDataset import NewAsyncInsertionDataset

def test_dataset_loading():
    print("="*80)
    print("🧪 Testing Dataset Loading")
    print("="*80)

    datasets = []
    total_samples = 0

    # Priority datasets
    priority_dataset_dirs = [
        "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
        "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]

    print("\n📦 Loading Priority Datasets (2x weight)...")
    for pattern in priority_dataset_dirs:
        expanded_paths = glob.glob(pattern)
        print(f"   Pattern: {pattern}")
        print(f"   Found {len(expanded_paths)} trajectories")

        for traj_dir in expanded_paths[:2]:  # Test first 2 only
            try:
                ds = insertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=8,
                    sensor_window_size=65,
                )
                datasets.append(ds)
                total_samples += len(ds)
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                print(f"   ✅ {Path(traj_dir).name}: {len(ds)} samples ({sensor_status})")
            except Exception as e:
                print(f"   ❌ {Path(traj_dir).name}: {e}")

    # Regular datasets
    regular_dataset_dirs = [
        "/home/najo/NAS/VLA/dataset/OCT_insertion/Captures*",
        "/home/najo/NAS/VLA/dataset/part1/ZED_Captures_*th",
    ]

    print("\n📦 Loading Regular Datasets...")
    for pattern in regular_dataset_dirs:
        expanded_paths = glob.glob(pattern)
        print(f"   Pattern: {pattern}")
        print(f"   Found {len(expanded_paths)} trajectories")

        for traj_dir in expanded_paths[:2]:  # Test first 2 only
            try:
                ds = insertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=8,
                    sensor_window_size=65,
                )
                datasets.append(ds)
                total_samples += len(ds)
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                print(f"   ✅ {Path(traj_dir).name}: {len(ds)} samples ({sensor_status})")
            except Exception as e:
                print(f"   ❌ {Path(traj_dir).name}: {e}")

    # New async datasets
    new_dataset_root = Path("/home/najo/NAS/VLA/Insertion_VLA/Make_dataset/New_dataset")

    if new_dataset_root.exists():
        print(f"\n📦 Loading New Async Datasets...")
        print(f"   Root: {new_dataset_root}")

        count = 0
        for task_dir in new_dataset_root.iterdir():
            if not task_dir.is_dir():
                continue
            for episode_dir in task_dir.iterdir():
                if episode_dir.is_dir() and episode_dir.name.startswith("episode_"):
                    if count >= 2:  # Test first 2 only
                        break
                    try:
                        ds = NewAsyncInsertionDataset(episode_dir=episode_dir)
                        datasets.append(ds)
                        total_samples += len(ds)
                        print(f"   ✅ {task_dir.name}/{episode_dir.name}: {len(ds)} samples")
                        count += 1
                    except Exception as e:
                        print(f"   ❌ {task_dir.name}/{episode_dir.name}: {e}")
            if count >= 2:
                break

    print("\n" + "="*80)
    print(f"📊 Summary:")
    print(f"   Total datasets loaded: {len(datasets)}")
    print(f"   Total samples: {total_samples}")
    print("="*80)

    # Test sampling
    if datasets:
        print("\n🔍 Testing sample access...")
        sample = datasets[0][0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Images: {len(sample['images'])} views")
        print(f"   Actions shape: {sample['actions'].shape}")
        print(f"   Sensor data: {sample['sensor_data'].shape if sample['sensor_data'] is not None else 'None'}")
        print(f"   Instruction: {sample['instruction']}")
        print("   ✅ Sample access successful!")

    print("\n" + "="*80)
    print("✅ Dataset loading test complete!")
    print("="*80)

if __name__ == "__main__":
    test_dataset_loading()
