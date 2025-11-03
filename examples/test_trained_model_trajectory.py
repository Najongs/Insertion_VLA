"""
Test script to evaluate trained VLA model on a demonstration trajectory

This script:
1. Loads a trained checkpoint (qwen_vla_sensor_best.pt or qwen_vla_sensor.pt)
2. Loads a demonstration from White_silicone_white_circle dataset
3. Generates predicted actions for the entire trajectory
4. Compares with ground truth actions
5. Visualizes results (trajectory plot, error metrics, sample images)

Usage:
    python examples/test_trained_model_trajectory.py \
        --checkpoint ./checkpoints/qwen_vla_sensor_best.pt \
        --demo-dir /home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170308 \
        --output-dir ./test_results
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_with_sensor import Not_freeze_QwenVLAWithSensor
from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor


def load_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load trained model from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model with same config as training
    # Force single GPU for inference
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = Not_freeze_QwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        finetune_vl="lora",  # Assuming LoRA was used
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        sensor_enabled=True,
        sensor_input_channels=1026,
        sensor_temporal_length=65,
        sensor_output_dim=3072,
        fusion_strategy="concat",
        image_resize_height=360,
        image_resize_width=640,
        device_map="cuda:0",  # Force single device
    ).to(device)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"✅ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    else:
        # Try loading directly
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded model weights (direct format)")

    model.eval()
    return model


def load_demo_trajectory(demo_dir: str, horizon: int = 8, sensor_window_size: int = 65):
    """
    Load a single demonstration trajectory

    Args:
        demo_dir: Path to demonstration directory
        horizon: Action horizon for the dataset
        sensor_window_size: Sensor window size (65 for 100ms @ 650Hz)

    Returns:
        dataset: Dataset object for the demonstration
    """
    print(f"📂 Loading demonstration from: {demo_dir}")

    dataset = insertionMeca500DatasetWithSensor(
        trajectory_dir=demo_dir,
        horizon=horizon,
        instruction="Approach the white square silicone",
        sensor_window_size=sensor_window_size,
        view_selection=['left', 'oak'],
        cache_sensor_windows=True  # Cache for faster loading
    )

    print(f"✅ Loaded {len(dataset)} timesteps")
    print(f"   Has sensor data: {dataset.has_sensor}")

    return dataset


def predict_trajectory(model, dataset, device: str = "cuda"):
    """
    Generate predictions for entire trajectory

    Args:
        model: Trained model
        dataset: Dataset containing the demonstration
        device: Device to run inference on

    Returns:
        predictions: List of predicted action sequences
        ground_truths: List of ground truth action sequences
        metadata: Additional info (images, sensor data, etc.)
    """
    print(f"\n🤖 Generating predictions for {len(dataset)} timesteps...")

    predictions = []
    ground_truths = []
    metadata = {
        'images': [],
        'sensor_data': [],
        'cache_keys': []
    }

    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Predicting"):
            sample = dataset[idx]

            # Prepare inputs
            images = sample['images']
            instruction = [sample['instruction']]
            gt_actions = sample['actions'].unsqueeze(0).to(device, dtype=torch.bfloat16)
            sensor_data = sample['sensor_data'].unsqueeze(0).to(device, dtype=torch.bfloat16) if sample['sensor_data'] is not None else None
            cache_key = [sample['cache_key']]

            # Run inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_actions, _ = model(
                    text_inputs=instruction,
                    image_inputs=[images],
                    z_chunk=gt_actions,  # Not used during inference, but required by forward()
                    cache_keys=cache_key,
                    sensor_data=sensor_data,
                )

            # Store results
            predictions.append(pred_actions.cpu().float().numpy()[0])  # [horizon, 7]
            ground_truths.append(gt_actions.cpu().float().numpy()[0])  # [horizon, 7]

            # Store metadata (only first few for visualization)
            if idx < 10:
                metadata['images'].append(images)
                metadata['cache_keys'].append(cache_key[0])
                if sensor_data is not None:
                    metadata['sensor_data'].append(sensor_data.cpu().float().numpy()[0])

    predictions = np.array(predictions)  # [T, horizon, 7]
    ground_truths = np.array(ground_truths)  # [T, horizon, 7]

    print(f"✅ Generated predictions: {predictions.shape}")

    return predictions, ground_truths, metadata


def compute_metrics(predictions, ground_truths):
    """
    Compute evaluation metrics

    Args:
        predictions: Predicted actions [T, horizon, 7]
        ground_truths: Ground truth actions [T, horizon, 7]

    Returns:
        metrics: Dictionary of metrics
    """
    print("\n📊 Computing metrics...")

    # MSE per dimension (averaged over timesteps and horizon)
    mse_per_dim = np.mean((predictions - ground_truths) ** 2, axis=(0, 1))

    # Overall MSE
    overall_mse = np.mean((predictions - ground_truths) ** 2)

    # MAE per dimension
    mae_per_dim = np.mean(np.abs(predictions - ground_truths), axis=(0, 1))

    # Overall MAE
    overall_mae = np.mean(np.abs(predictions - ground_truths))

    # Per-timestep error (for trajectory visualization)
    per_timestep_mse = np.mean((predictions - ground_truths) ** 2, axis=(1, 2))

    # Per-horizon error (first action vs full horizon)
    first_action_mse = np.mean((predictions[:, 0, :] - ground_truths[:, 0, :]) ** 2)
    full_horizon_mse = overall_mse

    # Correlation analysis (trend similarity)
    correlations = []
    for dim in range(predictions.shape[2]):  # For each action dimension
        # Use first action of each timestep for correlation
        pred_traj = predictions[:, 0, dim]
        gt_traj = ground_truths[:, 0, dim]
        if np.std(pred_traj) > 1e-6 and np.std(gt_traj) > 1e-6:
            corr = np.corrcoef(pred_traj, gt_traj)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)

    correlations = np.array(correlations)

    metrics = {
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'mse_per_dim': mse_per_dim,
        'mae_per_dim': mae_per_dim,
        'per_timestep_mse': per_timestep_mse,
        'first_action_mse': first_action_mse,
        'full_horizon_mse': full_horizon_mse,
        'correlations': correlations,
    }

    print(f"✅ Overall MSE: {overall_mse:.6f}")
    print(f"   Overall MAE: {overall_mae:.6f}")
    print(f"   First action MSE: {first_action_mse:.6f}")
    print(f"   Full horizon MSE: {full_horizon_mse:.6f}")
    print(f"   Per-dimension MSE: {mse_per_dim}")
    print(f"   Per-dimension MAE: {mae_per_dim}")
    print(f"   Correlations (trend): {correlations}")

    return metrics


def visualize_results(predictions, ground_truths, metrics, metadata, output_dir: Path):
    print(f"\n📈 Creating visualizations in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    action_names = ['dx', 'dy', 'dz', 'da', 'db', 'dr', 'gripper']

    # ✅ 1. Combined Position + Orientation Trajectory (3x2 layout)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    pos_dims = [0, 1, 2]
    ori_dims = [3, 4, 5]
    pos_names = ['dx (X-axis)', 'dy (Y-axis)', 'dz (Z-axis)']
    ori_names = ['da (Alpha)', 'db (Beta)', 'dr (Gamma)']

    for i in range(3):
        # Position subplot (left column)
        ax_pos = axes[i, 0]
        gt_pos = ground_truths[:, 0, pos_dims[i]]
        pred_pos = predictions[:, 0, pos_dims[i]]
        timesteps = np.arange(len(gt_pos))
        ax_pos.plot(timesteps, gt_pos, 'b-', label='GT', linewidth=2.8, marker='o', markersize=4, markevery=5)
        ax_pos.plot(timesteps, pred_pos, 'r--', label='Pred', linewidth=2.8, marker='s', markersize=4, markevery=5, alpha=0.8)
        ax_pos.set_title(pos_names[i], fontsize=17, fontweight='bold')
        ax_pos.set_ylabel('Δ Position', fontsize=16)
        ax_pos.grid(True, alpha=0.3)
        if i == 2:
            ax_pos.set_xlabel('Timestep', fontsize=15)
        if i == 0:
            ax_pos.legend(fontsize=14, loc='best')

        # Orientation subplot (right column)
        ax_ori = axes[i, 1]
        gt_ori = ground_truths[:, 0, ori_dims[i]]
        pred_ori = predictions[:, 0, ori_dims[i]]
        timesteps = np.arange(len(gt_ori))
        ax_ori.plot(timesteps, gt_ori, 'b-', label='GT', linewidth=2.8, marker='o', markersize=4, markevery=5)
        ax_ori.plot(timesteps, pred_ori, 'r--', label='Pred', linewidth=2.8, marker='s', markersize=4, markevery=5, alpha=0.8)
        ax_ori.set_title(ori_names[i], fontsize=17, fontweight='bold')
        ax_ori.set_ylabel('Δ Orientation', fontsize=16)
        ax_ori.grid(True, alpha=0.3)
        if i == 2:
            ax_ori.set_xlabel('Timestep', fontsize=15)
        if i == 0:
            ax_ori.legend(fontsize=14, loc='best')

    plt.suptitle('Pose Trajectory (Position + Orientation)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_pose_combined.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: trajectory_pose_combined.png")
    plt.close()

    # 2. Plot per-timestep MSE
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps = np.arange(len(metrics['per_timestep_mse']))
    ax.plot(timesteps, metrics['per_timestep_mse'], 'g-', linewidth=2)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('MSE')
    ax.set_title('Per-Timestep Mean Squared Error')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_timestep_error.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: per_timestep_error.png")
    plt.close()

    # 3. Bar plot of per-dimension errors and correlations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(7)
    ax1.bar(x, metrics['mse_per_dim'], color='steelblue', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_names, rotation=45)
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error per Action Dimension')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, metrics['mae_per_dim'], color='coral', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(action_names, rotation=45)
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error per Action Dimension')
    ax2.grid(True, alpha=0.3, axis='y')

    # Correlation plot (trend similarity)
    colors = ['green' if c > 0.8 else 'orange' if c > 0.5 else 'red' for c in metrics['correlations']]
    ax3.bar(x, metrics['correlations'], color=colors, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(action_names, rotation=45)
    ax3.set_ylabel('Correlation')
    ax3.set_ylim(-1, 1)
    ax3.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good (>0.8)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (>0.5)')
    ax3.set_title('Correlation (Trend Similarity)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'error_per_dimension.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: error_per_dimension.png")
    plt.close()

    # 4. Full horizon comparison (show all 8 actions for selected timesteps)
    # Select 4 representative timesteps
    T = predictions.shape[0]
    selected_timesteps = [0, T//3, 2*T//3, T-1]

    fig, axes = plt.subplots(4, 7, figsize=(21, 12))

    for row_idx, t in enumerate(selected_timesteps):
        for dim in range(7):
            ax = axes[row_idx, dim]

            # Plot horizon for this timestep and dimension
            horizon_steps = np.arange(predictions.shape[1])
            gt_horizon = ground_truths[t, :, dim]
            pred_horizon = predictions[t, :, dim]

            ax.plot(horizon_steps, gt_horizon, 'b-o', label='GT', linewidth=2, markersize=4)
            ax.plot(horizon_steps, pred_horizon, 'r--s', label='Pred', linewidth=2, markersize=4, alpha=0.7)

            if row_idx == 0:
                ax.set_title(f'{action_names[dim]}', fontsize=10)
            if dim == 0:
                ax.set_ylabel(f't={t}', fontsize=10)
            if row_idx == len(selected_timesteps) - 1:
                ax.set_xlabel('Horizon', fontsize=8)

            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='best')

    plt.suptitle('Full Horizon Comparison (Selected Timesteps)', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'horizon_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: horizon_comparison.png")
    plt.close()

    # 5. Sample images (if available)
    if metadata['images']:
        import cv2

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < len(metadata['images']):
                # Load first image from the view
                img_uri = metadata['images'][i][0]  # First view
                img_path = img_uri.replace('file://', '')

                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img)
                        ax.set_title(f'Timestep {i}', fontsize=10)
                        ax.axis('off')
                    else:
                        ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                        ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}', ha='center', va='center', fontsize=8)
                    ax.axis('off')
            else:
                ax.axis('off')

        plt.suptitle('Sample Images from Trajectory', fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_images.png', dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: sample_images.png")
        plt.close()

    # 6. Save metrics to JSON
    metrics_json = {
        'overall_mse': float(metrics['overall_mse']),
        'overall_mae': float(metrics['overall_mae']),
        'first_action_mse': float(metrics['first_action_mse']),
        'full_horizon_mse': float(metrics['full_horizon_mse']),
        'mse_per_dim': metrics['mse_per_dim'].tolist(),
        'mae_per_dim': metrics['mae_per_dim'].tolist(),
        'correlations': metrics['correlations'].tolist(),
        'action_names': action_names,
        'interpretation': {
            'mse': 'Lower is better - measures absolute error magnitude',
            'mae': 'Lower is better - measures average absolute error',
            'correlation': 'Higher is better (0~1) - measures trend similarity. >0.8 is good, >0.5 is fair',
            'note': 'High correlation with high MSE means the model captures the trend but has scale/offset issues'
        }
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Saved: metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Test trained VLA model on demonstration trajectory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/qwen_vla_sensor_best.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default="/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_20251027_170458",
        help="Path to demonstration directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=8,
        help="Action horizon"
    )
    parser.add_argument(
        "--sensor-window-size",
        type=int,
        default=65,
        help="Sensor window size (65 = 100ms @ 650Hz)"
    )

    args = parser.parse_args()

    print("="*80)
    print("🧪 Testing Trained VLA Model on Demonstration Trajectory")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Demo directory: {args.demo_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80)

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Available checkpoints:")
        ckpt_dir = Path("./checkpoints")
        if ckpt_dir.exists():
            for ckpt in ckpt_dir.glob("*.pt"):
                print(f"   - {ckpt.name}")
        return

    # Check if demo directory exists
    if not Path(args.demo_dir).exists():
        print(f"❌ Demo directory not found: {args.demo_dir}")
        return

    # Load model
    model = load_model(args.checkpoint, device=args.device)

    # Load demonstration
    dataset = load_demo_trajectory(
        args.demo_dir,
        horizon=args.horizon,
        sensor_window_size=args.sensor_window_size
    )

    # Generate predictions
    predictions, ground_truths, metadata = predict_trajectory(
        model, dataset, device=args.device
    )

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Visualize results
    output_dir = Path(args.output_dir)
    visualize_results(predictions, ground_truths, metrics, metadata, output_dir)

    print("\n" + "="*80)
    print("✅ Testing complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
