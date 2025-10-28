"""
Example: Load model from Hugging Face and run inference

This example shows how to:
1. Download model from Hugging Face
2. Load checkpoint
3. Run inference with sample data

Usage:
    python examples/huggingface_inference_example.py \
        --repo-id your-username/insertion-vla-diffusion-stage2 \
        --model-type diffusion \
        --training-stage stage2
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def download_model_if_needed(repo_id: str, output_dir: Path, token: str = None):
    """Download model from Hugging Face if not already present"""
    from huggingface_hub import snapshot_download

    model_dir = output_dir / repo_id.replace("/", "_")

    if model_dir.exists() and (model_dir / "pytorch_model.pt").exists():
        print(f"✅ Model already downloaded: {model_dir}")
        return model_dir

    print(f"📥 Downloading model from {repo_id}...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        token=token
    )
    print(f"✅ Downloaded to: {local_dir}")
    return Path(local_dir)


def load_model(model_dir: Path, model_type: str, training_stage: str):
    """Load model from downloaded checkpoint"""

    print(f"\n🔧 Loading {model_type} model ({training_stage})...")

    checkpoint_path = model_dir / "pytorch_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Import appropriate model class
    if model_type == "diffusion":
        if training_stage == "stage2":
            from models.model_with_sensor_diffusion import Not_freeze_QwenVLAWithSensorDiffusion
            model = Not_freeze_QwenVLAWithSensorDiffusion(
                vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                action_dim=7,
                horizon=8,
                sensor_enabled=True,
                diffusion_timesteps=100,
                finetune_vl="lora",
                lora_r=16,
                lora_alpha=32,
            )
        else:
            from models.model_with_sensor_diffusion import QwenVLAWithSensorDiffusion
            model = QwenVLAWithSensorDiffusion(
                vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                action_dim=7,
                horizon=8,
                sensor_enabled=True,
                diffusion_timesteps=100,
            )
    else:  # regression
        if training_stage == "stage2":
            from models.model_with_sensor import Not_freeze_QwenVLAWithSensor
            model = Not_freeze_QwenVLAWithSensor(
                vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                action_dim=7,
                horizon=8,
                sensor_enabled=True,
                finetune_vl="lora",
                lora_r=16,
                lora_alpha=32,
            )
        else:
            from models.model_with_sensor import QwenVLAWithSensor
            model = QwenVLAWithSensor(
                vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                action_dim=7,
                horizon=8,
                sensor_enabled=True,
            )

    # Load checkpoint
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if training_stage == "stage1":
        # Stage 1: Load sensor_encoder and action_expert
        if model.sensor_encoder and "sensor_encoder" in checkpoint:
            model.sensor_encoder.load_state_dict(checkpoint["sensor_encoder"])
            print("   ✅ Loaded Sensor Encoder")

        if "action_expert" in checkpoint:
            model.action_expert.load_state_dict(checkpoint["action_expert"])
            print("   ✅ Loaded Action Expert")
    else:
        # Stage 2: Load full model
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("   ✅ Loaded full model")
        else:
            raise ValueError("Stage 2 checkpoint should contain 'model_state_dict'")

    model.eval()
    model.to("cuda")
    print("✅ Model loaded and ready for inference!")

    return model


def run_inference_example(model, model_type: str):
    """Run inference with sample data"""

    print("\n🚀 Running inference with sample data...")

    # Prepare sample inputs
    text_inputs = ["Insert the needle into the target carefully"]

    # Note: In real usage, replace with actual image paths
    # For this example, we use placeholder paths
    image_inputs = [[
        "path/to/view1.jpg",
        "path/to/view2.jpg",
        "path/to/view3.jpg",
        "path/to/view4.jpg",
        "path/to/view5.jpg"
    ]]

    # Sample sensor data (B, T, C) where T=650, C=1026
    sensor_data = torch.randn(1, 650, 1026).to("cuda")

    # Run inference
    with torch.no_grad():
        if model_type == "diffusion":
            # Diffusion model: returns sampled actions
            actions = model(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                actions=None,  # Inference mode
                sensor_data=sensor_data,
                cache_keys=["sample_0"]
            )
            print(f"✅ Predicted actions: {actions.shape}")  # (1, 8, 7)
            print(f"   Sample action values:\n{actions[0, :3, :3]}")  # First 3 timesteps, 3 joints
        else:
            # Regression model: returns predicted actions and delta
            z_chunk = torch.zeros(1, 8, 7).to("cuda")
            pred_actions, delta = model(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                z_chunk=z_chunk,
                sensor_data=sensor_data,
                cache_keys=["sample_0"]
            )
            print(f"✅ Predicted actions: {pred_actions.shape}")  # (1, 8, 7)
            print(f"   Sample action values:\n{pred_actions[0, :3, :3]}")  # First 3 timesteps, 3 joints

    print("\n✅ Inference completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Load and run inference with Hugging Face model")

    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repository ID")
    parser.add_argument("--model-type", choices=["regression", "diffusion"], required=True,
                        help="Model type")
    parser.add_argument("--training-stage", choices=["stage1", "stage2"], required=True,
                        help="Training stage")
    parser.add_argument("--output-dir", type=str, default="./downloaded_models",
                        help="Directory for downloaded models")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token (for private repos)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference (just download and load)")

    args = parser.parse_args()

    try:
        # Download model if needed
        model_dir = download_model_if_needed(
            repo_id=args.repo_id,
            output_dir=Path(args.output_dir),
            token=args.token
        )

        # Load model
        model = load_model(
            model_dir=model_dir,
            model_type=args.model_type,
            training_stage=args.training_stage
        )

        # Run inference (unless skipped)
        if not args.skip_inference:
            run_inference_example(model, args.model_type)
        else:
            print("\n⏭️  Skipping inference (--skip-inference flag set)")

        print("\n✅ All done!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
