"""
Download and load VLA model from Hugging Face Hub

Usage:
    python download_from_huggingface.py \
        --repo-id username/insertion-vla-regression-stage2 \
        --output-dir ./downloaded_models \
        --model-type regression \
        --training-stage stage2

Then use in your code:
    from models.model_with_sensor import Not_freeze_QwenVLAWithSensor
    model = Not_freeze_QwenVLAWithSensor(...)
    checkpoint = torch.load("./downloaded_models/pytorch_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
"""

import argparse
import json
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from typing import Optional


def download_model(
    repo_id: str,
    output_dir: Path,
    token: Optional[str] = None,
    revision: str = "main"
):
    """Download entire model repository from Hugging Face"""

    print(f"📥 Downloading model from: {repo_id}")
    print(f"   Revision: {revision}")
    print(f"   Output directory: {output_dir}")

    try:
        # Download entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=token
        )

        print(f"\n✅ Download complete!")
        print(f"   Local path: {local_dir}")

        # List downloaded files
        files = list(Path(local_dir).glob("*"))
        print(f"\n📂 Downloaded files:")
        for f in files:
            print(f"   - {f.name}")

        return Path(local_dir)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise


def load_model_info(model_dir: Path):
    """Load model configuration and info"""

    print(f"\n📖 Loading model information...")

    # Load config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"\n⚙️  Model Configuration:")
        print(f"   Type: {config.get('model_type', 'unknown')}")
        print(f"   Training Stage: {config.get('training_stage', 'unknown')}")
        print(f"   Base Model: {config.get('base_model', 'unknown')}")
        return config
    else:
        print(f"⚠️  config.json not found")
        return None


def create_loading_script(
    model_dir: Path,
    model_type: str,
    training_stage: str
):
    """Create example loading script"""

    script_path = model_dir / "load_model_example.py"

    if model_type == "diffusion":
        if training_stage == "stage2":
            model_class = "Not_freeze_QwenVLAWithSensorDiffusion"
            import_line = "from models.model_with_sensor_diffusion import Not_freeze_QwenVLAWithSensorDiffusion"
        else:
            model_class = "QwenVLAWithSensorDiffusion"
            import_line = "from models.model_with_sensor_diffusion import QwenVLAWithSensorDiffusion"
    else:
        if training_stage == "stage2":
            model_class = "Not_freeze_QwenVLAWithSensor"
            import_line = "from models.model_with_sensor import Not_freeze_QwenVLAWithSensor"
        else:
            model_class = "QwenVLAWithSensor"
            import_line = "from models.model_with_sensor import QwenVLAWithSensor"

    script = f'''"""
Example script to load downloaded VLA model
"""

import torch
{import_line}

# Initialize model
model = {model_class}(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    hidden_dim=1024,
    sensor_enabled=True,
    fusion_strategy='concat',
'''

    if model_type == "diffusion":
        script += "    diffusion_timesteps=100,\n"

    if training_stage == "stage2":
        script += '''    finetune_vl="lora",
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
'''

    script += ''')

# Load checkpoint
checkpoint = torch.load("pytorch_model.pt", map_location="cpu")

'''

    if training_stage == "stage1":
        script += '''# Load Stage 1 weights (Sensor + Action Expert only)
if model.sensor_encoder:
    model.sensor_encoder.load_state_dict(checkpoint["sensor_encoder"])
    print("✅ Loaded Sensor Encoder")

model.action_expert.load_state_dict(checkpoint["action_expert"])
print("✅ Loaded Action Expert")
'''
    else:
        script += '''# Load full model weights
model.load_state_dict(checkpoint["model_state_dict"])
print("✅ Loaded full model")
'''

    script += '''
# Move to GPU
model.eval()
model.to("cuda")

print("🚀 Model ready for inference!")

# Example inference
text_inputs = ["Insert the needle into the target carefully"]
image_inputs = [[
    "path/to/view1.jpg",
    "path/to/view2.jpg",
    "path/to/view3.jpg",
    "path/to/view4.jpg",
    "path/to/view5.jpg"
]]
sensor_data = torch.randn(1, 650, 1026).to("cuda")  # Example sensor data

with torch.no_grad():
'''

    if model_type == "diffusion":
        script += '''    # Diffusion model inference
    actions = model(
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        actions=None,  # Inference mode
        sensor_data=sensor_data,
        cache_keys=["sample_0"]
    )
    print(f"Predicted actions: {actions.shape}")  # (1, 8, 7)
'''
    else:
        script += '''    # Regression model inference
    z_chunk = torch.zeros(1, 8, 7).to("cuda")
    pred_actions, delta = model(
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        z_chunk=z_chunk,
        sensor_data=sensor_data,
        cache_keys=["sample_0"]
    )
    print(f"Predicted actions: {pred_actions.shape}")  # (1, 8, 7)
'''

    with open(script_path, "w") as f:
        f.write(script)

    print(f"\n📝 Created loading example: {script_path}")


def main():
    parser = argparse.ArgumentParser(description="Download VLA model from Hugging Face Hub")

    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repository ID (e.g., username/model-name)")
    parser.add_argument("--output-dir", type=str, default="./downloaded_models",
                        help="Directory to save downloaded model")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API token (for private repos)")
    parser.add_argument("--revision", type=str, default="main",
                        help="Repository revision/branch to download")
    parser.add_argument("--model-type", choices=["regression", "diffusion"], required=True,
                        help="Model type for loading script generation")
    parser.add_argument("--training-stage", choices=["stage1", "stage2"], required=True,
                        help="Training stage for loading script generation")

    args = parser.parse_args()

    # Download model
    output_dir = Path(args.output_dir) / args.repo_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = download_model(
        repo_id=args.repo_id,
        output_dir=output_dir,
        token=args.token,
        revision=args.revision
    )

    # Load and display model info
    config = load_model_info(model_dir)

    # Create loading script
    create_loading_script(
        model_dir=model_dir,
        model_type=args.model_type,
        training_stage=args.training_stage
    )

    print(f"\n✅ Model ready to use!")
    print(f"\nNext steps:")
    print(f"1. cd {model_dir}")
    print(f"2. python load_model_example.py")


if __name__ == "__main__":
    main()
