"""
Upload trained VLA model to Hugging Face Hub

Supports both Stage 1 (Sensor+Action) and Stage 2 (Full model with LoRA) checkpoints.

Usage:
    python upload_to_huggingface.py \
        --checkpoint checkpoints/qwen_vla_sensor_best.pt \
        --model-type regression \
        --training-stage stage2 \
        --repo-id your-username/insertion-vla-regression-stage2 \
        --token YOUR_HF_TOKEN

Requirements:
    pip install huggingface_hub
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo, upload_folder
from typing import Dict, Any


def create_model_card(
    model_type: str,
    training_stage: str,
    checkpoint_info: Dict[str, Any],
    performance_metrics: Dict[str, float] = None
) -> str:
    """Generate model card (README.md) for Hugging Face"""

    stage_desc = {
        "stage1": "Stage 1: Sensor Encoder + Action Expert (VL frozen)",
        "stage2": "Stage 2: Full model with LoRA fine-tuning"
    }

    model_desc = {
        "regression": "Regression-based action prediction",
        "diffusion": "Diffusion policy (DDPM/DDIM) for action prediction"
    }

    card = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
tags:
- vision-language-action
- robotics
- sensor-fusion
- {"diffusion" if model_type == "diffusion" else "regression"}
- lora
- multi-modal
pipeline_tag: robotics
---

# Insertion VLA - {model_desc.get(model_type, model_type).title()}

**{stage_desc.get(training_stage, training_stage)}**

## Model Description

This is a Vision-Language-Action (VLA) model for robotic needle insertion with OCT/FPI sensor integration.

### Architecture

- **Base Model**: Qwen2.5-VL-3B-Instruct (frozen in Stage 1, LoRA fine-tuned in Stage 2)
- **Sensor Encoder**: 1D CNN + Transformer for OCT/FPI data (650Hz, 1026 channels)
- **Action Expert**: {"Diffusion Policy (DDPM/DDIM)" if model_type == "diffusion" else "Regression-based prediction"}
- **Fusion Strategy**: Concatenation of VL and sensor features
- **Action Space**: 7-DoF (6 joints + gripper)
- **Horizon**: 8 timesteps

### Training Stage

**{stage_desc.get(training_stage, training_stage)}**

"""

    if training_stage == "stage1":
        card += """
This checkpoint contains:
- ✅ Sensor Encoder weights
- ✅ Action Expert weights
- ❌ VL backbone (use base Qwen2.5-VL-3B-Instruct)

**Use this for**:
- Further training in Stage 2
- Inference with frozen VL backbone
"""
    else:
        card += """
This checkpoint contains:
- ✅ LoRA adapters for VL backbone
- ✅ Sensor Encoder weights
- ✅ Action Expert weights
- ✅ Complete model for inference

**Use this for**:
- Direct inference
- Fine-tuning on new tasks
"""

    # Add performance metrics if provided
    if performance_metrics:
        card += "\n### Performance Metrics\n\n"
        for metric, value in performance_metrics.items():
            card += f"- **{metric}**: {value}\n"

    # Add checkpoint info
    if checkpoint_info:
        card += "\n### Training Info\n\n"
        if "epoch" in checkpoint_info:
            card += f"- **Epochs**: {checkpoint_info['epoch']}\n"
        if "val_loss" in checkpoint_info:
            card += f"- **Validation Loss**: {checkpoint_info['val_loss']:.4f}\n"

    card += f"""

## Usage

### Installation

```bash
pip install torch transformers peft huggingface_hub
pip install qwen-vl-utils  # For Qwen VL processing
```

### Quick Start

```python
import torch
from transformers import AutoProcessor
from models.model_with_sensor{"_diffusion" if model_type == "diffusion" else ""} import {"Not_freeze_QwenVLAWithSensorDiffusion" if model_type == "diffusion" and training_stage == "stage2" else "QwenVLAWithSensorDiffusion" if model_type == "diffusion" else "Not_freeze_QwenVLAWithSensor" if training_stage == "stage2" else "QwenVLAWithSensor"}

# Load model
model = {"Not_freeze_QwenVLAWithSensorDiffusion" if model_type == "diffusion" and training_stage == "stage2" else "QwenVLAWithSensorDiffusion" if model_type == "diffusion" else "Not_freeze_QwenVLAWithSensor" if training_stage == "stage2" else "QwenVLAWithSensor"}(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    sensor_enabled=True,
"""

    if model_type == "diffusion":
        card += "    diffusion_timesteps=100,\n"

    if training_stage == "stage2":
        card += """    finetune_vl="lora",
    lora_r=16,
    lora_alpha=32,
"""

    card += """)

# Load checkpoint from Hugging Face
checkpoint = torch.load("pytorch_model.pt")
"""

    if training_stage == "stage1":
        card += """model.sensor_encoder.load_state_dict(checkpoint["sensor_encoder"])
model.action_expert.load_state_dict(checkpoint["action_expert"])
"""
    else:
        card += """model.load_state_dict(checkpoint["model_state_dict"])
"""

    card += """
model.eval()
model.to("cuda")

# Prepare inputs
text_inputs = ["Insert the needle into the target carefully"]
image_inputs = [[view1_path, view2_path, view3_path, view4_path, view5_path]]
sensor_data = torch.randn(1, 650, 1026)  # (B, T, C)
"""

    if model_type == "diffusion":
        card += """
# Generate actions (diffusion sampling)
with torch.no_grad():
    actions = model(
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        actions=None,  # Inference mode
        sensor_data=sensor_data,
        cache_keys=["sample_0"]
    )
# actions: (1, 8, 7) - 8 timesteps, 7-DoF actions
"""
    else:
        card += """z_chunk = torch.zeros(1, 8, 7)  # Initial action chunk

# Predict actions
with torch.no_grad():
    pred_actions, delta = model(
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        z_chunk=z_chunk,
        sensor_data=sensor_data,
        cache_keys=["sample_0"]
    )
# pred_actions: (1, 8, 7) - 8 timesteps, 7-DoF actions
"""

    card += """```

### Input Specification

- **Text**: Natural language instruction (str)
- **Images**: 5-view camera system (list of image paths)
  - View 1-4: ZED cameras (left images)
  - View 5: OAK camera
- **Sensor Data**: (B, 650, 1026) tensor
  - 650 timesteps @ 650Hz (1 second window)
  - 1026 channels: 1 force + 1025 A-scan values

### Output

- **Actions**: (B, 8, 7) tensor
  - 8 timesteps (action horizon)
  - 7-DoF: 6 joint angles + 1 gripper state

## Training Details

### Dataset

- **Modalities**: 5 RGB cameras + OCT/FPI sensor + robot state
- **Tasks**: Needle insertion into various targets
- **Episodes**: ~100 demonstrations
- **Frames**: ~50,000 total

### Hyperparameters

"""

    if training_stage == "stage1":
        card += """- Learning Rate: 1e-4 (action expert), 5e-4 (sensor encoder)
- Epochs: 20
- Batch Size: 4 per GPU
- Gradient Accumulation: 8 steps
"""
    else:
        card += """- Learning Rate: 1e-4 (action expert), 5e-4 (sensor), 1e-5 (VL LoRA)
- Epochs: 10
- Batch Size: 4 per GPU
- Gradient Accumulation: 8 steps
- LoRA Rank: 16
- LoRA Alpha: 32
"""

    if model_type == "diffusion":
        card += """- Diffusion Timesteps: 100 (training), 10 (DDIM inference)
- Noise Schedule: Cosine
"""

    card += """
## Citation

```bibtex
@article{insertion_vla_2024,
  title={Vision-Language-Action Model with Sensor Fusion for Robotic Needle Insertion},
  author={Your Name},
  year={2024}
}
```

## License

Apache 2.0

## Links

- **Repository**: https://github.com/yourusername/Insertion_VLA
- **Paper**: Coming soon
"""

    return card


def create_config_json(
    model_type: str,
    training_stage: str,
    checkpoint_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Create config.json for the model"""

    config = {
        "model_type": f"vla_{model_type}",
        "training_stage": training_stage,
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "architecture": {
            "sensor_encoder": {
                "input_channels": 1026,
                "temporal_length": 650,
                "hidden_dim": 512,
                "output_dim": 3072,
                "use_transformer": True,
                "num_transformer_layers": 2
            },
            "action_expert": {
                "type": model_type,
                "action_dim": 7,
                "horizon": 8,
                "hidden_dim": 1024,
                "fusion_strategy": "concat"
            }
        },
        "training_info": {
            "epoch": checkpoint_info.get("epoch", "unknown"),
            "val_loss": checkpoint_info.get("val_loss", None),
        }
    }

    if model_type == "diffusion":
        config["architecture"]["action_expert"]["diffusion_timesteps"] = 100

    if training_stage == "stage2":
        config["lora_config"] = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
        }

    return config


def prepare_upload_directory(
    checkpoint_path: Path,
    model_type: str,
    training_stage: str,
    output_dir: Path,
    performance_metrics: Dict[str, float] = None
) -> Path:
    """Prepare directory for Hugging Face upload"""

    print(f"📦 Preparing upload directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Save model weights
    print("💾 Saving model weights...")
    torch.save(checkpoint, output_dir / "pytorch_model.pt")

    # Create config.json
    print("⚙️  Creating config.json...")
    config = create_config_json(model_type, training_stage, checkpoint)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create model card
    print("📝 Creating model card (README.md)...")
    model_card = create_model_card(model_type, training_stage, checkpoint, performance_metrics)
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)

    # Create training args file if available
    if "training_stage" in checkpoint:
        training_info = {
            "training_stage": checkpoint["training_stage"],
            "epoch": checkpoint.get("epoch", None),
            "val_loss": checkpoint.get("val_loss", None),
        }
        with open(output_dir / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)

    print(f"✅ Upload directory prepared: {output_dir}")
    return output_dir


def upload_to_hub(
    repo_id: str,
    local_dir: Path,
    token: str,
    private: bool = False,
    commit_message: str = None
):
    """Upload model to Hugging Face Hub"""

    print(f"\n🚀 Uploading to Hugging Face Hub: {repo_id}")

    # Create repository if it doesn't exist
    api = HfApi()

    try:
        print(f"📁 Creating repository (if not exists)...")
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✅ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  Error creating repository: {e}")
        print("   Continuing with upload...")

    # Upload folder
    print(f"⬆️  Uploading files...")
    try:
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message or "Upload VLA model checkpoint",
            repo_type="model"
        )
        print(f"\n✅ Upload complete!")
        print(f"   Model URL: https://huggingface.co/{repo_id}")
        print(f"   Files uploaded: {list(local_dir.glob('*'))}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload VLA model to Hugging Face Hub")

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file (.pt)")
    parser.add_argument("--model-type", choices=["regression", "diffusion"], required=True,
                        help="Model type: regression or diffusion")
    parser.add_argument("--training-stage", choices=["stage1", "stage2"], required=True,
                        help="Training stage: stage1 or stage2")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="Hugging Face repository ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, required=True,
                        help="Hugging Face API token")

    # Optional arguments
    parser.add_argument("--output-dir", type=str, default="./hf_upload_temp",
                        help="Temporary directory for preparing upload")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private")
    parser.add_argument("--commit-message", type=str, default=None,
                        help="Custom commit message")

    # Performance metrics (optional)
    parser.add_argument("--success-rate", type=float, default=None,
                        help="Task success rate (%)")
    parser.add_argument("--path-error", type=float, default=None,
                        help="Path tracking error (mm)")
    parser.add_argument("--force-error", type=float, default=None,
                        help="Force control error (N)")
    parser.add_argument("--inference-time", type=float, default=None,
                        help="Inference time (ms)")

    args = parser.parse_args()

    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Prepare performance metrics
    performance_metrics = {}
    if args.success_rate is not None:
        performance_metrics["Success Rate"] = f"{args.success_rate}%"
    if args.path_error is not None:
        performance_metrics["Path Error"] = f"{args.path_error:.2f}mm"
    if args.force_error is not None:
        performance_metrics["Force Error"] = f"{args.force_error:.2f}N"
    if args.inference_time is not None:
        performance_metrics["Inference Time"] = f"{args.inference_time:.1f}ms"

    # Prepare upload directory
    output_dir = Path(args.output_dir)
    prepare_upload_directory(
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        training_stage=args.training_stage,
        output_dir=output_dir,
        performance_metrics=performance_metrics if performance_metrics else None
    )

    # Upload to Hugging Face
    upload_to_hub(
        repo_id=args.repo_id,
        local_dir=output_dir,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message
    )

    # Cleanup (optional)
    print(f"\n🧹 Temporary directory kept at: {output_dir}")
    print(f"   You can delete it manually if needed.")


if __name__ == "__main__":
    main()
