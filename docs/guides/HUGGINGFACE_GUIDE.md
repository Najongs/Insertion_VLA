# Hugging Face Model Sharing Guide

Complete guide for uploading and downloading trained VLA models to/from Hugging Face Hub.

## 📤 Uploading Models to Hugging Face

### Prerequisites

1. **Install Hugging Face Hub**:
```bash
pip install huggingface_hub
```

2. **Get Hugging Face Token**:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with **write** permissions
   - Save it securely

### Upload Commands

#### Stage 1 Model (Sensor + Action Expert)

**Regression:**
```bash
python upload_to_huggingface.py \
    --checkpoint checkpoints/qwen_vla_sensor_best.pt \
    --model-type regression \
    --training-stage stage1 \
    --repo-id your-username/insertion-vla-regression-stage1 \
    --token YOUR_HF_TOKEN \
    --success-rate 82.5 \
    --path-error 2.3 \
    --inference-time 150
```

**Diffusion:**
```bash
python upload_to_huggingface.py \
    --checkpoint checkpoints/diffusion_epoch20.pt \
    --model-type diffusion \
    --training-stage stage1 \
    --repo-id your-username/insertion-vla-diffusion-stage1 \
    --token YOUR_HF_TOKEN \
    --success-rate 85.0 \
    --path-error 2.0 \
    --inference-time 600
```

#### Stage 2 Model (Full Model with LoRA)

**Regression:**
```bash
python upload_to_huggingface.py \
    --checkpoint checkpoints/qwen_vla_sensor_best.pt \
    --model-type regression \
    --training-stage stage2 \
    --repo-id your-username/insertion-vla-regression-stage2 \
    --token YOUR_HF_TOKEN \
    --success-rate 87.5 \
    --path-error 1.8 \
    --inference-time 180
```

**Diffusion:**
```bash
python upload_to_huggingface.py \
    --checkpoint checkpoints/diffusion_epoch10.pt \
    --model-type diffusion \
    --training-stage stage2 \
    --repo-id your-username/insertion-vla-diffusion-stage2 \
    --token YOUR_HF_TOKEN \
    --success-rate 90.0 \
    --path-error 1.5 \
    --inference-time 650
```

### Upload Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--checkpoint` | ✅ | Path to checkpoint file (.pt) |
| `--model-type` | ✅ | `regression` or `diffusion` |
| `--training-stage` | ✅ | `stage1` or `stage2` |
| `--repo-id` | ✅ | Repository ID (username/model-name) |
| `--token` | ✅ | Hugging Face API token |
| `--output-dir` | ❌ | Temp directory (default: `./hf_upload_temp`) |
| `--private` | ❌ | Make repository private |
| `--commit-message` | ❌ | Custom commit message |
| `--success-rate` | ❌ | Task success rate (%) |
| `--path-error` | ❌ | Path tracking error (mm) |
| `--force-error` | ❌ | Force control error (N) |
| `--inference-time` | ❌ | Inference time (ms) |

### What Gets Uploaded

The script automatically creates:

1. **pytorch_model.pt** - Model checkpoint
2. **config.json** - Model configuration
3. **README.md** - Auto-generated model card with:
   - Model description
   - Architecture details
   - Usage examples
   - Training information
   - Performance metrics
4. **training_info.json** - Training metadata

---

## 📥 Downloading Models from Hugging Face

### Download Commands

**Regression Stage 2:**
```bash
python download_from_huggingface.py \
    --repo-id your-username/insertion-vla-regression-stage2 \
    --model-type regression \
    --training-stage stage2 \
    --output-dir ./downloaded_models
```

**Diffusion Stage 2:**
```bash
python download_from_huggingface.py \
    --repo-id your-username/insertion-vla-diffusion-stage2 \
    --model-type diffusion \
    --training-stage stage2 \
    --output-dir ./downloaded_models
```

**For Private Repositories:**
```bash
python download_from_huggingface.py \
    --repo-id your-username/private-model \
    --model-type regression \
    --training-stage stage2 \
    --token YOUR_HF_TOKEN
```

### Download Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--repo-id` | ✅ | Repository ID (username/model-name) |
| `--model-type` | ✅ | `regression` or `diffusion` |
| `--training-stage` | ✅ | `stage1` or `stage2` |
| `--output-dir` | ❌ | Download directory (default: `./downloaded_models`) |
| `--token` | ❌ | HF token (for private repos) |
| `--revision` | ❌ | Branch/revision (default: `main`) |

### What Gets Downloaded

- All model files from the repository
- Auto-generated **load_model_example.py** script

---

## 🚀 Using Downloaded Models

### Option 1: Using Auto-Generated Script

After downloading, a loading script is automatically created:

```bash
cd downloaded_models/your-username_insertion-vla-regression-stage2
python load_model_example.py
```

### Option 2: Manual Loading

**Regression Model (Stage 2):**
```python
import torch
from models.model_with_sensor import Not_freeze_QwenVLAWithSensor

# Initialize model
model = Not_freeze_QwenVLAWithSensor(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    sensor_enabled=True,
    finetune_vl="lora",
    lora_r=16,
    lora_alpha=32,
)

# Load checkpoint
checkpoint = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# Use model
model.eval()
model.to("cuda")
```

**Diffusion Model (Stage 2):**
```python
import torch
from models.model_with_sensor_diffusion import Not_freeze_QwenVLAWithSensorDiffusion

# Initialize model
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

# Load checkpoint
checkpoint = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# Use model
model.eval()
model.to("cuda")
```

**Stage 1 Models:**
```python
import torch
from models.model_with_sensor import QwenVLAWithSensor

# Initialize model
model = QwenVLAWithSensor(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    sensor_enabled=True,
)

# Load checkpoint (Stage 1: only sensor_encoder and action_expert)
checkpoint = torch.load("pytorch_model.pt", map_location="cpu")
model.sensor_encoder.load_state_dict(checkpoint["sensor_encoder"])
model.action_expert.load_state_dict(checkpoint["action_expert"])

# Use model
model.eval()
model.to("cuda")
```

---

## 🔄 Complete Workflow Example

### 1. Train Models

```bash
# Stage 1: Train Sensor + Action
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /path/to/dataset \
    --training-stage stage1 \
    --epochs 20

# Stage 2: Load Stage 1 + Add LoRA
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /path/to/dataset \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/diffusion_epoch20.pt \
    --finetune-vl lora \
    --epochs 10
```

### 2. Upload to Hugging Face

```bash
# Upload Stage 1
python upload_to_huggingface.py \
    --checkpoint checkpoints/diffusion_epoch20.pt \
    --model-type diffusion \
    --training-stage stage1 \
    --repo-id myusername/insertion-vla-diffusion-stage1 \
    --token $HF_TOKEN \
    --success-rate 85.0

# Upload Stage 2
python upload_to_huggingface.py \
    --checkpoint checkpoints/diffusion_epoch10.pt \
    --model-type diffusion \
    --training-stage stage2 \
    --repo-id myusername/insertion-vla-diffusion-stage2 \
    --token $HF_TOKEN \
    --success-rate 90.0
```

### 3. Share with Team

Send repository links:
- Stage 1: `https://huggingface.co/myusername/insertion-vla-diffusion-stage1`
- Stage 2: `https://huggingface.co/myusername/insertion-vla-diffusion-stage2`

### 4. Download and Use

```bash
# Team member downloads
python download_from_huggingface.py \
    --repo-id myusername/insertion-vla-diffusion-stage2 \
    --model-type diffusion \
    --training-stage stage2

# Run inference
cd downloaded_models/myusername_insertion-vla-diffusion-stage2
python load_model_example.py
```

---

## 📋 Model Naming Convention

Recommended repository naming:

```
{organization}/{project}-{model-type}-{stage}

Examples:
- mylab/insertion-vla-regression-stage1
- mylab/insertion-vla-regression-stage2
- mylab/insertion-vla-diffusion-stage1
- mylab/insertion-vla-diffusion-stage2

Or with version:
- mylab/insertion-vla-diffusion-v1.0
- mylab/insertion-vla-diffusion-v1.1
```

---

## 🔐 Security Best Practices

1. **Never commit tokens to git**:
   ```bash
   export HF_TOKEN=your_token_here
   python upload_to_huggingface.py --token $HF_TOKEN ...
   ```

2. **Use read tokens for downloads**:
   - Create separate tokens for reading/writing
   - Share read-only tokens with team

3. **Private repositories for unpublished work**:
   ```bash
   python upload_to_huggingface.py ... --private
   ```

4. **Model cards for documentation**:
   - Auto-generated model cards include usage examples
   - Edit README.md on Hugging Face web interface for details

---

## 🐛 Troubleshooting

### Error: "Repository not found"

**Solution**: Make sure the repository exists or will be created:
```bash
# Repository will be created automatically if it doesn't exist
python upload_to_huggingface.py ... --repo-id your-username/new-model
```

### Error: "Authentication failed"

**Solution**: Check your token permissions:
1. Go to https://huggingface.co/settings/tokens
2. Ensure token has **write** permissions
3. Use the correct token

### Error: "Checkpoint loading failed"

**Solution**: Verify checkpoint structure:
```python
import torch
ckpt = torch.load("checkpoint.pt")
print(ckpt.keys())  # Should show: training_stage, sensor_encoder, action_expert, etc.
```

### Model size too large

**Solution**: Use Git LFS (automatically handled by Hugging Face):
```bash
# Already configured by huggingface_hub
# No additional setup needed
```

### Download incomplete

**Solution**: Re-download with clean cache:
```bash
rm -rf downloaded_models/your-username_model-name
python download_from_huggingface.py --repo-id your-username/model-name ...
```

---

## 📊 Model Card Best Practices

The auto-generated model card includes:
- ✅ Model description
- ✅ Architecture details
- ✅ Usage examples
- ✅ Training information
- ✅ Performance metrics

**Customize after upload**:
1. Go to your model page on Hugging Face
2. Click "Edit model card"
3. Add:
   - Demo videos/GIFs
   - Additional examples
   - Limitations and biases
   - Intended use cases
   - Citation information

---

## 🔗 Useful Links

- **Hugging Face Hub Documentation**: https://huggingface.co/docs/hub
- **Model Cards Guide**: https://huggingface.co/docs/hub/model-cards
- **Hub API Reference**: https://huggingface.co/docs/huggingface_hub
- **Our Models**: https://huggingface.co/your-username

---

**Last Updated**: 2024-10-28
**Maintained By**: Project Team
