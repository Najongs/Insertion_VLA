# Training Guide for QwenVLA with Sensor Integration

## 📋 Overview

This guide covers training the QwenVLA model with the new integrated dataset system that supports **optional sensor data** from OCT/FPI sensors.

**New Features:**
- ✅ Unified dataset loading (handles datasets with/without sensor)
- ✅ Sensor encoder training
- ✅ Multiple fusion strategies (concat, cross_attention, gated)
- ✅ Weighted loss for sensor-equipped samples
- ✅ Automatic dummy data generation for non-sensor samples

---

## 📁 Updated Files

### New Training Script
- **`5st_VLA_TRAIN_VL_Lora_with_sensor.py`** - Updated training script with sensor support
  - Uses `model_with_sensor.Not_freeze_QwenVLAWithSensor`
  - Uses `IntegratedDataset` with `collate_fn_with_sensor`
  - Handles mixed batches (with/without sensor data)

### Original Files (Still Available)
- **`5st_VLA_TRAIN_VL_Lora.py`** - Original training script (without sensor)
- Use this if you want to train without sensor data

---

## 🚀 Quick Start

### 1. Build VL Feature Cache (Required, One-time)

First, build the VL feature cache for all datasets:

```bash
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode cache
```

**What this does:**
- Extracts VL features from Qwen2.5-VL-3B for all images
- Saves features to `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/`
- Only needs to be run once for all datasets
- Takes ~2-4 hours depending on dataset size

---

### 2. Training with Sensor Support

#### Option A: Train with Sensor Encoder (Recommended)

```bash
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --sensor-enabled \
    --fusion-strategy concat \
    --lr 1e-4 \
    --sensor-lr 5e-4 \
    --vl-lr 1e-5 \
    --finetune-vl lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --grad-accum-steps 8 \
    --batch-size 1 \
    --num-workers 4 \
    --sensor-loss-weight 2.0
```

**Key Arguments:**
- `--sensor-enabled`: Enable sensor encoder training
- `--fusion-strategy`: How to combine VL + sensor features
  - `concat`: Concatenate features (simple, effective)
  - `cross_attention`: Cross-attention fusion (more complex)
  - `gated`: Gated fusion (adaptive weighting)
  - `none`: No fusion (sensor encoder only)
- `--sensor-lr`: Learning rate for sensor encoder (typically higher)
- `--sensor-loss-weight`: Weight multiplier for samples with sensor data (2.0 = 2x weight)

#### Option B: Train without Sensor (Baseline)

```bash
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --finetune-vl lora \
    --grad-accum-steps 8 \
    --batch-size 1 \
    --num-workers 4
```

**Note:** Without `--sensor-enabled`, the model ignores sensor data even if present.

---

## ⚙️ Configuration Options

### Basic Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | **Required** | `cache` or `train` |
| `--lr` | `1e-4` | Base learning rate (action expert) |
| `--min-lr` | `1e-6` | Minimum LR for cosine decay |
| `--warmup-ratio` | `0.03` | Warmup ratio (3% of total steps) |
| `--hold-ratio` | `0.02` | Hold ratio after warmup |
| `--grad-accum-steps` | `8` | Gradient accumulation steps |
| `--sched-on` | `step` | Scheduler frequency: `step` or `epoch` |
| `--batch-size` | `1` | Batch size per GPU |
| `--num-workers` | `4` | DataLoader workers |

### VL Fine-tuning Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--finetune-vl` | `lora` | VL fine-tuning: `none`, `lora`, `full` |
| `--vl-lr` | `1e-5` | VL backbone learning rate |
| `--vision-lr` | `5e-6` | Vision encoder LR (if `full`) |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--unfreeze-last-n` | `2` | Unfreeze last N transformer layers (if `full`) |

### Sensor Options (NEW)

| Argument | Default | Description |
|----------|---------|-------------|
| `--sensor-enabled` | `False` | Enable sensor encoder |
| `--sensor-input-channels` | `1026` | 1 force + 1025 A-scan |
| `--sensor-temporal-length` | `650` | Sensor sequence length (650Hz × 1s) |
| `--sensor-output-dim` | `3072` | Must match VL dimension |
| `--fusion-strategy` | `concat` | Fusion type: `concat`, `cross_attention`, `gated`, `none` |
| `--sensor-lr` | `5e-4` | Sensor encoder learning rate |
| `--sensor-loss-weight` | `2.0` | Loss weight for sensor samples |

---

## 📊 Dataset Configuration

The script automatically loads all available datasets:

### Datasets with Sensor (5.5% of total)
```python
"/home/najo/NAS/VLA/Insertion_VLA/dataset/White_silicone_white_circle/recv_all_*"
"/home/najo/NAS/VLA/Insertion_VLA/dataset/Needle_insertion_eye_trocar/recv_all_*"
```
- 10 sessions
- 5,470 samples
- Sensor shape: (650, 1026)

### Datasets without Sensor (94.5% of total)
```python
"/home/najo/NAS/VLA/Insertion_VLA/dataset/OCT_insertion/Captures*"
"/home/najo/NAS/VLA/Insertion_VLA/dataset/part1/ZED_Captures_*th"
```
- 27 sessions
- 93,239 samples
- Uses dummy zeros for sensor data

**Total:** 98,709 samples across 37 sessions

---

## 🎯 Training Strategies

### Strategy 1: Mixed Training (Default)
Train on all data simultaneously. Sensor encoder learns from 5.5% of samples.

**Pros:**
- Simple, single-stage training
- Model generalizes to both scenarios

**Cons:**
- Sensor encoder gets limited training signal

**Recommendation:** Use `--sensor-loss-weight 2.0` to upweight sensor samples.

---

### Strategy 2: Two-Stage Training
Stage 1: Train on all data without sensor
Stage 2: Fine-tune with sensor enabled on sensor-rich data

```bash
# Stage 1: Baseline (50 epochs)
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --lr 1e-4 \
    --finetune-vl lora

# Stage 2: Add sensor (modify dataset paths in script first)
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --sensor-enabled \
    --lr 5e-5 \
    --sensor-lr 5e-4
```

---

### Strategy 3: Balanced Sampling (TODO)
Oversample sensor data to balance the dataset.

**Implementation needed:**
- Custom sampler that repeats sensor samples
- Weight = 93239 / 5470 ≈ 17x

---

## 💡 Tips & Best Practices

### Memory Management
- **Batch Size:** Start with `--batch-size 1` for 24GB GPUs
- **Gradient Accumulation:** Use `--grad-accum-steps 8` for effective batch size of 32
- **Workers:** `--num-workers 4` is usually optimal

### Learning Rates
- **Action Expert:** `--lr 1e-4` (highest, learns from scratch)
- **Sensor Encoder:** `--sensor-lr 5e-4` (high, learns from scratch)
- **VL LoRA:** `--vl-lr 1e-5` (low, fine-tuning pretrained)
- **Vision Encoder:** `--vision-lr 5e-6` (lowest, pretrained vision)

### Fusion Strategy Selection
1. **Start with `concat`** - Simple, fast, usually works well
2. **Try `cross_attention`** if concat plateaus - More parameters, slower
3. **Try `gated`** for adaptive fusion - Good when sensor quality varies

### Sensor Loss Weight
- `1.0`: Equal weight for all samples
- `2.0`: 2x weight for sensor samples (recommended start)
- `5.0`: Aggressive upweighting (if sensor learning is too slow)
- Monitor `train/sensor_ratio` in wandb to verify balance

---

## 📈 Monitoring Training

### WandB Metrics

**Training:**
- `train/loss_step`: Step-wise loss
- `train/loss_epoch`: Epoch average loss
- `train/lr`: Current learning rate
- `train/grad_norm`: Gradient norm (should be < 10)
- `train/sensor_samples`: Samples with sensor in current batch
- `train/sensor_ratio`: Ratio of sensor samples per epoch

**Validation:**
- `val/loss_epoch`: Validation loss (best model tracking)

**System:**
- `system/gpu_mem_GB`: GPU memory usage
- `system/cpu_mem_%`: CPU memory usage

**Model:**
- `params/trainable_M`: Trainable parameters (millions)
- `params/frozen_M`: Frozen parameters
- `lora/avg_weight_abs`: Average LoRA weight magnitude

---

## 🔍 Troubleshooting

### Issue: OOM (Out of Memory)
**Solutions:**
1. Reduce `--batch-size` to 1
2. Increase `--grad-accum-steps` to maintain effective batch size
3. Reduce `--num-workers` to 2
4. Use `--finetune-vl none` to freeze VL backbone

### Issue: Sensor encoder not learning
**Symptoms:** `train/sensor_ratio` shows low sensor coverage
**Solutions:**
1. Increase `--sensor-loss-weight` to 5.0 or higher
2. Use two-stage training (baseline → sensor)
3. Check if sensor data is loading correctly (inspect batch)

### Issue: Training very slow
**Solutions:**
1. Verify cache was built (`--mode cache`)
2. Use `--num-workers 4` or higher
3. Check disk I/O (move data to SSD if on HDD)
4. Reduce `--batch-size` if too many workers

### Issue: Loss not decreasing
**Solutions:**
1. Check learning rates (might be too low/high)
2. Verify data loading (inspect batch manually)
3. Try simpler fusion strategy (`concat`)
4. Increase warmup: `--warmup-ratio 0.05`

---

## 📂 Checkpoints

### Saved Files
- `./checkpoints/qwen_vla_sensor.pt` - Latest checkpoint (updated every epoch)
- `./checkpoints/qwen_vla_sensor_best.pt` - Best validation loss
- `./checkpoints/qwen_vla_sensor_final.pt` - Final checkpoint after all epochs

### Checkpoint Contents
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "val_loss": float
}
```

### Resume Training
The script automatically resumes from `./checkpoints/qwen_vla_sensor.pt` if it exists:
- Loads model weights (with `strict=False` for compatibility)
- Restores optimizer state
- Continues from last epoch
- Uses ReWarm scheduler to gradually increase LR

---

## 🧪 Testing the Model

After training, test inference:

```python
from model_with_sensor import Not_freeze_QwenVLAWithSensor
import torch

# Load model
model = Not_freeze_QwenVLAWithSensor(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    sensor_enabled=True,
    fusion_strategy='concat'
).cuda()

# Load checkpoint
ckpt = torch.load("./checkpoints/qwen_vla_sensor_best.pt")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prepare inputs
text_inputs = ["Insert into the white square silicone"]
image_inputs = [["/path/to/img1.jpg", "/path/to/img2.jpg"]]
sensor_data = torch.randn(1, 650, 1026).cuda()  # (B, T, C)

# Inference
with torch.no_grad():
    pred_actions, _ = model(
        text_inputs=text_inputs,
        image_inputs=image_inputs,
        sensor_data=sensor_data
    )

print(f"Predicted actions: {pred_actions.shape}")  # (B, horizon=8, action_dim=7)
```

---

## 📚 References

- **Dataset Documentation:** `FINAL_DATASET_SUMMARY.md`
- **Dataset Details:** `DATASET_README.md`
- **Model Architecture:** `model_with_sensor.py`
- **Dataset Loading:** `IntegratedDataset.py`
- **Preprocessing:** `preprocess_sensor_dataset.py`

---

## 🎓 Next Steps

1. ✅ **Build cache** - Run `--mode cache` once
2. ✅ **Baseline training** - Train without sensor first (optional)
3. ✅ **Sensor training** - Enable sensor encoder
4. 🔄 **Experiment with fusion strategies** - Try concat → cross_attention → gated
5. 🔄 **Tune sensor loss weight** - Find optimal balance
6. 🔄 **Ablation studies** - Compare with/without sensor
7. 🔄 **Deploy and evaluate** - Test on real robot

---

**Last Updated:** 2025-10-27
**Status:** ✅ Ready for Training
