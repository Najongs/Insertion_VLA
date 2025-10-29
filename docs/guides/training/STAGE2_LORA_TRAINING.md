# Stage 2: LoRA Fine-tuning Training Guide

## Overview

Stage 2 training continues from a Stage 1 checkpoint and adds LoRA fine-tuning to the VL (Vision-Language) backbone model.

### Training Flow

1. **Stage 1** (Completed)
   - ✅ VL model frozen
   - ✅ Train only Sensor Encoder + Diffusion Action Expert
   - ✅ Save checkpoint: `checkpoints/diffusion_epoch20.pt`

2. **Stage 2** (Current)
   - 📥 Load Stage 1 checkpoint (Sensor + Action Expert)
   - 🔓 Add LoRA to VL model (unfreeze adapters)
   - 🎯 Train entire model with different learning rates

## Model Architecture in Stage 2

```
┌─────────────────────────────────────────┐
│  Qwen2.5-VL-3B (with LoRA)              │
│  - LoRA rank: 16                        │
│  - Target: q_proj, k_proj, v_proj, o_proj│
│  - LR: 1e-5 (default)                   │
└─────────────┬───────────────────────────┘
              │ VL features (3072)
              │
┌─────────────▼───────────────────────────┐
│  Sensor Encoder (from Stage 1)          │
│  - Loaded from checkpoint               │
│  - LR: 1e-4 (default)                   │
└─────────────┬───────────────────────────┘
              │ Sensor features (3072)
              │
┌─────────────▼───────────────────────────┐
│  Diffusion Action Expert (from Stage 1) │
│  - Loaded from checkpoint               │
│  - Timesteps: 100                       │
│  - LR: 1e-4 (default)                   │
└─────────────┬───────────────────────────┘
              │
              ▼ Actions (B, 8, 7)
```

## Running Stage 2 Training

### Prerequisites

1. **Complete Stage 1 training first**
   ```bash
   torchrun --nproc_per_node=4 \
       training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
       --dataset_dir /home/najo/NAS/VLA/dataset \
       --training-stage stage1 \
       --epochs 20 \
       --batch_size 4 \
       --lr 1e-4
   ```

2. **Verify Stage 1 checkpoint exists**
   ```bash
   ls -lh checkpoints/diffusion_epoch20.pt
   ```

### Stage 2 Training Command

```bash
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/diffusion_epoch20.pt \
    --finetune-vl lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --grad_accum 8
```

### Arguments Explanation

**Stage 2 Specific:**
- `--training-stage stage2`: Enable Stage 2 mode
- `--stage1-checkpoint`: Path to Stage 1 checkpoint (required!)
- `--finetune-vl`: VL fine-tuning mode
  - `lora` (recommended): Add LoRA adapters
  - `full`: Unfreeze last N layers completely
  - `none`: Keep VL frozen (not typical for Stage 2)

**LoRA Configuration:**
- `--lora-r 16`: LoRA rank (lower = fewer params, faster)
- `--lora-alpha 32`: LoRA alpha (typically 2x rank)
- `--lora-dropout 0.05`: LoRA dropout

**Learning Rates:**
- `--lr 1e-4`: Learning rate for Sensor + Action Expert
- `--vl-lr 1e-5`: Learning rate for VL model (LoRA) - **typically 10x smaller!**

**Other:**
- `--epochs 10`: Fewer epochs than Stage 1 (fine-tuning)
- `--batch_size 4`: Per-GPU batch size
- `--grad_accum 8`: Gradient accumulation steps

## Key Differences from Stage 1

| Aspect | Stage 1 | Stage 2 |
|--------|---------|---------|
| VL Model | ❄️ Frozen | 🔓 LoRA adapters trainable |
| Sensor Encoder | 🎯 Train from scratch | 📥 Load from Stage 1 |
| Action Expert | 🎯 Train from scratch | 📥 Load from Stage 1 |
| Learning Rate | Single (1e-4) | Dual (VL: 1e-5, Others: 1e-4) |
| Epochs | 20+ | 10-15 |
| Checkpoint | Sensor + Action only | Full model state |
| Cache Mode | On (recommended) | Off (LoRA changes features) |

## Optimizer Configuration

In Stage 2, the optimizer uses **parameter groups** with different learning rates:

```python
optimizer = torch.optim.AdamW([
    {'params': vl_params, 'lr': 1e-5, 'weight_decay': 0.01},      # LoRA
    {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.01},   # Sensor + Action
], betas=(0.9, 0.95))
```

This is crucial because:
- VL model is pre-trained → needs smaller LR to avoid catastrophic forgetting
- Sensor/Action are already trained in Stage 1 → can use higher LR for fine-tuning

## Checkpoint Saving

**Stage 1:**
```python
checkpoint = {
    "sensor_encoder": model.sensor_encoder.state_dict(),
    "action_expert": model.action_expert.state_dict(),
    "training_stage": "stage1"
}
```

**Stage 2:**
```python
checkpoint = {
    "model_state_dict": model.state_dict(),  # Full model including LoRA
    "training_stage": "stage2"
}
```

## Expected Training Behavior

1. **Initial validation loss should be lower than Stage 1 end**
   - Because Sensor + Action are already trained

2. **Training loss may increase initially**
   - VL model adapting to the task through LoRA

3. **LoRA adds ~1-5% extra parameters**
   ```
   Base VL model: 3B params (frozen)
   LoRA adapters: ~50M params (trainable)
   Sensor Encoder: ~10M params (trainable)
   Action Expert: ~20M params (trainable)
   Total trainable: ~80M params
   ```

## Troubleshooting

### Issue 1: OOM (Out of Memory)
**Solution:** Reduce batch size or use gradient checkpointing
```bash
--batch_size 2 --grad_accum 16
```

### Issue 2: Loss not decreasing
**Possible causes:**
- `--vl-lr` too high → try 1e-6
- Cache mode still on → should be off for LoRA
- Stage 1 checkpoint not loaded properly

### Issue 3: "Missing keys" error when loading checkpoint
**Solution:** Make sure you're using the correct Stage 1 checkpoint
```bash
# Check checkpoint contents
python -c "import torch; ckpt = torch.load('checkpoints/diffusion_epoch20.pt'); print(ckpt.keys())"
```

## Monitoring Training

Watch for these metrics in WandB:

1. **train/loss**: Should decrease steadily but slower than Stage 1
2. **val/loss**: Should be lower than Stage 1 final validation loss
3. **train/lr**: Two different learning rates (check if both are updating)

## Example Training Run

```bash
# Stage 1 (already completed)
# Final loss: 0.4122, Val loss: 0.3231

# Stage 2
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/diffusion_epoch20.pt \
    --finetune-vl lora \
    --epochs 10 \
    --batch_size 4 \
    --lr 1e-4 \
    --vl-lr 1e-5

# Expected output:
# 📍 Stage 2: Training Full Model with LoRA
# 📥 Loading Stage 1 checkpoint from checkpoints/diffusion_epoch20.pt
#    ✅ Loaded Sensor Encoder from Stage 1
#    ✅ Loaded Diffusion Action Expert from Stage 1
# 💡 Applying LoRA fine-tuning to VL model...
#    LoRA trainable parameters: 50,331,648
# 🎯 Optimizer config:
#    VL params: 192 (lr=1e-05)
#    Other params: 84 (lr=0.0001)
```

## Advanced: Full Fine-tuning (Not Recommended)

If you want to unfreeze VL layers instead of using LoRA:

```bash
--finetune-vl full  # Unfreezes last 2 layers
```

**Warning:** This requires much more GPU memory and is more prone to overfitting!

## Next Steps After Stage 2

1. **Evaluate on validation set**
   - Compare with Stage 1 performance
   - Check if VL fine-tuning helped

2. **Test inference**
   ```bash
   python examples/test_sensor_model.py \
       --checkpoint checkpoints/diffusion_epoch10.pt \
       --stage stage2
   ```

3. **Optional: Continue training**
   ```bash
   --resume checkpoints/diffusion_epoch10.pt
   ```
