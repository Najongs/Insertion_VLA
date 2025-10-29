# 2-Stage Training Guide

Complete guide for training VLA models with 2-stage approach: Stage 1 (Sensor+Action) → Stage 2 (Full Model with LoRA).

## Why 2-Stage Training?

Training in two stages prevents initialization issues and ensures better convergence:

1. **Stage 1**: Train Sensor Encoder + Action Expert with **frozen VL backbone**
   - Learns sensor representation and action prediction
   - VL features remain unchanged
   - Faster training, lower memory

2. **Stage 2**: Load Stage 1 checkpoint → Add LoRA to VL → Train entire model
   - Fine-tunes VL backbone with LoRA adapters
   - Refines sensor and action modules together with VL
   - Better multi-modal alignment

## Quick Start

### Regression Model

#### Stage 1: Train Sensor + Action Expert
```bash
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --training-stage stage1 \
    --finetune-vl none \
    --sensor-enabled \
    --fusion-strategy concat \
    --lr 1e-4 \
    --sensor-lr 5e-4 \
    --epochs 20 \
    --batch-size 4 \
    --grad-accum-steps 8
```

**Result**: Checkpoint saved to `checkpoints/qwen_vla_sensor_best.pt` containing:
- `sensor_encoder` state dict
- `action_expert` state dict
- optimizer/scheduler states
- `training_stage: "stage1"`

#### Stage 2: Load Stage 1 + Add LoRA + Train Full Model
```bash
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/qwen_vla_sensor_best.pt \
    --finetune-vl lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --sensor-enabled \
    --fusion-strategy concat \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --sensor-lr 5e-4 \
    --epochs 10 \
    --batch-size 4 \
    --grad-accum-steps 8
```

**Result**: Full model checkpoint with LoRA-finetuned VL backbone.

---

### Diffusion Model

#### Stage 1: Train Sensor + Diffusion Action Expert
```bash
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /path/to/dataset \
    --training-stage stage1 \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --diffusion_timesteps 100
```

**Result**: Checkpoint saved to `checkpoints/diffusion_epoch20.pt` containing:
- `sensor_encoder` state dict
- `action_expert` state dict (diffusion model)
- optimizer/scheduler states
- `training_stage: "stage1"`

#### Stage 2: Load Stage 1 + Add LoRA + Train Full Model
```bash
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /path/to/dataset \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/diffusion_epoch20.pt \
    --finetune-vl lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --vl-lr 1e-5
```

**Result**: Full diffusion model checkpoint with LoRA-finetuned VL backbone.

---

## Detailed Arguments

### Common Arguments (Both Models)

| Argument | Default | Description |
|----------|---------|-------------|
| `--training-stage` | `stage1` | Training stage: `stage1` or `stage2` |
| `--stage1-checkpoint` | `None` | Path to Stage 1 checkpoint (required for stage2) |
| `--finetune-vl` | `lora` | VL fine-tuning: `none`, `lora`, or `full` |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha (scaling factor) |
| `--lora-dropout` | `0.05` | LoRA dropout rate |
| `--vl-lr` | `1e-5` | Learning rate for VL backbone (Stage 2) |

### Regression-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | - | `cache` or `train` |
| `--lr` | `1e-4` | Base learning rate |
| `--sensor-lr` | `5e-4` | Sensor encoder learning rate |
| `--grad-accum-steps` | `8` | Gradient accumulation steps |

### Diffusion-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--diffusion_timesteps` | `100` | Number of diffusion timesteps |
| `--grad_accum` | `8` | Gradient accumulation steps |
| `--sensor_loss_weight` | `2.0` | Loss weight for sensor samples |

---

## Checkpoint Structure

### Stage 1 Checkpoint
```python
{
    "epoch": 20,
    "sensor_encoder": OrderedDict(...),  # Sensor encoder weights
    "action_expert": OrderedDict(...),   # Action expert weights
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "val_loss": 0.0234,
    "training_stage": "stage1"
}
```

### Stage 2 Checkpoint
```python
{
    "epoch": 10,
    "model_state_dict": OrderedDict(...),  # Full model including VL with LoRA
    "optimizer_state_dict": {...},
    "scheduler_state_dict": {...},
    "val_loss": 0.0189,
    "training_stage": "stage2"
}
```

---

## Model Classes

### Regression Models

- **Stage 1**: `QwenVLAWithSensor` (frozen VL)
  - Located in `models/model_with_sensor.py`
  - VL backbone frozen
  - Trainable: Sensor Encoder + Action Expert

- **Stage 2**: `Not_freeze_QwenVLAWithSensor` (LoRA-enabled)
  - Located in `models/model_with_sensor.py`
  - Loads Stage 1 checkpoint via `stage1_checkpoint` parameter
  - Adds LoRA adapters to VL backbone
  - Trainable: All modules (VL LoRA + Sensor + Action)

### Diffusion Models

- **Stage 1**: `QwenVLAWithSensorDiffusion` (frozen VL)
  - Located in `models/model_with_sensor_diffusion.py`
  - VL backbone frozen
  - Trainable: Sensor Encoder + Diffusion Action Expert

- **Stage 2**: `Not_freeze_QwenVLAWithSensorDiffusion` (LoRA-enabled)
  - Located in `models/model_with_sensor_diffusion.py`
  - Loads Stage 1 checkpoint via `stage1_checkpoint` parameter
  - Adds LoRA adapters to VL backbone
  - Trainable: All modules (VL LoRA + Sensor + Diffusion Action)

---

## Training Tips

### Stage 1 Recommendations

1. **Higher Learning Rates**: Sensor encoder and action expert can use higher LR (1e-4 to 5e-4)
2. **More Epochs**: Train for 15-20 epochs to ensure convergence
3. **Monitor Sensor Loss**: Check if sensor fusion is working by comparing sensor vs non-sensor samples
4. **Checkpoint Selection**: Choose the best checkpoint based on validation loss

### Stage 2 Recommendations

1. **Lower VL Learning Rate**: Use 1e-5 to 1e-6 for VL backbone to avoid catastrophic forgetting
2. **Fewer Epochs**: 5-10 epochs is usually sufficient
3. **Gradient Clipping**: Use gradient clipping (1.0) to stabilize training
4. **Monitor Overfitting**: Early stopping if validation loss starts increasing

### Memory Optimization

- Stage 1: ~24GB per GPU (VL frozen)
- Stage 2: ~28GB per GPU (LoRA adds minimal overhead)
- Use `--grad-accum` to reduce batch size if OOM
- Consider mixed precision training (enabled by default)

---

## Troubleshooting

### Error: "Stage 2 training requires --stage1-checkpoint"
**Solution**: Provide path to Stage 1 checkpoint:
```bash
--stage1-checkpoint checkpoints/qwen_vla_sensor_best.pt
```

### Warning: "Stage 1 should use finetune-vl=none"
**Solution**: For Stage 1, always use `--finetune-vl none` or omit the argument.

### Error: Checkpoint loading mismatch
**Solution**: Ensure Stage 1 checkpoint contains `sensor_encoder` and `action_expert` keys.
```python
checkpoint = torch.load("checkpoint.pt")
print(checkpoint.keys())  # Should show: sensor_encoder, action_expert, ...
```

### Poor Stage 2 Performance
**Possible causes**:
1. Stage 1 not converged → Train Stage 1 longer
2. VL LR too high → Reduce `--vl-lr` to 1e-6
3. Sensor encoder forgetting → Reduce `--lr` and increase `--sensor-lr`

---

## Example Workflow

Complete example for training a diffusion model:

```bash
# 1. Stage 1: Train for 20 epochs
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/Insertion_VLA/dataset \
    --training-stage stage1 \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --diffusion_timesteps 100

# Wait for training to complete...
# Best checkpoint saved to: checkpoints/diffusion_epoch20.pt

# 2. Stage 2: Load Stage 1 + Add LoRA + Train for 10 epochs
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/Insertion_VLA/dataset \
    --training-stage stage2 \
    --stage1-checkpoint checkpoints/diffusion_epoch20.pt \
    --finetune-vl lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --vl-lr 1e-5

# Final model saved to: checkpoints/diffusion_epoch10.pt (Stage 2)
```

---

## Validation

To verify your checkpoints are correct:

```python
import torch

# Check Stage 1 checkpoint
ckpt1 = torch.load("checkpoints/qwen_vla_sensor_best.pt")
print("Stage 1 keys:", ckpt1.keys())
print("Training stage:", ckpt1.get("training_stage"))
# Expected: sensor_encoder, action_expert, optimizer_state_dict, ...

# Check Stage 2 checkpoint
ckpt2 = torch.load("checkpoints/qwen_vla_sensor_best.pt")  # After Stage 2
print("Stage 2 keys:", ckpt2.keys())
print("Training stage:", ckpt2.get("training_stage"))
# Expected: model_state_dict, optimizer_state_dict, ...
```

---

**Last Updated**: 2024-10-28
**Maintained By**: Project Team
