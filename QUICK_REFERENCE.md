# Quick Reference - QwenVLA Training with Sensor

## 🚀 Essential Commands

### Build Cache (Required, Run Once)
```bash
cd /home/najo/NAS/VLA/Insertion_VLA

torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode cache
```

---

### Train with Sensor (Recommended)
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
    --grad-accum-steps 8 \
    --sensor-loss-weight 2.0
```

---

### Train without Sensor (Baseline)
```bash
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --finetune-vl lora \
    --grad-accum-steps 8
```

---

## 📊 Dataset Summary

| Dataset | Sensor | Sessions | Samples |
|---------|--------|----------|---------|
| White_silicone_white_circle | ✅ Yes | 9 | 4,795 |
| Needle_insertion_eye_trocar | ✅ Yes | 1 | 675 |
| OCT_insertion | ❌ No | 7 | 27,606 |
| part1 | ❌ No | 20 | 65,633 |
| **TOTAL** | - | **37** | **98,709** |

**Sensor Coverage:** 5.5% (5,470 / 98,709)

---

## ⚙️ Key Arguments

| Argument | Value | Description |
|----------|-------|-------------|
| `--sensor-enabled` | flag | Enable sensor encoder |
| `--fusion-strategy` | `concat` | How to fuse VL + sensor |
| `--sensor-lr` | `5e-4` | Sensor learning rate |
| `--sensor-loss-weight` | `2.0` | Weight for sensor samples |
| `--lr` | `1e-4` | Action expert LR |
| `--vl-lr` | `1e-5` | VL backbone LR (LoRA) |
| `--finetune-vl` | `lora` | VL fine-tuning mode |
| `--grad-accum-steps` | `8` | Gradient accumulation |
| `--batch-size` | `1` | Batch per GPU |

---

## 🎯 Fusion Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `concat` | Concatenate features | **Start here** - Simple, fast |
| `cross_attention` | Cross-attention fusion | If concat plateaus |
| `gated` | Gated fusion (adaptive) | Variable sensor quality |
| `none` | No fusion | Sensor only (experimental) |

---

## 📁 Important Files

### Code
- `5st_VLA_TRAIN_VL_Lora_with_sensor.py` - Training script ✅ NEW
- `model_with_sensor.py` - Model with sensor encoder
- `IntegratedDataset.py` - Unified dataset loader

### Documentation
- `TRAINING_GUIDE.md` - **Read this first!**
- `TRAINING_UPDATE_SUMMARY.md` - What changed
- `FINAL_DATASET_SUMMARY.md` - Dataset statistics
- `DATASET_README.md` - Dataset technical details

### Checkpoints
- `./checkpoints/qwen_vla_sensor.pt` - Latest
- `./checkpoints/qwen_vla_sensor_best.pt` - Best validation
- `./checkpoints/qwen_vla_sensor_final.pt` - Final

---

## 🔍 Monitoring (WandB)

**Key Metrics:**
- `train/loss_step` - Step-wise loss
- `train/sensor_ratio` - % of sensor samples
- `val/loss_epoch` - Validation loss (for best model)
- `train/grad_norm` - Should be < 10
- `system/gpu_mem_GB` - GPU memory usage

---

## 🐛 Common Issues

### Cache Not Found
```bash
# Build cache first!
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py --mode cache
```

### OOM (Out of Memory)
```bash
# Reduce batch size
--batch-size 1 --grad-accum-steps 16
```

### Sensor Not Learning
```bash
# Increase sensor loss weight
--sensor-loss-weight 5.0
```

---

## 📈 Training Steps

1. ✅ **Preprocess Data** (Done - 98,709 samples)
2. ⏳ **Build Cache** (Run once, 2-4 hours)
3. 🎯 **Train Model** (50-100 epochs)
4. 📊 **Monitor WandB** (Check metrics)
5. ✅ **Save Best Model** (Automatic)
6. 🚀 **Deploy** (Export and test)

---

## 💡 Quick Tips

1. **Always build cache first** (`--mode cache`)
2. **Start with `concat` fusion** (simplest, usually works)
3. **Monitor `train/sensor_ratio`** (should be ~5.5%)
4. **Use `--sensor-loss-weight 2.0`** (balance sensor learning)
5. **Check `train/grad_norm`** (should be < 10, else exploding gradients)

---

## 📞 Need Help?

- **Training:** See `TRAINING_GUIDE.md`
- **Dataset:** See `FINAL_DATASET_SUMMARY.md`
- **Updates:** See `TRAINING_UPDATE_SUMMARY.md`

---

**Status:** ✅ Ready for Training
**Next:** Build cache, then train!

```bash
# Step 1: Build cache
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py --mode cache

# Step 2: Train
torchrun --nproc_per_node=4 \
    5st_VLA_TRAIN_VL_Lora_with_sensor.py --mode train --sensor-enabled
```
