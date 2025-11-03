# 🚀 Training Ready - Unified VLA Script

## ✅ 준비 완료

모든 학습 스크립트가 통합되어 준비되었습니다!

### 1. Dataset Status
- ✅ **Priority datasets** (with sensor):
  - White_silicone_white_circle: 9 trajectories
  - Needle_insertion_eye_trocar: 1 trajectory
- ✅ **Regular datasets** (without sensor):
  - OCT_insertion: 7 trajectories
  - part1: 20 trajectories
- ✅ **New async datasets**: 5 task types (Blue, Eye, Green, Red, White, Yellow points)

### 2. Cache Status
- ✅ Cache directory exists: `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/`
- ✅ Cache files: ~8GB (already built)
- ✅ No need to rebuild cache

### 3. Model & Code Status
- ✅ Model initialization tested: 127.06M trainable / 3881.68M total params
- ✅ Dataset loading tested: All formats working
- ✅ Sample access tested: Images, actions, sensor data all accessible

## 🏋️ How to Start Training

### Option 1: Quick Start (Recommended)
```bash
cd /home/najo/NAS/VLA/Insertion_VLA
bash scripts/start_training.sh
```

This will start training with:
- 2 GPUs
- Batch size: 1 per GPU
- Gradient accumulation: 8 steps
- LoRA fine-tuning on VL model
- Sensor encoder enabled

### Option 2: Manual Start
```bash
cd /home/najo/NAS/VLA/Insertion_VLA

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --lr 1e-4 \
    --sensor-lr 5e-4 \
    --vl-lr 1e-5 \
    --sensor-enabled \
    --finetune-vl lora \
    --training-stage stage2
```

### Option 3: Single GPU (for testing)
```bash
cd /home/najo/NAS/VLA/Insertion_VLA

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_port=29500 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --batch-size 1 \
    --grad-accum-steps 16 \
    --lr 1e-4 \
    --sensor-enabled \
    --finetune-vl lora \
    --training-stage stage2
```

## 📊 Training Configuration

### Effective Batch Size Calculation
- GPUs: 2
- Batch per GPU: 1
- Gradient accumulation: 8
- **Effective batch size**: 2 × 1 × 8 = **16 samples**

### Learning Rates
- Base (Action Expert): `1e-4`
- Sensor Encoder: `5e-4` (higher because it's new)
- VL Model (LoRA): `1e-5` (lower to preserve pretrained knowledge)

### Scheduler
- Type: Trapezoidal (Warmup → Hold → Cosine Decay)
- Warmup: 3% of total steps
- Hold: 2% of total steps
- Min LR: `1e-6`

### Dataset Weighting
- Priority datasets (with sensor): **2x weight**
- Regular datasets (without sensor): 1x weight
- This ensures the model sees more sensor data

## 📁 Output Locations

### Checkpoints
- Latest: `./checkpoints/qwen_vla_sensor.pt`
- Best: `./checkpoints/qwen_vla_sensor_best.pt`
- Final: `./checkpoints/qwen_vla_sensor_final.pt`

### Logs
- WandB project: `QwenVLA-Sensor`
- Run name: `train_sensor_MMDD_HHMM`

## 🔍 Monitoring Training

### During Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor checkpoint directory
watch -n 10 'ls -lh checkpoints/'

# Check WandB (if configured)
# Visit: https://wandb.ai/your-project/QwenVLA-Sensor
```

### Key Metrics to Watch
1. **train/loss_step**: Should decrease steadily
2. **train/sensor_samples**: Number of samples with sensor data
3. **train/grad_norm**: Should be < 5.0 (gradient clipping at 1.0)
4. **val/loss_epoch**: Validation loss (saved as best model)

## ⚠️ Important Notes

### Memory Management
- Each GPU will use ~20-24GB VRAM
- If OOM occurs, reduce `--batch-size` to 1 (already at minimum)
- Or increase `--grad-accum-steps` to maintain effective batch size

### Training Time Estimates
- Dataset size: ~2000+ samples per epoch
- Steps per epoch: ~125 (with batch_size=1, grad_accum=8, 2 GPUs)
- Time per epoch: ~15-20 minutes (depends on cache hits)
- For 100 epochs: ~25-30 hours

### Resuming Training
The script automatically resumes from the latest checkpoint if found:
- Loads model weights
- Loads optimizer state
- Continues from last epoch
- Re-warms learning rate

### If Training Fails
1. Check error messages in terminal
2. Verify GPU memory: `nvidia-smi`
3. Check dataset loading: `python scripts/test_dataset_loading.py`
4. Run dry-run again: `bash scripts/dry_run_training.sh`

## 🎯 Expected Results

Based on previous test results:
- **Overall MSE**: 0.23-0.45 (depends on demonstration)
- **Correlation**: 0.97-1.0 (excellent trend matching)
- **First action MSE**: Usually lower than full horizon

### Per-dimension Performance (from test)
- Position (xyz): MSE 0.05-0.63
- Orientation (abr): MSE 0.09-0.62
- All dimensions show >0.97 correlation

## 🚀 Ready to Go!

Everything is set up and tested. You can now start training with:

```bash
cd /home/najo/NAS/VLA/Insertion_VLA
bash scripts/start_training.sh
```

Good luck with your training! 🎉
