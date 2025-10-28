# Documentation Index

Complete documentation for Insertion VLA project.

## 📖 Active Documentation

### User Guides
- **[2-Stage Training Guide](guides/2STAGE_TRAINING.md)** - Complete guide for Stage 1 → Stage 2 training with LoRA
- **[Hugging Face Model Sharing](guides/HUGGINGFACE_GUIDE.md)** - Upload/download models to Hugging Face Hub
- **[Diffusion Quick Start](guides/DIFFUSION_QUICKSTART.md)** - Get started with diffusion policy training
- **[Diffusion Training Guide](guides/README_DIFFUSION.md)** - Detailed diffusion model training and comparison

### Main Documentation
- **[Project README](../README.md)** - Main project documentation with quick start

## 🗂️ Document Organization

```
docs/
├── README.md                           # This file
├── guides/                             # Active user guides
│   ├── DIFFUSION_QUICKSTART.md        # Diffusion quick start
│   └── README_DIFFUSION.md            # Diffusion training details
│
└── archive/                            # Outdated/superseded docs
    ├── DATASET_README.md              # Old dataset documentation
    ├── FINAL_DATASET_SUMMARY.md       # Dataset summary (superseded)
    ├── TRAINING_GUIDE.md              # Old training guide
    ├── TRAINING_UPDATE_SUMMARY.md     # Training updates (superseded)
    ├── QUICK_REFERENCE.md             # Old quick reference
    └── PROJECT_STRUCTURE.md           # Old structure doc
```

## 📚 Documentation by Topic

### Getting Started
1. Read [Project README](../README.md)
2. Install dependencies
3. Run tests: `python examples/test_sensor_model.py`
4. Follow [Quick Start](../README.md#-quick-start)

### Data Collection
- Main: [Project README - Data Collection](../README.md#2-data-collection)
- Scripts: `Make_dataset/Total_reciver.py`, `Improved_Jetson_sender.py`, `Robot_sender.py`

### Training

#### Regression Training (Baseline)
- Main README: [Training Section](../README.md#3-training)
- Script: `training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py`
- Quick command:
  ```bash
  torchrun --nproc_per_node=4 \
      training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
      --dataset_dir /path/to/dataset
  ```

#### Diffusion Training (Advanced)
- **Quick Start**: [guides/DIFFUSION_QUICKSTART.md](guides/DIFFUSION_QUICKSTART.md)
- **Detailed Guide**: [guides/README_DIFFUSION.md](guides/README_DIFFUSION.md)
- Script: `training/A5st_VLA_TRAIN_Diffusion_with_sensor.py`
- Quick command:
  ```bash
  torchrun --nproc_per_node=4 \
      training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
      --dataset_dir /path/to/dataset \
      --diffusion_timesteps 100
  ```

### Real-time Inference
- Main: [Project README - Real-time Inference](../README.md#4-real-time-inference)
- Script: `Make_dataset/Realtime_inference_receiver.py`
- Features:
  - Adaptive inference rate
  - Multi-view image buffer
  - Sensor circular buffer (650 samples @ 650Hz)
  - Optional data saving

### Model Implementation
- **Regression Model**: `models/model_with_sensor.py`
- **Diffusion Model**: `models/model_with_sensor_diffusion.py`
- **Dataset Loader**: `vla_datasets/IntegratedDataset.py`

### Testing
- Sensor model: `python examples/test_sensor_model.py`
- Diffusion model: `python examples/test_diffusion_model.py`

## 🔄 Document Update History

### 2024-10-28 (Latest)
- ✅ Created unified README.md
- ✅ Organized documentation into docs/ folder
- ✅ Moved active guides to docs/guides/
- ✅ Archived outdated documentation
- ✅ Added diffusion policy documentation

### Previous (Archived)
- Multiple overlapping README files
- No clear organization
- Outdated information mixed with current

## 📋 Quick Reference

### File Structure
```
Insertion_VLA/
├── README.md                    # 👈 START HERE
├── docs/
│   ├── README.md               # 👈 Documentation index
│   ├── guides/                 # Active guides
│   └── archive/                # Old docs
├── models/                     # Model implementations
├── training/                   # Training scripts
├── Make_dataset/               # Data collection & inference
├── vla_datasets/               # Dataset loaders
└── examples/                   # Test scripts
```

### Common Commands
```bash
# Test
python examples/test_sensor_model.py

# Train (Regression)
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py --dataset_dir /path/to/dataset

# Train (Diffusion)
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Diffusion_with_sensor.py --dataset_dir /path/to/dataset

# Inference
python Make_dataset/Realtime_inference_receiver.py --checkpoint checkpoints/best_model.pth --save-data
```

## 🆘 Getting Help

1. **Check main README**: [../README.md](../README.md)
2. **For diffusion**: [guides/DIFFUSION_QUICKSTART.md](guides/DIFFUSION_QUICKSTART.md)
3. **Check examples**: Run test scripts in `examples/`
4. **Review archived docs**: May contain additional details

## 📝 Contributing to Documentation

When updating documentation:
1. Update main [README.md](../README.md) for general changes
2. Add specialized guides to `docs/guides/`
3. Move outdated files to `docs/archive/`
4. Update this index file
5. Keep documentation clear and concise

---

**Last Updated**: 2024-10-28
**Maintained By**: Project Team
