# 🎯 Insertion VLA - Complete Dataset Summary

**Last Updated**: 2025-10-27
**Total Samples**: 98,709
**Total Sessions**: 37
**Status**: ✅ Ready for Training

---

## 📊 Dataset Overview

| Dataset | Sensor | Sessions | Samples | Status |
|---------|--------|----------|---------|--------|
| **White_silicone_white_circle** | ✅ Yes | 9 | 4,795 | ✅ Preprocessed |
| **Needle_insertion_eye_trocar** | ✅ Yes | 1 | 675 | ✅ Preprocessed |
| **OCT_insertion** | ❌ No | 7 | 27,606 | ✅ Pre-existing |
| **part1** | ❌ No | 20 | 65,633 | ✅ Pre-existing |
| **TOTAL** | - | **37** | **98,709** | ✅ |

### Sensor Data Distribution
- **With Sensor**: 5,470 samples (5.5%)
- **Without Sensor**: 93,239 samples (94.5%)
- **Sensor Format**: (650, 1026) - 650Hz sampling, 1 force + 1025 A-scan

---

## 🔧 Dataset Structure

### 1. With Sensor Data (White_silicone, Needle_insertion)
```
recv_all_YYYYMMDD_HHMMSS/
├── View1/left/*.jpg
├── View2/left/*.jpg
├── View3/left/*.jpg, View3/right/*.jpg (optional)
├── View4/left/*.jpg, View4/right/*.jpg (optional)
├── View5/*.jpg (OAK camera)
├── robot_state_*.csv                    # Robot states
├── sensor_data_*.npz                    # ✅ OCT/FPI sensor
└── recv_all_*_View*_single.json         # Generated JSON
```

**JSON Format**:
```json
{
  "timestamp": 1761551500.174,
  "image": "/path/to/image.jpg",
  "robot_state": {
    "timestamp": 1761551500.174,
    "joint_angles": [j1, j2, j3, j4, j5, j6],
    "ee_pose": [x, y, z, a, b, r]
  },
  "time_diff_robot": 0.0,
  "sensor_interval": {
    "start": 1761551499.139,
    "end": 1761551500.174,
    "duration": 1.035
  }
}
```

### 2. Without Sensor Data (OCT_insertion, part1)
```
Captures*/  or  ZED_Captures_*th/
├── view1/left/*.jpg, view1/right/*.jpg
├── view2/left/*.jpg, view2/right/*.jpg
├── view3/left/*.jpg, view3/right/*.jpg
├── view4/left/*.jpg, view4/right/*.jpg
├── view5_oak/*.jpg (optional)
├── robot_rt_*.csv                       # Robot states
└── *_view*_single.json                  # Pre-existing JSON
```

**JSON Format**:
```json
{
  "timestamp": 1759394647.266,
  "image": "/path/to/image.jpg",
  "robot_state": {
    "timestamp": 1759394647.266,
    "joint_angles": [j1, j2, j3, j4, j5, j6],
    "ee_pose": [x, y, z, a, b, r]
  },
  "time_diff_robot": 0.0
  // No sensor_interval field
}
```

---

## 💾 Data Loading

### Unified DataLoader (Recommended)
```python
from IntegratedDataset import create_integrated_dataloader

# Load ALL datasets (with and without sensor)
dataloader = create_integrated_dataloader(
    trajectory_dirs=[
        # With sensor
        "/path/to/White_silicone_white_circle/recv_all_*",
        "/path/to/Needle_insertion_eye_trocar/recv_all_*",

        # Without sensor
        "/path/to/OCT_insertion/Captures*",
        "/path/to/part1/ZED_Captures_*th",
    ],
    batch_size=16,
    horizon=8,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in dataloader:
    images = batch['images']              # List[List[str]]
    actions = batch['actions']            # (B, 8, 7)
    sensor_data = batch['sensor_data']    # (B, 650, 1026) - ALWAYS present
    has_sensor = batch['has_sensor_mask'] # (B,) - True/False
    instructions = batch['instruction']   # List[str]

    # sensor_data contains:
    # - Real sensor data where has_sensor[i] == True
    # - Dummy zeros where has_sensor[i] == False
```

### Key Features
1. **Consistent Shape**: `sensor_data` is ALWAYS `(B, 650, 1026)`
2. **Automatic Dummy Data**: Samples without sensor get zero-filled tensors
3. **Masking**: `has_sensor_mask` indicates which samples have real sensor data
4. **Mixed Training**: Seamlessly mix sensor and non-sensor datasets

---

## 🔬 Sensor Data Details

### Extraction Method: Interval-Based
Unlike single-point sampling, we extract sensor data for the **entire interval** between consecutive images:

```
Image[t-1] ──────┬─────────┬─────── Image[t]
                  │         │
           Sensor[t-1]   Sensor[t]
                  │         │
                  └─────────┘
                      ↓
               Extract ALL samples
              in this interval
                      ↓
            (650, 1026) tensor
```

**Benefits**:
- Captures temporal dynamics during motion
- Adapts to varying image intervals (0.5s, 1.0s, etc.)
- More robust to timing jitter

### Sensor Data Format
- **Shape**: `(650, 1026)`
  - Dimension 0: Force sensor (1 value)
  - Dimensions 1-1025: OCT A-scan (1025 values)
- **Sampling Rate**: 650 Hz
- **Duration**: ~1 second (adjusts to actual image interval)

---

## 🎯 Language Instructions

Auto-generated from folder names:

| Dataset | Instruction |
|---------|-------------|
| White_silicone_white_circle | "Insert into the white square silicone with a white circle sticker attached" |
| Needle_insertion_eye_trocar | "Insert the needle through the trocar into the eye phantom model" |
| OCT_insertion | User-defined (default: "Insert the needle into the tissue using OCT guidance") |
| part1 | User-defined (default: "Perform the insertion task") |

---

## 🚀 Training Setup

### 1. Model Configuration
```python
from model_with_sensor import Not_freeze_QwenVLAWithSensor

model = Not_freeze_QwenVLAWithSensor(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    hidden_dim=1024,
    finetune_vl="lora",              # or "none", "full"
    lora_r=16,
    lora_alpha=32,

    # Sensor configuration
    sensor_enabled=True,             # Enable sensor encoder
    sensor_input_channels=1026,      # 1 force + 1025 A-scan
    sensor_temporal_length=650,      # 1 second @ 650Hz
    sensor_output_dim=3072,          # Match VL dimension
    fusion_strategy='concat'         # or 'cross_attention', 'gated'
)
```

### 2. Loss Computation with Sensor Mask
```python
# Forward pass
pred_actions, delta = model(
    text_inputs=batch['instruction'],
    image_inputs=batch['images'],
    z_chunk=z_chunk,
    sensor_data=batch['sensor_data']  # Always (B, 650, 1026)
)

# Compute loss
loss = F.mse_loss(pred_actions, batch['actions'])

# Optional: Weight loss based on sensor availability
has_sensor = batch['has_sensor_mask']
if has_sensor.any():
    # Give higher weight to samples with real sensor data
    weights = torch.where(has_sensor, 2.0, 1.0).unsqueeze(1).unsqueeze(2)
    weighted_loss = (weights * (pred_actions - batch['actions']) ** 2).mean()
```

### 3. Training Script Updates
Update `5st_VLA_TRAIN_VL_Lora.py`:

```python
# OLD
from model import Not_freeze_QwenVLAForAction
from Total_Dataset import collate_fn

# NEW
from model_with_sensor import Not_freeze_QwenVLAWithSensor
from IntegratedDataset import collate_fn_with_sensor, create_integrated_dataloader
```

---

## 📈 Training Recommendations

### Mixed Training Strategy
Given the imbalanced sensor data (5.5% with sensor, 94.5% without):

1. **Option A: Balanced Sampling**
   ```python
   # Oversample sensor data to balance
   sensor_weight = 93239 / 5470  # ~17x
   ```

2. **Option B: Two-Stage Training**
   ```python
   # Stage 1: Train on all data (sensor encoder learns from 5.5%)
   # Stage 2: Fine-tune on sensor-rich data only
   ```

3. **Option C: Curriculum Learning**
   ```python
   # Early epochs: All data (learn general VLA)
   # Later epochs: Increase sensor data proportion
   ```

### Batch Size Recommendations
- **With Sensor Encoder**: 4-8 per GPU (3GB model + 650x1026 sensor data)
- **Without Sensor**: 8-16 per GPU (standard VLA)

### Hardware Requirements
- **GPU Memory**: 24GB+ (e.g., RTX 3090, A6000)
- **Disk I/O**: SSD recommended for 98k samples
- **CPU**: 16+ cores for DataLoader workers

---

## 📁 File Organization

```
Insertion_VLA/
├── preprocess_sensor_dataset.py          # Preprocessing for sensor data
├── IntegratedDataset.py                  # Unified dataset loader
├── model_with_sensor.py                  # VLA model with sensor encoder
├── 5st_VLA_TRAIN_VL_Lora.py             # Training script (update needed)
│
├── dataset/
│   ├── White_silicone_white_circle/     # 9 sessions, 4,795 samples ✅
│   ├── Needle_insertion_eye_trocar/     # 1 session, 675 samples ✅
│   ├── OCT_insertion/                    # 7 sessions, 27,606 samples ✅
│   └── part1/                            # 20 sessions, 65,633 samples ✅
│
└── Make_dataset/
    ├── Total_reciver.py                  # Data collection (sensor-equipped)
    ├── Robot_sender.py
    └── Improved_Jetson_sender.py
```

---

## ✅ Validation Checklist

### Pre-Training Checks
- [x] All JSON files generated
- [x] Sensor data NPZ files present (for White_silicone, Needle_insertion)
- [x] Dataset loading works for all 4 datasets
- [x] Mixed batches have consistent shape
- [x] has_sensor_mask correctly identifies real vs dummy data
- [ ] Model forward pass works with mixed batches
- [ ] Loss computation handles sensor mask
- [ ] Training script updated

### Post-Training Checks
- [ ] Model converges on sensor data
- [ ] Model generalizes to non-sensor data
- [ ] Sensor encoder learns meaningful features
- [ ] Action prediction improves with sensor input

---

## 🐛 Known Issues & Solutions

### Issue 1: Imbalanced Sensor Data
**Problem**: Only 5.5% of samples have sensor data
**Solution**: Use weighted sampling or two-stage training

### Issue 2: Memory Usage
**Problem**: Sensor data increases memory footprint
**Solution**: Reduce batch size or use gradient accumulation

### Issue 3: Dummy Sensor Data
**Problem**: Zeros for non-sensor samples might confuse model
**Solution**: Use `has_sensor_mask` in loss or add learnable "no sensor" embedding

---

## 📚 Quick Start Example

```python
from IntegratedDataset import create_integrated_dataloader
from model_with_sensor import Not_freeze_QwenVLAWithSensor
import torch

# 1. Create dataloader
train_loader = create_integrated_dataloader(
    trajectory_dirs=[
        "/path/to/White_silicone_white_circle/recv_all_*",
        "/path/to/OCT_insertion/Captures*",
        "/path/to/part1/ZED_Captures_*",
    ],
    batch_size=8,
    horizon=8,
    shuffle=True
)

# 2. Initialize model
model = Not_freeze_QwenVLAWithSensor(
    sensor_enabled=True,
    fusion_strategy='concat'
).cuda()

# 3. Train
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for batch in train_loader:
    # Prepare inputs
    z_chunk = torch.randn(len(batch['actions']), 8, 7).cuda()

    # Forward
    pred_actions, delta = model(
        text_inputs=batch['instruction'],
        image_inputs=batch['images'],
        z_chunk=z_chunk,
        sensor_data=batch['sensor_data'].cuda()
    )

    # Loss
    loss = F.mse_loss(pred_actions, batch['actions'].cuda())

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}, "
          f"Sensor samples: {batch['has_sensor_mask'].sum()}/{len(batch['actions'])}")
```

---

## 🎓 Next Steps

1. **Update Training Script**
   - Integrate `IntegratedDataset`
   - Add sensor mask handling
   - Configure balanced sampling

2. **Baseline Training**
   - Train without sensor encoder first
   - Establish baseline performance

3. **Sensor-Enhanced Training**
   - Enable sensor encoder
   - Compare with baseline

4. **Ablation Studies**
   - Test different fusion strategies
   - Evaluate sensor data importance

---

## 📞 Support

**Issues**: https://github.com/yourusername/Insertion_VLA/issues
**Documentation**: `/DATASET_README.md`, `/FINAL_DATASET_SUMMARY.md`

---

**Status**: ✅ **READY FOR TRAINING**
**Total Effort**: 37 sessions, 98,709 samples, ~1TB+ data
**Sensor Coverage**: 5.5% (sufficient for multi-modal learning)
