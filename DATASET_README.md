# Insertion VLA Dataset Integration - Complete Guide

## 📊 Dataset Overview

This repository supports training VLA models with **optional OCT/FPI sensor data** integration.

### Dataset Types

| Dataset | Sensor Data | Status | Samples |
|---------|------------|--------|---------|
| `White_silicone_white_circle` | ✅ Yes (650Hz) | ✅ Preprocessed | ~5,000 |
| `Needle_insertion_eye_trocar` | ✅ Yes (650Hz) | ✅ Preprocessed | ~500 |
| `OCT_insertion` | ❌ No | ℹ️ Pre-existing | TBD |
| `part1` | ❌ No | ℹ️ Pre-existing | TBD |
| `Bridge v2` | ❌ No | ℹ️ External | TBD |

---

## 🔧 Preprocessing Pipeline

### Step 1: Data Collection
Raw data is collected using `Make_dataset/Total_reciver.py`:
```
recv_all_YYYYMMDD_HHMMSS/
├── View1/left/*.jpg, View1/right/*.jpg  # ZED camera 1
├── View2/left/*.jpg, View2/right/*.jpg  # ZED camera 2
├── View3/left/*.jpg, View3/right/*.jpg  # ZED camera 3
├── View4/left/*.jpg, View4/right/*.jpg  # ZED camera 4
├── View5/*.jpg                          # OAK camera
├── robot_state_*.csv                    # Robot states (100Hz)
└── sensor_data_*.npz                    # OCT/FPI sensor (650Hz)
```

### Step 2: Preprocessing
Run `preprocess_sensor_dataset.py` to generate JSON files:

```bash
# Process all sessions
python3 preprocess_sensor_dataset.py

# Process single session
python3 preprocess_sensor_dataset.py --single_session /path/to/recv_all_*
```

**What it does:**
1. Matches each image to the closest robot state (within 10ms)
2. **Extracts sensor data for the INTERVAL between consecutive images**
   - Not a single timestamp, but the entire duration!
   - Example: If images are 1 second apart @ 650Hz → ~650 sensor samples
3. Auto-generates instructions from folder names
4. Handles missing right camera images
5. Generates JSON files per camera view

### Generated JSON Structure
```json
[
  {
    "timestamp": 1761551500.174,
    "image": "/absolute/path/to/image.jpg",
    "robot_state": {
      "timestamp": 1761551500.174,
      "joint_angles": [j1, j2, j3, j4, j5, j6],
      "ee_pose": [x, y, z, a, b, r]
    },
    "time_diff_robot": 0.0,
    "sensor_interval": {              // ← NEW!
      "start": 1761551499.139,
      "end": 1761551500.174,
      "duration": 1.035
    }
  }
]
```

---

## 💾 Dataset Loading

### Option 1: Single Dataset
```python
from IntegratedDataset import insertionMeca500DatasetWithSensor

dataset = insertionMeca500DatasetWithSensor(
    trajectory_dir="/path/to/recv_all_20251027_165107",
    horizon=8,
    sensor_window_size=650  # Number of sensor samples
)

sample = dataset[0]
# sample['images']: List of image paths
# sample['actions']: (8, 7) action sequence
# sample['sensor_data']: (650, 1026) sensor data [force, A-scan]
# sample['instruction']: Language instruction
```

### Option 2: Integrated DataLoader
```python
from IntegratedDataset import create_integrated_dataloader

dataloader = create_integrated_dataloader(
    trajectory_dirs=[
        "/path/to/White_silicone/recv_all_*",  # With sensor
        "/path/to/OCT_insertion/Captures1",     # No sensor
    ],
    batch_size=4,
    horizon=8,
    shuffle=True,
    num_workers=4
)

for batch in dataloader:
    images = batch['images']           # List[List[str]]
    actions = batch['actions']         # (B, 8, 7)
    sensor_data = batch['sensor_data'] # (B, 650, 1026) or None
    instructions = batch['instruction'] # List[str]
```

---

## 🔬 Sensor Data Details

### Format
- **Shape**: `(650, 1026)`
  - 650 samples @ 650Hz = 1 second
  - 1026 dimensions = 1 force + 1025 A-scan points

### Extraction Method
**Interval-based (NEW):**
- Extract all sensor data between consecutive images
- Example: Images at t=0s and t=1s → sensor data from [0s, 1s)
- Automatically adjusts for varying image intervals (0.5s, 1.0s, etc.)

### Sensor Data Flow
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

---

## 🎯 Language Instructions

Auto-generated from folder names:

| Folder | Instruction |
|--------|-------------|
| `White_silicone_white_circle` | "Insert into the white square silicone with a white circle sticker attached" |
| `Needle_insertion_eye_trocar` | "Insert the needle through the trocar into the eye phantom model" |
| Others | "Perform the insertion task" |

---

## 📁 File Organization

```
Insertion_VLA/
├── preprocess_sensor_dataset.py          # Main preprocessing script
├── IntegratedDataset.py                  # Dataset classes
├── model_with_sensor.py                  # VLA model with sensor encoder
├── 5st_VLA_TRAIN_VL_Lora.py             # Training script (to be updated)
│
├── dataset/
│   ├── White_silicone_white_circle/
│   │   ├── recv_all_20251027_165107/
│   │   │   ├── View*/left/*.jpg, right/*.jpg
│   │   │   ├── robot_state_*.csv
│   │   │   ├── sensor_data_*.npz         # ← Sensor data
│   │   │   └── *_View*_single.json       # ← Generated JSONs
│   │   └── recv_all_20251027_165241/
│   │       └── ...
│   │
│   ├── Needle_insertion_eye_trocar/
│   │   └── recv_all_20251027_172553/
│   │       └── ...
│   │
│   ├── OCT_insertion/Captures*/          # No sensor
│   └── part1/ZED_Captures_*/             # No sensor
│
└── Make_dataset/
    ├── Total_reciver.py                  # Data collection
    ├── Robot_sender.py
    └── Improved_Jetson_sender.py
```

---

## ✅ Preprocessing Results

**Completed Sessions:**
```
White_silicone_white_circle: 9 sessions
Needle_insertion_eye_trocar: 1 session
Total: 10 sessions, 5470 samples
```

**Camera Views per Session:**
- View1_left, View1_right (if exists)
- View2_left, View2_right (if exists)
- View3_left, View3_right (if exists)
- View4_left, View4_right (if exists)
- View5_oak (OAK camera)

**Image Intervals:**
- White_silicone: ~1.0s (mean), ~650 sensor samples
- Needle_insertion: ~0.5s (mean), ~325 sensor samples

---

## 🚀 Next Steps

1. **Update Training Script** (`5st_VLA_TRAIN_VL_Lora.py`)
   ```python
   # OLD
   from model import Not_freeze_QwenVLAForAction

   # NEW
   from model_with_sensor import Not_freeze_QwenVLAWithSensor
   ```

2. **Configure Model**
   ```python
   model = Not_freeze_QwenVLAWithSensor(
       vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
       action_dim=7,
       horizon=8,
       sensor_enabled=True,           # ← Enable sensor
       sensor_input_channels=1026,    # Force + A-scan
       sensor_temporal_length=650,    # 1 second @ 650Hz
       fusion_strategy='concat'       # or 'cross_attention', 'gated'
   )
   ```

3. **Use Integrated DataLoader**
   ```python
   from IntegratedDataset import create_integrated_dataloader

   train_loader = create_integrated_dataloader(
       trajectory_dirs=[
           "/path/to/White_silicone/recv_all_*",
           "/path/to/Needle_insertion/recv_all_*",
           "/path/to/OCT_insertion/Captures*",  # No sensor (will use zeros)
       ],
       batch_size=4,
       horizon=8
   )
   ```

4. **Train!**
   ```bash
   python3 5st_VLA_TRAIN_VL_Lora.py --sensor_enabled
   ```

---

## 🔍 Validation

### Check Sensor Data Loading
```python
from IntegratedDataset import insertionMeca500DatasetWithSensor

dataset = insertionMeca500DatasetWithSensor(
    trajectory_dir="/path/to/recv_all_*",
    horizon=8
)

sample = dataset[0]
sensor = sample['sensor_data']  # (650, 1026)

print(f"Shape: {sensor.shape}")
print(f"Non-zero samples: {(sensor.abs().sum(dim=1) > 0).sum()}")
print(f"Mean force: {sensor[:, 0].mean()}")
print(f"Mean A-scan: {sensor[:, 1:].mean()}")
```

**Expected Output:**
```
Shape: torch.Size([650, 1026])
Non-zero samples: 500-650  # Depends on interval
Mean force: ~1.0
Mean A-scan: ~50.0
```

---

## 📝 Notes

1. **Sensor Window Size**: Currently fixed at 650 samples (1 second @ 650Hz)
   - Can be adjusted via `sensor_window_size` parameter
   - Data is padded/truncated to this size

2. **Missing Sensor Data**:
   - Datasets without sensor (OCT_insertion, part1, Bridge) will have `sensor_data=None`
   - Model should handle this gracefully (see `collate_fn_with_sensor`)

3. **Right Camera Images**:
   - Some sessions have right camera images, some don't
   - Preprocessing automatically detects and handles this

4. **Instruction Generation**:
   - Based on parent folder name
   - Can be manually overridden in dataset constructor

---

## 🐛 Troubleshooting

### Issue: "No JSON trajectory files found"
- **Solution**: Run `preprocess_sensor_dataset.py` first

### Issue: "sensor_data is all zeros"
- **Check**: Is `sensor_data_*.npz` present in the directory?
- **Check**: Does JSON contain `sensor_interval` field?

### Issue: "Image intervals too large/small"
- **Check**: Image timestamps in filenames
- **Adjust**: `sensor_window_size` parameter

---

## 📚 References

- **Data Collection**: `Make_dataset/Total_reciver.py`
- **Preprocessing**: `preprocess_sensor_dataset.py`
- **Dataset**: `IntegratedDataset.py`
- **Model**: `model_with_sensor.py`
- **Training**: `5st_VLA_TRAIN_VL_Lora.py` (to be updated)

---

**Last Updated**: 2025-10-27
**Total Samples**: 5,470 (with sensor), TBD (without sensor)
**Status**: ✅ Ready for Training
