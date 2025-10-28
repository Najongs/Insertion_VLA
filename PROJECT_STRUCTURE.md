# 프로젝트 구조 정리

## 📁 디렉토리 구조

```
Insertion_VLA/
├── models/                          # 모델 정의
│   ├── __init__.py
│   ├── model.py                     # 원본 VLA 모델 (센서 없음)
│   └── model_with_sensor.py         # 센서 통합 VLA 모델
│
├── datasets/                        # 데이터셋 정의
│   ├── __init__.py
│   ├── IntegratedDataset.py         # 센서 데이터가 있거나 없는 통합 데이터셋
│   └── Total_Dataset.py             # 기타 데이터셋
│
├── training/                        # 학습 스크립트
│   ├── __init__.py
│   ├── train_encoder_action.py      # ✨ NEW: Encoder + Action Expert만 학습
│   ├── A5st_VLA_TRAIN_VL_Lora_with_sensor.py  # LoRA 학습 (전체)
│   ├── A5st_VLA_TRAIN_VL_Lora.py    # LoRA 학습 (센서 없음)
│   └── Make_VL_cache.py             # VL 피처 캐싱
│
├── preprocessing/                   # 전처리 스크립트
│   ├── __init__.py
│   ├── preprocess_sensor_dataset.py # 센서 데이터 전처리
│   └── preprocess_white_silicone.py # 특정 데이터셋 전처리
│
├── examples/                        # 사용 예제
│   ├── example_sensor_vla_usage.py  # 센서 VLA 사용 예제
│   └── test_sensor_model.py         # 모델 테스트
│
├── utils/                           # 유틸리티 함수
│   └── __init__.py
│
├── Make_dataset/                    # 데이터 수집 관련
│   ├── Robot_sender.py
│   ├── Total_reciver.py
│   └── Improved_Jetson_sender.py
│
├── checkpoints/                     # 학습된 모델 체크포인트
│
└── 기타 설정 파일들...
```

---

## 🚀 주요 학습 스크립트

### 1. **Encoder + Action Expert만 학습** (VL Backbone Frozen) ✨ NEW

**파일**: `training/train_encoder_action.py`

**특징**:
- ✅ VL Backbone 완전 Frozen (학습 안 함)
- ✅ SensorEncoder만 학습
- ✅ ActionExpert만 학습
- ✅ LoRA 적용 안 함

**사용법**:

```bash
# 1단계: VL 피처 캐시 생성 (한 번만 실행)
torchrun --nproc_per_node=4 training/train_encoder_action.py --mode cache

# 2단계: Encoder + Action Expert 학습
torchrun --nproc_per_node=4 training/train_encoder_action.py \
    --mode train \
    --sensor-enabled \
    --fusion-strategy concat \
    --batch-size 2 \
    --lr 5e-4 \
    --sensor-lr 5e-4 \
    --epochs 100
```

**주요 옵션**:
- `--mode`: `cache` (캐시 생성) 또는 `train` (학습)
- `--sensor-enabled`: 센서 인코더 활성화
- `--fusion-strategy`: 센서-VL 융합 전략 (`concat`, `cross_attention`, `gated`, `none`)
- `--lr`: Action Expert 학습률
- `--sensor-lr`: Sensor Encoder 학습률
- `--sensor-loss-weight`: 센서 데이터가 있는 샘플의 손실 가중치 (기본값: 2.0)
- `--batch-size`: 배치 크기
- `--epochs`: 총 에포크 수

---

### 2. **LoRA 학습** (VL Backbone Fine-tuning 포함)

**파일**: `training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py`

**특징**:
- ✅ VL Backbone에 LoRA 적용
- ✅ SensorEncoder 학습
- ✅ ActionExpert 학습

**사용법**:

```bash
# 캐시 생성
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py --mode cache

# LoRA 학습
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --finetune-vl lora \
    --sensor-enabled \
    --batch-size 2 \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --sensor-lr 5e-4
```

---

## 📦 모델 구조

### QwenVLAWithSensor (Frozen VL)
- `model_with_sensor.py:QwenVLAWithSensor`
- VL Backbone: **Frozen** (학습 안 함)
- Sensor Encoder: **Trainable**
- Action Expert: **Trainable**

### Not_freeze_QwenVLAWithSensor (LoRA/Full Fine-tuning)
- `model_with_sensor.py:Not_freeze_QwenVLAWithSensor`
- VL Backbone: **LoRA 또는 Full Fine-tuning**
- Sensor Encoder: **Trainable**
- Action Expert: **Trainable**

---

## 📊 데이터셋 구조

### 센서 데이터가 있는 데이터셋
- `White_silicone_white_circle/recv_all_*`
- `Needle_insertion_eye_trocar/recv_all_*`
- 센서 파일: `sensor_data_*.npz`
- 형식: `(650, 1026)` - 650Hz x 1초, 1 force + 1025 A-scan

### 센서 데이터가 없는 데이터셋
- `OCT_insertion/Captures*`
- `part1/ZED_Captures_*th`
- 센서 데이터 없음 → 모델은 자동으로 zero padding 처리

---

## 🔧 Import 경로 변경

### 기존 (Root에서 import)
```python
from model_with_sensor import QwenVLAWithSensor
from IntegratedDataset import collate_fn_with_sensor
```

### 변경 후 (패키지에서 import)
```python
from models.model_with_sensor import QwenVLAWithSensor
from datasets.IntegratedDataset import collate_fn_with_sensor
```

또는

```python
from models import QwenVLAWithSensor
from datasets import collate_fn_with_sensor
```

---

## 💡 학습 전략 비교

| 학습 방식 | VL Backbone | Sensor Encoder | Action Expert | 학습 속도 | 권장 상황 |
|----------|-------------|----------------|---------------|----------|----------|
| **Encoder+Action** | ❄️ Frozen | ✅ Train | ✅ Train | ⚡ 빠름 | 센서 데이터 활용, 빠른 실험 |
| **LoRA** | 🔥 LoRA | ✅ Train | ✅ Train | 🐢 중간 | VL 백본도 일부 학습 필요 |
| **Full** | 🔥 Full | ✅ Train | ✅ Train | 🐌 느림 | 전체 모델 최적화 |

---

## 📝 체크포인트 위치

- **Encoder+Action 학습**: `checkpoints/encoder_action.pt`
- **Best 모델**: `checkpoints/encoder_action_best.pt`
- **Final 모델**: `checkpoints/encoder_action_final.pt`
- **LoRA 학습**: `checkpoints/qwen_vla_sensor.pt`

---

## 🎯 권장 학습 순서

1. **1단계**: VL 피처 캐시 생성 (한 번만)
   ```bash
   torchrun --nproc_per_node=4 training/train_encoder_action.py --mode cache
   ```

2. **2단계**: Encoder + Action Expert 학습 (빠른 실험)
   ```bash
   torchrun --nproc_per_node=4 training/train_encoder_action.py --mode train --sensor-enabled
   ```

3. **3단계 (선택)**: LoRA로 전체 모델 Fine-tuning
   ```bash
   torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py --mode train --finetune-vl lora
   ```

---

## 📌 주요 변경 사항

### ✨ 새로 추가된 것
- `training/train_encoder_action.py`: Encoder + Action Expert만 학습하는 스크립트
- 패키지 구조로 재구성 (`models/`, `datasets/`, `training/` 등)

### 🔄 기존 파일 유지
- 루트 디렉토리의 기존 파일들은 그대로 유지됨
- 새로운 패키지 디렉토리는 복사본으로 생성됨

---

## ⚠️ 주의사항

1. **Import 경로**: 새로운 패키지 구조를 사용할 때는 import 경로를 수정해야 합니다
2. **체크포인트 호환성**: 다른 학습 스크립트로 만든 체크포인트는 호환되지 않을 수 있습니다
3. **캐시 공유**: 모든 학습 스크립트는 동일한 VL 피처 캐시를 공유합니다

---

## 🐛 문제 해결

### ImportError 발생 시
```bash
# Python path 설정
export PYTHONPATH="${PYTHONPATH}:/home/najo/NAS/VLA/Insertion_VLA"
```

### GPU 메모리 부족 시
- `--batch-size`를 줄이기 (예: 2 → 1)
- `--grad-accum-steps`를 늘리기 (예: 8 → 16)

---

## 📧 문의

문제가 발생하면 다음을 확인하세요:
1. 체크포인트가 올바른 위치에 있는지
2. Import 경로가 올바른지
3. 캐시가 생성되었는지 (`/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/`)
