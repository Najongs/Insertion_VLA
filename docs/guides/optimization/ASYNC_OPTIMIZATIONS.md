# Async VLA Optimizations

## 개요

비동기 VLA 모델의 실시간 성능을 위한 최적화 설정

## 목표 성능

- **Action Expert**: 10Hz (100ms period)
- **VLM Update**: ~2.6Hz (381ms with 640x360 @ 5 views)
- **VL Feature Reuse**: 4x (VLM updates every 400ms)
- **Total Output**: 80 actions/second (10Hz × 8-horizon)

## 주요 최적화

### 1. 이미지 해상도 최적화 (640x360)

**테스트 결과 (5 views):**

| 해상도 | VLM 시간 | 주파수 | VL Reuse | 비고 |
|--------|----------|--------|----------|------|
| 1280x720 | 1487ms | 0.67 Hz | 14x | 너무 느림 |
| 960x540 | 801ms | 1.25 Hz | 8x | 여전히 느림 |
| **640x360** | **381ms** | **2.62 Hz** | **4x** | ✅ **권장** |
| 480x270 | 230ms | 4.35 Hz | 2x | 품질 저하 |

**선택: 640x360**
- VLM 추론: 381ms (~2.6 Hz)
- VL feature reuse: 4x
- 이미지 품질: 충분히 좋음
- Action Expert는 10Hz로 계속 동작하면서 VL features를 4번 재사용

**적용 방법:**
```python
# models/model_with_sensor.py 및 model_with_sensor_async.py
model = Not_freeze_QwenVLAWithSensor(
    ...,
    image_resize_height=360,
    image_resize_width=640,
)
```

### 2. Weighted Sampling for Priority Datasets

**우선순위 데이터셋:**
- `Needle_insertion_eye_trocar`: 2x weight
- `White_silicone_white_circle`: 2x weight

**적용 방법:**

Distributed training에서는 WeightedRandomSampler를 사용할 수 없으므로, 데이터셋을 2번 추가하는 방식으로 구현:

```python
# Priority datasets - added 2x
priority_dataset_dirs = [
    "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
    "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
]

for pattern in priority_dataset_dirs:
    for traj_dir in glob.glob(pattern):
        ds = insertionMeca500DatasetWithSensor(...)
        datasets.append(ds)  # 1st time
        datasets.append(ds)  # 2nd time (2x weight)
```

**결과:**
- 우선순위 데이터셋이 일반 데이터셋보다 2배 더 자주 샘플링됨
- 학습 시 우선순위 태스크에 더 많은 가중치

### 3. Diffusion Action Expert 최적화 (선택사항)

**Diffusion을 사용하는 경우:**

DDIM steps 프로파일링 결과:

| DDIM Steps | 시간 | 주파수 | 10Hz 충족 | 비고 |
|------------|------|--------|----------|------|
| 5 | 9ms | 111 Hz | ✅ | 빠르지만 품질 낮음 |
| 10 | 19ms | 54 Hz | ✅ | 괜찮음 |
| 20 | 35ms | 28 Hz | ✅ | 좋은 품질 |
| **50** | **89ms** | **11 Hz** | **✅** | **권장** |
| 100 | 184ms | 5 Hz | ❌ | 너무 느림 |

**권장: 50 DDIM steps**
- Action Expert 추론: 89ms
- 10Hz 요구사항 충족
- 좋은 샘플 품질

**적용 방법:**
```python
# 추론 시
sampled_actions = model.action_expert.sample(
    vl_tokens,
    sensor_features,
    batch_size=batch_size,
    ddim_steps=50  # 50 steps로 설정
)
```

## Async Training 설정

### A6_VLA_TRAIN_ASYNC.py 파라미터

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    training/A6_VLA_TRAIN_ASYNC.py \
    --sensor-window-size 65 \
    --vlm-reuse-count 4 \
    --image-resize-height 360 \
    --image-resize-width 640 \
    --batch-size 1 \
    --lr 1e-4 \
    --finetune-vl lora \
    --stage1-checkpoint ./checkpoints/stage1_best.pt
```

### 주요 파라미터 설명

- `--sensor-window-size 65`: 센서 윈도우 100ms @ 650Hz
- `--vlm-reuse-count 4`: VL features를 4번 재사용
- `--image-resize-height 360`: 이미지 높이 360px
- `--image-resize-width 640`: 이미지 너비 640px

## 성능 계산

### VLM Update Frequency

640x360 해상도에서 5 views:
- VLM 시간: 381ms
- VLM 주파수: 2.62 Hz
- VL feature reuse: 4x

### Action Expert Frequency

- Action Expert 기본 주파수: 10 Hz
- 센서 윈도우: 65 samples (100ms @ 650Hz)
- VL features 재사용: 4회

### 전체 시스템 동작

```
Time:  0ms   100ms  200ms  300ms  400ms  500ms  ...
VLM:   [----381ms----]      [----381ms----]
       │              │      │              │
Action: 1     2     3     4     5     6     7    8 ...
       (10Hz)(10Hz)(10Hz)(10Hz)(10Hz)(10Hz)(10Hz)

VL Reuse: 1x   2x    3x    4x    1x    2x    3x   4x
```

- VLM은 ~381ms마다 새로운 VL features 생성
- Action Expert는 매 100ms마다 동작 (10Hz)
- 각 VL feature는 4번 재사용

## 학습 단계

### Stage 1: Frozen VL + Trainable Sensor + Action Expert
```bash
# Stage 1은 기존 방식대로 (sensor_window_size=650)
python training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --finetune-vl none \
    --sensor-enabled
```

### Stage 2: Async VLA with LoRA
```bash
# Stage 2에서 async 설정 적용
torchrun --nproc_per_node=4 \
    training/A6_VLA_TRAIN_ASYNC.py \
    --sensor-window-size 65 \
    --vlm-reuse-count 4 \
    --image-resize-height 360 \
    --image-resize-width 640 \
    --finetune-vl lora \
    --stage1-checkpoint ./checkpoints/stage1_best.pt
```

## 파일 수정 요약

### 1. 모델 파일
- `models/model_with_sensor.py`: `image_resize_height/width` 파라미터 추가
- `models/model_with_sensor_async.py`: 동일 파라미터 추가
- `models/model_with_sensor_diffusion.py`: dtype 호환성 수정

### 2. 학습 스크립트
- `training/A6_VLA_TRAIN_ASYNC.py`:
  - Image resize 파라미터 추가
  - VLM reuse count 기본값 변경 (17 → 4)
  - Priority dataset weighted sampling 구현

### 3. 데이터셋
- `vla_datasets/IntegratedDataset.py`: `priority_datasets` 파라미터 추가 (WeightedRandomSampler 지원)

### 4. 유틸리티
- `utils/profile_diffusion_speed.py`: Diffusion action expert 프로파일링 도구 추가

## 메모리 최적화

640x360 해상도 사용으로 인한 메모리 절약:
- 원본 1280x720: 921,600 pixels
- 640x360: 230,400 pixels
- **픽셀 감소: 4x** → GPU 메모리 절약

## 추론 예시

```python
from models.model_with_sensor_async import create_async_model

# 모델 생성
model = create_async_model(
    stage1_checkpoint="./checkpoints/stage1_best.pt",
    finetune_vl="lora",
    sensor_window_size=65,
    vlm_reuse_count=4,
    image_resize_height=360,
    image_resize_width=640,
)

# VL features 추출 (381ms, 한 번만 실행)
vl_features = model.extract_vl_features(
    text_inputs=["Insert needle into target"],
    image_inputs=[image_paths],  # 5 views
)

# Action Expert 반복 실행 (10Hz, VL features 재사용)
for i in range(4):  # 4번 재사용
    sensor_window = get_sensor_window(65)  # 100ms window

    actions = model.predict_actions_with_cached_vl(
        vl_features=vl_features,
        z_chunk=prev_actions,
        sensor_data=sensor_window,
    )

    execute_actions(actions)
    time.sleep(0.1)  # 10Hz
```

## 결론

이 최적화 설정으로:
- ✅ VLM 추론 속도 4배 향상 (640x360 해상도)
- ✅ 우선순위 데이터셋에 2배 가중치
- ✅ 10Hz Action Expert 달성
- ✅ Diffusion 사용 시 50 DDIM steps 권장
- ✅ 메모리 사용량 감소

실시간 로봇 제어를 위한 최적의 균형점!
