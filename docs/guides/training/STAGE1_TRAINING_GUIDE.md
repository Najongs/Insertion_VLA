# Stage 1 Training Guide (Optimized)

## 개요

Stage 1은 VL backbone을 frozen 상태로 유지하면서 Sensor Encoder와 Action Expert만 학습합니다.
**LoRA 없이 Stage 1만 진행하는 경우** 이 가이드를 따르세요.

## 최적화 적용

다음 최적화가 기본값으로 적용되었습니다:

1. **이미지 리사이즈: 640x360**
   - VLM 추론 시간 3.9배 단축 (1487ms → 381ms @ 5 views)
   - 캐시 빌드 시간도 단축

2. **센서 윈도우: 65 samples**
   - 100ms @ 650Hz (기존 650 samples = 1초)
   - 실시간 제어에 적합한 짧은 윈도우

3. **Weighted Sampling**
   - Priority datasets (Needle_insertion_eye_trocar, White_silicone_white_circle): **2x 가중치**
   - Regular datasets (OCT_insertion, part1): 1x 가중치

## 학습 단계

### 1단계: VL Feature Cache 빌드

먼저 VL features를 캐시로 저장합니다 (학습 속도 향상).

```bash
# 4개 GPU 사용 예시
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode cache \
    --sensor-enabled \
    --sensor-window-size 65 \
    --image-resize-height 360 \
    --image-resize-width 640
```

**예상 시간:**
- 기존 (1280x720): ~2-3시간
- 최적화 (640x360): ~30-45분 (약 4배 빠름)

**출력 예시:**
```
🚀 [Rank 0] Running in CACHE mode on 4 GPUs
🔬 Stage 1 Optimized Training
   - Sensor enabled: True
   - Sensor window: 65 samples (100ms @ 650Hz)
   - Image resize: 640x360
   - Priority datasets: 2x weight (Needle_insertion, White_silicone)

📦 Building integrated dataset...
✅ [Priority 2x] Added: recv_all_20251027_170308 (500 samples, WITH sensor)
✅ [Priority 2x] Added: recv_all_20251028_141523 (600 samples, WITH sensor)
✅ Added: Captures1 (800 samples, NO sensor)
✅ Added: ZED_Captures_4th (700 samples, NO sensor)

📊 Total dataset size: 4400 samples (priority datasets counted 2x)

⏳ Initializing VL-only model for cache building...
   Image resize: 640x360
   📐 Cache will use 640x360 images (230,400 pixels)

🔄 Building VL cache (distributed)...
[Rank 0] Processing batch 100/1100...
...
✅ Cache build complete. You can now run training with --mode train.
```

### 2단계: Stage 1 학습

캐시가 준비되면 Sensor Encoder와 Action Expert를 학습합니다.

```bash
# 4개 GPU 사용 예시
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --sensor-enabled \
    --sensor-window-size 65 \
    --image-resize-height 360 \
    --image-resize-width 640 \
    --finetune-vl none \
    --training-stage stage1 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --lr 1e-4 \
    --sensor-lr 5e-4 \
    --sensor-loss-weight 2.0
```

**주요 파라미터:**

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `--mode` | train | 학습 모드 |
| `--finetune-vl` | none | VL backbone frozen (LoRA 없음) |
| `--training-stage` | stage1 | Stage 1 학습 |
| `--sensor-window-size` | 65 | 100ms @ 650Hz |
| `--image-resize-height` | 360 | 이미지 높이 |
| `--image-resize-width` | 640 | 이미지 너비 |
| `--sensor-lr` | 5e-4 | Sensor encoder learning rate |
| `--sensor-loss-weight` | 2.0 | Sensor 데이터 loss 가중치 |
| `--batch-size` | 1 | GPU당 배치 크기 |
| `--grad-accum-steps` | 8 | Gradient accumulation |

**출력 예시:**
```
🚀 [Rank 0] Running in TRAIN mode on 4 GPUs
🔬 Stage 1 Optimized Training
   - Sensor enabled: True
   - Sensor window: 65 samples (100ms @ 650Hz)
   - Fusion strategy: concat
   - Sensor LR: 0.0005
   - Sensor loss weight: 2.0
   - Image resize: 640x360
   - Priority datasets: 2x weight (Needle_insertion, White_silicone)

📦 Building integrated dataset...
✅ [Priority 2x] Added: recv_all_20251027_170308 (500 samples, WITH sensor)
✅ [Priority 2x] Added: recv_all_20251028_141523 (600 samples, WITH sensor)
✅ Added: Captures1 (800 samples, NO sensor)
✅ Added: ZED_Captures_4th (700 samples, NO sensor)

📊 Total dataset size: 4400 samples
   Train: 4180 samples
   Val: 220 samples

⏳ Initializing full QwenVLA model for training...
🚀 Loading Trainable Qwen-VL-Sensor Model
   VL Fine-tuning: none
   Sensor Enabled: True
   Fusion Strategy: concat
   📐 Image resize: 640x360 (230,400 pixels)
🧊 Using frozen VL backbone.
✅ Model loaded

💡 Trainable parameters:
   - Action Expert: 12.3M
   - Sensor Encoder: 8.5M
   - Total trainable: 20.8M
   - Total frozen: 3.0B (VL backbone)

Epoch 1/100:
  [Step 100/522] loss: 0.012456, lr: 1.2e-5, grad: 0.82, sensor: 120/200
  [Step 200/522] loss: 0.009123, lr: 2.4e-5, grad: 0.65, sensor: 240/400
  ...

📊 Epoch 1 | Train: 0.008234 | Val: 0.009123
🏆 [Best] Validation improved → saved to ./checkpoints/qwen_vla_sensor_best.pt
```

## 학습 시간 예상

**GPU: 4x RTX 4090**
- Epoch당: ~15-20분
- Total (100 epochs): ~25-30시간

**GPU: 4x A100**
- Epoch당: ~10-12분
- Total (100 epochs): ~16-20시간

## 체크포인트

학습 중 체크포인트가 자동으로 저장됩니다:

- `./checkpoints/qwen_vla_sensor.pt`: 최신 체크포인트 (매 epoch)
- `./checkpoints/qwen_vla_sensor_best.pt`: Best validation loss
- `./checkpoints/qwen_vla_sensor_final.pt`: 최종 체크포인트

## 학습 재개

학습이 중단된 경우, 같은 명령어로 자동 재개됩니다:

```bash
# 자동으로 ./checkpoints/qwen_vla_sensor.pt에서 재개
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --sensor-enabled \
    --sensor-window-size 65 \
    --image-resize-height 360 \
    --image-resize-width 640 \
    --finetune-vl none \
    --training-stage stage1
```

## 검증

학습이 완료되면 체크포인트를 로드하여 추론 테스트:

```python
from models.model_with_sensor import Not_freeze_QwenVLAWithSensor

# 모델 로드
model = Not_freeze_QwenVLAWithSensor(
    vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    action_dim=7,
    horizon=8,
    finetune_vl="none",
    sensor_enabled=True,
    sensor_temporal_length=65,
    image_resize_height=360,
    image_resize_width=640,
)

# 체크포인트 로드
checkpoint = torch.load("./checkpoints/qwen_vla_sensor_best.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 추론
with torch.no_grad():
    actions, _ = model(
        text_inputs=["Insert needle into target"],
        image_inputs=[image_paths],  # 5 views
        z_chunk=torch.zeros(1, 8, 7),
        sensor_data=sensor_window,  # (1, 65, 1026)
    )
```

## WandB 모니터링

학습 중 WandB에 자동으로 로그가 기록됩니다:

- `train/loss_step`: Step별 학습 loss
- `train/loss_epoch`: Epoch별 평균 loss
- `val/loss_epoch`: Validation loss
- `train/lr`: Learning rate
- `train/grad_norm`: Gradient norm
- `train/sensor_samples`: Sensor 데이터 사용 샘플 수
- `train/nonsensor_samples`: Sensor 없는 샘플 수

## 트러블슈팅

### 1. CUDA Out of Memory

배치 크기 줄이기:
```bash
--batch-size 1 --grad-accum-steps 16  # 더 작은 배치, 더 많은 accumulation
```

### 2. Cache Miss Error

캐시를 다시 빌드:
```bash
rm -rf /home/najo/NAS/VLA/dataset/cache/qwen_vl_features/*
# 그 다음 cache 모드 다시 실행
```

### 3. Dataset 로딩 실패

데이터 경로 확인:
```python
# A5st_VLA_TRAIN_VL_Lora_with_sensor.py에서 수정
priority_dataset_dirs = [
    "YOUR_PATH/White_silicone_white_circle/recv_all_*",
    "YOUR_PATH/Needle_insertion_eye_trocar/recv_all_*",
]
```

## 다음 단계

Stage 1 학습 완료 후:

1. **Best checkpoint 선택**: `qwen_vla_sensor_best.pt` 사용 권장
2. **실제 로봇 테스트**: `examples/example_sensor_vla_usage.py` 참고
3. **추가 fine-tuning 필요 시**: Stage 2 (LoRA) 고려

## 요약

✅ **최적화 적용 완료:**
- 640x360 이미지 → 4배 빠른 학습
- 65 samples 센서 윈도우 → 실시간 제어 가능
- Priority datasets 2x → 더 나은 성능

✅ **Stage 1 only:**
- Frozen VL backbone (3B parameters)
- Trainable Sensor + Action Expert (20.8M parameters)
- LoRA 없이 빠른 학습 완료

준비되었습니다! 🚀
