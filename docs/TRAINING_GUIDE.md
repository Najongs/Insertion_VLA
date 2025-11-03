# Training Guide

## 학습 순서 (중요!)

### Step 0: 데이터셋 전처리 (최초 1회만)

**데이터 로딩이 느린 이유**: CSV를 매번 읽고 action delta를 계산하느라 시간이 오래 걸립니다.

**해결**: 타임스탬프 기반으로 주기에 맞춰 계산한 `data.pkl` 파일 생성

```bash
# 모든 데이터셋 전처리 (최초 1회만)
python preprocessing/Create_DataPKL_with_Timestamps.py
```

**주요 기능**:
- ✅ CSV 타임스탬프 자동 분석 (Robot Hz 감지)
- ✅ 타임스탬프 기반으로 10Hz 간격 action delta 계산
- ✅ data.pkl 저장 (이후 로딩 10배 이상 빠름)
- ✅ 센서 데이터 자동 포함 (있으면)

**소요 시간**: 5-10분 (모든 데이터셋)
**결과**: 각 에피소드에 `data.pkl` 파일 생성

**전처리 후**:
```
/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*/
├── data.pkl  ← 새로 생성됨!
├── robot_state_*.csv
├── sensor_data_*.npz
└── View1/, View2/, ...
```

---

### Step 1: VL Features 캐싱 (필수)
**학습 전에 반드시 VL features를 미리 캐싱하세요!**
- 학습 속도가 **10배 이상** 빨라집니다
- VLM forward pass를 건너뛸 수 있습니다
- GPU 메모리 사용량이 줄어듭니다

```bash
# Single GPU (추천)
python training/Build_VL_Cache_for_Training.py

# Multi-GPU (더 빠름)
torchrun --nproc_per_node=4 training/Build_VL_Cache_for_Training.py
```

**소요 시간**: 10-30분 (데이터셋 크기에 따라)
**결과**: `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/*.pt`

---

### Step 2: 모델 학습

캐싱 후 두 가지 학습 방법 중 선택:

## Option 1: 비동기 모델 (추천)

**특징**:
- VLM: 3.33Hz (300ms 주기)
- Action Expert: 10Hz (100ms 주기)
- Sensor window: 65 samples (100ms @ 650Hz)
- VL features 3x reuse

**학습 명령**:
```bash
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --lr 1e-4 \
  --sensor-enabled \
  --vlm-reuse-count 3 \
  --sensor-window-size 65 \
  --image-resize-height 360 \
  --image-resize-width 640
```

**파라미터 설명**:
- `--vlm-reuse-count 3`: VL features를 3번 재사용 (3.33Hz VLM)
- `--sensor-window-size 65`: 센서 윈도우 크기 (100ms)
- `--image-resize-*`: 이미지 리사이즈 (640x360으로 축소, 빠른 학습)

---

## Option 2: Diffusion 모델

**특징**:
- Diffusion-based action generation
- Sensor window: 650 samples (100ms @ 650Hz)
- 2-stage training 지원 (하지만 Stage 1만 사용 권장)

### Stage 1 Only (VL frozen, 추천)

**VL 모델을 학습하지 않으므로 Stage 1만 사용하세요!**

```bash
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
  --dataset_dir /home/najo/NAS/VLA/dataset \
  --training-stage stage1 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 1e-4 \
  --epochs 20 \
  --diffusion_timesteps 100
```

**파라미터 설명**:
- `--training-stage stage1`: VL frozen, Sensor + Action Expert만 학습
- `--diffusion_timesteps 100`: DDPM 타임스텝 수

### Stage 2 (LoRA, 사용 안 함)

~~VL 모델도 fine-tune하려면 Stage 2 사용~~ **→ 사용하지 않습니다**

---

## 데이터셋 가중치

새 데이터셋에 더 높은 가중치를 적용하여 학습합니다:

| 데이터셋 | 가중치 | 설명 |
|---------|--------|------|
| **Make_dataset/New_dataset/** | **3x** | 새로 수집한 데이터 (최우선) |
| White_silicone_white_circle | 2x | 우선순위 기존 데이터 |
| Needle_insertion_eye_trocar | 2x | 우선순위 기존 데이터 |
| OCT_insertion | 1x | 일반 기존 데이터 |
| part1 | 1x | 일반 기존 데이터 |

**결과**: 새 데이터가 학습 중 **3배 더 자주** 샘플링됩니다.

---

## 체크포인트

학습 중 체크포인트가 자동 저장됩니다:

### 비동기 모델
- `./checkpoints/qwen_vla_async.pt` - 최신 체크포인트
- `./checkpoints/qwen_vla_async_best.pt` - 최고 성능 체크포인트
- `./checkpoints/qwen_vla_async_final.pt` - 최종 체크포인트

### Diffusion 모델
- `./checkpoints/diffusion_stage1_latest.pt` - Stage 1 최신
- `./checkpoints/diffusion_stage1_best.pt` - Stage 1 최고 성능

---

## 모니터링

학습 진행 상황은 Weights & Biases에서 확인:

### 비동기 모델
- Project: `QwenVLA-Async`
- Metrics:
  - `train/loss_step`: 배치별 loss
  - `train/vl_reuse_counter`: VL feature reuse 횟수
  - `train/sensor_samples`: 센서 데이터 샘플 수
  - `val/loss_epoch`: Validation loss

### Diffusion 모델
- Project: `QwenVLA-Diffusion`
- Metrics:
  - `train/noise_loss`: Noise prediction loss
  - `train/diffusion_t`: Diffusion timestep 분포

---

## 성능 최적화 팁

### 1. VL 캐싱 (필수!)
- 학습 전 반드시 캐싱 수행
- 캐시가 없으면 학습이 매우 느림

### 2. Image Resize
```bash
--image-resize-height 360 \
--image-resize-width 640
```
- 이미지를 640x360으로 축소
- VLM 속도 향상, 메모리 절약

### 3. Gradient Accumulation
```bash
--grad-accum-steps 8
```
- 큰 batch size 효과
- GPU 메모리 절약

### 4. Mixed Precision (기본 활성화)
- BFloat16 자동 사용
- 속도 향상, 메모리 절약

### 5. Persistent Workers
- DataLoader의 `persistent_workers=False` 사용
- 메모리 누수 방지

---

## 트러블슈팅

### OOM (Out of Memory)
1. Batch size 줄이기: `--batch-size 2`
2. Gradient accumulation 늘리기: `--grad-accum-steps 16`
3. Image resize 적용: `--image-resize-height 360`

### 학습이 느림
1. **VL 캐싱 확인**: 캐시 파일이 있는지 확인
2. **VLM reuse count 증가**: `--vlm-reuse-count 4`
3. **Num workers 조정**: `--num-workers 8`

### Validation loss가 높음
1. 학습 epoch 늘리기: `--epochs 50`
2. Learning rate 조정: `--lr 5e-5`
3. Sensor loss weight 조정: `--sensor-loss-weight 3.0`

---

## FAQ

### Q: 데이터셋 전처리를 꼭 해야 하나요?
**A: 네, 필수입니다!** 전처리 없이 학습하면 데이터 로딩이 매우 느립니다.
- 전처리 후: 데이터 로딩 ~1초
- 전처리 전: 데이터 로딩 ~10초 이상

### Q: VL 캐싱을 꼭 해야 하나요?
**A: 네, 필수입니다!** 캐싱 없이 학습하면 10배 이상 느립니다.

### Q: Diffusion Stage 2를 사용해야 하나요?
**A: 아니오.** VL 모델을 학습하지 않으므로 Stage 1만 사용하세요.

### Q: 새 데이터셋이 정말 3배 더 샘플링되나요?
**A: 네.** WeightedRandomSampler를 사용해서 새 데이터가 3배 더 자주 선택됩니다.

### Q: 학습 시간은 얼마나 걸리나요?
**A: 데이터셋 크기에 따라 다르지만:**
- 비동기 모델: 4 GPU로 1 epoch당 ~1시간
- Diffusion 모델: 4 GPU로 1 epoch당 ~2시간

### Q: 캐시 파일을 삭제해도 되나요?
**A: 안 됩니다!** 캐시 파일을 삭제하면 다시 캐싱해야 합니다.

---

## 전체 학습 플로우 요약

```bash
# 1. 데이터셋 전처리 (최초 1회만)
python preprocessing/Create_DataPKL_with_Timestamps.py

# 2. VL 캐싱 (최초 1회만, 또는 데이터셋 추가 시)
python training/Build_VL_Cache_for_Training.py

# 3. 학습 실행
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
  --batch-size 4 \
  --vlm-reuse-count 3
```

**소요 시간 (전체)**:
- 데이터셋 전처리: ~10분
- VL 캐싱: ~20분
- 학습: 1 epoch당 ~1시간 (4 GPU)

---

## 다음 단계

학습 완료 후:
1. 체크포인트 평가
2. Inference 테스트
3. Real robot 배포

자세한 내용은 `docs/INFERENCE_GUIDE.md` 참조.
