# Quick Start Guide

빠르게 학습을 시작하는 가이드입니다.

## 📋 사전 준비

### 1. 데이터셋 전처리 (최초 1회만, 이미 완료됨)

```bash
python preprocessing/Create_DataPKL_with_Timestamps.py
```

✅ **완료 확인**: 각 에피소드에 `data.pkl` 파일 존재
✅ **10개 에피소드 전처리 완료**

### 2. VL 캐시 (이미 있음)

✅ **기존 캐시**: 8.1GB (100만 개 파일)
✅ **위치**: `/home/najo/NAS/VLA/dataset/cache/qwen_vl_features/`

새 데이터는 학습 첫 epoch에 자동 캐싱됩니다.

---

## 🚀 학습 시작

### Option 1: 비동기 모델 (추천)

**특징**:
- VLM: 3.33Hz (300ms 주기)
- Action Expert: 10Hz (100ms 주기)
- Sensor: 65 samples (100ms window)

**실행**:
```bash
bash scripts/train_async.sh
```

**또는 직접**:
```bash
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --vlm-reuse-count 3 \
  --sensor-window-size 65
```

---

### Option 2: Diffusion 모델

**특징**:
- Diffusion-based action generation
- Sensor: 650 samples (100ms window)
- Stage 1 only (VL frozen)

**실행**:
```bash
bash scripts/train_diffusion.sh
```

**또는 직접**:
```bash
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
  --dataset_dir /home/najo/NAS/VLA/dataset \
  --training-stage stage1 \
  --batch_size 4 \
  --grad_accum 8 \
  --epochs 20
```

---

## 📊 데이터셋 구성

| 데이터셋 | 가중치 | 샘플 수 |
|---------|--------|---------|
| **Make_dataset/New_dataset** | **3x** | 6개 에피소드 |
| White_silicone_white_circle | 2x | 9개 에피소드 |
| Needle_insertion_eye_trocar | 2x | 1개 에피소드 |

**총 16개 에피소드** (새 데이터가 3배 더 자주 샘플링됨)

---

## 🎯 학습 모니터링

### Weights & Biases

**비동기 모델**:
- Project: `QwenVLA-Async`
- URL: https://wandb.ai

**Diffusion 모델**:
- Project: `QwenVLA-Diffusion`
- URL: https://wandb.ai

### 주요 메트릭

**비동기 모델**:
- `train/loss_step`: 배치별 loss
- `train/vl_reuse_counter`: VL feature reuse 상태
- `train/sensor_samples`: 센서 데이터 사용량
- `val/loss_epoch`: Validation loss

**Diffusion 모델**:
- `train/noise_loss`: Noise prediction loss
- `train/diffusion_t`: Diffusion timestep 분포
- `val/loss_epoch`: Validation loss

---

## 💾 체크포인트

### 비동기 모델
```
./checkpoints/
├── qwen_vla_async.pt          # 최신
├── qwen_vla_async_best.pt     # 최고 성능
└── qwen_vla_async_final.pt    # 최종
```

### Diffusion 모델
```
./checkpoints/
├── diffusion_stage1_latest.pt # 최신
└── diffusion_stage1_best.pt   # 최고 성능
```

---

## ⏱️ 예상 소요 시간

### 비동기 모델
- **1 epoch**: ~1시간 (4 GPU)
  - 첫 epoch: 조금 느림 (새 데이터 캐싱)
  - 이후 epoch: 빠름 (모두 캐시 사용)
- **100 epochs**: ~100시간 (4일)

### Diffusion 모델
- **1 epoch**: ~2시간 (4 GPU)
- **20 epochs**: ~40시간 (2일)

---

## 🔧 파라미터 조정

### GPU 메모리 부족 시

```bash
# Batch size 줄이기
--batch-size 2 \
--grad-accum-steps 16

# 또는 이미지 해상도 줄이기
--image-resize-height 270 \
--image-resize-width 480
```

### 학습 속도 올리기

```bash
# VLM reuse 늘리기 (비동기 모델만)
--vlm-reuse-count 4

# Mixed precision 확인 (자동 활성화됨)
# BFloat16 사용 중
```

### 새 데이터 가중치 변경

`training/A6_VLA_TRAIN_ASYNC.py` 또는 `training/A5st_VLA_TRAIN_Diffusion_with_sensor.py`에서:

```python
dataset_weights.extend([3.0] * len(ds))  # 3.0을 원하는 값으로 변경
```

---

## 🐛 트러블슈팅

### 1. "data.pkl not found" 에러
```bash
# 해결: 데이터셋 전처리
python preprocessing/Create_DataPKL_with_Timestamps.py
```

### 2. 학습이 느림
- ✅ VL 캐시 확인: 첫 epoch은 캐시 생성으로 느릴 수 있음
- ✅ 두 번째 epoch부터 빠름

### 3. OOM (Out of Memory)
```bash
# Batch size 줄이기
--batch-size 2 --grad-accum-steps 16
```

### 4. Validation loss가 안 떨어짐
```bash
# Learning rate 조정
--lr 5e-5

# Sensor loss weight 조정
--sensor-loss-weight 3.0
```

---

## 📚 더 자세한 내용

- [전체 학습 가이드](TRAINING_GUIDE.md)
- [데이터셋 구조](../preprocessing/README.md)
- [모델 아키텍처](../models/README.md)

---

## 🎉 시작하기

**추천 순서**:

1. **비동기 모델 학습** (더 빠름):
```bash
bash scripts/train_async.sh
```

2. **Diffusion 모델 학습** (더 안정적):
```bash
bash scripts/train_diffusion.sh
```

두 모델을 동시에 학습할 수도 있습니다 (GPU가 충분하다면)!

**Happy Training! 🚀**
