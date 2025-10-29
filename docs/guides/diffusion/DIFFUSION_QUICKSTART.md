# Diffusion Policy Quick Start Guide

Regression 방식과 Diffusion 방식을 비교하기 위한 빠른 시작 가이드

## 1. 모델 테스트

```bash
# Diffusion 모델이 제대로 동작하는지 테스트
python examples/test_diffusion_model.py
```

예상 출력:
```
======================================================================
Diffusion VLA Model - Validation Tests
======================================================================

Test 1: Testing Diffusion Schedule...
  [✓ PASSED] Diffusion schedule
         dist(t=0)=0.012, dist(t=99)=2.345

Test 2: Testing Diffusion Action Expert...
  [✓ PASSED] Action expert forward
         Output shape: torch.Size([2, 8, 7])
  [✓ PASSED] Action expert sampling (DDPM 100 steps)
         Time: 1234.5ms
  [✓ PASSED] Action expert sampling (DDIM 10 steps)
         Time: 156.7ms (speedup: 7.9x)

...

Overall: 5/5 tests passed
✓ All tests passed! Diffusion model is ready.
```

## 2. 학습 실행

### Regression Baseline (기존)
```bash
# Single GPU
python training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --sensor_loss_weight 2.0

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar \
    --epochs 20 \
    --batch_size 4
```

### Diffusion Policy (신규)
```bash
# Single GPU
python training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --sensor_loss_weight 2.0 \
    --diffusion_timesteps 100

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar \
    --epochs 20 \
    --diffusion_timesteps 100
```

## 3. 모델 비교

### 학습 속도 비교
| 항목 | Regression | Diffusion |
|-----|-----------|----------|
| Epoch 시간 (4 GPUs) | ~30분 | ~40분 (+33%) |
| 메모리 사용 | ~24GB | ~28GB |
| 학습 안정성 | 높음 | 보통 |

### 추론 속도 비교
```bash
# Regression inference
python Make_dataset/Realtime_inference_receiver.py \
    --checkpoint checkpoints/regression_best.pt \
    --inference-rate 5.0  # 5Hz 가능

# Diffusion inference (느림 - DDIM 10 steps)
# TODO: Diffusion receiver 구현 필요
# 예상 속도: ~2Hz (DDIM 10), ~0.5Hz (DDPM 100)
```

## 4. 체크포인트 위치

```
checkpoints/
├── epoch1.pt          # Regression checkpoint
├── epoch2.pt
├── ...
├── diffusion_epoch1.pt   # Diffusion checkpoint
├── diffusion_epoch2.pt
└── ...
```

## 5. Wandb 모니터링

### Regression
- Project: `QwenVLA-Sensor`
- Run name: `train_sensor_MMDD_HHMM`

### Diffusion
- Project: `QwenVLA-Diffusion-Sensor`
- Run name: `train_diffusion_sensor_MMDD_HHMM`

URL: https://wandb.ai/your-username/

## 6. 주요 파라미터 튜닝

### Diffusion 특화 파라미터
```bash
# 더 많은 timesteps (더 나은 품질, 느린 학습)
--diffusion_timesteps 200

# 더 적은 timesteps (빠른 학습, 품질 저하)
--diffusion_timesteps 50

# Sensor data 가중치 조절
--sensor_loss_weight 3.0  # Sensor 데이터 더 중요하게
--sensor_loss_weight 1.5  # Sensor 데이터 덜 중요하게
```

### 학습 안정성 개선
```bash
# Gradient accumulation 증가 (메모리 부족 시)
--grad_accum 16

# Learning rate 감소 (학습 불안정 시)
--lr 5e-5

# Batch size 감소
--batch_size 2
```

## 7. 문제 해결

### GPU 메모리 부족
```bash
# Batch size 줄이기
--batch_size 2 --grad_accum 16

# Diffusion timesteps 줄이기
--diffusion_timesteps 50

# Hidden dimension 줄이기 (model_with_sensor_diffusion.py 수정)
hidden_dim=512  # 기본 1024
```

### 학습 불안정 (Loss가 발산)
```bash
# Learning rate 감소
--lr 5e-5

# Warmup 비율 증가 (scheduler 수정)
warmup_ratio=0.05  # 기본 0.03

# Gradient clipping 강화 (코드 수정)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

### 추론 속도가 너무 느림
```python
# DDIM steps 줄이기 (model_with_sensor_diffusion.py)
pred_actions = model.action_expert.sample(
    vl_tokens, sensor_features,
    ddim_steps=5  # 기본 10, 최소 3
)
```

## 8. 다음 단계

### 성능 평가
```bash
# TODO: Evaluation script 작성
python evaluation/compare_regression_vs_diffusion.py \
    --regression_ckpt checkpoints/regression_best.pt \
    --diffusion_ckpt checkpoints/diffusion_best.pt \
    --test_dataset /path/to/test
```

### 실시간 추론
```bash
# Regression (현재 사용 가능)
python Make_dataset/Realtime_inference_receiver.py \
    --checkpoint checkpoints/regression_best.pt \
    --save-data

# Diffusion (TODO: 구현 필요)
# python Make_dataset/Realtime_inference_receiver_diffusion.py \
#     --checkpoint checkpoints/diffusion_best.pt \
#     --ddim-steps 10
```

## 9. 예상 결과

### Regression Baseline
- **Success Rate**: 80-85%
- **Path Error**: 2.5mm (평균)
- **Inference Time**: 150ms
- **Training Time**: 10 hours (20 epochs, 4 GPUs)

### Diffusion Policy (예상)
- **Success Rate**: **85-90%** (+5%)
- **Path Error**: **2.0mm** (-20%)
- **Inference Time**: 600ms (DDIM 10) / 5s (DDPM 100)
- **Training Time**: 13 hours (20 epochs, 4 GPUs)

## 10. 참고 자료

- **Regression 모델**: `models/model_with_sensor.py`
- **Diffusion 모델**: `models/model_with_sensor_diffusion.py`
- **Regression 학습**: `training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py`
- **Diffusion 학습**: `training/A5st_VLA_TRAIN_Diffusion_with_sensor.py`
- **상세 README**: `training/README_DIFFUSION.md`

## 문의

문제가 발생하면 다음을 확인하세요:
1. GPU 메모리: `nvidia-smi`
2. Wandb 로그: https://wandb.ai
3. 학습 로그: `checkpoints/` 디렉토리
4. 테스트 결과: `python examples/test_diffusion_model.py`
