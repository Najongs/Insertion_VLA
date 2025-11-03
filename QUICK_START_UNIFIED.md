# 🚀 Quick Start - Unified Training Script

## ✅ 준비 완료!

모든 학습 스크립트가 하나로 통합되었습니다.

## 🎯 빠른 시작

### 방법 1: 인터랙티브 메뉴
```bash
cd /home/najo/NAS/VLA/Insertion_VLA
./scripts/start_training_unified.sh
```

### 방법 2: Diffusion 학습 (권장)
```bash
./scripts/start_training_unified.sh diffusion
```

### 방법 3: Regression 학습
```bash
# Step 1: 캐시 빌드 (한 번만)
./scripts/start_training_unified.sh regression-cache

# Step 2: 학습
./scripts/start_training_unified.sh regression-train
```

## 📊 모델 비교

| 특징 | Diffusion | Regression |
|------|-----------|------------|
| **캐시 빌드** | ❌ 불필요 | ✅ 필요 |
| **센서 윈도우** | 650 samples | 650 samples |
| **VLM 재사용** | 1x | 3x |
| **배치 크기** | 4 | 16 |
| **학습 속도** | 느림 | 빠름 |
| **추론 속도** | 느림 | 빠름 |
| **성능** | 더 좋음 | 좋음 |

## 📝 주요 변경사항

### ✅ 통합 완료
- `A5st_VLA_TRAIN_Diffusion_with_sensor.py` + `A5st_VLA_TRAIN_VL_Lora_with_sensor.py`
  → **`A5st_VLA_TRAIN_Unified.py`**

### ✅ 데이터셋 처리
- **센서 데이터 없는 old 데이터셋**: 자동으로 제로 패딩 처리
- **센서 데이터 있는 new 데이터셋**: 정상 로드
- `confidence` 값으로 구분 (센서 있음: 1.0, 없음: 0.5)

### ✅ collate 함수 수정
- `async_collate_fn_with_sensor` 사용
- `confidence` 키 자동 추가

## 🔧 Python 직접 실행

### Diffusion
```bash
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
    --model-type diffusion \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --epochs 20
```

### Regression
```bash
# 캐시 빌드
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
    --model-type regression \
    --mode cache

# 학습
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Unified.py \
    --model-type regression \
    --mode train \
    --batch_size 16 \
    --grad_accum 16 \
    --lr 5e-5
```

## 📈 모니터링

### WandB
- Diffusion: `QwenVLA-Unified-Diffusion`
- Regression: `QwenVLA-Unified-Regression`

### 체크포인트
- Diffusion: `./checkpoints/diffusion_*.pt`
- Regression: `./checkpoints/regression_*.pt`

## 🐛 문제 해결

### CUDA Out of Memory
```bash
--batch_size 2 --grad_accum 16
```

### confidence 키 에러
→ ✅ 수정됨! `AsyncIntegratedDataset`에 `confidence` 추가

### 센서 데이터 없음 에러
→ ✅ 수정됨! 자동으로 제로 패딩 처리

---

**모든 준비 완료! 학습을 시작하세요.** 🚀
