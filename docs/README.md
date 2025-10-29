# Insertion VLA 프로젝트 문서

Vision-Language-Action (VLA) 모델을 이용한 실시간 로봇 제어 시스템의 완전한 문서입니다.

## 📚 문서 구조

```
docs/
├── README.md (이 문서)
├── guides/
│   ├── training/           # 학습 가이드
│   ├── inference/          # 추론 가이드
│   ├── optimization/       # 최적화 가이드
│   ├── diffusion/          # Diffusion 모델 가이드
│   └── other/              # 기타 가이드
├── reference/              # 레퍼런스 문서
└── archive/                # 이전 버전 문서
```

---

## 🚀 빠른 시작

### 처음 시작하시나요?

1. **[QUICKSTART_INFERENCE.md](guides/inference/QUICKSTART_INFERENCE.md)** ⭐ - 5분 만에 추론 파이프라인 실행
2. **[ASYNC_VLA_PIPELINE_README.md](reference/ASYNC_VLA_PIPELINE_README.md)** - 전체 시스템 개요

### 학습부터 하려면?

1. **[STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md)** - Stage 1 학습 (권장)
2. **[STAGE2_LORA_TRAINING.md](guides/training/STAGE2_LORA_TRAINING.md)** - Stage 2 LoRA 학습 (선택)

---

## 📖 추론 가이드 (Inference)

실시간 추론을 위한 완전한 가이드입니다.

### 시작하기

| 문서 | 설명 | 난이도 |
|------|------|--------|
| **[QUICKSTART_INFERENCE.md](guides/inference/QUICKSTART_INFERENCE.md)** ⭐ | 5분 빠른 시작 가이드 | ⭐ 초급 |
| **[REALTIME_INFERENCE_GUIDE.md](guides/inference/REALTIME_INFERENCE_GUIDE.md)** | 실시간 추론 사용법 상세 | ⭐⭐ 중급 |
| **[INFERENCE_PIPELINE_TEST_GUIDE.md](guides/inference/INFERENCE_PIPELINE_TEST_GUIDE.md)** | 완전한 테스트 절차 | ⭐⭐⭐ 고급 |

### 주요 특징

- ⚡ **10Hz Action Expert**: 실시간 제어 (100ms latency)
- 🔄 **비동기 VLM**: 배경 스레드 (~2.6Hz)
- 📊 **센서 융합**: 65 samples (100ms @ 650Hz)
- 🎯 **VL Feature Reuse**: 4x 재사용으로 효율 극대화

### 빠른 실행

```bash
# 1. 시스템 검증
python utils/verify_inference_pipeline.py --robot-ip 10.130.41.110

# 2. Sender 시작 (Robot PC, Jetson, Sensor PC)
python Make_dataset/Optimized_Robot_sender.py --robot on
python Make_dataset/Optimized_Camera_sender.py --server-ip <Inference PC IP>

# 3. 추론 실행 (Inference PC)
python Make_dataset/Async_inference_receiver.py \
    --checkpoint ./checkpoints/qwen_vla_sensor_best.pt \
    --robot-ip 10.130.41.110 \
    --vl-reuse 4
```

---

## 🎓 학습 가이드 (Training)

VLA 모델 학습을 위한 가이드입니다.

### Stage 1: Frozen VL Backbone

| 문서 | 설명 | 권장 |
|------|------|------|
| **[STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md)** | Stage 1 학습 (Frozen VL + Trainable Sensor/Action) | ✅ 권장 |

**특징:**
- Frozen VL Backbone (3B parameters)
- Trainable Sensor Encoder + Action Expert (20.8M parameters)
- 빠른 학습 (25-30시간 on 4x RTX 4090)
- LoRA 없이 진행

**빠른 실행:**
```bash
# 1. VL Cache 빌드
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode cache \
    --sensor-enabled \
    --sensor-window-size 65 \
    --image-resize-height 360 \
    --image-resize-width 640

# 2. Stage 1 학습
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --sensor-enabled \
    --finetune-vl none \
    --training-stage stage1 \
    --sensor-window-size 65 \
    --image-resize-height 360 \
    --image-resize-width 640
```

### Stage 2: LoRA Fine-tuning (선택)

| 문서 | 설명 | 권장 |
|------|------|------|
| **[STAGE2_LORA_TRAINING.md](guides/training/STAGE2_LORA_TRAINING.md)** | Stage 2 LoRA 학습 | 📌 선택 |
| **[2STAGE_TRAINING.md](guides/training/2STAGE_TRAINING.md)** | 2단계 학습 전체 흐름 | 📌 선택 |

**특징:**
- Stage 1 체크포인트에서 시작
- LoRA로 VL Backbone 일부 fine-tuning
- 메모리 최적화 필요 (gradient checkpointing)

### 기타 학습 문서

| 문서 | 설명 |
|------|------|
| **[ASYNC_TRAINING.md](guides/training/ASYNC_TRAINING.md)** | 비동기 학습 관련 노트 |

---

## ⚡ 최적화 가이드 (Optimization)

성능 최적화 및 메모리 관리 가이드입니다.

| 문서 | 설명 | 핵심 내용 |
|------|------|----------|
| **[ASYNC_OPTIMIZATIONS.md](guides/optimization/ASYNC_OPTIMIZATIONS.md)** | 비동기 VLA 최적화 상세 | 이미지 리사이즈, 센서 윈도우, VL reuse |
| **[STAGE2_MEMORY_OPTIMIZATION.md](guides/optimization/STAGE2_MEMORY_OPTIMIZATION.md)** | Stage 2 메모리 최적화 | Gradient checkpointing, mixed precision |
| **[MODEL_IMPROVEMENTS.md](guides/optimization/MODEL_IMPROVEMENTS.md)** | 모델 개선사항 | 구조 변경, 성능 향상 |

### 주요 최적화

**1. 이미지 리사이즈 (640x360)**
- VLM 추론 시간: 1487ms → 381ms (**3.9배 빠름**)
- 실시간 제어 가능

**2. 센서 윈도우 최적화 (65 samples)**
- 기존: 650 samples (1초)
- 최적화: 65 samples (100ms @ 650Hz)
- 실시간 제어에 적합

**3. 비동기 VL-Action 분리**
- VLM: Background thread (~2.6Hz)
- Action Expert: Main thread (10Hz)
- VL features 4x 재사용

**4. Weighted Sampling**
- Priority datasets: 2x weight
- 중요한 데이터 강조 학습

---

## 🎯 Diffusion 모델 (Diffusion Policy)

Diffusion 기반 Action Expert를 사용하는 경우 참고하세요.

| 문서 | 설명 |
|------|------|
| **[README_DIFFUSION.md](guides/diffusion/README_DIFFUSION.md)** | Diffusion 모델 개요 |
| **[DIFFUSION_QUICKSTART.md](guides/diffusion/DIFFUSION_QUICKSTART.md)** | Diffusion 빠른 시작 |

**성능:**
- 50 DDIM steps: ~89ms (10Hz 제어 가능)
- 100 DDIM steps: ~184ms (더 정확하지만 느림)

---

## 📋 레퍼런스 (Reference)

전체 시스템 레퍼런스 및 기술 문서입니다.

| 문서 | 설명 |
|------|------|
| **[ASYNC_VLA_PIPELINE_README.md](reference/ASYNC_VLA_PIPELINE_README.md)** | 전체 시스템 종합 레퍼런스 |
| **[IMPORT_FIX_SUMMARY.md](reference/IMPORT_FIX_SUMMARY.md)** | Import 수정 사항 요약 |

---

## 🔧 기타 가이드

| 문서 | 설명 |
|------|------|
| **[HUGGINGFACE_GUIDE.md](guides/other/HUGGINGFACE_GUIDE.md)** | HuggingFace 사용 가이드 |

---

## 📦 Archive (이전 버전)

이전 버전의 문서들입니다. 현재 시스템과 일부 호환되지 않을 수 있습니다.

| 문서 | 설명 |
|------|------|
| [DATASET_README.md](archive/DATASET_README.md) | 구 데이터셋 가이드 |
| [FINAL_DATASET_SUMMARY.md](archive/FINAL_DATASET_SUMMARY.md) | 구 데이터셋 요약 |
| [TRAINING_GUIDE.md](archive/TRAINING_GUIDE.md) | 구 학습 가이드 |
| [TRAINING_UPDATE_SUMMARY.md](archive/TRAINING_UPDATE_SUMMARY.md) | 구 학습 업데이트 |
| [QUICK_REFERENCE.md](archive/QUICK_REFERENCE.md) | 구 빠른 참조 |
| [PROJECT_STRUCTURE.md](archive/PROJECT_STRUCTURE.md) | 구 프로젝트 구조 |

---

## 🎯 사용 시나리오별 가이드

### 시나리오 1: 빠르게 추론 테스트하기

```
1. QUICKSTART_INFERENCE.md (5분)
   └─→ 시스템 검증 → Sender 시작 → 추론 실행
```

### 시나리오 2: 처음부터 학습하기

```
1. STAGE1_TRAINING_GUIDE.md
   └─→ VL Cache 빌드 → Stage 1 학습
       └─→ QUICKSTART_INFERENCE.md
           └─→ 추론 테스트
```

### 시나리오 3: LoRA Fine-tuning 추가

```
1. STAGE1_TRAINING_GUIDE.md (완료 후)
   └─→ STAGE2_LORA_TRAINING.md
       └─→ LoRA 학습 → 추론 테스트
```

### 시나리오 4: 성능 최적화

```
1. ASYNC_OPTIMIZATIONS.md
   └─→ 이미지 리사이즈 조정
   └─→ VL reuse count 조정
   └─→ Weighted sampling 적용
```

### 시나리오 5: Diffusion 모델 사용

```
1. README_DIFFUSION.md (개요)
   └─→ DIFFUSION_QUICKSTART.md
       └─→ Diffusion Action Expert 학습/추론
```

---

## 🛠️ 유틸리티 도구

프로젝트에 포함된 유틸리티 도구들입니다.

| 도구 | 위치 | 설명 |
|------|------|------|
| `verify_inference_pipeline.py` | utils/ | 시스템 검증 (의존성, 네트워크, GPU) |
| `analyze_inference_logs.py` | utils/ | 로그 분석 및 성능 통계 |
| `profile_diffusion_speed.py` | utils/ | Diffusion 속도 프로파일링 |

**사용 예:**
```bash
# 시스템 검증
python utils/verify_inference_pipeline.py --robot-ip 10.130.41.110

# 로그 분석
python utils/analyze_inference_logs.py inference.log

# Diffusion 프로파일링
python utils/profile_diffusion_speed.py --checkpoint ./checkpoints/qwen_vla_sensor_best.pt
```

---

## 📊 성능 벤치마크

### 최적화 전후 비교

| 항목 | 최적화 전 | 최적화 후 | 개선 |
|------|-----------|-----------|------|
| **VLM 추론** | 1487ms | 381ms | **3.9배 빠름** |
| **센서 윈도우** | 650 samples (1s) | 65 samples (100ms) | **10배 감소** |
| **Action Rate** | ~2Hz | **10Hz** | **5배 향상** |
| **GPU 메모리** | ~12GB | ~8GB | 33% 절감 |

### 시스템 성능 (RTX 4090 기준)

| 항목 | 값 |
|------|-----|
| VLM 업데이트 주파수 | ~2.6 Hz |
| Action Expert 주파수 | 10 Hz |
| VL Feature Reuse | 4x |
| 총 Action 출력 | 80 actions/sec (10Hz × 8-horizon) |
| GPU 메모리 사용 | ~8GB |

---

## 🗺️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
├──────────────────┬──────────────────┬──────────────────────────┤
│ Robot Sender     │ Camera Sender    │ Sensor Sender            │
│ 10Hz (ZMQ PUB)   │ 5Hz (ZMQ PUSH)   │ 650Hz (UDP)              │
│ Port 5556        │ Port 5555        │ Port 9999                │
└────────┬─────────┴────────┬─────────┴────────┬─────────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
         ┌──────────────────▼───────────────────────────┐
         │     Async_inference_receiver.py              │
         │  ┌────────────────────────────────────────┐ │
         │  │ VLM Thread: ~381ms @ 640x360           │ │
         │  │ Updates VL features every ~385ms       │ │
         │  └────────────────────────────────────────┘ │
         │  ┌────────────────────────────────────────┐ │
         │  │ Action Expert: 10Hz                    │ │
         │  │ Reuses VL features 4x                  │ │
         │  │ Predicts actions every 100ms           │ │
         │  └────────────────────────────────────────┘ │
         └──────────────────────────────────────────────┘
                            │
                            ▼
                   Actions (10Hz, 8-horizon)
                            │
                            ▼
                   Robot Control (실시간)
```

---

## 🔍 문서 검색 가이드

### 키워드로 찾기

**학습 관련:**
- Stage 1 → [STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md)
- LoRA → [STAGE2_LORA_TRAINING.md](guides/training/STAGE2_LORA_TRAINING.md)
- 메모리 → [STAGE2_MEMORY_OPTIMIZATION.md](guides/optimization/STAGE2_MEMORY_OPTIMIZATION.md)
- Cache → [STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md)

**추론 관련:**
- 빠른 시작 → [QUICKSTART_INFERENCE.md](guides/inference/QUICKSTART_INFERENCE.md)
- 실시간 → [REALTIME_INFERENCE_GUIDE.md](guides/inference/REALTIME_INFERENCE_GUIDE.md)
- 테스트 → [INFERENCE_PIPELINE_TEST_GUIDE.md](guides/inference/INFERENCE_PIPELINE_TEST_GUIDE.md)
- Sender → [REALTIME_INFERENCE_GUIDE.md](guides/inference/REALTIME_INFERENCE_GUIDE.md)

**최적화 관련:**
- 이미지 리사이즈 → [ASYNC_OPTIMIZATIONS.md](guides/optimization/ASYNC_OPTIMIZATIONS.md)
- 센서 윈도우 → [ASYNC_OPTIMIZATIONS.md](guides/optimization/ASYNC_OPTIMIZATIONS.md)
- VL Reuse → [ASYNC_OPTIMIZATIONS.md](guides/optimization/ASYNC_OPTIMIZATIONS.md)
- GPU 메모리 → [STAGE2_MEMORY_OPTIMIZATION.md](guides/optimization/STAGE2_MEMORY_OPTIMIZATION.md)

**Diffusion 관련:**
- Diffusion → [README_DIFFUSION.md](guides/diffusion/README_DIFFUSION.md)
- DDIM → [DIFFUSION_QUICKSTART.md](guides/diffusion/DIFFUSION_QUICKSTART.md)

---

## ❓ FAQ

### Q1: 처음 시작하는데 어디서부터 봐야 하나요?

**A:** [QUICKSTART_INFERENCE.md](guides/inference/QUICKSTART_INFERENCE.md)부터 시작하세요. 5분 만에 전체 시스템을 실행해볼 수 있습니다.

### Q2: 학습은 어떻게 하나요?

**A:** [STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md)를 따라하세요. Stage 1 학습만으로도 충분한 성능을 얻을 수 있습니다.

### Q3: LoRA 학습이 꼭 필요한가요?

**A:** 아니요. Stage 1 학습만으로도 실시간 제어가 가능합니다. LoRA는 추가 성능 향상이 필요한 경우에만 고려하세요.

### Q4: GPU 메모리가 부족합니다.

**A:** [STAGE2_MEMORY_OPTIMIZATION.md](guides/optimization/STAGE2_MEMORY_OPTIMIZATION.md)를 참고하여 gradient checkpointing을 활성화하거나, batch size를 줄이세요.

### Q5: 추론 속도가 느립니다.

**A:** [ASYNC_OPTIMIZATIONS.md](guides/optimization/ASYNC_OPTIMIZATIONS.md)를 참고하여:
- 이미지 리사이즈가 640x360으로 설정되었는지 확인
- VL reuse count를 4 또는 6으로 증가
- GPU 사용량 확인 (`nvidia-smi`)

### Q6: 데이터가 수신되지 않습니다.

**A:** [INFERENCE_PIPELINE_TEST_GUIDE.md](guides/inference/INFERENCE_PIPELINE_TEST_GUIDE.md)의 트러블슈팅 섹션을 참고하세요. 주로 네트워크 또는 방화벽 문제입니다.

### Q7: Diffusion 모델을 사용하고 싶습니다.

**A:** [README_DIFFUSION.md](guides/diffusion/README_DIFFUSION.md)와 [DIFFUSION_QUICKSTART.md](guides/diffusion/DIFFUSION_QUICKSTART.md)를 참고하세요.

---

## 📞 지원 및 기여

문제가 발생하거나 개선 사항이 있으면:

1. **로그 수집**: `python utils/analyze_inference_logs.py inference.log`
2. **시스템 검증**: `python utils/verify_inference_pipeline.py`
3. **관련 문서 확인**: 위의 문서 검색 가이드 활용
4. **이슈 리포트**: 담당자에게 문의

---

## 📝 문서 히스토리

| 날짜 | 버전 | 변경 사항 |
|------|------|----------|
| 2025-10-29 | 1.0.0 | 초기 문서 구조화 및 종합 |
| - | - | 비동기 VLA 파이프라인 완성 |
| - | - | Stage 1 최적화 적용 |

---

## 🎉 준비 완료!

이제 시작할 준비가 되었습니다!

1. **추론 테스트**: [QUICKSTART_INFERENCE.md](guides/inference/QUICKSTART_INFERENCE.md) (5분)
2. **학습 시작**: [STAGE1_TRAINING_GUIDE.md](guides/training/STAGE1_TRAINING_GUIDE.md) (~30시간)
3. **전체 시스템 이해**: [ASYNC_VLA_PIPELINE_README.md](reference/ASYNC_VLA_PIPELINE_README.md)

**Happy coding! 🚀**

---

**마지막 업데이트:** 2025-10-29
**버전:** 1.0.0
**상태:** Production Ready
