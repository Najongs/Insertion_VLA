# Asynchronous VLA - Quick Start

실시간 비동기 VLA 모델 (VLM 3.33 Hz + Action Expert 10 Hz)

---

## 🚀 빠른 시작

### 1. VLM 속도 프로파일링 (선택)

```bash
python utils/profile_vlm_speed.py --num-samples 50
```

**예상 결과**: VLM ~311ms (3.21 Hz)

---

### 2. 비동기 학습

#### Stage 1: Sensor Encoder + Action Expert (VLM frozen)

```bash
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --training-stage stage1 \
    --finetune-vl none \
    --sensor-enabled \
    --sensor-window-size 65 \
    --batch-size 2 \
    --lr 1e-4
```

#### Stage 2: LoRA Fine-tuning (비동기 학습)

```bash
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
    --finetune-vl lora \
    --stage1-checkpoint checkpoints/qwen_vla_sensor_best.pt \
    --sensor-window-size 65 \
    --vlm-reuse-count 3 \
    --batch-size 2 \
    --lr 1e-4 \
    --vl-lr 1e-5 \
    --sensor-lr 5e-4
```

---

### 3. 비동기 추론

```bash
python examples/async_inference_example.py \
    --checkpoint checkpoints/qwen_vla_async_best.pt \
    --duration 10.0 \
    --vlm-hz 3.33 \
    --action-hz 10.0
```

**출력 예시:**
```
🚀 AsyncVLAInference initialized (TRUE async mode):
   VLM target period: 300ms (3.33 Hz)
   VLM actual time: ~311ms (measured)
   Action Expert: 100ms (10.0 Hz)

🎮 Starting TRUE async control loop for 10.0s...
   VLM runs in background thread (~311ms)
   Action Expert runs at 10 Hz in main loop

   🔄 Starting initial VLM update...
   ⏳ Waiting for first VLM completion...
   ⚡ [Step 10] Action: 24.3ms | VLM: 311.2ms (idle)
   🔄 [Step 13] Starting VLM update #1 (background)
   ⚡ [Step 20] Action: 23.8ms | VLM: 310.9ms (running)
   ...

📊 Control loop statistics:
   Total time: 10.02s
   Actions predicted: 98 (9.78 Hz)
   VLM updates started: 34
   VLM updates completed: 33 (3.29 Hz)
   Avg VLM time: 311.1ms
   Avg Action time: 24.2ms
```

---

## 📊 성능 (멀티뷰 5개 기준)

| 지표 | 기존 동기식 | 비동기 (개선) |
|------|-------------|---------------|
| VLM 실행 | 매번 (~1484ms) | 백그라운드 (1700ms 주기) |
| Action 생성 | ~1484ms | **100ms** ✅ |
| Actions/sec | ~0.67개 ❌ | **~10개** ✅ |
| 실시간 제어 | ❌ | ✅ |

**VLM 추론 시간:**
- 단일 이미지: ~311ms
- **멀티뷰 5개: ~1484ms** (약 4.8배 증가)

---

## 🔑 핵심 개념

### 학습 vs 추론

**학습 (Training):**
- VL 캐시 사용 (`cache_mode="on"`)
- VL features를 3번 재사용하며 학습
- 빠른 학습 속도

**추론 (Inference):**
- VL 캐시 비활성화 (`cache_mode="off"`)
- 실제 VLM 실행 (~311ms)
- VLM을 **백그라운드 스레드**에서 실행
- Action Expert는 메인 루프에서 10 Hz로 계속 동작

### 비동기 타임라인

```
t=0ms:     VLM 스레드 시작 (백그라운드, 311ms 소요)
t=0ms:     Action #1 예측 (메인 루프, 24ms)
t=100ms:   Action #2 예측 (VLM 실행 중...)
t=200ms:   Action #3 예측 (VLM 실행 중...)
t=311ms:   VLM 완료 → VL features 업데이트
t=300ms:   새 VLM 스레드 시작
t=300ms:   Action #4 예측 (새 VL features)
t=400ms:   Action #5 예측 (VLM 실행 중...)
...
```

---

## 📁 파일 구조

```
Insertion_VLA/
├── models/
│   ├── model_with_sensor.py              # 기존 동기식 모델
│   └── model_with_sensor_async.py        # 비동기 모델 ✨
│
├── training/
│   ├── A5st_VLA_TRAIN_VL_Lora_with_sensor.py  # 기존 학습
│   └── A6_VLA_TRAIN_ASYNC.py             # 비동기 학습 ✨
│
├── examples/
│   └── async_inference_example.py        # 비동기 추론 예제 ✨
│
├── utils/
│   └── profile_vlm_speed.py              # VLM 속도 프로파일링 ✨
│
├── docs/
│   ├── ASYNC_TRAINING.md                 # 상세 가이드 ✨
│   └── ...
│
└── README_ASYNC.md                        # 이 문서
```

---

## 🎯 주요 하이퍼파라미터

| 파라미터 | 값 (멀티뷰 5개) | 값 (단일 이미지) | 설명 |
|----------|-----------------|------------------|------|
| `sensor_window_size` | 65 | 65 | Sensor 윈도우 크기 (100ms @ 650Hz) |
| `vlm_reuse_count` | 17 | 3 | VL feature 재사용 횟수 |
| `vlm_update_hz` | 0.59 | 3.33 | VLM 업데이트 주파수 |
| `action_expert_hz` | 10.0 | 10.0 | Action Expert 실행 주파수 |

---

## 📖 상세 문서

전체 설명, FAQ, 트러블슈팅은 **[docs/ASYNC_TRAINING.md](docs/ASYNC_TRAINING.md)** 참조

---

## ⚡ 빠른 테스트

실제 데이터 없이 비동기 동작 테스트:

```bash
# 비동기 모델 테스트
python models/model_with_sensor_async.py

# 비동기 추론 예제 (더미 데이터)
python examples/async_inference_example.py --duration 5.0
```

---

**작성**: 2025-10-28
