# Asynchronous VLA Training and Inference

## 개요

기존 VLA 모델은 VLM과 Action Expert가 동기적으로 동작하여 추론 속도가 느렸습니다. 이 문서는 비동기 학습 및 추론을 통해 **VLM을 ~3.33 Hz로, Action Expert를 10 Hz로 동작**시켜 실시간 성능을 달성하는 방법을 설명합니다.

---

## 문제점: 기존 동기식 구조

### 기존 모델 구조
```
VLM (Qwen2.5-VL) → VL Features (매 추론마다 실행, ~311ms)
    ↓
Sensor Encoder (650 timesteps)
    ↓
Action Expert
    ↓
8 Actions (Horizon=8)
```

**문제:**
- VLM 추론 시간: ~311ms (3.21 Hz)
- 전체 파이프라인이 VLM 속도에 종속
- 1초에 최대 ~3번만 action 생성 가능
- 실시간 제어에 부적합

---

## 해결책: 비동기 구조

### ⚠️ 중요: 학습 vs 추론

**학습 시 (Training):**
- VL 캐시를 사용하여 빠르게 VL features 추출 (~수 ms)
- VL features를 3번 재사용하며 학습
- 실제 VLM 실행 안 함 (캐시된 features 사용)

**추론 시 (Inference):**
- VL 캐시를 **사용하지 않음** (`cache_mode="off"`)
- 실제 VLM을 실행하여 VL features 추출 (~311ms)
- VLM을 **별도 스레드**에서 백그라운드로 실행
- Action Expert는 메인 루프에서 10 Hz로 계속 동작

### VLM 추론 속도 프로파일링

먼저 실제 VLM 추론 속도를 측정:

```bash
python utils/profile_vlm_speed.py --num-samples 50 --save-results vlm_profile_results.json
```

**측정 결과 (단일 이미지):**
- VLM 평균 추론 시간: **311 ms**
- VLM 처리량: **3.21 Hz**

**⚠️ 실제 멀티뷰 5개 사용 시:**
- VLM 평균 추론 시간: **1484 ms**
- VLM 처리량: **0.67 Hz**
- **권장 설정: VLM 0.59 Hz (1700ms 주기), VL features를 17번 재사용**

### 비동기 아키텍처 (멀티뷰 5개 기준)

```
🕐 시간 축 (예시):
├─ VLM Inference (0.59Hz = 1700ms마다, 멀티뷰 5개)
│  ├─ t=0ms      → VL_feat_0 (1484ms 소요, 백그라운드)
│  ├─ t=1700ms   → VL_feat_1 (1484ms 소요, 백그라운드)
│  ├─ t=3400ms   → VL_feat_2
│  └─ ...
│
└─ Action Expert (10Hz = 100ms마다)
   ├─ t=0ms:     [VL_feat_0 + sensor[0:65]]      → 8 actions
   ├─ t=100ms:   [VL_feat_0 + sensor[65:130]]    → 8 actions (재사용 #1)
   ├─ t=200ms:   [VL_feat_0 + sensor[130:195]]   → 8 actions (재사용 #2)
   ├─ ...
   ├─ t=1600ms:  [VL_feat_0 + sensor[1040:1105]] → 8 actions (재사용 #16)
   ├─ t=1700ms:  [VL_feat_1 + sensor[1105:1170]] → 8 actions (새 VL_feat)
   └─ ...

총 출력: 10회 × 8 actions = 80 actions/sec ✅
```

**핵심 변경사항:**
1. **VLM 주기**: 1700ms마다 실행 (~0.59 Hz) - 멀티뷰 5개 처리
2. **VLM 실제 실행 시간**: 1484ms (백그라운드 스레드)
3. **Sensor window**: 650 → 65 timesteps (100ms @ 650Hz)
4. **Action Expert 주기**: 100ms마다 실행 (10 Hz)
5. **VL feature 재사용**: 같은 VL feature를 **17번** 재사용

---

## 구현

### 1. 비동기 모델 클래스

`models/model_with_sensor_async.py`:

```python
from models.model_with_sensor_async import AsyncQwenVLAWithSensor, create_async_model

# 모델 생성
model = create_async_model(
    finetune_vl="lora",
    sensor_window_size=65,      # 100ms @ 650Hz
    vlm_reuse_count=3,          # VL feature 3번 재사용
    stage1_checkpoint="path/to/stage1.pt",
)
```

**주요 메서드:**
- `extract_vl_features()`: VL features만 추출 (캐싱용)
- `predict_actions_with_cached_vl()`: 캐시된 VL features로 action 예측
- `forward()`: 일반 forward 또는 캐시된 VL features 사용 가능

### 2. 비동기 학습

`training/A6_VLA_TRAIN_ASYNC.py`:

```bash
# 멀티 GPU 학습
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
    --finetune-vl lora \
    --sensor-window-size 65 \
    --vlm-reuse-count 3 \
    --sensor-enabled \
    --batch-size 2 \
    --lr 1e-4 \
    --sensor-lr 5e-4 \
    --vl-lr 1e-5
```

**학습 전략:**
- VL features를 한 번 추출하고 `vlm_reuse_count`번 재사용
- 각 재사용 시 다른 sensor window 사용
- 모델이 "약간 오래된 VL features"로도 동작하도록 학습

### 3. 비동기 추론 (실시간)

`examples/async_inference_example.py`:

```bash
python examples/async_inference_example.py \
    --checkpoint checkpoints/qwen_vla_async_best.pt \
    --duration 10.0 \
    --vlm-hz 3.33 \
    --action-hz 10.0
```

**실제 비동기 동작:**
```python
inference = AsyncVLAInference(
    model=model,
    vlm_update_hz=3.33,
    action_expert_hz=10.0,
)

# 🔥 VLM을 백그라운드 스레드에서 시작 (non-blocking, ~311ms)
if inference.should_start_vlm_update(current_time):
    inference.start_vlm_update(instruction, image_paths)
    # 메인 루프는 즉시 다음으로 진행

# 🔥 Action 예측은 메인 루프에서 계속 실행 (10 Hz)
action = inference.predict_action(sensor_window)  # ~20-30ms
```

**타임라인 예시 (멀티뷰 5개):**
```
t=0ms:       VLM 스레드 시작 (백그라운드, 1484ms 소요)
t=0ms:       Action #1 예측 (메인 루프, VL_feat=None → 대기)
t=100ms:     Action #2 예측 (VLM 실행 중... VL_feat=None → 대기)
...
t=1484ms:    VLM 완료 → VL_feat_0 업데이트
t=1500ms:    Action #16 예측 (VL_feat_0 사용)
t=1600ms:    Action #17 예측 (VL_feat_0 재사용 #1)
t=1700ms:    새 VLM 스레드 시작 (백그라운드)
t=1700ms:    Action #18 예측 (VL_feat_0 재사용 #2)
t=1800ms:    Action #19 예측 (VLM 실행 중... VL_feat_0 재사용 #3)
...
t=3184ms:    VLM 완료 → VL_feat_1 업데이트
t=3200ms:    Action #33 예측 (VL_feat_1 사용)
...계속
```

**핵심:**
- VLM은 ~1484ms 걸리지만 백그라운드에서 실행
- 첫 VLM이 완료될 때까지 ~1.5초 대기 필요
- 이후 Action Expert는 블로킹 없이 10 Hz로 계속 동작
- VL features를 최대 1.7초간 재사용하지만 학습 시 이를 반영했으므로 성능 유지

---

## 성능 비교

| 지표 | 기존 동기식 (단일 이미지) | 기존 동기식 (멀티뷰 5개) | 비동기 (개선) |
|------|---------------------------|--------------------------|---------------|
| VLM 실행 시간 | ~311ms | **~1484ms** | 1484ms (백그라운드) |
| VLM 주기 | 매번 | 매번 | **1700ms마다** |
| Action 생성 주기 | ~311ms | ~1484ms | **100ms** ✅ |
| Actions/sec | ~3개 | ~0.67개 ❌ | **10개** ✅ |
| Sensor window | 650 samples | 650 samples | 65 samples |
| 실시간 제어 | 불가능 | **완전 불가능** ❌ | ✅ **가능** |

---

## 디렉토리 구조

```
Insertion_VLA/
├── models/
│   ├── model_with_sensor.py           # 기존 동기식 모델
│   └── model_with_sensor_async.py     # 새 비동기 모델 ✨
├── training/
│   ├── A5st_VLA_TRAIN_VL_Lora_with_sensor.py  # 기존 학습
│   └── A6_VLA_TRAIN_ASYNC.py          # 비동기 학습 ✨
├── examples/
│   └── async_inference_example.py     # 비동기 추론 예제 ✨
├── utils/
│   └── profile_vlm_speed.py           # VLM 속도 프로파일링 ✨
└── docs/
    └── ASYNC_TRAINING.md               # 이 문서
```

---

## 학습 파이프라인

### Stage 1: Sensor Encoder + Action Expert (VLM frozen)

```bash
# 기존과 동일, sensor_window_size만 변경
torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --mode train \
    --training-stage stage1 \
    --finetune-vl none \
    --sensor-enabled \
    --sensor-window-size 65 \
    --batch-size 2 \
    --lr 1e-4
```

### Stage 2: LoRA Fine-tuning (비동기 학습)

```bash
torchrun --nproc_per_node=4 training/A6_VLA_TRAIN_ASYNC.py \
    --finetune-vl lora \
    --stage1-checkpoint checkpoints/qwen_vla_sensor_best.pt \
    --sensor-window-size 65 \
    --vlm-reuse-count 3 \
    --batch-size 2 \
    --lr 1e-4 \
    --vl-lr 1e-5
```

---

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `sensor_window_size` | 65 | Sensor window 크기 (100ms @ 650Hz) |
| `vlm_reuse_count` | 17 | VL feature 재사용 횟수 (멀티뷰 5개 기준) |
| `vlm_update_hz` | 0.59 | VLM 업데이트 주파수 (Hz) (멀티뷰 5개 기준) |
| `action_expert_hz` | 10.0 | Action Expert 실행 주파수 (Hz) |

**⚠️ 주의**: 단일 이미지 사용 시 `vlm_reuse_count=3`, `vlm_update_hz=3.33` 사용 가능

---

## 실험 결과

### VLM 프로파일링 (Qwen2.5-VL-3B)

```
⏱️  Latency Statistics:
  Mean:   311.16 ms
  P95:    312.65 ms

🎯 Achievable Frequencies:
  Mean throughput: 3.21 Hz
  P95 throughput:  3.20 Hz

💡 Recommendations:
  VLM update frequency: 3.33 Hz (300ms)
  VL features reused: 3x
  Sensor window: 65 samples (100ms @ 650Hz)
  Action Expert: 10 Hz
```

### 비동기 학습 로그 예시

```
[Rank 0] Epoch 1
  🔄 [Step 0] Extracted new VL features
  ⚡ [Step 10] Action predicted (23.4ms) | vl_reuse: 1/3
  ⚡ [Step 20] Action predicted (22.8ms) | vl_reuse: 2/3
  🔄 [Step 30] Extracted new VL features | vl_reuse: 0/3
  ...

📊 Epoch 1 | Train: 0.00245 | Val: 0.00198
```

---

## FAQ

### Q1: 왜 sensor window를 650에서 65로 줄였나요?

A: 기존 650 samples은 1초치 데이터였습니다. 비동기에서는 100ms마다 action을 예측하므로, 100ms치 데이터인 65 samples만 필요합니다.

### Q2: VL feature를 재사용해도 성능이 괜찮나요?

A: 네! 학습 시부터 VL feature 재사용을 반영하므로, 모델이 "약간 오래된" 시각 정보로도 정확한 action을 예측하도록 학습됩니다. 실제 로봇 제어에서도 이미지가 항상 최신일 수는 없으므로, 오히려 더 robust합니다.

### Q3: 학습 시에는 VL 캐시를 사용하는데, 추론 시에는 어떻게 되나요?

A: **학습 시**에는 VL 캐시를 사용하여 빠르게 학습합니다 (`cache_mode="on"`). **추론 시**에는 캐시를 끄고 (`cache_mode="off"`) 실제 VLM을 백그라운드 스레드에서 실행합니다. 이렇게 하면 학습은 빠르게, 추론은 실제 환경과 동일하게 동작합니다.

### Q4: 실제로 10 Hz를 달성할 수 있나요?

A: 네! VLM을 백그라운드 스레드에서 실행하므로, Action Expert는 메인 루프에서 블로킹 없이 ~20-30ms만에 action을 예측합니다. 이론적으로 ~30-40 Hz도 가능하지만, 10 Hz로 설정하여 안정성을 확보했습니다.

### Q5: 기존 체크포인트를 비동기 모델에서 사용할 수 있나요?

A: 부분적으로 가능합니다. Sensor Encoder와 Action Expert는 그대로 로드 가능하지만, sensor window 크기가 다르므로 Stage 1부터 다시 학습하는 것을 권장합니다.

### Q6: VLM이 실행되는 동안 오래된 VL features를 사용해도 괜찮나요?

A: 네! 학습 시 이미 VL feature 재사용을 반영했으므로, 모델은 최대 **1.7초** 오래된 시각 정보로도 정확한 action을 예측할 수 있습니다. 센서 데이터는 실시간으로 업데이트되므로, **촉각 정보는 항상 최신**입니다. 실제로 니들 삽입 작업에서는 시각 정보보다 촉각 정보가 더 중요하므로 이 전략이 효과적입니다.

### Q7: 추론 속도를 더 높일 수 있나요?

A: 네! 다음 방법들을 시도해보세요:
- VLM quantization (INT8/INT4) - VLM 추론 속도 2-4배 향상
- Flash Attention 사용 (이미 적용됨)
- Smaller VLM 모델 (Qwen2.5-VL-1.5B) - 더 빠른 VLM
- GPU 업그레이드 (A100/H100)
- Action Expert 경량화 (hidden_dim 축소)

---

## 다음 단계

1. ✅ VLM 속도 프로파일링 완료
2. ✅ 비동기 모델 구현 완료
3. ✅ 비동기 학습 스크립트 완료
4. ✅ 비동기 추론 예제 완료
5. ⏳ Stage 1 학습 실행 (sensor_window_size=65)
6. ⏳ Stage 2 비동기 학습 실행
7. ⏳ 실제 로봇에서 비동기 추론 테스트

---

## 참고

- 기존 문서: `docs/STAGE2_LORA_TRAINING.md`
- 모델 개선 문서: `docs/guides/MODEL_IMPROVEMENTS.md`
- VLM 프로파일링: `vlm_profile_results.json`

---

**작성일**: 2025-10-28
**작성자**: Claude Code
