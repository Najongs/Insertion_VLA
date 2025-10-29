# Real-time Inference Guide

## 개요

최적화된 비동기 VLA 모델을 이용한 실시간 추론 가이드입니다.

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources                             │
├──────────────┬──────────────┬──────────────────────────────┤
│ Robot Sender │ Camera Sender│ Sensor Sender                │
│ (ZMQ PUB)    │ (ZMQ PUSH)   │ (UDP)                        │
│ Port 5556    │ Port 5555    │ Port 9999                    │
└──────┬───────┴──────┬───────┴──────┬────────────────────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
       ┌──────────────▼──────────────────────────────┐
       │    Async_inference_receiver.py              │
       │                                             │
       │  ┌─────────────────────────────────────┐   │
       │  │  VLM Thread (Background)            │   │
       │  │  - Updates VL features ~2.6Hz       │   │
       │  │  - Time: ~381ms @ 640x360           │   │
       │  └─────────────────────────────────────┘   │
       │                                             │
       │  ┌─────────────────────────────────────┐   │
       │  │  Action Expert Thread (10Hz)        │   │
       │  │  - Predicts actions every 100ms     │   │
       │  │  - Reuses VL features 4x            │   │
       │  │  - Time: ~20-30ms per prediction    │   │
       │  └─────────────────────────────────────┘   │
       └─────────────────────────────────────────────┘
                      │
                      ▼
              Action Commands (10Hz)
              80 actions/sec total
```

## 최적화 적용사항

1. **이미지 리사이즈: 640x360**
   - VLM 추론 시간: 1487ms → 381ms (3.9배 빠름)
   - 자동 리사이즈 적용

2. **센서 윈도우: 65 samples**
   - 100ms @ 650Hz (기존 650 samples = 1초)
   - 실시간 제어에 적합

3. **비동기 VL-Action 분리**
   - VLM: Background thread (~2.6Hz)
   - Action Expert: Main thread (10Hz)
   - VL features 4번 재사용

## 사용 방법

### 1단계: 데이터 송신 시작

#### Robot Sender (기존 코드 사용 가능)

```bash
# 터미널 1: Robot에서 실행
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset
python Robot_sender.py --robot on
```

**출력:**
```
✅ Starting robot data sampling to ./dataset/Robot_Data/robot_data_XXX.csv at 100.0 Hz...
✅ ZMQ Publisher initialized on *:5556
🚀 Robot control thread started!
```

#### Camera Sender (Jetson에서 실행)

```bash
# 터미널 2: Jetson에서 실행
python Improved_Jetson_sender.py
```

**출력:**
```
✅ Found 4 ZED cameras + 1 OAK camera
✅ ZMQ PUSH socket connected to tcp://10.130.41.XXX:5555
🎥 Capturing at 30 FPS...
```

#### Sensor Sender (별도 머신에서 실행)

센서 데이터는 UDP로 자동 전송됩니다 (기존 설정 사용).

### 2단계: 추론 시작

#### 기본 사용 (추론만)

```bash
# 터미널 3: 추론 서버에서 실행
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt
```

**예상 출력:**
```
================================================================================
Initializing Async VLA Inference Engine
================================================================================
Device: cuda
Model: Qwen/Qwen2.5-VL-3B-Instruct
Image Resize: 640x360
Sensor Window: 65 samples (100ms @ 650Hz)
Action Expert: 10.0 Hz
VL Reuse: 4x
Fusion Strategy: concat
Loading checkpoint: ../checkpoints/qwen_vla_sensor_best.pt
✅ Checkpoint loaded successfully
================================================================================

✅ Camera PULL listening on port 5555
✅ Robot SUB connected to 10.130.41.111:5556
✅ Sensor UDP Receiver started on port 9999
⏳ Calibrating sensor clock offset (first 50 batches)...

✅ Sensor Clock Offset Calibrated: 12.3 ms

================================================================================
Async Real-time Inference Started
================================================================================
Action Expert: 10.0 Hz (every 100ms)
VL Update: ~2.6 Hz (VL features reused 4x)
Image Resolution: 640x360
Sensor Window: 65 samples (100ms)
Device: cuda
Data Saving: Disabled

Waiting for data from all sources...
Press Ctrl+C to stop

🔄 [VL Update #1] Completed in 385ms
[ACTION #1] VL_reuse=1/4 | Actions[0]: [0.123, -0.045, 0.089, ...] | Time: 24.3ms | Sensor: 65/65
[ACTION #2] VL_reuse=2/4 | Actions[0]: [0.125, -0.043, 0.091, ...] | Time: 22.1ms | Sensor: 65/65
[ACTION #3] VL_reuse=3/4 | Actions[0]: [0.127, -0.041, 0.093, ...] | Time: 23.5ms | Sensor: 65/65
[ACTION #4] VL_reuse=4/4 | Actions[0]: [0.129, -0.039, 0.095, ...] | Time: 21.8ms | Sensor: 65/65
🔄 [VL Update #2] Completed in 378ms
[ACTION #5] VL_reuse=1/4 | Actions[0]: [0.131, -0.037, 0.097, ...] | Time: 24.1ms | Sensor: 65/65
...

--- Status (14:32:15) ---
VL Updates: 25 | VL avg: 381ms
Actions: 100 | Action avg: 23.2ms
Images recv: View1:3000, View2:3000, View3:3000, View4:3000, View5:3000
Sensor buffer: 65/65
Robot: J1=45.23°, Px=123.45mm
```

#### 데이터 저장 모드 (디버깅용)

```bash
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --save-data
```

저장되는 데이터:
- 이미지: `async_inference_YYYYMMDD_HHMMSS/View1-5/*.jpg`
- 로봇 상태: `robot_state_YYYYMMDD_HHMMSS.csv`
- 센서 데이터: `sensor_data_YYYYMMDD_HHMMSS.npz`
- 추론 결과: `inference_results_YYYYMMDD_HHMMSS.json`

#### VL Reuse 커스터마이징

```bash
# VL features를 3번만 재사용 (더 빈번한 VLM 업데이트)
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --vl-reuse 3

# VL features를 6번 재사용 (덜 빈번한 VLM 업데이트)
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --vl-reuse 6
```

## 성능 모니터링

### 주요 지표

**VL Update (Background)**
- 목표: ~2.6 Hz (380ms 간격)
- 실제: 375-390ms (정상)
- 문제: >500ms (GPU 과부하 또는 네트워크 지연)

**Action Prediction (10Hz)**
- 목표: ~100ms 간격 (10Hz)
- Action Expert 시간: 20-30ms (정상)
- 문제: >100ms (GPU 과부하)

**센서 버퍼**
- 정상: 60-65/65 samples
- 주의: <30/65 samples (센서 데이터 유실)

### 로그 해석

```
[ACTION #42] VL_reuse=2/4 | Actions[0]: [0.123, -0.045, 0.089, ...] | Time: 24.3ms | Sensor: 65/65
```

- `ACTION #42`: 42번째 action 예측
- `VL_reuse=2/4`: 현재 VL features 2번째 재사용 중 (총 4번 재사용 예정)
- `Actions[0]`: 첫 번째 horizon의 action 값
- `Time: 24.3ms`: Action Expert 추론 시간 (목표: <100ms)
- `Sensor: 65/65`: 센서 버퍼 상태 (충분히 찼음)

```
🔄 [VL Update #5] Completed in 385ms
```

- VL features 업데이트 완료
- 385ms 소요 (정상 범위: 375-400ms)

## 트러블슈팅

### 1. VL Update가 너무 느림 (>500ms)

**증상:**
```
🔄 [VL Update #10] Completed in 650ms
```

**원인:**
- GPU 메모리 부족
- 다른 프로세스가 GPU 사용 중
- 이미지 해상도가 640x360로 리사이즈되지 않음

**해결:**
```bash
# GPU 사용량 확인
nvidia-smi

# 다른 프로세스 종료
pkill -f "python.*train"

# 이미지 리사이즈 확인 (코드 내부에서 자동)
```

### 2. Action Expert가 너무 느림 (>50ms)

**증상:**
```
[ACTION #10] ... | Time: 89.5ms | ...
```

**원인:**
- GPU 메모리 부족
- 센서 윈도우가 650 samples로 설정됨 (잘못된 설정)

**해결:**
```bash
# 모델 재로드 확인
# 센서 윈도우가 65로 설정되어 있는지 확인
```

### 3. 센서 버퍼가 계속 비어있음

**증상:**
```
Sensor buffer: 5/65
```

**원인:**
- 센서 UDP 데이터 수신 안 됨
- 센서 sender가 실행되지 않음
- 방화벽 차단

**해결:**
```bash
# UDP 포트 확인
sudo netstat -ulpn | grep 9999

# 방화벽 확인
sudo ufw status
sudo ufw allow 9999/udp

# 센서 sender 재시작
```

### 4. 이미지가 수신되지 않음

**증상:**
```
[WAIT] VL Features: False | Images: False (0/5) | Sensor: True (65/65)
```

**원인:**
- Camera sender가 실행되지 않음
- ZMQ 연결 문제
- 네트워크 문제

**해결:**
```bash
# ZMQ 포트 확인
netstat -an | grep 5555

# Camera sender 재시작 (Jetson)
python Improved_Jetson_sender.py

# 네트워크 연결 확인
ping 10.130.41.XXX
```

## 추론 결과 활용

### JSON 형식

```json
[
  {
    "timestamp": 1234567890.123,
    "actions": [
      [0.123, -0.045, 0.089, 0.012, -0.034, 0.067, 0.5],  // Horizon 1
      [0.125, -0.043, 0.091, 0.014, -0.032, 0.069, 0.5],  // Horizon 2
      // ... (8 horizons total)
    ],
    "delta": [...],
    "inference_time": 0.0243,
    "vl_update_number": 12,
    "robot_state": {
      "joints": [45.2, 12.3, ...],
      "pose": [123.4, 56.7, ...],
      "timestamp": 1234567890.120
    }
  },
  ...
]
```

### Python으로 결과 읽기

```python
import json
import numpy as np

# Load results
with open("async_inference_20251029_143015/inference_results_20251029_143015.json") as f:
    results = json.load(f)

# Extract actions
for i, result in enumerate(results):
    timestamp = result['timestamp']
    actions = np.array(result['actions'])  # (8, 7)

    # Use first action in horizon
    first_action = actions[0]  # (7,)
    joints = first_action[:6]  # Joint commands
    gripper = first_action[6]  # Gripper command

    print(f"[{i}] Time: {timestamp:.3f} | Joints: {joints} | Gripper: {gripper:.2f}")
```

## 성능 요약

| 항목 | 값 | 비고 |
|------|-----|------|
| Action Expert 주파수 | 10 Hz | 100ms 간격 |
| VLM 업데이트 주파수 | ~2.6 Hz | 381ms @ 640x360 |
| VL feature reuse | 4x | 400ms마다 갱신 |
| 총 action 출력 | 80 actions/sec | 10Hz × 8 horizon |
| Action Expert 시간 | 20-30ms | GPU 여유 |
| VLM 추론 시간 | ~381ms | 5 views @ 640x360 |
| 센서 윈도우 | 65 samples | 100ms @ 650Hz |
| 메모리 사용 | ~8GB GPU | 최적화됨 |

## 다음 단계

1. ✅ **모델 체크포인트 준비**: Stage 1 학습 완료
2. ✅ **추론 코드 작성**: Async_inference_receiver.py
3. **실제 로봇 테스트**: 로봇에서 데이터 송수신 테스트
4. **성능 튜닝**: VL reuse count 조정
5. **실시간 제어**: Action commands를 로봇에 적용

준비되었습니다! 🚀
