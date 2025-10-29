# Inference Pipeline Testing Guide

## 개요

최적화된 비동기 VLA 모델의 완전한 실시간 추론 파이프라인을 테스트하는 가이드입니다.

## 시스템 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
├──────────────────┬──────────────────┬──────────────────────────┤
│ Robot PC         │ Jetson (Camera)  │ Sensor PC (C++)          │
│ - Robot_sender   │ - Camera_sender  │ - Sensor_sender          │
│ - 10Hz           │ - 5Hz            │ - 650Hz                  │
│ - ZMQ PUB        │ - ZMQ PUSH       │ - UDP                    │
│ - Port 5556      │ - Port 5555      │ - Port 9999              │
└────────┬─────────┴────────┬─────────┴────────┬─────────────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
         ┌──────────────────▼───────────────────────────┐
         │     Inference PC                             │
         │     Async_inference_receiver.py              │
         │                                              │
         │  ┌────────────────────────────────────────┐ │
         │  │  VLM Thread (Background, ~2.6Hz)       │ │
         │  │  Time: ~381ms @ 640x360                │ │
         │  └────────────────────────────────────────┘ │
         │                                              │
         │  ┌────────────────────────────────────────┐ │
         │  │  Action Expert Thread (10Hz)           │ │
         │  │  Time: ~20-30ms per prediction         │ │
         │  └────────────────────────────────────────┘ │
         └──────────────────────────────────────────────┘
                            │
                            ▼
                   Actions (10Hz)
```

## 준비사항

### 1. 하드웨어

- **Robot PC**: Mecademic 로봇 제어
- **Jetson**: ZED 카메라 4대 + OAK 카메라 1대
- **Sensor PC**: 센서 데이터 수집 (C++)
- **Inference PC**: GPU 서버 (RTX 4090 이상 권장)

### 2. 네트워크 설정

모든 PC가 같은 네트워크에 연결되어 있어야 합니다.

```bash
# IP 주소 예시
Robot PC:     10.130.41.110
Jetson:       10.130.41.111
Sensor PC:    10.130.41.112
Inference PC: 10.130.41.113
```

### 3. 소프트웨어 설치

**모든 PC (Python이 필요한 경우):**
```bash
pip install zmq numpy
```

**Jetson:**
```bash
pip install pyzmq opencv-python pyzed  # ZED SDK 필요
```

**Inference PC:**
```bash
pip install torch transformers zmq numpy pillow
```

### 4. 모델 체크포인트

Stage 1 학습 완료 후 생성된 체크포인트가 필요합니다:
```bash
./checkpoints/qwen_vla_sensor_best.pt
```

## 테스트 단계

### Phase 1: 개별 컴포넌트 테스트

각 컴포넌트를 개별적으로 테스트하여 정상 작동을 확인합니다.

#### 1.1 Robot Sender 테스트

**Robot PC에서 실행:**

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

# 시뮬레이션 모드 (로봇 없이 테스트)
python Optimized_Robot_sender.py --robot off
```

**예상 출력:**
```
================================================================================
🤖 Optimized Robot Sender
================================================================================
Mode: Simulation Mode
Send Rate: 10 Hz (optimized for Action Expert)
CSV Output: ./dataset/Robot_Data/robot_data_YYYYMMDD_HHMMSS.csv
ZMQ Port: 5556
================================================================================

🤖 Creating dummy robot for simulation...
✅ Starting robot data sampling to ./dataset/Robot_Data/... at 10.0 Hz...
✅ ZMQ Publisher bound to tcp://*:5556 at 10 Hz.
   Topic: 'robot_state', Payload Size: 68 bytes

================================================================================
✅ Optimized Robot Sender Started
================================================================================
Sampling & Sending at 10 Hz
Press Ctrl+C to stop

📊 Robot sampling: 50 samples collected (10.0 Hz avg)
📡 ZMQ sending: 50 messages (10.0 Hz avg, target: 10.0 Hz)
```

**검증 사항:**
- ✅ Send rate가 10Hz를 유지하는지 확인
- ✅ ZMQ 바인딩 성공 확인
- ✅ CSV 파일 생성 확인

**실제 로봇 테스트:**
```bash
# 로봇 연결 후
python Optimized_Robot_sender.py --robot on
```

#### 1.2 Camera Sender 테스트

**Jetson에서 실행:**

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

# 먼저 카메라 연결 확인
python -c "import pyzed.sl as sl; print('ZED SDK OK')"

# Camera sender 실행 (Inference PC IP 설정 필요)
python Optimized_Camera_sender.py --server-ip 10.130.41.113
```

**예상 출력:**
```
================================================================================
🎥 Optimized Camera Sender for Async VLA Inference
================================================================================
Mode: Real-time capture and send
Capture Rate: 5.0 Hz (200ms interval)
Image Size: 1280x720 (will be resized to 640x360 on receiver)
ZMQ Server: 10.130.41.113:5555
Views: 5 (ZED left x4 + OAK x1)
================================================================================

🔍 Searching for cameras...
✅ Found 4 ZED cameras
✅ Found 1 OAK camera

✅ ZMQ PUSH socket connected to tcp://10.130.41.113:5555

🎥 Starting synchronized capture at 5.0 Hz...
Press Ctrl+C to stop

📊 Capture stats: 25 frames sent | Avg: 5.0 Hz | Views: 5
```

**검증 사항:**
- ✅ 5개 카메라 모두 인식 확인
- ✅ 5Hz 캡처 확인
- ✅ ZMQ 연결 성공 확인

#### 1.3 Sensor Sender 테스트

**Sensor PC에서 실행 (C++ 코드):**

사용자가 제공한 C++ 센서 sender를 실행하여 650Hz UDP 전송 확인.

```bash
# 센서 sender 실행 예시 (사용자 제공 코드)
./sensor_sender --rate 650 --port 9999 --target 10.130.41.113
```

**검증 사항:**
- ✅ UDP 포트 9999 전송 확인
- ✅ 650Hz 전송 속도 확인

### Phase 2: 수신 테스트 (Inference PC)

각 데이터 소스가 정상적으로 수신되는지 확인합니다.

#### 2.1 Robot 데이터 수신 테스트

**Inference PC에서 실행:**

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

# ZMQ SUB 테스트
python -c "
import zmq
import struct
import time

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://10.130.41.110:5556')
socket.subscribe(b'robot_state')

print('Waiting for robot data...')
for i in range(10):
    topic, payload = socket.recv_multipart()
    ts, send_ts, force, *joints_pose = struct.unpack('<ddf12f', payload)
    print(f'[{i}] Timestamp: {ts:.3f}, Joints: {joints_pose[:6]}')
    time.sleep(0.1)
print('✅ Robot data received OK')
"
```

**예상 출력:**
```
Waiting for robot data...
[0] Timestamp: 1730188934.123, Joints: [10.5, 20.7, 30.3, 0.0, 45.0, 0.0]
[1] Timestamp: 1730188934.223, Joints: [10.6, 20.8, 30.4, 0.0, 45.0, 0.0]
...
✅ Robot data received OK
```

#### 2.2 Camera 데이터 수신 테스트

```bash
python -c "
import zmq
import struct
import time

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind('tcp://*:5555')

print('Waiting for camera data...')
for i in range(5):
    metadata_bytes = socket.recv()
    timestamp, view_count = struct.unpack('<dI', metadata_bytes[:12])
    print(f'[{i}] Timestamp: {timestamp:.3f}, Views: {view_count}')

    for v in range(view_count):
        jpg_data = socket.recv()
        print(f'  View {v+1}: {len(jpg_data)} bytes')

    time.sleep(0.2)
print('✅ Camera data received OK')
"
```

**예상 출력:**
```
Waiting for camera data...
[0] Timestamp: 1730188934.456, Views: 5
  View 1: 45678 bytes
  View 2: 46123 bytes
  View 3: 45890 bytes
  View 4: 46234 bytes
  View 5: 34567 bytes
...
✅ Camera data received OK
```

#### 2.3 Sensor 데이터 수신 테스트

```bash
python -c "
import socket
import struct
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 9999))
sock.settimeout(2.0)

print('Waiting for sensor data...')
for i in range(50):
    try:
        data, addr = sock.recvfrom(8192)
        print(f'[{i}] Received {len(data)} bytes from {addr}')
    except socket.timeout:
        print('❌ Timeout - no sensor data received')
        break
else:
    print('✅ Sensor data received OK')
"
```

### Phase 3: 통합 추론 테스트

모든 데이터 소스가 정상 작동하면, 완전한 추론 파이프라인을 실행합니다.

#### 3.1 기본 추론 실행

**Inference PC에서 실행:**

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --robot-ip 10.130.41.110 \
    --robot-port 5556 \
    --camera-port 5555 \
    --sensor-port 9999 \
    --vl-reuse 4
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
✅ Robot SUB connected to 10.130.41.110:5556
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
[ACTION #1] VL_reuse=1/4 | Actions[0]: [0.123, -0.045, 0.089, 0.012, -0.034, 0.067, 0.5] | Time: 24.3ms | Sensor: 65/65
[ACTION #2] VL_reuse=2/4 | Actions[0]: [0.125, -0.043, 0.091, 0.014, -0.032, 0.069, 0.5] | Time: 22.1ms | Sensor: 65/65
[ACTION #3] VL_reuse=3/4 | Actions[0]: [0.127, -0.041, 0.093, 0.016, -0.030, 0.071, 0.5] | Time: 23.5ms | Sensor: 65/65
[ACTION #4] VL_reuse=4/4 | Actions[0]: [0.129, -0.039, 0.095, 0.018, -0.028, 0.073, 0.5] | Time: 21.8ms | Sensor: 65/65
🔄 [VL Update #2] Completed in 378ms
[ACTION #5] VL_reuse=1/4 | Actions[0]: [0.131, -0.037, 0.097, 0.020, -0.026, 0.075, 0.5] | Time: 24.1ms | Sensor: 65/65
...

--- Status (14:32:15) ---
VL Updates: 25 | VL avg: 381ms
Actions: 100 | Action avg: 23.2ms
Images recv: View1:500, View2:500, View3:500, View4:500, View5:500
Sensor buffer: 65/65
Robot: J1=45.23°, Px=123.45mm
```

#### 3.2 데이터 저장 모드 (디버깅용)

```bash
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --robot-ip 10.130.41.110 \
    --robot-port 5556 \
    --camera-port 5555 \
    --sensor-port 9999 \
    --vl-reuse 4 \
    --save-data
```

저장되는 데이터:
- 이미지: `async_inference_YYYYMMDD_HHMMSS/View1-5/*.jpg`
- 로봇 상태: `robot_state_YYYYMMDD_HHMMSS.csv`
- 센서 데이터: `sensor_data_YYYYMMDD_HHMMSS.npz`
- 추론 결과: `inference_results_YYYYMMDD_HHMMSS.json`

### Phase 4: 성능 검증

#### 4.1 타이밍 검증

**정상 범위:**

| 항목 | 목표 | 정상 범위 | 문제 |
|------|------|-----------|------|
| VL Update | ~381ms | 375-400ms | >500ms |
| Action Expert | ~25ms | 20-30ms | >50ms |
| Action Period | 100ms | 95-105ms | >110ms |
| Sensor Buffer | 65/65 | 60-65 | <30 |

**로그 확인:**
```bash
# VL Update 시간 추출
grep "VL Update" inference.log | awk '{print $6}' | sed 's/ms//' > vl_times.txt
python -c "
import numpy as np
times = np.loadtxt('vl_times.txt')
print(f'VL Update: mean={times.mean():.1f}ms, std={times.std():.1f}ms, max={times.max():.1f}ms')
"

# Action Expert 시간 추출
grep "ACTION" inference.log | grep "Time:" | awk '{print $9}' | sed 's/ms//' > action_times.txt
python -c "
import numpy as np
times = np.loadtxt('action_times.txt')
print(f'Action Expert: mean={times.mean():.1f}ms, std={times.std():.1f}ms, max={times.max():.1f}ms')
"
```

#### 4.2 데이터 흐름 검증

```python
import json
import numpy as np

# 추론 결과 로드
with open('async_inference_YYYYMMDD_HHMMSS/inference_results_YYYYMMDD_HHMMSS.json') as f:
    results = json.load(f)

# 타임스탬프 분석
timestamps = [r['timestamp'] for r in results]
intervals = np.diff(timestamps)

print(f"Total actions: {len(results)}")
print(f"Duration: {timestamps[-1] - timestamps[0]:.1f}s")
print(f"Action rate: {len(results) / (timestamps[-1] - timestamps[0]):.1f} Hz")
print(f"Interval: mean={intervals.mean()*1000:.1f}ms, std={intervals.std()*1000:.1f}ms")

# 액션 값 분석
actions_first = np.array([r['actions'][0] for r in results])  # First horizon
print(f"\nAction statistics (first horizon):")
print(f"  Mean: {actions_first.mean(axis=0)}")
print(f"  Std:  {actions_first.std(axis=0)}")
print(f"  Min:  {actions_first.min(axis=0)}")
print(f"  Max:  {actions_first.max(axis=0)}")
```

## 트러블슈팅

### 문제 1: 데이터 수신 안 됨

**증상:**
```
[WAIT] VL Features: False | Images: False (0/5) | Sensor: False (0/65)
```

**원인 및 해결:**

1. **네트워크 연결 문제**
```bash
# 각 PC에서 ping 테스트
ping 10.130.41.110  # Robot PC
ping 10.130.41.111  # Jetson
ping 10.130.41.112  # Sensor PC
```

2. **방화벽 차단**
```bash
# 포트 열기
sudo ufw allow 5555/tcp
sudo ufw allow 5556/tcp
sudo ufw allow 9999/udp
```

3. **Sender가 실행되지 않음**
```bash
# 각 PC에서 sender 프로세스 확인
ps aux | grep sender
```

### 문제 2: VL Update가 너무 느림

**증상:**
```
🔄 [VL Update #10] Completed in 650ms
```

**원인 및 해결:**

1. **GPU 메모리 부족**
```bash
nvidia-smi
# 다른 프로세스 종료
pkill -f "python.*train"
```

2. **이미지 리사이즈 미적용**
```python
# Async_inference_receiver.py에서 확인
# processor.image_processor가 640x360으로 설정되었는지 확인
```

3. **Multi-view 이미지 수 확인**
```bash
# 5개 view가 모두 수신되는지 확인
grep "Images recv" inference.log
```

### 문제 3: Action Expert가 너무 느림

**증상:**
```
[ACTION #10] ... | Time: 89.5ms | ...
```

**원인 및 해결:**

1. **센서 윈도우 크기 확인**
```python
# 65 samples로 설정되어 있는지 확인
--sensor-window-size 65
```

2. **GPU 과부하**
```bash
nvidia-smi
# GPU 사용률이 95% 이상이면 VL reuse count 증가
--vl-reuse 6
```

### 문제 4: 센서 버퍼가 계속 비어있음

**증상:**
```
Sensor buffer: 5/65
```

**원인 및 해결:**

1. **UDP 수신 확인**
```bash
sudo netstat -ulpn | grep 9999
```

2. **Sensor sender 확인**
```bash
# Sensor PC에서 프로세스 확인
ps aux | grep sensor_sender
```

3. **Clock offset 문제**
```bash
# Async_inference_receiver.py 로그에서 clock offset 확인
# 정상: 10-20ms
# 비정상: >100ms
```

### 문제 5: ZMQ 연결 실패

**증상:**
```
zmq.error.ZMQError: Address already in use
```

**해결:**
```bash
# 포트를 사용하는 프로세스 확인 및 종료
lsof -i :5555
kill -9 <PID>

# 또는 다른 포트 사용
--camera-port 5557
```

## 성능 체크리스트

실제 테스트 시 다음 항목을 확인하세요:

- [ ] **Robot Sender**: 10Hz 유지, ZMQ 바인딩 성공
- [ ] **Camera Sender**: 5Hz 유지, 5개 view 모두 전송
- [ ] **Sensor Sender**: 650Hz 유지, UDP 전송 성공
- [ ] **Receiver**: 모든 데이터 소스 수신 확인
- [ ] **VL Update**: 375-400ms 유지
- [ ] **Action Expert**: 20-30ms 유지
- [ ] **Action Period**: 100ms 간격 유지
- [ ] **Sensor Buffer**: 60-65/65 유지
- [ ] **메모리 사용**: GPU <10GB, CPU <50%
- [ ] **데이터 저장**: 모든 데이터 정상 저장 (--save-data 시)

## 다음 단계

테스트 완료 후:

1. **성능 로그 분석**: VL/Action 시간, 데이터 흐름 확인
2. **Action 명령 검증**: 출력된 action 값이 합리적인지 확인
3. **실제 로봇 적용**: Action 명령을 로봇에 전송하여 제어 테스트
4. **Fine-tuning**: 필요시 모델 재학습 또는 VL reuse count 조정

준비되었습니다! 🚀
