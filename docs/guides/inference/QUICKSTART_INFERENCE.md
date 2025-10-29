# Quick Start: 실시간 추론 파이프라인

## 📋 준비사항 체크리스트

- [ ] Stage 1 학습 완료 (`checkpoints/qwen_vla_sensor_best.pt` 존재)
- [ ] 모든 PC가 같은 네트워크에 연결
- [ ] Robot PC, Jetson, Sensor PC, Inference PC 준비
- [ ] 필요한 패키지 설치 완료

## 🚀 5분 Quick Start

### 1단계: 시스템 검증 (1분)

**Inference PC에서 실행:**

```bash
cd /home/najo/NAS/VLA/Insertion_VLA

# 시스템 체크
python utils/verify_inference_pipeline.py \
    --robot-ip 10.130.41.110 \
    --jetson-ip 10.130.41.111 \
    --checkpoint ./checkpoints/qwen_vla_sensor_best.pt
```

**예상 출력:**
```
================================================================================
Async VLA Inference Pipeline Verification
================================================================================

1. Checking Dependencies
   ✅ ZMQ: Available
   ✅ PyTorch: Available (version 2.0.1+cu118)

2. Checking Network Connectivity
   ✅ Robot PC (10.130.41.110): Reachable on port 5556
   ✅ Jetson (10.130.41.111): Host is reachable

3. Checking Port Availability
   ✅ Camera port 5555: Available
   ✅ Sensor port 9999: Available

4. Checking Model Checkpoint
   ✅ Checkpoint: epoch=95, model_state_dict=OK, optimizer_state_dict=OK

5. Checking GPU Status
   ✅ GPU: GPU 0: NVIDIA RTX 4090 | Total: 24.0GB | Free: 22.5GB | Used: 0.3GB

Summary
   Total checks: 7
   ✅ Passed: 7
   ❌ Failed: 0
   ⚠️  Warnings: 0

   🎉 All checks passed! System is ready for inference.
```

모든 체크가 통과되면 다음 단계로 진행하세요.

### 2단계: 데이터 송신 시작 (2분)

#### Terminal 1: Robot Sender (Robot PC)

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

# 시뮬레이션 모드로 테스트
python Optimized_Robot_sender.py --robot off

# 실제 로봇 사용시
# python Optimized_Robot_sender.py --robot on
```

**확인사항:** `📡 ZMQ sending: ... (10.0 Hz avg)` 출력 확인

#### Terminal 2: Camera Sender (Jetson)

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

# Inference PC의 IP 주소로 수정
python Optimized_Camera_sender.py --server-ip 10.130.41.113
```

**확인사항:** `📊 Capture stats: ... Avg: 5.0 Hz` 출력 확인

#### Terminal 3: Sensor Sender (Sensor PC)

```bash
# 사용자가 제공한 C++ 센서 sender 실행
./sensor_sender --rate 650 --port 9999 --target 10.130.41.113
```

**확인사항:** 650Hz UDP 전송 확인

### 3단계: 추론 실행 (2분)

#### Terminal 4: Inference Receiver (Inference PC)

```bash
cd /home/najo/NAS/VLA/Insertion_VLA/Make_dataset

python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --robot-ip 10.130.41.110 \
    --vl-reuse 4
```

**예상 출력:**
```
================================================================================
Async Real-time Inference Started
================================================================================
Action Expert: 10.0 Hz (every 100ms)
VL Update: ~2.6 Hz (VL features reused 4x)
...

🔄 [VL Update #1] Completed in 385ms
[ACTION #1] VL_reuse=1/4 | Actions[0]: [0.123, ...] | Time: 24.3ms | Sensor: 65/65
[ACTION #2] VL_reuse=2/4 | Actions[0]: [0.125, ...] | Time: 22.1ms | Sensor: 65/65
...
```

**성공 지표:**
- ✅ VL Update: 375-400ms
- ✅ Action Expert: 20-30ms
- ✅ Sensor: 60-65/65
- ✅ Action rate: ~10 Hz

## 📊 성능 모니터링

### 실시간 모니터링

추론 중 다음 지표를 확인하세요:

```
--- Status (14:32:15) ---
VL Updates: 25 | VL avg: 381ms     ← 375-400ms 정상
Actions: 100 | Action avg: 23.2ms   ← 20-30ms 정상
Images recv: View1:3000, ...        ← 모든 view 수신 확인
Sensor buffer: 65/65                ← 60-65 정상
Robot: J1=45.23°, Px=123.45mm      ← 로봇 상태 확인
```

### 로그 분석 (추론 종료 후)

```bash
# 로그 파일 분석
python utils/analyze_inference_logs.py inference.log

# JSON 결과 분석
python utils/analyze_inference_logs.py \
    --results async_inference_20251029_143015/inference_results_20251029_143015.json
```

## 🔧 일반적인 문제 해결

### 문제 1: "No data received"

**증상:**
```
[WAIT] VL Features: False | Images: False (0/5) | Sensor: False (0/65)
```

**해결:**
1. 모든 sender가 실행 중인지 확인
2. 네트워크 연결 확인: `ping 10.130.41.110`
3. 방화벽 확인: `sudo ufw allow 5555/tcp && sudo ufw allow 5556/tcp && sudo ufw allow 9999/udp`

### 문제 2: VL Update가 느림 (>500ms)

**해결:**
```bash
# GPU 사용량 확인
nvidia-smi

# 다른 프로세스 종료
pkill -f "python.*train"

# 이미지 리사이즈 확인 (코드에서 자동 적용됨)
```

### 문제 3: Action Expert가 느림 (>50ms)

**해결:**
```bash
# Sensor 윈도우 크기 확인 (65 samples인지)
# VL reuse count 증가
python Async_inference_receiver.py ... --vl-reuse 6
```

### 문제 4: Sensor buffer가 비어있음 (<30/65)

**해결:**
```bash
# UDP 수신 확인
sudo netstat -ulpn | grep 9999

# Sensor sender 재시작
```

## 📁 데이터 저장 (선택사항)

디버깅이나 분석을 위해 데이터를 저장하려면:

```bash
python Async_inference_receiver.py \
    --checkpoint ../checkpoints/qwen_vla_sensor_best.pt \
    --robot-ip 10.130.41.110 \
    --vl-reuse 4 \
    --save-data
```

저장되는 파일:
- `async_inference_YYYYMMDD_HHMMSS/View1-5/*.jpg` - 이미지
- `robot_state_YYYYMMDD_HHMMSS.csv` - 로봇 상태
- `sensor_data_YYYYMMDD_HHMMSS.npz` - 센서 데이터
- `inference_results_YYYYMMDD_HHMMSS.json` - 추론 결과

## 🎯 성능 최적화 팁

### VL Reuse Count 조정

```bash
# 더 빈번한 VLM 업데이트 (더 높은 정확도, 약간 느림)
--vl-reuse 3

# 덜 빈번한 VLM 업데이트 (더 빠름, 정확도 약간 감소)
--vl-reuse 6
```

### 권장 설정 (RTX 4090 기준)

| 시나리오 | VL Reuse | VLM Rate | Action Rate | 특징 |
|---------|----------|----------|-------------|------|
| 균형 (권장) | 4 | ~2.6 Hz | 10 Hz | 최적의 균형 |
| 고정확도 | 3 | ~3.5 Hz | 10 Hz | VLM 더 자주 업데이트 |
| 고속 | 6 | ~1.7 Hz | 10 Hz | GPU 부담 감소 |

## 📚 추가 문서

- **전체 테스트 가이드**: [INFERENCE_PIPELINE_TEST_GUIDE.md](INFERENCE_PIPELINE_TEST_GUIDE.md)
- **실시간 추론 가이드**: [REALTIME_INFERENCE_GUIDE.md](REALTIME_INFERENCE_GUIDE.md)
- **최적화 상세**: [ASYNC_OPTIMIZATIONS.md](ASYNC_OPTIMIZATIONS.md)
- **Stage 1 학습**: [STAGE1_TRAINING_GUIDE.md](STAGE1_TRAINING_GUIDE.md)

## ✅ 체크리스트

추론 시작 전:
- [ ] 시스템 검증 완료 (`verify_inference_pipeline.py`)
- [ ] 모든 sender 실행 및 정상 작동 확인
- [ ] GPU 메모리 충분 (>10GB 여유)
- [ ] 네트워크 연결 정상

추론 중:
- [ ] VL Update: 375-400ms 유지
- [ ] Action Expert: 20-30ms 유지
- [ ] Sensor buffer: 60-65/65 유지
- [ ] 모든 view 이미지 수신 확인

추론 후:
- [ ] 로그 분석 (`analyze_inference_logs.py`)
- [ ] Action 값 합리성 확인
- [ ] 데이터 저장 확인 (`--save-data` 사용 시)

## 🎉 성공!

모든 지표가 정상이면 추론 파이프라인이 성공적으로 동작하는 것입니다.

다음 단계:
1. 저장된 action 값 검토
2. 실제 로봇에 action 명령 적용 테스트
3. 필요시 VL reuse count 조정
4. 추가 fine-tuning 고려

문제가 발생하면 [INFERENCE_PIPELINE_TEST_GUIDE.md](INFERENCE_PIPELINE_TEST_GUIDE.md)의 트러블슈팅 섹션을 참고하세요.
