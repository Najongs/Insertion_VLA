# Diffusion vs Regression 비교 실험

현재 VLA 모델의 action expert를 **Diffusion Policy**로 대체하여 성능을 비교합니다.

## 모델 비교

### 1. **Regression Baseline** (기존)
- **모델**: `models/model_with_sensor.py` - `QwenActionExpertWithSensor`
- **학습**: Direct regression (MSE loss on action delta)
- **추론**: Single forward pass → actions
- **장점**: 빠른 추론 속도, 간단한 학습
- **단점**: Uni-modal prediction, 불확실성 표현 어려움

### 2. **Diffusion Policy** (신규)
- **모델**: `models/model_with_sensor_diffusion.py` - `DiffusionActionExpert`
- **학습**: Denoising diffusion (MSE loss on noise prediction)
- **추론**: Iterative denoising (10-100 steps)
- **장점**: Multi-modal distribution, 더 나은 탐색, 복잡한 행동 표현
- **단점**: 느린 추론 속도 (DDIM으로 완화 가능)

## 핵심 차이점

### Forward Process
```python
# Regression
pred_actions = action_expert(vl_features, sensor_features, z_chunk)
loss = MSE(pred_actions, gt_actions)

# Diffusion
noisy_actions = add_noise(gt_actions, timestep)
eps_pred = action_expert(noisy_actions, timestep, vl_features, sensor_features)
loss = MSE(eps_pred, noise)
```

### Inference
```python
# Regression (1 step)
actions = model.forward(images, sensor_data)

# Diffusion (10-100 steps, iterative denoising)
actions = model.sample(images, sensor_data, ddim_steps=10)
```

## 사용법

### 1. Regression Baseline 학습 (기존)
```bash
# Single GPU
python training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4

# Multi-GPU
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_VL_Lora_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --epochs 20
```

### 2. Diffusion Policy 학습 (신규)
```bash
# Single GPU
python training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --epochs 20 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 1e-4 \
    --diffusion_timesteps 100

# Multi-GPU
torchrun --nproc_per_node=4 \
    training/A5st_VLA_TRAIN_Diffusion_with_sensor.py \
    --dataset_dir /home/najo/NAS/VLA/dataset \
    --epochs 20 \
    --diffusion_timesteps 100
```

### 3. 추론 비교

#### Regression (빠름)
```python
from models.model_with_sensor import QwenVLAWithSensor

model = QwenVLAWithSensor(...)
model.load_state_dict(torch.load("checkpoints/regression_best.pt"))

# 1회 forward pass
pred_actions, delta = model(text_inputs, image_inputs, z_chunk, sensor_data)
```

#### Diffusion (느림, 하지만 더 나은 성능 기대)
```python
from models.model_with_sensor_diffusion import QwenVLAWithSensorDiffusion

model = QwenVLAWithSensorDiffusion(diffusion_timesteps=100)
model.load_state_dict(torch.load("checkpoints/diffusion_best.pt"))

# Iterative denoising (DDIM 10 steps)
pred_actions = model.forward(text_inputs, image_inputs, actions=None, sensor_data=None)
# OR explicit sampling
pred_actions = model.action_expert.sample(vl_tokens, sensor_features, ddim_steps=10)
```

## 실험 세팅 권장사항

### Diffusion 하이퍼파라미터
| 파라미터 | 추천값 | 설명 |
|---------|--------|------|
| `diffusion_timesteps` | 100 | 학습 시 noise schedule 길이 |
| `ddim_steps` | 10 | 추론 시 denoising steps (빠르게) |
| `schedule` | 'cosine' | Beta schedule (cosine > linear) |
| `hidden_dim` | 512-1024 | Action expert 크기 |

### 학습 파라미터 (동일하게 유지)
- Batch size: 4 (per GPU)
- Gradient accumulation: 8
- Learning rate: 1e-4
- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: Warmup(3%) → Hold(2%) → Cosine decay

### 평가 메트릭
1. **성공률** (Success Rate): 작업 완료 여부
2. **경로 오차** (Path Error): Predicted vs GT trajectory L2 distance
3. **추론 시간** (Inference Time):
   - Regression: ~100-200ms
   - Diffusion (DDIM 10): ~500-1000ms
   - Diffusion (DDPM 100): ~5-10s
4. **다양성** (Diversity): Multi-modal distribution quality

## 파일 구조
```
models/
├── model_with_sensor.py              # Regression baseline
└── model_with_sensor_diffusion.py   # Diffusion policy

training/
├── A5st_VLA_TRAIN_VL_Lora_with_sensor.py    # Regression 학습
├── A5st_VLA_TRAIN_Diffusion_with_sensor.py  # Diffusion 학습
└── README_DIFFUSION.md                       # 이 파일

Make_dataset/
└── Realtime_inference_receiver.py    # 실시간 추론 (regression 전용)
    # TODO: Diffusion 버전도 추가 가능
```

## 주요 구현 세부사항

### Diffusion Schedule
```python
# Cosine schedule (더 안정적)
betas = cosine_beta_schedule(timesteps=100)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Forward diffusion (add noise)
x_t = sqrt(alpha_cumprod[t]) * x_0 + sqrt(1 - alpha_cumprod[t]) * noise

# Reverse diffusion (predict noise)
eps_pred = model(x_t, t, condition)
x_0_pred = (x_t - sqrt(1-alpha) * eps_pred) / sqrt(alpha)
```

### Action Expert Architecture
```python
class DiffusionActionExpert:
    - Timestep embedding (sinusoidal, 128-dim)
    - Condition encoder (VL + Sensor fusion)
    - Action embedding (encode noisy actions)
    - Temporal transformer (process sequence)
    - Output head (predict noise)
```

### Sensor Fusion (동일)
- **concat**: Concatenate VL + sensor features
- **cross_attention**: Cross-attention between VL and sensor
- **gated**: Gated fusion with learned gates

## 예상 결과

### 성능 비교 (예상)
| 메트릭 | Regression | Diffusion |
|-------|-----------|----------|
| Success Rate | Baseline | **+5-10%** |
| Path Error (mm) | Baseline | **-10-20%** |
| Inference (ms) | 150 | 600 (DDIM 10) |
| Training Time | 1x | 1.2-1.5x |

### Trade-offs
- **Diffusion 사용 권장**: 정밀도가 중요한 경우 (needle insertion)
- **Regression 사용 권장**: 실시간 반응 속도가 중요한 경우

## Wandb 로깅

### Regression
- Project: `QwenVLA-Sensor`
- Metrics: `train/loss`, `train/lr`, `val/loss`

### Diffusion
- Project: `QwenVLA-Diffusion-Sensor`
- Metrics: `train/loss`, `train/lr`, `val/loss`, `train/timestep_dist`

## 참고 논문

1. **DDPM** - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
2. **DDIM** - Denoising Diffusion Implicit Models (Song et al., 2021) - 빠른 샘플링
3. **Diffusion Policy** - Diffusion Policy (Chi et al., 2023) - Robotics에 적용

## 다음 단계

1. ✅ Diffusion 모델 구현
2. ✅ Diffusion 학습 코드 작성
3. ⏳ 두 모델 학습 및 비교
4. ⏳ 실시간 추론 receiver에 diffusion 추가
5. ⏳ 결과 분석 및 논문 작성
