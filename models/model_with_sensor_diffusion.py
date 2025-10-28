"""
Vision-Language-Action Model with Sensor + Diffusion Policy
Extends model_with_sensor.py with diffusion-based action prediction

Key Changes:
- DiffusionActionExpert: Replaces direct regression with DDPM-based denoising
- Training: Predicts noise instead of action deltas
- Inference: Iterative denoising process (DDPM/DDIM)

Diffusion Process:
- Forward: a_t = sqrt(alpha_t) * a_0 + sqrt(1-alpha_t) * epsilon
- Reverse: Predict epsilon, recover a_0 iteratively
- Benefits: Multi-modal action distribution, better exploration
"""

import os
from pathlib import Path
import hashlib, fcntl
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

# Import base components
from models.model_with_sensor import (
    SensorEncoder,
    QwenVLAWithSensor as BaseQwenVLAWithSensor
)


# =====================================
# 1️⃣ Diffusion Schedule Utilities
# =====================================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
    More stable than linear schedule
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear schedule (original DDPM)"""
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionSchedule:
    """
    Manages diffusion noise schedule and sampling utilities
    """
    def __init__(self, timesteps=100, schedule='cosine', device='cuda'):
        self.timesteps = timesteps
        self.device = device

        # Compute betas and alphas
        if schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store for training
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Precompute values for forward diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Precompute values for reverse diffusion q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))

    def register_buffer(self, name, tensor):
        """Helper to store tensors as buffers"""
        setattr(self, name, tensor.to(self.device))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: q(x_t | x_0)
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """
        Predict x_0 from x_t and predicted noise
        x_0 = (x_t - sqrt(1-alpha_cumprod) * eps) / sqrt(alpha_cumprod)
        """
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_recip * x_t - sqrt_recipm1 * eps

    def p_mean_variance(self, x_t, t, eps_pred):
        """
        Compute mean and variance for p(x_{t-1} | x_t)
        """
        # Predict x_0
        pred_x0 = self.predict_x0_from_eps(x_t, t, eps_pred)

        # Clip x_0 for stability
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Compute mean
        model_mean = (
            self.sqrt_recip_alphas[t].view(-1, 1, 1) *
            (x_t - self.betas[t].view(-1, 1, 1) * eps_pred / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1))
        )

        # Variance
        model_variance = self.posterior_variance[t].view(-1, 1, 1)
        model_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1, 1)

        return model_mean, model_variance, model_log_variance, pred_x0


# =====================================
# 2️⃣ Diffusion Action Expert
# =====================================
class DiffusionActionExpert(nn.Module):
    """
    Diffusion-based Action Expert with Sensor Fusion

    Architecture:
    - Condition encoder: Fuses VL features + sensor features
    - Noise predictor: U-Net style temporal model
    - Training: Predict noise at random timestep
    - Inference: Iterative denoising (DDPM or DDIM)

    Args:
        vl_dim: Vision-Language feature dimension
        sensor_dim: Sensor feature dimension
        action_dim: Action dimension (7)
        horizon: Action prediction horizon (8)
        timesteps: Number of diffusion steps (100)
        fusion_strategy: How to fuse VL and sensor features
    """
    def __init__(self,
                 vl_dim=3072,
                 sensor_dim=3072,
                 action_dim=7,
                 horizon=8,
                 hidden_dim=512,
                 timesteps=100,
                 fusion_strategy='concat',
                 nhead=8,
                 num_layers=4,
                 dropout=0.1):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.timesteps = timesteps
        self.fusion_strategy = fusion_strategy

        # Diffusion schedule
        self.diffusion = DiffusionSchedule(timesteps=timesteps, schedule='cosine')

        # Timestep embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 🔹 Condition encoder (VL + Sensor fusion)
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.cond_proj = nn.Linear(fused_dim, hidden_dim)
        elif fusion_strategy == 'cross_attention':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'gated':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
            self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'none':
            self.cond_proj = nn.Linear(vl_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # 🔹 Action embedding (encode noisy actions)
        self.action_embed = nn.Linear(action_dim, hidden_dim)

        # 🔹 Temporal transformer (process action sequence)
        # Input: [time_embed + cond_embed + action_embed]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🔹 Output head (predict noise)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        print(f"✅ DiffusionActionExpert initialized:")
        print(f"   Timesteps: {timesteps}, Fusion: {fusion_strategy}")
        print(f"   Action shape: (B, {horizon}, {action_dim})")

    def forward(self, noisy_actions, timesteps, vl_tokens, sensor_features=None):
        """
        Forward pass for training

        Args:
            noisy_actions: (B, H, A) - actions with noise
            timesteps: (B,) - diffusion timestep
            vl_tokens: (B, seq_len, vl_dim) or (B, 1, vl_dim)
            sensor_features: (B, sensor_dim) - optional

        Returns:
            eps_pred: (B, H, A) - predicted noise
        """
        B, H, A = noisy_actions.shape

        # 1. Timestep embedding
        t_embed = self.timestep_embedding(timesteps)  # (B, 128)
        t_embed = self.time_embed(t_embed)  # (B, hidden_dim)
        t_embed = t_embed.unsqueeze(1).expand(-1, H, -1)  # (B, H, hidden_dim)

        # 2. Condition embedding (VL + Sensor)
        cond_embed = self._encode_condition(vl_tokens, sensor_features)  # (B, hidden_dim)
        cond_embed = cond_embed.unsqueeze(1).expand(-1, H, -1)  # (B, H, hidden_dim)

        # 3. Action embedding
        action_embed = self.action_embed(noisy_actions)  # (B, H, hidden_dim)

        # 4. Combine embeddings
        x = t_embed + cond_embed + action_embed  # (B, H, hidden_dim)

        # 5. Temporal processing
        x = self.temporal_encoder(x)  # (B, H, hidden_dim)

        # 6. Predict noise
        eps_pred = self.output_head(x)  # (B, H, A)

        return eps_pred

    def _encode_condition(self, vl_tokens, sensor_features):
        """Encode and fuse VL + Sensor features"""
        if self.fusion_strategy == 'concat' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)  # (B, vl_dim)
            fused = torch.cat([vl_pooled, sensor_features], dim=-1)
            cond = self.cond_proj(fused)  # (B, hidden_dim)

        elif self.fusion_strategy == 'cross_attention' and sensor_features is not None:
            vl_feat = self.vl_proj(vl_tokens)  # (B, seq_len, hidden_dim)
            sensor_feat = self.sensor_proj(sensor_features).unsqueeze(1)  # (B, 1, hidden_dim)
            attn_out, _ = self.cross_attn(sensor_feat, vl_feat, vl_feat)
            cond = self.cond_proj(attn_out.squeeze(1))

        elif self.fusion_strategy == 'gated' and sensor_features is not None:
            vl_pooled = vl_tokens.mean(dim=1)
            vl_feat = self.vl_proj(vl_pooled)
            sensor_feat = self.sensor_proj(sensor_features)
            gate = self.gate(torch.cat([vl_feat, sensor_feat], dim=-1))
            fused = gate * vl_feat + (1 - gate) * sensor_feat
            cond = self.cond_proj(fused)

        else:  # 'none' or sensor not provided
            vl_pooled = vl_tokens.mean(dim=1)
            cond = self.cond_proj(vl_pooled)

        return cond  # (B, hidden_dim)

    @staticmethod
    def timestep_embedding(timesteps, dim=128):
        """
        Sinusoidal timestep embedding (same as Transformer positional encoding)
        """
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    @torch.no_grad()
    def sample(self, vl_tokens, sensor_features=None, batch_size=1, ddim_steps=None):
        """
        Sample actions via iterative denoising (DDPM or DDIM)

        Args:
            vl_tokens: (B, seq_len, vl_dim)
            sensor_features: (B, sensor_dim)
            batch_size: Number of samples
            ddim_steps: If provided, use DDIM sampling (faster)

        Returns:
            actions: (B, H, A) - denoised actions
        """
        device = vl_tokens.device
        H, A = self.horizon, self.action_dim

        # Start from pure noise
        x = torch.randn(batch_size, H, A, device=device)

        # DDIM sampling (faster)
        if ddim_steps is not None:
            return self._ddim_sample(x, vl_tokens, sensor_features, ddim_steps)

        # DDPM sampling (full)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)

            # Compute mean and variance
            mean, variance, log_variance, _ = self.diffusion.p_mean_variance(x, t_batch, eps_pred)

            # Sample x_{t-1}
            noise = torch.randn_like(x) if t > 0 else 0.0
            x = mean + torch.sqrt(variance) * noise

        return x

    @torch.no_grad()
    def _ddim_sample(self, x, vl_tokens, sensor_features, ddim_steps):
        """
        DDIM sampling (faster, deterministic)
        Uses subset of timesteps with eta=0 (deterministic)
        """
        device = x.device
        batch_size = x.shape[0]

        # Select subset of timesteps
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(0, self.timesteps, step_size))[:ddim_steps]
        timesteps = list(reversed(timesteps))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            eps_pred = self.forward(x, t_batch, vl_tokens, sensor_features)

            # Predict x_0
            pred_x0 = self.diffusion.predict_x0_from_eps(x, t_batch, eps_pred)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Get next alpha
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                alpha_next = self.diffusion.alphas_cumprod[t_next]
            else:
                alpha_next = torch.tensor(1.0, device=device)

            alpha_t = self.diffusion.alphas_cumprod[t]

            # DDIM update (eta=0, deterministic)
            sigma_t = 0.0  # eta=0 for deterministic
            x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next - sigma_t**2) * eps_pred

        return x


# =====================================
# 3️⃣ Full VLA Model with Diffusion
# =====================================
class QwenVLAWithSensorDiffusion(BaseQwenVLAWithSensor):
    """
    VLA model with Diffusion Policy
    Inherits from base model but replaces action expert with diffusion version
    """
    def __init__(self,
                 vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 action_dim=7,
                 horizon=8,
                 hidden_dim=1024,
                 cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
                 # Sensor encoder params
                 sensor_enabled=True,
                 sensor_input_channels=1026,
                 sensor_temporal_length=650,
                 sensor_hidden_dim=512,
                 sensor_output_dim=3072,
                 # Fusion params
                 fusion_strategy='concat',
                 # Diffusion params
                 diffusion_timesteps=100,
                 ):
        # Don't call parent __init__ to avoid creating base action expert
        nn.Module.__init__(self)

        print(f"🚀 Loading Qwen-VL-Sensor-Diffusion Model")
        print(f"   Sensor Enabled: {sensor_enabled}")
        print(f"   Fusion Strategy: {fusion_strategy}")
        print(f"   Diffusion Timesteps: {diffusion_timesteps}")

        self.sensor_enabled = sensor_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True
        self.cache_limit_gb = 20.0
        self.strict_cache = False
        self.action_dim = action_dim
        self.horizon = horizon

        # VL Model (Frozen)
        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = self._load_qwen_with_fallback(vl_model_name)

        # Sensor Encoder (Trainable)
        if sensor_enabled:
            self.sensor_encoder = SensorEncoder(
                input_channels=sensor_input_channels,
                temporal_length=sensor_temporal_length,
                hidden_dim=sensor_hidden_dim,
                output_dim=sensor_output_dim,
                use_transformer=True,
                num_transformer_layers=2
            ).to(dtype=torch.bfloat16, device="cuda")
        else:
            self.sensor_encoder = None

        # 🔥 Diffusion Action Expert (Trainable)
        self.action_expert = DiffusionActionExpert(
            vl_dim=self.vl_model.config.hidden_size,
            sensor_dim=sensor_output_dim if sensor_enabled else 0,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            timesteps=diffusion_timesteps,
            fusion_strategy=fusion_strategy if sensor_enabled else 'none'
        ).to(dtype=torch.bfloat16, device="cuda")

        # Freeze VL model
        print("🧊 Freezing Qwen-VL parameters...")
        for p in self.vl_model.parameters():
            p.requires_grad = False
        print("✅ VL Model frozen. Sensor Encoder and Diffusion Action Expert are trainable.")

    def _load_qwen_with_fallback(self, vl_model_name):
        """Reuse from base class"""
        return super(QwenVLAWithSensorDiffusion, self)._load_qwen_with_fallback(vl_model_name)

    def forward(self,
                text_inputs,
                image_inputs,
                actions=None,  # Ground truth actions for training
                sensor_data=None,
                cache_keys=None,
                cache: bool = True):
        """
        Forward pass for training/inference

        Training mode (actions provided):
            - Add noise to actions at random timestep
            - Predict noise
            - Return loss

        Inference mode (actions=None):
            - Sample actions via diffusion

        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths
            actions: (B, H, A) - ground truth actions (training only)
            sensor_data: (B, T, C) - sensor data
            cache_keys: Cache keys for VL features
            cache: Whether to use caching

        Returns:
            Training: eps_pred, eps_target, timesteps
            Inference: sampled_actions
        """
        device = next(self.parameters()).device
        use_cache = bool(cache and self.cache_enabled)

        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # Process VL features (with caching) - same as base model
        vl_tokens = self._encode_vision_features(text_inputs, image_inputs, cache_keys, use_cache, device)

        # Process Sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)

        # Training mode: predict noise
        if actions is not None and self.training:
            B, H, A = actions.shape
            actions = actions.to(device=device, dtype=vl_tokens.dtype)

            # Sample random timesteps
            timesteps = torch.randint(0, self.action_expert.timesteps, (B,), device=device).long()

            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.action_expert.diffusion.q_sample(actions, timesteps, noise)

            # Predict noise
            with torch.autocast(device.type, dtype=torch.bfloat16):
                eps_pred = self.action_expert(noisy_actions, timesteps, vl_tokens, sensor_features)

            return eps_pred, noise, timesteps

        # Inference mode: sample actions
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                sampled_actions = self.action_expert.sample(
                    vl_tokens, sensor_features,
                    batch_size=vl_tokens.shape[0],
                    ddim_steps=10  # Use DDIM for faster inference
                )
            return sampled_actions

    def _encode_vision_features(self, text_inputs, image_inputs, cache_keys, use_cache, device):
        """Encode VL features with caching - reuse from base class"""
        # Same logic as base model
        pooled_vl_tokens_dict = {}
        miss_items = []

        if use_cache:
            for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
                cache_path = self._cache_path(key, txt, views)
                if cache_path.exists():
                    pooled = torch.load(cache_path, map_location="cpu")
                    pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                    pooled_vl_tokens_dict[key] = pooled
                else:
                    miss_items.append((txt, views, key))
        else:
            miss_items = list(zip(text_inputs, image_inputs, cache_keys))

        def preprocess_message(args):
            txt, views, key = args
            msg_content = [{"type": "image", "image": v} for v in views if v is not None]
            msg_content.append({"type": "text", "text": txt})
            messages = [{"role": "user", "content": msg_content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            vision_inputs, video_inputs = process_vision_info(messages)
            return key, txt, views, text, vision_inputs, video_inputs

        if miss_items and use_cache and getattr(self, "strict_cache", False):
            missing_keys = [key for _, _, key in miss_items]
            raise FileNotFoundError(f"Missing cached features for keys: {missing_keys}")

        if miss_items:
            with ThreadPoolExecutor(max_workers=24) as executor:
                results = list(executor.map(preprocess_message, miss_items))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, text, vision_inputs, video_inputs in results:
                    if use_cache:
                        cache_path = self._cache_path(key, txt, views)
                        if cache_path.exists():
                            pooled = torch.load(cache_path, map_location="cpu")
                            pooled = pooled.pin_memory().to(device=device, non_blocking=True, dtype=torch.bfloat16)
                            pooled_vl_tokens_dict[key] = pooled
                            continue

                    inputs = self.processor(
                        text=[text],
                        images=vision_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16, non_blocking=True)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    if use_cache:
                        cache_path = self._cache_path(key, txt, views)
                        self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), cache_path)
                        self._enforce_cache_limit()

                    pooled_vl_tokens_dict[key] = pooled.to(dtype=torch.bfloat16)

        pooled_vl_tokens = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(pooled_vl_tokens, dim=0)  # (B, 1, vl_dim)

        return vl_tokens

    def _cache_path(self, key, txt, views):
        """Reuse from base class"""
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu, path):
        """Reuse from base class"""
        tmp = path.with_suffix(".pt.tmp")
        with open(str(path) + ".lock", "w") as lockfile:
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX)
                if path.exists():
                    return
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self):
        """Reuse from base class"""
        limit_gb = getattr(self, "cache_limit_gb", 20.0)
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > limit_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > limit_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)


# =====================================
# 4️⃣ Trainable Version with LoRA (Stage 2)
# =====================================
class Not_freeze_QwenVLAWithSensorDiffusion(nn.Module):
    """
    Trainable VLA model with Sensor + Diffusion Policy
    - Supports LoRA fine-tuning of VL backbone (Stage 2)
    - Load Stage 1 checkpoint (Sensor + Action Expert only)
    - Add LoRA to VL model
    - Train entire model
    """
    def __init__(self,
                 vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                 action_dim=7,
                 horizon=8,
                 hidden_dim=1024,
                 cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
                 finetune_vl="none",  # "none" | "lora" | "full"
                 lora_r=16,
                 lora_alpha=32,
                 lora_dropout=0.05,
                 unfreeze_last_n=2,
                 # Sensor encoder params
                 sensor_enabled=True,
                 sensor_input_channels=1026,
                 sensor_temporal_length=650,
                 sensor_hidden_dim=512,
                 sensor_output_dim=3072,
                 # Fusion params
                 fusion_strategy='concat',
                 # Diffusion params
                 diffusion_timesteps=100,
                 # Stage 2 checkpoint loading
                 stage1_checkpoint=None,
                 ):
        super().__init__()

        print(f"🚀 Loading Trainable Qwen-VL-Sensor-Diffusion Model")
        print(f"   VL Fine-tuning: {finetune_vl}")
        print(f"   Sensor Enabled: {sensor_enabled}")
        print(f"   Fusion Strategy: {fusion_strategy}")
        print(f"   Diffusion Timesteps: {diffusion_timesteps}")
        if stage1_checkpoint:
            print(f"   Stage 1 Checkpoint: {stage1_checkpoint}")

        self.sensor_enabled = sensor_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_mode = "on"
        self.cache_limit_gb = 20.0
        self.strict_cache = False
        self.action_dim = action_dim
        self.horizon = horizon
        self.finetune_vl = finetune_vl

        # VL Model
        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = self._load_qwen_with_fallback(vl_model_name)

        # Enable gradient checkpointing to save memory
        if hasattr(self.vl_model, 'gradient_checkpointing_enable'):
            self.vl_model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled for VL model")

        # Sensor Encoder (Trainable)
        if sensor_enabled:
            self.sensor_encoder = SensorEncoder(
                input_channels=sensor_input_channels,
                temporal_length=sensor_temporal_length,
                hidden_dim=sensor_hidden_dim,
                output_dim=sensor_output_dim,
                use_transformer=True,
                num_transformer_layers=2
            ).to(dtype=torch.bfloat16)
        else:
            self.sensor_encoder = None

        # Diffusion Action Expert (Trainable)
        self.action_expert = DiffusionActionExpert(
            vl_dim=self.vl_model.config.hidden_size,
            sensor_dim=sensor_output_dim if sensor_enabled else 0,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            timesteps=diffusion_timesteps,
            fusion_strategy=fusion_strategy if sensor_enabled else 'none'
        ).to(dtype=torch.bfloat16)

        # Load Stage 1 checkpoint if provided
        if stage1_checkpoint:
            self._load_stage1_checkpoint(stage1_checkpoint)

        # Freeze VL model initially
        for p in self.vl_model.parameters():
            p.requires_grad = False

        # Apply fine-tuning strategy
        if finetune_vl == "lora":
            print("💡 Applying LoRA fine-tuning to VL model...")
            self._inject_lora_to_vl(lora_r, lora_alpha, lora_dropout)
            # Must use cache_mode="off" to actually train LoRA
            self.cache_mode = "off"
            print("   ⚠️  Cache OFF: Computing VL forward pass every iteration (required for LoRA training)")
            print("   💡 Tip: Use small batch_size (1-2) + gradient_accumulation for memory efficiency")
        elif finetune_vl == "full":
            print(f"💡 Unfreezing last {unfreeze_last_n} layers...")
            self._selective_unfreeze_vl(unfreeze_last_n)
            self.cache_mode = "off"
            print("   ⚠️  Cache OFF: Computing VL forward pass every iteration")
        else:
            print("🧊 Using frozen VL backbone with cache enabled.")

    def _load_qwen_with_fallback(self, vl_model_name):
        """Load Qwen-VL with attention implementation fallback

        Note: Do not use device_map="auto" when using DDP!
        DDP requires each process to have the full model on its own device.
        """
        dtype_candidates = [torch.bfloat16, torch.float16]
        attn_candidates = ["flash_attention_2", "sdpa"]

        for dtype in dtype_candidates:
            for impl in attn_candidates:
                try:
                    print(f"🧠 Trying attn_implementation={impl} with dtype={dtype}...")
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vl_model_name,
                        torch_dtype=dtype,
                        attn_implementation=impl,
                        low_cpu_mem_usage=True,
                    )
                    print(f"✅ Successfully loaded with {impl} ({dtype})")
                    self.attn_backend = impl
                    self.model_dtype = dtype
                    return model
                except Exception as e:
                    print(f"⚠️ {impl} ({dtype}) failed: {e}")

        for dtype in dtype_candidates:
            try:
                print(f"🧠 Trying default attention with dtype={dtype}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vl_model_name,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                print(f"✅ Successfully loaded with default attention ({dtype})")
                self.attn_backend = "default"
                self.model_dtype = dtype
                return model
            except Exception as e:
                print(f"⚠️ Default ({dtype}) failed: {e}")

        raise RuntimeError("❌ All dtype/attention fallback attempts failed.")

    def _load_stage1_checkpoint(self, checkpoint_path):
        """Load Stage 1 checkpoint (Sensor Encoder + Action Expert only)"""
        print(f"📥 Loading Stage 1 checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load sensor encoder state
        if self.sensor_enabled and 'sensor_encoder' in checkpoint:
            self.sensor_encoder.load_state_dict(checkpoint['sensor_encoder'])
            print("   ✅ Loaded Sensor Encoder from Stage 1")

        # Load action expert state
        if 'action_expert' in checkpoint:
            self.action_expert.load_state_dict(checkpoint['action_expert'])
            print("   ✅ Loaded Diffusion Action Expert from Stage 1")

        print("✅ Stage 1 checkpoint loaded successfully")

    def _inject_lora_to_vl(self, r, alpha, dropout):
        """Apply LoRA to VL model"""
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
        )
        self.vl_model = get_peft_model(self.vl_model, cfg)
        for n, p in self.vl_model.named_parameters():
            p.requires_grad = "lora" in n

        trainable = sum(p.numel() for p in self.vl_model.parameters() if p.requires_grad)
        print(f"   LoRA trainable parameters: {trainable:,}")

    def _selective_unfreeze_vl(self, last_n=2):
        """Unfreeze last N layers of VL model"""
        blocks = None
        for attr in ["model.layers", "transformer.blocks", "layers"]:
            try:
                blocks = eval(f"self.vl_model.{attr}")
                break
            except Exception:
                pass
        if blocks is None:
            for p in self.vl_model.parameters():
                p.requires_grad = True
            return
        for i, blk in enumerate(blocks):
            trainable = i >= (len(blocks) - last_n)
            for p in blk.parameters():
                p.requires_grad = trainable

    def set_cache(self, enabled: bool = True):
        self.cache_mode = "on" if enabled else "off"

    def set_strict_cache(self, enabled: bool = True):
        self.strict_cache = enabled

    def set_cache_limit(self, limit_gb: float):
        try:
            self.cache_limit_gb = float(limit_gb)
        except (TypeError, ValueError):
            raise ValueError("limit_gb must be convertible to float") from None

    def _cache_path(self, key, txt, views):
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu, path):
        tmp = path.with_suffix(".pt.tmp")
        with open(str(path) + ".lock", "w") as lockfile:
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            if not path.exists():
                torch.save(tensor_cpu, tmp)
                os.replace(tmp, path)
            fcntl.flock(lockfile, fcntl.LOCK_UN)

    def _enforce_cache_limit(self):
        limit_gb = getattr(self, "cache_limit_gb", 20.0)
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > limit_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > limit_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)

    def _encode_lazy_cache(self, text_inputs, image_inputs, cache_keys, device):
        pooled_vl_tokens_dict = {}
        miss_items = []

        for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
            path = self._cache_path(key, txt, views)
            if path.exists():
                pooled = torch.load(path, map_location="cpu")
                pooled_vl_tokens_dict[key] = pooled.to(device, dtype=torch.bfloat16)
            else:
                miss_items.append((txt, views, key))

        if miss_items and getattr(self, "strict_cache", False):
            missing_keys = [key for _, _, key in miss_items]
            raise FileNotFoundError(f"Missing cached features for keys: {missing_keys}")

        if miss_items:
            def preprocess(args):
                txt, views, key = args
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                messages = [{"role": "user", "content": msg_content}]
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                vision_inputs, video_inputs = process_vision_info(messages)
                return key, txt, views, text, vision_inputs, video_inputs

            with ThreadPoolExecutor(max_workers=4) as ex:
                results = list(ex.map(preprocess, miss_items))

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for key, txt, views, text, vision_inputs, video_inputs in results:
                    path = self._cache_path(key, txt, views)
                    inputs = self.processor(
                        text=[text],
                        images=vision_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    vl_device = next(self.vl_model.parameters()).device
                    inputs = inputs.to(device=vl_device, dtype=torch.bfloat16)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), path)
                    self._enforce_cache_limit()
                    pooled_vl_tokens_dict[key] = pooled

        ordered = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(ordered, dim=0)
        return vl_tokens

    def forward(self,
                text_inputs,
                image_inputs,
                actions=None,
                sensor_data=None,
                cache_keys=None):
        """
        Forward pass for training/inference

        Training mode (actions provided):
            - Add noise to actions at random timestep
            - Predict noise
            - Return loss

        Inference mode (actions=None):
            - Sample actions via diffusion
        """
        device = next(self.parameters()).device
        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # Process VL features
        if self.cache_mode == "off":
            # Process each sample individually to avoid OOM
            pooled_vl_tokens_list = []

            for txt, views in zip(text_inputs, image_inputs):
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                msg_batch = [{"role": "user", "content": msg_content}]

                text = self.processor.apply_chat_template(msg_batch, tokenize=False, add_generation_prompt=False)
                vision_inputs, video_inputs = process_vision_info(msg_batch)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    inputs = self.processor(
                        text=[text],
                        images=vision_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(device=device, dtype=torch.bfloat16)

                    outputs = self.vl_model(**inputs, output_hidden_states=True, return_dict=True)
                    vl_tokens = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                    pooled = vl_tokens.mean(dim=1)  # (1, hidden_dim)
                    pooled_vl_tokens_list.append(pooled)

            # Concatenate all samples
            pooled_vl_tokens = torch.cat(pooled_vl_tokens_list, dim=0).unsqueeze(1)  # (B, 1, hidden_dim)
        else:
            pooled_vl_tokens = self._encode_lazy_cache(text_inputs, image_inputs, cache_keys, device)

        # Process Sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)

        # Training mode: predict noise
        if actions is not None and self.training:
            B, H, A = actions.shape
            actions = actions.to(device=device, dtype=pooled_vl_tokens.dtype)

            # Sample random timesteps
            timesteps = torch.randint(0, self.action_expert.timesteps, (B,), device=device).long()

            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.action_expert.diffusion.q_sample(actions, timesteps, noise)

            # Predict noise
            with torch.autocast(device.type, dtype=torch.bfloat16):
                eps_pred = self.action_expert(noisy_actions, timesteps, pooled_vl_tokens, sensor_features)

            # Memory cleanup
            del pooled_vl_tokens
            if sensor_features is not None:
                del sensor_features
            torch.cuda.empty_cache()

            return eps_pred, noise, timesteps

        # Inference mode: sample actions
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                sampled_actions = self.action_expert.sample(
                    pooled_vl_tokens, sensor_features,
                    batch_size=pooled_vl_tokens.shape[0],
                    ddim_steps=10  # Use DDIM for faster inference
                )

            # Memory cleanup
            del pooled_vl_tokens
            if sensor_features is not None:
                del sensor_features
            torch.cuda.empty_cache()

            return sampled_actions


__all__ = [
    'DiffusionSchedule',
    'DiffusionActionExpert',
    'QwenVLAWithSensorDiffusion',
    'Not_freeze_QwenVLAWithSensorDiffusion',
]
