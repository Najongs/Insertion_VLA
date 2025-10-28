"""
Vision-Language-Action Model with Sensor Encoder Integration
Combines Qwen2.5-VL-3B with OCT/FPI Sensor Data for Enhanced Robot Control

Key Features:
- SensorEncoder: Processes (650, 1026) OCT/FPI temporal data
- Multi-modal fusion: VL features + Sensor features
- Flexible fusion strategies: concat, cross-attention, gating
- Maintains backward compatibility with existing model.py
"""

import os
from pathlib import Path
import hashlib, fcntl
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model


# =====================================
# 1️⃣ Sensor Encoder Module
# =====================================
class SensorEncoder(nn.Module):
    """
    Encodes OCT/FPI sensor data: (B, T, C) where T=650, C=1026 (1 force + 1025 A-scan)

    Architecture:
    - 1D Convolutional layers for temporal feature extraction
    - Optional Transformer layers for long-range dependencies
    - Projection to match VL feature dimension

    Args:
        input_channels: 1026 (1 force + 1025 A-scan)
        temporal_length: 650 (1 second at 650Hz)
        hidden_dim: Internal hidden dimension
        output_dim: Output feature dimension (default: 3072 to match Qwen VL)
        num_conv_layers: Number of 1D conv layers
        use_transformer: Whether to use transformer layers
        num_transformer_layers: Number of transformer layers
        dropout: Dropout rate
    """
    def __init__(self,
                 input_channels=1026,
                 temporal_length=650,
                 hidden_dim=512,
                 output_dim=3072,
                 num_conv_layers=4,
                 use_transformer=True,
                 num_transformer_layers=2,
                 nhead=8,
                 dropout=0.1,
                 gradient_checkpointing=False,
                 interpolation_mode='linear'):
        super().__init__()

        self.input_channels = input_channels
        self.temporal_length = temporal_length
        self.output_dim = output_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.interpolation_mode = interpolation_mode  # 'linear', 'nearest', or 'cubic'

        # 🔹 1D Convolutional backbone for temporal feature extraction
        conv_layers = []
        current_channels = input_channels
        current_length = temporal_length

        for i in range(num_conv_layers):
            out_channels = hidden_dim if i == 0 else hidden_dim * 2 if i == 1 else hidden_dim * 2
            conv_layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_channels = out_channels
            current_length = (current_length + 1) // 2  # Track length reduction

        self.conv_backbone = nn.Sequential(*conv_layers)
        self.final_temporal_length = current_length

        # 🔹 Optional Transformer layers for long-range temporal dependencies
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=current_channels,
                nhead=nhead,
                dim_feedforward=current_channels * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 🔹 Temporal pooling and projection to output dimension
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.projection = nn.Sequential(
            nn.Linear(current_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # 🔹 Optional: Separate encoders for force and A-scan (advanced)
        # self.force_encoder = nn.Linear(1, hidden_dim)
        # self.ascan_encoder = nn.Conv1d(1025, hidden_dim, kernel_size=1)

        print(f"✅ SensorEncoder initialized:")
        print(f"   Input: (B, {temporal_length}, {input_channels})")
        print(f"   Conv layers: {num_conv_layers} → Final temporal length: {self.final_temporal_length}")
        print(f"   Transformer: {use_transformer} ({num_transformer_layers} layers)")
        print(f"   Output: (B, {output_dim})")

    def forward(self, sensor_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_data: (B, T, C) where T can be variable, C=1026
        Returns:
            sensor_features: (B, output_dim)

        Note: Supports variable-length inputs through automatic interpolation
        """
        B, T, C = sensor_data.shape

        # Validate channel dimension
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {C}")

        # Handle variable temporal length with automatic interpolation
        if T != self.temporal_length:
            # Interpolate to target length using configurable interpolation mode
            # (B, T, C) → (B, C, T) → interpolate → (B, C, T_target) → (B, T_target, C)
            x = sensor_data.transpose(1, 2)  # (B, C, T)

            # Select interpolation mode based on configuration
            if self.interpolation_mode == 'cubic':
                # Cubic interpolation requires at least 4 points
                if T >= 4:
                    x = F.interpolate(x, size=self.temporal_length, mode='cubic', align_corners=False)
                else:
                    x = F.interpolate(x, size=self.temporal_length, mode='linear', align_corners=False)
            else:
                x = F.interpolate(x, size=self.temporal_length, mode=self.interpolation_mode, align_corners=False if self.interpolation_mode == 'linear' else None)

            sensor_data = x.transpose(1, 2)  # (B, T_target, C)

        # (B, T, C) → (B, C, T) for Conv1d
        x = sensor_data.transpose(1, 2)

        # 1D Convolutional feature extraction with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(self.conv_backbone, x, use_reentrant=False)
        else:
            x = self.conv_backbone(x)  # (B, hidden_dim*2, T')

        # Optional Transformer for temporal modeling
        if self.use_transformer:
            x = x.transpose(1, 2)  # (B, T', hidden_dim*2)
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)  # (B, T', hidden_dim*2)
            x = x.transpose(1, 2)  # (B, hidden_dim*2, T')

        # Temporal pooling
        x = self.temporal_pool(x).squeeze(-1)  # (B, hidden_dim*2)

        # Project to output dimension
        sensor_features = self.projection(x)  # (B, output_dim)

        return sensor_features


# =====================================
# 2️⃣ Enhanced Action Expert with Sensor Fusion
# =====================================
class QwenActionExpertWithSensor(nn.Module):
    """
    Enhanced Action Expert that fuses VL features with Sensor features

    Fusion Strategies:
    - 'concat': Concatenate VL and sensor features
    - 'cross_attention': Cross-attention between VL and sensor
    - 'gated': Gated fusion with learned gates
    - 'none': Use only VL features (backward compatible)

    Args:
        vl_dim: Vision-Language feature dimension (3072)
        sensor_dim: Sensor feature dimension (3072)
        action_dim: Action dimension (7)
        horizon: Action prediction horizon (8)
        fusion_strategy: One of ['concat', 'cross_attention', 'gated', 'none']
    """
    def __init__(self,
                 vl_dim=3072,
                 sensor_dim=3072,
                 action_dim=7,
                 horizon=8,
                 hidden_dim=1024,
                 nhead=8,
                 num_layers=4,
                 fusion_strategy='concat',
                 dropout=0.1):
        super().__init__()

        self.horizon = horizon
        self.fusion_strategy = fusion_strategy

        # 🔹 Fusion layer based on strategy
        if fusion_strategy == 'concat':
            fused_dim = vl_dim + sensor_dim
            self.fusion_proj = nn.Linear(fused_dim, hidden_dim)
        elif fusion_strategy == 'cross_attention':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'gated':
            self.vl_proj = nn.Linear(vl_dim, hidden_dim)
            self.sensor_proj = nn.Linear(sensor_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_strategy == 'none':
            self.fusion_proj = nn.Linear(vl_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # 🔹 Positional embeddings for action chunks
        self.pos_embed = nn.Parameter(torch.randn(1, horizon, hidden_dim))

        # 🔹 Temporal decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 🔹 Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

        print(f"✅ QwenActionExpertWithSensor initialized with fusion strategy: {fusion_strategy}")

    def forward(self,
                vl_tokens: torch.Tensor,
                z_chunk: torch.Tensor,
                sensor_features: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vl_tokens: (B, 1, vl_dim) or (B, seq_len, vl_dim)
            z_chunk: (B, H, action_dim) - action chunks for temporal decoding
            sensor_features: (B, sensor_dim) - optional sensor features
        Returns:
            pred_actions: (B, H, action_dim)
            delta: (B, H, action_dim)
        """
        B, H, A = z_chunk.shape

        # 🔹 Fuse VL and Sensor features based on strategy
        if self.fusion_strategy == 'concat' and sensor_features is not None:
            # Concatenate VL and sensor features
            vl_pooled = vl_tokens.mean(dim=1)  # (B, vl_dim)
            fused = torch.cat([vl_pooled, sensor_features], dim=-1)  # (B, vl_dim + sensor_dim)
            cond = self.fusion_proj(fused).unsqueeze(1)  # (B, 1, hidden_dim)

        elif self.fusion_strategy == 'cross_attention' and sensor_features is not None:
            # Cross-attention between VL and sensor
            vl_feat = self.vl_proj(vl_tokens)  # (B, seq_len, hidden_dim)
            sensor_feat = self.sensor_proj(sensor_features).unsqueeze(1)  # (B, 1, hidden_dim)
            attn_out, _ = self.cross_attn(sensor_feat, vl_feat, vl_feat)  # (B, 1, hidden_dim)
            cond = self.fusion_proj(attn_out)  # (B, 1, hidden_dim)

        elif self.fusion_strategy == 'gated' and sensor_features is not None:
            # Gated fusion
            vl_pooled = vl_tokens.mean(dim=1)  # (B, vl_dim)
            vl_feat = self.vl_proj(vl_pooled)  # (B, hidden_dim)
            sensor_feat = self.sensor_proj(sensor_features)  # (B, hidden_dim)
            gate = self.gate(torch.cat([vl_feat, sensor_feat], dim=-1))  # (B, hidden_dim)
            fused = gate * vl_feat + (1 - gate) * sensor_feat  # (B, hidden_dim)
            cond = self.fusion_proj(fused).unsqueeze(1)  # (B, 1, hidden_dim)

        else:
            # No fusion or sensor not provided - use only VL features
            vl_pooled = vl_tokens.mean(dim=1, keepdim=True)  # (B, 1, vl_dim)
            cond = self.fusion_proj(vl_pooled)  # (B, 1, hidden_dim)

        # 🔹 Temporal decoding with positional embeddings
        tgt = self.pos_embed.repeat(B, 1, 1)  # (B, H, hidden_dim)
        decoded = self.temporal_decoder(tgt, cond)  # (B, H, hidden_dim)

        # 🔹 Predict action deltas
        delta = self.output_head(decoded)  # (B, H, action_dim)
        pred_actions = z_chunk + delta

        return pred_actions, delta


# =====================================
# 3️⃣ Full Vision-Language-Action-Sensor Model (Frozen VL)
# =====================================
class QwenVLAWithSensor(nn.Module):
    """
    Full VLA model with Sensor Encoder integration
    - Frozen Qwen-VL backbone with caching
    - Trainable Sensor Encoder
    - Trainable Action Expert with multi-modal fusion
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
                 ):
        super().__init__()

        print(f"🚀 Loading Qwen-VL-Sensor Model")
        print(f"   Sensor Enabled: {sensor_enabled}")
        print(f"   Fusion Strategy: {fusion_strategy}")

        self.sensor_enabled = sensor_enabled
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True
        self.cache_limit_gb = 20.0
        self.strict_cache = False
        self.action_dim = action_dim
        self.horizon = horizon

        # 🔹 VL Model (Frozen)
        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = self._load_qwen_with_fallback(vl_model_name)

        # 🔹 Sensor Encoder (Trainable)
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

        # 🔹 Action Expert (Trainable)
        self.action_expert = QwenActionExpertWithSensor(
            vl_dim=self.vl_model.config.hidden_size,
            sensor_dim=sensor_output_dim if sensor_enabled else 0,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            fusion_strategy=fusion_strategy if sensor_enabled else 'none'
        ).to(dtype=torch.bfloat16, device="cuda")

        # Freeze VL model
        print("🧊 Freezing Qwen-VL parameters...")
        for p in self.vl_model.parameters():
            p.requires_grad = False
        print("✅ VL Model frozen. Sensor Encoder and Action Expert are trainable.")

    def _load_qwen_with_fallback(self, vl_model_name):
        """Load Qwen-VL with attention implementation fallback"""
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
                        device_map="cuda",
                        low_cpu_mem_usage=True,
                    )
                    print(f"✅ Successfully loaded with {impl} ({dtype})")
                    self.attn_backend = impl
                    self.model_dtype = dtype
                    return model
                except Exception as e:
                    print(f"⚠️ {impl} ({dtype}) failed: {e}")

        # Final fallback: default attention
        for dtype in dtype_candidates:
            try:
                print(f"🧠 Trying default attention with dtype={dtype}...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vl_model_name,
                    torch_dtype=dtype,
                    device_map="cuda",
                    low_cpu_mem_usage=True,
                )
                print(f"✅ Successfully loaded with default attention ({dtype})")
                self.attn_backend = "default"
                self.model_dtype = dtype
                return model
            except Exception as e:
                print(f"⚠️ Default ({dtype}) failed: {e}")

        raise RuntimeError("❌ All dtype/attention fallback attempts failed.")

    def set_cache(self, enabled: bool = True):
        self.cache_enabled = enabled

    def set_strict_cache(self, enabled: bool = True):
        self.strict_cache = enabled

    def set_cache_limit(self, limit_gb: float):
        try:
            self.cache_limit_gb = float(limit_gb)
        except (TypeError, ValueError):
            raise ValueError("limit_gb must be convertible to float") from None

    def _cache_path(self, key: str, txt: str, views: list[str | None]) -> Path:
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
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
        limit_gb = getattr(self, "cache_limit_gb", 20.0)
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        if total_bytes > limit_gb * (1024 ** 3):
            all_files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
            while total_bytes > limit_gb * (1024 ** 3) and all_files:
                f = all_files.pop(0)
                total_bytes -= f.stat().st_size
                f.unlink(missing_ok=True)
            print(f"⚠️ Cache limit exceeded. Trimmed to {limit_gb}GB.")

    def encode_vision(self, text_inputs, image_inputs, cache_keys, cache: bool = True):
        device = next(self.vl_model.parameters()).device
        horizon = getattr(self, "horizon", 8)
        action_dim = getattr(self, "action_dim", 7)
        dtype = getattr(self, "model_dtype", torch.bfloat16)
        dummy = torch.zeros(len(text_inputs), horizon, action_dim, device=device, dtype=dtype)
        with torch.inference_mode():
            self(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                z_chunk=dummy,
                sensor_data=None,
                cache_keys=cache_keys,
                cache=cache,
            )

    def forward(self,
                text_inputs,
                image_inputs,
                z_chunk,
                sensor_data=None,
                cache_keys=None,
                cache: bool = True):
        """
        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths (multi-view)
            z_chunk: (B, H, action_dim) - action chunks
            sensor_data: (B, T, C) - optional sensor data where T=650, C=1026
            cache_keys: Cache keys for VL features
            cache: Whether to use caching
        Returns:
            pred_actions: (B, H, action_dim)
            delta: (B, H, action_dim)
        """
        device = next(self.parameters()).device
        use_cache = bool(cache and self.cache_enabled)

        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # 🔹 Process VL features (with caching)
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

        # 🔹 Process Sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)  # (B, sensor_dim)

        # 🔹 Action prediction with multi-modal fusion
        z_chunk = z_chunk.to(device=device, dtype=vl_tokens.dtype)
        pred_actions, delta = self.action_expert(vl_tokens, z_chunk, sensor_features)

        return pred_actions, delta


# =====================================
# 4️⃣ Trainable Version with LoRA/Full Fine-tuning
# =====================================
class Not_freeze_QwenVLAWithSensor(nn.Module):
    """
    Trainable VLA model with Sensor integration
    - Supports LoRA fine-tuning of VL backbone
    - Full fine-tuning option
    - Trainable Sensor Encoder and Action Expert
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
                 # Stage 2 checkpoint loading
                 stage1_checkpoint=None,
                 ):
        super().__init__()

        print(f"🚀 Loading Trainable Qwen-VL-Sensor Model")
        print(f"   VL Fine-tuning: {finetune_vl}")
        print(f"   Sensor Enabled: {sensor_enabled}")
        print(f"   Fusion Strategy: {fusion_strategy}")
        if stage1_checkpoint:
            print(f"   Stage 1 Checkpoint: {stage1_checkpoint}")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.finetune_vl = finetune_vl
        self.sensor_enabled = sensor_enabled
        self.cache_mode = "on"
        self.cache_limit_gb = 20.0
        self.strict_cache = False
        self.action_dim = action_dim
        self.horizon = horizon

        # 🔹 VL Model
        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = self._load_qwen_with_fallback(vl_model_name)

        # 🔹 Sensor Encoder (Trainable)
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

        # 🔹 Action Expert (Trainable)
        self.action_expert = QwenActionExpertWithSensor(
            vl_dim=self.vl_model.config.hidden_size,
            sensor_dim=sensor_output_dim if sensor_enabled else 0,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            fusion_strategy=fusion_strategy if sensor_enabled else 'none'
        ).to(dtype=torch.bfloat16)

        # 🔹 Load Stage 1 checkpoint if provided (before LoRA injection)
        if stage1_checkpoint:
            self._load_stage1_checkpoint(stage1_checkpoint)

        # 🔹 Fine-tuning setup
        for p in self.vl_model.parameters():
            p.requires_grad = False

        if finetune_vl == "lora":
            print("💡 Applying LoRA fine-tuning...")
            self._inject_lora_to_vl(lora_r, lora_alpha, lora_dropout)
        elif finetune_vl == "full":
            print(f"💡 Unfreezing last {unfreeze_last_n} layers...")
            self._selective_unfreeze_vl(unfreeze_last_n)
        else:
            print("🧊 Using frozen VL backbone.")

        if finetune_vl in ["lora", "full"]:
            self.cache_mode = "off"
        else:
            self.cache_mode = "on"

    def set_cache(self, enabled: bool = True):
        self.cache_mode = "on" if enabled else "off"

    def set_strict_cache(self, enabled: bool = True):
        self.strict_cache = enabled

    def set_cache_limit(self, limit_gb: float):
        try:
            self.cache_limit_gb = float(limit_gb)
        except (TypeError, ValueError):
            raise ValueError("limit_gb must be convertible to float") from None

    def _load_qwen_with_fallback(self, vl_model_name):
        """Load Qwen-VL with attention implementation fallback"""
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
                        device_map="auto",
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
                    device_map="auto",
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
            print("   ✅ Loaded Regression Action Expert from Stage 1")

        print("✅ Stage 1 checkpoint loaded successfully")

    def _inject_lora_to_vl(self, r, alpha, dropout):
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

    def _cache_path(self, key: str, txt: str, views: list[str | None]) -> Path:
        vlist = [v for v in views if v is not None]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path):
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
            print(f"⚠️ Cache trimmed to {limit_gb}GB.")

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
                    self._enforce_cache_limit(max_gb=20)
                    pooled_vl_tokens_dict[key] = pooled

        ordered = [pooled_vl_tokens_dict[k] for k in cache_keys if k in pooled_vl_tokens_dict]
        vl_tokens = torch.cat(ordered, dim=0)
        return vl_tokens

    def encode_vision(self, text_inputs, image_inputs, cache_keys, cache: bool = True):
        if not cache:
            prev_mode = self.cache_mode
            self.cache_mode = "off"
        else:
            prev_mode = self.cache_mode
            self.cache_mode = "on"

        device = next(self.parameters()).device
        horizon = getattr(self, "horizon", 8)
        action_dim = getattr(self, "action_dim", 7)
        dtype = getattr(self, "model_dtype", torch.bfloat16)
        dummy = torch.zeros(len(text_inputs), horizon, action_dim, device=device, dtype=dtype)
        try:
            with torch.inference_mode():
                self(
                    text_inputs=text_inputs,
                    image_inputs=image_inputs,
                    z_chunk=dummy,
                    sensor_data=None,
                    cache_keys=cache_keys,
                )
        finally:
            self.cache_mode = prev_mode

    def forward(self,
                text_inputs,
                image_inputs,
                z_chunk,
                sensor_data=None,
                cache_keys=None):
        """
        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths (multi-view)
            z_chunk: (B, H, action_dim) - action chunks
            sensor_data: (B, T, C) - optional sensor data where T=650, C=1026
            cache_keys: Cache keys for VL features
        Returns:
            pred_actions: (B, H, action_dim)
            delta: (B, H, action_dim)
        """
        device = next(self.parameters()).device
        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # 🔹 Process VL features
        if self.cache_mode == "off":
            msg_batch = []
            for txt, views in zip(text_inputs, image_inputs):
                msg_content = [{"type": "image", "image": v} for v in views if v is not None]
                msg_content.append({"type": "text", "text": txt})
                msg_batch.append({"role": "user", "content": msg_content})

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
                vl_tokens = outputs.hidden_states[-1]
                pooled_vl_tokens = vl_tokens.mean(dim=1, keepdim=True)
        else:
            pooled_vl_tokens = self._encode_lazy_cache(text_inputs, image_inputs, cache_keys, device)

        # 🔹 Process Sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)

        # 🔹 Action prediction with multi-modal fusion
        z_chunk = z_chunk.to(device=device, dtype=pooled_vl_tokens.dtype)
        pred_actions, delta = self.action_expert(pooled_vl_tokens, z_chunk, sensor_features)

        # Memory cleanup
        del pooled_vl_tokens
        if sensor_features is not None:
            del sensor_features
        torch.cuda.empty_cache()

        return pred_actions, delta


# =====================================
# Backward Compatibility Aliases
# =====================================
# Keep original classes for backward compatibility
from models.model import QwenActionExpert as QwenActionExpert_Original
from models.model import QwenVLAForAction as QwenVLAForAction_Original
from models.model import Not_freeze_QwenVLAForAction as Not_freeze_QwenVLAForAction_Original

__all__ = [
    'SensorEncoder',
    'QwenActionExpertWithSensor',
    'QwenVLAWithSensor',
    'Not_freeze_QwenVLAWithSensor',
    # Backward compatibility
    'QwenActionExpert_Original',
    'QwenVLAForAction_Original',
    'Not_freeze_QwenVLAForAction_Original',
]
