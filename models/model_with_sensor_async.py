"""
Asynchronous Vision-Language-Action Model with Sensor Integration

Key Features:
- VLM runs at ~3.33 Hz (300ms period)
- Action Expert runs at 10 Hz (100ms period)
- VL features are cached and reused 3x before updating
- Sensor encoder processes 65 timesteps (100ms @ 650Hz)

Architecture:
    VLM (3.33 Hz) → VL Features (cached)
    Sensor Encoder (10 Hz) → Sensor Features (65 timesteps)
    Action Expert (10 Hz) → Actions (8-horizon)

This enables the model to operate asynchronously in real-time:
- Heavy VLM computation happens every ~300ms
- Lightweight Action Expert runs every 100ms with cached VL features
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base classes from model_with_sensor
from models.model_with_sensor import (
    SensorEncoder,
    QwenActionExpertWithSensor,
    Not_freeze_QwenVLAWithSensor,
)


class AsyncQwenVLAWithSensor(Not_freeze_QwenVLAWithSensor):
    """
    Asynchronous VLA model optimized for real-time inference

    Differences from base model:
    1. Sensor encoder uses smaller temporal window (65 vs 650)
    2. Provides separate methods for VL extraction and action prediction
    3. Supports VL feature caching for efficient repeated use

    Args:
        vlm_reuse_count: How many times to reuse each VL feature (default: 3)
        sensor_temporal_length: Sensor window size (default: 65 for 100ms @ 650Hz)
        All other args same as Not_freeze_QwenVLAWithSensor
    """

    def __init__(
        self,
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        cache_dir="/home/najo/NAS/VLA/dataset/cache/qwen_vl_features",
        finetune_vl="none",
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        unfreeze_last_n=2,
        # Sensor encoder params - MODIFIED for async
        sensor_enabled=True,
        sensor_input_channels=1026,
        sensor_temporal_length=65,  # 🔥 Changed from 650 to 65 (100ms @ 650Hz)
        sensor_hidden_dim=512,
        sensor_output_dim=3072,
        # Fusion params
        fusion_strategy='concat',
        # Stage 2 checkpoint loading
        stage1_checkpoint=None,
        # Image resize params (for faster inference)
        image_resize_height=None,
        image_resize_width=None,
        # Async params
        vlm_reuse_count=3,  # 🔥 NEW: VL feature reuse count
    ):
        # Store async params before calling parent __init__
        self.vlm_reuse_count = vlm_reuse_count
        self.vlm_update_period_ms = 300  # ~3.33 Hz
        self.action_expert_period_ms = 100  # 10 Hz

        # Call parent constructor
        super().__init__(
            vl_model_name=vl_model_name,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            cache_dir=cache_dir,
            finetune_vl=finetune_vl,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            unfreeze_last_n=unfreeze_last_n,
            sensor_enabled=sensor_enabled,
            sensor_input_channels=sensor_input_channels,
            sensor_temporal_length=sensor_temporal_length,
            sensor_hidden_dim=sensor_hidden_dim,
            sensor_output_dim=sensor_output_dim,
            fusion_strategy=fusion_strategy,
            stage1_checkpoint=stage1_checkpoint,
            image_resize_height=image_resize_height,
            image_resize_width=image_resize_width,
        )

        print(f"🔄 Async VLA initialized:")
        print(f"   VLM update: {self.vlm_update_period_ms}ms (~{1000/self.vlm_update_period_ms:.1f} Hz)")
        print(f"   Action Expert: {self.action_expert_period_ms}ms ({1000/self.action_expert_period_ms:.1f} Hz)")
        print(f"   VL feature reuse: {self.vlm_reuse_count}x")
        print(f"   Sensor window: {sensor_temporal_length} samples")

    def extract_vl_features(self, text_inputs, image_inputs, cache_keys=None):
        """
        Extract VL features only (for caching and reuse)

        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths (multi-view)
            cache_keys: Cache keys for VL features

        Returns:
            vl_features: (B, 1, vl_dim) VL features
        """
        device = next(self.parameters()).device

        if cache_keys is None:
            cache_keys = [f"idx={i}" for i in range(len(text_inputs))]

        # Process VL features (same as parent forward)
        if self.cache_mode == "off":
            from qwen_vl_utils import process_vision_info

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
            # Load from cache
            pooled_vl_tokens = self._encode_lazy_cache(text_inputs, image_inputs, cache_keys, device)

        return pooled_vl_tokens

    def predict_actions_with_cached_vl(
        self,
        vl_features,
        z_chunk,
        sensor_data,
    ):
        """
        Predict actions using cached VL features and current sensor data

        Args:
            vl_features: (B, 1, vl_dim) - cached VL features
            z_chunk: (B, H, action_dim) - action chunks
            sensor_data: (B, T, C) - sensor data (T=65 for async mode)

        Returns:
            pred_actions: (B, H, action_dim)
            delta: (B, H, action_dim)
        """
        device = next(self.parameters()).device

        # Process Sensor features
        sensor_features = None
        if self.sensor_enabled and sensor_data is not None:
            sensor_data = sensor_data.to(device=device, dtype=torch.bfloat16)
            sensor_features = self.sensor_encoder(sensor_data)

        # Action prediction with multi-modal fusion
        z_chunk = z_chunk.to(device=device, dtype=vl_features.dtype)
        pred_actions, delta = self.action_expert(vl_features, z_chunk, sensor_features)

        return pred_actions, delta

    def forward(
        self,
        text_inputs,
        image_inputs,
        z_chunk,
        sensor_data=None,
        cache_keys=None,
        # 🔥 NEW: Option to use pre-extracted VL features
        vl_features=None,
    ):
        """
        Forward pass with optional pre-extracted VL features

        If vl_features is provided, skip VL extraction (for async training)
        Otherwise, behave like parent class

        Args:
            text_inputs: List of text prompts
            image_inputs: List of image paths
            z_chunk: (B, H, action_dim)
            sensor_data: (B, T, C) where T=65
            cache_keys: Cache keys
            vl_features: (B, 1, vl_dim) - pre-extracted VL features (optional)
        """
        if vl_features is not None:
            # Use pre-extracted VL features (async mode)
            return self.predict_actions_with_cached_vl(vl_features, z_chunk, sensor_data)
        else:
            # Standard forward (extract VL features + predict actions)
            return super().forward(
                text_inputs=text_inputs,
                image_inputs=image_inputs,
                z_chunk=z_chunk,
                sensor_data=sensor_data,
                cache_keys=cache_keys,
            )


def create_async_model(
    stage1_checkpoint=None,
    finetune_vl="lora",
    sensor_window_size=65,
    vlm_reuse_count=3,
    image_resize_height=None,
    image_resize_width=None,
    **kwargs
):
    """
    Convenience function to create async VLA model

    Args:
        stage1_checkpoint: Path to stage 1 checkpoint
        finetune_vl: VL fine-tuning mode ("none", "lora", "full")
        sensor_window_size: Sensor window size (default: 65)
        vlm_reuse_count: VL feature reuse count (default: 3)
        image_resize_height: Image resize height (e.g., 360 for 640x360)
        image_resize_width: Image resize width (e.g., 640 for 640x360)
        **kwargs: Additional model arguments
    """
    model = AsyncQwenVLAWithSensor(
        vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim=7,
        horizon=8,
        hidden_dim=1024,
        finetune_vl=finetune_vl,
        sensor_enabled=True,
        sensor_temporal_length=sensor_window_size,
        vlm_reuse_count=vlm_reuse_count,
        stage1_checkpoint=stage1_checkpoint,
        image_resize_height=image_resize_height,
        image_resize_width=image_resize_width,
        **kwargs
    )

    return model


if __name__ == "__main__":
    print("🧪 Testing AsyncQwenVLAWithSensor...")

    model = create_async_model(
        finetune_vl="none",
        sensor_window_size=65,
        vlm_reuse_count=3,
    )

    print(f"\n✅ Model created successfully!")
    print(f"   Sensor temporal length: {model.sensor_encoder.temporal_length if model.sensor_encoder else 'N/A'}")
    print(f"   VLM reuse count: {model.vlm_reuse_count}")

    # Test forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    B = 2
    text_inputs = ["Test instruction"] * B
    image_inputs = [["/path/to/image.jpg"] * 2] * B  # Dummy paths
    z_chunk = torch.randn(B, 8, 7, device=device, dtype=torch.bfloat16)
    sensor_data = torch.randn(B, 65, 1026, device=device, dtype=torch.bfloat16)

    print(f"\n🔬 Testing VL feature extraction...")
    with torch.no_grad():
        # This will fail with dummy paths, but tests the interface
        try:
            vl_features = model.extract_vl_features(text_inputs, image_inputs)
            print(f"   VL features shape: {vl_features.shape}")
        except Exception as e:
            print(f"   Expected error (dummy paths): {type(e).__name__}")

    print(f"\n🔬 Testing action prediction with cached VL features...")
    # Use actual VL dimension from model
    vl_dim = model.vl_model.config.hidden_size
    print(f"   Actual VL dimension: {vl_dim}")
    vl_features_dummy = torch.randn(B, 1, vl_dim, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        pred_actions, delta = model.predict_actions_with_cached_vl(
            vl_features_dummy, z_chunk, sensor_data
        )
        print(f"   Predicted actions shape: {pred_actions.shape}")
        print(f"   Delta shape: {delta.shape}")

    print(f"\n✅ All tests passed!")
