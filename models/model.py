import os
import hashlib
import fcntl
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenActionExpert(nn.Module):
    """Predicts action sequences conditioned on VL embeddings."""

    def __init__(
        self,
        vl_dim: int = 3072,
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 1024,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon

        self.context_proj = nn.Sequential(
            nn.Linear(vl_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        vl_tokens: torch.Tensor,
        z_chunk: torch.Tensor,
        _: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if vl_tokens.ndim != 3:
            raise ValueError(f"vl_tokens must be 3D (B, 1, D); got {vl_tokens.shape}")
        if z_chunk.ndim != 3:
            raise ValueError(f"z_chunk must be 3D (B, H, A); got {z_chunk.shape}")

        _, horizon, _ = z_chunk.shape
        if horizon != self.horizon:
            raise ValueError(
                f"z_chunk horizon mismatch: expected {self.horizon}, got {horizon}"
            )

        vl_context = vl_tokens.squeeze(1)
        vl_context = self.context_proj(vl_context)
        vl_context = vl_context.unsqueeze(1).expand(-1, horizon, -1)

        x = torch.cat([vl_context, z_chunk], dim=-1)
        x = self.input_proj(x)
        x = self.temporal_encoder(x)
        delta = self.delta_head(x)
        pred = z_chunk + delta
        return pred, delta


class _QwenVisionLanguageMixin:
    cache_limit_gb: float = 20.0

    def set_cache(self, enabled: bool = True) -> None:
        self.cache_enabled = bool(enabled)

    def set_strict_cache(self, strict: bool = False) -> None:
        self.strict_cache = bool(strict)

    def set_cache_limit(self, max_gb: float) -> None:
        self.cache_limit_gb = float(max(0.0, max_gb))

    def _cache_path(self, key: str, txt: str, views: List[Optional[str]]) -> Path:
        vlist = [v for v in views if v]
        raw = key + "||" + txt + "||" + "|".join(vlist)
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{h}.pt"

    @staticmethod
    def _atomic_save(tensor_cpu: torch.Tensor, path: Path) -> None:
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

    def _enforce_cache_limit(self) -> None:
        if self.cache_limit_gb <= 0:
            return
        total_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        limit = self.cache_limit_gb * (1024 ** 3)
        if total_bytes <= limit:
            return
        files = sorted(self.cache_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
        for file in files:
            if total_bytes <= limit:
                break
            try:
                size = file.stat().st_size
                file.unlink(missing_ok=True)
                total_bytes -= size
            except FileNotFoundError:
                continue

    def _prepare_message(self, txt: str, views: List[Optional[str]]):
        content = [{"type": "image", "image": v} for v in views if v]
        content.append({"type": "text", "text": txt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        vision_inputs, video_inputs = process_vision_info(messages)
        return rendered, vision_inputs, video_inputs

    def encode_vision(
        self,
        text_inputs: Iterable[str],
        image_inputs: Iterable[List[Optional[str]]],
        cache_keys: Optional[Iterable[str]] = None,
        *,
        cache: Optional[bool] = None,
    ) -> torch.Tensor:
        if cache_keys is None:
            cache_keys = [f"idx={i}" for i, _ in enumerate(text_inputs)]
        cache_keys = list(cache_keys)
        text_inputs = list(text_inputs)
        image_inputs = list(image_inputs)

        if not (len(text_inputs) == len(image_inputs) == len(cache_keys)):
            raise ValueError("text_inputs, image_inputs, cache_keys must have same length")

        device = next(self.vl_model.parameters()).device
        use_cache = self.cache_enabled if cache is None else bool(cache)

        pooled_dict: dict[str, torch.Tensor] = {}
        missing: list[tuple[str, List[Optional[str]], str]] = []

        for txt, views, key in zip(text_inputs, image_inputs, cache_keys):
            path = self._cache_path(key, txt, views)
            if use_cache and path.exists():
                pooled = torch.load(path, map_location="cpu")
                pooled_dict[key] = pooled.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            else:
                if use_cache and self.strict_cache:
                    raise FileNotFoundError(
                        f"Cache miss for key={key}. Run cache build or disable strict cache mode."
                    )
                missing.append((txt, views, key))

        if missing:
            payloads = []
            for txt, views, key in missing:
                rendered, vision_inputs, video_inputs = self._prepare_message(txt, views)
                payloads.append((key, txt, views, rendered, vision_inputs, video_inputs))

            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=self.model_dtype
            ):
                for key, txt, views, rendered, vision_inputs, video_inputs in payloads:
                    inputs = self.processor(
                        text=[rendered],
                        images=vision_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(device=device, dtype=self.model_dtype, non_blocking=True)

                    outputs = self.vl_model(
                        **inputs, output_hidden_states=True, return_dict=True
                    )
                    vl_tokens = outputs.hidden_states[-1]
                    pooled = vl_tokens.mean(dim=1, keepdim=True)

                    if use_cache:
                        path = self._cache_path(key, txt, views)
                        self._atomic_save(pooled.detach().to("cpu", dtype=torch.float16), path)
                        self._enforce_cache_limit()

                    pooled_dict[key] = pooled.to(dtype=torch.bfloat16)

        ordered = [pooled_dict[key] for key in cache_keys if key in pooled_dict]
        if not ordered:
            raise RuntimeError("No VL features produced for the provided batch")
        return torch.cat(ordered, dim=0)


class QwenVLAForAction(nn.Module, _QwenVisionLanguageMixin):
    """Frozen Qwen2.5-VL backbone with trainable action expert."""

    def __init__(
        self,
        vl_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 1024,
        cache_dir: str = "./vl_cache",
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True
        self.strict_cache = False
        self.cache_limit_gb = float(self.cache_limit_gb)

        self.processor = AutoProcessor.from_pretrained(vl_model_name)
        self.vl_model = self._load_qwen_with_fallback(vl_model_name)
        for param in self.vl_model.parameters():
            param.requires_grad = False
        self.vl_model.eval()

        self.action_expert = QwenActionExpert(
            vl_dim=self.vl_model.config.hidden_size,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
        ).to(dtype=torch.bfloat16)

    def _load_qwen_with_fallback(self, vl_model_name: str):
        dtype_candidates = [torch.bfloat16, torch.float16]
        attn_candidates = ["flash_attention_2", "sdpa"]

        for dtype in dtype_candidates:
            for impl in attn_candidates:
                try:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        vl_model_name,
                        torch_dtype=dtype,
                        attn_implementation=impl,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                    self.attn_backend = impl
                    self.model_dtype = dtype
                    return model
                except Exception:
                    continue

        for dtype in dtype_candidates:
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vl_model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
                self.attn_backend = "default"
                self.model_dtype = dtype
                return model
            except Exception:
                continue

        raise RuntimeError("Failed to load Qwen2.5-VL model with available configurations")

    def forward(
        self,
        *,
        text_inputs: Iterable[str],
        image_inputs: Iterable[List[Optional[str]]],
        z_chunk: torch.Tensor,
        cache_keys: Optional[Iterable[str]] = None,
        cache: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vl_tokens = self.encode_vision(text_inputs, image_inputs, cache_keys, cache=cache)
        z_chunk = z_chunk.to(device=vl_tokens.device, dtype=vl_tokens.dtype)
        pred, delta = self.action_expert(vl_tokens, z_chunk, None)
        return pred, delta


class Not_freeze_QwenVLAForAction(QwenVLAForAction):
    """Trainable variant with optional VL fine-tuning via LoRA/full unfreeze."""

    def __init__(
        self,
        vl_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        action_dim: int = 7,
        horizon: int = 8,
        hidden_dim: int = 1024,
        cache_dir: str = "./vl_cache",
        finetune_vl: str = "none",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        unfreeze_last_n: int = 2,
    ) -> None:
        super().__init__(
            vl_model_name=vl_model_name,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            cache_dir=cache_dir,
        )

        self.finetune_vl = finetune_vl
        self.set_cache_limit(self.cache_limit_gb)

        if finetune_vl == "none":
            self.set_cache(True)
        elif finetune_vl == "lora":
            self._inject_lora_to_vl(lora_r, lora_alpha, lora_dropout)
            self.set_cache(False)
        elif finetune_vl == "full":
            self._selective_unfreeze_vl(unfreeze_last_n)
            self.set_cache(False)
        else:
            raise ValueError("finetune_vl must be one of ['none', 'lora', 'full']")

    def _inject_lora_to_vl(self, r: int, alpha: int, dropout: float) -> None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        self.vl_model = get_peft_model(self.vl_model, config)
        for name, param in self.vl_model.named_parameters():
            param.requires_grad = "lora" in name
        self.vl_model.train()

    def _selective_unfreeze_vl(self, last_n: int = 2) -> None:
        layers = None
        for attr in ["model.layers", "transformer.blocks", "layers"]:
            try:
                layers = eval(f"self.vl_model.{attr}")
                break
            except Exception:
                continue
        if layers is None:
            for param in self.vl_model.parameters():
                param.requires_grad = True
            return

        total = len(layers)
        for idx, block in enumerate(layers):
            trainable = idx >= total - last_n
            for param in block.parameters():
                param.requires_grad = trainable

        self.vl_model.train()

