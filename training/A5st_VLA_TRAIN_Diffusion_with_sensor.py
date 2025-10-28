"""
Diffusion-based Training Script for QwenVLA with Sensor Integration

Key Differences from regression baseline:
- Model: QwenVLAWithSensorDiffusion (diffusion-based action expert)
- Loss: MSE on predicted noise (eps_pred vs eps_target)
- Training: Sample random timesteps, add noise to actions, predict noise
- Inference: Iterative denoising (DDPM/DDIM)

Usage:
    # Single GPU
    python training/A5st_VLA_TRAIN_Diffusion_with_sensor.py --dataset_dir /path/to/dataset

    # Multi-GPU
    torchrun --nproc_per_node=4 training/A5st_VLA_TRAIN_Diffusion_with_sensor.py --dataset_dir /path/to/dataset
"""

import argparse
import wandb
import io, shutil, threading, queue, time
import os
import sys
import re
import math
import glob
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data import random_split

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision("high")

# Import diffusion model and dataset
from models.model_with_sensor_diffusion import QwenVLAWithSensorDiffusion, Not_freeze_QwenVLAWithSensorDiffusion
from vla_datasets.IntegratedDataset import collate_fn_with_sensor, create_integrated_dataloader

# Import cache builder (from same directory)
import importlib.util
cache_module_path = Path(__file__).parent / "Make_VL_cache.py"
spec = importlib.util.spec_from_file_location("Make_VL_cache", cache_module_path)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
build_vl_cache_distributed_optimized = cache_module.build_vl_cache_distributed_optimized

# ======== I/O & Checkpoint Utils ========
STAGING_DIR = Path("/dev/shm/qwen_vla_stage")
CKPT_DIR = Path("./checkpoints")
STAGING_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def _atomic_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if src != tmp:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def copy_to_local_then_load(src_path: Path, map_location):
    """네트워크 파일을 로컬 스테이징으로 빠르게 복사 후 torch.load"""
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))
    local_copy = STAGING_DIR / src_path.name
    shutil.copy2(src_path, local_copy)
    try:
        return torch.load(local_copy, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(local_copy, map_location=map_location)

class AsyncCheckpointWriter:
    """학습은 그대로 진행, 저장은 백그라운드 스레드가 처리"""
    def __init__(self, max_queue=2, sync_every=0):
        self.q = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.stop = False
        self.sync_every = sync_every
        self.thread.start()

    def _worker(self):
        last_sync = time.time()
        while not self.stop:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            state_dict, final_dst = payload["state"], Path(payload["dst"])
            local_tmp = STAGING_DIR / (final_dst.name + f".{int(time.time())}.pt")
            torch.save(state_dict, local_tmp, _use_new_zipfile_serialization=True)
            if self.sync_every > 0 and (time.time() - last_sync) < self.sync_every:
                continue
            _atomic_move(local_tmp, final_dst)
            last_sync = time.time()

    def submit(self, state_dict, final_dst: Path):
        if self.q.full():
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
        self.q.put({"state": state_dict, "dst": str(final_dst)})

    def close(self):
        self.stop = True
        self.thread.join(timeout=5)

def build_trapezoid_scheduler(
    optimizer,
    total_steps: int,
    *,
    base_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.03,
    hold_ratio: float = 0.02,
):
    """LLM 스타일: Warmup -> Hold -> Cosine Decay"""
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            cos_val = 0.5 * (1.0 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos_val

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)
    prev_lr = base_lr * lr_lambda(0)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched

# ===========================================================
# 초기화
# ===========================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank

# ===========================================================
# Diffusion 학습 루프
# ===========================================================
def Train_Diffusion(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/qwen_vla_diffusion.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
):
    """
    Diffusion-based training loop

    Key differences from regression:
    1. Model returns (eps_pred, eps_target, timesteps) instead of (pred_actions, delta)
    2. Loss: MSE between predicted noise and ground truth noise
    3. No need for z_chunk (initial action estimate)
    """
    loss_fn = nn.MSELoss(reduction='none')  # Per-sample loss for weighting
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-Diffusion-Sensor",
            name=f"train_diffusion_sensor_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_diffusion_sensor_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
                "diffusion_timesteps": model.module.action_expert.timesteps if hasattr(model, 'module') else model.action_expert.timesteps,
            }
        )

    global_step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0
        total_nonsensor_samples = 0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)

            # Handle sensor data
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 🔥 Diffusion forward: returns (eps_pred, eps_target, timesteps)
                eps_pred, eps_target, timesteps = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    actions=gt_actions,  # Ground truth actions
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data if sensor_enabled else None,
                )

                # Compute loss (MSE on noise)
                loss_per_sample = loss_fn(eps_pred, eps_target).mean(dim=[1, 2])  # (B,)

                # Weighted loss based on confidence and sensor availability
                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                if sensor_enabled and has_sensor_mask is not None:
                    # Higher weight for samples with real sensor data
                    sensor_weights = torch.where(has_sensor_mask,
                                                 torch.tensor(sensor_loss_weight, device=device),
                                                 torch.tensor(1.0, device=device))
                    weights = weights * sensor_weights

                    # Track statistics
                    total_sensor_samples += has_sensor_mask.sum().item()
                    total_nonsensor_samples += (~has_sensor_mask).sum().item()

                # Weighted average loss
                loss = (loss_per_sample * weights).mean()
                loss = loss / grad_accum_steps

            # Backward
            loss.backward()

            # Gradient accumulation
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None and sched_on == "step":
                    scheduler.step()

                global_step += 1

            total_loss += loss.item() * grad_accum_steps

            # Logging
            if rank == 0 and step % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss.item() * grad_accum_steps:.4f}",
                    "lr": f"{current_lr:.2e}"
                })

                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps,
                    "train/lr": current_lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

        # Epoch end
        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        avg_loss = total_loss / len(data_loader)

        if rank == 0:
            print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")
            if sensor_enabled:
                print(f"  Sensor samples: {total_sensor_samples}, Non-sensor: {total_nonsensor_samples}")

            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/sensor_samples": total_sensor_samples,
                "epoch/nonsensor_samples": total_nonsensor_samples,
            })

            # Save checkpoint
            ckpt_path = CKPT_DIR / f"diffusion_epoch{epoch+1}.pt"

            # Determine what to save based on training stage
            model_module = model.module if hasattr(model, "module") else model
            training_stage = getattr(Train_Diffusion, '_training_stage', 'stage1')

            if training_stage == 'stage1':
                # Stage 1: Save only Sensor Encoder + Action Expert
                state = {
                    "epoch": epoch + 1,
                    "sensor_encoder": model_module.sensor_encoder.state_dict() if model_module.sensor_enabled else None,
                    "action_expert": model_module.action_expert.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "loss": avg_loss,
                    "training_stage": "stage1",
                }
                print(f"💾 [Stage 1] Saving Sensor + Action Expert only")
            else:
                # Stage 2: Save full model state
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "loss": avg_loss,
                    "training_stage": "stage2",
                }
                print(f"💾 [Stage 2] Saving full model state")

            writer.submit(state, ckpt_path)
            print(f"   Checkpoint saved: {ckpt_path}")

        # Validation (optional)
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_loss = validate_diffusion(model, val_loader, device, sensor_enabled, sensor_loss_weight)
            if rank == 0:
                print(f"[Validation] Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "val/epoch": epoch + 1})

    if rank == 0 and writer:
        writer.close()
        wandb.finish()

def validate_diffusion(model, val_loader, device, sensor_enabled, sensor_loss_weight):
    """Validation for diffusion model"""
    # Keep model in train mode for forward pass (but use no_grad)
    # This is needed because the model checks self.training to decide whether to return training outputs
    was_training = model.training
    model.train()

    loss_fn = nn.MSELoss(reduction='none')
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=(dist.get_rank() != 0)):
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eps_pred, eps_target, timesteps = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    actions=gt_actions,
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data,
                )

                loss_per_sample = loss_fn(eps_pred, eps_target).mean(dim=[1, 2])
                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                if sensor_enabled and has_sensor_mask is not None:
                    sensor_weights = torch.where(has_sensor_mask,
                                                 torch.tensor(sensor_loss_weight, device=device),
                                                 torch.tensor(1.0, device=device))
                    weights = weights * sensor_weights

                loss = (loss_per_sample * weights).mean()
                total_loss += loss.item()

    # Restore original training state
    if was_training:
        model.train()
    else:
        model.eval()

    return total_loss / len(val_loader)

# ===========================================================
# Main
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description='Train Diffusion VLA with Sensor - 2 Stage Training')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sensor_loss_weight', type=float, default=2.0)
    parser.add_argument('--diffusion_timesteps', type=int, default=100, help='Number of diffusion steps')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # 2-Stage Training Arguments
    parser.add_argument('--training-stage', type=str, choices=['stage1', 'stage2'], default='stage1',
                        help='Training stage: stage1 (Sensor+Action only, VL frozen) or stage2 (Full model with LoRA)')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                        help='Path to Stage 1 checkpoint (required for stage2)')

    # LoRA Arguments (for Stage 2)
    parser.add_argument('--finetune-vl', choices=['none', 'lora', 'full'], default='lora',
                        help='VL fine-tuning mode for Stage 2')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--vl-lr', type=float, default=1e-5, help='Learning rate for VL (Stage 2)')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Validate Stage 2 requirements
    if args.training_stage == 'stage2' and not args.stage1_checkpoint:
        raise ValueError("❌ Stage 2 training requires --stage1-checkpoint")

    if rank == 0:
        print(f"🚀 Training Diffusion VLA with Sensor Integration")
        print(f"   Training Stage: {args.training_stage.upper()}")
        print(f"   World Size: {world_size}")
        print(f"   Diffusion Timesteps: {args.diffusion_timesteps}")
        print(f"   Dataset: {args.dataset_dir}")
        if args.training_stage == 'stage2':
            print(f"   Stage 1 Checkpoint: {args.stage1_checkpoint}")
            print(f"   VL Fine-tuning: {args.finetune_vl}")

    # Note: VL cache will be built on-the-fly during training
    # For faster training, you can pre-build cache using Make_VL_cache.py

    # Load model based on training stage
    if args.training_stage == 'stage1':
        # Stage 1: Frozen VL, train Sensor + Action Expert only
        if rank == 0:
            print("📍 Stage 1: Training Sensor Encoder + Diffusion Action Expert (VL frozen)")
        model = QwenVLAWithSensorDiffusion(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=True,
            fusion_strategy='concat',
            diffusion_timesteps=args.diffusion_timesteps,
        ).to(device)

    else:  # stage2
        # Stage 2: Load Stage 1 checkpoint, add LoRA, train entire model
        if rank == 0:
            print("📍 Stage 2: Training Full Model with LoRA")
        model = Not_freeze_QwenVLAWithSensorDiffusion(
            vl_model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=True,
            fusion_strategy='concat',
            diffusion_timesteps=args.diffusion_timesteps,
            finetune_vl=args.finetune_vl,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            stage1_checkpoint=args.stage1_checkpoint,
        )

        # Move entire model to device (including VL model with LoRA)
        model = model.to(device)

        # Explicitly move VL model to ensure PEFT wrapped model is on correct device
        if hasattr(model, 'vl_model'):
            model.vl_model = model.vl_model.to(device)

        if rank == 0:
            print(f"✅ Model moved to device: {device}")

    # DDP with find_unused_parameters for Stage 2 (when using cache with LoRA)
    find_unused = (args.training_stage == 'stage2' and args.finetune_vl != 'none')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=find_unused,
                gradient_as_bucket_view=False)  # Required for gradient checkpointing

    # Set static graph for gradient checkpointing compatibility
    if args.training_stage == 'stage2':
        model._set_static_graph()

    # Create dataloader
    train_loader, val_loader = create_integrated_dataloader(
        dataset_root=Path(args.dataset_dir),
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        val_split=args.val_split,
        distributed=True,
        rank=rank,
        world_size=world_size,
    )

    # Optimizer - Different learning rates for VL and other components in Stage 2
    if args.training_stage == 'stage2' and args.finetune_vl != 'none':
        # Separate parameter groups for VL (LoRA) and others
        vl_params = []
        other_params = []

        model_module = model.module if hasattr(model, 'module') else model

        for name, param in model.named_parameters():
            if param.requires_grad:
                # LoRA parameters or VL model parameters
                if 'vl_model' in name or 'lora' in name.lower():
                    vl_params.append(param)
                else:
                    other_params.append(param)

        if rank == 0:
            print(f"🎯 Optimizer config:")
            print(f"   VL params: {len(vl_params)} (lr={args.vl_lr})")
            print(f"   Other params: {len(other_params)} (lr={args.lr})")

        optimizer = torch.optim.AdamW([
            {'params': vl_params, 'lr': args.vl_lr, 'weight_decay': 0.01},
            {'params': other_params, 'lr': args.lr, 'weight_decay': 0.01},
        ], betas=(0.9, 0.95))
    else:
        # Stage 1: Single learning rate
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = build_trapezoid_scheduler(
        optimizer,
        total_steps=total_steps,
        base_lr=args.lr,
        min_lr=1e-6,
        warmup_ratio=0.03,
        hold_ratio=0.02,
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = copy_to_local_then_load(Path(args.resume), map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)

    # Store training stage for checkpoint saving
    Train_Diffusion._training_stage = args.training_stage

    # Train
    Train_Diffusion(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        grad_accum_steps=args.grad_accum,
        device=device,
        scheduler=scheduler,
        sched_on="step",
        val_loader=val_loader,
        start_epoch=start_epoch,
        sensor_enabled=True,
        sensor_loss_weight=args.sensor_loss_weight,
    )

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
