"""
Asynchronous VLA Training Script

Key Differences from standard training:
1. Uses AsyncQwenVLAWithSensor model
2. Sensor window size: 65 samples (100ms @ 650Hz)
3. VL features are reused 3x before updating (simulating 3.33Hz VLM)
4. Training mimics real-time async behavior

Training Strategy:
- Extract VL features once
- Reuse VL features for 3 consecutive action predictions with different sensor windows
- This teaches the model to work with slightly outdated VL features (realistic scenario)
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

# ✅ Import async model
from models.model_with_sensor_async import AsyncQwenVLAWithSensor, create_async_model

# ✅ Use existing dataset with smaller sensor window
from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor, collate_fn_with_sensor

# Import utilities from regular training script
from training.A5st_VLA_TRAIN_VL_Lora_with_sensor import (
    setup_distributed,
    build_trapezoid_scheduler,
    build_rewarm_scheduler,
    AsyncCheckpointWriter,
    copy_to_local_then_load,
    STAGING_DIR,
    CKPT_DIR,
)

# ===========================================================
# Async Training Loop
# ===========================================================
def Train_Async(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/qwen_vla_async.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
    vlm_reuse_count=3,  # 🔥 NEW: VL feature reuse
):
    """
    Async training loop with VL feature reuse

    Strategy:
    - Every vlm_reuse_count batches, extract new VL features
    - Between updates, reuse cached VL features
    - This simulates real-time async behavior where VLM runs slower than Action Expert
    """
    loss_fn = nn.MSELoss()
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-Async",
            name=f"async_train_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_async_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
                "vlm_reuse_count": vlm_reuse_count,
            }
        )

    global_step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0
        total_nonsensor_samples = 0
        vl_feature_cache = None  # 🔥 Cache for VL features
        reuse_counter = 0  # 🔥 Counter for VL feature reuse

        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)

            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 🔥 Extract VL features only when cache is empty or expired
                if vl_feature_cache is None or reuse_counter >= vlm_reuse_count:
                    # 🔥 IMPORTANT: Use cache for training (fast VL extraction)
                    # Set cache mode to "on" for training efficiency
                    if hasattr(model.module, 'cache_mode'):
                        model.module.cache_mode = "on"

                    # Extract fresh VL features (uses cache, fast)
                    vl_features = model.module.extract_vl_features(
                        text_inputs=instructions,
                        image_inputs=image_inputs,
                        cache_keys=batch["cache_keys"],
                    )
                    vl_feature_cache = vl_features
                    reuse_counter = 0

                    if rank == 0 and step % 50 == 0:
                        print(f"\n   🔄 [Step {step}] Extracted new VL features (cached)")

                reuse_counter += 1

                # 🔥 Predict actions using cached VL features
                pred_actions, _ = model.module.predict_actions_with_cached_vl(
                    vl_features=vl_feature_cache,
                    z_chunk=gt_actions,
                    sensor_data=sensor_data if sensor_enabled else None,
                )

                # Weighted loss
                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                if sensor_enabled and has_sensor_mask is not None:
                    sensor_weights = torch.where(has_sensor_mask,
                                                 torch.tensor(sensor_loss_weight, device=device),
                                                 torch.tensor(1.0, device=device))
                    weights = weights * sensor_weights

                weights = weights / weights.mean()
                loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1,2])
                loss = (loss_each * weights).mean() / grad_accum_steps

            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            if sensor_enabled and has_sensor_mask is not None:
                total_sensor_samples += has_sensor_mask.sum().item()
                total_nonsensor_samples += (~has_sensor_mask).sum().item()

            # Gradient step
            if (step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and sched_on == "step":
                    scheduler.step()

                global_step += 1

                # Logging
                lr = optimizer.param_groups[0]["lr"]
                if rank == 0:
                    postfix_dict = {
                        "loss": f"{loss.item() * grad_accum_steps:.6f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{grad_norm:.2f}",
                        "vl_reuse": f"{reuse_counter}/{vlm_reuse_count}",
                    }
                    if sensor_enabled:
                        postfix_dict["sensor"] = f"{total_sensor_samples}/{total_sensor_samples+total_nonsensor_samples}"
                    pbar.set_postfix(postfix_dict)

                    log_dict = {
                        "train/loss_step": loss.item() * grad_accum_steps,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/vl_reuse_counter": reuse_counter,
                        "global_step": global_step
                    }
                    if sensor_enabled:
                        log_dict["train/sensor_samples"] = total_sensor_samples
                        log_dict["train/nonsensor_samples"] = total_nonsensor_samples
                    wandb.log(log_dict)

        # Epoch average
        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)
                    sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None

                    pred_actions, _ = model(
                        text_inputs=batch["instruction"],
                        image_inputs=batch["images"],
                        z_chunk=gt_actions,
                        cache_keys=batch["cache_keys"],
                        sensor_data=sensor_data if sensor_enabled else None,
                    )
                    weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)
                    weights = weights / weights.mean()
                    loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1,2])
                    loss = (loss_each * weights).mean() / grad_accum_steps
                    val_loss_sum += loss.item()
                    val_count += 1
            val_loss = val_loss_sum / max(1, val_count)
            model.train()

        # Checkpoint saving
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            frozen = total_params - trainable

            import psutil, gc
            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            log_dict = {
                "epoch": epoch + 1,
                "train/loss_epoch": avg_loss,
                "val/loss_epoch": val_loss if val_loss else None,
                "params/trainable_M": trainable / 1e6,
                "params/frozen_M": frozen / 1e6,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": optimizer.param_groups[0]["lr"],
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples

            wandb.log(log_dict)

            print(f"\n📊 Epoch {epoch+1} | Train: {avg_loss:.8f} | Val: {val_loss:.8f if val_loss else 'N/A'}")

            # Save checkpoint
            model_module = model.module if hasattr(model, "module") else model
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model_module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
            }

            if not hasattr(Train_Async, "_best_loss"):
                Train_Async._best_loss = float("inf")

            is_best = val_loss is not None and val_loss < Train_Async._best_loss
            if is_best:
                Train_Async._best_loss = val_loss
                best_path = Path(save_path).parent / "qwen_vla_async_best.pt"
                torch.save(ckpt_data, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")
            else:
                tmp_path = Path(save_path).with_suffix(".tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, save_path)
                print(f"💾 [Sync] Latest checkpoint updated: {save_path}")

    if rank == 0 and writer is not None:
        writer.close()

    if rank == 0:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--hold-ratio", type=float, default=0.02)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--sched-on", choices=["step", "epoch"], default="step")

    # LoRA / Fine-tuning
    parser.add_argument("--finetune-vl", choices=["none", "lora", "full"], default="lora")
    parser.add_argument("--vl-lr", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Sensor
    parser.add_argument("--sensor-enabled", action="store_true", default=True)
    parser.add_argument("--sensor-window-size", type=int, default=65,  # 🔥 Changed from 650
                        help="Sensor window size (65 = 100ms @ 650Hz)")
    parser.add_argument("--sensor-lr", type=float, default=5e-4)
    parser.add_argument("--sensor-loss-weight", type=float, default=2.0)

    # Async
    parser.add_argument("--vlm-reuse-count", type=int, default=17,
                        help="How many times to reuse VL features (default: 17 for multi-view 5)")

    # Data
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Checkpoint
    parser.add_argument("--stage1-checkpoint", type=str, default=None)

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"🚀 Async VLA Training")
        print(f"   Sensor window: {args.sensor_window_size} samples")
        print(f"   VLM reuse count: {args.vlm_reuse_count}x")

    # Build dataset
    from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor
    import glob

    datasets = []
    sensor_dataset_dirs = [
        "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
        "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]
    nosensor_dataset_dirs = [
        "/home/najo/NAS/VLA/dataset/OCT_insertion/Captures*",
        "/home/najo/NAS/VLA/dataset/part1/ZED_Captures_*th",
    ]

    all_dirs = sensor_dataset_dirs + nosensor_dataset_dirs
    for pattern in all_dirs:
        expanded_paths = glob.glob(pattern)
        for traj_dir in expanded_paths:
            try:
                ds = insertionMeca500DatasetWithSensor(
                    trajectory_dir=traj_dir,
                    horizon=8,
                    sensor_window_size=args.sensor_window_size,  # 🔥 Use async window size
                )
                datasets.append(ds)
                if rank == 0:
                    print(f"✅ Added: {Path(traj_dir).name} ({len(ds)} samples)")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ Failed to load {traj_dir}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    full_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    if rank == 0:
        print(f"\n📊 Total dataset size: {len(full_dataset)} samples")

    # Create async model
    model = create_async_model(
        stage1_checkpoint=args.stage1_checkpoint,
        finetune_vl=args.finetune_vl,
        sensor_window_size=args.sensor_window_size,
        vlm_reuse_count=args.vlm_reuse_count,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    ).to(device)

    # Split dataset
    total_len = len(full_dataset)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

    # DDP Samplers
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        collate_fn=collate_fn_with_sensor,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        collate_fn=collate_fn_with_sensor,
        persistent_workers=False,
        pin_memory=False,
    )

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer
    def wd_filter(name, param):
        if param.ndim == 1: return False
        if name.endswith(".bias"): return False
        return True

    ae_named = list(model.module.action_expert.named_parameters())
    vl_named = list(model.module.vl_model.named_parameters())

    ae_decay = [p for n,p in ae_named if wd_filter(n,p) and p.requires_grad]
    ae_n_decay = [p for n,p in ae_named if not wd_filter(n,p) and p.requires_grad]

    vl_decay = [p for n,p in vl_named if wd_filter(n,p) and p.requires_grad]
    vl_n_decay = [p for n,p in vl_named if not wd_filter(n,p) and p.requires_grad]

    sensor_decay, sensor_n_decay = [], []
    if args.sensor_enabled and hasattr(model.module, 'sensor_encoder'):
        sensor_named = list(model.module.sensor_encoder.named_parameters())
        sensor_decay = [p for n,p in sensor_named if wd_filter(n,p) and p.requires_grad]
        sensor_n_decay = [p for n,p in sensor_named if not wd_filter(n,p) and p.requires_grad]

    param_groups = [
        {"params": ae_decay, "lr": args.lr, "weight_decay": 0.01},
        {"params": ae_n_decay, "lr": args.lr, "weight_decay": 0.0},
    ]

    if args.sensor_enabled and (sensor_decay or sensor_n_decay):
        param_groups += [
            {"params": sensor_decay, "lr": args.sensor_lr, "weight_decay": 0.01},
            {"params": sensor_n_decay, "lr": args.sensor_lr, "weight_decay": 0.0},
        ]

    if args.finetune_vl == "lora":
        param_groups += [
            {"params": vl_decay, "lr": args.vl_lr, "weight_decay": 0.01},
            {"params": vl_n_decay, "lr": args.vl_lr, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler
    total_epochs = 100
    iters_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(iters_per_epoch / max(1, args.grad_accum_steps))
    total_steps = steps_per_epoch * total_epochs

    scheduler = build_trapezoid_scheduler(
        optimizer,
        total_steps=total_steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        hold_ratio=args.hold_ratio,
    )

    save_path = './checkpoints/qwen_vla_async.pt'

    # Train
    Train_Async(
        model,
        train_loader,
        optimizer,
        num_epochs=total_epochs,
        grad_accum_steps=args.grad_accum_steps,
        device=device,
        save_path=save_path,
        scheduler=scheduler,
        sched_on=args.sched_on,
        val_loader=val_loader,
        start_epoch=0,
        sensor_enabled=args.sensor_enabled,
        sensor_loss_weight=args.sensor_loss_weight,
        vlm_reuse_count=args.vlm_reuse_count,
    )

    if rank == 0:
        final_path = Path("./checkpoints/qwen_vla_async_final.pt")
        torch.save({
            "epoch": total_epochs - 1,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }, final_path)
        print(f"✅ Final checkpoint saved at {final_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["PYTHONBUFFERED"] = "1"
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
