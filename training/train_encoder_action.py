"""
Training Script for Sensor Encoder + Action Expert ONLY
(VL Backbone is FROZEN - No LoRA, No Fine-tuning)

This script trains:
1. SensorEncoder (Trainable)
2. QwenActionExpertWithSensor (Trainable)
3. Qwen-VL Backbone (FROZEN - for feature extraction only)

Usage:
    # Build cache first (one-time)
    torchrun --nproc_per_node=4 training/train_encoder_action.py --mode cache

    # Train encoder and action expert
    torchrun --nproc_per_node=4 training/train_encoder_action.py --mode train \
        --sensor-enabled --fusion-strategy concat --batch-size 2
"""

import argparse
import wandb
import io, shutil, threading, queue, time
import os
import re
import math
import glob
import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

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

# Seed for reproducibility
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

# ✅ Import sensor-enabled model (FROZEN VL version)
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.model_with_sensor import QwenVLAWithSensor  # ✅ Use FROZEN VL version
from vla_datasets.IntegratedDataset import collate_fn_with_sensor
from training.Make_VL_cache import build_vl_cache_distributed_optimized

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
    """Copy network file to local staging then torch.load"""
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))
    local_copy = STAGING_DIR / src_path.name
    shutil.copy2(src_path, local_copy)
    try:
        return torch.load(local_copy, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(local_copy, map_location=map_location)

class AsyncCheckpointWriter:
    """Background thread for checkpoint writing"""
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
    """
    Warmup -> Hold -> Cosine Decay scheduler
    """
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
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))
            return floor + (1 - floor) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def build_rewarm_scheduler(
    optimizer,
    total_steps: int,
    *,
    prev_lr: float,
    target_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.05,
    hold_ratio: float = 0.05,
):
    """
    ReWarm Scheduler for resumed training
    """
    assert target_lr > 0 and min_lr > 0
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    decay_steps = max(1, total_steps - warmup_steps - hold_steps)

    floor = min_lr / target_lr
    start = max(1e-12, prev_lr / target_lr)

    def lr_lambda(step: int):
        if step < warmup_steps:
            prog = (step + 1) / max(1, warmup_steps)
            return start + (1.0 - start) * prog
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))
            return floor + (1.0 - floor) * cos

    for g in optimizer.param_groups:
        g["lr"] = target_lr

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched

# ===========================================================
# 1️⃣ Distributed Setup
# ===========================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank

# ===========================================================
# 2️⃣ Training Loop
# ===========================================================
def Train(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/encoder_action.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,
    sensor_loss_weight=2.0,
):
    loss_fn = nn.MSELoss()
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-EncoderAction",
            name=f"encoder_action_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_ea_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
                "trainable_modules": "SensorEncoder + ActionExpert ONLY (VL Frozen)",
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

            # ✅ Handle sensor data
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ✅ Pass sensor_data to model
                pred_actions, _ = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    z_chunk=gt_actions,
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data if sensor_enabled else None,
                )

                # ✅ Weighted loss based on sensor availability
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

            # ✅ Track sensor statistics
            if sensor_enabled and has_sensor_mask is not None:
                total_sensor_samples += has_sensor_mask.sum().item()
                total_nonsensor_samples += (~has_sensor_mask).sum().item()

            # === Gradient step ===
            if (step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and sched_on == "step":
                    scheduler.step()

                global_step += 1

                # === Logging ===
                lr = optimizer.param_groups[0]["lr"]
                if rank == 0:
                    postfix_dict = {
                        "loss": f"{loss.item() * grad_accum_steps:.6f}",
                        "lr": f"{lr:.2e}",
                        "grad": f"{grad_norm:.2f}"
                    }
                    if sensor_enabled:
                        postfix_dict["sensor"] = f"{total_sensor_samples}/{total_sensor_samples+total_nonsensor_samples}"
                    pbar.set_postfix(postfix_dict)

                    log_dict = {
                        "train/loss_step": loss.item() * grad_accum_steps,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "global_step": global_step
                    }
                    if sensor_enabled:
                        log_dict["train/sensor_samples"] = total_sensor_samples
                        log_dict["train/nonsensor_samples"] = total_nonsensor_samples
                    wandb.log(log_dict)

        # === Epoch average ===
        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        # === Scheduler per epoch ===
        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # === Validation Loop ===
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
                    loss = (loss_each * weights).mean()
                    val_loss_sum += loss.item()
                    val_count += 1
            val_loss = val_loss_sum / max(1, val_count)
            model.train()

        # === Epoch summary ===
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
                "params/frozen_ratio": frozen / total_params,
                "system/gpu_mem_GB": gpu_mem,
                "system/cpu_mem_%": cpu_mem,
                "lr/base_lr": optimizer.param_groups[0]["lr"],
            }

            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            wandb.log(log_dict)

            print(f"[DEBUG] GPU {gpu_mem:.2f} GB / CPU {cpu_mem:.1f}% used "
                  f"| Trainable {trainable/1e6:.2f}M / Frozen {frozen/1e6:.2f}M")

            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | "
                  f"Val: {val_loss:.8f}" if val_loss else f"\n📊 Epoch {epoch+1} Train Loss: {avg_loss:.8f}")

            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_path = Path(save_path)

            # === Best model saving ===
            if not hasattr(Train, "_best_loss"):
                Train._best_loss = float("inf")

            is_best = val_loss is not None and val_loss < Train._best_loss

            if is_best:
                Train._best_loss = val_loss
                best_path = save_dir / "encoder_action_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                }, best_path)
                print(f"🏆 [Best] Validation improved → saved to {best_path}")

            else:
                tmp_path = base_path.with_suffix(".tmp")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                }, tmp_path)
                os.replace(tmp_path, base_path)
                print(f"💾 [Sync] Latest checkpoint updated: {base_path}")

    if rank == 0 and writer is not None:
        writer.close()

    if rank == 0:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cache", "train"], required=True,
                        help="Mode: 'cache' to build feature cache, 'train' to train encoder+action")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Base learning rate for action expert")
    parser.add_argument("--sensor-lr", type=float, default=5e-4,
                        help="Learning rate for sensor encoder")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--hold-ratio", type=float, default=0.02)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--sched-on", choices=["step", "epoch"], default="step")

    # ✅ Sensor-related options
    parser.add_argument("--sensor-enabled", action="store_true",
                        help="Enable sensor encoder training")
    parser.add_argument("--sensor-input-channels", type=int, default=1026)
    parser.add_argument("--sensor-temporal-length", type=int, default=650)
    parser.add_argument("--sensor-output-dim", type=int, default=3072)
    parser.add_argument("--fusion-strategy", choices=["concat", "cross_attention", "gated", "none"],
                        default="concat")
    parser.add_argument("--sensor-loss-weight", type=float, default=2.0)

    # ✅ Dataset & training
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"🚀 [Rank {rank}] Running in {args.mode.upper()} mode on {world_size} GPUs")
        print(f"🔬 Sensor enabled: {args.sensor_enabled}")
        if args.sensor_enabled:
            print(f"   - Fusion strategy: {args.fusion_strategy}")
            print(f"   - Sensor LR: {args.sensor_lr}")
            print(f"   - Action Expert LR: {args.lr}")
            print(f"   - Sensor loss weight: {args.sensor_loss_weight}")
        print(f"📌 Training ONLY: SensorEncoder + ActionExpert (VL Backbone FROZEN)")

    # ✅ Build integrated dataset
    if rank == 0:
        print("📦 Building integrated dataset...")

    from vla_datasets.IntegratedDataset import insertionMeca500DatasetWithSensor
    import glob

    datasets = []

    # With sensor data
    sensor_dataset_dirs = [
        "/home/najo/NAS/VLA/dataset/White_silicone_white_circle/recv_all_*",
        "/home/najo/NAS/VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]

    # Without sensor data
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
                    horizon=8
                )
                datasets.append(ds)
                sensor_status = "WITH sensor" if ds.has_sensor else "NO sensor"
                if rank == 0:
                    print(f"✅ Added: {Path(traj_dir).name} ({len(ds)} samples, {sensor_status})")
            except Exception as e:
                if rank == 0:
                    print(f"⚠️ Failed to load {traj_dir}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded!")

    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    if rank == 0:
        print(f"\n📊 Total dataset size: {len(full_dataset)} samples")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ===========================================================
    # 3️⃣ Cache Mode
    # ===========================================================
    if args.mode == "cache":
        if rank == 0:
            print("⏳ Initializing VL-only model for cache building...")

        processor = AutoProcessor.from_pretrained(vl_model_name)
        vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vl_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda",
            low_cpu_mem_usage=True,
        )

        class DummyVLA:
            def __init__(self, vl_model, processor):
                self.vl_model = vl_model
                self.processor = processor
                self.cache_dir = Path("/home/najo/NAS/VLA/dataset/cache/qwen_vl_features")
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Bind instance methods from QwenVLAWithSensor
                from models.model_with_sensor import QwenVLAWithSensor
                self._cache_path = QwenVLAWithSensor._cache_path.__get__(self)
                self._enforce_cache_limit = QwenVLAWithSensor._enforce_cache_limit.__get__(self)
                self._atomic_save = QwenVLAWithSensor._atomic_save

            def eval(self):
                self.vl_model.eval()
                return self

        dummy_model = DummyVLA(vl_model, processor)

        build_vl_cache_distributed_optimized(
            dummy_model, full_dataset, device=device,
            rank_sharded_cache=False
        )

        dist.barrier()
        if rank == 0:
            print("✅ Cache build complete. You can now run training with --mode train.")
        dist.destroy_process_group()
        return

    # ===========================================================
    # 4️⃣ Training Mode
    # ===========================================================
    if args.mode == "train":
        if rank == 0:
            print("⏳ Initializing QwenVLA model (FROZEN VL) for training...")

        # ✅ Use FROZEN VL version (QwenVLAWithSensor)
        model = QwenVLAWithSensor(
            vl_model_name=vl_model_name,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            sensor_enabled=args.sensor_enabled,
            sensor_input_channels=args.sensor_input_channels,
            sensor_temporal_length=args.sensor_temporal_length,
            sensor_output_dim=args.sensor_output_dim,
            fusion_strategy=args.fusion_strategy,
        ).to(device)

        # ✅ Verify VL backbone is frozen
        if rank == 0:
            vl_params_trainable = sum(p.numel() for p in model.vl_model.parameters() if p.requires_grad)
            if vl_params_trainable == 0:
                print("✅ Confirmed: VL Backbone is FROZEN (0 trainable params)")
            else:
                print(f"⚠️ WARNING: VL Backbone has {vl_params_trainable} trainable params!")

        # Split dataset
        total_len = len(full_dataset)
        val_len = int(total_len * 0.05)
        train_len = total_len - val_len
        train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

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

        # === Optimizer (ONLY for trainable parts) ===
        def wd_filter(name, param):
            if param.ndim == 1: return False
            if name.endswith(".bias"): return False
            return True

        # ✅ Collect trainable parameters ONLY from action_expert and sensor_encoder
        ae_named = list(model.module.action_expert.named_parameters())
        ae_decay = [p for n,p in ae_named if wd_filter(n,p) and p.requires_grad]
        ae_n_decay = [p for n,p in ae_named if not wd_filter(n,p) and p.requires_grad]

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

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        # === Load checkpoint if exists ===
        ckpt_path = "./checkpoints/encoder_action.pt"
        start_epoch = 0
        if os.path.exists(ckpt_path):
            if rank == 0:
                print(f"🔄 Found checkpoint at {ckpt_path}, resuming training...")
            checkpoint = copy_to_local_then_load(Path(ckpt_path), map_location=device)

            try:
                model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("✅ Loaded model weights (partial, strict=False)")
            except KeyError:
                model.module.load_state_dict(checkpoint, strict=False)

            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except ValueError:
                    print("⚠️ Optimizer group mismatch — skipping optimizer state load.")

            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                if rank == 0:
                    print(f"✅ Resumed from epoch {start_epoch}")
        else:
            if rank == 0:
                print("🆕 No checkpoint found, starting from scratch.")

        # === Scheduler ===
        remaining_epochs = max(1, args.epochs - start_epoch)
        iters_per_epoch = len(train_loader)
        steps_per_epoch = math.ceil(iters_per_epoch / max(1, args.grad_accum_steps))
        total_steps = steps_per_epoch * remaining_epochs

        current_lr = optimizer.param_groups[0]["lr"]
        target_lr = args.lr

        if start_epoch > 0:
            scheduler = build_rewarm_scheduler(
                optimizer,
                total_steps=total_steps,
                prev_lr=current_lr,
                target_lr=target_lr,
                min_lr=args.min_lr,
                warmup_ratio=0.05,
                hold_ratio=0.05,
            )
            if rank == 0:
                print(f"🔁 [ReWarm] {current_lr:.2e} → {target_lr:.2e}, then cosine → {args.min_lr:.2e}")
        else:
            scheduler = build_trapezoid_scheduler(
                optimizer,
                total_steps=total_steps,
                base_lr=target_lr,
                min_lr=args.min_lr,
                warmup_ratio=args.warmup_ratio,
                hold_ratio=args.hold_ratio,
            )
            if rank == 0:
                print(f"🆕 [New] 0 → {target_lr:.2e} (warmup/hold/cosine)")

        save_path = './checkpoints/encoder_action.pt'

        # === Train ===
        Train(
            model,
            train_loader,
            optimizer,
            num_epochs=remaining_epochs,
            grad_accum_steps=args.grad_accum_steps,
            device=device,
            save_path=save_path,
            scheduler=scheduler,
            sched_on=args.sched_on,
            val_loader=val_loader,
            start_epoch=start_epoch,
            sensor_enabled=args.sensor_enabled,
            sensor_loss_weight=args.sensor_loss_weight,
        )

        # === Final checkpoint ===
        if rank == 0:
            final_path = Path("./checkpoints/encoder_action_final.pt")
            torch.save({
                "epoch": args.epochs - 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            }, final_path)
            print(f"✅ [Final] Final checkpoint saved at {final_path}")

        dist.destroy_process_group()
        if rank == 0:
            print("🧹 DDP process group destroyed. Training complete.")

if __name__ == "__main__":
    os.environ["PYTHONBUFFERED"] = "1"
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
