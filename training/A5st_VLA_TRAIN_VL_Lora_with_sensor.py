"""
Updated Training Script for QwenVLA with Sensor Integration

Changes from original:
- Uses model_with_sensor.Not_freeze_QwenVLAWithSensor
- Uses IntegratedDataset with create_integrated_dataloader
- Handles mixed datasets (with/without sensor data)
- Supports sensor encoder training
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

# ✅ NEW: Import sensor-enabled model and dataset
from model_with_sensor import Not_freeze_QwenVLAWithSensor
from IntegratedDataset import collate_fn_with_sensor, create_integrated_dataloader
from Make_VL_cache import build_vl_cache_distributed_optimized

# ======== I/O & Checkpoint Utils ========
STAGING_DIR = Path("/dev/shm/qwen_vla_stage")   # 로컬 RAM/NVMe (없으면 /tmp 권장)
CKPT_DIR     = Path("./checkpoints")            # NAS 또는 공유 디렉토리
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
    shutil.copy2(src_path, local_copy)  # 보통 이 경로가 훨씬 빠름
    # PyTorch 2.4+면 weights_only=True가 빠르고 안전
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
        self.sync_every = sync_every  # 0이면 즉시 처리
        self.thread.start()

    def _worker(self):
        last_sync = time.time()
        while not self.stop:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            state_dict, final_dst = payload["state"], Path(payload["dst"])
            # 1) 로컬 스테이징에 먼저 저장 (CPU 텐서)
            local_tmp = STAGING_DIR / (final_dst.name + f".{int(time.time())}.pt")
            # 모델/옵티마이저 텐서를 CPU로 복사한 상태를 받는 것이 이상적
            torch.save(state_dict, local_tmp, _use_new_zipfile_serialization=True)
            # 2) 필요 시 배치/주기 동기화
            if self.sync_every > 0 and (time.time() - last_sync) < self.sync_every:
                continue
            # 3) 원자적 교체로 최종 목적지에 반영
            _atomic_move(local_tmp, final_dst)
            last_sync = time.time()

    def submit(self, state_dict, final_dst: Path):
        # 큐가 가득 차 있으면 가장 오래된 걸 버리고 최신으로 교체(학습 지연 방지)
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
    LLM 스타일: Warmup -> Hold(선택) -> Cosine Decay
    - warmup_ratio: 전체 step 대비 워밍업 비율
    - hold_ratio  : 워밍업 후 고정 유지 비율 (0 가능)
    - 나머지는 cosine으로 min_lr까지 감쇠
    """
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps   = int(total_steps * hold_ratio)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)
    floor = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_steps:
            # 선형 워밍업: 0 -> 1
            return (step + 1) / max(1, warmup_steps)
        elif step < warmup_steps + hold_steps:
            # 유지: 1.0
            return 1.0
        else:
            # 코사인 감쇠: 1 -> floor
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))  # 1 -> 0
            return floor + (1 - floor) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

def build_rewarm_scheduler(
    optimizer,
    total_steps: int,
    *,
    prev_lr: float,        # 마지막 학습에서 쓰던 실 lr (예: 1e-8)
    target_lr: float = 1e-4,
    min_lr: float = 1e-6,
    warmup_ratio: float = 0.05,
    hold_ratio: float = 0.05,
):
    """
    🔁 ReWarm Scheduler (절대 lr 스케줄을 factor로 표현)
      - base_lr = target_lr 로 잡고
      - factor를 prev_lr/target_lr → 1.0 으로 warmup
      - hold 후 1.0 → (min_lr/target_lr) 로 cosine decay
    """
    assert target_lr > 0 and min_lr > 0
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps   = int(total_steps * hold_ratio)
    decay_steps  = max(1, total_steps - warmup_steps - hold_steps)

    floor = min_lr / target_lr                 # decay 최하 한계 (factor)
    start = max(1e-12, prev_lr / target_lr)    # warmup 시작 factor(아주 작을 수 있음)

    def lr_lambda(step: int):
        if step < warmup_steps:
            # 선형: start → 1.0
            prog = (step + 1) / max(1, warmup_steps)
            return start + (1.0 - start) * prog
        elif step < warmup_steps + hold_steps:
            return 1.0
        else:
            # cosine: 1.0 → floor
            t = (step - warmup_steps - hold_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * t))   # 1 → 0
            return floor + (1.0 - floor) * cos

    # ① base_lr를 target_lr로 맞춤
    for g in optimizer.param_groups:
        g["lr"] = target_lr

    sched = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ② 첫 step 전에 실 lr을 prev_lr로 맞춰 시작 (warmup 시작점)
    for g in optimizer.param_groups:
        g["lr"] = prev_lr
    return sched

# ===========================================================
# 1️⃣ 초기화
# ===========================================================
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank

# ===========================================================
# 2️⃣ 학습 루프
# ===========================================================
def Train(
    model,
    data_loader,
    optimizer,
    num_epochs=3,
    grad_accum_steps=8,
    device="cuda",
    save_path="./checkpoints/qwen_vla.pt",
    scheduler=None,
    sched_on="step",
    val_loader=None,
    start_epoch=0,
    sensor_enabled=True,  # ✅ NEW
    sensor_loss_weight=2.0,  # ✅ NEW: Weight for samples with sensor
):
    loss_fn = nn.MSELoss()
    rank = dist.get_rank()
    writer = AsyncCheckpointWriter(max_queue=2, sync_every=0) if rank == 0 else None

    model.train()
    if rank == 0:
        wandb.init(
            project="QwenVLA-Sensor",  # ✅ Changed project name
            name=f"train_sensor_{time.strftime('%m%d_%H%M')}",
            resume="allow",
            id=f"qvla_sensor_{int(time.time())}",
            settings=wandb.Settings(start_method="thread", _disable_stats=True),
            config={
                "lr": optimizer.param_groups[0]["lr"],
                "grad_accum_steps": grad_accum_steps,
                "epochs": num_epochs,
                "scheduler": sched_on,
                "sensor_enabled": sensor_enabled,
                "sensor_loss_weight": sensor_loss_weight,
            }
        )

    global_step = 0
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_sensor_samples = 0  # ✅ NEW: Track sensor samples
        total_nonsensor_samples = 0  # ✅ NEW
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(data_loader), total=len(data_loader),
                    desc=f"[Rank {rank}] Epoch {epoch+1}",
                    disable=(rank != 0))

        for step, batch in pbar:
            instructions = batch["instruction"]
            image_inputs = batch["images"]
            gt_actions = batch["actions"].to(device, dtype=torch.bfloat16)

            # ✅ NEW: Handle sensor data
            sensor_data = batch["sensor_data"].to(device, dtype=torch.bfloat16) if sensor_enabled else None
            has_sensor_mask = batch["has_sensor_mask"].to(device) if sensor_enabled else None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ✅ NEW: Pass sensor_data to model
                pred_actions, _ = model(
                    text_inputs=instructions,
                    image_inputs=image_inputs,
                    z_chunk=gt_actions,
                    cache_keys=batch["cache_keys"],
                    sensor_data=sensor_data if sensor_enabled else None,
                )

                # ✅ NEW: Weighted loss based on sensor availability
                weights = torch.tensor(batch["confidence"], device=device, dtype=torch.bfloat16)

                if sensor_enabled and has_sensor_mask is not None:
                    # Higher weight for samples with real sensor data
                    sensor_weights = torch.where(has_sensor_mask,
                                                 torch.tensor(sensor_loss_weight, device=device),
                                                 torch.tensor(1.0, device=device))
                    weights = weights * sensor_weights

                weights = weights / weights.mean()
                loss_each = (pred_actions.float() - gt_actions.float()).pow(2).mean(dim=[1,2])
                loss = (loss_each * weights).mean() / grad_accum_steps

            loss.backward()
            total_loss += loss.item() * grad_accum_steps

            # ✅ NEW: Track sensor statistics
            if sensor_enabled and has_sensor_mask is not None:
                total_sensor_samples += has_sensor_mask.sum().item()
                total_nonsensor_samples += (~has_sensor_mask).sum().item()

            # === gradient step ===
            if (step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None and sched_on == "step":
                    scheduler.step()

                global_step += 1

                # === ✅ tqdm 및 wandb 로깅 ===
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

        # === epoch 평균 ===
        avg_loss_tensor = torch.tensor(total_loss / len(data_loader), device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

        # === scheduler per epoch ===
        if scheduler is not None and sched_on == "epoch":
            scheduler.step()

        # === ✅ Validation Loop ===
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



        # === epoch 종료 후 ===
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            frozen = total_params - trainable

            import psutil, gc
            gpu_mem = torch.cuda.memory_allocated()/1e9
            cpu_mem = psutil.virtual_memory().percent
            gc.collect()

            # === 추가 로깅 ===
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

            # ✅ NEW: Log sensor statistics
            if sensor_enabled:
                log_dict["train/epoch_sensor_samples"] = total_sensor_samples
                log_dict["train/epoch_nonsensor_samples"] = total_nonsensor_samples
                log_dict["train/sensor_ratio"] = total_sensor_samples / max(1, total_sensor_samples + total_nonsensor_samples)

            if len(optimizer.param_groups) > 1:
                log_dict["lr/vl_lr"] = optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 2:
                log_dict["lr/vision_lr"] = optimizer.param_groups[2]["lr"]

            wandb.log(log_dict)

            print(f"[DEBUG] GPU {gpu_mem:.2f} GB / CPU {cpu_mem:.1f}% used "
                  f"| Trainable {trainable/1e6:.2f}M / Frozen {frozen/1e6:.2f}M")

            # === LoRA 파라미터 로깅 (선택) ===
            lora_params = {n: p for n, p in model.named_parameters() if "lora_" in n}
            if lora_params:
                avg_abs = np.mean([p.data.abs().mean().item() for p in lora_params.values()])
                wandb.log({"lora/avg_weight_abs": avg_abs})

            print(f"\n📊 Epoch {epoch+1} Summary | Train: {avg_loss:.8f} | "
                  f"Val: {val_loss:.8f}" if val_loss else f"\n📊 Epoch {epoch+1} Train Loss: {avg_loss:.8f}")

            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            base_path = Path(save_path)

            # === Best model 저장 여부 판단 ===
            if not hasattr(Train, "_best_loss"):
                Train._best_loss = float("inf")

            # Determine what to save based on training stage
            model_module = model.module if hasattr(model, "module") else model
            training_stage = getattr(Train, '_training_stage', 'stage2')

            if training_stage == 'stage1':
                # Stage 1: Save only Sensor Encoder + Action Expert
                sensor_enabled = getattr(Train, '_sensor_enabled', True)
                ckpt_data = {
                    "epoch": epoch,
                    "sensor_encoder": model_module.sensor_encoder.state_dict() if sensor_enabled else None,
                    "action_expert": model_module.action_expert.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                    "training_stage": "stage1",
                }
                stage_label = "[Stage 1]"
            else:
                # Stage 2: Save full model state
                ckpt_data = {
                    "epoch": epoch,
                    "model_state_dict": model_module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_loss": val_loss,
                    "training_stage": "stage2",
                }
                stage_label = "[Stage 2]"

            is_best = val_loss is not None and val_loss < Train._best_loss

            if is_best:
                Train._best_loss = val_loss
                best_path = save_dir / "qwen_vla_sensor_best.pt"
                torch.save(ckpt_data, best_path)
                print(f"🏆 {stage_label} [Best] Validation improved → saved to {best_path}")

            else:
                # Best가 아닌 경우 기존 latest만 덮어쓰기
                tmp_path = base_path.with_suffix(".tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, base_path)
                print(f"💾 {stage_label} [Sync] Latest checkpoint updated: {base_path}")

    if rank == 0 and writer is not None:
        writer.close()

    if rank == 0:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    # ===== 기존 VLA 기본 학습 옵션 =====
    parser.add_argument("--mode", choices=["cache", "train"], required=True,
                        help="Mode: 'cache' to build feature cache, 'train' to train action expert")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--hold-ratio", type=float, default=0.02)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--sched-on", choices=["step", "epoch"], default="step")

    # ===== LoRA / Fine-tuning 옵션 =====
    parser.add_argument("--finetune-vl", choices=["none", "lora", "full"], default="lora",
                        help="Qwen-VL fine-tuning mode")
    parser.add_argument("--vl-lr", type=float, default=1e-5,
                        help="learning rate for VL backbone (LoRA/full mode)")
    parser.add_argument("--vision-lr", type=float, default=5e-6,
                        help="learning rate for vision encoder if full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--unfreeze-last-n", type=int, default=2,
                        help="number of transformer blocks to unfreeze when full fine-tuning")

    # ✅ NEW: Sensor-related options
    parser.add_argument("--sensor-enabled", action="store_true",
                        help="Enable sensor encoder training")
    parser.add_argument("--sensor-input-channels", type=int, default=1026,
                        help="Sensor input channels (1 force + 1025 A-scan)")
    parser.add_argument("--sensor-temporal-length", type=int, default=650,
                        help="Sensor temporal length (650Hz * 1s)")
    parser.add_argument("--sensor-output-dim", type=int, default=3072,
                        help="Sensor encoder output dimension (should match VL dimension)")
    parser.add_argument("--fusion-strategy", choices=["concat", "cross_attention", "gated", "none"],
                        default="concat", help="Sensor-VL fusion strategy")
    parser.add_argument("--sensor-lr", type=float, default=5e-4,
                        help="Learning rate for sensor encoder")
    parser.add_argument("--sensor-loss-weight", type=float, default=2.0,
                        help="Loss weight multiplier for samples with sensor data")

    # ✅ NEW: Dataset paths
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of dataloader workers")

    # 🔥 NEW: 2-Stage Training Arguments
    parser.add_argument('--training-stage', type=str, choices=['stage1', 'stage2'], default='stage2',
                        help='Training stage: stage1 (Sensor+Action only, VL frozen) or stage2 (Full model with LoRA)')
    parser.add_argument('--stage1-checkpoint', type=str, default=None,
                        help='Path to Stage 1 checkpoint (required for stage2 with finetune-vl != none)')

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"🚀 [Rank {rank}] Running in {args.mode.upper()} mode on {world_size} GPUs")
        print(f"🔬 Sensor enabled: {args.sensor_enabled}")
        if args.sensor_enabled:
            print(f"   - Fusion strategy: {args.fusion_strategy}")
            print(f"   - Sensor LR: {args.sensor_lr}")
            print(f"   - Sensor loss weight: {args.sensor_loss_weight}")

    # ✅ NEW: Build integrated dataset manually
    if rank == 0:
        print("📦 Building integrated dataset...")

    from IntegratedDataset import insertionMeca500DatasetWithSensor
    import glob

    datasets = []

    # With sensor data
    sensor_dataset_dirs = [
        "/home/najo/NAS/VLA/Insertion_VLA/dataset/White_silicone_white_circle/recv_all_*",
        "/home/najo/NAS/VLA/Insertion_VLA/dataset/Needle_insertion_eye_trocar/recv_all_*",
    ]

    # Without sensor data
    nosensor_dataset_dirs = [
        "/home/najo/NAS/VLA/Insertion_VLA/dataset/OCT_insertion/Captures*",
        "/home/najo/NAS/VLA/Insertion_VLA/dataset/part1/ZED_Captures_*th",
    ]

    # Expand wildcards and load datasets
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

    # Concatenate all datasets
    if len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        full_dataset = ConcatDataset(datasets)

    if rank == 0:
        print(f"\n📊 Total dataset size: {len(full_dataset)} samples")

    vl_model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ===========================================================
    # 3️⃣ 캐시 생성 모드
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

                # ✅ 인스턴스 메서드만 바인딩
                self._cache_path = Not_freeze_QwenVLAWithSensor._cache_path.__get__(self)
                self._enforce_cache_limit = Not_freeze_QwenVLAWithSensor._enforce_cache_limit.__get__(self)

                # ✅ staticmethod는 그대로 복사
                self._atomic_save = Not_freeze_QwenVLAWithSensor._atomic_save

            def eval(self):
                self.vl_model.eval()
                return self

        dummy_model = DummyVLA(vl_model, processor)

        # 캐시 빌드
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
    # 4️⃣ 학습 모드
    # ===========================================================
    if args.mode == "train":
        if rank == 0:
            print("⏳ Initializing full QwenVLA model for training...")

        # ✅ NEW: Validate Stage 2 requirements
        if args.training_stage == 'stage2' and args.finetune_vl != 'none' and not args.stage1_checkpoint:
            print("⚠️  WARNING: Stage 2 training without Stage 1 checkpoint. Training from scratch.")

        if args.training_stage == 'stage1' and args.finetune_vl != 'none':
            print("⚠️  WARNING: Stage 1 should use finetune-vl=none. Overriding to 'none'.")
            args.finetune_vl = 'none'

        # ✅ NEW: Initialize sensor-enabled model
        model = Not_freeze_QwenVLAWithSensor(
            vl_model_name=vl_model_name,
            action_dim=7,
            horizon=8,
            hidden_dim=1024,
            finetune_vl=args.finetune_vl,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            unfreeze_last_n=args.unfreeze_last_n,

            # ✅ NEW: Sensor configuration
            sensor_enabled=args.sensor_enabled,
            sensor_input_channels=args.sensor_input_channels,
            sensor_temporal_length=args.sensor_temporal_length,
            sensor_output_dim=args.sensor_output_dim,
            fusion_strategy=args.fusion_strategy,

            # 🔥 NEW: Stage 1 checkpoint loading
            stage1_checkpoint=args.stage1_checkpoint,
        ).to(device)

        # 전체 데이터셋 분할
        total_len = len(full_dataset)
        val_len = int(total_len * 0.05)   # 5% validation
        train_len = total_len - val_len
        train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

        # DDP용 Sampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

        # ✅ NEW: Use sensor-aware collate function
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

        # === Optimizer 구성 ===
        def wd_filter(name, param):
            if param.ndim == 1: return False
            if name.endswith(".bias"): return False
            return True

        ae_named = list(model.module.action_expert.named_parameters())
        vl_named = list(model.module.vl_model.named_parameters())

        ae_decay    = [p for n,p in ae_named if wd_filter(n,p) and p.requires_grad]
        ae_n_decay  = [p for n,p in ae_named if not wd_filter(n,p) and p.requires_grad]

        vl_decay   = [p for n,p in vl_named if wd_filter(n,p) and p.requires_grad]
        vl_n_decay = [p for n,p in vl_named if not wd_filter(n,p) and p.requires_grad]

        # ✅ NEW: Sensor encoder parameters
        sensor_decay, sensor_n_decay = [], []
        if args.sensor_enabled and hasattr(model.module, 'sensor_encoder'):
            sensor_named = list(model.module.sensor_encoder.named_parameters())
            sensor_decay = [p for n,p in sensor_named if wd_filter(n,p) and p.requires_grad]
            sensor_n_decay = [p for n,p in sensor_named if not wd_filter(n,p) and p.requires_grad]

        vision_decay, vision_n_decay = [], []
        for n,p in vl_named:
            if not p.requires_grad:
                continue
            if "vision" in n or "visual" in n or "vision_tower" in n:
                (vision_decay if wd_filter(n,p) else vision_n_decay).append(p)

        # === LR 설정 ===
        param_groups = [
            {"params": ae_decay,        "lr": args.lr,      "weight_decay": 0.01},
            {"params": ae_n_decay,      "lr": args.lr,      "weight_decay": 0.0},
        ]

        # ✅ NEW: Add sensor encoder param group
        if args.sensor_enabled and (sensor_decay or sensor_n_decay):
            param_groups += [
                {"params": sensor_decay,    "lr": args.sensor_lr,  "weight_decay": 0.01},
                {"params": sensor_n_decay,  "lr": args.sensor_lr,  "weight_decay": 0.0},
            ]

        if args.finetune_vl == "lora":
            param_groups += [
                {"params": vl_decay,    "lr": args.vl_lr,   "weight_decay": 0.01},
                {"params": vl_n_decay,  "lr": args.vl_lr,   "weight_decay": 0.0},
            ]
        elif args.finetune_vl == "full":
            param_groups += [
                {"params": vision_decay,    "lr": args.vision_lr,  "weight_decay": 0.01},
                {"params": vision_n_decay,  "lr": args.vision_lr,  "weight_decay": 0.0},
                {"params": vl_decay,        "lr": args.vl_lr,      "weight_decay": 0.01},
                {"params": vl_n_decay,      "lr": args.vl_lr,      "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


        # =======================================================
        # ✅ 체크포인트 복원 (model+optim) + epoch 복원
        # =======================================================
        ckpt_path = "./checkpoints/qwen_vla_sensor.pt"
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
                    print("⚠️ Optimizer group mismatch detected — skipping optimizer state load.")


            # ✅ 여기서 epoch 반드시 복원
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                if rank == 0:
                    print(f"✅ Resumed from epoch {start_epoch}")
        else:
            if rank == 0:
                print("🆕 No checkpoint found, starting from scratch.")

        # ===== 남은 epoch/step 계산 =====
        total_epochs = 100  # Adjust as needed
        remaining_epochs = max(1, total_epochs - start_epoch)
        iters_per_epoch = len(train_loader)
        steps_per_epoch = math.ceil(iters_per_epoch / max(1, args.grad_accum_steps))
        total_steps = steps_per_epoch * remaining_epochs

        # ===== 스케줄러 설정 =====
        current_lr = optimizer.param_groups[0]["lr"]
        target_lr  = args.lr

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

        save_path = './checkpoints/qwen_vla_sensor.pt'

        # Store training stage for checkpoint saving
        Train._training_stage = args.training_stage
        Train._sensor_enabled = args.sensor_enabled

        # =======================================================
        # ✅ Train 호출
        # =======================================================
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

        # =======================================================
        # ✅ 안전 저장 보장
        # =======================================================
        if rank == 0 and "writer" in locals() and writer is not None:
            print("💾 Flushing async checkpoints to disk...")
            writer.flush()
            writer.close()
            print("✅ All pending checkpoints flushed safely.")

        # ✅ 최종 체크포인트 저장
        if rank == 0:
            final_path = Path("./checkpoints/qwen_vla_sensor_final.pt")
            torch.save({
                "epoch": total_epochs - 1,
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
