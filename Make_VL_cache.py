import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from tqdm import tqdm

from Total_Dataset import collate_fn


def _batch_iterator(
    dataset,
    *,
    batch_size: int,
    sampler: Optional[Iterable[int]] = None,
):
    indices = sampler if sampler is not None else range(len(dataset))

    batch = []
    for idx in indices:
        batch.append(dataset[idx])
        if len(batch) == batch_size:
            yield collate_fn(batch)
            batch.clear()
    if batch:
        yield collate_fn(batch)


def build_vl_cache_distributed_optimized(
    model,
    dataset,
    *,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    rank_sharded_cache: bool = False,
    max_cache_gb: float = 20.0,
) -> None:
    """Populate VL feature cache for the provided dataset.

    Args:
        model: Object exposing ``vl_model``, ``processor`` and ``encode_vision``.
        dataset: Dataset returning dict samples compatible with ``collate_fn``.
        device: Target torch device.
        batch_size: Number of samples processed per step.
        num_workers: Unused (kept for backwards compatibility).
        rank_sharded_cache: When ``True`` only keeps cache built by current rank.
        max_cache_gb: Cache upper bound enforced after each write.
    """

    del num_workers  # kept for API compatibility

    _ = device  # Kept for backward compatibility

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    if hasattr(model, "set_cache_limit"):
        model.set_cache_limit(max_cache_gb)
    else:
        setattr(model, "cache_limit_gb", max_cache_gb)
    if hasattr(model, "set_cache"):
        model.set_cache(True)
    if hasattr(model, "eval"):
        model.eval()

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        sampler.set_epoch(0)

    dataset_len = len(dataset)
    if dataset_len == 0:
        if rank == 0:
            tqdm.write("No samples available for VL cache build.")
        return

    iterator = _batch_iterator(dataset, batch_size=batch_size, sampler=sampler)
    total_steps = max(1, math.ceil(dataset_len / (batch_size * max(1, world_size))))

    progress = tqdm(
        iterator,
        total=total_steps,
        desc="Building VL cache",
        disable=rank != 0,
    )

    with torch.inference_mode():
        for batch in progress:
            instructions = batch["instruction"]
            images = batch["images"]
            keys = batch["cache_keys"]
            if hasattr(model, "encode_vision"):
                model.encode_vision(instructions, images, keys, cache=True)
            else:
                # Fall back to a forward pass to populate the cache
                device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda")
                actions = batch["actions"].to(device=device, dtype=torch.bfloat16)
                sensor = batch.get("sensor_data")
                if isinstance(sensor, torch.Tensor):
                    sensor = sensor.to(device=device, dtype=torch.bfloat16)
                model(
                    text_inputs=instructions,
                    image_inputs=images,
                    z_chunk=actions,
                    sensor_data=sensor,
                    cache_keys=keys,
                )

    if distributed and not rank_sharded_cache:
        dist.barrier()

