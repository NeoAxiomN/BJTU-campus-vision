from .device import resolve_device

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
	enabled: bool
	rank: int
	local_rank: int
	world_size: int

	@property
	def is_main(
		self,
	) -> bool:
		return self.rank == 0


def setup_distributed(
	requested_device: str,
) -> tuple[DistributedState, torch.device]:
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	rank = int(os.environ.get("RANK", "0"))
	local_rank = int(os.environ.get("LOCAL_RANK", "0"))
	enabled = world_size > 1
	state = DistributedState(enabled=enabled, rank=rank, local_rank=local_rank, world_size=world_size)
	if not enabled:
		return state, _resolve_single_process_device(requested_device)
	if requested_device not in {"auto", "cuda"}:
		raise RuntimeError("Distributed training requires --device cuda or --device auto.")
	if not torch.cuda.is_available():
		raise RuntimeError("Distributed CUDA training was requested, but CUDA is not available.")
	torch.cuda.set_device(local_rank)
	if not dist.is_initialized():
		dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
	return state, torch.device("cuda", local_rank)


def cleanup_distributed(
	state: DistributedState,
):
	if state.enabled and dist.is_initialized():
		dist.destroy_process_group()


def reduce_mean(
	value: float,
	device: torch.device,
	state: DistributedState,
) -> float:
	if not state.enabled:
		return value
	tensor = torch.tensor(value, device=device)
	dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
	tensor /= state.world_size
	return float(tensor.item())


def unwrap_model(
	model,
):
	if hasattr(model, "module"):
		return model.module
	return model


def _resolve_single_process_device(
	requested_device: str,
) -> torch.device:
	return resolve_device(requested_device)
