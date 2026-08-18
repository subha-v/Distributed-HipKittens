#!/usr/bin/env python3
"""Mixture-of-Kittens-style synthetic inputs for the PF4H prefill benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MoKSyntheticConfig:
    world_size: int = 8
    tokens_per_rank: int = 4096
    hidden_dim: int = 7168
    intermediate_dim: int = 2048
    num_experts: int = 256
    topk: int = 8
    seed_base: int = 1234
    weight_chunk_elements: int = 32 * 1024 * 1024
    # Optional measured-skew replay: path to a JSON file holding a 256-bin
    # expert-popularity histogram (see _load_route_hist for the format).  When
    # set, router logits become log(p) + Gumbel noise, so top-k sampling is
    # without-replacement proportional to the measured popularity instead of
    # the i.i.d.-normal balanced default.  Relative paths resolve against this
    # module's directory so the same value works on host and in-container.
    route_hist_path: str | None = None

    @property
    def local_experts(self) -> int:
        if self.num_experts % self.world_size:
            raise ValueError(
                f"{self.num_experts} experts cannot be divided across "
                f"{self.world_size} ranks"
            )
        return self.num_experts // self.world_size

    def validate(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.tokens_per_rank <= 0:
            raise ValueError("tokens_per_rank must be positive")
        if not 0 < self.topk <= self.num_experts:
            raise ValueError("topk must be in [1, num_experts]")
        if self.hidden_dim % 128 or self.intermediate_dim % 128:
            raise ValueError("hidden and intermediate dimensions must be multiples of 128")
        if self.weight_chunk_elements <= 0:
            raise ValueError("weight_chunk_elements must be positive")
        _ = self.local_experts


def _sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_route_hist(
    path: str, num_experts: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a measured expert-popularity histogram for routing replay.

    Format: JSON object with "counts" (num_experts non-negative numbers,
    normalized here) and optionally "meta" (provenance, recorded verbatim).
    Returns (probs float64 CPU tensor summing to 1, metadata dict).
    """

    resolved = path
    if not os.path.isabs(resolved):
        resolved = os.path.join(os.path.dirname(os.path.abspath(__file__)), resolved)
    with open(resolved, "r") as handle:
        payload = json.load(handle)
    counts = payload.get("counts")
    if not isinstance(counts, list) or len(counts) != num_experts:
        raise ValueError(
            f"route hist {resolved!r} must hold exactly {num_experts} counts, "
            f"got {type(counts).__name__} of length "
            f"{len(counts) if isinstance(counts, list) else 'n/a'}"
        )
    probs = torch.tensor(counts, dtype=torch.float64)
    if (probs < 0).any() or probs.sum() <= 0:
        raise ValueError(f"route hist {resolved!r} counts must be >=0 with a positive sum")
    probs = probs / probs.sum()
    with open(resolved, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    top1 = float(probs.max())
    metadata = {
        "route_hist_path": resolved,
        "route_hist_sha256": digest,
        "route_hist_meta": payload.get("meta", {}),
        "route_hist_top1_share": top1,
        "route_hist_top1_over_uniform": top1 * num_experts,
        "route_hist_zero_bins": int((probs == 0).sum()),
    }
    return probs, metadata


def _local_route_stats(
    expert_ids: torch.Tensor,
    router_weights: torch.Tensor,
    *,
    local_experts: int,
    world_size: int,
) -> dict[str, Any]:
    ids = expert_ids.detach().to(torch.int64).cpu()
    weights = router_weights.detach().to(torch.float32).cpu()
    destinations = ids // local_experts
    fanout = torch.tensor(
        [torch.unique(row).numel() for row in destinations], dtype=torch.int64
    )
    fanout_values, fanout_counts = torch.unique(fanout, return_counts=True)
    expert_counts = torch.bincount(ids.flatten(), minlength=world_size * local_experts)
    destination_counts = expert_counts.reshape(world_size, local_experts).sum(dim=1)
    return {
        "fanout_histogram": {
            str(int(value)): int(count)
            for value, count in zip(fanout_values, fanout_counts)
        },
        "fanout_mean": float(fanout.to(torch.float64).mean()),
        "fanout_max": int(fanout.max()),
        "expert_assignment_min": int(expert_counts.min()),
        "expert_assignment_max": int(expert_counts.max()),
        "expert_assignments_per_destination_rank": [
            int(value) for value in destination_counts
        ],
        "router_weight_sum_min": float(weights.sum(dim=1).min()),
        "router_weight_sum_max": float(weights.sum(dim=1).max()),
        "expert_ids_sha256": _sha256_tensor(ids.to(torch.int32)),
        "router_weights_sha256": _sha256_tensor(weights),
    }


def generate_router_and_hidden(
    rank: int,
    device: torch.device | str,
    config: MoKSyntheticConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Generator, dict[str, Any]]:
    """Generate the MoK router logits, top-k routes, weights, and BF16 activations."""

    config.validate()
    if not 0 <= rank < config.world_size:
        raise ValueError(f"rank {rank} is outside world size {config.world_size}")

    generator = torch.Generator(device=device).manual_seed(config.seed_base + rank)
    route_hist_metadata: dict[str, Any] = {}
    if config.route_hist_path:
        probs, route_hist_metadata = _load_route_hist(
            config.route_hist_path, config.num_experts
        )
        # Gumbel-top-k: argtop-k of log(p) + Gumbel noise draws k experts
        # without replacement with probability proportional to p at every
        # stage, reproducing the measured marginal popularity.  Zero-count
        # experts get -inf via the clamp's floor and are never selected.
        log_probs = (
            torch.log(probs.clamp_min(1e-300))
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
        )
        uniform = torch.rand(
            config.tokens_per_rank,
            config.num_experts,
            generator=generator,
            device=device,
        )
        gumbel = -torch.log(-torch.log(uniform.clamp(min=1e-20, max=1.0 - 1e-7)))
        router_logits = log_probs + gumbel
    else:
        router_logits = torch.randn(
            config.tokens_per_rank,
            config.num_experts,
            generator=generator,
            device=device,
        )
    topk_values, topk_experts = torch.topk(router_logits, config.topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    del router_logits, topk_values

    hidden = torch.randn(
        config.tokens_per_rank,
        config.hidden_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    metadata = _local_route_stats(
        topk_experts,
        router_weights,
        local_experts=config.local_experts,
        world_size=config.world_size,
    )
    metadata.update(
        {
            "rank": rank,
            "seed": config.seed_base + rank,
            "hidden_dtype": str(hidden.dtype),
            "hidden_shape": list(hidden.shape),
            "expert_ids_dtype": str(topk_experts.dtype),
            "router_weights_dtype": str(router_weights.dtype),
            "route_family": "measured_hist" if config.route_hist_path else "iid_normal",
        }
    )
    if route_hist_metadata:
        metadata.update(route_hist_metadata)
    return (
        hidden,
        topk_experts.to(torch.int32),
        router_weights,
        generator,
        metadata,
    )


def _normal_bf16(
    shape: tuple[int, ...],
    *,
    std: float,
    generator: torch.Generator,
    device: torch.device | str,
    chunk_elements: int,
) -> torch.Tensor:
    """Fill a BF16 tensor from chunked normal samples."""

    total = math.prod(shape)
    output = torch.empty(total, dtype=torch.bfloat16, device=device)
    for begin in range(0, total, chunk_elements):
        end = min(begin + chunk_elements, total)
        sample = torch.randn(
            end - begin,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        sample.mul_(std)
        output[begin:end].copy_(sample)
    return output.reshape(shape)


def generate_mok_synthetic_inputs(
    rank: int,
    device: torch.device | str,
    config: MoKSyntheticConfig,
) -> dict[str, Any]:
    """Generate all setup-time tensors shared by production and PF4H.

    MoK uses BF16 routed weights (and prequantizes its MXFP8 variant). This
    helper therefore returns BF16 logical weights from the same rank-local
    normal stream. The executor quantizes them per 128x128 and applies AITER's
    production weight shuffle before either measured arm runs.
    """

    hidden, expert_ids, router_weights, generator, route_metadata = (
        generate_router_and_hidden(rank, device, config)
    )
    local_experts = config.local_experts
    w13_bf16 = _normal_bf16(
        (local_experts, 2 * config.intermediate_dim, config.hidden_dim),
        std=config.hidden_dim**-0.5,
        generator=generator,
        device=device,
        chunk_elements=config.weight_chunk_elements,
    )
    w2_bf16 = _normal_bf16(
        (local_experts, config.hidden_dim, config.intermediate_dim),
        std=config.intermediate_dim**-0.5,
        generator=generator,
        device=device,
        chunk_elements=config.weight_chunk_elements,
    )
    metadata = {
        "generator": "torch.Generator(device).manual_seed(seed_base + rank)",
        "generation_order": [
            "router_logits",
            "hidden",
            "routed_w13",
            "routed_w2",
        ],
        "config": asdict(config),
        "route": route_metadata,
        "weights": {
            "w13_logical_shape": list(w13_bf16.shape),
            "w2_logical_shape": list(w2_bf16.shape),
            "logical_dtype": str(w13_bf16.dtype),
            "quantization": "executor: AITER per-128x128 FP8 + shuffle_weight",
            "setup_only": True,
        },
    }
    return {
        "hidden": hidden,
        "topk_experts": expert_ids,
        "router_weights": router_weights,
        "w13_bf16": w13_bf16,
        "w2_bf16": w2_bf16,
        "metadata": metadata,
    }
