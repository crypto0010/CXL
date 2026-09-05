"""Exact dynamic-programming placement over three placements.

The v1 "dynamic programming solver" was a single greedy pass whose second
branch sent every weighted layer to the FPGA unconditionally — which is why
every published manifest had 100% FPGA placement and zero transfers.

This is a genuine DP.  State is (layer, placement, host-memory-used,
fpga-resident-used); the transition adds the placement's own cost plus the
activation crossing when compute location changes.  Memory usage is tracked
in fixed-size buckets so the two capacity constraints are exact to bucket
granularity, and the argmin table gives an exact backtrack.

Complexity per layer: |P|^2 x H x F with |P| = 3, H = host buckets, F = FPGA
buckets (1 when weights are streamed).  Vectorised with numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from partitioner.cost_model import COMPUTE_LOCATION, PLACEMENTS, CostModel
from partitioner.graph import LayerInfo

INF = float("inf")


@dataclass
class LayerDecision:
    name: str
    op_type: str
    placement: str
    cost_ms: float                       # placement cost for this layer
    boundary_ms: float                   # activation crossing paid before this layer
    candidates_ms: dict[str, float]      # cost under every feasible placement
    arithmetic_intensity: float
    link_crossover_bytes_per_s: float | None


@dataclass
class PartitionResult:
    assignments: list[str]
    total_latency_ms: float
    gpu_memory_bytes: int                # host-resident weight bytes
    fpga_memory_bytes: int               # peak DDR2 resident footprint
    num_transfers: int                   # compute-location changes
    fpga_weight_traffic_bytes: int = 0   # bytes streamed to DDR2 per inference
    pool_fault_bytes: int = 0            # bytes faulted host-ward per inference
    activation_transfer_bytes: int = 0
    decisions: list[LayerDecision] = field(default_factory=list)
    breakdown_ms: dict[str, float] = field(default_factory=dict)

    @property
    def placement_counts(self) -> dict[str, int]:
        return {p: self.assignments.count(p) for p in PLACEMENTS}


def _buckets(nbytes: int, bucket: int) -> int:
    return int(math.ceil(nbytes / bucket)) if nbytes > 0 else 0


def partition_model(layers: list[LayerInfo], cost_model: CostModel,
                    streaming: bool | None = None,
                    allowed: tuple[str, ...] = PLACEMENTS,
                    bucket_bytes: int | None = None,
                    max_states: int = 1 << 18) -> PartitionResult:
    """Choose a placement for every layer minimising modelled latency.

    ``streaming`` is a backwards-compatible alias: ``False`` sets
    ``hw.fpga_weights_resident = True`` for this call.  The default follows
    the cost model's hardware parameters.

    Memory usage is tracked in buckets.  With ``bucket_bytes=None`` the
    bucket size is chosen per dimension so that the total state count stays
    under ``max_states`` — fine granularity when only one capacity is
    active, coarser when both are.  Rounding is conservative (ceil), so a
    plan the solver accepts always fits.
    """
    hw = cost_model.hw
    resident = hw.fpga_weights_resident if streaming is None else (not streaming)
    saved = hw.fpga_weights_resident
    hw.fpga_weights_resident = resident
    try:
        return _solve(layers, cost_model, allowed, bucket_bytes, resident, max_states)
    finally:
        hw.fpga_weights_resident = saved


def _choose_buckets(host_cap: int, fpga_cap: int, resident: bool,
                    bucket: int | None, max_states: int) -> tuple[int, int]:
    """Pick (host_bucket, fpga_bucket) so that H*F <= max_states."""
    if bucket is not None:
        return bucket, bucket
    page = 4096
    if not resident or fpga_cap <= 0:
        h = max(page, -(-host_cap // max_states))         # ceil division
        return h, page
    if host_cap <= 0:
        return page, max(page, -(-fpga_cap // max_states))
    side = int(math.sqrt(max_states))
    return (max(page, -(-host_cap // side)), max(page, -(-fpga_cap // side)))


def _solve(layers, cm: CostModel, allowed, bucket, resident, max_states) -> PartitionResult:
    hw = cm.hw
    P = [p for p in PLACEMENTS if p in allowed]
    n = len(layers)
    hb, fb = _choose_buckets(hw.host_weight_budget_bytes, hw.fpga_ddr2_capacity_bytes,
                             resident, bucket, max_states)
    H = _buckets(hw.host_weight_budget_bytes, hb) + 1
    F = (_buckets(hw.fpga_ddr2_capacity_bytes, fb) + 1) if resident else 1

    # Per-layer, per-placement cost and feasibility.
    place_cost = np.full((n, len(P)), INF)
    for i, layer in enumerate(layers):
        for j, p in enumerate(P):
            ok = (p == "gpu"
                  or (p == "pool" and cm.pool_feasible(layer))
                  or (p == "fpga" and cm.fpga_feasible(layer)))
            if ok:
                place_cost[i, j] = cm.placement_cost_ms(layer, p)

    # dp[j, h, f]: best cost with layer i on P[j], host buckets h, fpga buckets f.
    dp = np.full((len(P), H, F), INF)
    back = np.zeros((n, len(P), H, F), dtype=np.int8)   # index of previous placement

    def usage_shift(j: int, wbytes: int):
        """How placement P[j] moves the (h, f) usage for a layer of wbytes."""
        p = P[j]
        if p == "gpu":
            return _buckets(wbytes, hb), 0
        return 0, (_buckets(wbytes, fb) if resident else 0)   # pool and fpga both occupy DDR2

    # Layer 0: no boundary.
    for j in range(len(P)):
        if place_cost[0, j] == INF:
            continue
        dh, df = usage_shift(j, layers[0].weight_bytes)
        if dh < H and df < F:
            dp[j, dh, df] = place_cost[0, j]
            back[0, j, dh, df] = j

    for i in range(1, n):
        layer, prev = layers[i], layers[i - 1]
        new = np.full_like(dp, INF)
        for j, p in enumerate(P):
            if place_cost[i, j] == INF:
                continue
            dh, df = usage_shift(j, layer.weight_bytes)
            if dh >= H or df >= F:
                continue
            # Candidate from every previous placement, shifted by usage.
            best = np.full((H - dh, F - df), INF)
            best_from = np.zeros((H - dh, F - df), dtype=np.int8)
            for jp, pp in enumerate(P):
                src = dp[jp, :H - dh, :F - df]
                cand = src + place_cost[i, j] + cm.boundary_cost_ms(prev, pp, p)
                better = cand < best
                best = np.where(better, cand, best)
                best_from = np.where(better, jp, best_from)
            new[j, dh:, df:] = best
            back[i, j, dh:, df:] = best_from
        dp = new

    # Terminal: any usage.
    j_end, h_end, f_end = np.unravel_index(int(np.argmin(dp)), dp.shape)
    total = float(dp[j_end, h_end, f_end])
    if not math.isfinite(total):
        raise RuntimeError("No feasible placement: check host budget, DDR2 capacity, "
                           "and FPGA-supported op set.")

    # Backtrack.
    assign_idx = [0] * n
    j, h, f = int(j_end), int(h_end), int(f_end)
    for i in range(n - 1, -1, -1):
        assign_idx[i] = j
        jp = int(back[i, j, h, f])
        dh, df = usage_shift(j, layers[i].weight_bytes)
        h, f, j = h - dh, f - df, jp
    assignments = [P[j] for j in assign_idx]

    return _summarise(layers, assignments, cm, place_cost, P, total, resident)


def _summarise(layers, assignments, cm: CostModel, place_cost, P, total, resident):
    decisions: list[LayerDecision] = []
    breakdown = {"gpu_compute": 0.0, "fpga_compute": 0.0, "pool_fault": 0.0,
                 "weight_stream": 0.0, "dispatch": 0.0, "activation_xfer": 0.0}
    host_bytes = fpga_traffic = pool_bytes = act_bytes = 0
    transfers = 0
    ddr2_layers: list[int] = []
    for i, (layer, p) in enumerate(zip(layers, assignments)):
        boundary = 0.0
        if i > 0:
            boundary = cm.boundary_cost_ms(layers[i - 1], assignments[i - 1], p)
            if COMPUTE_LOCATION[assignments[i - 1]] != COMPUTE_LOCATION[p]:
                transfers += 1
                act_bytes += layers[i - 1].output_tensor_bytes
        breakdown["activation_xfer"] += boundary
        if p == "gpu":
            host_bytes += layer.weight_bytes
            breakdown["gpu_compute"] += cm.gpu_compute_ms(layer)
        elif p == "pool":
            pool_bytes += layer.weight_bytes
            ddr2_layers.append(layer.weight_bytes)
            breakdown["pool_fault"] += cm.pool_weight_fault_ms(layer)
            breakdown["gpu_compute"] += cm.gpu_compute_ms(layer)
        else:
            ddr2_layers.append(layer.weight_bytes)
            breakdown["weight_stream"] += cm.nmc_weight_stream_ms(layer)
            breakdown["dispatch"] += cm.nmc_dispatch_ms()
            breakdown["fpga_compute"] += cm.fpga_compute_ms(layer)
            if not resident:
                fpga_traffic += layer.weight_bytes
        decisions.append(LayerDecision(
            name=layer.name, op_type=layer.op_type, placement=p,
            cost_ms=float(place_cost[i, P.index(p)]), boundary_ms=boundary,
            candidates_ms={q: float(place_cost[i, k]) for k, q in enumerate(P)
                           if math.isfinite(place_cost[i, k])},
            arithmetic_intensity=cm.arithmetic_intensity(layer),
            link_crossover_bytes_per_s=cm.link_crossover_bytes_per_s(layer),
        ))
    fpga_mem = (sum(ddr2_layers) if resident else max(ddr2_layers, default=0))
    return PartitionResult(
        assignments=assignments, total_latency_ms=total,
        gpu_memory_bytes=host_bytes, fpga_memory_bytes=fpga_mem,
        num_transfers=transfers, fpga_weight_traffic_bytes=fpga_traffic,
        pool_fault_bytes=pool_bytes, activation_transfer_bytes=act_bytes,
        decisions=decisions, breakdown_ms=breakdown,
    )
