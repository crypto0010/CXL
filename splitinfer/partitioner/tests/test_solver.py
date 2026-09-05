"""Solver tests.  The critical one is optimality against brute force —
it is what distinguishes a dynamic program from the greedy pass it replaces."""
import itertools
import math

import pytest

from partitioner.cost_model import COMPUTE_LOCATION, PLACEMENTS, CostModel, HardwareParams
from partitioner.graph import LayerInfo
from partitioner.solver import PartitionResult, partition_model

MB = 1024 * 1024


def _hw(**kw):
    return HardwareParams.measured_uart_substrate(**kw)


def _fc(name, k, n, batch=1):
    return LayerInfo(name, "MatMul", ["x", "W"], ["y"], k * n * 4, batch * n * 4,
                     input_activation_bytes=batch * k * 4,
                     weight_shapes=[(k, n)], output_shape=(batch, n))


def _relu(name, width, batch=1):
    return LayerInfo(name, "Relu", ["x"], ["y"], 0, batch * width * 4,
                     input_activation_bytes=batch * width * 4, output_shape=(batch, width))


def _chain(widths, batch=1):
    layers, prev = [], widths[0]
    for i, w in enumerate(widths[1:]):
        layers.append(_fc(f"fc{i}", prev, w, batch))
        layers.append(_relu(f"relu{i}", w, batch))
        prev = w
    return layers


def _brute_force(layers, cm, resident):
    """Exhaustive minimum over every feasible assignment (small N only)."""
    hw = cm.hw
    best, best_assign = math.inf, None
    for assign in itertools.product(PLACEMENTS, repeat=len(layers)):
        host = sum(l.weight_bytes for l, p in zip(layers, assign) if p == "gpu")
        if host > hw.host_weight_budget_bytes:
            continue
        ddr2 = [l.weight_bytes for l, p in zip(layers, assign) if p != "gpu"]
        if resident and sum(ddr2) > hw.fpga_ddr2_capacity_bytes:
            continue
        ok = all((p == "gpu") or (p == "pool" and cm.pool_feasible(l))
                 or (p == "fpga" and cm.fpga_feasible(l)) for l, p in zip(layers, assign))
        if not ok:
            continue
        cost = 0.0
        for i, (l, p) in enumerate(zip(layers, assign)):
            cost += cm.placement_cost_ms(l, p)
            if i > 0:
                cost += cm.boundary_cost_ms(layers[i - 1], assign[i - 1], p)
        if cost < best:
            best, best_assign = cost, assign
    return best, best_assign


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("link_bw", [11_520.0, 1e6, 1e8])
def test_dp_matches_brute_force(resident, link_bw):
    layers = _chain([256, 512, 1024, 512, 8])          # 8 layers -> 3^8 = 6561 combos
    hw = _hw(link_bw_bytes_per_s=link_bw, fpga_weights_resident=resident,
             host_weight_budget_bytes=2 * MB)            # tight: forces some off-host
    cm = CostModel(hw)
    got = partition_model(layers, cm, bucket_bytes=64 * 1024)
    want_cost, want_assign = _brute_force(layers, cm, resident)
    assert got.total_latency_ms == pytest.approx(want_cost, rel=1e-9)


def test_returns_result_with_all_fields():
    r = partition_model(_chain([128, 256, 8]), CostModel(_hw()))
    assert isinstance(r, PartitionResult)
    assert len(r.assignments) == 4
    assert all(a in PLACEMENTS for a in r.assignments)
    assert len(r.decisions) == 4
    assert set(r.breakdown_ms) >= {"gpu_compute", "fpga_compute", "pool_fault"}


def test_greedy_bug_regression_weighted_layer_not_forced_to_fpga():
    """D4 regression: v1 sent every weighted layer to the FPGA unconditionally.
    With a fast GPU, a slow link, and plenty of host memory, the right answer
    is all-GPU."""
    layers = _chain([256, 512, 512, 8])
    r = partition_model(layers, CostModel(_hw(host_weight_budget_bytes=1024 * MB)))
    assert all(a == "gpu" for a in r.assignments), r.assignments


def test_host_budget_forces_layers_off_host():
    layers = _chain([1024, 1024, 1024, 1024, 8])    # 3 x 4 MiB + small
    hw = _hw(host_weight_budget_bytes=5 * MB)        # can hold one big layer
    r = partition_model(layers, CostModel(hw), bucket_bytes=256 * 1024)
    assert r.gpu_memory_bytes <= hw.host_weight_budget_bytes
    assert any(a != "gpu" for a in r.assignments)


def test_ddr2_capacity_respected_when_resident():
    layers = _chain([2048, 2048, 2048, 2048, 8])    # 3 x 16 MiB
    hw = _hw(host_weight_budget_bytes=0, fpga_ddr2_capacity_bytes=40 * MB,
             fpga_weights_resident=True)
    with pytest.raises(RuntimeError):
        partition_model(layers, CostModel(hw), bucket_bytes=MB)   # 48 MiB > 40 MiB, host has 0


def test_streaming_peak_is_max_layer_not_sum():
    layers = _chain([2048, 2048, 2048, 2048, 8])
    hw = _hw(host_weight_budget_bytes=0, fpga_weights_resident=False)
    r = partition_model(layers, CostModel(hw), bucket_bytes=MB)
    assert r.fpga_memory_bytes == 2048 * 2048 * 4
    assert r.fpga_weight_traffic_bytes >= 3 * 2048 * 2048 * 4 or r.pool_fault_bytes > 0


def test_unsupported_op_never_placed_on_fpga():
    layers = _chain([128, 256, 8])
    layers.insert(2, LayerInfo("cat", "Concat", ["a", "b"], ["c"], 0, 1024,
                               input_activation_bytes=1024, output_shape=(1, 256)))
    r = partition_model(layers, CostModel(_hw(host_weight_budget_bytes=0)))
    assert r.assignments[2] != "fpga"


def test_transfers_counted_at_compute_location_changes():
    layers = _chain([128, 256, 8])
    r = partition_model(layers, CostModel(_hw()))
    changes = sum(1 for i in range(1, len(r.assignments))
                  if COMPUTE_LOCATION[r.assignments[i]] != COMPUTE_LOCATION[r.assignments[i - 1]])
    assert r.num_transfers == changes


def test_allowed_subset_restricts_placements():
    layers = _chain([128, 256, 8])
    r = partition_model(layers, CostModel(_hw(host_weight_budget_bytes=0)), allowed=("pool",))
    assert all(a == "pool" for a in r.assignments)


def test_legacy_streaming_kwarg_still_works():
    layers = _chain([128, 256, 8])
    a = partition_model(layers, CostModel(_hw()), streaming=True)
    b = partition_model(layers, CostModel(_hw()), streaming=False)
    assert isinstance(a, PartitionResult) and isinstance(b, PartitionResult)
