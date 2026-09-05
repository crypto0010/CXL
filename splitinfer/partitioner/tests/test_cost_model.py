import math
import pytest
from partitioner.graph import LayerInfo
from partitioner.cost_model import CostModel, HardwareParams, PLACEMENTS


def _hw(**kw):
    return HardwareParams.measured_uart_substrate(**kw)


def _fc(name="fc", k=1024, n=1024, batch=1, dtype_bytes=4):
    return LayerInfo(name, "MatMul", ["x", "W"], ["y"], k * n * dtype_bytes,
                     batch * n * dtype_bytes, input_activation_bytes=batch * k * dtype_bytes,
                     weight_shapes=[(k, n)], output_shape=(batch, n))


def test_measured_substrate_is_uart_not_usb_bulk():
    """D6 regression: v1 defaulted to 40 MB/s; the link is 11.5 KB/s."""
    hw = _hw()
    assert hw.link_bw_bytes_per_s == pytest.approx(11_520.0)
    assert hw.link_rtt_ms >= 1.0


def test_flops_from_real_shapes():
    cm = CostModel(_hw())
    assert cm.flops(_fc(k=1024, n=512, batch=4)) == 2 * 4 * 1024 * 512


def test_weight_streaming_is_priced():
    """D5 regression: streamed weights must cost link time."""
    cm = CostModel(_hw(fpga_weights_resident=False))
    layer = _fc(k=1024, n=1024)          # 4 MiB of FP32 weights
    t = cm.nmc_weight_stream_ms(layer)
    assert t >= (layer.weight_bytes / 11_520.0) * 1000.0   # at least the wire time


def test_resident_weights_are_free_to_stream():
    cm = CostModel(_hw(fpga_weights_resident=True))
    assert cm.nmc_weight_stream_ms(_fc()) == 0.0


def test_pool_fault_pays_one_rtt_per_page_batch():
    hw = _hw(pool_prefetch_pages=1)
    cm = CostModel(hw)
    layer = _fc(k=1024, n=1)             # 4 KiB = exactly one page
    t = cm.pool_weight_fault_ms(layer)
    assert t == pytest.approx((4096 / 11_520.0) * 1000.0 + hw.link_rtt_ms)


def test_prefetch_reduces_pool_round_trips():
    layer = _fc(k=1024, n=64)            # 256 KiB = 64 pages
    demand = CostModel(_hw(pool_prefetch_pages=1)).pool_weight_fault_ms(layer)
    batched = CostModel(_hw(pool_prefetch_pages=64)).pool_weight_fault_ms(layer)
    assert batched < demand


def test_boundary_cost_only_when_compute_location_changes():
    cm = CostModel(_hw())
    prev = _fc()
    assert cm.boundary_cost_ms(prev, "gpu", "pool") == 0.0     # both compute on GPU
    assert cm.boundary_cost_ms(prev, "gpu", "fpga") > 0.0
    assert cm.boundary_cost_ms(prev, "fpga", "pool") > 0.0


def test_every_placement_has_a_cost():
    cm = CostModel(_hw())
    for p in PLACEMENTS:
        assert cm.placement_cost_ms(_fc(), p) > 0


def test_unsupported_op_is_fpga_infeasible():
    cm = CostModel(_hw())
    assert cm.fpga_feasible(LayerInfo("c", "Concat", ["a"], ["b"], 0, 64)) is False
    assert cm.fpga_feasible(_fc()) is True


def test_over_capacity_layer_is_infeasible_on_fpga_and_pool():
    cm = CostModel(_hw())
    big = _fc(k=8192, n=8192)            # 256 MiB > 128 MiB DDR2
    assert not cm.fpga_feasible(big)
    assert not cm.pool_feasible(big)


def test_demand_paging_makes_pool_rtt_bound():
    """With one round trip per page, pool cost is dominated by RTT, not
    bandwidth: the crossover is +inf (fpga always wins) for a 1024-page layer."""
    cm = CostModel(_hw(fpga_weights_resident=True, pool_prefetch_pages=1))
    assert cm.link_crossover_bytes_per_s(_fc(k=1024, n=1024)) == math.inf


def test_crossover_bandwidth_is_finite_with_batched_prefetch():
    kw = dict(fpga_weights_resident=True, fpga_int8_gops=0.03, gpu_gflops=2000,
              pool_prefetch_pages=1024)
    cm = CostModel(_hw(**kw))
    layer = _fc(k=1024, n=1024)
    b = cm.link_crossover_bytes_per_s(layer)
    assert b is not None and math.isfinite(b) and b > 0
    slow = CostModel(_hw(link_bw_bytes_per_s=b / 10, **kw))
    fast = CostModel(_hw(link_bw_bytes_per_s=b * 10, **kw))
    assert slow.placement_cost_ms(layer, "fpga") < slow.placement_cost_ms(layer, "pool")
    assert fast.placement_cost_ms(layer, "pool") < fast.placement_cost_ms(layer, "fpga")
