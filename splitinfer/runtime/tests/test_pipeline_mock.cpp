/* splitinfer/runtime/tests/test_pipeline_mock.cpp
 * Tests for the pipeline orchestrator using mock executors.
 */

#include "splitinfer/manifest.h"
#include "splitinfer/pipeline.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

// ─── Mock executors ──────────────────────────────────────────────────────────

struct MockGpuExecutor : splitinfer::GpuExecutorBase {
    std::vector<std::string> executed_layers;

    bool execute(const std::string& layer_name,
                 const void*, size_t,
                 void*, size_t) override {
        executed_layers.push_back(layer_name);
        return true;
    }
};

struct MockFpgaExecutor : splitinfer::FpgaExecutorBase {
    std::vector<std::string> executed_layers;
    int transfer_to_fpga_calls = 0;
    int transfer_to_host_calls = 0;

    bool execute(const std::string& layer_name,
                 uint32_t /*nmc_op*/,
                 const void*, size_t,
                 void*, size_t) override {
        executed_layers.push_back(layer_name);
        return true;
    }

    bool transfer_to_host(uint32_t, size_t, void*) override {
        ++transfer_to_host_calls;
        return true;
    }

    bool transfer_to_fpga(const void*, size_t, uint32_t) override {
        ++transfer_to_fpga_calls;
        return true;
    }
};

// ─── Test helpers ─────────────────────────────────────────────────────────────

static int pass_count = 0;
static int fail_count = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", #expr, __FILE__, __LINE__); \
        ++fail_count; \
    } else { \
        ++pass_count; \
    } \
} while (0)

// ─── Embedded test manifest ───────────────────────────────────────────────────

static const char* TEST_JSON = R"({
  "version": "1.0",
  "layers": [
    {
      "name": "conv1",
      "op_type": "Conv",
      "device": "GPU",
      "weight_bytes": 4096,
      "output_tensor_bytes": 8192,
      "inputs": ["input_0"],
      "outputs": ["conv1_out"]
    },
    {
      "name": "embed1",
      "op_type": "Gather",
      "device": "FPGA",
      "weight_bytes": 16384,
      "output_tensor_bytes": 2048,
      "inputs": ["conv1_out"],
      "outputs": ["embed1_out"]
    },
    {
      "name": "relu1",
      "op_type": "Relu",
      "device": "GPU",
      "weight_bytes": 0,
      "output_tensor_bytes": 2048,
      "inputs": ["embed1_out"],
      "outputs": ["output_0"]
    }
  ],
  "transfers": [
    {
      "after_layer": "conv1",
      "before_layer": "embed1",
      "from_device": "GPU",
      "to_device": "FPGA",
      "tensor_names": ["conv1_out"],
      "tensor_bytes": 8192
    }
  ],
  "summary": {
    "total_layers": 3,
    "gpu_layers": 2,
    "fpga_layers": 1,
    "estimated_latency_ms": 12.5,
    "gpu_memory_bytes": 12288,
    "fpga_memory_bytes": 18432,
    "num_transfers": 1
  }
})";

int main() {
    splitinfer::Manifest manifest;
    bool parse_ok = splitinfer::parse_manifest(std::string(TEST_JSON), manifest);
    CHECK(parse_ok);
    if (!parse_ok) {
        std::printf("Cannot continue: manifest parse failed\n");
        return 1;
    }

    MockGpuExecutor  gpu;
    MockFpgaExecutor fpga;

    splitinfer::Pipeline pipeline(manifest, gpu, fpga);
    bool run_ok = pipeline.run();
    CHECK(run_ok);

    // ── Dispatch correctness ──────────────────────────────────────────────────

    // GPU executor should have seen conv1 and relu1
    CHECK(gpu.executed_layers.size() == 2u);
    if (gpu.executed_layers.size() == 2u) {
        CHECK(gpu.executed_layers[0] == "conv1");
        CHECK(gpu.executed_layers[1] == "relu1");
    }

    // FPGA executor should have seen embed1
    CHECK(fpga.executed_layers.size() == 1u);
    if (!fpga.executed_layers.empty()) {
        CHECK(fpga.executed_layers[0] == "embed1");
    }

    // Transfer: GPU→FPGA should have triggered transfer_to_fpga once
    CHECK(fpga.transfer_to_fpga_calls == 1);
    // No FPGA→GPU transfer in this manifest
    CHECK(fpga.transfer_to_host_calls == 0);

    // ── Telemetry ─────────────────────────────────────────────────────────────

    auto stats = pipeline.get_telemetry().get_stats();
    CHECK(stats.inference_count == 1u);
    CHECK(stats.total_latency_ms >= 0.0);
    CHECK(stats.gpu_time_ms >= 0.0);
    CHECK(stats.fpga_time_ms >= 0.0);
    CHECK(stats.bytes_transferred == 8192);

    // ── Second run accumulates ────────────────────────────────────────────────

    pipeline.run();
    auto stats2 = pipeline.get_telemetry().get_stats();
    CHECK(stats2.inference_count == 2u);
    CHECK(stats2.bytes_transferred == 8192 * 2);

    // ── Reset ─────────────────────────────────────────────────────────────────

    pipeline.get_telemetry().reset();
    auto stats3 = pipeline.get_telemetry().get_stats();
    CHECK(stats3.inference_count == 0u);
    CHECK(stats3.bytes_transferred == 0);

    // ── Prefetch accounting (Task #14) ────────────────────────────────────────
    //
    // The test manifest has exactly 1 GPU→FPGA transfer (after conv1).
    // With prefetch ENABLED (default), each pass should record:
    //   prefetch_attempts += 1
    //   prefetch_hits     += 1
    //   prefetch_hit_rate  = 100%
    //
    // With prefetch DISABLED, the same transfer is still issued and counted
    // as an attempt, but recorded as a miss:
    //   prefetch_attempts += 1
    //   prefetch_hits     += 0
    //   prefetch_hit_rate  = 0%
    //
    // We can't directly compare wall-clock latencies in a unit test (mocks
    // don't sleep), but we can verify the counter logic is correct.

    pipeline.get_telemetry().reset();
    CHECK(pipeline.is_prefetch_enabled() == true);  // default
    pipeline.run();
    auto stats_pre_on = pipeline.get_telemetry().get_stats();
    CHECK(stats_pre_on.prefetch_attempts == 1u);
    CHECK(stats_pre_on.prefetch_hits     == 1u);
    CHECK(stats_pre_on.prefetch_hit_rate == 1.0);

    pipeline.set_prefetch_enabled(false);
    CHECK(pipeline.is_prefetch_enabled() == false);
    pipeline.get_telemetry().reset();
    pipeline.run();
    auto stats_pre_off = pipeline.get_telemetry().get_stats();
    CHECK(stats_pre_off.prefetch_attempts == 1u);
    CHECK(stats_pre_off.prefetch_hits     == 0u);
    CHECK(stats_pre_off.prefetch_hit_rate == 0.0);

    // ── New per-bucket time accounting (Task #17) ─────────────────────────────
    // The new TelemetryStats fields gpu_time_ms, fpga_time_ms, transfer_time_ms,
    // sync_time_ms should all be non-negative and finite.  Mocks return
    // immediately so all values should be very small (microseconds at most).

    pipeline.set_prefetch_enabled(true);  // restore default for any future runs
    pipeline.get_telemetry().reset();
    pipeline.run();
    auto stats_breakdown = pipeline.get_telemetry().get_stats();
    CHECK(stats_breakdown.gpu_time_ms      >= 0.0);
    CHECK(stats_breakdown.fpga_time_ms     >= 0.0);
    CHECK(stats_breakdown.transfer_time_ms >= 0.0);
    CHECK(stats_breakdown.sync_time_ms     >= 0.0);

    std::printf("test_pipeline: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
