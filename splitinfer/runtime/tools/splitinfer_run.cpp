/* splitinfer/runtime/tools/splitinfer_run.cpp
 * CLI entry point for SplitInfer.
 *
 * Usage: splitinfer_run <manifest.json>
 *
 * Loads the partition manifest, runs one inference pass with stub
 * (no-op) GPU and FPGA executors, and prints timing telemetry.
 * Intended for smoke-testing the runtime before real TensorRT /
 * EdgeCoh executors are integrated.
 */

#include "splitinfer/manifest.h"
#include "splitinfer/pipeline.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// ─── Stub executors (no-op) ───────────────────────────────────────────────────

struct StubGpuExecutor : splitinfer::GpuExecutorBase {
    bool execute(const std::string& layer_name,
                 const void*, size_t input_bytes,
                 void*, size_t output_bytes) override {
        std::printf("  [GPU] %-30s  in=%zu B  out=%zu B\n",
                    layer_name.c_str(), input_bytes, output_bytes);
        return true;
    }
};

struct StubFpgaExecutor : splitinfer::FpgaExecutorBase {
    bool execute(const std::string& layer_name,
                 uint32_t nmc_op,
                 const void*, size_t input_bytes,
                 void*, size_t output_bytes) override {
        std::printf("  [FPGA nmc_op=%u] %-24s  in=%zu B  out=%zu B\n",
                    nmc_op, layer_name.c_str(), input_bytes, output_bytes);
        return true;
    }

    bool transfer_to_host(uint32_t fpga_addr, size_t len, void*) override {
        std::printf("  [XFER FPGA→HOST] addr=0x%08X  %zu B\n", fpga_addr, len);
        return true;
    }

    bool transfer_to_fpga(const void*, size_t len, uint32_t fpga_addr) override {
        std::printf("  [XFER HOST→FPGA] addr=0x%08X  %zu B\n", fpga_addr, len);
        return true;
    }
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

static const char* device_str(splitinfer::Device d) {
    return d == splitinfer::Device::GPU ? "GPU" : "FPGA";
}

static void print_summary(const splitinfer::Manifest& m) {
    const auto& s = m.summary;
    std::printf("\n=== Manifest Summary ===\n");
    std::printf("  Version             : %s\n", m.version.c_str());
    std::printf("  Total layers        : %d  (GPU=%d  FPGA=%d)\n",
                s.total_layers, s.gpu_layers, s.fpga_layers);
    std::printf("  Estimated latency   : %.2f ms\n", s.estimated_latency_ms);
    std::printf("  GPU memory          : %lld B\n", (long long)s.gpu_memory_bytes);
    std::printf("  FPGA memory         : %lld B\n", (long long)s.fpga_memory_bytes);
    std::printf("  Device transfers    : %d\n", s.num_transfers);

    std::printf("\nLayer schedule:\n");
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const auto& l = m.layers[i];
        std::printf("  [%2zu] %-30s  %-4s  op=%-12s  w=%lld B  out=%lld B\n",
                    i, l.name.c_str(), device_str(l.device), l.op_type.c_str(),
                    (long long)l.weight_bytes, (long long)l.output_tensor_bytes);
    }
    if (!m.transfers.empty()) {
        std::printf("\nTransfers:\n");
        for (const auto& t : m.transfers) {
            std::printf("  after=%-20s before=%-20s %s→%s  %lld B\n",
                        t.after_layer.c_str(), t.before_layer.c_str(),
                        device_str(t.from_device), device_str(t.to_device),
                        (long long)t.tensor_bytes);
        }
    }
}

static void print_telemetry(const splitinfer::TelemetryStats& s) {
    std::printf("\n=== Telemetry ===\n");
    std::printf("  Inference runs      : %llu\n", (unsigned long long)s.inference_count);
    std::printf("  Total latency       : %.3f ms\n", s.total_latency_ms);
    std::printf("  Avg latency         : %.3f ms\n", s.avg_latency_ms);
    std::printf("  Min latency         : %.3f ms\n",
                s.min_latency_ms < 1e18 ? s.min_latency_ms : 0.0);
    std::printf("  Max latency         : %.3f ms\n", s.max_latency_ms);
    std::printf("  GPU time            : %.3f ms\n", s.gpu_time_ms);
    std::printf("  FPGA time           : %.3f ms\n", s.fpga_time_ms);
    std::printf("  Bytes transferred   : %lld B\n", (long long)s.bytes_transferred);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <manifest.json>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string path = argv[1];
    splitinfer::Manifest manifest;

    if (!splitinfer::load_manifest(path, manifest)) {
        std::fprintf(stderr, "Error: failed to load manifest from '%s'\n", path.c_str());
        return EXIT_FAILURE;
    }

    print_summary(manifest);

    StubGpuExecutor  gpu;
    StubFpgaExecutor fpga;

    splitinfer::Pipeline pipeline(manifest, gpu, fpga);

    std::printf("\nRunning inference (stub executors)...\n");
    bool ok = pipeline.run();
    if (!ok) {
        std::fprintf(stderr, "Warning: one or more layers reported an error.\n");
    }

    print_telemetry(pipeline.get_telemetry().get_stats());

    std::printf("\nDone. Status: %s\n", ok ? "OK" : "ERROR");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
