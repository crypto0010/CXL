/* splitinfer/runtime/tools/splitinfer_run.cpp
 * CLI entry point for SplitInfer.
 *
 * Usage: splitinfer_run <manifest.json> [--real]
 *
 * By default uses stub (no-op) executors for smoke testing.
 * With --real, instantiates real GpuExecutor and FpgaExecutor.
 *
 * Ablation environment variables (for E4 experiment):
 *   SPLITINFER_NO_NMC=1       Force all FPGA layers to GPU
 *   SPLITINFER_NO_PREFETCH=1  Disable prefetch (not yet implemented)
 *   SPLITINFER_NO_PIPELINE=1  Disable pipelining (not yet implemented)
 */

#include "splitinfer/manifest.h"
#include "splitinfer/pipeline.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/// Apply SPLITINFER_NO_NMC: force all FPGA layers to GPU.
static void apply_ablation_no_nmc(splitinfer::Manifest& m) {
    int moved = 0;
    for (auto& layer : m.layers) {
        if (layer.device == splitinfer::Device::FPGA) {
            layer.device = splitinfer::Device::GPU;
            moved++;
        }
    }
    // Clear transfers since everything is on GPU now.
    m.transfers.clear();
    m.summary.gpu_layers += moved;
    m.summary.fpga_layers = 0;
    m.summary.num_transfers = 0;
    std::printf("  [ablation] NO_NMC: moved %d layers from FPGA to GPU\n", moved);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <manifest.json> [--real]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string path = argv[1];
    bool use_real = false;

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--real") == 0) {
            use_real = true;
        }
    }

    splitinfer::Manifest manifest;

    if (!splitinfer::load_manifest(path, manifest)) {
        std::fprintf(stderr, "Error: failed to load manifest from '%s'\n", path.c_str());
        return EXIT_FAILURE;
    }

    // Check ablation environment variables.
    if (std::getenv("SPLITINFER_NO_NMC")) {
        apply_ablation_no_nmc(manifest);
    }
    if (std::getenv("SPLITINFER_NO_PREFETCH")) {
        std::printf("  [ablation] NO_PREFETCH: noted (prefetch not yet implemented)\n");
    }
    if (std::getenv("SPLITINFER_NO_PIPELINE")) {
        std::printf("  [ablation] NO_PIPELINE: noted (double-buffering not yet implemented)\n");
    }

    print_summary(manifest);

    bool ok;

    if (use_real) {
        std::printf("\nUsing real executors (GpuExecutor + FpgaExecutor)...\n");
        splitinfer::GpuExecutor  gpu;
        splitinfer::FpgaExecutor fpga;

        splitinfer::Pipeline pipeline(manifest, gpu, fpga);

        std::printf("\nRunning inference...\n");
        ok = pipeline.run();

        print_telemetry(pipeline.get_telemetry().get_stats());
    } else {
        std::printf("\nRunning inference (stub executors)...\n");
        StubGpuExecutor  gpu;
        StubFpgaExecutor fpga;

        splitinfer::Pipeline pipeline(manifest, gpu, fpga);
        ok = pipeline.run();

        print_telemetry(pipeline.get_telemetry().get_stats());
    }

    if (!ok) {
        std::fprintf(stderr, "Warning: one or more layers reported an error.\n");
    }

    std::printf("\nDone. Status: %s\n", ok ? "OK" : "ERROR");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
