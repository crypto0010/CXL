/* splitinfer/runtime/tools/splitinfer_run.cpp
 * CLI entry point for SplitInfer.
 *
 * Usage: splitinfer_run <manifest.json> [--real] [--warmup N] [--iterations N]
 *
 * By default uses stub (no-op) executors for smoke testing.
 * With --real, instantiates real GpuExecutor and FpgaExecutor.
 *
 * Measurement flags (for E2 performance sweeps):
 *   --warmup N      run N inferences and discard their latencies (default: 0)
 *   --iterations N  run N measured inferences (default: 1)
 *   --json          emit a machine-readable STATS_JSON line after the run
 *
 * When --iterations > 1, the program reports per-iteration statistics
 * (median, p5, p95, std) computed over the measured iterations only.
 * The STATS_JSON: prefix line at the end is consumed by e2_run.py.
 *
 * KNOWN LIMITATION (--iterations with --real):
 *   --iterations > 1 with --real is currently BROKEN due to an FPGA
 *   protocol desync that happens between successive Pipeline::run()
 *   calls in the same process.  After the first inference succeeds,
 *   subsequent iterations time out on every layer's ACK wait — the
 *   most likely cause is that the FPGA's edgecoh_controller is left
 *   in a non-idle state by the first call, and our host-side
 *   FpgaExecutor doesn't emit an inter-inference sync marker.
 *
 *   Workaround: for --real measurement, call splitinfer_run with
 *   --iterations 1 and spawn the process once per measurement from
 *   an outer loop (e2_run.py does this).  For stub executors the
 *   --iterations flag works correctly.
 *
 * Ablation environment variables (for E4 experiment):
 *   SPLITINFER_NO_NMC=1       Force all FPGA layers to GPU
 *   SPLITINFER_NO_PREFETCH=1  Disable prefetch
 *   SPLITINFER_NO_PIPELINE=1  Alias for NO_PREFETCH (spec uses both names)
 */

#include "splitinfer/manifest.h"
#include "splitinfer/pipeline.h"
#include "splitinfer/gpu_executor.h"
#include "splitinfer/fpga_executor.h"
#include "splitinfer/telemetry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ─── Stub executors (no-op) ───────────────────────────────────────────────────

struct StubGpuExecutor : splitinfer::GpuExecutorBase {
    bool verbose = true;
    bool execute(const std::string& layer_name,
                 const void*, size_t input_bytes,
                 void*, size_t output_bytes) override {
        if (verbose) {
            std::printf("  [GPU] %-30s  in=%zu B  out=%zu B\n",
                        layer_name.c_str(), input_bytes, output_bytes);
        }
        return true;
    }
};

struct StubFpgaExecutor : splitinfer::FpgaExecutorBase {
    bool verbose = true;
    bool execute(const std::string& layer_name,
                 uint32_t nmc_op,
                 const void*, size_t input_bytes,
                 void*, size_t output_bytes) override {
        if (verbose) {
            std::printf("  [FPGA nmc_op=%u] %-24s  in=%zu B  out=%zu B\n",
                        nmc_op, layer_name.c_str(), input_bytes, output_bytes);
        }
        return true;
    }
    bool transfer_to_host(uint32_t fpga_addr, size_t len, void*) override {
        if (verbose) {
            std::printf("  [XFER FPGA→HOST] addr=0x%08X  %zu B\n", fpga_addr, len);
        }
        return true;
    }
    bool transfer_to_fpga(const void*, size_t len, uint32_t fpga_addr) override {
        if (verbose) {
            std::printf("  [XFER HOST→FPGA] addr=0x%08X  %zu B\n", fpga_addr, len);
        }
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
}

// ─── Statistics over a vector of per-iteration latencies ─────────────────────
//
// We compute the statistics directly over a sorted copy instead of reusing
// Telemetry's running min/max/avg, because proper percentile estimation
// requires the full distribution.  For 30+ iterations this is O(n log n)
// and takes microseconds — negligible vs the workload.

struct IterStats {
    size_t count = 0;
    double mean  = 0.0;
    double std   = 0.0;
    double min   = 0.0;
    double max   = 0.0;
    double p5    = 0.0;
    double p50   = 0.0;
    double p95   = 0.0;
};

/// Linear interpolation percentile — matches numpy/Python statistics default.
/// For small sample sizes this is the most honest estimator since exact
/// percentiles require more samples than we typically have.
static double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (sorted.size() == 1) return sorted[0];
    double k = (static_cast<double>(sorted.size()) - 1.0) * (p / 100.0);
    size_t f = static_cast<size_t>(k);
    size_t c = std::min(f + 1, sorted.size() - 1);
    double frac = k - static_cast<double>(f);
    return sorted[f] + (sorted[c] - sorted[f]) * frac;
}

static IterStats compute_stats(const std::vector<double>& latencies_ms) {
    IterStats s;
    s.count = latencies_ms.size();
    if (s.count == 0) return s;

    s.mean = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
             static_cast<double>(s.count);

    // Sample standard deviation (N-1 denominator for unbiased estimate).
    if (s.count > 1) {
        double sq_sum = 0.0;
        for (double x : latencies_ms) {
            double d = x - s.mean;
            sq_sum += d * d;
        }
        s.std = std::sqrt(sq_sum / static_cast<double>(s.count - 1));
    }

    std::vector<double> sorted(latencies_ms);
    std::sort(sorted.begin(), sorted.end());
    s.min = sorted.front();
    s.max = sorted.back();
    s.p5  = percentile(sorted, 5.0);
    s.p50 = percentile(sorted, 50.0);
    s.p95 = percentile(sorted, 95.0);
    return s;
}

static void print_iter_stats(const IterStats& s, bool prefetch_on, int warmup) {
    std::printf("\n=== Per-Iteration Statistics ===\n");
    std::printf("  Warmup (discarded)  : %d iterations\n", warmup);
    std::printf("  Measured count      : %zu iterations\n", s.count);
    std::printf("  Mean latency        : %.3f ms\n", s.mean);
    std::printf("  Std  latency        : %.3f ms  (%.1f%% CV)\n",
                s.std, s.mean > 0.0 ? (s.std / s.mean) * 100.0 : 0.0);
    std::printf("  Min  latency        : %.3f ms\n", s.min);
    std::printf("  p5   latency        : %.3f ms\n", s.p5);
    std::printf("  p50  latency        : %.3f ms  (median)\n", s.p50);
    std::printf("  p95  latency        : %.3f ms\n", s.p95);
    std::printf("  Max  latency        : %.3f ms\n", s.max);
    std::printf("  Prefetch            : %s\n", prefetch_on ? "ON" : "OFF");
}

/// Emit a JSON stats block — consumed by e2_run.py and other scripts that
/// want structured data without scraping stdout.  Printed to stdout on its
/// own line prefixed with "STATS_JSON: " so consumers can grep for it.
static void print_stats_json(const IterStats& s, bool prefetch_on, int warmup,
                              double total_ms) {
    std::printf("STATS_JSON: {"
                "\"count\":%zu,"
                "\"mean_ms\":%.4f,"
                "\"std_ms\":%.4f,"
                "\"min_ms\":%.4f,"
                "\"p5_ms\":%.4f,"
                "\"p50_ms\":%.4f,"
                "\"p95_ms\":%.4f,"
                "\"max_ms\":%.4f,"
                "\"total_wall_ms\":%.3f,"
                "\"warmup\":%d,"
                "\"prefetch\":%s"
                "}\n",
                s.count, s.mean, s.std, s.min, s.p5, s.p50, s.p95, s.max,
                total_ms, warmup, prefetch_on ? "true" : "false");
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
    m.transfers.clear();
    m.summary.gpu_layers += moved;
    m.summary.fpga_layers = 0;
    m.summary.num_transfers = 0;
    std::printf("  [ablation] NO_NMC: moved %d layers from FPGA to GPU\n", moved);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s <manifest.json> [--real] [--warmup N] [--iterations N] [--json]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    const std::string path = argv[1];
    bool use_real = false;
    int  warmup     = 0;
    int  iterations = 1;
    bool emit_json  = false;

    // Simple arg parser (avoids dragging in a dependency).
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--real") == 0) {
            use_real = true;
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
            if (warmup < 0) warmup = 0;
        } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
            if (iterations < 1) iterations = 1;
        } else if (std::strcmp(argv[i], "--json") == 0) {
            emit_json = true;
        }
    }

    splitinfer::Manifest manifest;
    if (!splitinfer::load_manifest(path, manifest)) {
        std::fprintf(stderr, "Error: failed to load manifest from '%s'\n", path.c_str());
        return EXIT_FAILURE;
    }

    if (std::getenv("SPLITINFER_NO_NMC")) apply_ablation_no_nmc(manifest);
    bool prefetch_disabled =
        (std::getenv("SPLITINFER_NO_PREFETCH") != nullptr) ||
        (std::getenv("SPLITINFER_NO_PIPELINE") != nullptr);
    if (prefetch_disabled) {
        std::printf("  [ablation] prefetch DISABLED via SPLITINFER_NO_PREFETCH/PIPELINE\n");
    }

    print_summary(manifest);

    // When running many iterations we suppress the per-layer stub trace.
    bool verbose_stubs = (iterations == 1 && warmup == 0);

    using Clock = std::chrono::steady_clock;
    using MsD   = std::chrono::duration<double, std::milli>;

    auto run_once = [](splitinfer::Pipeline& p) -> std::pair<bool, double> {
        auto t0 = Clock::now();
        bool ok = p.run();
        auto t1 = Clock::now();
        return {ok, std::chrono::duration_cast<MsD>(t1 - t0).count()};
    };

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(iterations));
    bool all_ok = true;
    double total_wall_ms = 0.0;

    auto run_measurements = [&](splitinfer::Pipeline& pipeline) {
        auto pass_start = Clock::now();

        // Warmup phase — latencies discarded.
        for (int i = 0; i < warmup; ++i) {
            auto [ok, _dt] = run_once(pipeline);
            if (!ok) all_ok = false;
        }
        // Reset telemetry so warmup runs don't pollute the running totals
        // the pipeline itself accumulates.
        pipeline.get_telemetry().reset();

        // Measured phase.
        for (int i = 0; i < iterations; ++i) {
            auto [ok, dt] = run_once(pipeline);
            if (!ok) all_ok = false;
            latencies_ms.push_back(dt);
        }

        total_wall_ms = std::chrono::duration_cast<MsD>(Clock::now() - pass_start).count();
    };

    if (use_real) {
        std::printf("\nUsing real executors (GpuExecutor + FpgaExecutor)...\n");
        splitinfer::GpuExecutor  gpu;
        splitinfer::FpgaExecutor fpga;
        splitinfer::Pipeline pipeline(manifest, gpu, fpga);
        if (prefetch_disabled) pipeline.set_prefetch_enabled(false);

        std::printf("Running inference (prefetch=%s, warmup=%d, iterations=%d)...\n",
                    pipeline.is_prefetch_enabled() ? "ON" : "OFF", warmup, iterations);
        run_measurements(pipeline);
    } else {
        std::printf("\nRunning inference (stub executors, warmup=%d, iterations=%d)...\n",
                    warmup, iterations);
        StubGpuExecutor  gpu;  gpu.verbose  = verbose_stubs;
        StubFpgaExecutor fpga; fpga.verbose = verbose_stubs;
        splitinfer::Pipeline pipeline(manifest, gpu, fpga);
        if (prefetch_disabled) pipeline.set_prefetch_enabled(false);
        run_measurements(pipeline);
    }

    if (!all_ok) {
        std::fprintf(stderr, "Warning: one or more layers reported an error.\n");
    }

    IterStats stats = compute_stats(latencies_ms);
    print_iter_stats(stats, !prefetch_disabled, warmup);
    if (emit_json) {
        print_stats_json(stats, !prefetch_disabled, warmup, total_wall_ms);
    }

    std::printf("\nDone. Status: %s\n", all_ok ? "OK" : "ERROR");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
