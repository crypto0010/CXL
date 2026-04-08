#ifndef SPLITINFER_TELEMETRY_H
#define SPLITINFER_TELEMETRY_H

#include <cstdint>

namespace splitinfer {

/// Per-pass measurements gathered during a single Pipeline::run() invocation.
/// Passed to Telemetry::record_inference() so the per-pass counters can be
/// accumulated into running totals.
///
/// Spec § 7.3 mandates separate buckets for compute vs transfer vs sync —
/// the original telemetry lumped everything into gpu_time_ms / fpga_time_ms,
/// which made it impossible to attribute latency.  This version splits them.
struct PassMetrics {
    double  total_ms        = 0.0;  ///< End-to-end wall time for this pass.
    double  gpu_compute_ms  = 0.0;  ///< Time inside gpu_.execute() calls.
    double  fpga_compute_ms = 0.0;  ///< Time inside fpga_.execute() calls.
    double  transfer_ms     = 0.0;  ///< Time inside transfer_to_fpga / transfer_to_host calls.
    double  sync_ms         = 0.0;  ///< Time waiting for prefetched transfers to drain.
    int64_t transfer_bytes  = 0;    ///< Bytes moved across device boundaries.
    /* Prefetch accounting (spec § 7.3 secondary metric: prefetch_hit_rate) */
    uint64_t prefetch_attempts = 0; ///< How many transfers were eligible for prefetch.
    uint64_t prefetch_hits     = 0; ///< How many of those were issued strictly before their consumer's execute() — i.e. successfully overlapped.
};

struct TelemetryStats {
    uint64_t inference_count;        ///< Total number of inference calls recorded.
    double   total_latency_ms;       ///< Cumulative wall-clock time across all calls (ms).
    double   avg_latency_ms;         ///< Average latency per call (ms).
    double   min_latency_ms;         ///< Minimum single-call latency (ms).
    double   max_latency_ms;         ///< Maximum single-call latency (ms).
    double   gpu_time_ms;            ///< Cumulative time inside gpu_.execute() (ms).
    double   fpga_time_ms;           ///< Cumulative time inside fpga_.execute() (ms).
    double   transfer_time_ms;       ///< Cumulative time in cross-device transfers (ms).
    double   sync_time_ms;           ///< Cumulative time waiting for prefetched transfers (ms).
    int64_t  bytes_transferred;      ///< Total bytes moved across device boundaries.
    uint64_t prefetch_attempts;      ///< Total transfers eligible for prefetch.
    uint64_t prefetch_hits;          ///< Total transfers actually overlapped with compute.
    double   prefetch_hit_rate;      ///< prefetch_hits / prefetch_attempts (0..1, NaN if 0 attempts).
};

class Telemetry {
public:
    Telemetry();

    /// Reset all accumulators to zero.
    void reset();

    /// Record a completed inference pass with full per-pass breakdown.
    /// Recomputes derived stats (avg_latency_ms, prefetch_hit_rate).
    void record_inference(const PassMetrics& m);

    /// Return a snapshot of the current statistics.
    TelemetryStats get_stats() const;

private:
    TelemetryStats stats_;
};

} // namespace splitinfer
#endif
