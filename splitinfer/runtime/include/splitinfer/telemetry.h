#ifndef SPLITINFER_TELEMETRY_H
#define SPLITINFER_TELEMETRY_H

#include <cstdint>

namespace splitinfer {

struct TelemetryStats {
    uint64_t inference_count;        ///< Total number of inference calls recorded.
    double   total_latency_ms;       ///< Cumulative wall-clock time across all calls (ms).
    double   avg_latency_ms;         ///< Average latency per call (ms).
    double   min_latency_ms;         ///< Minimum single-call latency (ms).
    double   max_latency_ms;         ///< Maximum single-call latency (ms).
    double   gpu_time_ms;            ///< Cumulative time spent in GPU executor (ms).
    double   fpga_time_ms;           ///< Cumulative time spent in FPGA executor (ms).
    int64_t  bytes_transferred;      ///< Total bytes moved across device boundaries.
};

class Telemetry {
public:
    Telemetry();

    /// Reset all accumulators to zero.
    void reset();

    /// Record a completed inference pass.
    /// @param total_ms   End-to-end wall time for the pass (ms).
    /// @param gpu_ms     Time spent in GPU layers (ms).
    /// @param fpga_ms    Time spent in FPGA layers (ms).
    /// @param xfer_bytes Bytes transferred between devices during this pass.
    void record_inference(double total_ms, double gpu_ms, double fpga_ms,
                          int64_t xfer_bytes);

    /// Return a snapshot of the current statistics.
    TelemetryStats get_stats() const;

private:
    TelemetryStats stats_;
};

} // namespace splitinfer
#endif
