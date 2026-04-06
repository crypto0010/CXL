/* splitinfer/runtime/src/telemetry.cpp */

#include "splitinfer/telemetry.h"

#include <algorithm>
#include <limits>

namespace splitinfer {

Telemetry::Telemetry() {
    reset();
}

void Telemetry::reset() {
    stats_.inference_count  = 0;
    stats_.total_latency_ms = 0.0;
    stats_.avg_latency_ms   = 0.0;
    stats_.min_latency_ms   = std::numeric_limits<double>::max();
    stats_.max_latency_ms   = 0.0;
    stats_.gpu_time_ms      = 0.0;
    stats_.fpga_time_ms     = 0.0;
    stats_.bytes_transferred = 0;
}

void Telemetry::record_inference(double total_ms, double gpu_ms, double fpga_ms,
                                 int64_t xfer_bytes) {
    ++stats_.inference_count;
    stats_.total_latency_ms += total_ms;
    stats_.avg_latency_ms    = stats_.total_latency_ms / static_cast<double>(stats_.inference_count);
    stats_.min_latency_ms    = std::min(stats_.min_latency_ms, total_ms);
    stats_.max_latency_ms    = std::max(stats_.max_latency_ms, total_ms);
    stats_.gpu_time_ms      += gpu_ms;
    stats_.fpga_time_ms     += fpga_ms;
    stats_.bytes_transferred += xfer_bytes;
}

TelemetryStats Telemetry::get_stats() const {
    return stats_;
}

} // namespace splitinfer
