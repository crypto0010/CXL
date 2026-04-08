/* splitinfer/runtime/src/telemetry.cpp */

#include "splitinfer/telemetry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace splitinfer {

Telemetry::Telemetry() {
    reset();
}

void Telemetry::reset() {
    stats_.inference_count   = 0;
    stats_.total_latency_ms  = 0.0;
    stats_.avg_latency_ms    = 0.0;
    stats_.min_latency_ms    = std::numeric_limits<double>::max();
    stats_.max_latency_ms    = 0.0;
    stats_.gpu_time_ms       = 0.0;
    stats_.fpga_time_ms      = 0.0;
    stats_.transfer_time_ms  = 0.0;
    stats_.sync_time_ms      = 0.0;
    stats_.bytes_transferred = 0;
    stats_.prefetch_attempts = 0;
    stats_.prefetch_hits     = 0;
    stats_.prefetch_hit_rate = std::numeric_limits<double>::quiet_NaN();
}

void Telemetry::record_inference(const PassMetrics& m) {
    ++stats_.inference_count;
    stats_.total_latency_ms += m.total_ms;
    stats_.avg_latency_ms    = stats_.total_latency_ms / static_cast<double>(stats_.inference_count);
    stats_.min_latency_ms    = std::min(stats_.min_latency_ms, m.total_ms);
    stats_.max_latency_ms    = std::max(stats_.max_latency_ms, m.total_ms);
    stats_.gpu_time_ms      += m.gpu_compute_ms;
    stats_.fpga_time_ms     += m.fpga_compute_ms;
    stats_.transfer_time_ms += m.transfer_ms;
    stats_.sync_time_ms     += m.sync_ms;
    stats_.bytes_transferred += m.transfer_bytes;
    stats_.prefetch_attempts += m.prefetch_attempts;
    stats_.prefetch_hits     += m.prefetch_hits;
    stats_.prefetch_hit_rate  = (stats_.prefetch_attempts > 0)
        ? static_cast<double>(stats_.prefetch_hits) /
          static_cast<double>(stats_.prefetch_attempts)
        : std::numeric_limits<double>::quiet_NaN();
}

TelemetryStats Telemetry::get_stats() const {
    return stats_;
}

} // namespace splitinfer
