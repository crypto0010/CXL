#ifndef SPLITINFER_MANIFEST_H
#define SPLITINFER_MANIFEST_H

#include <string>
#include <vector>

namespace splitinfer {

enum class Device { GPU, FPGA };

struct LayerEntry {
    std::string name;
    std::string op_type;
    Device device;
    int64_t weight_bytes;
    int64_t output_tensor_bytes;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

struct TransferEntry {
    std::string after_layer;
    std::string before_layer;
    Device from_device;
    Device to_device;
    std::vector<std::string> tensor_names;
    int64_t tensor_bytes;
};

struct ManifestSummary {
    int total_layers;
    int gpu_layers;
    int fpga_layers;
    double estimated_latency_ms;
    int64_t gpu_memory_bytes;
    int64_t fpga_memory_bytes;
    int num_transfers;
};

struct Manifest {
    std::string version;
    std::vector<LayerEntry> layers;
    std::vector<TransferEntry> transfers;
    ManifestSummary summary;
};

bool parse_manifest(const std::string& json_str, Manifest& out);
bool load_manifest(const std::string& path, Manifest& out);

} // namespace splitinfer
#endif
