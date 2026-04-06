/* splitinfer/runtime/tests/test_manifest.cpp
 * Tests for the C++ manifest loader.
 */

#include "splitinfer/manifest.h"

#include <cassert>
#include <cstdio>
#include <string>

// Embedded test JSON — 3 layers (2 GPU, 1 FPGA) and 1 transfer.
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

int main() {
    splitinfer::Manifest m;
    bool ok = splitinfer::parse_manifest(std::string(TEST_JSON), m);

    // parse succeeds
    CHECK(ok);

    // version
    CHECK(m.version == "1.0");

    // layer count
    CHECK(m.layers.size() == 3u);

    // layer 0 — GPU conv
    if (m.layers.size() >= 1) {
        const auto& l0 = m.layers[0];
        CHECK(l0.name == "conv1");
        CHECK(l0.op_type == "Conv");
        CHECK(l0.device == splitinfer::Device::GPU);
        CHECK(l0.weight_bytes == 4096);
        CHECK(l0.output_tensor_bytes == 8192);
        CHECK(l0.inputs.size() == 1u);
        CHECK(l0.inputs[0] == "input_0");
        CHECK(l0.outputs.size() == 1u);
        CHECK(l0.outputs[0] == "conv1_out");
    }

    // layer 1 — FPGA gather
    if (m.layers.size() >= 2) {
        const auto& l1 = m.layers[1];
        CHECK(l1.name == "embed1");
        CHECK(l1.op_type == "Gather");
        CHECK(l1.device == splitinfer::Device::FPGA);
        CHECK(l1.weight_bytes == 16384);
        CHECK(l1.output_tensor_bytes == 2048);
    }

    // layer 2 — GPU relu
    if (m.layers.size() >= 3) {
        const auto& l2 = m.layers[2];
        CHECK(l2.name == "relu1");
        CHECK(l2.device == splitinfer::Device::GPU);
    }

    // transfer count
    CHECK(m.transfers.size() == 1u);

    if (!m.transfers.empty()) {
        const auto& t = m.transfers[0];
        CHECK(t.after_layer == "conv1");
        CHECK(t.before_layer == "embed1");
        CHECK(t.from_device == splitinfer::Device::GPU);
        CHECK(t.to_device == splitinfer::Device::FPGA);
        CHECK(t.tensor_names.size() == 1u);
        CHECK(t.tensor_names[0] == "conv1_out");
        CHECK(t.tensor_bytes == 8192);
    }

    // summary
    CHECK(m.summary.total_layers == 3);
    CHECK(m.summary.gpu_layers == 2);
    CHECK(m.summary.fpga_layers == 1);
    CHECK(m.summary.num_transfers == 1);
    CHECK(m.summary.estimated_latency_ms > 12.0 && m.summary.estimated_latency_ms < 13.0);
    CHECK(m.summary.gpu_memory_bytes == 12288);
    CHECK(m.summary.fpga_memory_bytes == 18432);

    std::printf("test_manifest: %d passed, %d failed\n", pass_count, fail_count);
    return fail_count == 0 ? 0 : 1;
}
