/* Lowered program (see partitioner/lowering.py for the producer). */
#ifndef SPLITINFER_V2_PROGRAM_H
#define SPLITINFER_V2_PROGRAM_H
#include <cstdint>
#include <string>
#include <vector>

namespace splitinfer2 {

struct Segment { uint32_t addr = 0, offset = 0, length = 0; std::string layer, kind; };
struct InputSpec { std::string name, kind; uint32_t ddr2_addr = 0; int count = 0; double scale = 1; int rows = 0; };
struct Part { std::string src, tensor; uint32_t addr = 0; int len = 0; };
struct LayerRec {
    std::string name, kind, placement;
    // gather
    uint32_t table_addr = 0, idx_addr = 0; int rows = 0, dim = 0, dim_bytes = 0, n_idx = 0; std::string index_input;
    // concat
    std::vector<Part> parts; int len = 0;
    // fc
    uint32_t in_addr = 0, w_addr = 0, b_addr = 0, acc_addr = 0, out_addr = 0;
    int K = 0, K_pad = 0, N = 0, M_pad = 0, mult = 0, shift = 0, relu = 0;
    bool final_ = false, has_bias = false;
};
struct VecInput { std::string name; size_t off = 0, len = 0; };
struct Vector { std::vector<VecInput> inputs; size_t expect_off = 0; int expect_count = 0; std::vector<double> fp32_ref; };

struct Program {
    std::string dir;
    uint32_t image_bytes = 0, layout_end = 0;
    std::vector<Segment> segments;
    std::vector<InputSpec> inputs;
    std::vector<LayerRec> layers;
    std::string output_layer; int output_count = 0; double dequant_scale = 1; bool output_int32 = true;
    std::vector<Vector> vectors;
    std::vector<uint8_t> image, vecdata;

    bool load(const std::string& dir, std::string* err);
    const Segment* segment(const std::string& layer, const std::string& kind) const;
    const InputSpec* input(const std::string& name) const;
};

} // namespace
#endif
