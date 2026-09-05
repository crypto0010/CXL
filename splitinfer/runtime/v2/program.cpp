#include "program.h"
#include "json.h"
#include <fstream>
#include <sstream>

namespace splitinfer2 {

static bool slurp(const std::string& p, std::vector<uint8_t>& out) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {}); return true;
}

bool Program::load(const std::string& d, std::string* err) {
    dir = d;
    std::ifstream f(d + "/program.json"); if (!f) { if (err) *err = "cannot open program.json"; return false; }
    std::stringstream ss; ss << f.rdbuf();
    sjson::Value j;
    try { j = sjson::parse(ss.str()); } catch (std::exception& e) { if (err) *err = e.what(); return false; }
    image_bytes = (uint32_t)j["ddr2"]["image_bytes"].as_int();
    layout_end  = (uint32_t)j["ddr2"]["layout_end"].as_int();
    for (size_t i = 0; i < j["segments"].size(); i++) { const auto& s = j["segments"][i];
        segments.push_back({(uint32_t)s["addr"].as_int(), (uint32_t)s["offset"].as_int(), (uint32_t)s["length"].as_int(), s["layer"].as_str(), s["kind"].as_str()}); }
    for (size_t i = 0; i < j["inputs"].size(); i++) { const auto& s = j["inputs"][i]; InputSpec in;
        in.name = s["name"].as_str(); in.kind = s["kind"].as_str(); in.ddr2_addr = (uint32_t)s["ddr2_addr"].as_int();
        in.count = (int)s["count"].as_int(); in.scale = s["scale"].as_num(1); in.rows = (int)s["rows"].as_int(); inputs.push_back(in); }
    for (size_t i = 0; i < j["layers"].size(); i++) { const auto& s = j["layers"][i]; LayerRec L;
        L.name = s["name"].as_str(); L.kind = s["kind"].as_str(); L.placement = s["placement"].as_str();
        L.table_addr = (uint32_t)s["table_addr"].as_int(); L.idx_addr = (uint32_t)s["idx_addr"].as_int();
        L.rows = (int)s["rows"].as_int(); L.dim = (int)s["dim"].as_int(); L.dim_bytes = (int)s["dim_bytes"].as_int();
        L.n_idx = (int)s["n_idx"].as_int(); L.index_input = s["index_input"].as_str();
        L.len = (int)s["len"].as_int();
        for (size_t k = 0; k < s["parts"].size(); k++) { const auto& p = s["parts"][k];
            L.parts.push_back({p["src"].as_str(), p["tensor"].as_str(), (uint32_t)p["addr"].as_int(), (int)p["len"].as_int()}); }
        L.in_addr = (uint32_t)s["in_addr"].as_int(); L.w_addr = (uint32_t)s["w_addr"].as_int(); L.b_addr = (uint32_t)s["b_addr"].as_int();
        L.acc_addr = (uint32_t)s["acc_addr"].as_int(); L.out_addr = (uint32_t)s["out_addr"].as_int();
        L.K = (int)s["K"].as_int(); L.K_pad = (int)s["K_pad"].as_int(); L.N = (int)s["N"].as_int(); L.M_pad = (int)s["M_pad"].as_int();
        L.mult = (int)s["mult"].as_int(); L.shift = (int)s["shift"].as_int(); L.relu = (int)s["relu"].as_int();
        L.final_ = s["final"].as_bool(); L.has_bias = s["has_bias"].as_bool();
        layers.push_back(L); }
    output_layer = j["output"]["layer"].as_str(); output_count = (int)j["output"]["count"].as_int();
    dequant_scale = j["output"]["dequant_scale"].as_num(1); output_int32 = j["output"]["int32"].as_bool(true);
    const auto& vv = j["vectors"];
    for (size_t i = 0; i < vv["entries"].size(); i++) { const auto& e = vv["entries"][i]; Vector v;
        for (size_t k = 0; k < e["inputs"].size(); k++) { const auto& in = e["inputs"][k];
            v.inputs.push_back({in["name"].as_str(), (size_t)in["off"].as_int(), (size_t)in["len"].as_int()}); }
        v.expect_off = (size_t)e["expect_off"].as_int(); v.expect_count = (int)e["expect_count"].as_int();
        for (size_t k = 0; k < e["fp32_ref"].size(); k++) v.fp32_ref.push_back(e["fp32_ref"][k].as_num());
        vectors.push_back(v); }
    if (!slurp(d + "/image.bin", image)) { if (err) *err = "cannot read image.bin"; return false; }
    if (vv.has("file") && !slurp(d + "/" + vv["file"].as_str(), vecdata)) { if (err) *err = "cannot read vectors"; return false; }
    if (image.size() != image_bytes) { if (err) *err = "image size mismatch"; return false; }
    return true;
}

const Segment* Program::segment(const std::string& layer, const std::string& kind) const {
    for (auto& s : segments) if (s.layer == layer && s.kind == kind) return &s;
    return nullptr;
}
const InputSpec* Program::input(const std::string& name) const {
    for (auto& i : inputs) if (i.name == name) return &i;
    return nullptr;
}
} // namespace
