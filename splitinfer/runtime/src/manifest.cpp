/* splitinfer/runtime/src/manifest.cpp
 * Simple JSON parser for the SplitInfer partition manifest.
 * No external dependencies — uses only the C++ standard library.
 */

#include "splitinfer/manifest.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <cstdlib>

namespace splitinfer {

// ─── low-level helpers ────────────────────────────────────────────────────────

static void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace((unsigned char)s[pos])) ++pos;
}

// Returns the raw content of the next JSON string (without surrounding quotes).
static std::string parse_json_string(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"')
        throw std::runtime_error("Expected '\"' at pos " + std::to_string(pos));
    ++pos; // skip opening quote
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\') {
            ++pos;
            if (pos >= s.size()) break;
            switch (s[pos]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                default:   result += s[pos]; break;
            }
        } else {
            result += s[pos];
        }
        ++pos;
    }
    if (pos < s.size()) ++pos; // skip closing quote
    return result;
}

// Skip one complete JSON value (string, number, object, array, literal).
static void skip_value(const std::string& s, size_t& pos);

static void skip_object(const std::string& s, size_t& pos) {
    // pos points at '{'
    ++pos;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') { ++pos; return; }
    while (pos < s.size()) {
        skip_ws(s, pos);
        parse_json_string(s, pos); // key
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ':') ++pos;
        skip_value(s, pos);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        if (pos < s.size() && s[pos] == '}') { ++pos; return; }
    }
}

static void skip_array(const std::string& s, size_t& pos) {
    ++pos;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') { ++pos; return; }
    while (pos < s.size()) {
        skip_value(s, pos);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; continue; }
        if (pos < s.size() && s[pos] == ']') { ++pos; return; }
    }
}

static void skip_value(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) return;
    char c = s[pos];
    if (c == '"') { parse_json_string(s, pos); }
    else if (c == '{') { skip_object(s, pos); }
    else if (c == '[') { skip_array(s, pos); }
    else {
        // number or literal (true/false/null)
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' &&
               s[pos] != ']' && !std::isspace((unsigned char)s[pos]))
            ++pos;
    }
}

// ─── mid-level helpers ────────────────────────────────────────────────────────

// Find the value associated with `key` inside the JSON object region [begin,end).
// On success pos_out points just after the colon (before the value).
static bool find_key(const std::string& s, size_t begin, size_t end,
                     const std::string& key, size_t& value_pos) {
    size_t pos = begin;
    skip_ws(s, pos);
    if (pos < end && s[pos] == '{') ++pos;
    while (pos < end) {
        skip_ws(s, pos);
        if (pos >= end || s[pos] == '}') break;
        std::string k = parse_json_string(s, pos);
        skip_ws(s, pos);
        if (pos < end && s[pos] == ':') ++pos;
        skip_ws(s, pos);
        if (k == key) {
            value_pos = pos;
            return true;
        }
        skip_value(s, pos);
        skip_ws(s, pos);
        if (pos < end && s[pos] == ',') ++pos;
    }
    return false;
}

static std::string extract_string(const std::string& s, size_t begin, size_t end,
                                  const std::string& key) {
    size_t vpos;
    if (!find_key(s, begin, end, key, vpos)) return "";
    return parse_json_string(s, vpos);
}

static int64_t extract_int(const std::string& s, size_t begin, size_t end,
                            const std::string& key) {
    size_t vpos;
    if (!find_key(s, begin, end, key, vpos)) return 0;
    return std::stoll(s.substr(vpos, end - vpos));
}

static double extract_double(const std::string& s, size_t begin, size_t end,
                              const std::string& key) {
    size_t vpos;
    if (!find_key(s, begin, end, key, vpos)) return 0.0;
    return std::stod(s.substr(vpos, end - vpos));
}

static std::vector<std::string> extract_string_array(const std::string& s,
                                                      size_t begin, size_t end,
                                                      const std::string& key) {
    std::vector<std::string> result;
    size_t vpos;
    if (!find_key(s, begin, end, key, vpos)) return result;
    if (vpos >= end || s[vpos] != '[') return result;
    ++vpos; // skip '['
    while (vpos < end && s[vpos] != ']') {
        skip_ws(s, vpos);
        if (vpos < end && s[vpos] == ']') break;
        if (vpos < end && s[vpos] == '"')
            result.push_back(parse_json_string(s, vpos));
        else
            break;
        skip_ws(s, vpos);
        if (vpos < end && s[vpos] == ',') ++vpos;
    }
    return result;
}

// Collect top-level objects inside a named JSON array, returning spans [begin,end).
static std::vector<std::pair<size_t,size_t>> extract_objects(const std::string& s,
                                                               size_t doc_begin,
                                                               size_t doc_end,
                                                               const std::string& key) {
    std::vector<std::pair<size_t,size_t>> spans;
    size_t vpos;
    if (!find_key(s, doc_begin, doc_end, key, vpos)) return spans;
    if (vpos >= doc_end || s[vpos] != '[') return spans;
    ++vpos; // skip '['
    while (vpos < doc_end) {
        skip_ws(s, vpos);
        if (vpos >= doc_end || s[vpos] == ']') break;
        if (s[vpos] != '{') break;
        size_t obj_start = vpos;
        skip_object(s, vpos);
        spans.push_back({obj_start, vpos});
        skip_ws(s, vpos);
        if (vpos < doc_end && s[vpos] == ',') ++vpos;
    }
    return spans;
}

// ─── device string → enum ─────────────────────────────────────────────────────

static Device parse_device(const std::string& s) {
    if (s == "FPGA" || s == "fpga") return Device::FPGA;
    return Device::GPU;
}

// ─── public API ───────────────────────────────────────────────────────────────

bool parse_manifest(const std::string& json_str, Manifest& out) {
    try {
        const size_t N = json_str.size();

        out.version = extract_string(json_str, 0, N, "version");

        // ── layers ──
        auto layer_spans = extract_objects(json_str, 0, N, "layers");
        out.layers.clear();
        for (auto [b, e] : layer_spans) {
            LayerEntry le;
            le.name              = extract_string(json_str, b, e, "name");
            le.op_type           = extract_string(json_str, b, e, "op_type");
            le.device            = parse_device(extract_string(json_str, b, e, "device"));
            le.weight_bytes      = extract_int(json_str, b, e, "weight_bytes");
            le.output_tensor_bytes = extract_int(json_str, b, e, "output_tensor_bytes");
            le.inputs            = extract_string_array(json_str, b, e, "inputs");
            le.outputs           = extract_string_array(json_str, b, e, "outputs");
            out.layers.push_back(std::move(le));
        }

        // ── transfers ──
        auto xfer_spans = extract_objects(json_str, 0, N, "transfers");
        out.transfers.clear();
        for (auto [b, e] : xfer_spans) {
            TransferEntry te;
            te.after_layer  = extract_string(json_str, b, e, "after_layer");
            te.before_layer = extract_string(json_str, b, e, "before_layer");
            te.from_device  = parse_device(extract_string(json_str, b, e, "from_device"));
            te.to_device    = parse_device(extract_string(json_str, b, e, "to_device"));
            te.tensor_names = extract_string_array(json_str, b, e, "tensor_names");
            te.tensor_bytes = extract_int(json_str, b, e, "tensor_bytes");
            out.transfers.push_back(std::move(te));
        }

        // ── summary ──
        size_t sum_pos;
        if (find_key(json_str, 0, N, "summary", sum_pos)) {
            // find matching closing brace
            size_t depth = 1, p = sum_pos + 1; // skip '{'
            while (p < N && depth > 0) {
                if (json_str[p] == '{') ++depth;
                else if (json_str[p] == '}') --depth;
                ++p;
            }
            size_t sb = sum_pos, se = p;
            out.summary.total_layers         = static_cast<int>(extract_int(json_str, sb, se, "total_layers"));
            out.summary.gpu_layers           = static_cast<int>(extract_int(json_str, sb, se, "gpu_layers"));
            out.summary.fpga_layers          = static_cast<int>(extract_int(json_str, sb, se, "fpga_layers"));
            out.summary.estimated_latency_ms = extract_double(json_str, sb, se, "estimated_latency_ms");
            out.summary.gpu_memory_bytes     = extract_int(json_str, sb, se, "gpu_memory_bytes");
            out.summary.fpga_memory_bytes    = extract_int(json_str, sb, se, "fpga_memory_bytes");
            out.summary.num_transfers        = static_cast<int>(extract_int(json_str, sb, se, "num_transfers"));
        }

        return true;
    } catch (const std::exception& ex) {
        return false;
    }
}

bool load_manifest(const std::string& path, Manifest& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_manifest(ss.str(), out);
}

} // namespace splitinfer
