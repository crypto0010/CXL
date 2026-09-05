#include "executor.h"
#include "cxlwin/cxlwin.h"
extern "C" {
#include <edgecoh/messages.h>
#include <edgecoh/transport.h>
}
#include <chrono>
#include <cstdio>
#include <cstring>

namespace splitinfer2 {

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static int pad16(int n) { return (n + 15) / 16 * 16; }

/* ── Host ────────────────────────────────────────────────────────────── */
class HostExecutor : public Executor {
    std::vector<uint8_t> mem;      /* DDR2-shaped host arena: weights + activations at their DDR2 addresses */
public:
    const char* mode() const override { return "host"; }
    bool prepare(const Program& p, std::string* err) override {
        mem.assign(p.layout_end + 64, 0);
        for (auto& s : p.segments) if (!p.streaming || s.kind == "table") std::memcpy(mem.data() + s.addr, p.image.data() + s.offset, s.length);
        (void)err; return true;
    }
    bool run(const Program& p, const std::map<std::string, std::vector<uint8_t>>& in,
             std::vector<int32_t>& out, RunMetrics& m) override {
        auto t0 = Clock::now();
        for (auto& is : p.inputs) { auto it = in.find(is.name); if (it == in.end()) return false;
            std::memcpy(mem.data() + is.ddr2_addr, it->second.data(), it->second.size()); }
        m.input_ms = ms_since(t0);
        for (auto& L : p.layers) {
            auto tl = Clock::now();
            if (L.stream) for (auto& s : p.segments) if (s.layer == L.name) std::memcpy(mem.data() + s.addr, p.image.data() + s.offset, s.length);
            if (L.kind == "gather") {
                int32_t idx; std::memcpy(&idx, mem.data() + L.idx_addr, 4);
                kernels::gather((int8_t*)(mem.data() + L.table_addr), L.dim_bytes, L.dim, idx, (int8_t*)(mem.data() + L.out_addr));
            } else if (L.kind == "fc") {
                kernels::fc_acc((int8_t*)(mem.data() + L.w_addr), L.M_pad, L.K_pad, (int8_t*)(mem.data() + L.in_addr), (int32_t*)(mem.data() + L.acc_addr));
                if (!L.final_) kernels::epilogue((int32_t*)(mem.data() + L.acc_addr), (int32_t*)(mem.data() + L.b_addr), L.M_pad, L.mult, L.shift, L.relu, (int8_t*)(mem.data() + L.out_addr));
            } /* concat: no-op by construction */
            m.layers.push_back({L.name, L.kind, ms_since(tl)});
        }
        auto tf = Clock::now();
        const LayerRec* last = nullptr; for (auto& L : p.layers) if (L.name == p.output_layer) last = &L;
        out.resize(last->N);
        for (int i = 0; i < last->N; i++) { int32_t a, b; std::memcpy(&a, mem.data() + last->acc_addr + i * 4, 4);
            std::memcpy(&b, mem.data() + last->b_addr + i * 4, 4); out[i] = a + b; }
        m.output_ms = ms_since(tf);
        m.total_ms = ms_since(t0);
        return true;
    }
};

/* ── EdgeCoh helpers (shared by NMC and Pool) ────────────────────────── */
struct Link {
    edgecoh_transport* t; RunMetrics* m = nullptr;
    static constexpr int TO = 30000;
    bool ack() { edgecoh_header_t h; if (edgecoh_recv_header(t, &h, TO) < 0) return false;
                 if (m) m->link_bytes_in += sizeof(h); return h.msg_type == EDGECOH_MSG_ACK || h.msg_type == EDGECOH_MSG_NMC_DONE; }
    bool write(uint32_t addr, const void* src, size_t len) {
        const uint8_t* p = (const uint8_t*)src;
        while (len) {
            size_t chunk = len > 4096 ? 4096 : len;
            edgecoh_data_write_msg_t msg; std::memset(&msg, 0, sizeof msg);
            msg.header.msg_type = EDGECOH_MSG_DATA_WRITE; msg.header.payload_len = (uint32_t)(4 + chunk); msg.ddr2_addr = addr;
            std::vector<uint8_t> buf(sizeof msg + chunk);
            int n = edgecoh_serialize(&msg, buf.data(), (int)sizeof msg); std::memcpy(buf.data() + n, p, chunk);
            if (edgecoh_transport_send(t, buf.data(), n + (int)chunk) != n + (int)chunk) return false;
            if (m) { m->link_bytes_out += n + chunk; m->link_msgs++; }
            if (!ack()) return false;
            p += chunk; addr += (uint32_t)chunk; len -= chunk;
        }
        return true;
    }
    bool read(uint32_t addr, void* dst, size_t len) {
        edgecoh_data_read_msg_t msg; std::memset(&msg, 0, sizeof msg);
        msg.header.msg_type = EDGECOH_MSG_DATA_READ; msg.header.payload_len = 8; msg.ddr2_addr = addr; msg.read_len = (uint32_t)len;
        uint8_t buf[32]; int n = edgecoh_serialize(&msg, buf, sizeof buf);
        if (edgecoh_transport_send(t, buf, n) != n) return false;
        if (m) { m->link_bytes_out += n; m->link_msgs++; }
        edgecoh_header_t h; if (edgecoh_recv_header(t, &h, TO) < 0 || h.msg_type != EDGECOH_MSG_DATA_RESPONSE) return false;
        size_t got = 0; uint8_t* d = (uint8_t*)dst;
        while (got < len) { int r = edgecoh_transport_recv(t, d + got, (int)std::min<size_t>(4096, len - got), TO); if (r <= 0) return false; got += r; }
        if (m) m->link_bytes_in += sizeof(h) + len;
        return true;
    }
    bool nmc(uint8_t op, uint32_t tb, uint32_t rows, uint32_t cols, uint32_t ia, uint32_t il, uint32_t oa) {
        edgecoh_nmc_exec_msg_t msg; std::memset(&msg, 0, sizeof msg);
        msg.header.msg_type = EDGECOH_MSG_NMC_EXEC; msg.header.payload_len = 25;
        msg.nmc_op = op; msg.table_base_addr = tb; msg.table_rows = rows; msg.table_cols = cols;
        msg.input_addr = ia; msg.input_len = il; msg.output_addr = oa;
        uint8_t buf[64]; int n = edgecoh_serialize(&msg, buf, sizeof buf);
        if (edgecoh_transport_send(t, buf, n) != n) return false;
        if (m) { m->link_bytes_out += n; m->link_msgs++; }
        return ack();
    }
    bool load_image(const Program& p) {
        for (auto& s : p.segments) {
            bool streamed = p.streaming && (s.kind == "weight" || s.kind == "bias");
            if (!streamed && !write(s.addr, p.image.data() + s.offset, s.length)) return false;
        }
        return true;
    }
    bool stream_layer(const Program& p, const LayerRec& L) {
        for (auto& s : p.segments) if (s.layer == L.name && !write(s.addr, p.image.data() + s.offset, s.length)) return false;
        return true;
    }
};

/* ── NMC ─────────────────────────────────────────────────────────────── */
class NmcExecutor : public Executor {
    Link link; std::vector<int32_t> bias_last;
public:
    explicit NmcExecutor(edgecoh_transport* t) : link{t} {}
    const char* mode() const override { return "nmc"; }
    bool prepare(const Program& p, std::string* err) override {
        RunMetrics tmp; link.m = &tmp;
        if (!link.load_image(p)) { if (err) *err = "image load failed"; return false; }
        const LayerRec* last = nullptr; for (auto& L : p.layers) if (L.name == p.output_layer) last = &L;
        auto* bs = p.segment(last->name, "bias"); bias_last.resize(last->M_pad);
        std::memcpy(bias_last.data(), p.image.data() + bs->offset, bs->length);
        return true;
    }
    bool run(const Program& p, const std::map<std::string, std::vector<uint8_t>>& in,
             std::vector<int32_t>& out, RunMetrics& m) override {
        link.m = &m;
        auto t0 = Clock::now();
        for (auto& is : p.inputs) { auto it = in.find(is.name); if (it == in.end()) return false;
            std::vector<uint8_t> padded(it->second); padded.resize(pad16((int)padded.size()), 0);
            if (!link.write(is.ddr2_addr, padded.data(), padded.size())) return false; }
        m.input_ms = ms_since(t0);
        for (auto& L : p.layers) {
            auto tl = Clock::now();
            if (L.stream) { auto ts = Clock::now(); if (!link.stream_layer(p, L)) return false; m.weight_stream_ms += ms_since(ts); }
            if (L.kind == "gather") {
                if (!link.nmc(0x01, L.table_addr, L.rows, L.dim_bytes, L.idx_addr, L.n_idx, L.out_addr)) return false;
            } else if (L.kind == "fc") {
                if (!link.nmc(0x02, L.w_addr, L.M_pad, L.K_pad, L.in_addr, 0, L.acc_addr)) return false;
                if (!L.final_) {
                    uint32_t tb = 4u | ((uint32_t)((L.relu ? 0x80 : 0) | (L.shift & 31)) << 8);
                    if (!link.nmc(0x03, tb, L.M_pad / 4, L.b_addr, L.acc_addr, (uint32_t)L.mult, L.out_addr)) return false;
                }
            }
            m.layers.push_back({L.name, L.kind, ms_since(tl)});
        }
        auto tf = Clock::now();
        const LayerRec* last = nullptr; for (auto& L : p.layers) if (L.name == p.output_layer) last = &L;
        out.resize(last->N);
        std::vector<int32_t> acc(last->N);
        if (!link.read(last->acc_addr, acc.data(), acc.size() * 4)) return false;
        for (int i = 0; i < last->N; i++) out[i] = acc[i] + bias_last[i];
        m.output_ms = ms_since(tf);
        m.total_ms = ms_since(t0);
        return true;
    }
};

/* ── Pool ────────────────────────────────────────────────────────────── */
class PoolExecutor : public Executor {
    edgecoh_transport* t; unsigned prefetch; bool warm;
    cxlwin_t* win = nullptr; uint8_t* base = nullptr;
    std::vector<uint8_t> act;      /* host-side activations (DDR2-shaped arena) */
public:
    PoolExecutor(edgecoh_transport* t_, unsigned pf, bool w) : t(t_), prefetch(pf), warm(w) {}
    ~PoolExecutor() override { if (win) cxlwin_destroy(win); }
    const char* mode() const override { return "pool"; }
    bool prepare(const Program& p, std::string* err) override {
        if (p.streaming) { if (err) *err = "pool mode requires DDR2-resident weights (program is streaming)"; return false; }
        RunMetrics tmp; Link link{t, &tmp};
        if (!link.load_image(p)) { if (err) *err = "image load failed"; return false; }
        cxlwin_backend_t be; if (cxlwin_backend_edgecoh_create(&be, t, 0) != 0) { if (err) *err = "backend"; return false; }
        cxlwin_config_t cfg{}; cfg.size = p.layout_end; cfg.dev_base = 0; cfg.prefetch_pages = prefetch; cfg.strict = 0;
        win = cxlwin_create(&cfg, &be); if (!win) { if (err) *err = "cxlwin_create"; return false; }
        base = (uint8_t*)cxlwin_base(win);
        act.assign(p.layout_end + 64, 0);
        return true;
    }
    void reset_between_iterations() override { if (!warm && win) cxlwin_invalidate(win, 0, 0); }
    bool run(const Program& p, const std::map<std::string, std::vector<uint8_t>>& in,
             std::vector<int32_t>& out, RunMetrics& m) override {
        cxlwin_reset_stats(win);
        auto t0 = Clock::now();
        for (auto& is : p.inputs) { auto it = in.find(is.name); if (it == in.end()) return false;
            std::memcpy(act.data() + is.ddr2_addr, it->second.data(), it->second.size()); }
        m.input_ms = ms_since(t0);
        for (auto& L : p.layers) {
            auto tl = Clock::now();
            if (L.kind == "gather") {
                int32_t idx; std::memcpy(&idx, act.data() + L.idx_addr, 4);
                /* table lives in the window: this dereference faults the row's page(s) in */
                kernels::gather((int8_t*)(base + L.table_addr), L.dim_bytes, L.dim, idx, (int8_t*)(act.data() + L.out_addr));
            } else if (L.kind == "fc") {
                kernels::fc_acc((int8_t*)(base + L.w_addr), L.M_pad, L.K_pad, (int8_t*)(act.data() + L.in_addr), (int32_t*)(act.data() + L.acc_addr));
                if (!L.final_) kernels::epilogue((int32_t*)(act.data() + L.acc_addr), (int32_t*)(base + L.b_addr), L.M_pad, L.mult, L.shift, L.relu, (int8_t*)(act.data() + L.out_addr));
            }
            m.layers.push_back({L.name, L.kind, ms_since(tl)});
        }
        auto tf = Clock::now();
        const LayerRec* last = nullptr; for (auto& L : p.layers) if (L.name == p.output_layer) last = &L;
        out.resize(last->N);
        for (int i = 0; i < last->N; i++) { int32_t a, b; std::memcpy(&a, act.data() + last->acc_addr + i * 4, 4);
            std::memcpy(&b, base + last->b_addr + i * 4, 4); out[i] = a + b; }
        m.output_ms = ms_since(tf);
        cxlwin_stats_t s; cxlwin_get_stats(win, &s);
        m.faults = s.read_faults; m.pages_fetched = s.pages_fetched; m.fetch_calls = s.fetch_calls;
        m.link_bytes_in = s.bytes_in; m.link_msgs = s.fetch_calls;
        m.total_ms = ms_since(t0);
        return true;
    }
};

std::unique_ptr<Executor> make_host_executor() { return std::make_unique<HostExecutor>(); }
std::unique_ptr<Executor> make_nmc_executor(edgecoh_transport* t) { return std::make_unique<NmcExecutor>(t); }
std::unique_ptr<Executor> make_pool_executor(edgecoh_transport* t, unsigned pf, bool warm) { return std::make_unique<PoolExecutor>(t, pf, warm); }

} // namespace
