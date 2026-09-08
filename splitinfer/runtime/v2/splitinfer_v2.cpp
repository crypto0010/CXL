/* splitinfer_v2 — run a lowered program in one of three execution modes and
 * verify it against the lowering's expected outputs.
 *
 *   splitinfer_v2 <program_dir> --mode host|nmc|pool --transport emul|usb
 *                 [--iterations N] [--warmup W] [--vectors K] [--json]
 *                 [--prefetch-pages P] [--pool-warm]
 *                 [--emul-link-bw BYTES/S] [--emul-rtt-ms MS] [--emul-clock-hz HZ]
 *
 * Every iteration's output is compared with the expected INT32 vector; a
 * mismatch is reported and the exit code is non-zero.  Latencies are
 * reported as full distributions (median, p5/p95/p99, max, CV) — never a
 * median alone.
 */
#include "executor.h"
#include "program.h"
extern "C" {
#include <edgecoh/transport.h>
#include <edgecoh/transport_emul.h>
}
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace splitinfer2;

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0; std::sort(v.begin(), v.end());
    double r = p / 100.0 * (v.size() - 1); size_t lo = (size_t)r; double f = r - lo;
    return lo + 1 < v.size() ? v[lo] * (1 - f) + v[lo + 1] * f : v[lo];
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <program_dir> [options]\n", argv[0]); return 2; }
    std::string dir = argv[1], mode = "host", transport = "emul";
    int iterations = 1, warmup = 0, nvec = -1; bool json = false, warm = false;
    unsigned prefetch = 1; double link_bw = 0, rtt_ms = 0, clock_hz = 0; int max_layers = -1; bool verbose = false, verify_layers = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
        if (a == "--mode") mode = next(); else if (a == "--transport") transport = next();
        else if (a == "--iterations") iterations = std::atoi(next().c_str()); else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--vectors") nvec = std::atoi(next().c_str()); else if (a == "--json") json = true;
        else if (a == "--prefetch-pages") prefetch = (unsigned)std::atoi(next().c_str()); else if (a == "--pool-warm") warm = true;
        else if (a == "--emul-link-bw") link_bw = std::atof(next().c_str()); else if (a == "--emul-rtt-ms") rtt_ms = std::atof(next().c_str());
        else if (a == "--emul-clock-hz") clock_hz = std::atof(next().c_str());
        else if (a == "--max-layers") max_layers = std::atoi(next().c_str()); else if (a == "--verbose") verbose = true;
        else if (a == "--verify-layers") verify_layers = true;
        else if (a == "--verify-layers") verify_layers = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    Program p; std::string err;
    if (!p.load(dir, &err)) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    if (nvec < 0 || nvec > (int)p.vectors.size()) nvec = (int)p.vectors.size();

    edgecoh_transport* t = nullptr;
    if (mode != "host") {
        if (transport == "emul") {
            edgecoh_emul_params_t ep{}; ep.ddr2_bytes = 128u << 20; ep.link_bytes_per_s = link_bw; ep.link_rtt_s = rtt_ms / 1000.0;
            ep.engine_clock_hz = clock_hz; ep.mac_per_cycle = 0.684;
            t = edgecoh_transport_open_emul(&ep);
        } else t = edgecoh_transport_open(0x0403, 0x6010);
        if (!t) { std::fprintf(stderr, "transport open failed\n"); return 1; }
    }
    std::unique_ptr<Executor> ex;
    if (mode == "host") ex = make_host_executor(); else if (mode == "nmc") ex = make_nmc_executor(t);
    else if (mode == "pool") ex = make_pool_executor(t, prefetch, warm);
    else { std::fprintf(stderr, "bad mode\n"); return 2; }

    ex->set_debug(max_layers, verbose);
    auto tp = std::chrono::steady_clock::now();
    if (!ex->prepare(p, &err)) { std::fprintf(stderr, "prepare failed: %s\n", err.c_str()); return 1; }
    double prepare_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();

    auto inputs_for = [&](const Vector& v) { std::map<std::string, std::vector<uint8_t>> in;
        for (auto& vi : v.inputs) in[vi.name] = std::vector<uint8_t>(p.vecdata.begin() + vi.off, p.vecdata.begin() + vi.off + vi.len); return in; };

    std::vector<double> lat; std::vector<RunMetrics> mets; int mismatches = 0, runs = 0;
    double max_abs_err_fp32 = 0, max_ref = 0;
    for (int it = 0; it < warmup + iterations; it++) {
        const Vector& v = p.vectors[it % nvec];
        auto in = inputs_for(v);
        std::vector<int32_t> out; RunMetrics m;
        ex->reset_between_iterations();
        if (!ex->run(p, in, out, m)) { std::fprintf(stderr, "run failed at iteration %d\n", it); return 1; }
        const int32_t* expect = (const int32_t*)(p.vecdata.data() + v.expect_off);
        bool ok = max_layers >= 0 || ((int)out.size() == v.expect_count && std::memcmp(out.data(), expect, out.size() * 4) == 0);
        if (it >= warmup) {
            runs++; lat.push_back(m.total_ms); mets.push_back(m);
            if (!ok) { mismatches++; if (mismatches <= 3) { std::fprintf(stderr, "MISMATCH iter %d: got %d expect %d\n", it, out.empty() ? 0 : out[0], expect[0]); } }
            for (size_t k = 0; k < out.size() && k < v.fp32_ref.size(); k++) {
                double deq = out[k] * p.dequant_scale; max_abs_err_fp32 = std::max(max_abs_err_fp32, std::fabs(deq - v.fp32_ref[k]));
                max_ref = std::max(max_ref, std::fabs(v.fp32_ref[k])); }
        }
    }
    if (verify_layers && mode == "nmc") {
        /* run the host executor on the LAST input used, then compare every DDR2 region */
        const Vector& v = p.vectors[(warmup + iterations - 1) % nvec];
        auto host = make_host_executor(); std::string e2; host->prepare(p, &e2);
        std::vector<int32_t> ho; RunMetrics hm; host->run(p, inputs_for(v), ho, hm);
        std::fprintf(stderr, "verify-layers: comparing DDR2 against host arena for the last input\n");
        int bad = ex->verify_layers(p, *host->arena());
        std::fprintf(stderr, "verify-layers: %d region(s) differ\n", bad);
    }
    if (verify_layers && mode == "nmc") {
        const Vector& v = p.vectors[(warmup + iterations - 1) % nvec];
        auto host = make_host_executor(); std::string e2; host->prepare(p, &e2);
        std::vector<int32_t> ho; RunMetrics hm; host->run(p, inputs_for(v), ho, hm);
        std::fprintf(stderr, "verify-layers: comparing DDR2 against host arena for the last input\n");
        int bad = ex->verify_layers(p, *host->arena());
        std::fprintf(stderr, "verify-layers: %d region(s) differ\n", bad);
    }
    double mean = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
    double var = 0; for (double x : lat) var += (x - mean) * (x - mean); double sd = lat.size() > 1 ? std::sqrt(var / (lat.size() - 1)) : 0;
    uint64_t bytes_out = 0, bytes_in = 0, msgs = 0, faults = 0, pages = 0, fetches = 0;
    for (auto& m : mets) { bytes_out += m.link_bytes_out; bytes_in += m.link_bytes_in; msgs += m.link_msgs; faults += m.faults; pages += m.pages_fetched; fetches += m.fetch_calls; }
    std::map<std::string, double> per_kind; for (auto& m : mets) for (auto& l : m.layers) per_kind[l.kind] += l.ms / mets.size();
    double stream_ms = 0; for (auto& m : mets) stream_ms += m.weight_stream_ms / mets.size();

    if (!json) {
        std::printf("mode=%s transport=%s iterations=%d (warmup %d) prepare=%.1f ms\n", ex->mode(), transport.c_str(), runs, warmup, prepare_ms);
        std::printf("latency ms: median %.3f  mean %.3f  sd %.3f  p5 %.3f  p95 %.3f  p99 %.3f  max %.3f  cv %.1f%%\n",
                    pct(lat, 50), mean, sd, pct(lat, 5), pct(lat, 95), pct(lat, 99), *std::max_element(lat.begin(), lat.end()), mean > 0 ? 100 * sd / mean : 0);
        std::printf("per-inference link: %.1f B out, %.1f B in, %.1f msgs", (double)bytes_out / runs, (double)bytes_in / runs, (double)msgs / runs);
        if (mode == "pool") std::printf(", %.1f faults, %.1f pages, %.1f fetch RTTs", (double)faults / runs, (double)pages / runs, (double)fetches / runs);
        std::printf("\nper-kind ms:"); for (auto& kv : per_kind) std::printf(" %s %.3f", kv.first.c_str(), kv.second);
        if (p.streaming) std::printf("  [weight streaming %.3f ms/inference, %.1f MiB per inference, DDR2 slot 2x%.1f MiB]", stream_ms, (double)p.total_weight_bytes / 1048576.0, (double)p.slot_bytes / 1048576.0);
        std::printf("\n");
        std::printf("correctness: %d/%d bit-exact vs expected; max |err| vs FP32 %.4f (normalised %.4f)\n", runs - mismatches, runs, max_abs_err_fp32, max_ref > 0 ? max_abs_err_fp32 / max_ref : 0);
    } else {
        std::printf("{\"mode\":\"%s\",\"transport\":\"%s\",\"iterations\":%d,\"warmup\":%d,\"prepare_ms\":%.3f,"
                    "\"stats\":{\"count\":%zu,\"mean_ms\":%.4f,\"std_ms\":%.4f,\"min_ms\":%.4f,\"p5_ms\":%.4f,\"median_ms\":%.4f,\"p95_ms\":%.4f,\"p99_ms\":%.4f,\"max_ms\":%.4f,\"cv_pct\":%.2f},"
                    "\"link\":{\"bytes_out\":%.1f,\"bytes_in\":%.1f,\"msgs\":%.1f,\"faults\":%.1f,\"pages\":%.1f,\"fetch_rtts\":%.1f},"
                    "\"streaming\":%s,\"weight_stream_ms\":%.3f,\"total_weight_bytes\":%llu,\"correctness\":{\"bit_exact\":%d,\"total\":%d,\"max_abs_err_fp32\":%.6f,\"max_norm_err_fp32\":%.6f},\"per_kind_ms\":{",
                    ex->mode(), transport.c_str(), runs, warmup, prepare_ms, lat.size(), mean, sd, *std::min_element(lat.begin(), lat.end()), pct(lat, 5), pct(lat, 50), pct(lat, 95), pct(lat, 99), *std::max_element(lat.begin(), lat.end()), mean > 0 ? 100 * sd / mean : 0,
                    (double)bytes_out / runs, (double)bytes_in / runs, (double)msgs / runs, (double)faults / runs, (double)pages / runs, (double)fetches / runs,
                    p.streaming ? "true" : "false", stream_ms, (unsigned long long)p.total_weight_bytes,
                    runs - mismatches, runs, max_abs_err_fp32, max_ref > 0 ? max_abs_err_fp32 / max_ref : 0);
        bool first = true; for (auto& kv : per_kind) { std::printf("%s\"%s\":%.4f", first ? "" : ",", kv.first.c_str(), kv.second); first = false; }
        std::printf("}}\n");
    }
    if (t) edgecoh_transport_close(t);
    return mismatches ? 3 : 0;
}
