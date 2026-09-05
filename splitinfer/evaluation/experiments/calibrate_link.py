#!/usr/bin/env python3
"""Measure the EdgeCoh link: per-message RTT and sustained bandwidth in each
direction.  Writes a HardwareParams-override JSON for the partitioner so the
cost model's link figures are MEASURED, not assumed.

Uses splitinfer_v2's transport through a tiny C helper?  No — keeps it in
Python over the tty (usb) or via the emulator through ctypes, so the
measurement code is independent of the runtime under test.
"""
import argparse, ctypes, json, os, statistics, struct, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "build", "protocol", "libedgecoh.a")


def _load():
    # Build a tiny shared lib from the static one so ctypes can use it.
    so = "/tmp/libedgecoh_ctypes.so"
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(LIB):
        os.system(f"gcc -shared -o {so} -Wl,--whole-archive {LIB} -Wl,--no-whole-archive -lpthread 2>/dev/null")
    lib = ctypes.CDLL(so)
    lib.edgecoh_transport_open.restype = ctypes.c_void_p
    lib.edgecoh_transport_open_emul.restype = ctypes.c_void_p
    lib.edgecoh_transport_open_emul.argtypes = [ctypes.c_void_p]
    lib.edgecoh_transport_send.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.edgecoh_transport_recv.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.edgecoh_transport_close.argtypes = [ctypes.c_void_p]
    return lib


class EmulParams(ctypes.Structure):
    _fields_ = [("ddr2_bytes", ctypes.c_size_t), ("link_bytes_per_s", ctypes.c_double), ("link_rtt_s", ctypes.c_double),
                ("engine_clock_hz", ctypes.c_double), ("mac_per_cycle", ctypes.c_double)]


def hdr(t, tid=0, plen=0): return struct.pack("<BBHI", t, 0, tid, plen)


def recv_exact(lib, t, n, timeout=20000):
    buf = ctypes.create_string_buffer(n); got = 0
    while got < n:
        r = lib.edgecoh_transport_recv(t, ctypes.cast(ctypes.addressof(buf) + got, ctypes.c_char_p), n - got, timeout)
        if r <= 0: raise RuntimeError("recv timeout")
        got += r
    return buf.raw


def barrier_rtt(lib, t, n):
    lat = []
    for _ in range(n):
        m = hdr(0x03); t0 = time.perf_counter(); lib.edgecoh_transport_send(t, m, len(m)); recv_exact(lib, t, 8)
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def write_bw(lib, t, size, reps):
    data = bytes(range(256)) * (size // 256 + 1); data = data[:size]; ts = []
    for _ in range(reps):
        m = hdr(0x10, 0, 4 + size) + struct.pack("<I", 0x10000) + data
        t0 = time.perf_counter(); lib.edgecoh_transport_send(t, m, len(m)); recv_exact(lib, t, 8)
        ts.append(time.perf_counter() - t0)
    return ts


def read_bw(lib, t, size, reps):
    ts = []
    for _ in range(reps):
        m = hdr(0x11, 0, 8) + struct.pack("<II", 0x10000, size)
        t0 = time.perf_counter(); lib.edgecoh_transport_send(t, m, len(m)); recv_exact(lib, t, 8 + size)
        ts.append(time.perf_counter() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--transport", choices=["usb", "emul"], default="usb")
    ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--emul-bw", type=float, default=11520); ap.add_argument("--emul-rtt-ms", type=float, default=5)
    a = ap.parse_args()
    lib = _load()
    if a.transport == "emul":
        p = EmulParams(128 << 20, a.emul_bw, a.emul_rtt_ms / 1000, 0, 0.684); t = lib.edgecoh_transport_open_emul(ctypes.byref(p))
    else:
        t = lib.edgecoh_transport_open(0x0403, 0x6010)
    if not t: sys.exit("transport open failed")
    rtt = barrier_rtt(lib, t, a.reps)
    sizes = [64, 1024, 4096]
    res = {"transport": a.transport, "barrier_rtt_ms": {"median": statistics.median(rtt), "p95": sorted(rtt)[int(0.95 * (len(rtt) - 1))], "n": len(rtt)}, "write": {}, "read": {}}
    for s in sizes:
        w = write_bw(lib, t, s, max(3, a.reps // 4)); r = read_bw(lib, t, s, max(3, a.reps // 4))
        res["write"][s] = {"median_s": statistics.median(w), "bytes_per_s": s / statistics.median(w)}
        res["read"][s] = {"median_s": statistics.median(r), "bytes_per_s": s / statistics.median(r)}
    # Fit t = rtt + bytes / bw over the write sizes (least squares on two largest)
    s1, s2 = sizes[-2], sizes[-1]; t1, t2 = res["write"][s1]["median_s"], res["write"][s2]["median_s"]
    bw = (s2 - s1) / max(t2 - t1, 1e-9); fixed = t2 - s2 / bw
    res["fit"] = {"link_bw_bytes_per_s": bw, "per_msg_overhead_ms": fixed * 1000}
    res["hardware_params_override"] = {"link_bw_bytes_per_s": bw, "link_rtt_ms": max(statistics.median(rtt), fixed * 1000)}
    lib.edgecoh_transport_close(t)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f: json.dump(res, f, indent=1)
    print(f"RTT median {res['barrier_rtt_ms']['median']:.3f} ms; fitted link {bw:.0f} B/s, per-msg {fixed*1000:.3f} ms -> {a.out}")


if __name__ == "__main__":
    main()
