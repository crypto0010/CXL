"""INT8 lowering: ONNX model + placement manifest -> DDR2 image + program.

This is what makes the runtime *compute* rather than dispatch empty
messages.  It produces three artifacts:

  program.json  — per-layer records with DDR2 addresses, dimensions and
                  requantisation parameters, in topological order.
  image.bin     — every DDR2-resident constant (INT8 tables / weights, INT32
                  biases), with segment descriptors in program.json.
  vectors.bin   — N test inputs with expected outputs, so every execution
                  mode (NMC on FPGA, NMC on the emulator, POOL through
                  cxlwin, host reference) is checked for bit-exact integer
                  agreement and for FP32 accuracy against onnxruntime.

Numerics: symmetric per-tensor INT8, INT32 accumulate, fixed-point requant
(mult * 2^-shift) with ReLU fused — identical to the FPGA EPILOGUE engine,
so host and device produce the same integers.

Supported ops (the scope of the paper's workloads): Gather (embedding),
Concat, MatMul (+ Add bias, + Relu, fused), Identity.  Anything else raises.

Alignment rules (from fpga/src): all DDR2 regions 16-byte aligned; FC weight
rows [M x K] INT8 with K padded to 16 and M padded to 16; embedding dim
bytes a multiple of 16.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import numpy_helper

ALIGN = 16
DDR2_BYTES = 128 * 1024 * 1024


def _pad16(n: int) -> int:
    return (n + 15) // 16 * 16


def _align(a: int) -> int:
    return (a + ALIGN - 1) // ALIGN * ALIGN


def _requant_params(ratio: float) -> tuple[int, int]:
    """mult, shift such that mult * 2^-shift ~= ratio, mult < 2^15."""
    if ratio <= 0:
        return 0, 0
    shift = int(math.floor(math.log2(32767.0 / ratio)))
    shift = max(0, min(31, shift))
    mult = int(round(ratio * (1 << shift)))
    mult = max(0, min(32767, mult))
    return mult, shift


def _sat8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -128, 127).astype(np.int8)


@dataclass
class Segment:
    addr: int
    offset: int
    length: int
    layer: str
    kind: str


@dataclass
class Layer:
    name: str
    kind: str                # gather | concat | fc
    placement: str
    params: dict = field(default_factory=dict)


class _Allocator:
    def __init__(self, base: int = 0):
        self.next = _align(base)

    def alloc(self, nbytes: int) -> int:
        a = self.next
        self.next = _align(a + max(nbytes, 1))
        if self.next > DDR2_BYTES:
            raise MemoryError(f"DDR2 layout exceeds {DDR2_BYTES} bytes")
        return a


class Lowering:
    def __init__(self, model: onnx.ModelProto, manifest: dict, calibration_inputs: int = 16,
                 seed: int = 0):
        self.model = model
        self.graph = model.graph
        self.manifest = manifest
        self.placement = {l["name"]: l["placement"] for l in manifest["layers"]}
        self.inits = {i.name: numpy_helper.to_array(i) for i in self.graph.initializer}
        self.rng = np.random.default_rng(seed)
        self.n_calib = calibration_inputs
        self.image = bytearray()
        self.segments: list[Segment] = []
        self.layers: list[Layer] = []
        self.inputs: list[dict] = []
        self.alloc = _Allocator(0)

    # ── FP32 reference via onnxruntime ────────────────────────────────────

    def _make_inputs(self, n: int) -> list[dict[str, np.ndarray]]:
        feeds = []
        for _ in range(n):
            f = {}
            for vi in self.graph.input:
                if vi.name in self.inits:
                    continue
                dims = [d.dim_value if d.dim_value > 0 else 1 for d in vi.type.tensor_type.shape.dim]
                et = vi.type.tensor_type.elem_type
                if et == onnx.TensorProto.FLOAT:
                    f[vi.name] = self.rng.standard_normal(dims).astype(np.float32)
                elif et in (onnx.TensorProto.INT64, onnx.TensorProto.INT32):
                    # index into the table this input feeds
                    rows = self._table_rows_for_index(vi.name)
                    f[vi.name] = self.rng.integers(0, rows, size=dims).astype(
                        np.int64 if et == onnx.TensorProto.INT64 else np.int32)
                else:
                    raise ValueError(f"unsupported input dtype {et} for {vi.name}")
            feeds.append(f)
        return feeds

    def _table_rows_for_index(self, idx_name: str) -> int:
        for n in self.graph.node:
            if n.op_type == "Gather" and n.input[1] == idx_name:
                return int(self.inits[n.input[0]].shape[0])
        return 1

    def _run_fp32(self, feeds: list[dict], want: list[str]) -> list[dict[str, np.ndarray]]:
        import onnxruntime as ort
        m = onnx.ModelProto(); m.CopyFrom(self.model)
        existing = {o.name for o in m.graph.output}
        for w in want:
            if w not in existing:
                m.graph.output.append(onnx.ValueInfoProto(name=w))
        sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
        outs = []
        for f in feeds:
            vals = sess.run(want, f)
            outs.append(dict(zip(want, vals)))
        return outs

    # ── Image helpers ─────────────────────────────────────────────────────

    def _place(self, data: np.ndarray, layer: str, kind: str) -> int:
        b = data.tobytes()
        addr = self.alloc.alloc(len(b))
        off = len(self.image)
        self.image += b
        pad = _align(len(self.image)) - len(self.image)
        self.image += bytes(pad)
        self.segments.append(Segment(addr, off, len(b), layer, kind))
        return addr

    # ── Lowering ──────────────────────────────────────────────────────────

    def lower(self) -> dict:
        nodes = list(self.graph.node)
        # Fuse MatMul -> Add -> Relu chains.
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for n in nodes:
            for i in n.input:
                consumers.setdefault(i, []).append(n)
        fused_into: dict[str, str] = {}      # node name -> fc layer name
        fc_chains: dict[str, dict] = {}
        for n in nodes:
            if n.op_type != "MatMul":
                continue
            chain = {"matmul": n, "add": None, "relu": None}
            out = n.output[0]
            c = consumers.get(out, [])
            if len(c) == 1 and c[0].op_type == "Add" and any(i in self.inits for i in c[0].input):
                chain["add"] = c[0]; fused_into[c[0].name] = n.name; out = c[0].output[0]
                c = consumers.get(out, [])
            if len(c) == 1 and c[0].op_type == "Relu":
                chain["relu"] = c[0]; fused_into[c[0].name] = n.name; out = c[0].output[0]
            chain["out"] = out
            fc_chains[n.name] = chain

        # Calibration: activation ranges at every fc input / concat output / graph input.
        act_names = [c["matmul"].input[0] for c in fc_chains.values()]
        for n in nodes:
            if n.op_type == "Concat":
                act_names.append(n.output[0])
        graph_inputs = [vi.name for vi in self.graph.input if vi.name not in self.inits]
        act_names += [g for g in graph_inputs
                      if next(vi for vi in self.graph.input if vi.name == g).type.tensor_type.elem_type == onnx.TensorProto.FLOAT]
        act_names = list(dict.fromkeys(act_names))
        feeds = self._make_inputs(self.n_calib)
        outs = self._run_fp32(feeds, act_names + [self.graph.output[0].name])
        maxabs = {a: max(float(np.max(np.abs(o[a]))) for o in outs) or 1.0 for a in act_names}
        scale = {a: maxabs[a] / 127.0 for a in act_names}

        # Concat groups share one scale: embedding tables and the dense input
        # that feed a concat are quantised with the concat output's scale.
        tensor_scale: dict[str, float] = dict(scale)
        concat_nodes = [n for n in nodes if n.op_type == "Concat"]
        for cn in concat_nodes:
            s = scale[cn.output[0]]
            for i in cn.input:
                tensor_scale[i] = s

        # Walk the graph in order and emit layers.
        produced_addr: dict[str, int] = {}     # tensor name -> DDR2 addr of INT8 activation
        produced_len: dict[str, int] = {}
        pending_concat_parts: dict[str, list] = {}
        for n in nodes:
            place = self.placement.get(n.name, "gpu")
            if n.op_type == "Gather":
                table = self.inits[n.input[0]].astype(np.float32)
                rows, dim = table.shape
                dim_bytes = _pad16(dim)          # INT8 per element
                s = tensor_scale.get(n.output[0], maxabs.get(n.output[0], float(np.max(np.abs(table))) or 1.0) / 127.0)
                tq = np.zeros((rows, dim_bytes), dtype=np.int8)
                tq[:, :dim] = _sat8(np.round(table / s))
                t_addr = self._place(tq, n.name, "table")
                idx_name = n.input[1]
                idx_addr = self.alloc.alloc(16)
                out_addr = self.alloc.alloc(dim_bytes)
                self.inputs.append({"name": idx_name, "kind": "index", "ddr2_addr": idx_addr, "count": 1,
                                    "rows": int(rows)})
                self.layers.append(Layer(n.name, "gather", place, {
                    "table_addr": t_addr, "rows": int(rows), "dim": int(dim), "dim_bytes": dim_bytes,
                    "idx_addr": idx_addr, "n_idx": 1, "out_addr": out_addr, "scale": s,
                    "index_input": idx_name}))
                produced_addr[n.output[0]] = out_addr; produced_len[n.output[0]] = dim
            elif n.op_type == "Concat":
                parts = []
                total = 0
                for i in n.input:
                    if i in produced_addr:
                        parts.append({"src": "layer", "tensor": i, "addr": produced_addr[i], "len": produced_len[i]})
                        total += produced_len[i]
                    else:                          # graph input (dense)
                        vi = next(v for v in self.graph.input if v.name == i)
                        ln = int(np.prod([d.dim_value or 1 for d in vi.type.tensor_type.shape.dim]))
                        addr = self.alloc.alloc(_pad16(ln))
                        self.inputs.append({"name": i, "kind": "dense", "ddr2_addr": addr, "count": ln,
                                            "scale": tensor_scale[i]})
                        parts.append({"src": "input", "tensor": i, "addr": addr, "len": ln})
                        total += ln
                out_addr = self.alloc.alloc(_pad16(total))
                self.layers.append(Layer(n.name, "concat", place, {
                    "parts": parts, "out_addr": out_addr, "len": total, "scale": scale[n.output[0]]}))
                produced_addr[n.output[0]] = out_addr; produced_len[n.output[0]] = total
            elif n.op_type == "MatMul":
                ch = fc_chains[n.name]
                x_name = n.input[0]
                W = self.inits[n.input[1]].astype(np.float32)          # [K, N]
                K, N = W.shape
                if x_name not in produced_addr:
                    # graph input feeding the first FC directly
                    vi = next(v for v in self.graph.input if v.name == x_name)
                    addr = self.alloc.alloc(_pad16(K))
                    self.inputs.append({"name": x_name, "kind": "dense", "ddr2_addr": addr, "count": K,
                                        "scale": tensor_scale[x_name]})
                    produced_addr[x_name] = addr; produced_len[x_name] = K
                s_x = tensor_scale[x_name]
                s_w = float(np.max(np.abs(W))) / 127.0 or 1.0
                Wt = np.zeros((_pad16(N), _pad16(K)), dtype=np.int8)     # [M_pad x K_pad]
                Wt[:N, :K] = _sat8(np.round(W.T / s_w))
                w_addr = self._place(Wt, n.name, "weight")
                b = np.zeros(_pad16(N), dtype=np.int32)
                if ch["add"] is not None:
                    bname = next(i for i in ch["add"].input if i in self.inits)
                    b[:N] = np.round(self.inits[bname].astype(np.float32).reshape(-1) / (s_w * s_x)).astype(np.int32)
                b_addr = self._place(b, n.name, "bias")
                acc_addr = self.alloc.alloc(_pad16(N) * 4)
                out_name = ch["out"]
                final = out_name == self.graph.output[0].name or (
                    len(consumers.get(out_name, [])) == 1 and consumers[out_name][0].op_type == "Identity"
                    and consumers[out_name][0].output[0] == self.graph.output[0].name)
                if final:
                    out_addr = acc_addr; mult = shift = 0; s_y = s_w * s_x
                else:
                    out_addr = self.alloc.alloc(_pad16(N))
                    s_y = tensor_scale.get(out_name, scale.get(out_name))
                    if s_y is None:
                        # output feeds something we did not calibrate; find the consumer's scale
                        s_y = next(scale[c.input[0]] for c in consumers.get(out_name, []) if c.input[0] in scale)
                    mult, shift = _requant_params(s_w * s_x / s_y)
                self.layers.append(Layer(n.name, "fc", place, {
                    "in_addr": produced_addr[x_name], "K": int(K), "K_pad": _pad16(K),
                    "N": int(N), "M_pad": _pad16(N), "w_addr": w_addr, "b_addr": b_addr,
                    "acc_addr": acc_addr, "out_addr": out_addr,
                    "mult": mult, "shift": shift, "relu": int(ch["relu"] is not None),
                    "final": bool(final), "in_scale": s_x, "w_scale": s_w, "out_scale": s_y,
                    "has_bias": ch["add"] is not None}))
                produced_addr[out_name] = out_addr; produced_len[out_name] = N
            elif n.op_type in ("Add", "Relu") and n.name in fused_into:
                continue
            elif n.op_type == "Identity":
                produced_addr[n.output[0]] = produced_addr[n.input[0]]
                produced_len[n.output[0]] = produced_len[n.input[0]]
            else:
                raise NotImplementedError(f"lowering does not support {n.op_type} ({n.name})")

        last_fc = next(l for l in reversed(self.layers) if l.kind == "fc")
        program = {
            "version": "2.0",
            "batch": 1,
            "ddr2": {"image_bytes": len(self.image), "layout_end": self.alloc.next},
            "segments": [s.__dict__ for s in self.segments],
            "inputs": self.inputs,
            "layers": [{"name": l.name, "kind": l.kind, "placement": l.placement, **l.params}
                       for l in self.layers],
            "output": {"layer": last_fc.name, "count": last_fc.params["N"],
                       "dequant_scale": last_fc.params["out_scale"], "int32": last_fc.params["final"]},
        }
        self.program = program
        self._feeds = feeds
        self._fp32_out = [o[self.graph.output[0].name] for o in outs]
        return program

    # ── Reference execution (integer, same semantics as the FPGA) ─────────

    def run_reference(self, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Execute the lowered program on the host with exact integer semantics.
        Returns (output INT32 acc+bias vector, per-layer INT8 activations)."""
        mem: dict[str, np.ndarray] = {}
        acts: dict[str, np.ndarray] = {}
        prog = self.program
        for inp in prog["inputs"]:
            v = feed[inp["name"]]
            if inp["kind"] == "index":
                mem[inp["name"]] = v.reshape(-1).astype(np.int32)
            else:
                mem[inp["name"]] = _sat8(np.round(v.reshape(-1).astype(np.float32) / inp["scale"]))
        by_addr: dict[int, np.ndarray] = {}
        for inp in prog["inputs"]:
            if inp["kind"] == "dense":
                by_addr[inp["ddr2_addr"]] = mem[inp["name"]]
        for L in prog["layers"]:
            if L["kind"] == "gather":
                seg = next(s for s in self.segments if s.layer == L["name"] and s.kind == "table")
                tq = np.frombuffer(self.image[seg.offset:seg.offset + seg.length], dtype=np.int8).reshape(L["rows"], L["dim_bytes"])
                idx = int(mem[L["index_input"]][0])
                row = tq[idx, :L["dim"]].copy()
                by_addr[L["out_addr"]] = row; acts[L["name"]] = row
            elif L["kind"] == "concat":
                parts = []
                for p in L["parts"]:
                    parts.append(by_addr[p["addr"]] if p["src"] == "layer" else mem[p["tensor"]])
                v = np.concatenate(parts).astype(np.int8)
                by_addr[L["out_addr"]] = v; acts[L["name"]] = v
            elif L["kind"] == "fc":
                x = by_addr[L["in_addr"]]
                xk = np.zeros(L["K_pad"], dtype=np.int8); xk[:len(x)] = x
                wseg = next(s for s in self.segments if s.layer == L["name"] and s.kind == "weight")
                W = np.frombuffer(self.image[wseg.offset:wseg.offset + wseg.length], dtype=np.int8).reshape(L["M_pad"], L["K_pad"])
                bseg = next(s for s in self.segments if s.layer == L["name"] and s.kind == "bias")
                b = np.frombuffer(self.image[bseg.offset:bseg.offset + bseg.length], dtype=np.int32)
                acc = (W.astype(np.int32) @ xk.astype(np.int32)).astype(np.int32)
                if L["final"]:
                    out = (acc.astype(np.int64) + b.astype(np.int64))[:L["N"]].astype(np.int32)
                    by_addr[L["out_addr"]] = out; acts[L["name"]] = out
                else:
                    s = (acc.astype(np.int64) + b.astype(np.int64)) * L["mult"]
                    s = s >> L["shift"]
                    if L["relu"]:
                        s = np.maximum(s, 0)
                    y = _sat8(s)[:L["N"]]
                    by_addr[L["out_addr"]] = y; acts[L["name"]] = y
        last = prog["output"]["layer"]
        return acts[last], acts

    # ── Artifacts ─────────────────────────────────────────────────────────

    def write(self, out_dir: str, n_vectors: int = 8) -> dict:
        import os
        os.makedirs(out_dir, exist_ok=True)
        prog = self.program
        feeds = self._feeds[:n_vectors]
        fp32 = self._fp32_out[:n_vectors]
        vec = bytearray(); entries = []
        max_rel_err = 0.0
        for f, ref in zip(feeds, fp32):
            e = {"inputs": [], "expect_off": 0, "expect_count": 0}
            for inp in prog["inputs"]:
                v = f[inp["name"]].reshape(-1)
                if inp["kind"] == "index":
                    data = v.astype(np.int32).tobytes()
                else:
                    data = _sat8(np.round(v.astype(np.float32) / inp["scale"])).tobytes()
                e["inputs"].append({"name": inp["name"], "off": len(vec), "len": len(data)})
                vec += data
            out, _ = self.run_reference(f)
            e["expect_off"] = len(vec); e["expect_count"] = int(out.size)
            vec += out.astype(np.int32).tobytes()
            deq = out.astype(np.float64) * prog["output"]["dequant_scale"]
            e["fp32_ref"] = [float(x) for x in ref.reshape(-1)]
            e["int8_dequant"] = [float(x) for x in deq]
            # normalised error: max |err| / max |ref| over the vector (relative
            # error is meaningless for near-zero outputs)
            rel = float(np.max(np.abs(deq - ref.reshape(-1))) / (np.max(np.abs(ref.reshape(-1))) + 1e-9))
            max_rel_err = max(max_rel_err, rel)
            entries.append(e)
        prog["vectors"] = {"file": "vectors.bin", "count": len(entries), "entries": entries,
                           "max_norm_err_vs_fp32": max_rel_err}
        with open(os.path.join(out_dir, "program.json"), "w") as fh:
            json.dump(prog, fh, indent=1)
        with open(os.path.join(out_dir, "image.bin"), "wb") as fh:
            fh.write(self.image)
        with open(os.path.join(out_dir, "vectors.bin"), "wb") as fh:
            fh.write(vec)
        return prog


def lower_model(model_path: str, manifest_path: str, out_dir: str, n_vectors: int = 8,
                calibration_inputs: int = 16, seed: int = 0) -> dict:
    model = onnx.load(model_path)
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    lw = Lowering(model, manifest, calibration_inputs=calibration_inputs, seed=seed)
    lw.lower()
    return lw.write(out_dir, n_vectors=n_vectors)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="INT8 lowering -> DDR2 image + program")
    ap.add_argument("model"); ap.add_argument("manifest"); ap.add_argument("out_dir")
    ap.add_argument("--vectors", type=int, default=8)
    ap.add_argument("--calib", type=int, default=16)
    a = ap.parse_args()
    p = lower_model(a.model, a.manifest, a.out_dir, a.vectors, a.calib)
    print(f"layers={len(p['layers'])} image={p['ddr2']['image_bytes']/2**20:.2f} MiB "
          f"layout_end={p['ddr2']['layout_end']/2**20:.2f} MiB "
          f"max_norm_err_vs_fp32={p['vectors']['max_norm_err_vs_fp32']:.4f}")
