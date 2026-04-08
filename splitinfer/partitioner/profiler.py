"""GPU layer profiler — measures per-layer latency on the Jetson via onnxruntime."""

from __future__ import annotations
import json, sys, time, tempfile, warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_ONNX_NP_DTYPE = {
    TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64,
    TensorProto.FLOAT16: np.float16, TensorProto.INT8: np.int8,
    TensorProto.INT16: np.int16, TensorProto.INT32: np.int32,
    TensorProto.INT64: np.int64, TensorProto.UINT8: np.uint8,
    TensorProto.BOOL: np.bool_,
}


@dataclass
class LayerProfile:
    name: str
    op_type: str
    latency_ms: float
    std_ms: float
    peak_memory_bytes: int | None = None


@dataclass
class ProfileResult:
    profiles: list[LayerProfile] = field(default_factory=list)
    provider: str = ""

    def as_dict(self) -> dict[str, float]:
        return {p.name: p.latency_ms for p in self.profiles}

    def to_json(self) -> str:
        entries = [{"name": p.name, "op_type": p.op_type,
                    "latency_ms": round(p.latency_ms, 4),
                    "std_ms": round(p.std_ms, 4),
                    **({"peak_memory_bytes": p.peak_memory_bytes}
                       if p.peak_memory_bytes is not None else {})}
                   for p in self.profiles]
        total = sum(p.latency_ms for p in self.profiles)
        return json.dumps({"provider": self.provider, "layers": entries,
                           "total_ms": round(total, 4)}, indent=2)


def _resolve_shape(shape: list, default_dim: int = 1) -> list[int]:
    """Replace symbolic / zero / missing dims with *default_dim*."""
    out: list[int] = []
    for d in shape:
        if isinstance(d, int) and d > 0:
            out.append(d)
        else:
            out.append(default_dim)
    return out


def _tensor_shape(vi: onnx.ValueInfoProto) -> list:
    tp = vi.type.tensor_type
    if not tp.HasField("shape"):
        return []
    return [d.dim_value if d.dim_value > 0 else d.dim_param or 0
            for d in tp.shape.dim]


def _tensor_elem_type(vi: onnx.ValueInfoProto) -> int:
    return vi.type.tensor_type.elem_type


def _make_random_input(shape: list[int], elem_type: int) -> np.ndarray:
    dt = _ONNX_NP_DTYPE.get(elem_type, np.float32)
    if np.issubdtype(dt, np.integer):
        return np.zeros(shape, dtype=dt)
    return np.random.randn(*shape).astype(dt)


def _query_gpu_memory() -> int | None:
    """Return current GPU memory usage in bytes, or None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used)
    except Exception:
        return None


class GpuProfiler:
    """Profiles each ONNX graph node on the GPU (or CPU fallback)."""

    def __init__(self, model: onnx.ModelProto, *, warmup: int = 10,
                 runs: int = 100, default_dim: int = 1):
        self.model = model
        self.warmup = warmup
        self.runs = runs
        self.default_dim = default_dim
        # Shape-infer so every edge has type/shape info.
        try:
            self.model = onnx.shape_inference.infer_shapes(self.model)
        except Exception:
            pass
        self._value_info: dict[str, onnx.ValueInfoProto] = {}
        for vi in (*self.model.graph.value_info, *self.model.graph.input,
                   *self.model.graph.output):
            self._value_info[vi.name] = vi
        self._init_data: dict[str, onnx.TensorProto] = {
            init.name: init for init in self.model.graph.initializer}

    # ------------------------------------------------------------------
    def _build_single_node_model(self, node: onnx.NodeProto) -> onnx.ModelProto | None:
        """Build a minimal single-op model for *node*."""
        inputs: list[onnx.ValueInfoProto] = []
        initializers: list[onnx.TensorProto] = []

        for inp_name in node.input:
            if not inp_name:
                continue
            if inp_name in self._init_data:
                initializers.append(self._init_data[inp_name])
                continue
            vi = self._value_info.get(inp_name)
            if vi is None:
                return None  # can't resolve — skip layer
            inputs.append(vi)

        outputs: list[onnx.ValueInfoProto] = []
        for out_name in node.output:
            if not out_name:
                continue
            vi = self._value_info.get(out_name)
            if vi is not None:
                outputs.append(vi)
            else:
                # Fallback: untyped output placeholder
                outputs.append(helper.make_tensor_value_info(out_name,
                               TensorProto.FLOAT, None))

        graph = helper.make_graph([node], f"profile_{node.name}",
                                  inputs, outputs, initializers)
        mp = helper.make_model(graph)
        mp.ir_version = self.model.ir_version
        del mp.opset_import[:]
        for oi in self.model.opset_import:
            new = mp.opset_import.add()
            new.CopyFrom(oi)
        try:
            onnx.checker.check_model(mp)
        except onnx.checker.ValidationError:
            pass  # best-effort
        return mp

    # ------------------------------------------------------------------
    def _make_feeds(self, inputs: list[onnx.ValueInfoProto]) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        for vi in inputs:
            raw_shape = _tensor_shape(vi)
            shape = _resolve_shape(raw_shape, self.default_dim)
            elem_type = _tensor_elem_type(vi)
            feeds[vi.name] = _make_random_input(shape if shape else [1], elem_type)
        return feeds

    # ------------------------------------------------------------------
    @staticmethod
    def _pick_provider() -> tuple[str, list[str]]:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return "CPUExecutionProvider", ["CPUExecutionProvider"]

    # ------------------------------------------------------------------
    def profile(self) -> ProfileResult:
        import onnxruntime as ort

        provider_name, providers = self._pick_provider()
        result = ProfileResult(provider=provider_name)

        for node in self.model.graph.node:
            sub_model = self._build_single_node_model(node)
            if sub_model is None:
                warnings.warn(f"Skipping layer {node.name!r}: cannot build subgraph")
                continue

            # Write to temp file — ort.InferenceSession needs a path or bytes.
            raw = sub_model.SerializeToString()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # suppress warnings
            try:
                sess = ort.InferenceSession(raw, opts, providers=providers)
            except Exception as exc:
                warnings.warn(f"Skipping layer {node.name!r}: {exc}")
                continue

            feed_vis = [vi for vi in (*self.model.graph.input,
                                      *self.model.graph.value_info)
                        if vi.name in {i for i in node.input
                                       if i not in self._init_data and i}]
            feeds = self._make_feeds(feed_vis)

            # --- warmup ---
            for _ in range(self.warmup):
                sess.run(None, feeds)

            # --- measure ---
            mem_before = _query_gpu_memory()
            times: list[float] = []
            for _ in range(self.runs):
                t0 = time.perf_counter()
                sess.run(None, feeds)
                times.append((time.perf_counter() - t0) * 1000.0)
            mem_after = _query_gpu_memory()

            peak_mem: int | None = None
            if mem_before is not None and mem_after is not None:
                peak_mem = max(mem_after - mem_before, 0)

            arr = np.array(times)
            result.profiles.append(LayerProfile(
                name=node.name, op_type=node.op_type,
                latency_ms=float(np.median(arr)),
                std_ms=float(np.std(arr)),
                peak_memory_bytes=peak_mem,
            ))

        return result


# ── CLI ──────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Profile ONNX model layers on GPU")
    parser.add_argument("model", type=str, help="Path to ONNX model")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--default-dim", type=int, default=1,
                        help="Value for dynamic/unknown dims")
    args = parser.parse_args()

    model = onnx.load(args.model)
    profiler = GpuProfiler(model, warmup=args.warmup, runs=args.runs,
                           default_dim=args.default_dim)
    result = profiler.profile()
    print(result.to_json())


if __name__ == "__main__":
    main()
