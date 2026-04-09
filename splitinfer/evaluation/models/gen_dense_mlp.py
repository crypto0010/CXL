#!/usr/bin/env python3
"""gen_dense_mlp.py — synthesize a multi-GB dense MLP for memory-capacity
experiments.

Why this exists:
    DLRM-style models (sparse gather lookups) are mmap-friendly in
    onnxruntime: even a 10 GB DLRM can be "loaded" with only a few MB
    of resident memory because each inference touches just a handful
    of embedding rows.  For the SplitInfer capacity-unlock story we
    need a model where EVERY inference touches ALL weights, which
    means a dense multiply.  A single-layer GEMM with 2 GB of weights
    forces the full 2 GB to be resident during inference (CPU has to
    stream every weight through the FMA units), producing honest
    memory pressure on the Jetson.

Architecture:
    Input:  [1, D_in]
    Layer k (k=0..num_layers-1):
        y = ReLU(x @ W_k + b_k)
    W_k ∈ [in_dim, out_dim]  FP32

    By default we generate a 3-layer MLP with ~1.8 GB of weights per
    layer (close to ONNX's single-file 2 GB cap), giving ~5.4 GB total.
    With --streaming the weights live in an external .data sidecar and
    the generator never holds more than one chunk in RAM.

Usage:
    gen_dense_mlp.py <out.onnx>                      # default ~5 GB
    gen_dense_mlp.py <out.onnx> --dim 10000          # 10K->10K->10K->1K
    gen_dense_mlp.py <out.onnx> --dim 12000 --streaming --num-layers 3
"""

import argparse
import os
import sys

import onnx
from onnx import helper, TensorProto


def _make_external_tensor(name: str, dims, offset: int, length: int,
                          data_basename: str) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.data_type = TensorProto.FLOAT
    t.dims.extend(dims)
    t.data_location = TensorProto.EXTERNAL
    for k, v in [("location", data_basename),
                 ("offset",   str(offset)),
                 ("length",   str(length))]:
        e = t.external_data.add()
        e.key = k
        e.value = v
    return t


def build_dense_mlp_streaming(out_path: str,
                              input_dim: int,
                              hidden_dim: int,
                              output_dim: int,
                              num_layers: int,
                              batch: int) -> None:
    """Write a dense MLP with weights in an external sidecar."""
    data_path = out_path + ".data"
    data_basename = os.path.basename(data_path)

    CHUNK = 4 * 1024 * 1024
    import numpy as np
    zero_chunk = np.zeros(CHUNK // 4, dtype=np.float32).tobytes()

    offsets: dict[str, tuple[int, int]] = {}

    print(f"  [streaming] writing weights to {data_basename}...")
    with open(data_path, "wb") as fd:
        def write_zero_tensor(name: str, nbytes: int) -> None:
            off = fd.tell()
            if off % 8 != 0:
                pad = 8 - (off % 8)
                fd.write(b"\x00" * pad)
                off += pad
            remaining = nbytes
            while remaining > 0:
                write_n = min(remaining, CHUNK)
                if write_n == CHUNK:
                    fd.write(zero_chunk)
                else:
                    fd.write(b"\x00" * write_n)
                remaining -= write_n
            offsets[name] = (off, nbytes)

        # Layer dims: [input_dim, hidden_dim, hidden_dim, ..., hidden_dim, output_dim]
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        total_weight_bytes = 0
        for j in range(len(dims) - 1):
            in_d, out_d = dims[j], dims[j + 1]
            wb = in_d * out_d * 4
            bb =         out_d * 4
            write_zero_tensor(f"W{j}", wb)
            write_zero_tensor(f"b{j}", bb)
            total_weight_bytes += wb + bb
            print(f"    layer {j}: W=[{in_d},{out_d}] ({wb/1024**3:.2f} GB)")

    data_bytes = os.path.getsize(data_path)
    print(f"  [streaming] wrote {data_bytes/1024**3:.2f} GB to {data_basename}")

    # Build the model
    initializers: list[TensorProto] = []
    nodes = []
    prev = "input"
    dims_list = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

    for j in range(len(dims_list) - 1):
        in_d, out_d = dims_list[j], dims_list[j + 1]
        off_w, len_w = offsets[f"W{j}"]
        off_b, len_b = offsets[f"b{j}"]
        initializers.append(
            _make_external_tensor(f"W{j}", [in_d, out_d], off_w, len_w, data_basename))
        initializers.append(
            _make_external_tensor(f"b{j}", [out_d],       off_b, len_b, data_basename))

        mm_out  = f"mm{j}"
        add_out = f"add{j}"
        nodes.append(helper.make_node("MatMul", [prev, f"W{j}"], [mm_out],  name=f"matmul{j}"))
        nodes.append(helper.make_node("Add",    [mm_out, f"b{j}"], [add_out], name=f"add{j}"))

        if j < len(dims_list) - 2:
            relu_out = f"relu{j}"
            nodes.append(helper.make_node("Relu", [add_out], [relu_out], name=f"relu{j}"))
            prev = relu_out
        else:
            nodes.append(helper.make_node("Identity", [add_out], ["output"], name="output_id"))
            prev = "output"

    graph_inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, input_dim])]
    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, output_dim])]
    graph = helper.make_graph(nodes, "dense_mlp_streaming",
                              graph_inputs, graph_outputs,
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, out_path)

    header_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path} ({header_mb:.2f} MB header + "
          f"{data_bytes/1024**3:.2f} GB external data)")
    print(f"  Architecture: {' -> '.join(str(d) for d in dims_list)}  |  batch={batch}")


def main():
    parser = argparse.ArgumentParser(description="Generate dense-MLP ONNX model for capacity experiments.")
    parser.add_argument("output", help="Path to write the .onnx file")
    parser.add_argument("--input-dim",   type=int, default=10000)
    parser.add_argument("--hidden-dim",  type=int, default=20000)
    parser.add_argument("--output-dim",  type=int, default=1000)
    parser.add_argument("--num-layers",  type=int, default=3)
    parser.add_argument("--batch",       type=int, default=1)
    args = parser.parse_args()
    build_dense_mlp_streaming(
        out_path=args.output,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
