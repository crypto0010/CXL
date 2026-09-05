#!/usr/bin/env python3
"""Generate a synthetic DLRM-style ONNX model.

Usage:
  gen_dlrm.py <output_path>                       # default: small (~14 MB)
  gen_dlrm.py <output_path> --rows 200000         # scaled (~1.3 GB)
  gen_dlrm.py <output_path> --tables 26 --rows 100000 --cols 64

Architecture:
  N embedding tables x R rows x C-dim (Gather)
  Dense input: 128 features
  Top MLP: (N*C + 128) -> 1024 -> 256 -> 1

The "small" default (26 x 1000 x 64 = 14 MB) is for smoke tests.
The "scaled" preset (26 x 200000 x 64 = 1.3 GB) is for E1's
capability-unlock story — a model where the embedding tables alone
strain the Jetson Orin Nano's GPU memory (8 GB LPDDR5 shared).
"""

import sys
import os
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_dlrm_streaming(out_path: str,
                         num_tables: int,
                         table_rows: int,
                         table_cols: int,
                         dense_dim: int,
                         batch_size: int) -> None:
    """Memory-efficient DLRM generator for multi-GB models.

    Writes weight tensors directly to an external .onnx.data sidecar
    file as they are generated, referencing them from the main .onnx
    via TensorProto.external_data.  Peak memory usage is O(chunk_size)
    rather than O(total_weights), so we can build 10+ GB models without
    OOM-killing the generator process.

    ONNX external data format:
        TensorProto {
            name: "emb_table_0"
            dims: [rows, cols]
            data_type: FLOAT
            data_location: EXTERNAL
            external_data: [
                { key: "location", value: "model.onnx.data" }
                { key: "offset",   value: "<byte offset>" }
                { key: "length",   value: "<byte length>" }
            ]
        }
    """
    BATCH = batch_size
    data_path = out_path + ".data"
    data_basename = os.path.basename(data_path)

    # ─── Pass 1: open sidecar file and write all weight blobs ─────────────────
    print(f"  [streaming] writing weights to {data_basename}...")
    CHUNK = 4 * 1024 * 1024  # 4 MB write chunks
    zero_chunk = np.zeros(CHUNK // 4, dtype=np.float32).tobytes()  # 4 MB of FP32 zeros

    offsets: dict[str, tuple[int, int]] = {}  # name -> (offset, length)

    with open(data_path, "wb") as fd:
        def write_zero_tensor(name: str, nbytes: int) -> None:
            off = fd.tell()
            # Align to 8 bytes for ONNX compatibility
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

        # Embedding tables: num_tables × (table_rows × table_cols × 4B)
        table_bytes = table_rows * table_cols * 4
        for i in range(num_tables):
            write_zero_tensor(f"emb_table_{i}", table_bytes)

        # MLP weights: (NT*TC + DD)->1024->256->1
        mlp_dims = [num_tables * table_cols + dense_dim, 1024, 256, 1]
        for j in range(len(mlp_dims) - 1):
            in_dim, out_dim = mlp_dims[j], mlp_dims[j + 1]
            write_zero_tensor(f"mlp_w{j}", in_dim * out_dim * 4)
            write_zero_tensor(f"mlp_b{j}",           out_dim * 4)

    total_data_bytes = os.path.getsize(data_path)
    print(f"  [streaming] wrote {total_data_bytes / (1024**3):.2f} GB to {data_basename}")

    # ─── Pass 2: build the ONNX model referencing external data ──────────────
    print(f"  [streaming] building main .onnx with external data references...")

    def make_external_tensor(name: str, dims: list[int]) -> TensorProto:
        off, length = offsets[name]
        t = TensorProto()
        t.name = name
        t.data_type = TensorProto.FLOAT
        t.dims.extend(dims)
        t.data_location = TensorProto.EXTERNAL
        # external_data entries are StringStringEntryProto
        for k, v in [("location", data_basename),
                     ("offset",   str(off)),
                     ("length",   str(length))]:
            e = t.external_data.add()
            e.key = k
            e.value = v
        return t

    initializers: list[TensorProto] = []
    nodes = []

    # Embedding tables + gathers
    emb_outputs = []
    for i in range(num_tables):
        initializers.append(make_external_tensor(
            f"emb_table_{i}", [table_rows, table_cols]))
        nodes.append(helper.make_node(
            "Gather",
            inputs=[f"emb_table_{i}", f"emb_idx_{i}"],
            outputs=[f"emb_out_{i}"],
            name=f"gather_{i}",
            axis=0,
        ))
        emb_outputs.append(f"emb_out_{i}")

    # Concat
    nodes.append(helper.make_node(
        "Concat",
        inputs=emb_outputs + ["dense_input"],
        outputs=["concat_out"],
        name="concat_all",
        axis=1,
    ))

    # MLP
    mlp_dims = [num_tables * table_cols + dense_dim, 1024, 256, 1]
    prev = "concat_out"
    for j in range(len(mlp_dims) - 1):
        in_dim, out_dim = mlp_dims[j], mlp_dims[j + 1]
        initializers.append(make_external_tensor(f"mlp_w{j}", [in_dim, out_dim]))
        initializers.append(make_external_tensor(f"mlp_b{j}", [out_dim]))

        nodes.append(helper.make_node("MatMul", [prev, f"mlp_w{j}"], [f"mlp_mm{j}"],  name=f"mlp_matmul{j}"))
        nodes.append(helper.make_node("Add",    [f"mlp_mm{j}", f"mlp_b{j}"], [f"mlp_add{j}"], name=f"mlp_add_op{j}"))
        if j < len(mlp_dims) - 2:
            nodes.append(helper.make_node("Relu", [f"mlp_add{j}"], [f"mlp_relu{j}"], name=f"mlp_relu{j}"))
            prev = f"mlp_relu{j}"
        else:
            nodes.append(helper.make_node("Identity", [f"mlp_add{j}"], ["output"], name="output_id"))
            prev = "output"

    graph_inputs = [helper.make_tensor_value_info("dense_input", TensorProto.FLOAT, [BATCH, dense_dim])]
    for i in range(num_tables):
        graph_inputs.append(
            helper.make_tensor_value_info(f"emb_idx_{i}", TensorProto.INT64, [BATCH]))
    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [BATCH, 1])]

    graph = helper.make_graph(nodes, "dlrm_synthetic_streaming",
                              graph_inputs, graph_outputs,
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, out_path)

    header_mb = os.path.getsize(out_path) / (1024 * 1024)
    data_gb   = total_data_bytes / (1024 ** 3)
    print(f"  Saved: {out_path} ({header_mb:.1f} MB header + {data_gb:.2f} GB external data)")
    print(f"  Tables: {num_tables} x {table_rows}x{table_cols}  |  MLP: {mlp_dims}  |  batch={BATCH}")


def build_dlrm(out_path: str,
               num_tables: int = 26,
               table_rows: int = 1000,
               table_cols: int = 64,
               dense_dim:  int = 128,
               batch_size: int = 1,
               seed: int | None = 0) -> None:
    """In-memory builder.  Weights are seeded random by default (seed=None
    gives all-zero weights, the v1 behaviour).  v1 built every synthetic
    model with zero weights, which made correctness validation vacuous and
    INT8 quantisation degenerate; timing numbers were unaffected."""
    rng = np.random.default_rng(seed) if seed is not None else None
    NUM_TABLES = num_tables
    TABLE_ROWS = table_rows
    TABLE_COLS = table_cols
    DENSE_DIM  = dense_dim
    BATCH      = batch_size

    nodes        = []
    initializers = []

    # Embedding tables (Gather ops)
    emb_outputs = []
    for i in range(NUM_TABLES):
        table_name = f"emb_table_{i}"
        idx_name   = f"emb_idx_{i}"
        out_name   = f"emb_out_{i}"
        if rng is None:
            weight = np.zeros((TABLE_ROWS, TABLE_COLS), dtype=np.float32)
        else:
            weight = (rng.standard_normal((TABLE_ROWS, TABLE_COLS)) * 0.5).astype(np.float32)
        initializers.append(numpy_helper.from_array(weight, name=table_name))
        nodes.append(helper.make_node(
            "Gather",
            inputs=[table_name, idx_name],
            outputs=[out_name],
            name=f"gather_{i}",
            axis=0,
        ))
        emb_outputs.append(out_name)

    # Concatenate all embeddings + dense input
    concat_out = "concat_out"
    nodes.append(helper.make_node(
        "Concat",
        inputs=emb_outputs + ["dense_input"],
        outputs=[concat_out],
        name="concat_all",
        axis=1,
    ))

    # Top MLP
    mlp_dims = [NUM_TABLES * TABLE_COLS + DENSE_DIM, 1024, 256, 1]
    prev = concat_out
    for j in range(len(mlp_dims) - 1):
        in_dim, out_dim = mlp_dims[j], mlp_dims[j + 1]
        w_name   = f"mlp_w{j}"
        b_name   = f"mlp_b{j}"
        mm_out   = f"mlp_mm{j}"
        add_out  = f"mlp_add{j}"

        if rng is None:
            W = np.zeros((in_dim, out_dim), dtype=np.float32)
            B = np.zeros((out_dim,),        dtype=np.float32)
        else:
            W = (rng.standard_normal((in_dim, out_dim)) / np.sqrt(in_dim)).astype(np.float32)
            B = (rng.standard_normal((out_dim,)) * 0.05).astype(np.float32)
        initializers.append(numpy_helper.from_array(W, name=w_name))
        initializers.append(numpy_helper.from_array(B, name=b_name))

        nodes.append(helper.make_node("MatMul", [prev, w_name], [mm_out],  name=f"mlp_matmul{j}"))
        nodes.append(helper.make_node("Add",    [mm_out, b_name], [add_out], name=f"mlp_add_op{j}"))

        if j < len(mlp_dims) - 2:
            relu_out = f"mlp_relu{j}"
            nodes.append(helper.make_node("Relu", [add_out], [relu_out], name=f"mlp_relu{j}"))
            prev = relu_out
        else:
            nodes.append(helper.make_node("Identity", [add_out], ["output"], name="output_id"))
            prev = "output"

    # Graph inputs / outputs
    # First dim is the batch size; gather indices are 1D of length BATCH so
    # each call processes BATCH samples through the embedding tables.
    graph_inputs = [helper.make_tensor_value_info("dense_input", TensorProto.FLOAT, [BATCH, DENSE_DIM])]
    for i in range(NUM_TABLES):
        graph_inputs.append(
            helper.make_tensor_value_info(f"emb_idx_{i}", TensorProto.INT64, [BATCH])
        )

    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [BATCH, 1])]

    graph = helper.make_graph(nodes, "dlrm_synthetic", graph_inputs, graph_outputs,
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    # ONNX protobuf has a 2 GB single-file hard limit because wire-format
    # lengths are 32-bit varints.  For models larger than ~1.8 GB we must
    # use external data format, where weights live in sibling files and
    # the .onnx just carries metadata + shape info.  We estimate the
    # serialized size from the initializer sizes and auto-switch modes.
    est_weight_bytes = sum(
        int(np.prod(i.dims) if i.dims else 1)
        * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(i.data_type)).itemsize
        for i in initializers
    )
    external_threshold = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB soft cutoff
    use_external = est_weight_bytes > external_threshold

    if use_external:
        print(f"  Weights estimated at {est_weight_bytes / (1024**3):.1f} GB — "
              f"using external data format (splits weights into sibling files)")
        # Convert to external first, THEN check, THEN save.
        # Using all_tensors_to_one_file=True keeps a single sidecar.
        from onnx.external_data_helper import convert_model_to_external_data
        convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=os.path.basename(out_path) + ".data",
            size_threshold=1024,  # inline tiny tensors, external for rest
        )
        onnx.save_model(model, out_path,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=os.path.basename(out_path) + ".data",
                        size_threshold=1024,
                        convert_attribute=False)
    else:
        onnx.checker.check_model(model)
        onnx.save(model, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    data_path = out_path + ".data"
    total_mb = size_mb
    if os.path.exists(data_path):
        total_mb += os.path.getsize(data_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB header"
          f"{f', {total_mb - size_mb:.1f} MB external data, {total_mb:.1f} MB total' if use_external else ''})")
    print(f"  Tables: {NUM_TABLES} x {TABLE_ROWS}x{TABLE_COLS}  |  MLP: {mlp_dims}  |  batch={BATCH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic DLRM ONNX model.")
    parser.add_argument("output", help="Path to write the .onnx file")
    parser.add_argument("--tables", type=int, default=26,
                        help="Number of embedding tables (default: 26)")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Rows per embedding table (default: 1000; "
                             "use 200000 for the scaled E1 model)")
    parser.add_argument("--cols", type=int, default=64,
                        help="Embedding dimension (default: 64)")
    parser.add_argument("--dense-dim", type=int, default=128,
                        help="Dense feature input dimension (default: 128)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Static batch size baked into the model (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random weights")
    parser.add_argument("--zero-weights", action="store_true",
                        help="all-zero weights (v1 behaviour; NOT for correctness runs)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use memory-efficient streaming generator (required "
                             "for multi-GB models — writes weights directly to "
                             "an external .data sidecar without holding them all "
                             "in RAM).  Peak memory stays ~50 MB regardless of "
                             "model size.")
    args = parser.parse_args()
    if args.streaming:
        build_dlrm_streaming(args.output,
                             num_tables=args.tables,
                             table_rows=args.rows,
                             table_cols=args.cols,
                             dense_dim=args.dense_dim,
                             batch_size=args.batch)
    else:
        build_dlrm(args.output,
                   num_tables=args.tables,
                   table_rows=args.rows,
                   table_cols=args.cols,
                   dense_dim=args.dense_dim,
                   batch_size=args.batch,
                   seed=None if args.zero_weights else args.seed)
