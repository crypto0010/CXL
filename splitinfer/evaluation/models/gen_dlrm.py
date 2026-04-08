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


def build_dlrm(out_path: str,
               num_tables: int = 26,
               table_rows: int = 1000,
               table_cols: int = 64,
               dense_dim:  int = 128) -> None:
    NUM_TABLES = num_tables
    TABLE_ROWS = table_rows
    TABLE_COLS = table_cols
    DENSE_DIM  = dense_dim

    nodes        = []
    initializers = []

    # Embedding tables (Gather ops)
    emb_outputs = []
    for i in range(NUM_TABLES):
        table_name = f"emb_table_{i}"
        idx_name   = f"emb_idx_{i}"
        out_name   = f"emb_out_{i}"
        weight = np.zeros((TABLE_ROWS, TABLE_COLS), dtype=np.float32)
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

        W = np.zeros((in_dim, out_dim), dtype=np.float32)
        B = np.zeros((out_dim,),        dtype=np.float32)
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
    graph_inputs = [helper.make_tensor_value_info("dense_input", TensorProto.FLOAT, [1, DENSE_DIM])]
    for i in range(NUM_TABLES):
        graph_inputs.append(
            helper.make_tensor_value_info(f"emb_idx_{i}", TensorProto.INT64, [1])
        )

    graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])]

    graph = helper.make_graph(nodes, "dlrm_synthetic", graph_inputs, graph_outputs,
                              initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"  Tables: {NUM_TABLES} x {TABLE_ROWS}x{TABLE_COLS}  |  MLP: {mlp_dims}")


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
    args = parser.parse_args()
    build_dlrm(args.output,
               num_tables=args.tables,
               table_rows=args.rows,
               table_cols=args.cols,
               dense_dim=args.dense_dim)
