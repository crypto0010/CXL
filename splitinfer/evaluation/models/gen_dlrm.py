#!/usr/bin/env python3
"""Generate a synthetic DLRM-style ONNX model.

Usage: gen_dlrm.py <output_path>

Architecture:
  26 embedding tables x 1000 rows x 64-dim (Gather)
  Dense input: 128 features
  Top MLP: (26*64 + 128) -> 1024 -> 256 -> 1
"""

import sys
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def build_dlrm(out_path: str) -> None:
    NUM_TABLES = 26
    TABLE_ROWS = 1000
    TABLE_COLS = 64
    DENSE_DIM  = 128

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
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)
    build_dlrm(sys.argv[1])
