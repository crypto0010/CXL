#!/usr/bin/env python3
"""Export MobileBERT to ONNX format.

Usage: gen_mobilebert.py <output_path>

Requires: torch, transformers
"""

import sys
import os


def export_mobilebert(out_path: str) -> None:
    import torch
    from transformers import MobileBertModel, MobileBertConfig

    cfg   = MobileBertConfig()
    model = MobileBertModel(cfg).eval()
    dummy_input_ids      = torch.zeros(1, 128, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

    torch.onnx.export(
        model,
        ({"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask},),
        out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=17,
        dynamic_axes={
            "input_ids":      {1: "seq"},
            "attention_mask": {1: "seq"},
        },
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)
    export_mobilebert(sys.argv[1])
