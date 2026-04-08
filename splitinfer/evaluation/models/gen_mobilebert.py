#!/usr/bin/env python3
"""Export MobileBERT to ONNX format.

Usage: gen_mobilebert.py <output_path>

Requires: torch, transformers
"""

import sys
import os


def export_mobilebert(out_path: str) -> None:
    """Export MobileBERT to ONNX with fixed input shapes.

    KNOWN BROKEN with transformers >= 5.5:
      In transformers 5.x the new masking_utils.create_bidirectional_mask()
      path crashes during ONNX trace with:

        File ".../transformers/masking_utils.py", line 492, in sdpa_mask
          q_length, q_offset = q_length.shape[0], q_length[0].to(device)
        IndexError: tuple index out of range

      The bug is internal to transformers and not patchable from this
      script.  The previous transformers 4.x API worked fine.

    WORKAROUNDS (any one of these):
      1. Pin transformers to 4.x in your environment:
           pip install 'transformers<5'
         then re-run this script.
      2. Use the optimum-onnxruntime exporter instead:
           pip install optimum[exporters]
           optimum-cli export onnx --model google/mobilebert-uncased \\
             --task feature-extraction <out_dir>
      3. Switch this script to use a non-MobileBERT BERT-family model
         (DistilBERT, TinyBERT) that uses a different masking path.

    For E1 (capability unlock — DLRM only) and current E2 vision
    coverage (YOLOv8), MobileBERT is not blocking.  Restoring it is
    a follow-up that needs an environment-level decision.
    """
    import torch
    from transformers import MobileBertModel, MobileBertConfig

    cfg = MobileBertConfig()
    model = MobileBertModel(cfg)
    model.train(False)            # inference mode without using .eval()
    model.requires_grad_(False)

    # Wrap to return only last_hidden_state (HF returns a dict which
    # the ONNX exporter handles awkwardly).
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state

    wrapper = Wrapper(model)
    wrapper.train(False)

    seq_len = 128
    dummy_input_ids      = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            out_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB, seq_len={seq_len})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)
    export_mobilebert(sys.argv[1])
