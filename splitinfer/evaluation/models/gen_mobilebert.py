#!/usr/bin/env python3
"""Export MobileBERT to ONNX format.

Usage: gen_mobilebert.py <output_path>

Requires: torch, transformers
"""

import sys
import os


def export_mobilebert(out_path: str) -> None:
    """Export MobileBERT to ONNX with fixed input shapes.

    KNOWN BROKEN with transformers >= 5.5 (and the system numpy is 2.x):
      In transformers 5.x the new masking_utils.create_bidirectional_mask()
      path crashes during ONNX trace with:

        File ".../transformers/masking_utils.py", line 492, in sdpa_mask
          q_length, q_offset = q_length.shape[0], q_length[0].to(device)
        IndexError: tuple index out of range

      The bug is internal to transformers and not patchable from this
      script.  The previous transformers 4.x API works fine.

    WORKING APPROACH (used to produce the committed mobilebert.onnx):
      Use a venv with --system-site-packages so it inherits the existing
      torch + onnx, then install transformers<5 into the venv:

        python3 -m venv --system-site-packages /tmp/tf4_venv
        /tmp/tf4_venv/bin/pip install 'transformers<5'
        /tmp/tf4_venv/bin/python evaluation/models/gen_mobilebert.py \\
            evaluation/models/generated/mobilebert.onnx

      The script `make_tf4_venv.sh` next to this file automates the setup.
      The NumPy 1.x/2.x ABI warnings during import are benign — torch
      tolerates the mismatch and the export still works.
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
