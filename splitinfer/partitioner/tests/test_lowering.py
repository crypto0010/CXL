import json
import os
import subprocess
import sys

import numpy as np
import onnx
import pytest

from partitioner.lowering import Lowering, _requant_params

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "evaluation", "models")


@pytest.fixture(scope="module")
def dlrm(tmp_path_factory):
    d = tmp_path_factory.mktemp("dlrm")
    model = os.path.join(d, "dlrm.onnx")
    subprocess.run([sys.executable, os.path.join(MODELS, "gen_dlrm.py"), model], check=True,
                   capture_output=True)
    manifest = os.path.join(d, "m.json")
    subprocess.run([sys.executable, "-m", "partitioner.cli", model, "-o", manifest, "--resident",
                    "--host-budget-mb", "0"], check=True, capture_output=True, cwd=ROOT)
    return model, manifest, str(d)


def test_requant_params_precision():
    for r in (0.5, 0.01, 3.7, 1e-4):
        m, s = _requant_params(r)
        assert 0 <= m < 32768 and 0 <= s <= 31
        assert abs(m / (1 << s) - r) / r < 1e-3


def test_dlrm_lowers_and_reference_tracks_fp32(dlrm):
    model, manifest, out = dlrm
    lw = Lowering(onnx.load(model), json.load(open(manifest)), calibration_inputs=8)
    prog = lw.lower()
    kinds = [l["kind"] for l in prog["layers"]]
    assert kinds.count("gather") == 26 and kinds.count("concat") == 1 and kinds.count("fc") == 3
    # alignment rules
    for l in prog["layers"]:
        for k, v in l.items():
            if k.endswith("_addr"):
                assert v % 16 == 0, f"{l['name']}.{k} unaligned"
        if l["kind"] == "fc":
            assert l["K_pad"] % 16 == 0 and l["M_pad"] % 16 == 0
    prog = lw.write(out, n_vectors=4)
    assert prog["vectors"]["max_norm_err_vs_fp32"] < 0.15, prog["vectors"]["max_norm_err_vs_fp32"]
    assert os.path.getsize(os.path.join(out, "image.bin")) == prog["ddr2"]["image_bytes"]


def test_fc_chain_is_fused(dlrm):
    model, manifest, out = dlrm
    lw = Lowering(onnx.load(model), json.load(open(manifest)), calibration_inputs=4)
    prog = lw.lower()
    fcs = [l for l in prog["layers"] if l["kind"] == "fc"]
    assert fcs[0]["relu"] == 1 and fcs[0]["has_bias"] and not fcs[0]["final"]
    assert fcs[-1]["final"] and fcs[-1]["mult"] == 0


def test_reference_is_deterministic_integer(dlrm):
    model, manifest, out = dlrm
    lw = Lowering(onnx.load(model), json.load(open(manifest)), calibration_inputs=4)
    lw.lower()
    f = lw._feeds[0]
    a, _ = lw.run_reference(f)
    b, _ = lw.run_reference(f)
    assert a.dtype == np.int32 and np.array_equal(a, b)
