#!/usr/bin/env python
"""export_onnx.py — exporteer een getraind schaal-arm-gewicht (``<arm>/model_best.pt``) naar ONNX
als KANDIDAAT voor de bibliotheek, met de conventies van ``psgscoring/data/arousal_unet_v1.json``:
opset 17, invoer ``x[1,C,n]`` (50 Hz, genormaliseerd zoals in de bench, n deelbaar door 128),
uitvoer ``p[1,n]`` = kans per sample (sigmoid zit in de graaf; de bibliotheek drempelt de kans).

Schrijft ``<arm>/arousal_unet_<arm>_kandidaat.onnx`` + ``.json`` (sha256 van bron en ONNX, config,
τ uit ``tau.json``, maximale absolute afwijking tegen torch op CPU in float32, op twee lengtes).
Raakt de bibliotheek niet; inbouw als ``unet_v2`` is een aparte beslissing.

Gebruik:  bench/eeg/_venv/bin/python bench/eeg/unet_scale/export_onnx.py --arm MS
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER.parent / "unet50"))
from model import UNet1D  # noqa: E402

OUT_DEFAULT = Path("/srv/CODE/docs/arousal_unet_schaal_20261025")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


class MetSigmoid(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    d = a.out / a.arm
    bron = d / "model_best.pt"
    ck = torch.load(bron, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = UNet1D(**cfg)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapped = MetSigmoid(net).eval()
    c = int(cfg["in_ch"])
    n0 = 128 * 300  # 12,8 min @ 50 Hz
    x0 = torch.randn(1, c, n0)
    onnx_pad = d / f"arousal_unet_{a.arm}_kandidaat.onnx"
    torch.onnx.export(
        wrapped, (x0,), str(onnx_pad), opset_version=a.opset, input_names=["x"], output_names=["p"],
        dynamic_axes={"x": {2: "n"}, "p": {1: "n"}}, dynamo=False, do_constant_folding=True)
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_pad)))
    sess = ort.InferenceSession(str(onnx_pad), providers=["CPUExecutionProvider"])
    afw = {}
    for n in (n0, 128 * 3000):  # 12,8 min en 2,1 u
        g = torch.Generator().manual_seed(n)
        x = torch.randn(1, c, n, generator=g)
        with torch.no_grad():
            pt = wrapped(x).numpy()
        po = sess.run(["p"], {"x": x.numpy()})[0]
        assert po.shape == pt.shape, (po.shape, pt.shape)
        afw[str(n)] = float(np.max(np.abs(po - pt)))
    tau = json.loads((d / "tau.json").read_text())
    meta = {
        "arm": a.arm,
        "source": str(bron.relative_to(a.out.parent)) if str(bron).startswith(str(a.out.parent)) else str(bron),
        "source_sha256": sha256_of(bron),
        "onnx_sha256": sha256_of(onnx_pad),
        "config": cfg,
        "epoch": ck.get("epoch"),
        "val_f1": tau.get("val_f1"),
        "tau_val": tau.get("tau_val"),
        "tau_neutral": tau.get("tau_neutral"),
        "opset": a.opset,
        "input": f"x[1,{c},n] = (EEG, EOG, kin-EMG{', flow' if c == 4 else ''}) @ 50 Hz, genormaliseerd, n deelbaar door 128",
        "output": "p[1,n] kans per sample",
        "max_abs_diff_vs_torch": afw,
        "fs": 50,
        "pad_multiple": 128,
        "preprocessing": "als arousal_unet_v1.json (bench data_scale.py: mne resample 50 Hz, lopend gemiddelde/RMS 18 min verwijderd, clip ±20, float16-rondgang)",
        "postprocessing": "1 s lopend gemiddelde, drempel tau, gaten ≤ 1 s samengevoegd, ≥ 3 s, slaappoort, daarna de 10 s-samenvoeging van de keten",
        "status": "KANDIDAAT — niet in de bibliotheek; schaalmeting docs/arousal_unet_schaal_replicatie_20261026.md",
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
    }
    (d / f"arousal_unet_{a.arm}_kandidaat.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: meta[k] for k in ("arm", "source_sha256", "onnx_sha256", "epoch", "val_f1", "tau_val", "max_abs_diff_vs_torch")}, indent=1))
    print("bytes", onnx_pad.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
