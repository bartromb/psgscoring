#!/usr/bin/env python3
"""Post-hoc, buiten de regel (zie preregistratie §Per event): per alleen-druk-apneu van de
`aasm_v3_breath_dual+conf@0.50`-arm (inclusief de vervallen events) de thermistordaling op
BEIDE schalen — vooraf (mediaan 60 s ervoor = bibliotheekmaat, 0,72-dossier) en robuust
(p90 over ±120 s) — en de NSRR-koppeling (apneu / enig event, IoU ≥ 0,20).
Schrijft per_event.jsonl en een samenvatting; de EDF's worden alleen gelezen."""
import json, sys
from pathlib import Path
import numpy as np
D = Path(__file__).parent
sys.path.insert(0, str(D.parent.parent / "scripts")); sys.path.insert(0, str(D.parent.parent))
from validate_mesa import parse_nsrr
from psgscoring.postprocess import flow_envelope, envelope_drop
from psgscoring.utils import detect_channels as pneumo_detect_channels
MESA = Path("/srv/DATA/MESA/mesa/polysomnography"); REF = "aasm15"; ARM = "aasm_v3_breath_dual+conf@0.50"

def iou(a0, a1, b0, b1):
    i = max(0.0, min(a1, b1) - max(a0, b0)); u = max(a1, b1) - min(a0, b0); return i / u if u > 0 else 0.0

def robust_drop(env, sf, t0, t1, win=120.0, pct=90):
    i0, i1 = int(t0 * sf), int(t1 * sf); b0 = int(max(0.0, t0 - win) * sf); b1 = int(min(len(env) / sf, t1 + win) * sf)
    if i1 <= i0 or i1 > len(env): return None
    seg = np.concatenate([env[b0:i0], env[i1:b1]])
    if seg.size < int(20 * sf): return None
    bl = float(np.percentile(seg, pct)); ev = float(np.median(env[i0:i1]))
    return None if bl <= 0 else float(np.clip(1.0 - ev / bl, -1.0, 1.0))

def een(r):
    import mne
    rec = r["recording"]; arm = r["profiles"].get(ARM) or {}
    if "apneas" not in arm: return None
    dropped = ((arm.get("dual_sensor_apnea") or {}).get("confirmation") or {}).get("dropped") or []
    events = [dict(e, _status="behouden") for e in arm["apneas"] if e.get("corroboration") == "pressure_only"] + \
             [dict(e, _status="vervallen", dual_confirmation="vervallen") for e in dropped if e.get("corroboration") == "pressure_only"]
    if not events: return {"recording": rec, "events": []}
    raw = mne.io.read_raw_edf(str(MESA / "edfs" / f"{rec}.edf"), preload=False, verbose="ERROR")
    ch = pneumo_detect_channels(raw.ch_names); th = ch.get("flow_thermistor")
    if not th or th not in raw.ch_names: return {"recording": rec, "events": [], "error": f"geen thermistor in {raw.ch_names}"}
    x = raw.get_data(picks=[th])[0]; sf = float(raw.info["sfreq"])
    env = flow_envelope(x, sf)
    _h, refs, _t = parse_nsrr(MESA / "annotations-events-nsrr" / f"{rec}-nsrr.xml", r["duration_h"] * 3600.0)
    ref_all = [(a, b) for a, b, t in refs.get(REF, [])]
    ref_ap = [(a, b) for a, b, t in refs.get(REF, []) if "hypopnea" not in str(t).lower()]
    uit = []
    for e in events:
        a0 = float(e["onset_s"]); a1 = a0 + float(e["duration_s"])
        uit.append({"rec": rec, "onset_s": a0, "duration_s": round(a1 - a0, 2), "status": e["_status"], "conf": e.get("dual_confirmation"),
                    "d_lib": e.get("thermistor_drop"), "d_vooraf": envelope_drop(env, sf, a0, a1), "r_robuust": robust_drop(env, sf, a0, a1),
                    "desat": e.get("desaturation_pct"),
                    "nsrr_apneu": any(iou(a0, a1, b0, b1) >= 0.2 for b0, b1 in ref_ap), "nsrr_enig": any(iou(a0, a1, b0, b1) >= 0.2 for b0, b1 in ref_all)})
    return {"recording": rec, "events": uit, "thermistor": th}

if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else D / "mesa.json"
    d = json.load(open(src)); rows = d["results"] if isinstance(d, dict) and "results" in d else d
    rows = [r for r in rows if "error" not in r]
    from concurrent.futures import ProcessPoolExecutor
    out = D / "posthoc_per_event.jsonl"; fh = out.open("w"); n = 0
    with ProcessPoolExecutor(max_workers=int(sys.argv[2]) if len(sys.argv) > 2 else 8) as ex:
        for res in ex.map(een, rows):
            if res is None: continue
            for e in res["events"]: fh.write(json.dumps(e) + "\n"); n += 1
            if res.get("error"): print(res["recording"], res["error"][:120])
    fh.close(); print("events", n, "→", out)
    # samenvatting: klasse per schaal × status → NSRR-match
    ev = [json.loads(l) for l in out.read_text().splitlines()]
    def kl(v): return "?" if v is None else "A" if v >= 0.72 else "C" if v >= 0.30 else "B"
    for schaal in ("d_vooraf", "r_robuust"):
        print(f"\n== schaal {schaal}: klasse × bevestiging → n, NSRR-apneu, enig NSRR-event ==")
        tab = {}
        for e in ev:
            k = (kl(e[schaal]), e["conf"] or "n/a"); t = tab.setdefault(k, [0, 0, 0]); t[0] += 1; t[1] += e["nsrr_apneu"]; t[2] += e["nsrr_enig"]
        for k, (a, b, c) in sorted(tab.items()): print(f"  {k[0]:2} {k[1]:11} n={a:6} apneu {b/a:.2f} enig {c/a:.2f}")
    # drempelveeg op de robuuste schaal: precisie/recall van 'd ≥ τ' voor NSRR-apneu onder alleen-druk-apneus zonder gevolg
    print("\n== veeg τ op r_robuust, events zonder desat-bevestiging (conf ∉ desat/arousal): precisie / recall tegen NSRR-apneu ==")
    zg = [e for e in ev if e["conf"] in ("thermistor", "vervallen") and e["r_robuust"] is not None]
    pos = sum(e["nsrr_apneu"] for e in zg)
    for tau in (0.5, 0.6, 0.72, 0.8, 0.85, 0.9):
        sel = [e for e in zg if e["r_robuust"] >= tau]; tp = sum(e["nsrr_apneu"] for e in sel)
        print(f"  τ {tau:.2f}: n={len(sel):5} precisie {tp/len(sel) if sel else 0:.2f} recall {tp/pos if pos else 0:.2f}")
