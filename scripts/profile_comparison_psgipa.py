#!/usr/bin/env python3
"""Profielvergelijking op PSG-IPA — AHI per profiel tegen de scoordermediaan.

Draait een willekeurige set profielen op de vijf PSG-IPA-opnames met het
manuele hypnogram van scorer 1, zodat teller én noemer bij alle profielen
identiek zijn en alleen het respiratoire scoren nog verschilt. Zelfde conventie
als `validate_psgipa.py` en paper v31.

Neem altijd minstens één BEKEND profiel mee als anker. PSG-IPA-AHI's zijn
harnasgevoelig: cijfers uit twee runs naast elkaar leggen verandert twee dingen
tegelijk. Met een anker in dezelfde run is elke vergelijking intern.

    cd /home/bart/CODE/psgscoring
    PYTHONPATH=$PWD python scripts/profile_comparison_psgipa.py \
        --data-dir /home/bart/PSG-IPA \
        --profiles aasm_v3_rec aasm_v3_breath aasm_v3_prob \
        --out /tmp/vergelijking.json

VALKUIL — waarom PYTHONPATH er staat. In `~/.local/lib/python3.12/site-packages`
staat een psgscoring **0.2.91**, van vóór het profielregister. Vanuit de repo
pakt `import psgscoring` de bron; vanuit elke andere map die stokoude versie, en
dan meet je stilzwijgend niets. Herkenbaar aan een logregel die niet in de bron
staat: "Signal quality assessment failed: The truth value of an array…".
`--expect-version` faalt daarom hard, en de controle draait ook in elke worker —
ProcessPoolExecutor start verse interpreters die de werkmap niet erven.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RECORDINGS = [f"SN{i}" for i in range(1, 6)]


def _require(profiles, expect_version=None):
    """Faal luid in plaats van stil de verkeerde versie meten."""
    import psgscoring
    from psgscoring.profiles import PROFILES
    if expect_version and psgscoring.__version__ != expect_version:
        raise SystemExit(
            f"VERKEERDE psgscoring: {psgscoring.__version__} uit "
            f"{psgscoring.__file__} — verwacht {expect_version} uit {REPO}")
    missing = [p for p in profiles if p not in PROFILES]
    if missing:
        raise SystemExit(f"profielen bestaan niet in deze versie: {missing}")
    return psgscoring.__version__


def _harness():
    spec = importlib.util.spec_from_file_location("vp", REPO / "validate_psgipa.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vp"] = mod
    spec.loader.exec_module(mod)
    return mod


def analyse(sn_id, data_dir, profiles, expect_version):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))          # ook in de worker
    import mne
    import psgscoring
    _require(profiles, expect_version)
    mne.set_log_level("ERROR")
    vp = _harness()
    data_dir = Path(data_dir)

    psg = data_dir / "Resp_events" / "PSG" / f"{sn_id}_Respiration.edf"
    if not psg.exists():
        return {"recording": sn_id, "error": f"PSG ontbreekt: {psg}"}
    raw = mne.io.read_raw_edf(str(psg), preload=True, verbose=False)
    dur = float(raw.times[-1])

    refs, hypno1 = [], None
    for i, f in enumerate(vp.find_scorer_files(data_dir, sn_id)):
        ahi, _tst, hyp = vp.parse_scorer_file(f, dur)
        if ahi is not None:
            refs.append(ahi)
            if i == 0:
                hypno1 = hyp
    if not refs or hypno1 is None:
        return {"recording": sn_id, "error": "geen bruikbare scoordersdata"}

    out = {
        "recording": sn_id,
        "psgscoring": psgscoring.__version__,
        "n_scorers": len(refs),
        "ref_median": round(float(median(refs)), 2),
        "ref_lo": round(float(min(refs)), 2),
        "ref_hi": round(float(max(refs)), 2),
        "tst_min": round(sum(1 for s in hypno1
                             if s in ("N1", "N2", "N3", "R")) * 0.5, 1),
        "profiles": {},
    }
    for prof in profiles:
        try:
            r = psgscoring.run_pneumo_analysis(raw, hypno=hypno1,
                                               scoring_profile=prof)
            resp = r.get("respiratory") or {}
            s, ev = resp.get("summary") or {}, resp.get("events") or []
            ahi = s.get("ahi_total")
            out["profiles"][prof] = {
                "ahi": None if ahi is None else round(float(ahi), 2),
                "n_events": len(ev),
                "n_apnea": sum(1 for e in ev
                               if e.get("type") in ("obstructive", "central",
                                                    "mixed", "uncertain")),
                "n_hypopnea": sum(1 for e in ev
                                  if "hypopnea" in str(e.get("type", ""))),
            }
        except Exception as e:                                    # noqa: BLE001
            out["profiles"][prof] = {"error": f"{type(e).__name__}: {e}"[:160]}
    return out


def _severity(a):
    return "normal" if a < 5 else "mild" if a < 15 else "moderate" if a < 30 else "severe"


def report(rows, profiles):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("geen bruikbare resultaten"); return
    short = {p: p.replace("aasm_v3_", "") for p in profiles}
    hdr = (f"{'':5s} {'human':>6s} {'range':>13s} {'TST':>6s} "
           + " ".join(f"{short[p]:>9s}" for p in profiles))
    print(f"\npsgscoring {ok[0]['psgscoring']} · n = {len(ok)} · hypnogram scorer 1\n")
    print(hdr); print("-" * len(hdr))
    for r in ok:
        line = (f"{r['recording']:5s} {r['ref_median']:6.2f} "
                f"{r['ref_lo']:5.2f}-{r['ref_hi']:6.2f} {r['tst_min']:6.0f} ")
        line += " ".join(
            f"{r['profiles'][p].get('ahi', float('nan')):9.2f}" for p in profiles)
        print(line)
    print(f"\n{'profile':24s} {'bias':>7s} {'|bias|':>7s} {'in range':>9s} {'severity':>9s}")
    for p in profiles:
        vals = [(r["profiles"][p].get("ahi"), r) for r in ok]
        if any(v is None for v, _ in vals):
            print(f"{p:24s}   (onvolledig)"); continue
        b = [v - r["ref_median"] for v, r in vals]
        inr = sum(1 for v, r in vals if r["ref_lo"] <= v <= r["ref_hi"])
        sev = sum(1 for v, r in vals if _severity(v) == _severity(r["ref_median"]))
        print(f"{p:24s} {sum(b)/len(b):+7.2f} {sum(abs(x) for x in b)/len(b):7.2f} "
              f"{inr:6d}/{len(ok)} {sev:6d}/{len(ok)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/bart/PSG-IPA")
    ap.add_argument("--profiles", nargs="+", required=True,
                    help="minstens één bekend profiel als anker")
    ap.add_argument("--recordings", nargs="+", default=RECORDINGS)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--expect-version", default=None,
                    help="faal als psgscoring een andere versie is (aanrader)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    v = _require(a.profiles, a.expect_version)
    print(f"psgscoring {v} uit {REPO} — {len(a.profiles)} profielen aanwezig",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(analyse, sn, a.data_dir, a.profiles, a.expect_version)
                for sn in a.recordings]
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(f"  klaar: {r['recording']}"
                  + (f"  ({r['error']})" if "error" in r else ""), flush=True)
    rows.sort(key=lambda d: d["recording"])
    report(rows, a.profiles)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"\ngeschreven: {a.out}")


if __name__ == "__main__":
    main()
