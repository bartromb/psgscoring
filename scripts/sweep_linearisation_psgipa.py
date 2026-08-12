#!/usr/bin/env python3
"""Blok 1: linearisatie aan, en `hypopnea_strictness` opnieuw afleiden.

Eén experiment, geen twee. Linearisatie aanzetten verkleint de gemeten
flowreductie, dus bij ongewijzigde strictness vallen er events af en schiet de
bias negatief door. Wie alleen het eerste doet, reproduceert de conclusie van
v0.14.0 en trekt hem opnieuw ten onrechte.

BESLISREGEL — vooraf vastgelegd in de CHANGELOG bij v0.16.0, niet hier:
`hypopnea_strictness` wordt opnieuw afgeleid als het punt waar de AHI-bias
tegen de PSG-IPA-scoordermediaan door nul gaat. NIET het F1-maximum, want dat
is fitten op de uitkomstmaat. Bij twee punten die nul omsluiten: de kleinste
|bias|; bij gelijkspel de LAGERE strictness, want de gemeten faalmodus op MESA
is onderdetectie.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \
        python scripts/sweep_linearisation_psgipa.py --out sweep.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path("/home/bart/PSG-IPA")
RECS = [f"SN{i}" for i in range(1, 6)]


def _harness():
    spec = importlib.util.spec_from_file_location("vp", REPO / "validate_psgipa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["vp"] = m
    spec.loader.exec_module(m)
    return m


def _register(strictness, linearise, profiel="aasm_v3_breath"):
    """Zet aasm_v3_breath op deze combinatie en ververs de legacy-dicts.

    De legacy-dicts worden bij import uit profiles.py afgeleid, dus een
    wijziging aan het dataclass-profiel bereikt de detector pas na een reload
    van constants én het doorgeven aan de modules die de dict vasthouden.
    Zonder dat meet je stilzwijgend het oude punt.
    """
    from psgscoring.profiles import PROFILES as P
    pp = P[profiel].post_processing
    pp.hypopnea_strictness = float(strictness)
    pp.hypopnea_force_linearisation = bool(linearise)
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    # Controle: bereikt het veld de detector werkelijk?
    d = C.SCORING_PROFILES[profiel]
    assert abs(d["HYPOPNEA_STRICTNESS"] - float(strictness)) < 1e-9
    assert d["HYPOPNEA_FORCE_LINEARISATION"] is bool(linearise)


def een(args):
    sn, strictness, linearise, profiel = args
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mne
    import psgscoring
    mne.set_log_level("ERROR")
    vp = _harness()
    _register(strictness, linearise, profiel)

    raw = mne.io.read_raw_edf(
        str(DATA / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"),
        preload=True, verbose=False)
    dur = float(raw.times[-1])
    refs, hyp1 = [], None
    for i, f in enumerate(vp.find_scorer_files(DATA, sn)):
        a, _t, h = vp.parse_scorer_file(f, dur)
        if a is not None:
            refs.append(a)
            if i == 0:
                hyp1 = h
    r = psgscoring.run_pneumo_analysis(raw, hypno=hyp1,
                                       scoring_profile=profiel)
    resp = r.get("respiratory") or {}
    s = resp.get("summary") or {}
    ev = resp.get("events") or []
    return {
        "recording": sn, "profiel": profiel, "strictness": float(strictness),
        "linearised": bool(linearise),
        "ahi": float(s.get("ahi_total") or 0.0),
        "ref_median": float(median(refs)),
        "ref_lo": float(min(refs)), "ref_hi": float(max(refs)),
        "n_hypopnea": sum(1 for e in ev if "hypopnea" in str(e.get("type", ""))),
        "n_apnea": sum(1 for e in ev if e.get("type") in
                       ("obstructive", "central", "mixed", "uncertain")),
        # provenance: draaide de nieuwe tak echt?
        "prov_lin": resp.get("hypopnea_linearised"),
        "prov_shared": resp.get("hypopnea_channel_shared"),
        "prov_forced": resp.get("hypopnea_linearisation_forced"),
    }


def _sev(a):
    return "normal" if a < 5 else "mild" if a < 15 else "moderate" if a < 30 else "severe"


def samenvat(rows):
    """Per (strictness, linearised): bias, MAE, binnen range, severity."""
    uit = {}
    for r in rows:
        uit.setdefault((r["strictness"], r["linearised"]), []).append(r)
    regels = []
    for (st, lin), rs in sorted(uit.items()):
        d = [x["ahi"] - x["ref_median"] for x in rs]
        regels.append({
            "strictness": st, "linearised": lin, "n": len(rs),
            "bias": sum(d) / len(d),
            "mae": sum(abs(x) for x in d) / len(d),
            "in_range": sum(1 for x in rs
                            if x["ref_lo"] <= x["ahi"] <= x["ref_hi"]),
            "severity": sum(1 for x in rs
                            if _sev(x["ahi"]) == _sev(x["ref_median"])),
            "n_hyp": sum(x["n_hypopnea"] for x in rs),
            "n_apn": sum(x["n_apnea"] for x in rs),
        })
    return regels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--strictness", nargs="+", type=float, default=None)
    ap.add_argument("--linearised", nargs="+", type=int, default=[1])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sts = a.strictness or [0.30 + 0.025 * i for i in range(13)]  # 0.30..0.60

    jobs = [(sn, st, bool(l), a.profile) for l in a.linearised for st in sts for sn in RECS]
    print(f"  {len(jobs)} runs — {len(sts)} strictness-waarden x "
          f"{len(a.linearised)} lin-standen x 5 opnames", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(een, j) for j in jobs]):
            rows.append(f.result())
            if len(rows) % 5 == 0:
                print(f"    {len(rows)}/{len(jobs)}", flush=True)

    regels = samenvat(rows)
    print(f"\n  {'strict':>7s} {'lin':>4s} {'bias':>7s} {'MAE':>6s} "
          f"{'range':>6s} {'sev':>5s} {'hyp':>5s} {'apn':>5s}")
    for g in regels:
        print(f"  {g['strictness']:7.3f} {str(g['linearised']):>4s} "
              f"{g['bias']:+7.2f} {g['mae']:6.2f} {g['in_range']:4d}/5 "
              f"{g['severity']:3d}/5 {g['n_hyp']:5d} {g['n_apn']:5d}")

    # Nulbias-kruising, volgens de vooraf vastgelegde regel.
    lin = [g for g in regels if g["linearised"]]
    lin.sort(key=lambda g: g["strictness"])
    for x, y in zip(lin, lin[1:]):
        if x["bias"] * y["bias"] <= 0:
            keuze = min((x, y), key=lambda g: (abs(g["bias"]), g["strictness"]))
            print(f"\n  nulbias-kruising tussen {x['strictness']:.3f} "
                  f"({x['bias']:+.2f}) en {y['strictness']:.3f} "
                  f"({y['bias']:+.2f})")
            print(f"  -> gekozen punt {keuze['strictness']:.3f} "
                  f"(bias {keuze['bias']:+.2f}, MAE {keuze['mae']:.2f})")
            break
    else:
        print("\n  GEEN nulbias-kruising binnen het bereik — rapporteer dat.")

    if a.out:
        a.out.write_text(json.dumps({"rows": rows, "summary": regels}, indent=2))
        print(f"\n  geschreven: {a.out}")


if __name__ == "__main__":
    main()
