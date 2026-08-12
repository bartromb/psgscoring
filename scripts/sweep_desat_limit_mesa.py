#!/usr/bin/env python3
"""Blok 2B op MESA: hoe vaak bevestigt één desaturatie meerdere hypopneus?

PSG-IPA gaf een echte nul (grootste groep = 2, op één opname; CAISR's limiet
van 2 degradeert daar niets). MESA is het cohort waar de overdetectie is
gemeten, dus daar hoort de vraag opnieuw gesteld.

Rapporteert per opname de groepsverdeling ZONDER te begrenzen (limiet hoog),
zodat de omvang van het probleem zichtbaar is voordat er een waarde gekozen
wordt, plus het aantal dat een limiet van 2 respectievelijk 3 zou degraderen.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \
        python scripts/sweep_desat_limit_mesa.py --n 30 --out mesa_desat.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EDF_DIR = Path("/home/bart/MESA/mesa/polysomnography/edfs")


def _hypno_uit_nsrr(xml_path, dur_s):
    """Hergebruikt de staging-parser van het subtyperingsscript."""
    spec = importlib.util.spec_from_file_location(
        "sub", REPO / "scripts" / "subtype_agreement_mesa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sub"] = m
    spec.loader.exec_module(m)
    return m._ref_apneas(xml_path, dur_s)


def _register(limiet, profiel):
    from psgscoring.profiles import PROFILES as P
    P[profiel].post_processing.max_events_per_desaturation = limiet
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    assert C.SCORING_PROFILES[profiel]["MAX_EVENTS_PER_DESATURATION"] == limiet


def een(args):
    fn, profiel, verwacht = args
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mne
    import psgscoring
    mne.set_log_level("ERROR")
    if verwacht and psgscoring.__version__ != verwacht:
        raise SystemExit(f"verkeerde psgscoring: {psgscoring.__version__}")
    try:
        # Limiet 999 = niets wordt gedegradeerd, maar de groepsstatistiek
        # wordt wel berekend. Zo meet je de omvang voor je een waarde kiest.
        _register(999, profiel)
        raw = mne.io.read_raw_edf(str(fn), preload=True, verbose=False)
        dur = float(raw.times[-1])
        # Zonder hypnogram klapt de gegradeerde detector en valt de pipeline
        # stil terug op de envelope-detector ("breath-graded detector failed").
        # Dan meet je een ander profiel dan je denkt; de NSRR-staging is
        # beschikbaar en hoort gebruikt.
        xml = (fn.parent.parent / "annotations-events-nsrr"
               / f"{fn.stem}-nsrr.xml")
        if not xml.exists():
            return {"recording": fn.stem, "error": "geen NSRR-annotatie"}
        _ref, hypno = _hypno_uit_nsrr(xml, dur)
        if not any(s in ("N1", "N2", "N3", "R") for s in hypno):
            return {"recording": fn.stem, "error": "geen slaap in de staging"}
        r = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                           scoring_profile=profiel)
        resp = r.get("respiratory") or {}
        st = resp.get("desat_reuse_limit") or {}
        ev = resp.get("events") or []
        det = {str((e.get("classify_detail") or {}).get("detector"))
               for e in ev if e.get("classify_detail")}
        return {
            "recording": fn.stem,
            "ahi": float((resp.get("summary") or {}).get("ahi_total") or 0.0),
            "n_hypopnea": sum(1 for e in ev
                              if "hypopnea" in str(e.get("type", ""))),
            # provenance: draaide werkelijk de gegradeerde detector?
            "detectors": sorted(det),
            "stats": st,
        }
    except Exception as e:
        return {"recording": fn.stem, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--expect-version", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    alle = sorted(EDF_DIR.glob("mesa-sleep-*.edf"))
    random.Random(a.seed).shuffle(alle)
    keuze = alle[:a.n]
    print(f"  {len(keuze)} van {len(alle)} opnames, profiel {a.profile}",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(een, (fn, a.profile, a.expect_version))
                               for fn in keuze]):
            rows.append(f.result())
            if len(rows) % 5 == 0:
                print(f"    {len(rows)}/{len(keuze)}", flush=True)

    # Altijd schrijven, ook bij nul bruikbare opnames: juist dan zitten de
    # foutmeldingen erin en zonder bestand is de oorzaak weg.
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"\n  geschreven: {a.out}")

    ok = [r for r in rows if "error" not in r and r.get("stats")]
    print(f"\n  bruikbaar: {len(ok)} van {len(rows)}")
    fouten = Counter(r.get("error", "").split(":")[0]
                     for r in rows if "error" in r)
    for f_, n in fouten.most_common():
        print(f"    fout: {f_} — {n}x")
    detectors = Counter(d for r in ok for d in r.get("detectors") or [])
    if detectors:
        print(f"    detectoren: {dict(detectors)}")
    if not ok:
        print("  GEEN bruikbare opnames — rapporteer dat, verzin geen getal.")
        return

    tot_hyp = sum(r["n_hypopnea"] for r in ok)
    tot_grp = sum(r["stats"].get("n_events_grouped", 0) for r in ok)
    tot_des = sum(r["stats"].get("n_desaturations", 0) for r in ok)
    grootste = Counter(r["stats"].get("max_group_size", 0) for r in ok)

    print(f"  hypopneus totaal      : {tot_hyp}")
    print(f"  desaturaties totaal   : {tot_des}")
    print(f"  aan een desat gekoppeld: {tot_grp} "
          f"({100.0 * tot_grp / max(tot_hyp, 1):.1f} % van de hypopneus)")
    print("\n  grootste groep per opname:")
    for k in sorted(grootste):
        print(f"    {k:2d} events op één desaturatie : {grootste[k]:3d} opnames")

    # Wat zou een limiet van 2 / 3 degraderen? De groepsgroottes zitten niet
    # per stuk in de stats, dus dit is de ondergrens uit max_group_size.
    n2 = sum(1 for r in ok if r["stats"].get("max_group_size", 0) > 2)
    n3 = sum(1 for r in ok if r["stats"].get("max_group_size", 0) > 3)
    print(f"\n  opnames waar limiet 2 iets zou doen: {n2} van {len(ok)}")
    print(f"  opnames waar limiet 3 iets zou doen: {n3} van {len(ok)}")


if __name__ == "__main__":
    main()
