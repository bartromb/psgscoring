#!/usr/bin/env python3
"""Wat doet de gerepareerde RIP-poort op MESA?

Draait elke opname twee keer — `rip_quality_scale_free` uit en aan — en
rapporteert wat er verschuift: de poortmodus, het aandeel `uncertain`, en
`ahi_total`. Die laatste beweegt mee omdat kale `uncertain` BUITEN `ahi_total`
valt; het is dus geen typeringswijziging maar een indexwijziging.

De AHI-kolom wordt gemeten NA de kalibratie van de drempel, nooit ervoor: de
drempel is op de vorm van het signaal gekozen en mag niet op de uitkomst
worden afgesteld. Zie de beslisregel in CHANGELOG v0.17.0.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \
        python scripts/effect_rip_gate_mesa.py --n 25 --out rip.json
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
from statistics import median

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EDF_DIR = Path("/home/bart/MESA/mesa/polysomnography/edfs")


def _hypno_uit_nsrr(xml_path, dur_s):
    spec = importlib.util.spec_from_file_location(
        "sub", REPO / "scripts" / "subtype_agreement_mesa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sub"] = m
    spec.loader.exec_module(m)
    return m._ref_apneas(xml_path, dur_s)


def _register(scale_free, profiel):
    from psgscoring.profiles import PROFILES as P
    P[profiel].post_processing.rip_quality_scale_free = bool(scale_free)
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    assert C.SCORING_PROFILES[profiel]["RIP_QUALITY_SCALE_FREE"] is bool(scale_free)


def _tel(resp):
    ev = resp.get("events") or []
    s = resp.get("summary") or {}
    t = Counter(str(e.get("type", "")) for e in ev)
    return {
        "ahi_total": float(s.get("ahi_total") or 0.0),
        "ahi_incl_uncertain": float(s.get("ahi_incl_uncertain") or 0.0),
        "n_uncertain": t.get("uncertain", 0),
        "n_obstructive": t.get("obstructive", 0),
        "n_central": t.get("central", 0),
        "n_mixed": t.get("mixed", 0),
        "mode": (resp.get("_sq") or {}).get("recommended_mode"),
    }


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
        xml = (fn.parent.parent / "annotations-events-nsrr"
               / f"{fn.stem}-nsrr.xml")
        if not xml.exists():
            return {"recording": fn.stem, "error": "geen NSRR-annotatie"}
        raw = mne.io.read_raw_edf(str(fn), preload=True, verbose=False)
        _ref, hypno = _hypno_uit_nsrr(xml, float(raw.times[-1]))
        if not any(s in ("N1", "N2", "N3", "R") for s in hypno):
            return {"recording": fn.stem, "error": "geen slaap in de staging"}

        uit = {"recording": fn.stem}
        for sf_aan in (False, True):
            _register(sf_aan, profiel)
            r = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                               scoring_profile=profiel)
            resp = dict(r.get("respiratory") or {})
            resp["_sq"] = r.get("signal_quality") or {}
            uit["aan" if sf_aan else "uit"] = _tel(resp)
        return uit
    except Exception as e:
        return {"recording": fn.stem, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aasm_v3_rec")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--expect-version", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    alle = sorted(EDF_DIR.glob("mesa-sleep-*.edf"))
    random.Random(a.seed).shuffle(alle)
    keuze = alle[:a.n]
    print(f"  {len(keuze)} opnames, profiel {a.profile}, twee runs elk",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(een, (fn, a.profile, a.expect_version))
                               for fn in keuze]):
            rows.append(f.result())
            if len(rows) % 5 == 0:
                print(f"    {len(rows)}/{len(keuze)}", flush=True)

    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"\n  geschreven: {a.out}")

    ok = [r for r in rows if "error" not in r]
    print(f"\n  bruikbaar: {len(ok)} van {len(rows)}")
    for f_, n in Counter(r.get("error", "").split(":")[0]
                         for r in rows if "error" in r).most_common():
        print(f"    fout: {f_} — {n}x")
    if not ok:
        print("  GEEN bruikbare opnames — rapporteer dat, verzin geen getal.")
        return

    print("\n  poortmodus:")
    for tak in ("uit", "aan"):
        c = Counter(r[tak]["mode"] for r in ok)
        print(f"    {tak:3s}: {dict(c)}")

    def som(tak, k):
        return sum(r[tak][k] for r in ok)

    print(f"\n  {'':22s} {'uit':>10s} {'aan':>10s}")
    for k in ("n_uncertain", "n_obstructive", "n_central", "n_mixed"):
        print(f"  {k:22s} {som('uit', k):10d} {som('aan', k):10d}")

    for k in ("ahi_total", "ahi_incl_uncertain"):
        u = median(r["uit"][k] for r in ok)
        n = median(r["aan"][k] for r in ok)
        print(f"  {k:22s} {u:10.2f} {n:10.2f}   (mediaan)")

    d = [r["aan"]["ahi_total"] - r["uit"]["ahi_total"] for r in ok]
    print(f"\n  ahi_total verschuift gemiddeld {sum(d) / len(d):+.2f}/u "
          f"(mediaan {median(d):+.2f}, max {max(d):+.2f})")
    print(f"  opnames waar ahi_total verandert: "
          f"{sum(1 for x in d if abs(x) > 0.005)} van {len(ok)}")


if __name__ == "__main__":
    main()
