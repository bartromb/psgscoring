#!/usr/bin/env python3
"""Blok 2A: het koppelvenster tussen event-einde en arousal verruimen.

Idee overgenomen uit CAISR-resp (25 s tegen onze 15 s); zie
`docs/third_party_comparison.md` rij 1 voor de herkomst. Alleen de waarde komt
over, geen code.

BESLISREGEL — vooraf vastgelegd in docs/third_party_comparison.md, niet hier:
de grootste waarde die de **event-precisie** niet onder die van het huidige
venster (15 s) brengt. Niet het F1-maximum en niet de bias, want een ruimer
venster koopt per definitie recall; de vraag is uitsluitend wat dat aan
precisie kost. Bij gelijke precisie wint de grootste waarde.

De precisie is per opname de MEDIAAN over de twaalf scoorders — één scoorder
als referentie nemen meet die scoorder, niet de detector.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \
        python scripts/sweep_arousal_window_psgipa.py --out venster.json
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


def _register(window_s, profiel):
    """Zet het venster en ververs de legacy-dicts.

    De legacy-dicts worden bij import uit profiles.py afgeleid, dus een
    wijziging aan het dataclass-profiel bereikt de detector pas na een reload
    van constants én het doorgeven aan de modules die de dict vasthouden.
    """
    from psgscoring.profiles import PROFILES as P
    P[profiel].post_processing.rule1b_arousal_window_s = float(window_s)
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    # Controle: bereikt het veld de detector werkelijk?
    d = C.SCORING_PROFILES[profiel]
    assert abs(d["RULE1B_AROUSAL_WINDOW_S"] - float(window_s)) < 1e-9, \
        f"venster bereikte de dict niet: {d['RULE1B_AROUSAL_WINDOW_S']}"


def een(args):
    sn, window_s, profiel, verwacht = args
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mne
    import psgscoring
    mne.set_log_level("ERROR")
    if verwacht and psgscoring.__version__ != verwacht:
        raise SystemExit(
            f"verkeerde psgscoring: {psgscoring.__version__} != {verwacht} "
            f"({psgscoring.__file__})")
    vp = _harness()
    _register(window_s, profiel)

    raw = mne.io.read_raw_edf(
        str(DATA / "Resp_events" / "PSG" / f"{sn}_Respiration.edf"),
        preload=True, verbose=False)
    dur = float(raw.times[-1])

    refs, hyp1, scorer_sets = [], None, []
    for i, f in enumerate(vp.find_scorer_files(DATA, sn)):
        a, _t, h = vp.parse_scorer_file(f, dur)
        if a is not None:
            refs.append(a)
            if i == 0:
                hyp1 = h
        scorer_sets.append(vp.event_set(f, dur))

    r = psgscoring.run_pneumo_analysis(raw, hypno=hyp1, scoring_profile=profiel)
    resp = r.get("respiratory") or {}
    s = resp.get("summary") or {}
    ev = resp.get("events") or []

    algo = [(float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"]),
             str(e.get("type", "")))
            for e in ev if e.get("duration_s")]
    # Een lege lijst geeft precisie 0,0 die op een meting lijkt in plaats van
    # op een defect. Laat hem klappen zolang de detector wél events opleverde.
    assert len(algo) == len(ev), (
        f"{len(ev) - len(algo)} van {len(ev)} events verloren bij het omzetten "
        f"— veldnamen gewijzigd? keys={sorted(ev[0]) if ev else []}")

    # Per scoorder de event-metriek; mediaan over de twaalf.
    per = [vp.match_events(algo, ss, iou_thresh=0.20, type_aware=False,
                           optimal=False)
           for ss in scorer_sets if ss]
    prec = [m["precision"] for m in per]
    rec = [m["recall"] for m in per]
    f1 = [m["f1"] for m in per]

    return {
        "recording": sn, "profiel": profiel, "window_s": float(window_s),
        "ahi": float(s.get("ahi_total") or 0.0),
        "ref_median": float(median(refs)), "ref_lo": float(min(refs)),
        "ref_hi": float(max(refs)),
        "precision": float(median(prec)) if prec else 0.0,
        "recall": float(median(rec)) if rec else 0.0,
        "f1": float(median(f1)) if f1 else 0.0,
        "n_scorers": len(per),
        "n_hypopnea": sum(1 for e in ev if "hypopnea" in str(e.get("type", ""))),
        "n_apnea": sum(1 for e in ev if e.get("type") in
                       ("obstructive", "central", "mixed", "uncertain")),
    }


def _sev(a):
    return "normal" if a < 5 else "mild" if a < 15 else "moderate" if a < 30 else "severe"


def samenvat(rows):
    uit = {}
    for r in rows:
        uit.setdefault(r["window_s"], []).append(r)
    regels = []
    for w, rs in sorted(uit.items()):
        d = [x["ahi"] - x["ref_median"] for x in rs]
        regels.append({
            "window_s": w, "n": len(rs),
            "precision": sum(x["precision"] for x in rs) / len(rs),
            "recall": sum(x["recall"] for x in rs) / len(rs),
            "f1": sum(x["f1"] for x in rs) / len(rs),
            "bias": sum(d) / len(d),
            "mae": sum(abs(x) for x in d) / len(d),
            "in_range": sum(1 for x in rs if x["ref_lo"] <= x["ahi"] <= x["ref_hi"]),
            "severity": sum(1 for x in rs if _sev(x["ahi"]) == _sev(x["ref_median"])),
            "n_hyp": sum(x["n_hypopnea"] for x in rs),
            "n_apn": sum(x["n_apnea"] for x in rs),
        })
    return regels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--windows", nargs="+", type=float,
                    default=[15.0, 20.0, 25.0, 30.0])
    ap.add_argument("--expect-version", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    jobs = [(sn, w, a.profile, a.expect_version) for w in a.windows for sn in RECS]
    print(f"  {len(jobs)} runs — {len(a.windows)} vensters x 5 opnames "
          f"op {a.profile}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(een, j) for j in jobs]):
            rows.append(f.result())
            if len(rows) % 5 == 0:
                print(f"    {len(rows)}/{len(jobs)}", flush=True)

    regels = samenvat(rows)
    print(f"\n  {'venster':>8s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
          f"{'bias':>7s} {'MAE':>6s} {'range':>6s} {'sev':>5s} {'hyp':>5s} {'apn':>5s}")
    for g in regels:
        print(f"  {g['window_s']:6.0f} s {g['precision']:6.3f} {g['recall']:7.3f} "
              f"{g['f1']:6.3f} {g['bias']:+7.2f} {g['mae']:6.2f} "
              f"{g['in_range']:4d}/5 {g['severity']:3d}/5 "
              f"{g['n_hyp']:5d} {g['n_apn']:5d}")

    # Beslisregel: grootste venster dat de precisie niet onder die van 15 s brengt.
    basis = next((g for g in regels if g["window_s"] == 15.0), regels[0])
    houdbaar = [g for g in regels if g["precision"] >= basis["precision"] - 1e-9]
    keuze = max(houdbaar, key=lambda g: g["window_s"]) if houdbaar else basis
    print(f"\n  basis {basis['window_s']:.0f} s: precisie {basis['precision']:.3f}")
    print(f"  -> gekozen venster {keuze['window_s']:.0f} s "
          f"(precisie {keuze['precision']:.3f}, recall {keuze['recall']:.3f}, "
          f"F1 {keuze['f1']:.3f})")
    if keuze["window_s"] == basis["window_s"]:
        print("  -> elke verruiming kost precisie; het venster blijft 15 s.")

    if a.out:
        a.out.write_text(json.dumps({"rows": rows, "summary": regels}, indent=2))
        print(f"\n  geschreven: {a.out}")


if __name__ == "__main__":
    main()
