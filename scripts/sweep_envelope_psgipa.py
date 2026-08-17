#!/usr/bin/env python3
"""De enveloppe-as tegen menselijke scoring op PSG-IPA.

Vijf armen die in precies één opzicht verschillen — hoe de amplitude-enveloppe
wordt gebouwd — tegen de twaalf scoorders van elke opname. `aasm_v3_rec` draait
mee als anker in DEZELFDE run: PSG-IPA-AHI's zijn harnasgevoelig, dus cijfers
uit twee runs naast elkaar leggen verandert twee dingen tegelijk.

BESLISREGEL — vooraf vastgelegd in de CHANGELOG bij de enveloppe-as, niet hier:

  * `hilbert_chunked` mag default worden als, en alleen als, bias én event-F1
    onveranderd zijn tot op de gerapporteerde precisie én de grensoffsets niet
    verschuiven. Het is een geheugen- en snelheidsoptimalisatie; kost hij
    nauwkeurigheid, dan is hij het niet waard.
  * `rectify_lowpass`, `breath_amplitude` en `envelope_fs` zijn
    methodewijzigingen. Die blijven uit ongeacht hoe ze meten, tenzij fysiologisch
    beargumenteerd en als afwijking gerapporteerd. Een betere F1 op één cohort
    is niet voldoende.

Dat staat er omdat dit script anders een fitprocedure wordt: met vijf armen en
vier maten is er altijd wel een arm die op één maat wint.

CONVENTIES — gelijk aan `sweep_arousal_window_psgipa.py` en paper v31, zodat de
getallen naast de bestaande tabellen te leggen zijn:

  * hypnogram van scoorder 1 voor alle armen, dus teller én noemer identiek;
  * precisie/recall/F1 per opname = MEDIAAN over de twaalf scoorders. Eén
    scoorder als referentie nemen meet die scoorder, niet de detector;
  * matcher in legacy-modus (IoU 0,20, type-onbewust, greedy) — de tak waar de
    gepubliceerde cijfers aan hangen;
  * `PSGSCORING_AROUSAL_DERIVATION=single`, zoals elke eerdere PSG-IPA-meting.
    Zonder dat draaien de armen op multi-derivatie en is de vergelijking met de
    bestaande tabellen scheef.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \
        python scripts/sweep_envelope_psgipa.py --out enveloppe.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path("/home/bart/PSG-IPA")
RECS = [f"SN{i}" for i in range(1, 6)]

# Anker eerst. De vier armen daarna, in de volgorde van de opties.
PROFILES = [
    "aasm_v3_rec",
    "aasm_v3_env_chunked",
    "aasm_v3_env_rectify",
    "aasm_v3_env_breath",
    "aasm_v3_env_decimated",
]


def _harness():
    spec = importlib.util.spec_from_file_location("vp", REPO / "validate_psgipa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["vp"] = m
    spec.loader.exec_module(m)
    return m


def _require(profiles, verwacht):
    """Faal luid in plaats van stil de verkeerde versie of een leeg profiel te meten.

    In `~/.local/lib/python3.12/site-packages` staat een psgscoring van vóór het
    profielregister. Een worker start een verse interpreter en erft de werkmap
    niet, dus deze controle hoort in de worker en niet alleen in main.
    """
    import psgscoring
    from psgscoring.profiles import PROFILES as P
    if verwacht and psgscoring.__version__ != verwacht:
        raise SystemExit(
            f"verkeerde psgscoring: {psgscoring.__version__} != {verwacht} "
            f"({psgscoring.__file__})")
    ontbreekt = [p for p in profiles if p not in P]
    if ontbreekt:
        raise SystemExit(f"profielen bestaan niet in deze versie: {ontbreekt}")
    # De as moet de detector werkelijk bereiken; een profiel dat wél bestaat maar
    # zijn enveloppeveld niet doorgeeft, meet het anker vijf keer.
    import psgscoring.constants as C
    for p in profiles:
        d = C.SCORING_PROFILES[p]
        pp = P[p].post_processing
        assert d.get("ENVELOPE_METHOD") == pp.envelope_method, (
            f"{p}: enveloppeveld bereikt de legacy-dict niet")
    return psgscoring.__version__


def een(args):
    sn, profiel, verwacht = args
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mne

    import psgscoring
    mne.set_log_level("ERROR")
    _require([profiel], verwacht)
    vp = _harness()

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
    if not refs or hyp1 is None:
        return {"recording": sn, "profiel": profiel, "error": "geen scoordersdata"}

    r = psgscoring.run_pneumo_analysis(raw, hypno=hyp1, scoring_profile=profiel)
    resp = r.get("respiratory") or {}
    s = resp.get("summary") or {}
    ev = resp.get("events") or []

    algo = [(float(e["onset_s"]), float(e["onset_s"]) + float(e["duration_s"]),
             str(e.get("type", "")))
            for e in ev if e.get("duration_s")]
    # Een lege lijst geeft precisie 0,0 die op een meting lijkt in plaats van op
    # een defect. Laat hem klappen zolang de detector wél events opleverde.
    assert len(algo) == len(ev), (
        f"{len(ev) - len(algo)} van {len(ev)} events verloren bij het omzetten "
        f"— veldnamen gewijzigd? keys={sorted(ev[0]) if ev else []}")

    per = [vp.match_events(algo, ss, iou_thresh=0.20, type_aware=False,
                           optimal=False)
           for ss in scorer_sets if ss]

    return {
        "recording": sn, "profiel": profiel,
        "psgscoring": psgscoring.__version__,
        "ahi": float(s.get("ahi_total") or 0.0),
        "ref_median": float(median(refs)),
        "ref_lo": float(min(refs)), "ref_hi": float(max(refs)),
        "precision": float(median(m["precision"] for m in per)) if per else 0.0,
        "recall": float(median(m["recall"] for m in per)) if per else 0.0,
        "f1": float(median(m["f1"] for m in per)) if per else 0.0,
        "n_scorers": len(per),
        "n_events": len(ev),
        "n_hypopnea": sum(1 for e in ev if "hypopnea" in str(e.get("type", ""))),
        "n_apnea": sum(1 for e in ev if e.get("type") in
                       ("obstructive", "central", "mixed", "uncertain")),
    }


def _sev(a):
    return "normal" if a < 5 else "mild" if a < 15 else "moderate" if a < 30 else "severe"


def samenvat(rows, profiles):
    uit = []
    for p in profiles:
        rs = [r for r in rows if r.get("profiel") == p and "error" not in r]
        if not rs:
            continue
        d = [x["ahi"] - x["ref_median"] for x in rs]
        uit.append({
            "profiel": p, "n": len(rs),
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
    return uit


def rapport(rows, samen, profiles):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("geen bruikbare resultaten")
        return
    kort = {p: p.replace("aasm_v3_", "").replace("env_", "") for p in profiles}

    n_rec = len({r["recording"] for r in ok})
    print(f"\npsgscoring {ok[0]['psgscoring']} · n = {n_rec}"
          f" · hypnogram scoorder 1 · arousal-modus "
          f"{os.environ.get('PSGSCORING_AROUSAL_DERIVATION', '(profiel)')}\n")

    hdr = (f"{'':5s} {'human':>6s} {'range':>13s} "
           + " ".join(f"{kort[p]:>10s}" for p in profiles))
    print("AHI per opname")
    print(hdr)
    print("-" * len(hdr))
    for sn in sorted({r["recording"] for r in ok}):
        rs = {r["profiel"]: r for r in ok if r["recording"] == sn}
        a = next(iter(rs.values()))
        line = f"{sn:5s} {a['ref_median']:6.2f} {a['ref_lo']:5.2f}-{a['ref_hi']:6.2f} "
        line += " ".join(f"{rs[p]['ahi']:10.2f}" if p in rs else f"{'—':>10s}"
                         for p in profiles)
        print(line)

    print(f"\n{'profiel':24s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
          f"{'bias':>7s} {'MAE':>6s} {'in range':>9s} {'severity':>9s} {'events':>7s}")
    print("-" * 95)
    anker = samen[0] if samen else None
    for s in samen:
        merk = "" if s is anker else (
            f"   ΔF1 {s['f1'] - anker['f1']:+.3f}  Δbias {s['bias'] - anker['bias']:+.2f}")
        print(f"{s['profiel']:24s} {s['precision']:6.3f} {s['recall']:7.3f} "
              f"{s['f1']:6.3f} {s['bias']:+7.2f} {s['mae']:6.2f} "
              f"{s['in_range']:6d}/{s['n']} {s['severity']:6d}/{s['n']} "
              f"{s['n_apn'] + s['n_hyp']:7d}{merk}")

    if anker:
        print(f"\nAnker = {anker['profiel']}. Verschillen zijn binnen deze run "
              f"berekend; PSG-IPA-AHI's zijn harnasgevoelig en niet vergelijkbaar "
              f"met cijfers uit een andere run.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", nargs="+", default=PROFILES,
                    help="anker eerst; standaard aasm_v3_rec + de vier armen")
    ap.add_argument("--recordings", nargs="+", default=RECS)
    ap.add_argument("--expect-version", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    if not os.environ.get("PSGSCORING_AROUSAL_DERIVATION"):
        print("LET OP: PSGSCORING_AROUSAL_DERIVATION staat niet op 'single'; "
              "de cijfers zijn dan niet vergelijkbaar met de eerdere "
              "PSG-IPA-tabellen.", file=sys.stderr)

    v = _require(a.profiles, a.expect_version)
    jobs = [(sn, p, a.expect_version) for p in a.profiles for sn in a.recordings]
    print(f"psgscoring {v} uit {REPO} — {len(jobs)} runs "
          f"({len(a.profiles)} profielen x {len(a.recordings)} opnames)", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for f in as_completed([ex.submit(een, j) for j in jobs]):
            r = f.result()
            rows.append(r)
            print(f"  klaar: {r['recording']} {r['profiel']}"
                  + (f"  ({r['error']})" if "error" in r else
                     f"  AHI {r['ahi']:.2f}  F1 {r['f1']:.3f}"), flush=True)

    rows.sort(key=lambda d: (a.profiles.index(d["profiel"]), d["recording"]))
    samen = samenvat(rows, a.profiles)
    rapport(rows, samen, a.profiles)

    if a.out:
        a.out.write_text(json.dumps({"per_opname": rows, "samenvatting": samen},
                                    indent=2))
        print(f"\ngeschreven: {a.out}")


if __name__ == "__main__":
    main()
