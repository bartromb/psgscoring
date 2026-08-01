#!/usr/bin/env python3
"""Held-out validatie op MESA/NSRR: aasm_v3_breath tegenover aasm_v3_rec.

Waarvoor dit dient
------------------
Het werkpunt van ``aasm_v3_breath`` (``hypopnea_strictness`` 0,50) is gekozen
op PSG-IPA — vijf opnames, en dezelfde vijf waarop het resultaat gerapporteerd
werd. Dat is een fit, geen validatie. MESA is held-out: geen enkele parameter
is erop afgesteld, en dit script stelt er ook niets op af. Het meet alleen.

De uitkomst is een GEPAARD verschil per opname. Absolute overeenstemming met
MESA is niet de vraag en zou ook misleiden, want MESA is met een andere regel
gescoord (zie hieronder); beide profielen draaien onder exact dezelfde
condities tegen dezelfde referentie, dus het verschil is wél zinvol.

Twee referentiesets, want MESA's regel is niet die van AASM v3 Rule 1A
--------------------------------------------------------------------
MESA scoorde hypopneeën met >=4% desaturatie en labelde de gevallen die dat
niet haalden — de 3%-of-arousal-gevallen — als ``Unsure``. Beide profielen
implementeren Rule 1A (>=3% OF arousal), dus:

  ``mesa4``   Hypopnea + apneus. MESA's eigen AHI-definitie.
  ``rule1a``  idem + Unsure. Dichter bij wat Rule 1A hoort te vangen.

Beide worden gerapporteerd. Een profiel dat op ``rule1a`` wint en op ``mesa4``
verliest heeft geen fout gemaakt — het volgt een andere regel dan MESA.

Het hypnogram komt uit de NSRR-annotatie, niet uit YASA, zodat het verschil
tussen de profielen alleen de respiratoire scoring betreft.

Gebruik
-------
    python scripts/validate_mesa.py --data-dir /home/bart/MESA/mesa \\
        --n 50 --seed 20260801 --workers 5 --output-json mesa_holdout.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mne  # noqa: E402

mne.set_log_level("ERROR")

from validate_psgipa import (  # noqa: E402
    LEGACY_MATCHER,
    match_events,
    severity,
)

EPOCH_S = 30.0

STAGE_MAP = {
    "wake": "W",
    "stage 1 sleep": "N1",
    "stage 2 sleep": "N2",
    "stage 3 sleep": "N3",
    "stage 4 sleep": "N3",
    "rem sleep": "R",
}

# EventConcept -> eventtype voor de matcher
RESP_MAP = {
    "obstructive apnea": "obstructive",
    "central apnea": "central",
    "mixed apnea": "mixed",
    "hypopnea": "hypopnea",
    "unsure": "hypopnea",
}
REFERENCES = {
    "mesa4": {"obstructive apnea", "central apnea", "mixed apnea", "hypopnea"},
    "rule1a": {"obstructive apnea", "central apnea", "mixed apnea", "hypopnea",
               "unsure"},
}
PROFILES = ("aasm_v3_rec", "aasm_v3_breath")


# ─────────────────────────────────────────────────────────────────
#  NSRR-annotatie
# ─────────────────────────────────────────────────────────────────

def parse_nsrr(xml_path, signal_duration_s):
    """(hypno, events_per_referentie, tst_h) uit een NSRR-annotatiebestand.

    ``events`` is per referentieset een lijst (onset, offset, type), in
    hetzelfde formaat als ``validate_psgipa.event_set``.
    """
    root = ET.parse(str(xml_path)).getroot()
    n_epochs = int(math.ceil(signal_duration_s / EPOCH_S))
    hypno = ["W"] * n_epochs
    raw_events = []

    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").split("|")[0].strip().lower()
        try:
            start = float(ev.findtext("Start") or "nan")
            dur = float(ev.findtext("Duration") or "nan")
        except ValueError:
            continue
        if not (np.isfinite(start) and np.isfinite(dur)):
            continue
        if start < 0 or start >= signal_duration_s:
            continue

        st = STAGE_MAP.get(concept)
        if st is not None:
            ep0 = int(start // EPOCH_S)
            for i in range(max(1, int(round(dur / EPOCH_S)))):
                if 0 <= ep0 + i < n_epochs:
                    hypno[ep0 + i] = st
            continue

        if concept in RESP_MAP:
            raw_events.append((start, start + dur, RESP_MAP[concept], concept))

    tst_h = sum(1 for s in hypno if s in ("N1", "N2", "N3", "R")) * EPOCH_S / 3600.0
    events = {
        name: [(a, b, t) for a, b, t, c in raw_events if c in concepts]
        for name, concepts in REFERENCES.items()
    }
    return hypno, events, tst_h


# ─────────────────────────────────────────────────────────────────
#  Eén opname
# ─────────────────────────────────────────────────────────────────

def analyse_one(args):
    rec_id, data_dir = args
    data_dir = Path(data_dir)
    edf = data_dir / "polysomnography" / "edfs" / f"{rec_id}.edf"
    xml = (data_dir / "polysomnography" / "annotations-events-nsrr"
           / f"{rec_id}-nsrr.xml")
    try:
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"EDF: {e}"}

    dur = float(raw.times[-1])
    try:
        hypno, refs, tst_h = parse_nsrr(xml, dur)
    except Exception as e:  # noqa: BLE001
        return {"recording": rec_id, "error": f"XML: {e}"}
    if tst_h < 1.0:
        return {"recording": rec_id, "error": f"TST {tst_h:.2f} h te kort"}

    import psgscoring

    out = {"recording": rec_id, "tst_h": tst_h, "duration_h": dur / 3600.0,
           "n_ref": {k: len(v) for k, v in refs.items()},
           "ahi_ref": {k: len(v) / tst_h for k, v in refs.items()},
           "profiles": {}}

    for prof in PROFILES:
        try:
            res = psgscoring.run_pneumo_analysis(
                raw, hypno=hypno, scoring_profile=prof)
        except Exception as e:  # noqa: BLE001
            out["profiles"][prof] = {"error": str(e)}
            continue
        r = res.get("respiratory", {}) or {}
        summ = r.get("summary", {}) or {}
        algo = [(float(e["onset_s"]),
                 float(e["onset_s"]) + float(e.get("duration_s") or 0.0),
                 e.get("type"))
                for e in r.get("events", []) if e.get("onset_s") is not None]
        rec = {
            "ahi": float(summ.get("ahi_total") or summ.get("ahi") or 0.0),
            "n_events": len(algo),
            "n_hypopnea": sum(1 for e in algo if "hypopnea" in str(e[2])),
            "n_apnea": sum(1 for e in algo
                           if "apnea" in str(e[2]) and "hypo" not in str(e[2])),
            "match": {},
        }
        for name, ref in refs.items():
            m = match_events(algo, ref, **LEGACY_MATCHER)
            rec["match"][name] = {k: m[k] for k in
                                  ("f1", "precision", "recall", "tp", "fp", "fn")}
        bd = r.get("breath_detector")
        if bd:
            rec["breath_detector"] = bd
        out["profiles"][prof] = rec
    return out


# ─────────────────────────────────────────────────────────────────
#  Gepaarde statistiek
# ─────────────────────────────────────────────────────────────────

def wilcoxon_signed_rank(deltas):
    """Tweezijdige p-waarde, normale benadering met continuiteitscorrectie.

    Zelf geimplementeerd zodat het script geen scipy nodig heeft. Nulverschillen
    vallen af, zoals gebruikelijk bij Wilcoxon.
    """
    d = [x for x in deltas if abs(x) > 1e-12]
    n = len(d)
    if n < 6:
        return None, n
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sd == 0:
        return None, n
    z = (abs(w_plus - mu) - 0.5) / sd
    p = math.erfc(z / math.sqrt(2.0))
    return p, n


def report(rows, ref_name):
    ok = [r for r in rows if "error" not in r
          and all(p in r["profiles"] and "error" not in r["profiles"][p]
                  for p in PROFILES)]
    if not ok:
        print(f"  geen bruikbare opnames voor referentie {ref_name}")
        return

    print(f"\n{'═' * 72}")
    print(f"  REFERENTIE: {ref_name}   (n = {len(ok)} opnames)")
    print(f"{'═' * 72}")
    print(f"  {'profiel':16s} {'F1 med':>7s} {'F1 gem':>7s} {'prec':>6s} "
          f"{'rec':>6s} {'bias':>7s} {'MAE':>6s} {'sev':>7s}")
    stats = {}
    for prof in PROFILES:
        f1 = [r["profiles"][prof]["match"][ref_name]["f1"] for r in ok]
        pr = [r["profiles"][prof]["match"][ref_name]["precision"] for r in ok]
        rc = [r["profiles"][prof]["match"][ref_name]["recall"] for r in ok]
        d = [r["profiles"][prof]["ahi"] - r["ahi_ref"][ref_name] for r in ok]
        sev = sum(1 for r in ok
                  if severity(r["profiles"][prof]["ahi"])
                  == severity(r["ahi_ref"][ref_name]))
        stats[prof] = f1
        print(f"  {prof:16s} {median(f1):7.3f} {np.mean(f1):7.3f} "
              f"{median(pr):6.3f} {median(rc):6.3f} "
              f"{np.mean(d):+7.2f} {np.mean(np.abs(d)):6.2f} "
              f"{sev:3d}/{len(ok):<3d}")

    delta = [b - a for a, b in zip(stats[PROFILES[0]], stats[PROFILES[1]])]
    better = sum(1 for x in delta if x > 0)
    worse = sum(1 for x in delta if x < 0)
    p, n_eff = wilcoxon_signed_rank(delta)
    print(f"\n  GEPAARD  {PROFILES[1]} - {PROFILES[0]}:")
    print(f"    mediaan dF1 {median(delta):+.4f}   gemiddeld {np.mean(delta):+.4f}")
    print(f"    beter {better}/{len(ok)}   slechter {worse}/{len(ok)}   "
          f"gelijk {len(ok) - better - worse}")
    print(f"    Wilcoxon (n={n_eff}): "
          + ("p niet bepaald, te weinig verschillen" if p is None
             else f"p = {p:.4g}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=50,
                    help="aantal held-out opnames (0 = alle)")
    ap.add_argument("--seed", type=int, default=20260801,
                    help="steekproefzaad; vastleggen en niet variëren")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument("--recordings", nargs="+", default=None,
                    help="expliciete opname-ids (overschrijft de steekproef)")
    a = ap.parse_args()

    edf_dir = a.data_dir / "polysomnography" / "edfs"
    xml_dir = a.data_dir / "polysomnography" / "annotations-events-nsrr"
    ids = sorted(p.stem for p in edf_dir.glob("*.edf")
                 if (xml_dir / f"{p.stem}-nsrr.xml").exists())
    if a.recordings:
        picked = list(a.recordings)
    elif a.n and a.n < len(ids):
        picked = sorted(random.Random(a.seed).sample(ids, a.n))
    else:
        picked = ids

    print(f"MESA held-out validatie — {len(picked)} van {len(ids)} opnames "
          f"(seed {a.seed})")
    print(f"profielen: {', '.join(PROFILES)}")
    print("het werkpunt is op PSG-IPA gekozen en wordt hier NIET aangepast\n")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(analyse_one,
                                     [(x, str(a.data_dir)) for x in picked]), 1):
            rows.append(r)
            if "error" in r:
                print(f"  [{i}/{len(picked)}] {r['recording']}: FOUT {r['error']}")
            else:
                f = {p: r["profiles"].get(p, {}).get("match", {})
                      .get("rule1a", {}).get("f1") for p in PROFILES}
                print(f"  [{i}/{len(picked)}] {r['recording']}  "
                      + "  ".join(f"{p.split('_')[-1]} F1 "
                                  f"{'--' if f[p] is None else format(f[p], '.3f')}"
                                  for p in PROFILES))

    for ref in REFERENCES:
        report(rows, ref)

    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        print(f"\n  {n_err} opnames overgeslagen wegens fouten")

    if a.output_json:
        a.output_json.write_text(json.dumps(
            {"seed": a.seed, "n_requested": len(picked), "profiles": list(PROFILES),
             "results": rows}, indent=2))
        print(f"\n  JSON weggeschreven: {a.output_json}")


if __name__ == "__main__":
    main()
