#!/usr/bin/env python3
"""
sweep_event_locked_window.py — het event-locked venster op beide cohorten,
parallel over opnames.

WAT ER GEMETEN WORDT
--------------------
Twee grootheden per arm, en een vlag is alleen aanvaardbaar als ze allebei
kloppen:

  koppeling `ons x mens`  onze arousals tegen MENSELIJKE events. Teller en
                          noemer uit verschillende bronnen, dus een vlag die
                          simpelweg meer arousals oplevert kan hem niet
                          optillen.
  telling / referentie    onze arousaltelling gedeeld door de menselijke.
                          Vooraf vastgelegde grens: **1,10** (uit
                          docs/arousal-recall-diagnose.md, gezet om niet terug
                          te vallen in de 1,47x-overtelling van vóór v0.26).

PARALLELLISME
-------------
Eén proces per OPNAME, alle armen binnen dat proces. De EDF wordt dan één keer
geladen in plaats van één keer per arm -- bij vijf armen scheelt dat een factor
vijf aan I/O en geheugen.

De Z6 bevriest THERMISCH rond 83 °C bij aanhoudende all-core belasting, niet
door geheugen (zie scripts/thermal_guard.sh). Draai daarom onder
`systemd-run --user --property=CPUQuota=...` met de guard ernaast, en houd
`--workers` ruim onder het aantal cores.

Gebruik:
    python scripts/sweep_event_locked_window.py --cohort psgipa --workers 5
    python scripts/sweep_event_locked_window.py --cohort mesa --n 20 --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import median

# EEN thread per worker. Zonder dit opent elke worker via numpy/MNE zijn eigen
# BLAS-threadpool ter grootte van het hele CPU-aantal; tien workers leveren dan
# honderden draaiende threads op. Een CPUQuota knijpt de CPU-TIJD af maar niet
# het AANTAL threads, en aanhoudende all-core belasting is de heetste last die
# er bestaat.
#
# Op 25-08-2026 bevroor de Z6 hierop tijdens deze meting: van 68 naar 82 °C in
# tien seconden, bij crit 84, terwijl de quota op vijftien kernen stond en de
# load op 24-33. Zie scripts/thermal_guard.sh.
#
# Moet VOOR de numpy/MNE-import staan, anders is de threadpool al aangemaakt.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

EPOCH_S = 30.0
RESP_TXT = re.compile(
    r"\b(apnea|apnoea|apneu|hypopnea|hypopnoea|hypopneu|rera)\b", re.IGNORECASE)
SLEEP_XML = {"Stage 1 sleep|1", "Stage 2 sleep|2", "Stage 3 sleep|3",
             "Stage 4 sleep|4", "REM sleep|5"}


def _f1(ons, mens, iou=0.20):
    """Event-level F1 tussen twee arousallijsten, projectstandaard IoU 0,20.

    De koppelingsfractie zegt of er een arousal OP een respiratoir event volgt;
    F1 zegt of onze arousals staan waar de MENSELIJKE staan. Dat is de
    strengere vraag: een vlag kan de koppeling optillen door meer arousals bij
    event-eindes te zetten zonder dat die met menselijke arousals samenvallen.

    Beide lijsten zijn (onset, duur)-paren. Hergebruikt de matcher van
    psgscoring.agreement -- niet nagebouwd, want dan meet je je eigen nabouw.
    """
    from psgscoring.agreement import _match
    a = [{"onset_s": o, "duration_s": max(d, 0.1), "type": "arousal"}
         for o, d in ons]
    b = [{"onset_s": o, "duration_s": max(d, 0.1), "type": "arousal"}
         for o, d in mens]
    if not a or not b:
        return None
    pairs, only_a, only_b = _match(a, b, iou)
    n = len(pairs)
    noemer = 2 * n + len(only_a) + len(only_b)
    return (2 * n / noemer) if noemer else None


def _koppel(events, arousal_onsets):
    """Fractie events met een arousal in het koppelvenster."""
    from psgscoring.arousal import arousal_couples_to_event
    if not events:
        return None
    n = sum(1 for o, d in events
            if any(arousal_couples_to_event(x, o, o + d, 15.0)
                   for x in arousal_onsets))
    return n / len(events)


def _armen(raw, hypno, profiel, cmap, art, drempels):
    """Alle armen op één al geladen opname."""
    import psgscoring
    uit = {}
    for naam, waarde in drempels:
        os.environ["PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD"] = (
            "" if waarde is None else str(waarde))
        res = psgscoring.run_pneumo_analysis(
            raw.copy(), hypno=hypno, channel_map=cmap,
            artifact_epochs=art, scoring_profile=profiel)
        _ev = (res.get("arousal") or {}).get("events", [])
        ar = [float(e["onset_s"]) for e in _ev if e.get("onset_s") is not None]
        ar_dur = [(float(e["onset_s"]), float(e.get("duration_s") or 3.0))
                  for e in _ev if e.get("onset_s") is not None]
        n_ev = len((res.get("respiratory") or {}).get("events", []))
        # Hoeveel arousals het VENSTER heeft toegelaten. Zonder dit getal is
        # een uitblijvend effect niet te onderscheiden van een vlag die
        # helemaal niet gedraaid heeft -- en op de breath-profielen ziet het
        # venster alleen de apneus, want de hypopneeen bestaan bij de
        # arousalstap nog niet.
        nested = ((res.get("arousal") or {}).get("arousals") or {}).get(
            "summary") or {}
        uit[naam] = {"n_arousals": len(ar), "onsets": ar, "spans": ar_dur,
                     "n_onze_ev": n_ev,
                     "n_window": nested.get("n_event_locked")}
    return uit


# ── PSG-IPA ───────────────────────────────────────────────────────────
def _psgipa_spans(txt_path, dur_s):
    """(onset, duur) van de arousals van één scoorder. `parse_scorer` geeft
    alleen onsets terug en F1 heeft de duur nodig."""
    uit = []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            q = [x.strip() for x in line.split(",")]
            if len(q) < 5 or "arousal" not in q[4].lower():
                continue
            try:
                o, d = float(q[2]), float(q[3])
            except ValueError:
                continue
            if 0 <= o < dur_s:
                uit.append((o, d or 3.0))
    return uit


def _psgipa_een(sn, data_dir, profiel, drempels):
    from measure_our_arousal_coupling_psgipa import parse_scorer
    root = Path(data_dir) / "Resp_events"
    psg = root / "PSG" / f"{sn}_Respiration.edf"
    hdr = mne.io.read_raw_edf(psg, preload=False, verbose=False)
    dur = hdr.n_times / hdr.info["sfreq"]
    per, hypnos, spans = [], [], []
    for sc in sorted((root / "Annotations" / "manual")
                     .glob(f"{sn}_Respiration_manual_scorer*.txt")):
        h, ev, ar = parse_scorer(sc, dur)
        per.append((ev, ar)); hypnos.append(h)
        spans.append(_psgipa_spans(sc, dur))
    n_ep = min(len(h) for h in hypnos)
    hypno = [max(set(x), key=x.count)
             for x in ([h[i] for h in hypnos] for i in range(n_ep))]
    raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
    armen = _armen(raw, hypno, profiel, None, None, drempels)
    mens_ar = median([len(ar) for _e, ar in per])
    rij = {"rec": sn, "mens_ar": mens_ar,
           "mm": median([_koppel(ev, ar) for ev, ar in per]), "armen": {}}
    for naam, a in armen.items():
        f1s = [x for x in (_f1(a["spans"], sp) for sp in spans) if x is not None]
        rij["armen"][naam] = {
            "n": a["n_arousals"], "n_window": a["n_window"],
            "om": median([_koppel(ev, a["onsets"]) for ev, _ar in per]),
            "f1": median(f1s) if f1s else None,
            "n_mens_ev": median([len(ev) for ev, _a in per]),
        }
    return rij


# ── MESA ──────────────────────────────────────────────────────────────
def _nsrr(xml_path):
    root = ET.parse(xml_path).getroot()
    events, arousals, tst = [], [], 0.0
    for ev in root.iter("ScoredEvent"):
        c = ev.findtext("EventConcept") or ""
        try:
            o = float(ev.findtext("Start") or -1)
            d = float(ev.findtext("Duration") or 0)
        except ValueError:
            continue
        if c in SLEEP_XML:
            tst += d
        cl = c.lower()
        if "arousal" in cl:
            arousals.append((o, d or 3.0))
        elif "apnea" in cl or "hypopnea" in cl:
            events.append((o, d))
    return events, arousals, tst / 3600.0


def _mesa_een(stem, mesa_dir, profiel, drempels):
    from validate_mesa import parse_nsrr
    edfs = Path(mesa_dir) / "polysomnography" / "edfs"
    xmls = Path(mesa_dir) / "polysomnography" / "annotations-events-nsrr"
    edf, xml = edfs / f"{stem}.edf", xmls / f"{stem}-nsrr.xml"
    h = mne.io.read_raw_edf(edf, preload=False, verbose=False)
    dur = h.n_times / h.info["sfreq"]
    hypno, _r, _t = parse_nsrr(xml, dur)
    events, ar_spans, _tst = _nsrr(xml)
    arousals = [o for o, _d in ar_spans]
    if not arousals or not events:
        return None
    houden = {c for c in h.ch_names
              if (c.upper().startswith("EEG") and not c.upper().endswith("_OFF"))
              or c in ("EMG", "Pres", "Therm", "Thor", "Abdo", "SpO2")}
    raw = mne.io.read_raw_edf(
        edf, exclude=[c for c in h.ch_names if c not in houden],
        preload=True, verbose=False)
    armen = _armen(raw, hypno, profiel, None, None, drempels)
    rij = {"rec": stem, "mens_ar": len(arousals),
           "mm": _koppel(events, arousals), "armen": {}}
    for naam, a in armen.items():
        rij["armen"][naam] = {"n": a["n_arousals"], "n_window": a["n_window"],
                              "om": _koppel(events, a["onsets"]),
                              "f1": _f1(a["spans"], ar_spans),
                              "n_mens_ev": len(events)}
    return rij


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["psgipa", "mesa"], required=True)
    ap.add_argument("--data-dir", default=str(Path.home() / "PSG-IPA"))
    ap.add_argument("--mesa-dir", default="/home/bart/MESA/mesa")
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--arms", default="uit,0.75,0.70,0.65,0.60")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    drempels = [(x, None if x == "uit" else float(x))
                for x in a.arms.split(",")]

    if a.cohort == "psgipa":
        taken = [("SN1",), ("SN2",), ("SN3",), ("SN4",), ("SN5",)]
        fn, vaste = _psgipa_een, (a.data_dir, a.profile, drempels)
    else:
        edfs = sorted((Path(a.mesa_dir) / "polysomnography" / "edfs")
                      .glob("mesa-sleep-*.edf"))[:a.n]
        taken = [(e.stem,) for e in edfs]
        fn, vaste = _mesa_een, (a.mesa_dir, a.profile, drempels)

    rijen = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(fn, *t, *vaste): t[0] for t in taken}
        for f in as_completed(futs):
            naam = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"{naam}: FOUT {e}", flush=True)
                continue
            if r is None:
                continue
            rijen.append(r)
            print(f"{naam}: mens {r['mens_ar']:>4} ar, koppeling "
                  f"{r['mm']*100:>5.1f}% | " +
                  "  ".join(
                      f"{k}={v['n']}(+{v['n_window'] or 0})"
                      f"/{v['om']*100:.0f}%/F1 "
                      f"{(v['f1'] or 0):.2f}"
                      for k, v in r["armen"].items()), flush=True)

    if not rijen:
        return
    Path(a.out).write_text(json.dumps(rijen, indent=2, default=float))
    print(f"\nn = {len(rijen)}   cohort = {a.cohort}")
    print("\n── per opname ──")
    kop = f"{'opname':<16}{'mens':>6}" + "".join(
        f"{n:>16}" for n, _w in drempels)
    print(kop); print("-" * len(kop))
    for r in sorted(rijen, key=lambda x: x["rec"]):
        cells = "".join(
            f"{r['armen'][n]['n']:>5}/{(r['armen'][n]['f1'] or 0):.2f}"
            f"/{r['armen'][n]['om']*100:>3.0f}%"
            for n, _w in drempels)
        print(f"{r['rec']:<16}{r['mens_ar']:>6}{cells}")
    print("   (telling / F1 / koppeling)")

    print(f"\n{'arm':>6}{'koppeling ons x mens':>22}{'F1 vs mens':>14}"
          f"{'telling / referentie':>22}{'via venster':>12}")
    print("-" * 76)
    num_mm = sum(r["mm"] * r["armen"][drempels[0][0]]["n_mens_ev"] for r in rijen)
    den_mm = sum(r["armen"][drempels[0][0]]["n_mens_ev"] for r in rijen)
    for naam, _w in drempels:
        num = sum(r["armen"][naam]["om"] * r["armen"][naam]["n_mens_ev"]
                  for r in rijen)
        den = sum(r["armen"][naam]["n_mens_ev"] for r in rijen)
        ratio = (sum(r["armen"][naam]["n"] for r in rijen)
                 / sum(r["mens_ar"] for r in rijen))
        vlag = "" if ratio <= 1.10 else "   << boven de grens 1,10"
        w = sum((r["armen"][naam]["n_window"] or 0) for r in rijen)
        f1s = [r["armen"][naam]["f1"] for r in rijen
               if r["armen"][naam]["f1"] is not None]
        f1m = median(f1s) if f1s else float("nan")
        print(f"{naam:>6}{num/den*100:>21.1f}%{f1m:>14.3f}{ratio:>22.2f}"
              f"{w:>12}{vlag}")
    print(f"{'mens':>6}{num_mm/den_mm*100:>21.1f}%{'—':>14}{1.00:>22.2f}")
    print("\nVooraf: de koppeling moet stijgen EN de telling onder 1,10 blijven.")


if __name__ == "__main__":
    main()
