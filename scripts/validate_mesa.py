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

De referentie moet gerecONSTRUEERD worden, niet geteld
------------------------------------------------------
In de NSRR-annotatie is elk ``Hypopnea``- en ``Unsure``-label een flowdaling
van >=30%; de desaturatie-eis zit er NIET in. NSRR past die achteraf toe door
elk event te koppelen aan de losse ``SpO2 desaturation``-events. Wie de labels
rauw telt, komt 2 tot 4 keer te hoog uit en concludeert vervolgens ten
onrechte dat het algoritme massaal onderdetecteert.

De reconstructie telt alleen events die in een slaap-epoch beginnen, en
koppelt elke hypopnee aan een SpO2-desaturatie die begint binnen
[start, einde + 30 s] — en, voor de arousal-varianten, aan een arousal binnen
[start, einde + 5 s].

Drie referentiesets, elk geijkt tegen zijn eigen gepubliceerde NSRR-kolom
(60 opnames; herhaal met ``--verify-reference``):

  ``aasm15``  PRIMAIR. Alle apneus + hypopneeen met >=3% desaturatie OF een
              arousal. Dat is letterlijk AASM v3 Rule 1A, dus de regel die
              beide profielen implementeren.
              vs ``nsrr_ahi_hp3r_aasm15``: bias +0,32, MAE 0,72, r = 0,999
  ``oahi3``   obstructieve apneus + hypopneeen met >=3% desaturatie, ZONDER
              arousal-tak. vs ``oahi35``: bias +0,11, MAE 0,84, r = 0,998
  ``oahi4``   idem met >=4%. MESA's eigen kopcijfer.
              vs ``oahi45``: bias +0,19, MAE 0,66, r = 0,998

De oahi-varianten staan erbij voor continuiteit met eerdere rondes, maar ze
crediteren geen hypopnee die alleen door een arousal kwalificeert terwijl
Rule 1A dat wel doet. Daardoor tellen zulke events als false positive en
drukken ze zowel precisie als recall van BEIDE profielen. ``aasm15`` heeft
dat probleem niet en is daarom de maat waarop geconcludeerd wordt.

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

APNEA_MAP = {"obstructive apnea": "obstructive", "mixed apnea": "mixed",
             "central apnea": "central"}
OBSTRUCTIVE_SCOPE = {"obstructive", "mixed"}
HYPOPNEA_CONCEPTS = {"hypopnea", "unsure"}

DESAT_LINK_WINDOW_S = 30.0
AROUSAL_LINK_WINDOW_S = 5.0

# Referentiesets, elk met de NSRR-kolom waartegen de reconstructie geijkt is.
# ``aasm15`` is de primaire: dat is letterlijk de regel die beide profielen
# implementeren (AASM v3 Rule 1A). De oahi-varianten staan erbij voor
# continuiteit met de eerste ronde en omdat oahi45 MESA's eigen kopcijfer is.
REFERENCES = {
    "aasm15": {"desat": 3.0, "arousal": True, "apneas": "all",
               "column": "nsrr_ahi_hp3r_aasm15", "source": "harmonized"},
    "oahi3":  {"desat": 3.0, "arousal": False, "apneas": "obstructive",
               "column": "oahi35", "source": "dataset"},
    "oahi4":  {"desat": 4.0, "arousal": False, "apneas": "obstructive",
               "column": "oahi45", "source": "dataset"},
}
PRIMARY_REFERENCE = "aasm15"

PROFILES = ("aasm_v3_rec", "aasm_v3_breath")


# ─────────────────────────────────────────────────────────────────
#  NSRR-annotatie
# ─────────────────────────────────────────────────────────────────

def parse_nsrr(xml_path, signal_duration_s,
               link_window_s=DESAT_LINK_WINDOW_S):
    """(hypno, events_per_referentie, tst_h) uit een NSRR-annotatiebestand.

    ``events`` is per referentieset een lijst (onset, offset, type), in
    hetzelfde formaat als ``validate_psgipa.event_set``. Zie de moduledocstring
    voor waarom de labels niet rauw geteld mogen worden.
    """
    root = ET.parse(str(xml_path)).getroot()
    n_epochs = int(math.ceil(signal_duration_s / EPOCH_S))
    hypno = ["W"] * n_epochs
    hypopneas, apneas, desats, arousals = [], [], [], []

    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").split("|")[0].strip().lower()
        ev_type = (ev.findtext("EventType") or "").lower()
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
        elif concept in HYPOPNEA_CONCEPTS:
            hypopneas.append((start, start + dur))
        elif concept in APNEA_MAP:
            apneas.append((start, start + dur, APNEA_MAP[concept]))
        elif concept == "spo2 desaturation":
            try:
                drop = (float(ev.findtext("SpO2Baseline"))
                        - float(ev.findtext("SpO2Nadir")))
            except (TypeError, ValueError):
                continue
            desats.append((start, drop))
        elif "arousal" in ev_type or "arousal" in concept:
            arousals.append(start)

    tst_h = sum(1 for s in hypno if s in ("N1", "N2", "N3", "R")) * EPOCH_S / 3600.0
    arousals.sort()

    def asleep(t):
        ep = int(t // EPOCH_S)
        return 0 <= ep < n_epochs and hypno[ep] in ("N1", "N2", "N3", "R")

    events = {}
    for name, spec in REFERENCES.items():
        thr = spec["desat"]
        out = []
        for a, b in hypopneas:
            if not asleep(a):
                continue
            qualifies = any(a <= s <= b + link_window_s and d >= thr
                            for s, d in desats)
            if not qualifies and spec["arousal"]:
                qualifies = any(a <= x <= b + AROUSAL_LINK_WINDOW_S
                                for x in arousals)
            if qualifies:
                out.append((a, b, "hypopnea"))
        for a, b, t in apneas:
            if not asleep(a):
                continue
            if spec["apneas"] == "obstructive" and t not in OBSTRUCTIVE_SCOPE:
                continue
            out.append((a, b, t))
        events[name] = sorted(out)
    return hypno, events, tst_h


# ─────────────────────────────────────────────────────────────────
#  Eén opname
# ─────────────────────────────────────────────────────────────────

def _configs(strictness_values, profiles=None):
    """[(label, profielnaam, strictness_of_None), ...].

    De basislijn `aasm_v3_rec` draait ALTIJD mee, zodat elke vergelijking
    gepaard is op dezelfde opname in plaats van tussen runs.

    `profiles` breidt de set uit voorbij de oorspronkelijke breath-vs-rec
    vraag. Dat is nodig voor de assen die het paper wil scheiden:

      hypopneedetector   regelcascade (`rec`)  ×  ademteug-gegradeerd (`breath`)
      arousal-as         drempel               ×  gegradeerd (`prob`)
      flowsensor         mono                  ×  duaal (`*_dual`)

    MESA is de enige cohort waar die derde as te meten valt: de montage draagt
    `Pres`, `Flow` én `Therm`. PSG-IPA heeft één flowkanaal, en daar zijn de
    duale profielen aantoonbaar identiek aan hun ouder (gemeten 9 aug 2026).
    """
    cfg = [("aasm_v3_rec", "aasm_v3_rec", None)]
    if strictness_values:
        cfg += [(f"breath@{s:.2f}", "aasm_v3_breath", float(s))
                for s in strictness_values]
        return cfg
    for p in (profiles or ["aasm_v3_breath"]):
        if p != "aasm_v3_rec":
            cfg.append((p, p, None))
    return cfg


def _apply_strictness(value):
    """Zet de strictness van aasm_v3_breath in dit werkproces.

    De legacy-dicts worden uit profiles.py afgeleid bij import, dus die moeten
    opnieuw gerenderd en doorgegeven worden aan de modules die ze vasthouden.
    """
    import importlib

    from psgscoring.profiles import PROFILES as _P
    _P["aasm_v3_breath"].post_processing.hypopnea_strictness = value
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES


def _apply_rip_scale_free(aan: bool):
    """Zet `rip_quality_scale_free` op ALLE profielen in dit werkproces.

    Bedoeld om te meten wat de gerepareerde RIP-poort met de MESA-bias doet:
    kale `uncertain` valt buiten `ahi_total`, dus een werkende poort verhoogt
    de index. Zet dit NIET aan voor een reproductie van paper v31/v37 —
    `mesa_shhs` hoort daarvoor op het absolute gedrag te blijven.
    """
    import importlib

    from psgscoring.profiles import PROFILES as _P
    for _p in _P.values():
        _p.post_processing.rip_quality_scale_free = bool(aan)
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    for _n, _d in C.SCORING_PROFILES.items():
        assert _d["RIP_QUALITY_SCALE_FREE"] is bool(aan), _n


def _apply_desat_limit(limiet):
    """Zet `max_events_per_desaturation` op ALLE profielen in dit werkproces."""
    import importlib

    from psgscoring.profiles import PROFILES as _P
    for _p in _P.values():
        _p.post_processing.max_events_per_desaturation = limiet
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    for _n, _d in C.SCORING_PROFILES.items():
        assert _d["MAX_EVENTS_PER_DESATURATION"] == limiet, _n


def _apply_thermistor_gate(gate):
    """Zet `thermistor_gate` op ALLE profielen in dit werkproces."""
    import importlib

    from psgscoring.profiles import PROFILES as _P
    for _p in _P.values():
        _p.post_processing.thermistor_gate = gate
    import psgscoring.constants as C
    importlib.reload(C)
    import psgscoring.pipeline as PL
    import psgscoring.respiratory as R
    PL.SCORING_PROFILES = C.SCORING_PROFILES
    R.SCORING_PROFILES = C.SCORING_PROFILES
    for _n, _d in C.SCORING_PROFILES.items():
        assert _d["THERMISTOR_GATE"] == gate, _n


def analyse_one(args):
    # `profiles` moet MEE door de tuple: de worker draait in een eigen proces
    # en kan de argparse-namespace niet zien. Zonder dit rekent hij de
    # standaardset terwijl report() de uitgebreide labels eist — dan valt
    # elke opname af en meldt het harnas 'geen bruikbare opnames'.
    (rec_id, data_dir, strictness_values, profiles, rip_scale_free,
     desat_limit, therm_gate) = args
    data_dir = Path(data_dir)
    if rip_scale_free:
        _apply_rip_scale_free(True)
    if desat_limit is not None:
        _apply_desat_limit(int(desat_limit))
    if therm_gate:
        _apply_thermistor_gate(str(therm_gate))
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

    for label, prof, strict in _configs(strictness_values, profiles):
        try:
            if strict is not None:
                _apply_strictness(strict)
            res = psgscoring.run_pneumo_analysis(
                raw, hypno=hypno, scoring_profile=prof)
        except Exception as e:  # noqa: BLE001
            out["profiles"][label] = {"error": str(e)}
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
        # De eventlijst zelf mee opslaan: dan kan een referentiewijziging
        # offline hermatched worden zonder de pijplijn opnieuw te draaien.
        rec["events"] = [(round(a, 2), round(b, 2), str(t)) for a, b, t in algo]
        bd = r.get("breath_detector")
        if bd:
            rec["breath_detector"] = bd
        out["profiles"][label] = rec
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


def report(rows, ref_name, labels):
    ok = [r for r in rows if "error" not in r
          and all(p in r["profiles"] and "error" not in r["profiles"][p]
                  for p in labels)]
    if not ok:
        print(f"  geen bruikbare opnames voor referentie {ref_name}")
        return

    print(f"\n{'═' * 78}")
    print(f"  REFERENTIE: {ref_name}   (n = {len(ok)} opnames)")
    print(f"{'═' * 78}")
    print(f"  {'configuratie':16s} {'F1 med':>7s} {'F1 gem':>7s} {'prec':>6s} "
          f"{'rec':>6s} {'bias':>7s} {'MAE':>6s} {'r':>6s} {'sev':>7s}")
    stats = {}
    for lab in labels:
        f1 = [r["profiles"][lab]["match"][ref_name]["f1"] for r in ok]
        pr = [r["profiles"][lab]["match"][ref_name]["precision"] for r in ok]
        rc = [r["profiles"][lab]["match"][ref_name]["recall"] for r in ok]
        algo = np.array([r["profiles"][lab]["ahi"] for r in ok])
        ref = np.array([r["ahi_ref"][ref_name] for r in ok])
        d = algo - ref
        sev = sum(1 for x, y in zip(algo, ref) if severity(x) == severity(y))
        stats[lab] = f1
        rr = float(np.corrcoef(algo, ref)[0, 1]) if len(ok) > 2 else float("nan")
        print(f"  {lab:16s} {median(f1):7.3f} {np.mean(f1):7.3f} "
              f"{median(pr):6.3f} {median(rc):6.3f} "
              f"{d.mean():+7.2f} {np.abs(d).mean():6.2f} {rr:6.3f} "
              f"{sev:3d}/{len(ok):<3d}")

    # Severity-fouten uitsplitsen: de richting is klinisch niet symmetrisch.
    print("\n  severity-fouten (referentie -> algoritme):")
    for lab in labels:
        wrong = {}
        for r in ok:
            rv = severity(r["ahi_ref"][ref_name])
            av = severity(r["profiles"][lab]["ahi"])
            if rv != av:
                wrong[(rv, av)] = wrong.get((rv, av), 0) + 1
        txt = ", ".join(f"{a}->{b}:{n}" for (a, b), n
                        in sorted(wrong.items(), key=lambda x: -x[1]))
        print(f"    {lab:16s} {txt or 'geen'}")

    base = labels[0]
    for lab in labels[1:]:
        delta = [b - a for a, b in zip(stats[base], stats[lab])]
        better = sum(1 for x in delta if x > 0)
        worse = sum(1 for x in delta if x < 0)
        p, n_eff = wilcoxon_signed_rank(delta)
        print(f"\n  GEPAARD  {lab} - {base}:")
        print(f"    mediaan dF1 {median(delta):+.4f}   "
              f"gemiddeld {np.mean(delta):+.4f}")
        print(f"    beter {better}/{len(ok)}   slechter {worse}/{len(ok)}   "
              f"gelijk {len(ok) - better - worse}")
        print(f"    Wilcoxon (n={n_eff}): "
              + ("p niet bepaald, te weinig verschillen" if p is None
                 else f"p = {p:.4g}"))


def verify_reference(data_dir, limit=60):
    """Toon dat de gereconstrueerde referentie NSRR's eigen oahi reproduceert.

    Zonder deze controle is elk cijfer in dit script onbetrouwbaar: rauwe
    labeltellingen liggen 2-4x te hoog en zouden onderdetectie suggereren die
    er niet is.
    """
    import csv

    sources = {}
    for tag, fname, idcol in (
            ("dataset", "mesa-sleep-dataset-0.8.0.csv", "mesaid"),
            ("harmonized", "mesa-sleep-harmonized-dataset-0.8.0.csv", "nsrrid")):
        path = data_dir / "datasets" / fname
        if not path.exists():
            print(f"  CSV niet gevonden: {path}")
            continue
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            sources[tag] = {row[idcol]: row for row in csv.DictReader(fh)}
    if not sources:
        return

    xml_dir = data_dir / "polysomnography" / "annotations-events-nsrr"
    got = {k: [] for k in REFERENCES}
    ref = {k: [] for k in REFERENCES}

    for xml in sorted(xml_dir.glob("*.xml"))[:limit]:
        rid = xml.stem.replace("-nsrr", "").replace("mesa-sleep-", "").lstrip("0")
        # De signaalduur is hier niet nodig: de annotatie bestrijkt de opname.
        try:
            _h, events, tst = parse_nsrr(xml, 1e9)
        except Exception:  # noqa: BLE001
            continue
        if tst < 1.0:
            continue
        for name, spec in REFERENCES.items():
            table = sources.get(spec["source"], {})
            if rid not in table:
                continue
            try:
                published = float(table[rid][spec["column"]])
            except (KeyError, ValueError):
                continue
            got[name].append(len(events[name]) / tst)
            ref[name].append(published)

    print(f"\n  Referentiereconstructie vs gepubliceerde NSRR-waarden "
          f"(desat-venster {DESAT_LINK_WINDOW_S:.0f} s, "
          f"arousal-venster {AROUSAL_LINK_WINDOW_S:.0f} s)")
    print(f"  {'set':9s} {'kolom':22s} {'n':>4s} {'bias':>8s} {'MAE':>7s} {'r':>7s}")
    for name, spec in REFERENCES.items():
        if len(got[name]) < 3:
            print(f"  {name:9s} te weinig data")
            continue
        d = np.asarray(got[name]) - np.asarray(ref[name])
        r = float(np.corrcoef(got[name], ref[name])[0, 1])
        mark = "  <- primair" if name == PRIMARY_REFERENCE else ""
        print(f"  {name:9s} {spec['column']:22s} {len(d):4d} {d.mean():+8.2f} "
              f"{np.abs(d).mean():7.2f} {r:7.3f}{mark}")


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
    ap.add_argument("--verify-reference", action="store_true",
                    help="controleer de referentie tegen gepubliceerde oahi "
                         "en stop daarna")
    ap.add_argument("--thermistor-gate", default=None,
                    choices=["envelope_agreement", "respiratory_band",
                             "breath_coherence"],
                    help="forceer deze poort op alle profielen; weglaten = "
                         "profieldefault")
    ap.add_argument("--desat-limit", type=int, default=None,
                    help="max_events_per_desaturation op alle profielen; "
                         "weglaten = None = huidig gedrag")
    ap.add_argument("--rip-scale-free", action="store_true",
                    help="meet met de gerepareerde RIP-poort; NIET voor "
                         "een reproductie van paper v31/v37")
    ap.add_argument("--profiles", nargs="+", default=None,
                    help="profielen naast de vaste basislijn aasm_v3_rec; "
                         "default aasm_v3_breath")
    ap.add_argument("--strictness", nargs="+", type=float, default=None,
                    help="draai aasm_v3_breath op deze strictness-waarden, "
                         "alle op dezelfde opnames zodat de vergelijking "
                         "gepaard is")
    ap.add_argument("--exclude-seed", type=int, default=None,
                    help="sluit de steekproef van dit zaad uit, zodat een "
                         "tweede ronde disjunct is van de eerste")
    ap.add_argument("--exclude-n", type=int, default=None,
                    help="omvang van de uit te sluiten eerdere steekproef")
    a = ap.parse_args()

    if a.verify_reference:
        verify_reference(a.data_dir)
        return

    edf_dir = a.data_dir / "polysomnography" / "edfs"
    xml_dir = a.data_dir / "polysomnography" / "annotations-events-nsrr"
    ids = sorted(p.stem for p in edf_dir.glob("*.edf")
                 if (xml_dir / f"{p.stem}-nsrr.xml").exists())

    excluded = set()
    if a.exclude_seed is not None and a.exclude_n:
        excluded = set(random.Random(a.exclude_seed).sample(ids, a.exclude_n))

    pool = [x for x in ids if x not in excluded]
    if a.recordings:
        picked = list(a.recordings)
    elif a.n and a.n < len(pool):
        picked = sorted(random.Random(a.seed).sample(pool, a.n))
    else:
        picked = pool

    labels = [c[0] for c in _configs(a.strictness, a.profiles)]
    print(f"MESA-validatie — {len(picked)} van {len(pool)} beschikbare opnames "
          f"(seed {a.seed})")
    if excluded:
        print(f"uitgesloten: {len(excluded)} opnames uit de eerdere steekproef "
              f"(seed {a.exclude_seed}); overlap = "
              f"{len(set(picked) & excluded)}")
    print(f"configuraties: {', '.join(labels)}\n")

    rows = []
    jobs = [(x, str(a.data_dir), a.strictness, a.profiles, a.rip_scale_free,
             a.desat_limit, a.thermistor_gate) for x in picked]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(analyse_one, jobs), 1):
            rows.append(r)
            if "error" in r:
                print(f"  [{i}/{len(picked)}] {r['recording']}: FOUT {r['error']}")
            else:
                parts = []
                for lab in labels:
                    v = (r["profiles"].get(lab, {}).get("match", {})
                         .get(PRIMARY_REFERENCE, {}).get("f1"))
                    parts.append(f"{lab} {'--' if v is None else format(v, '.3f')}")
                print(f"  [{i}/{len(picked)}] {r['recording']}  "
                      + "  ".join(parts))

    for ref in REFERENCES:
        report(rows, ref, labels)

    n_err = sum(1 for r in rows if "error" in r)
    if n_err:
        print(f"\n  {n_err} opnames overgeslagen wegens fouten")

    if a.output_json:
        # Provenance. De run van 2026-08-09 legde de arousal-modus NIET vast,
        # waardoor achteraf niet te bewijzen viel of hij op de profiel-default
        # (`multi`) of op `single` draaide. Zonder deze velden is een
        # herhaling niet met de vorige te vergelijken.
        import os as _os

        import psgscoring as _ps
        import psgscoring.constants as _C
        _prof = _C.SCORING_PROFILES
        meta = {
            "psgscoring_version": _ps.__version__,
            "arousal_derivation_env": _os.environ.get(
                "PSGSCORING_AROUSAL_DERIVATION") or "(niet gezet — profieldefault)",
            "arousal_derivation_per_profile": {
                n: _prof[n].get("AROUSAL_DERIVATION_MODE")
                for n in labels if n in _prof},
            "rip_quality_scale_free": {
                n: _prof[n].get("RIP_QUALITY_SCALE_FREE")
                for n in labels if n in _prof},
            "stability_filter_all_hypopnea_subtypes": {
                n: _prof[n].get("STABILITY_FILTER_ALL_HYPOPNEA_SUBTYPES")
                for n in labels if n in _prof},
            "rip_scale_free_flag": bool(a.rip_scale_free),
            "desat_limit": a.desat_limit,
            "thermistor_gate_forced": a.thermistor_gate,
            "thermistor_gate_per_profile": {
                n: _prof[n].get("THERMISTOR_GATE") for n in labels if n in _prof},
        }
        a.output_json.write_text(json.dumps(
            {"seed": a.seed, "n_requested": len(picked), "configs": labels,
             "exclude_seed": a.exclude_seed, "exclude_n": a.exclude_n,
             "meta": meta, "results": rows}, indent=2))
        print(f"\n  JSON weggeschreven: {a.output_json}")
        print(f"  provenance: psgscoring {meta['psgscoring_version']}, "
              f"arousal-env {meta['arousal_derivation_env']}")


if __name__ == "__main__":
    main()
