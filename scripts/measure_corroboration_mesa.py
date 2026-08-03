#!/usr/bin/env python3
"""Hoeveel apneus vallen in elk corroboratievakje? Gemeten op MESA.

MESA is de enige beschikbare set met twee bruikbare flowsensoren
(``Pres`` + ``Therm``) EN een referentie. PSG-IPA heeft maar één flowkanaal en
kan deze vraag niet beantwoorden.

Per opname draait de detectie twee keer — apneu op de thermistor, apneu op de
neusdruk — en worden de twee eventlijsten gekruist:

  ``both``             beide sensoren zien het -> apneu
  ``thermistor_only``  alleen de thermistor    -> verdacht artefact
  ``pressure_only``    alleen de neusdruk      -> vermoedelijk mondademhaling

De signaalkwaliteitspoort wordt hier bewust omzeild: het doel is het VOLLEDIGE
beeld, inclusief opnames waar de poort de thermistor zou afkeuren. De gemeten
envelope-overeenstemming staat per opname in de uitvoer, zodat zichtbaar wordt
of die samenhangt met welk vakje domineert — en dus of de drempel van 0,40
ergens op slaat.

Gebruik:
    python scripts/measure_corroboration_mesa.py [--n 25] [--seed 20260803]
"""
from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
mne.set_log_level("ERROR")

D = Path("/home/bart/MESA/mesa/polysomnography")


def run(args):
    rid, sensor = args
    import psgscoring
    import psgscoring.signal_quality as SQ

    SQ.THERMISTOR_AGREEMENT_MIN = -1.0        # poort uit: volledig beeld
    from psgscoring.utils import detect_channels

    from validate_mesa import parse_nsrr

    raw = mne.io.read_raw_edf(str(D / "edfs" / f"{rid}.edf"),
                              preload=True, verbose=False)
    hyp, refs, tst = parse_nsrr(
        D / "annotations-events-nsrr" / f"{rid}-nsrr.xml", float(raw.times[-1]))
    cm = {k: v for k, v in detect_channels(raw.ch_names).items() if v}
    if sensor == "pressure":
        cm.pop("flow_thermistor", None)
    # Een rol WEGLATEN volstaat niet: channel_map_from_user() begint bij
    # auto-detectie en legt de gebruikerskaart eroverheen, dus de sleutel komt
    # terug. De kaart moet daarom hard afgedwongen worden, anders draaien
    # beide armen op hetzelfde kanaal en meet je nul verschil.
    import psgscoring.pipeline as PL
    PL.channel_map_from_user = lambda _user, _names: dict(cm)
    o = psgscoring.run_pneumo_analysis(raw, hypno=hyp, channel_map=cm,
                                       scoring_profile="aasm_v3_rec")
    r = o["respiratory"]
    # "uncertain" is een apneu die de effort-classifier niet kon subtyperen
    # (respiratory.py:1541). Op MESA zijn de RIP-banden onbruikbaar, dus daar
    # belandt vrijwel elke apneu. Hem weglaten telt nul apneus.
    ap = [e for e in r.get("events", [])
          if str(e.get("type")) in ("obstructive", "central", "mixed",
                                    "uncertain", "apnea")]
    fc = (o.get("meta", {}) or {}).get("flow_channels", {}) or {}
    return {
        "rid": rid, "sensor": sensor, "apneas": ap,
        "ahi": float((r.get("summary") or {}).get("ahi_total") or 0.0),
        "ref": (len(refs.get("aasm15", [])) / tst) if tst else 0.0,
        "agreement": (fc.get("thermistor_check") or {}).get("agreement"),
        "used": fc.get("apnea_sensor"),
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--n", type=int, default=25)
    ap_.add_argument("--seed", type=int, default=20260803)
    ap_.add_argument("--workers", type=int, default=12)
    a = ap_.parse_args()

    from psgscoring.postprocess import corroborate_apnea_events

    ids = sorted(p.stem for p in (D / "edfs").glob("*.edf")
                 if (D / "annotations-events-nsrr" / f"{p.stem}-nsrr.xml").exists())
    picked = sorted(random.Random(a.seed).sample(ids, a.n))
    print(f"MESA corroboratiemeting — {a.n} opnames, seed {a.seed}\n", flush=True)

    jobs = [(r, s) for r in picked for s in ("thermistor", "pressure")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run, jobs))

    tot = {"n_both": 0, "n_thermistor_only": 0, "n_pressure_only": 0}
    per = []
    for rid in picked:
        g = {x["sensor"]: x for x in rows if x["rid"] == rid}
        if len(g) != 2:
            continue
        _ev, d = corroborate_apnea_events(g["thermistor"]["apneas"],
                                          g["pressure"]["apneas"])
        for k in tot:
            tot[k] += d[k]
        per.append((rid, d, g["thermistor"]["agreement"],
                    g["thermistor"]["ahi"], g["pressure"]["ahi"],
                    g["pressure"]["ref"]))

    print(f"{'opname':18s} {'overeenk':>8s} {'beide':>6s} {'th-only':>8s} "
          f"{'pr-only':>8s} | {'AHI th':>7s} {'AHI pr':>7s} {'ref':>6s}")
    for rid, d, ag, ahit, ahip, ref in per:
        print(f"{rid:18s} {str(ag):>8s} {d['n_both']:6d} "
              f"{d['n_thermistor_only']:8d} {d['n_pressure_only']:8d} | "
              f"{ahit:7.1f} {ahip:7.1f} {ref:6.1f}")

    s = sum(tot.values())
    print(f"\nTOTAAL over {len(per)} opnames — {s} apneu-detecties:")
    for k, v in tot.items():
        print(f"  {k:18s} {v:5d}  ({100 * v / s:5.1f}%)" if s
              else f"  {k:18s}     0")


if __name__ == "__main__":
    main()
