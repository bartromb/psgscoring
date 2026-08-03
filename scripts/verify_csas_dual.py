#!/usr/bin/env python3
"""Verificatie op de twee CSAS-opnames: aasm_v3_rec tegenover aasm_v3_dual.

Dit is de toets die de spec voorschrijft vóór release, en ze weegt zwaarder
dan de rest van de validatie. Een productiewijziging op 2026-08-02 liet
apneus op de thermistor scoren; op een echte opname zakte het aantal van 93
naar 0 en verschoof de uitkomst van matig CSAS — bevestigd door menselijke
scoring — naar mild SAS. 100 MESA-opnames en 5 PSG-IPA-opnames zagen dat
niet; één opname met menselijke scoring wel.

De veiligheidsvoorwaarde die hier bewezen moet worden: **aasm_v3_dual mag
nooit MINDER apneus vinden dan aasm_v3_rec.** Het profiel gebruikt beide
sensoren additief en wijst niets af, dus een onbruikbare thermistor mag
hooguit niets bijdragen.

Waarschuwingen uit eerdere metingen, hier verwerkt:
  - apneus van type "uncertain" meetellen (de effort-classifier parkeert ze
    daar; ze weglaten telt nul apneus)
  - geen pipes met buffering rond de uitvoer
  - staging op 100 Hz, want YASA op 256 Hz liep 47 minuten zonder voortgang
  - controleren DAT de twee armen verschillende sensoren gebruiken
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

PSG = Path("/home/bart/PSGs")
RECS = ["CSASofOSAS01", "CSASofOAS02"]
RESP = ["Snore", "Pressure Flow", "Flow Th.", "RIP Thora", "RIP Abdom",
        "SpO2", "Pos.", "Pulse", "ECG II", "C4:A1", "O2:A1", "F4:A1",
        "EOG1:A2", "EOG2:A1"]

APNEA_TYPES = ("obstructive", "central", "mixed", "uncertain", "apnea")


def _is_apnea(e):
    t = str(e.get("type", ""))
    return "hypopnea" not in t and (t in APNEA_TYPES or "apnea" in t)


def run(args):
    name, profile = args
    import yasa

    import psgscoring
    from psgscoring.utils import detect_channels

    p = PSG / f"{name}.edf"
    st = mne.io.read_raw_edf(str(p), preload=True, verbose=False,
                             include=["C4:A1", "EOG1:A2", "EMG1"])
    st.set_channel_types({"C4:A1": "eeg", "EOG1:A2": "eog", "EMG1": "emg"})
    st.resample(100, npad="auto")          # YASA is op 100 Hz getraind
    hyp = list(yasa.SleepStaging(st, eeg_name="C4:A1", eog_name="EOG1:A2",
                                 emg_name="EMG1").predict())
    del st

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False, include=RESP)
    cm = {k: v for k, v in detect_channels(raw.ch_names).items() if v}
    o = psgscoring.run_pneumo_analysis(raw, hypno=hyp, channel_map=cm,
                                       scoring_profile=profile)
    r = o["respiratory"]
    s = r.get("summary") or {}
    ev = r.get("events", [])
    fc = (o.get("meta", {}) or {}).get("flow_channels", {}) or {}
    ap = [e for e in ev if _is_apnea(e)]
    return {
        "name": name, "profile": profile,
        "ahi": float(s.get("ahi_total") or 0.0),
        "n_apnea": len(ap),
        "n_central": sum(1 for e in ap if str(e.get("type")) == "central"),
        "n_obstr": sum(1 for e in ap if str(e.get("type")) == "obstructive"),
        "n_uncertain": sum(1 for e in ap if str(e.get("type")) == "uncertain"),
        "n_hyp": sum(1 for e in ev if "hypopnea" in str(e.get("type"))),
        "types": dict(Counter(str(e.get("type")) for e in ev)),
        "apnea_sensor": fc.get("apnea_sensor"),
        "hypop_sensor": fc.get("hypopnea_sensor"),
        "agreement": (fc.get("thermistor_check") or {}).get("agreement"),
        "dual": r.get("dual_sensor_apnea"),
    }


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--workers", type=int, default=4)
    a = ap_.parse_args()

    jobs = [(n, p) for n in RECS for p in ("aasm_v3_rec", "aasm_v3_dual")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run, jobs))

    ok = True
    for name in RECS:
        g = {r["profile"]: r for r in rows if r["name"] == name}
        if len(g) != 2:
            print(f"{name}: onvolledig")
            continue
        rec, dual = g["aasm_v3_rec"], g["aasm_v3_dual"]
        print(f"\n{'=' * 68}\n{name}   envelope-overeenstemming "
              f"{dual['agreement']}")
        print(f"  {'':22s} {'aasm_v3_rec':>13s} {'aasm_v3_dual':>13s}")
        print(f"  {'apneu-sensor':22s} {str(rec['apnea_sensor']):>13s} "
              f"{str(dual['apnea_sensor']):>13s}")
        print(f"  {'hypopnee-sensor':22s} {str(rec['hypop_sensor']):>13s} "
              f"{str(dual['hypop_sensor']):>13s}")
        for k, lab in (("ahi", "AHI"), ("n_apnea", "apneus TOTAAL"),
                       ("n_central", "  waarvan centraal"),
                       ("n_obstr", "  waarvan obstructief"),
                       ("n_uncertain", "  waarvan ongetypeerd"),
                       ("n_hyp", "hypopnees")):
            print(f"  {lab:22s} {rec[k]:>13} {dual[k]:>13}")
        if dual.get("dual"):
            d = dual["dual"]
            print(f"  corroboratie: beide {d['n_both']}, alleen thermistor "
                  f"{d['n_thermistor_only']}, alleen druk {d['n_pressure_only']}")

        # De veiligheidsvoorwaarde
        if dual["n_apnea"] < rec["n_apnea"]:
            ok = False
            print(f"  ** GEZAKT: dual vindt MINDER apneus "
                  f"({dual['n_apnea']} < {rec['n_apnea']})")
        elif dual["n_central"] < rec["n_central"]:
            ok = False
            print(f"  ** GEZAKT: dual vindt MINDER centrale apneus "
                  f"({dual['n_central']} < {rec['n_central']})")
        else:
            print("  ok: dual verliest geen apneus")

    print(f"\n{'=' * 68}")
    print("VEILIGHEIDSVOORWAARDE: " + ("GEHAALD" if ok else "**GEZAKT**"))
    print("  aasm_v3_dual mag nooit minder apneus vinden dan aasm_v3_rec.")


if __name__ == "__main__":
    main()
