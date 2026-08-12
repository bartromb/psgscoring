#!/usr/bin/env python
"""
inventory_flow_channels.py — welke flowsensoren draagt een cohort werkelijk?

Read-only. Scoort niets, schrijft niets in de dataset, laadt geen signalen
verder dan nodig voor de kanaalinventaris en de poortmeting.

Twee vragen die een AHI-vergelijking tussen cohorten ongeldig kunnen maken
zonder dat er iets aan de scoring mankeert:

1. DRAAGT DE OPNAME EEN TWEEDE FLOWSENSOR, EN ZIET psgscoring HEM?

   De MESA/NSRR-documentatie vermeldt een apparatuurwissel: de oude
   configuratie draagt een kanaal ``Therm``, de nieuwe een kanaal
   ``Aux_AC``. Geen enkel patroon in ``flow_thermistor`` matcht ``Aux_AC``,
   dus op die opnames blijft de rol leeg en reduceert elk duaal profiel
   stilzwijgend tot zijn één-sensor-ouder. Een gerapporteerde
   thermistor-passage van 45% is dan geen sensoreigenschap maar een
   kanaalnaam. Dit script telt beide groepen apart.

2. WERD DE HYPOPNEE-ENVELOPPE WORTEL-GELINEARISEERD?

   ``_setup_hypop_channel`` past AASM Regel 3 alleen toe wanneer het
   hypopnee-kanaal een ANDER kanaal is dan het apneu-kanaal. Op een montage
   met één flowkanaal wordt de reeds berekende ``flow_env`` hergebruikt en
   is er geen linearisatie. Zonder linearisatie meet een echte flowreductie
   van 50% als 75% amplitudereductie, dus reducties worden overschat en meer
   kandidaten halen het 30%-criterium. Een operating point dat op een
   één-kanaals cohort is gekalibreerd, is op een twee-kanaals cohort te
   streng — precies de richting van een negatieve AHI-bias.

   Het script leidt die tak per opname af uit de kanaalinventaris, zonder te
   scoren, zodat je vóór elke sweep weet welke opnames onder welke conventie
   vallen.

Gebruik
-------
    PYTHONPATH=$PWD python scripts/inventory_flow_channels.py \
        --data-dir /path/to/mesa \
        --pattern '*.edf' --limit 200 \
        --out flow_inventory.csv

    # met de NSRR-hulpkanaalnaam meegenomen, ter vergelijking
    PSGSCORING_NSRR_AUX_AC=1 PYTHONPATH=$PWD python \
        scripts/inventory_flow_channels.py --data-dir /path/to/mesa \
        --out flow_inventory_auxac.csv

Het verschil tussen die twee runs IS het antwoord op vraag 1.

Met ``--measure-gate`` worden de flowkanalen geladen en draaien beide
thermistorpoorten (envelope-agreement en respiratory-band) op elke opname
die twee kanalen draagt. Dat is trager en leest signaaldata; zonder die vlag
worden alleen headers gelezen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Kanaalnamen die op een tweede flowsensor kunnen wijzen maar door geen enkel
# patroon worden opgeëist. Puur voor rapportage: dit script kent nooit een rol
# toe, het meldt alleen wat er onopgeëist blijft liggen.
_UNCLAIMED_HINTS = ("aux", "therm", "flow", "cannula", "nasal", "pres", "oro")


def _load_raw_header(path: Path):
    import mne
    return mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")


def _row_for(path: Path, measure_gate: bool) -> dict:
    from psgscoring.utils import detect_channels

    row: dict = {
        "recording": path.stem,
        "error": "",
        "flow_pressure": "",
        "flow_thermistor": "",
        "flow_generic": "",
        "n_channels": 0,
        "unclaimed_flow_like": "",
        "dual_sensor": False,
        "hypopnea_linearised": "",
        "gate_envelope_agreement": "",
        "gate_envelope_usable": "",
        "gate_band_power": "",
        "gate_band_usable": "",
    }
    try:
        raw = _load_raw_header(path)
    except Exception as e:  # noqa: BLE001 — één kapot bestand stopt de run niet
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    names = list(raw.ch_names)
    row["n_channels"] = len(names)
    ch = detect_channels(names)
    fp = ch.get("flow_pressure")
    ft = ch.get("flow_thermistor")
    row["flow_pressure"] = fp or ""
    row["flow_thermistor"] = ft or ""
    row["flow_generic"] = ch.get("flow") or ""

    claimed = {v for v in ch.values() if v}
    row["unclaimed_flow_like"] = "|".join(
        n for n in names
        if n not in claimed and any(h in n.lower() for h in _UNCLAIMED_HINTS)
    )

    # Dezelfde afleiding als _resolve_flow_channels + _setup_hypop_channel:
    # apneu krijgt de thermistor waar die er is, hypopnee de druk. Vallen ze
    # op hetzelfde kanaal samen, dan wordt flow_env hergebruikt en is er geen
    # wortellinearisatie. Let op: dit is de kanaalafleiding VOOR de poort. Een
    # thermistor die door de poort wordt afgewezen valt onder een vervangend
    # profiel alsnog terug op één kanaal — daarom staan de poortmetingen
    # ernaast in plaats van erin verwerkt.
    apnea_ch = ft or fp or row["flow_generic"] or None
    hypop_ch = fp or ft or row["flow_generic"] or None
    row["dual_sensor"] = bool(fp and ft)
    if apnea_ch and hypop_ch:
        row["hypopnea_linearised"] = apnea_ch != hypop_ch

    if measure_gate and fp and ft:
        try:
            import numpy as np  # noqa: F401  (via mne getters)
            from psgscoring.signal_quality import (
                assess_flow_sensor_agreement, assess_thermistor_band_power)
            raw.load_data(verbose="ERROR")
            sf = float(raw.info["sfreq"])
            d_fp = raw.get_data(picks=[fp])[0]
            d_ft = raw.get_data(picks=[ft])[0]
            env = assess_flow_sensor_agreement(d_fp, sf, d_ft, sf)
            band = assess_thermistor_band_power(d_ft, sf)
            row["gate_envelope_agreement"] = env.get("agreement")
            row["gate_envelope_usable"] = env.get("usable")
            row["gate_band_power"] = band.get("agreement", band.get("fraction"))
            row["gate_band_usable"] = band.get("usable")
        except Exception as e:  # noqa: BLE001
            row["error"] = f"gate: {type(e).__name__}: {e}"
    return row


def _summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    thermistor = Counter(
        "named" if r["flow_thermistor"] else
        ("unclaimed_candidate" if r["unclaimed_flow_like"] else "absent")
        for r in ok
    )
    lin = Counter(str(r["hypopnea_linearised"]) for r in ok)
    return {
        "n_files": len(rows),
        "n_readable": len(ok),
        "n_errors": len(rows) - len(ok),
        "thermistor_role": dict(thermistor),
        "dual_sensor": sum(1 for r in ok if r["dual_sensor"]),
        "hypopnea_linearised": dict(lin),
        "aux_ac_pattern_enabled": os.environ.get(
            "PSGSCORING_NSRR_AUX_AC", "").strip().lower() in (
                "1", "true", "yes", "on"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--pattern", default="*.edf")
    ap.add_argument("--limit", type=int, default=0, help="0 = alle bestanden")
    ap.add_argument("--out", type=Path, default=Path("flow_inventory.csv"))
    ap.add_argument("--measure-gate", action="store_true",
                    help="laad signalen en meet beide thermistorpoorten")
    args = ap.parse_args(argv)

    files = sorted(args.data_dir.rglob(args.pattern))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"geen bestanden gevonden onder {args.data_dir}", file=sys.stderr)
        return 1

    rows = []
    for i, f in enumerate(files, 1):
        rows.append(_row_for(f, args.measure_gate))
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", file=sys.stderr)

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = _summarise(rows)
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
