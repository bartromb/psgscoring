#!/usr/bin/env python3
"""Herijking van THERMISTOR_BAND_POWER_MIN: ademband-fractie per montage,
in BEIDE decimatiemodi.

WAAROM
------
De drempel 0,70 is afgeleid op negen Somnomedics-montages (`Flow Th.`):

    0,982  0,981  0,977  0,970   |   0,441  0,396  0,318  0,036  0,000

Die negen getallen zijn gemeten MET aliasing: `respiratory_band_power`
decimeerde tot 11-10-2026 met `x[::q]`, zonder anti-aliasfilter, zodat
alles boven 4 Hz (netbrom, ECG, pulsartefact) in de noemer van de breuk
vouwde. Met het filter aan (`thermistor_band_power_antialias`, default)
kunnen beide klassen opschuiven en ligt het midden van het gat mogelijk
ergens anders. Dit script meet dezelfde opnames in beide modi, zodat de
herijking op de server kan draaien waar die opnames staan -- hier is geen
data.

GEBRUIK
-------
    python scripts/calibrate_thermistor_band_power.py \
        --edf-dir /pad/naar/edfs --channel "Flow Th." --channel "Therm" \
        [--pattern "*.edf"] [--csv uit.csv]

Per bestand wordt het EERSTE opgegeven kanaal genomen dat in de montage
voorkomt (hoofdletterongevoelig, exacte naam voor substring). Het kanaal
wordt VOLLEDIG geladen: een gedeeltelijke lezing van een EDF met gemengde
samplefrequenties geeft andere waarden dan een volledige (handboek §9).

Zonder data: `--selftest` draait de synthetische reproductie (ademhaling
op 0,20 Hz met een 50 Hz-toon van 0,7x de amplitude) en moet 0,671 tegen
~1,000 laten zien.

UITVOER
-------
Eén regel per bestand (bestand, kanaal, sf, fractie met aliasing, fractie
met filter, verschil, piek in Hz) en een samenvatting: mediaan per modus,
de gesorteerde fracties per modus en de huidige drempel. De beslisregel
blijft dezelfde als bij de oorspronkelijke ijking -- het midden van het
grootste gat tussen de twee klassen -- en wordt hier NIET automatisch
toegepast: welke montages "goed" en "slecht" zijn is een oordeel over de
opname, niet over het getal.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psgscoring.signal_quality import (  # noqa: E402
    THERMISTOR_BAND_POWER_MIN,
    respiratory_band_power,
)


def _kies_kanaal(ch_names: list[str], gewenst: list[str]) -> str | None:
    lower = {c.lower(): c for c in ch_names}
    for w in gewenst:
        if w.lower() in lower:
            return lower[w.lower()]
    for w in gewenst:
        for lc, orig in lower.items():
            if w.lower() in lc:
                return orig
    return None


def _laad_kanaal(pad: Path, gewenst: list[str]):
    """(signaal, sf, kanaalnaam) of None als geen gewenst kanaal bestaat."""
    import mne
    mne.set_log_level("ERROR")
    kop = mne.io.read_raw_edf(pad, preload=False, verbose="ERROR")
    naam = _kies_kanaal(list(kop.ch_names), gewenst)
    if naam is None:
        return None
    # Alleen dit kanaal, maar WEL volledig geladen (zie module-docstring).
    exclude = [c for c in kop.ch_names if c != naam]
    raw = mne.io.read_raw_edf(pad, exclude=exclude, preload=True,
                              verbose="ERROR")
    x = raw.get_data(picks=[naam])[0]
    return np.asarray(x, dtype=float), float(raw.info["sfreq"]), naam


def meet(x: np.ndarray, sf: float) -> dict:
    oud = respiratory_band_power(x, sf, antialias=False)
    nieuw = respiratory_band_power(x, sf, antialias=True)
    return {
        "sf": sf,
        "fractie_alias": oud["fraction"],
        "fractie_antialias": nieuw["fraction"],
        "verschil": (None if oud["fraction"] is None or nieuw["fraction"] is None
                     else round(nieuw["fraction"] - oud["fraction"], 3)),
        "piek_hz": nieuw["peak_hz"],
        "n_vensters": nieuw["n_windows"],
    }


def _print_regel(naam: str, kanaal: str, m: dict) -> None:
    def f(v):
        return "  n/a " if v is None else f"{v:6.3f}"
    print(f"{naam:40s} {kanaal:14s} {m['sf']:6.1f} Hz  alias {f(m['fractie_alias'])}"
          f"  filter {f(m['fractie_antialias'])}  Δ {f(m['verschil'])}"
          f"  piek {f(m['piek_hz'])} Hz")


def _samenvatting(rijen: list[dict]) -> None:
    for modus in ("fractie_alias", "fractie_antialias"):
        w = sorted(r[modus] for r in rijen if r[modus] is not None)
        if not w:
            continue
        print(f"\n{modus}: n={len(w)}  mediaan {np.median(w):.3f}")
        print("  gesorteerd: " + "  ".join(f"{v:.3f}" for v in w))
        onder = [v for v in w if v < THERMISTOR_BAND_POWER_MIN]
        print(f"  onder de huidige drempel {THERMISTOR_BAND_POWER_MIN:.2f}: "
              f"{len(onder)} van {len(w)}")
    print("\nDe drempel is het midden van het grootste gat tussen de klassen "
          "'draagt ademhaling' en 'draagt geen ademhaling'. Welke montage "
          "waar hoort is een oordeel over de opname; dit script velt het "
          "niet.")


def selftest() -> int:
    """De synthetische reproductie: 0,671 met aliasing, ~1,000 met filter."""
    ok = True
    for sf in (128.0, 200.0, 250.0, 256.0, 512.0):
        t = np.arange(int(600 * sf)) / sf
        x = (np.sin(2 * np.pi * 0.20 * t)
             + 0.7 * np.sin(2 * np.pi * 50.0 * t)
             + np.random.default_rng(3).normal(scale=0.01, size=t.size))
        m = meet(x, sf)
        _print_regel("synthetisch 0,20 Hz + 50 Hz-toon", "-", m)
        ok &= abs(m["fractie_alias"] - 1 / 1.49) < 0.02
        ok &= m["fractie_antialias"] > 0.95
    print("selftest:", "OK" if ok else "MISLUKT")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edf-dir", type=Path, help="map met EDF/BDF-bestanden")
    ap.add_argument("--channel", action="append", default=[],
                    help="thermistorkanaalnaam; herhaalbaar, eerste treffer wint")
    ap.add_argument("--pattern", default="*.edf")
    ap.add_argument("--csv", type=Path, help="schrijf de rijen ook als CSV")
    ap.add_argument("--selftest", action="store_true",
                    help="synthetische reproductie zonder data")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.edf_dir or not args.channel:
        ap.error("--edf-dir en minstens één --channel zijn vereist "
                 "(of --selftest)")

    paden = sorted(args.edf_dir.glob(args.pattern))
    if not paden:
        print(f"geen bestanden op {args.edf_dir}/{args.pattern}", file=sys.stderr)
        return 2

    rijen: list[dict] = []
    for pad in paden:
        try:
            geladen = _laad_kanaal(pad, args.channel)
        except Exception as e:                                  # noqa: BLE001
            print(f"  overslaan {pad.name}: {e}", file=sys.stderr)
            continue
        if geladen is None:
            print(f"  overslaan {pad.name}: geen van {args.channel} aanwezig",
                  file=sys.stderr)
            continue
        x, sf, kanaal = geladen
        m = meet(x, sf)
        _print_regel(pad.name, kanaal, m)
        rijen.append({"bestand": pad.name, "kanaal": kanaal, **m})

    if not rijen:
        return 2
    _samenvatting(rijen)

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rijen[0]))
            w.writeheader()
            w.writerows(rijen)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
