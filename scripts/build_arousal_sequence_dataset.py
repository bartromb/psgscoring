#!/usr/bin/env python3
"""Golfvormvensters + AROUSALmaskers voor een sequentiemodel.

WAAROM DIT BESTAAT
==================
Gemeten op 18 MESA-opnames (2760 menselijke arousals): **62,7 % wordt nooit als
kandidaat voorgesteld**. Niet te weinig kandidaten -- we genereren er 908 per
opname tegen 154 menselijke -- maar de VERKEERDE. Het spectrale criterium vindt
honderden plekken die de scoorder niet ziet en mist twee derde van wat hij wel
ziet.

Geen classifier en geen werkpunt haalt die 62,7 % terug: ze staan niet op tafel.
Alleen de 10,7 % die de classifier verwerpt is met een drempel bereikbaar, en
die koop je met slechtere localisatie (F1 0,443 -> 0,370 van werkpunt 0,70 naar
0,30).

Een model dat de golfvorm leest heeft geen kandidatenstap en dus geen
recall-plafond. Dat is de enige reparatierichting die de meting overlaat.

64 Hz IN PLAATS VAN 8
=====================
Arousals zijn gedefinieerd op een frequentieVERSCHUIVING naar alfa, theta en
beta (tot 30 Hz volgens de AASM). Bij 8 Hz is de Nyquist 4 Hz en is dat
per constructie onzichtbaar. 64 Hz is het minimum dat de betaband draagt.


WAAROM DIT NAAST DE TABELLARISCHE SET BESTAAT
=============================================
`build_training_dataset.py` levert 32 features per KANDIDAAT, en die kandidaten
komen uit de regelgebaseerde keten. Daarmee erft elk model dat erop traint het
recall-plafond van die keten. Gemeten op PSG-IPA: de regels wezen 28 kandidaten
af die een MEERDERHEID van de scoorders wél markeerde -- 7 % van alles wat een
meerderheid zag, en geen classifier haalt dat terug.

Dit script slaat die stap over. Het levert de RUWE golfvorm en een masker van
waar de events liggen, zodat een model zelf mag bepalen wat een event is. De
AASM-regels verhuizen daarmee van de ingang naar de uitgang: wat het model
voorstelt, wordt achteraf aan de manual getoetst.

KEUZES, EN WAAROM
=================
**8 Hz.** Respiratoire events duren >= 10 s en ademhaling zit onder 1 Hz. Bij
8 Hz is de Nyquist 4 Hz, ruim boven alles wat telt, en een nacht van 10 uur
wordt 288 000 samples in plaats van 9 miljoen bij 256 Hz. Dat scheelt twee
ordes in geheugen en rekentijd, en er gaat geen informatie verloren die de
AASM-regels gebruiken.

**Vier kanalen.** Flow (neusdruk), thorax, abdomen, SpO2. Precies de vier
waarop de regels beslissen -- niet meer, zodat een vergelijking eerlijk is.

**Per opname genormaliseerd.** Robuuste z-score op de mediaan en de MAD, want
EDF-eenheden verschillen per montage en een netwerk dat de eenheid leert, leert
het apparaat in plaats van de patiënt. SpO2 blijft in procenten: dat IS een
absolute schaal, en 88 % betekent overal hetzelfde.

**Masker uit dezelfde referentie als alle andere metingen.** `aasm15` --
3 % desaturatie OF arousal, alle apneus, dus AASM v3 Rule 1A. Zie
`scripts/validate_mesa.py` voor de reconstructie.

Gebruik (in de DL-venv, want torch en psgscoring naast elkaar):
    ~/CODE/.venv-dl/bin/python scripts/build_sequence_dataset.py \\
        --data-dir ~/MESA/mesa --limit 150 --output seq_mesa.npz
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore")

FS = 64.0           #: doelbemonstering (Hz) -- beta tot 30 Hz vraagt dit
KANALEN = ("eeg", "emg", "eog")


def _resample(x, sf_in, n_uit):
    """Lineair naar de doellengte. Geen anti-aliasfilter nodig: de
    respiratoire kanalen zijn al bandbeperkt onder 3 Hz door de acquisitie,
    en SpO2 verandert in seconden, niet in milliseconden."""
    if x is None or len(x) < 2:
        return np.zeros(n_uit, dtype=np.float32)
    oud = np.linspace(0.0, 1.0, len(x))
    nieuw = np.linspace(0.0, 1.0, n_uit)
    return np.interp(nieuw, oud, x).astype(np.float32)


def _robuust(x):
    """Z-score op mediaan en MAD. Een netwerk dat de EDF-eenheid leert, leert
    het apparaat; deze normalisatie haalt dat weg zonder de VORM aan te tasten."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) / 0.6745
    if mad < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - med) / mad, -10.0, 10.0).astype(np.float32)


def _een_opname(rec_id: str, data_dir: Path):
    import mne
    mne.set_log_level("ERROR")
    sys.path.insert(0, str(REPO / "scripts"))
    from validate_mesa import parse_nsrr

    from psgscoring.constants import CHANNEL_PATTERNS

    edf = data_dir / "polysomnography" / "edfs" / f"{rec_id}.edf"
    xml = (data_dir / "polysomnography" / "annotations-events-nsrr"
           / f"{rec_id}-nsrr.xml")
    if not (edf.exists() and xml.exists()):
        return None

    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
    dur = float(raw.times[-1])
    n = int(dur * FS)
    if n < int(3600 * FS):          # korter dan een uur: geen nacht
        return None

    # Kanalen zoeken met dezelfde patronen die de keten gebruikt, zodat het
    # model op dezelfde signalen kijkt als de regels.
    def zoek(rol):
        pats = CHANNEL_PATTERNS.get(rol, [])
        for ch in raw.ch_names:
            k = ch.lower()
            if any(p in k for p in pats):
                return ch
        return None

    namen = {"eeg": zoek("eeg"), "emg": zoek("emg_chin") or zoek("emg"),
             "eog": zoek("eog")}
    if namen["eeg"] is None:
        return None   # zonder EEG is een arousal per AASM niet scoorbaar

    kanalen = []
    for rol in KANALEN:
        nm = namen.get(rol)
        if nm is None:
            kanalen.append(np.zeros(n, dtype=np.float32))
            continue
        sig = raw.get_data(picks=[nm])[0]
        r = _resample(sig, raw.info["sfreq"], n)
        kanalen.append(_robuust(r))
    X = np.stack(kanalen)                       # (4, n)

    import xml.etree.ElementTree as ET
    hypno, _refs, _tst = parse_nsrr(xml, dur)
    masker = np.zeros(n, dtype=np.float32)
    for ev in ET.parse(str(xml)).getroot().iter("ScoredEvent"):
        if "arousal" not in (ev.findtext("EventConcept") or "").lower():
            continue
        try:
            s0 = float(ev.findtext("Start"))
            d0 = float(ev.findtext("Duration") or 3.0)
        except (TypeError, ValueError):
            continue
        a, b = int(s0 * FS), int(min(s0 + max(d0, 3.0), dur) * FS)
        if 0 <= a < b <= n:
            masker[a:b] = 1.0
    # Slaapmasker: buiten de slaap wordt niet gescoord, en een model dat daar
    # traint leert wake-artefacten als "geen event" -- ruis in beide richtingen.
    slaap = np.zeros(n, dtype=np.float32)
    for i, st in enumerate(hypno):
        if st != "W":
            a, b = int(i * 30 * FS), int((i + 1) * 30 * FS)
            slaap[a:min(b, n)] = 1.0
    return X, masker, slaap, rec_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    xml_dir = args.data_dir / "polysomnography" / "annotations-events-nsrr"
    ids = [p.stem.replace("-nsrr", "") for p in sorted(xml_dir.glob("*.xml"))]
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} opnames -> {FS:.0f} Hz, {len(KANALEN)} kanalen")

    tmp_dir = args.output.parent / (args.output.stem + "_delen")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    Xs, Ms, Ss, Rs = [], [], [], []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_een_opname, i, args.data_dir): i for i in ids}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:                            # noqa: BLE001
                print(f"  {futs[f]}: MISLUKT — {e}", flush=True)
                continue
            if r is None:
                print(f"  {futs[f]}: overgeslagen (kanalen of duur)", flush=True)
                continue
            X, m, s, rid = r
            Xs.append(X); Ms.append(m); Ss.append(s); Rs.append(rid)
            # Tussentijds wegschrijven. De vorige versie bewaarde alles in
            # geheugen en schreef pas aan het eind; toen de opslag faalde was
            # 25 minuten rekenwerk weg.
            np.save(tmp_dir / f"{rid}.npy",
                    np.array([X, m, s], dtype=object), allow_pickle=True)
            print(f"  {rid}: {X.shape[1]/FS/3600:.1f} u, "
                  f"{100*m.mean():.1f} % in event", flush=True)

    if not Xs:
        print("niets opgeleverd")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ONGELIJKE LENGTES. `np.array(lijst, dtype=object)` broadcast wanneer alle
    # elementen dezelfde EERSTE dimensie hebben -- hier (4, T) met wisselende
    # T -- en faalt met "could not broadcast input array from shape (4,259191)
    # into shape (4,)". Dat gebeurde NA het verwerken van alle 150 opnames, dus
    # het kostte de hele run. Een lege object-array vullen omzeilt het.
    def _obj(lijst):
        a = np.empty(len(lijst), dtype=object)
        for i, v in enumerate(lijst):
            a[i] = v
        return a

    np.savez_compressed(
        args.output,
        X=_obj(Xs), mask=_obj(Ms), sleep=_obj(Ss), rec=np.array(Rs),
        fs=FS, kanalen=np.array(KANALEN))
    uren = sum(x.shape[1] for x in Xs) / FS / 3600
    frac = float(np.mean([m.mean() for m in Ms]))
    print(f"\n{len(Xs)} opnames, {uren:.0f} uur, {100*frac:.1f} % van de tijd "
          f"in een event -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
