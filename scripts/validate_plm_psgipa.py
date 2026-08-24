#!/usr/bin/env python3
"""
validate_plm_psgipa.py — PLM-detectie tegen twaalf menselijke scoorders.

WAAROM DIT HARNAS BESTAAT
-------------------------
De MESA-meting van 24-08 liet zien dat onze PLM-detectie fors overdetecteert
(mediane index 3,6x boven de NSRR-annotatie, en 72-74 PLM's op twee nachten
waar de scoorder er NUL telde). MESA is daarvoor een zwakke referentie: het
annoteert alleen het LINKERbeen, en er is één scoorder.

PSG-IPA is de sterke referentie. Vijf opnames, **twaalf scoorders per opname**,
beide benen apart geannoteerd (`EMG LAT` / `EMG RAT`), met de gescoorde
bewegingen gekoppeld aan het kanaal waarop ze gezien zijn.

DE OPZET DIE DE VRAAG BEANTWOORDT
---------------------------------
Een PLM-telling hangt aan twee dingen: welke BEWEGINGEN je vindt, en welke
daarvan je tot een SERIE rekent. Die twee scheiden, anders weet je van een
verschil niet waar het vandaan komt.

Daarom gaan de menselijke bewegingen door **exact dezelfde** keten als de onze:
bilaterale samenvoeging, slaapfilter, respiratoire uitsluiting, seriedetectie.
Het enige verschil tussen de twee armen is dan de LM-lijst zelf.

De scoordersspreiding is geen ruis maar de maat waaraan een algoritme zich
hoort te meten: ligt onze telling binnen de spreiding van de twaalf, dan is ze
niet slechter dan een willekeurige menselijke scoorder.

Gebruik:
    python scripts/validate_plm_psgipa.py --data-dir ~/PSG-IPA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")

from psgscoring.plm import (
    _detect_series,
    _exclude_resp_associated,
    _merge_bilateral,
)

RECORDINGS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
EPOCH_S = 30.0
SLEEP = {"N1", "N2", "N3", "R"}


class StageParseError(RuntimeError):
    """Een stadiumtekst die de parser niet kent.

    HARD, niet stil. De eerste versie van deze parser testte op het CIJFER
    ("1", "2", "3") terwijl PSG-IPA "Sleep stage N1" schrijft. Ze gaf dus None
    voor alle NREM, het hypnogram bleef W behalve in REM, en 568 van de 846
    epochs vielen als wake weg -- in BEIDE armen, ook de onze. Er kwam een
    volledige, plausibel ogende tabel uit waarin de mens 21 bewegingen telde
    op een nacht met er 700.

    Een parser die niets herkent hoort te stoppen, niet door te tellen.
    """


def _stage(desc: str) -> str | None:
    """W/N1/N2/N3/R uit een annotatietekst, of None als het geen stadium is."""
    d = desc.strip().lower()
    if "sleep stage" not in d:
        return None
    tail = d.split("sleep stage")[-1].strip()
    if not tail:
        raise StageParseError(f"lege stadiumtekst: {desc!r}")
    if tail[0] == "w":
        return "W"
    if tail[0] == "r":
        return "R"
    cijfers = [c for c in tail if c.isdigit()]
    if not cijfers:
        raise StageParseError(f"onbekend stadium: {desc!r}")
    n = cijfers[0]
    if n == "1":
        return "N1"
    if n == "2":
        return "N2"
    if n in ("3", "4"):
        return "N3"
    raise StageParseError(f"onbekend stadium: {desc!r}")


def parse_scorer(edf_path: Path, dur_s: float):
    """(hypno, lms_left, lms_right) van één scoorder."""
    ann = mne.read_annotations(str(edf_path))
    n_ep = int(np.ceil(dur_s / EPOCH_S))
    hypno = ["W"] * n_ep
    lms_l, lms_r = [], []
    for onset, d, desc in zip(ann.onset, ann.duration, ann.description):
        onset, d, desc = float(onset), float(d), str(desc)
        if not (0 <= onset < dur_s):
            continue
        st = _stage(desc)
        if st is not None:
            ep0 = int(onset // EPOCH_S)
            for i in range(max(1, round(d / EPOCH_S))):
                if 0 <= ep0 + i < n_ep:
                    hypno[ep0 + i] = st
            continue
        if "limb movement" not in desc.lower():
            continue
        lm = {"onset_s": onset, "duration_s": d or 0.5, "amplitude_uv": 0.0}
        # De koppeling staat in de annotatiebeschrijving noch in de EDF+-tekst;
        # ze zit in het .txt-bestand ernaast. Zonder zijde: alles links, en de
        # bilaterale samenvoeging heeft dan niets te doen -- dat zou de
        # menselijke telling kunstmatig verhogen. Vandaar _sides_from_txt().
        lms_l.append(lm)
    return hypno, lms_l, lms_r


def _sides_from_txt(txt_path: Path, dur_s: float):
    """Bewegingen mét zijde, uit het tekstbestand naast de EDF.

    Kolom 6 is het gekoppelde kanaal (`EMG LAT` / `EMG RAT`). Zonder die
    kolom zou elke beweging als één been geteld worden en zou de bilaterale
    samenvoeging niets doen -- precies de stap die op SN3 556 bewegingen per
    been tot 97 terugbrengt.
    """
    lms_l, lms_r = [], []
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        next(f, None)
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6 or "limb movement" not in parts[4].lower():
                continue
            try:
                onset, d = float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if not (0 <= onset < dur_s):
                continue
            lm = {"onset_s": onset, "duration_s": d or 0.5, "amplitude_uv": 0.0}
            (lms_r if "rat" in parts[5].lower() else lms_l).append(lm)
    return lms_l, lms_r


def through_the_chain(lms_l, lms_r, hypno, resp_ends):
    """De menselijke bewegingen door EXACT dezelfde keten als de onze."""
    merged = _merge_bilateral(lms_l, lms_r)
    merged.sort(key=lambda x: x["onset_s"])
    sleep_lms = []
    for lm in merged:
        ep = int(lm["onset_s"] // EPOCH_S)
        st = hypno[ep] if ep < len(hypno) else "W"
        if st in SLEEP:
            sleep_lms.append(lm)
    eligible, _n_resp = _exclude_resp_associated(sleep_lms, resp_ends)
    _series, n_plm = _detect_series(eligible)
    tst_h = sum(1 for s in hypno if s in SLEEP) * EPOCH_S / 3600.0
    # Tweede grendel, voor het geval een toekomstige montage een stadium
    # draagt dat wél parseert maar verkeerd. Een scoorder die minder dan een
    # half uur slaap ziet over een hele nacht heeft niet gescoord, of wij
    # lezen hem verkeerd -- allebei een fout, geen resultaat.
    if tst_h < 0.5:
        raise StageParseError(
            f"slechts {tst_h:.2f} u slaap uit dit hypnogram "
            f"({sum(1 for s in hypno if s == 'W')} van {len(hypno)} epochs W) "
            f"-- de stadiumparser leest dit bestand niet")
    return {
        "n_lm_sleep": len(sleep_lms),
        "n_eligible": len(eligible),
        "n_plm": n_plm,
        "tst_h": round(tst_h, 2),
        "plmi": round(n_plm / tst_h, 2) if tst_h > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path.home() / "PSG-IPA"))
    ap.add_argument("--profile", default="aasm_v3_breath")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--single-leg", choices=["lat", "rat"], default=None,
        help="Alleen dit been gebruiken, in BEIDE armen, zonder bilaterale "
             "samenvoeging. Bootst de MESA-opzet na (één kanaal, annotatie "
             "van één been) op een cohort met een betrouwbare referentie, om "
             "te scheiden of een cohortverschil aan de montage ligt of aan de "
             "referentie.")
    a = ap.parse_args()
    root = Path(a.data_dir)
    import psgscoring

    alles = []
    for sn in RECORDINGS:
        psg = root / "Limb_movements" / "PSG" / f"{sn}_LimbMovements.edf"
        if not psg.exists():
            print(f"{sn}: geen PSG", flush=True)
            continue
        hdr = mne.io.read_raw_edf(psg, preload=False, verbose=False)
        dur = hdr.n_times / hdr.info["sfreq"]

        scorers = sorted((root / "Limb_movements" / "Annotations" / "manual")
                         .glob(f"{sn}_LimbMovements_manual_scorer*.edf"))
        if not scorers:
            print(f"{sn}: geen scoorders", flush=True)
            continue

        hypnos, per_scorer = [], []
        for sc in scorers:
            hypno, _l, _r = parse_scorer(sc, dur)
            lms_l, lms_r = _sides_from_txt(sc.with_suffix(".txt"), dur)
            hypnos.append(hypno)
            per_scorer.append((hypno, lms_l, lms_r))

        # Consensushypnogram: modus per epoch over de scoorders.
        n_ep = min(len(h) for h in hypnos)
        hypno_cons = []
        for i in range(n_ep):
            stemmen = [h[i] for h in hypnos]
            hypno_cons.append(max(set(stemmen), key=stemmen.count))

        raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
        if a.single_leg:
            weg = "EMG RAT" if a.single_leg == "lat" else "EMG LAT"
            if weg in raw.ch_names:
                raw.drop_channels([weg])
        out = psgscoring.run_pneumo_analysis(
            raw, hypno=hypno_cons, scoring_profile=a.profile)

        # ONZE respiratoire events, ook aan de menselijke arm gegeven. Onze
        # keten sluit bewegingen rond een event-einde uit; de menselijke arm
        # met een lege lijst laten draaien zou onze telling kunstmatig
        # verlagen -- de vergelijking moet op één regelset staan.
        resp_ends = [float(e["onset_s"]) + float(e["duration_s"])
                     for e in (out.get("respiratory", {}) or {}).get("events", [])
                     if e.get("onset_s") is not None
                     and e.get("duration_s") is not None]
        if a.single_leg == "lat":
            per_scorer = [(h, l, []) for h, l, _r in per_scorer]
        elif a.single_leg == "rat":
            per_scorer = [(h, r, []) for h, _l, r in per_scorer]
        mens = [through_the_chain(l, r, h, resp_ends) for h, l, r in per_scorer]
        s = (out.get("plm") or {}).get("summary") or {}
        ons = {"n_lm_sleep": s.get("n_lm_sleep"),
               "n_eligible": s.get("n_plm_eligible"),
               "n_plm": s.get("n_plm"),
               "plmi": s.get("plm_index"),
               "kanalen": out["meta"].get("plm_channels")}

        lm_h = [m["n_lm_sleep"] for m in mens]
        plm_h = [m["n_plm"] for m in mens]
        plmi_h = [m["plmi"] for m in mens if m["plmi"] is not None]
        rij = {"rec": sn, "n_scorers": len(mens), "ons": ons,
               "mens_lm": lm_h, "mens_plm": plm_h, "mens_plmi": plmi_h}
        alles.append(rij)
        print(
            f"{sn}: scoorders n={len(mens)} | "
            f"LM mens {median(lm_h):.0f} [{min(lm_h)}-{max(lm_h)}] "
            f"vs ons {ons['n_lm_sleep']} | "
            f"PLM mens {median(plm_h):.0f} [{min(plm_h)}-{max(plm_h)}] "
            f"vs ons {ons['n_plm']} | "
            f"PLMI mens {median(plmi_h):.1f} vs ons {ons['plmi']}",
            flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(alles, indent=2))
    if alles:
        print("\n── samenvatting ──")
        r_lm = [r["ons"]["n_lm_sleep"] / median(r["mens_lm"])
                for r in alles if median(r["mens_lm"]) > 0
                and r["ons"]["n_lm_sleep"] is not None]
        r_plm = [r["ons"]["n_plm"] / median(r["mens_plm"])
                 for r in alles if median(r["mens_plm"]) > 0
                 and r["ons"]["n_plm"] is not None]
        if r_lm:
            print(f"LM-ratio ons/mens : mediaan {median(r_lm):.2f} "
                  f"({', '.join(f'{x:.2f}' for x in r_lm)})")
        if r_plm:
            print(f"PLM-ratio ons/mens: mediaan {median(r_plm):.2f} "
                  f"({', '.join(f'{x:.2f}' for x in r_plm)})")
        binnen = sum(1 for r in alles
                     if r["ons"]["n_plm"] is not None and r["mens_plm"]
                     and min(r["mens_plm"]) <= r["ons"]["n_plm"] <= max(r["mens_plm"]))
        print(f"binnen de scoordersspreiding: {binnen}/{len(alles)}")


if __name__ == "__main__":
    main()
