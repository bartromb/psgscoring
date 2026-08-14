#!/usr/bin/env python3
"""Blok 3.1 — apneu-SUBTYPERING tegen de NSRR-referentie. Geen codewijziging.

psgscoring valideert AHI, event-F1, percentiel en severity. De subtypering komt
in geen enkele tabel voor, terwijl er klinisch meer van afhangt dan van de AHI
zelf: obstructief versus centraal bepaalt CPAP versus ASV. De beslissing valt
in een regelcascade die nooit tegen een referentie is gehouden.

TWEE GESCHEIDEN VRAGEN, TWEE TABELLEN. Vermengen meet geen van beide:

  1. DETECTIE      vindt het systeem de apneu?  F1, precisie, recall,
                   typeloos gematcht (IoU >= 0,20, de bestaande matcher).
  2. CLASSIFICATIE mits gevonden, welk type?  Verwarringsmatrix en kappa,
                   UITSLUITEND over gematchte paren.

`uncertain` krijgt een EIGEN KOLOM en telt niet als fout. Een apneu waarvan het
type niet bepaald kon worden is een conventie van psgscoring die de referentie
niet kent; die als misclassificatie tellen schrijft een conventieverschil toe
aan het algoritme.

Gestratificeerd op Cheyne-Stokes: fix 3 herclassificeert obstructief naar
centraal op CSR-nachten, dus verwacht dat verschillen daar geconcentreerd
zitten en niet gelijkmatig verdeeld. Zonder stratificatie lijkt dat ruis.

    cd psgscoring
    PYTHONPATH=$PWD PSGSCORING_AROUSAL_DERIVATION=single \\
        python scripts/subtype_agreement_mesa.py --n 60 --out subtype.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REF_TYPES = ("obstructive", "central", "mixed")
ALGO_TYPES = ("obstructive", "central", "mixed", "uncertain")


def _mesa():
    spec = importlib.util.spec_from_file_location(
        "vm", REPO / "scripts" / "validate_mesa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["vm"] = m
    spec.loader.exec_module(m)
    return m


def _vp():
    spec = importlib.util.spec_from_file_location("vp", REPO / "validate_psgipa.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["vp"] = m
    spec.loader.exec_module(m)
    return m


def _ref_apneas(xml_path, dur_s):
    """Apneus uit de NSRR-annotatie, met hun type. Alleen in slaap begonnen.

    Hypopneeen blijven buiten beschouwing: dit gaat over apneu-subtypering.
    """
    import xml.etree.ElementTree as ET
    import math
    vm = _mesa()
    root = ET.parse(str(xml_path)).getroot()
    n_ep = int(math.ceil(dur_s / vm.EPOCH_S))
    hypno = ["W"] * n_ep
    ruw = []
    for ev in root.iter("ScoredEvent"):
        c = (ev.findtext("EventConcept") or "").split("|")[0].strip().lower()
        try:
            st = float(ev.findtext("Start") or "nan")
            du = float(ev.findtext("Duration") or "nan")
        except ValueError:
            continue
        if st < 0 or st >= dur_s:
            continue
        stage = vm.STAGE_MAP.get(c)
        if stage is not None:
            e0 = int(st // vm.EPOCH_S)
            for i in range(max(1, int(round(du / vm.EPOCH_S)))):
                if 0 <= e0 + i < n_ep:
                    hypno[e0 + i] = stage
        elif c in vm.APNEA_MAP:
            ruw.append((st, st + du, vm.APNEA_MAP[c]))

    def asleep(t):
        e = int(t // vm.EPOCH_S)
        return 0 <= e < n_ep and hypno[e] in ("N1", "N2", "N3", "R")

    return [(a, b, t) for a, b, t in ruw if asleep(a)], hypno


def een(args):
    rec_id, data_dir, profiel = args
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import mne
    import psgscoring
    mne.set_log_level("ERROR")
    vp = _vp()
    data_dir = Path(data_dir)
    edf = data_dir / "polysomnography" / "edfs" / f"{rec_id}.edf"
    xml = (data_dir / "polysomnography" / "annotations-events-nsrr"
           / f"{rec_id}-nsrr.xml")
    if not (edf.exists() and xml.exists()):
        return {"recording": rec_id, "error": "bestand ontbreekt"}
    try:
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
        dur = float(raw.times[-1])
        ref, hypno = _ref_apneas(xml, dur)
        if not ref:
            return {"recording": rec_id, "error": "geen referentie-apneus"}
        r = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                           scoring_profile=profiel)
        resp = r.get("respiratory") or {}
        algo = [(float(e["onset_s"]),
                 float(e["onset_s"]) + float(e.get("duration_s") or 0.0),
                 str(e.get("type")))
                for e in (resp.get("events") or [])
                if e.get("onset_s") is not None
                and str(e.get("type")) in ALGO_TYPES]

        # 1. DETECTIE — typeloos, precies de matcher van paper v31.
        m = vp.match_events(algo, ref, iou_thresh=0.20,
                            type_aware=False, optimal=False)

        # 2. CLASSIFICATIE — alleen over gematchte paren.
        paren = []
        for pa in m.get("matched_pairs", []):
            try:
                ia, ib = pa[0], pa[1]
                paren.append((algo[ia][2], ref[ib][2]))
            except Exception:
                continue

        csr = bool(((r.get("cheyne_stokes") or {}).get("csr_detected")))
        # Zonder werkende effortbanden is effort-gebaseerde typering onmogelijk
        # en wordt ALLES `uncertain`. Dat meet de dataset, niet de classificatie,
        # dus het moet een stratificatie-as zijn en geen stilzwijgend nulresultaat.
        sq = r.get("signal_quality") or {}
        rip_mode = sq.get("recommended_mode") if isinstance(sq, dict) else None
        # De ECG-effort-tak herclassificeert events naar centraal op grond van
        # de hartslagmodulatie. Waar hij vuurt verwacht je de verschillen
        # geconcentreerd, dus dat is een stratificatie-as (briefing 3.1).
        n_ecg = int(resp.get("n_ecg_reclassified_central") or 0)
        return {
            "recording": rec_id, "csr": csr,
            "rip_mode": rip_mode,
            "rip_ok": rip_mode not in (None, "unreliable"),
            "n_ecg_reclassified": n_ecg,
            "ecg_branch": n_ecg > 0,
            "n_ref": len(ref), "n_algo": len(algo),
            "detect": {k: m.get(k) for k in
                       ("tp", "fp", "fn", "precision", "recall", "f1")},
            "pairs": paren,
            "ref_mix": dict(Counter(t for _a, _b, t in ref)),
            "algo_mix": dict(Counter(t for _a, _b, t in algo)),
        }
    except Exception as e:                                        # noqa: BLE001
        return {"recording": rec_id, "error": f"{type(e).__name__}: {e}"[:140]}


def _kappa(paren, klassen):
    """Cohen kappa over de gematchte paren, klassen expliciet meegegeven."""
    n = len(paren)
    if n == 0:
        return float("nan")
    obs = sum(1 for a, b in paren if a == b) / n
    ca = Counter(a for a, _ in paren)
    cb = Counter(b for _, b in paren)
    exp = sum((ca[k] / n) * (cb[k] / n) for k in klassen)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def rapport(rows, titel):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        print(f"  {titel}: geen bruikbare opnames")
        return
    import statistics as st
    print(f"\n{'=' * 72}\n  {titel}   (n = {len(ok)} opnames)\n{'=' * 72}")

    # ── 1. DETECTIE ─────────────────────────────────────────────
    f1 = [r["detect"]["f1"] for r in ok if r["detect"]["f1"] is not None]
    pr = [r["detect"]["precision"] for r in ok if r["detect"]["precision"] is not None]
    rc = [r["detect"]["recall"] for r in ok if r["detect"]["recall"] is not None]
    print("\n  TABEL 1 — DETECTIE (typeloos, IoU >= 0,20)")
    print(f"    F1        mediaan {st.median(f1):.3f}   gemiddeld {st.mean(f1):.3f}")
    print(f"    precisie  mediaan {st.median(pr):.3f}")
    print(f"    recall    mediaan {st.median(rc):.3f}")
    print(f"    referentie-apneus {sum(r['n_ref'] for r in ok)}   "
          f"algoritme-apneus {sum(r['n_algo'] for r in ok)}")

    # ── 2. CLASSIFICATIE ────────────────────────────────────────
    paren = [p for r in ok for p in r["pairs"]]
    print(f"\n  TABEL 2 — CLASSIFICATIE (alleen gematchte events, n = {len(paren)})")
    if not paren:
        print("    geen gematchte paren")
        return
    print(f"    {'ref \\\\ algo':>14s}" + "".join(f"{a:>13s}" for a in ALGO_TYPES)
          + f"{'totaal':>9s}")
    for rt in REF_TYPES:
        rij = [sum(1 for a, b in paren if b == rt and a == at) for at in ALGO_TYPES]
        tot = sum(rij)
        print(f"    {rt:>14s}" + "".join(f"{x:13d}" for x in rij) + f"{tot:9d}")
    kol = [sum(1 for a, _b in paren if a == at) for at in ALGO_TYPES]
    print(f"    {'totaal':>14s}" + "".join(f"{x:13d}" for x in kol)
          + f"{len(paren):9d}")

    # kappa: uncertain telt NIET als fout, dus buiten de kappa houden.
    zonder = [(a, b) for a, b in paren if a != "uncertain"]
    acc = sum(1 for a, b in zonder if a == b) / len(zonder) if zonder else float("nan")
    n_unc = len(paren) - len(zonder)
    print(f"\n    uncertain: {n_unc} van {len(paren)} "
          f"({100 * n_unc / len(paren):.1f} %) — eigen kolom, geen fout")
    if n_unc:
        verd = Counter(b for a, b in paren if a == "uncertain")
        print("      komen uit tegen referentie: "
              + ", ".join(f"{k} {v}" for k, v in sorted(verd.items())))
    print(f"    accuraatheid (uncertain uitgesloten): {acc:.3f}  "
          f"op n = {len(zonder)}")
    print(f"    Cohen kappa                        : "
          f"{_kappa(zonder, REF_TYPES):.3f}")
    for rt in REF_TYPES:
        sub = [(a, b) for a, b in zonder if b == rt]
        if sub:
            print(f"      recall {rt:12s}: "
                  f"{sum(1 for a, b in sub if a == b) / len(sub):.3f}  (n={len(sub)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/home/bart/MESA/mesa"))
    ap.add_argument("--profile", default="aasm_v3_rec")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    edfs = a.data_dir / "polysomnography" / "edfs"
    xmls = a.data_dir / "polysomnography" / "annotations-events-nsrr"
    ids = sorted(p.stem for p in edfs.glob("*.edf")
                 if (xmls / f"{p.stem}-nsrr.xml").exists())
    pick = sorted(random.Random(a.seed).sample(ids, min(a.n, len(ids))))
    print(f"  {len(pick)} van {len(ids)} opnames, profiel {a.profile}", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(een, (r, str(a.data_dir), a.profile)) for r in pick]
        for f in as_completed(futs):
            rows.append(f.result())
            if len(rows) % 10 == 0:
                print(f"    {len(rows)}/{len(pick)}", flush=True)

    rapport(rows, "ALLE OPNAMES")
    ok = [r for r in rows if "error" not in r]
    # EERST stratificeren op bruikbare effortbanden: zonder die is de vraag
    # naar subtypering niet gesteld, laat staan beantwoord.
    rip_ok = [r for r in ok if r.get("rip_ok")]
    rip_bad = [r for r in ok if not r.get("rip_ok")]
    from collections import Counter as _C
    print(f"\n  RIP-kwaliteit: {_C(r.get('rip_mode') for r in ok)}")
    print(f"  bruikbare effortbanden: {len(rip_ok)} van {len(ok)}")
    if rip_ok:
        rapport(rip_ok, "EFFORTBANDEN BRUIKBAAR — hier is de vraag te beantwoorden")
    if rip_bad:
        rapport(rip_bad, "EFFORTBANDEN ONBRUIKBAAR — meet de dataset, niet de classificatie")
    ecg = [r for r in rip_ok if r.get("ecg_branch")]
    if ecg:
        rapport(ecg, f"ECG-EFFORT-TAK VUURT  ({len(ecg)} van {len(rip_ok)})")
    geen_ecg = [r for r in rip_ok if not r.get("ecg_branch")]
    if geen_ecg and ecg:
        rapport(geen_ecg, "ECG-EFFORT-TAK VUURT NIET")

    csr = [r for r in rip_ok if r.get("csr")]
    nocsr = [r for r in rip_ok if not r.get("csr")]
    print(f"\n  binnen de bruikbare set: {len(csr)} met CSR, {len(nocsr)} zonder")
    if csr:
        rapport(csr, "BRUIKBAAR + Cheyne-Stokes")
    if nocsr:
        rapport(nocsr, "BRUIKBAAR, zonder Cheyne-Stokes")

    fouten = [r for r in rows if "error" in r]
    if fouten:
        print(f"\n  {len(fouten)} opnames overgeslagen, eerste: {fouten[0]['error']}")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"\n  geschreven: {a.out}")


if __name__ == "__main__":
    main()
