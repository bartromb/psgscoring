#!/usr/bin/env python3
"""
diag_arousal_triage.py — waar sneuvelen de arousals: in de filter of in de
kandidaatgeneratie?

WAAROM DIT SCRIPT BESTAAT
-------------------------
Een arousal-index die te laag is tegenover de eventfrequentie heeft twee
mogelijke oorzaken, en ze vragen tegengestelde reparaties:

* **filter-limiet** — de kandidaten bestaan, de LGBM verwerpt ze. Dan helpt
  herkalibreren van het werkpunt of het opschonen van de EMG-features.
* **recall-limiet** — de kandidaatgeneratie biedt ze nooit aan. Dan is
  herkalibreren kalibreren op het verkeerde ding: de filter kan niet
  terugvinden wat er niet is.

Drie armen op dezelfde opname scheiden die twee. `pre_lgbm_n_arousals` (sinds
v0.27.1 ook door de multi-derivatie heen doorgegeven) geeft de kandidatenpool,
en de respiratoire arousalfractie zegt of de gevonden arousals op de plek
liggen waar de prior het hoogst is.

Draait de VOLLE pijplijn per arm -- de fractie n_resp_arousals/n_resp_events
vraagt de respiratoire scoring en de koppeling, en die zijn niet los te
benaderen zonder een tweede waarheid te maken.

Gebruik:
    python scripts/diag_arousal_triage.py REC.edf --ctx ctx.json
    python scripts/diag_arousal_triage.py REC.edf --hypno hypno.json \
        --profile aasm_v3_breath
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
mne.set_log_level("ERROR")


def _arm(raw, hypno, profile, channel_map, art_epochs, *, lgbm, threshold):
    """Eén arm. Env-overrides, niet de registry muteren."""
    import psgscoring

    saved = {k: os.environ.get(k) for k in
             ("PSGSCORING_AROUSAL_LGBM", "PSGSCORING_AROUSAL_LGBM_THRESHOLD")}
    os.environ["PSGSCORING_AROUSAL_LGBM"] = "1" if lgbm else "0"
    if threshold is not None:
        os.environ["PSGSCORING_AROUSAL_LGBM_THRESHOLD"] = str(threshold)
    else:
        os.environ.pop("PSGSCORING_AROUSAL_LGBM_THRESHOLD", None)
    try:
        out = psgscoring.run_pneumo_analysis(
            raw.copy(), hypno=hypno, channel_map=channel_map,
            artifact_epochs=art_epochs, scoring_profile=profile)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    ar = out.get("arousal") or {}
    s = ar.get("summary") or {}
    nested = (ar.get("arousals") or {}).get("summary") or {}
    resp = (out.get("respiratory") or {}).get("summary") or {}
    n_ev = resp.get("n_ah_total")
    n_resp_ar = s.get("n_respiratory_arousals")
    return {
        "pre_lgbm": (ar.get("arousals") or {}).get("pre_lgbm_n_arousals")
                    or nested.get("lgbm_n_pre"),
        "n_post":   nested.get("lgbm_n_post"),
        "n_arousals": len(ar.get("events") or []),
        "ai":       s.get("arousal_index"),
        "n_resp_ar": n_resp_ar,
        "n_resp_ev": n_ev,
        "frac":     (round(n_resp_ar / n_ev, 3)
                     if n_resp_ar is not None and n_ev else None),
        "ahi":      resp.get("ahi_total"),
        "rdi":      s.get("rdi"),
        "lgbm_ok":  nested.get("lgbm_available"),
        "reden":    nested.get("lgbm_skipped_reason"),
        "eeg":      (out["meta"].get("channels_used") or {}).get("eeg"),
        "derivs":   nested.get("derivations"),
        "per_deriv": nested.get("n_per_derivation"),
        "emg":      out["meta"].get("arousal_emg_channel"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edf")
    ap.add_argument("--ctx", help="JSON met hypno/pneumo_channels/eeg_ch/emg_ch/art_epochs")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--sweep", default="0.50,0.60,0.70,0.80")
    ap.add_argument(
        "--extra", default="",
        help="Kanalen die YASAFlaskified óók in de pneumo-raw zet maar die "
             "niet in de jobconfig staan (komma-gescheiden). Nodig om de "
             "klinische run exact na te bouwen: _detect_pneumo_channels haalt "
             "meer EEG binnen dan de config noemt, en de arousalstap kiest "
             "zijn afleiding uit wat er ÍS.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ctx = json.loads(Path(a.ctx).read_text()) if a.ctx else {}
    hypno = ctx.get("hypno") or []
    if not hypno:
        raise SystemExit("geen hypnogram: de armen zijn dan niet vergelijkbaar "
                         "met de klinische run")
    profile = a.profile or ctx.get("profile") or "aasm_v3_breath"
    cmap = dict(ctx.get("pneumo_channels") or {})
    if ctx.get("emg_ch"):
        cmap["emg"] = ctx["emg_ch"]
    art = ctx.get("art_epochs") or []

    # Dezelfde kanalenset als YASAFlaskified samenstelt: pneumo + EEG + kin-EMG.
    hdr = mne.io.read_raw_edf(a.edf, preload=False, verbose=False)
    nodig = [c for c in list(cmap.values()) + [ctx.get("eeg_ch")]
             + [x.strip() for x in a.extra.split(",")] if c]
    # De arousalstap kiest zelf een EEG uit wat er is; alle EEG-afleidingen
    # meenemen zou een ander kanaal kunnen opleveren dan de klinische run.
    nodig = [c for c in dict.fromkeys(nodig) if c in hdr.ch_names]
    raw = mne.io.read_raw_edf(
        a.edf, exclude=[c for c in hdr.ch_names if c not in set(nodig)],
        preload=True, verbose=False)
    print(f"kanalen: {raw.ch_names}", flush=True)
    print(f"hypnogram: {len(hypno)} epochs, {len(art)} artefact-epochs, "
          f"profiel {profile}\n", flush=True)

    armen = [("1. LGBM UIT (regelgebaseerd)", False, None),
             ("2. LGBM AAN, cutoff 0.80", True, 0.80)]
    for t in [float(x) for x in a.sweep.split(",")]:
        if abs(t - 0.80) > 1e-9:
            armen.append((f"3. LGBM AAN, cutoff {t:.2f}", True, t))

    rijen = {}
    kop = (f"{'arm':<30}{'kand':>6}{'na':>6}{'AI':>7}{'resp-ar':>9}"
           f"{'/events':>9}{'fractie':>9}{'RDI':>7}")
    print(kop); print("-" * len(kop), flush=True)
    for naam, lgbm, thr in armen:
        r = _arm(raw, hypno, profile, cmap, art, lgbm=lgbm, threshold=thr)
        rijen[naam] = r
        print(f"{naam:<30}{r['pre_lgbm'] or '—'!s:>6}"
              f"{r['n_post'] or '—'!s:>6}{r['ai']!s:>7}"
              f"{r['n_resp_ar']!s:>9}{r['n_resp_ev']!s:>9}"
              f"{r['frac']!s:>9}{r['rdi']!s:>7}", flush=True)

    eerste = next(iter(rijen.values()))
    print(f"\nEEG: {eerste['eeg']}   kin-EMG: {eerste['emg']}   "
          f"AHI: {eerste['ahi']}", flush=True)
    for naam, r in rijen.items():
        if r.get("derivs"):
            print(f"   {naam:<30} afleidingen {r['derivs']} "
                  f"{r['per_deriv']}", flush=True)

    ai_uit = rijen[armen[0][0]]["ai"]
    print("\n── beslisregel (vooraf vastgelegd) ──")
    if ai_uit is None:
        print("arm 1 gaf geen index — niets te beslissen")
    elif ai_uit >= 35:
        print(f"arm 1 = {ai_uit}/u ≥ 35  →  FILTER-limiet dominant  →  D2 en D5 eerst")
    elif ai_uit <= 25:
        print(f"arm 1 = {ai_uit}/u ≤ 25  →  RECALL-limiet dominant  →  D3 en D4 eerst")
    else:
        print(f"arm 1 = {ai_uit}/u tussen 25 en 35  →  beide sporen, "
              f"volgorde D3/D4 → D2 → D5")

    if a.out:
        Path(a.out).write_text(json.dumps(rijen, indent=2))


if __name__ == "__main__":
    main()
