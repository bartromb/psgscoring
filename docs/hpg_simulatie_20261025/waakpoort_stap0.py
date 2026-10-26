#!/usr/bin/env python
"""waakpoort_stap0.py — boekhouding vóór de waakpoort-preregistratie (26-10-2026).

Vraag: van de 23 604 events die de polygrafieroute (`hpg`-arm) in W-epochs scoort, hoeveel liggen op
een dood of ongeldig signaal (sensor los: flow vlak én/óf SpO2 buiten bereik) en hoeveel op geldig
signaal (wakker liggen)? En wat zou een analysevenster op signaalgeldigheid alleen al wegnemen?

Leest de EDF's rechtstreeks (MNE, één kanaal per keer zodat de eigen bemonstering blijft staan) en de NSRR-stadia uit de XML;
importeert psgscoring NIET (de bibliotheek mag intussen bewerkt worden). Post hoc op de meetset
(n150 = afstelset): de getallen hier zijn een ontwerpgrond, geen validatie.

Per 30 s-epoch en kanaal: IQR van het signaal (p75−p25). Referentie per kanaal = mediane IQR over de
NSRR-slaapepochs. "dood" = IQR < 2 % van de referentie (of exact 0). flow_dood = Pres dood én Therm
dood (ontbreekt een kanaal, dan telt het andere). SpO2 ongeldig = mediaan < 50 of > 100, of < 50 %
van de samples in [50, 100]. sensor_los = flow_dood of SpO2 ongeldig.
Regels (post hoc): A = venster van de eerste tot de laatste run van ≥ 4 epochs (2 min) zonder
sensor_los; B = A plus runs ≥ 6 epochs (3 min) sensor_los binnen het venster eruit; C = orakel
(NSRR-slaap). Per regel: events over, uren, index, Δ t.o.v. psg-AHI en hpg_hypno-AHI, ernst.
"""
from __future__ import annotations
import argparse, json, math, statistics as st, sys, xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HIER = Path(__file__).resolve().parent
MESA = Path("/srv/DATA/MESA/mesa/polysomnography")
EPOCH_S = 30.0
SLAAP = ("N1", "N2", "N3", "R")
DOOD_FRAC = 0.02
MIN_GELDIG_RUN = 4    # 2 min
MIN_DOOD_RUN = 6      # 3 min


def stadia(xml_path: Path, dur_s: float) -> list[str]:
    root = ET.parse(str(xml_path)).getroot()
    n = int(math.ceil(dur_s / EPOCH_S)); h = ["W"] * n
    m = {"0": "W", "1": "N1", "2": "N2", "3": "N3", "4": "N3", "5": "R"}
    for ev in root.iter("ScoredEvent"):
        if (ev.findtext("EventType") or "").startswith("Stages"):
            code = (ev.findtext("EventConcept") or "").split("|")[-1].strip()
            s = float(ev.findtext("Start") or 0); d = float(ev.findtext("Duration") or 0)
            a, b = int(round(s / EPOCH_S)), int(round((s + d) / EPOCH_S))
            for i in range(max(a, 0), min(b, n)):
                h[i] = m.get(code, "W")
    return h


def epoch_iqr(x: np.ndarray, fs: float, n_ep: int) -> np.ndarray:
    k = int(round(fs * EPOCH_S)); out = np.full(n_ep, np.nan)
    for i in range(n_ep):
        seg = x[i * k:(i + 1) * k]
        if seg.size >= k // 2:
            q = np.nanpercentile(seg, [25, 75]); out[i] = q[1] - q[0]
    return out


def runs(mask: np.ndarray):
    """[(start, einde_excl)] van aaneengesloten True."""
    r, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            r.append((i, j)); i = j
        else:
            i += 1
    return r


def severity(ahi):
    return "Normal" if ahi < 5 else "Mild" if ahi < 15 else "Moderate" if ahi < 30 else "Severe"


def nacht(row: dict) -> dict:
    rec = row["recording"]; dur_s = row["duration_h"] * 3600.0
    n_ep = int(math.ceil(dur_s / EPOCH_S))
    hyp = stadia(MESA / "annotations-events-nsrr" / f"{rec}-nsrr.xml", dur_s)
    slaap = np.array([s in SLAAP for s in hyp])
    tst_h = float(slaap.sum() * EPOCH_S / 3600.0)
    import mne
    pad = str(MESA / "edfs" / f"{rec}.edf")
    labels = list(mne.io.read_raw_edf(pad, preload=False, verbose="error").ch_names)
    def sig(naam):
        # één kanaal per keer lezen: zo blijft de eigen bemonstering staan (MNE brengt anders alles naar de hoogste)
        if naam not in labels: return None
        raw = mne.io.read_raw_edf(pad, include=[naam], preload=True, verbose="error")
        return raw.get_data()[0].astype(np.float64), float(raw.info["sfreq"])
    iqr, ref, dood = {}, {}, {}
    for ch in ("Pres", "Therm", "Flow", "Thor", "Abdo"):
        v = sig(ch)
        if v is None: continue
        q = epoch_iqr(v[0], v[1], n_ep); iqr[ch] = q
        r = float(np.nanmedian(q[slaap])) if slaap.any() else float("nan"); ref[ch] = r
        dood[ch] = (q <= DOOD_FRAC * r) | (q == 0) | np.isnan(q) if (r > 0 and not math.isnan(r)) else np.ones(n_ep, bool)
    flow_chs = [c for c in ("Pres", "Therm") if c in dood and ref[c] > 0]
    flow_dood = np.logical_and.reduce([dood[c] for c in flow_chs]) if flow_chs else np.ones(n_ep, bool)
    eff_chs = [c for c in ("Thor", "Abdo") if c in dood and ref[c] > 0]
    eff_dood = np.logical_and.reduce([dood[c] for c in eff_chs]) if eff_chs else np.ones(n_ep, bool)
    v = sig("SpO2")
    spo2_ong = np.ones(n_ep, bool)
    if v is not None:
        x, fs = v; k = int(round(fs * EPOCH_S))
        for i in range(n_ep):
            seg = x[i * k:(i + 1) * k]
            if seg.size:
                ok = (seg >= 50) & (seg <= 100); med = float(np.median(seg))
                spo2_ong[i] = not (50 <= med <= 100 and ok.mean() >= 0.5)
    los = flow_dood | spo2_ong
    # regel A: venster eerste..laatste geldige run van ≥ MIN_GELDIG_RUN
    geldig_runs = [r for r in runs(~los) if r[1] - r[0] >= MIN_GELDIG_RUN]
    if geldig_runs:
        a0, a1 = geldig_runs[0][0], geldig_runs[-1][1]
    else:
        a0, a1 = 0, n_ep
    binnen_A = np.zeros(n_ep, bool); binnen_A[a0:a1] = True
    # regel B: A minus dode runs ≥ MIN_DOOD_RUN binnen het venster
    binnen_B = binnen_A.copy()
    for r in runs(los & binnen_A):
        if r[1] - r[0] >= MIN_DOOD_RUN: binnen_B[r[0]:r[1]] = False
    ev = row["arms"]["hpg"]["events"]
    tel = dict(W=0, W_flow_dood=0, W_spo2_ong=0, W_los=0, W_geldig=0, W_voor_A=0, W_na_A=0, W_in_A=0, W_in_B=0,
               S=0, S_los=0, S_in_A=0, S_in_B=0, W_eff_dood=0)
    eerste_s = int(np.argmax(slaap)) if slaap.any() else None
    laatste_s = int(n_ep - 1 - np.argmax(slaap[::-1])) if slaap.any() else None
    tel.update(W_staart_los=0, W_staart_geldig=0, W_staart=0)
    for a, b, typ in ev:
        i = min(int(a // EPOCH_S), n_ep - 1)
        if slaap[i]:
            tel["S"] += 1; tel["S_los"] += int(los[i]); tel["S_in_A"] += int(binnen_A[i]); tel["S_in_B"] += int(binnen_B[i])
        else:
            tel["W"] += 1; tel["W_flow_dood"] += int(flow_dood[i]); tel["W_spo2_ong"] += int(spo2_ong[i])
            tel["W_los"] += int(los[i]); tel["W_geldig"] += int(not los[i]); tel["W_eff_dood"] += int(eff_dood[i])
            tel["W_voor_A"] += int(i < a0); tel["W_na_A"] += int(i >= a1); tel["W_in_A"] += int(binnen_A[i]); tel["W_in_B"] += int(binnen_B[i])
            if laatste_s is not None and i > laatste_s:
                tel["W_staart"] += 1; tel["W_staart_los"] += int(los[i]); tel["W_staart_geldig"] += int(not los[i])
    uren = dict(A=float(binnen_A.sum() * EPOCH_S / 3600), B=float(binnen_B.sum() * EPOCH_S / 3600), C=tst_h, TIB=row["tib_h"])
    n_in = dict(A=tel["W_in_A"] + tel["S_in_A"], B=tel["W_in_B"] + tel["S_in_B"], C=tel["S"], TIB=len(ev))
    idx = {k: (n_in[k] / uren[k] if uren[k] > 0 else None) for k in uren}
    return {"recording": rec, "n_ep": n_ep, "tst_h_xml": tst_h, "tst_h_rij": row["tst_h"], "kanalen": {c: (c in iqr) for c in ("Pres", "Therm", "Flow", "Thor", "Abdo")},
            "ref_iqr": ref, "ep_flow_dood": int(flow_dood.sum()), "ep_spo2_ong": int(spo2_ong.sum()), "ep_los": int(los.sum()),
            "ep_los_in_slaap": int((los & slaap).sum()), "ep_W": int((~slaap).sum()), "ep_W_los": int((los & ~slaap).sum()),
            "venster_A": [a0, a1], "ep_voor_A": a0, "ep_na_A": n_ep - a1, "ep_B_weg_binnen_A": int((binnen_A & ~binnen_B).sum()),
            "slaap_buiten_A": int((slaap & ~binnen_A).sum()), "slaap_buiten_B": int((slaap & ~binnen_B).sum()),
            "tel": tel, "uren": uren, "n_in": n_in, "index": idx,
            "psg_ahi": row["arms"]["psg"]["ahi"], "hpg_hypno_ahi": row["arms"]["hpg_hypno"]["ahi"], "hpg_ahi": row["arms"]["hpg"]["ahi"],
            "ahi_ref": row["ahi_ref"]["aasm15"], "sev_ref": row["severity_ref"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rijen", type=Path, default=Path("/srv/CODE/docs/hpg_simulatie_20261025_raw/rijen.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    R = json.load(open(a.rijen))
    if a.n: R = R[: a.n]
    uit = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, u in enumerate(ex.map(nacht, R), 1):
            uit.append(u)
            if i % 10 == 0 or i == len(R): print(f"{i}/{len(R)}", flush=True)
    tot = lambda k: int(sum(u["tel"][k] for u in uit))
    med = lambda xs: float(st.median([x for x in xs if x is not None]))
    S = {"n": len(uit), "tst_afwijking_max_h": max(abs(u["tst_h_xml"] - u["tst_h_rij"]) for u in uit),
         "W_events": tot("W"), "W_flow_dood": tot("W_flow_dood"), "W_spo2_ong": tot("W_spo2_ong"), "W_los": tot("W_los"), "W_geldig": tot("W_geldig"),
         "W_eff_dood": tot("W_eff_dood"), "W_staart": tot("W_staart"), "W_staart_los": tot("W_staart_los"), "W_staart_geldig": tot("W_staart_geldig"),
         "W_voor_A": tot("W_voor_A"), "W_na_A": tot("W_na_A"), "W_in_A": tot("W_in_A"), "W_in_B": tot("W_in_B"),
         "S_events": tot("S"), "S_los": tot("S_los"), "S_buiten_A": tot("S") - tot("S_in_A"), "S_buiten_B": tot("S") - tot("S_in_B"),
         "ep_los_totaal": int(sum(u["ep_los"] for u in uit)), "ep_los_in_slaap": int(sum(u["ep_los_in_slaap"] for u in uit)),
         "ep_W_totaal": int(sum(u["ep_W"] for u in uit)), "ep_W_los": int(sum(u["ep_W_los"] for u in uit)),
         "slaapepochs_buiten_A": int(sum(u["slaap_buiten_A"] for u in uit)), "slaapepochs_buiten_B": int(sum(u["slaap_buiten_B"] for u in uit)),
         "ep_voor_A_med_h": med([u["ep_voor_A"] * EPOCH_S / 3600 for u in uit]), "ep_na_A_med_h": med([u["ep_na_A"] * EPOCH_S / 3600 for u in uit]),
         "nachten_zonder_Pres": sum(1 for u in uit if not u["kanalen"]["Pres"]), "nachten_zonder_Therm": sum(1 for u in uit if not u["kanalen"]["Therm"])}
    for k in ("A", "B", "C", "TIB"):
        d_psg = [u["index"][k] - u["psg_ahi"] for u in uit if u["index"][k] is not None]
        d_hh = [u["index"][k] - u["hpg_hypno_ahi"] for u in uit if u["index"][k] is not None]
        S[f"regel_{k}"] = {"uren_med": med([u["uren"][k] for u in uit]), "index_med": med([u["index"][k] for u in uit]),
                           "d_vs_psg_med": med(d_psg), "d_vs_psg_abs_ge5": int(sum(abs(x) >= 5 for x in d_psg)), "d_vs_psg_min": min(d_psg), "d_vs_psg_max": max(d_psg),
                           "d_vs_hpg_hypno_med": med(d_hh), "d_vs_hpg_hypno_abs_ge5": int(sum(abs(x) >= 5 for x in d_hh)),
                           "ernst_gelijk_ref": int(sum(severity(u["index"][k]) == u["sev_ref"] for u in uit if u["index"][k] is not None)),
                           "ernst_gelijk_psg": int(sum(severity(u["index"][k]) == severity(u["psg_ahi"]) for u in uit if u["index"][k] is not None))}
    (HIER / "waakpoort_stap0.json").write_text(json.dumps({"samenvatting": S, "per_nacht": uit}, indent=1))
    print(json.dumps(S, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
