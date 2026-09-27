#!/usr/bin/env python3
"""Replicatieharnas voor de bevroren U-Net-arousaldetector (preregistratie 27-09-2026).

    bench/eeg/_venv/bin/python scripts/arousal_unet_replicatie.py <cohort> <stap> [opties]

cohort: shhs (150 verse SHHS1-nachten) | mesa (76 verse MESA-nachten) | psgipa (SN1-5)
        | mesaval (de 79/80 MESA-validatienachten van de bench; alleen voor de bewakers)
stap:   unet      bevroren model -> kans -> events (tau 0,35) -> slaappoort -> 10 s-samenvoeging;
                  schrijft ook referentie (NSRR-arousals) en hypnogram per nacht
        baseline  psgscoring 0.34.2 `aasm_v3_rec`, productie-aanroep (zelfde hypnogram); daarna,
                  als de U-Net-events er al staan, dezelfde aanroep met die events als
                  `arousal_events` (doorwerking op RERA/RDI, koppeling, PLM)
        eval      gepaarde vergelijking per nacht: projectmatcher (IoU 0,20) + onset +-5 s,
                  count-ratio, tertielen van de referentie-arousalindex, REM/NREM, Wilcoxon,
                  de vooraf vastgelegde beslisregel, en de doorwerkingscijfers

Alles wat gemeten wordt staat in docs/arousal_unet_preregistratie_20260927.md; dit script
voegt geen keuze toe. Uitvoer buiten git: $UNET_REPL_OUT (default
/srv/CODE/docs/arousal_unet_20260927). Er wordt niets onder /srv/DATA geschreven.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, hashlib, json, math, shutil, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "bench" / "eeg"
for p in (REPO, REPO / "scripts", REPO / "bench", BENCH / "unet50", BENCH / "common"):
    sys.path.insert(0, str(p))

OUT = Path(os.environ.get("UNET_REPL_OUT", "/srv/CODE/docs/arousal_unet_20260927"))
MODEL = OUT / "model_frozen_cc91bb86.pt"
MODEL_SHA = "cc91bb867c167d7387cf619c8dd005ba294d30f640d0fe0a59c2eaeb3a798566"
TAU = 0.35
PROFILE = "aasm_v3_rec"
DATA = Path(os.environ.get("PSGSCORING_DATA_ROOT", "/srv/DATA"))
SHHS_EDF = DATA / "SHHS/shhs/polysomnography/edfs/shhs1"
SHHS_XML = DATA / "SHHS/shhs/polysomnography/annotations-events-nsrr/shhs1"
MESA_EDF = DATA / "MESA/mesa/polysomnography/edfs"
MESA_XML = DATA / "MESA/mesa/polysomnography/annotations-events-nsrr"
PSGIPA = DATA / "PSG-IPA/Resp_events/PSG"
# SHHS1-montage (NSRR): EEG = C4-A1, EEG(sec) = C3-A2, EOG(L)/EOG(R), EMG = kin. Het model kreeg
# in validatie C4-M1 + EOG-L + EMG (train.VAL_CH), dus hier de linker-equivalenten.
SHHS_CH = ["EEG", "EOG(L)", "EMG"]
# Respiratoire rollen zoals /srv/DATA/SHHS-validation/score_shhs.py ze eerder gaf (NEW AIR is
# een thermische sensor; de rol raakt alleen de doorwerkingscijfers, niet de arousals).
SHHS_CMAP = {"flow_pressure": "NEW AIR", "thorax": "THOR RES", "abdomen": "ABDO RES"}
SNS = ["SN1", "SN2", "SN3", "SN4", "SN5"]
SLEEP = {"N1", "N2", "N3", "R"}
MATCHERS = [("project", 0.20), ("onset", 5.0)]


# ── nachten en referentie ──────────────────────────────────────────────────────────────────
def ids_for(cohort: str) -> list[str]:
    if cohort == "psgipa":
        return list(SNS)
    f = {"shhs": OUT / "ids_shhs150.txt", "mesa": OUT / "ids_mesa76.txt",
         "mesaval": BENCH / "unet50" / "ids_val.txt"}[cohort]
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def edf_xml(cohort: str, rec: str) -> tuple[Path, Path | None]:
    if cohort == "shhs":
        return SHHS_EDF / f"{rec}.edf", SHHS_XML / f"{rec}-nsrr.xml"
    if cohort in ("mesa", "mesaval"):
        return MESA_EDF / f"{rec}.edf", MESA_XML / f"{rec}-nsrr.xml"
    return PSGIPA / f"{rec}_Respiration.edf", None


def out_dir(cohort: str) -> Path:
    d = OUT / "out" / cohort
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_ref(cohort: str, rec: str, dur_s: float | None = None):
    """(hypno, [(onset, offset)], dur_s) -- NSRR-xml, of de bench-referentie voor PSG-IPA."""
    if cohort == "psgipa":
        meta = json.loads((BENCH / "ref" / f"{rec}_hypno.json").read_text())
        from evaluate import lees_events
        ar = [(a, b) for a, b, _ in lees_events(BENCH / "ref" / f"{rec}_ref_arousal.csv")]
        return meta["hypno"], ar, float(meta["dur_s"])
    from data import parse_mesa_xml
    edf, xml = edf_xml(cohort, rec)
    if dur_s is None:
        import mne
        mne.set_log_level("ERROR")
        h = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)
        dur_s = h.n_times / h.info["sfreq"]
    hypno, ar = parse_mesa_xml(xml, dur_s)
    return hypno, ar, dur_s


def write_events(path: Path, events, typ: str = "arousal") -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["onset_s", "offset_s", "type"])
        for a, b in events:
            w.writerow([f"{a:.3f}", f"{b:.3f}", typ])


def write_ref(cohort: str, rec: str, hypno, ar, dur_s: float) -> None:
    d = out_dir(cohort)
    if not (d / f"{rec}_ref.csv").exists():
        write_events(d / f"{rec}_ref.csv", ar)
    if not (d / f"{rec}_hypno.json").exists():
        tst_h = sum(1 for s in hypno if s in SLEEP) * 30.0 / 3600.0
        (d / f"{rec}_hypno.json").write_text(json.dumps(
            {"hypno": hypno, "dur_s": dur_s, "tst_h": round(tst_h, 4)}))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def min_interval_s() -> float:
    from psgscoring.profiles import get_profile
    return float(get_profile(PROFILE).post_processing.arousal_min_interval_s)


def merge_10s(events, interval_s: float):
    """De 10 s-samenvoeging van de huidige keten, op tuples."""
    from psgscoring.arousal import enforce_min_arousal_interval
    dicts = [{"onset_s": float(a), "end_s": float(b), "duration_s": float(b - a)} for a, b in events]
    uit = enforce_min_arousal_interval(dicts, interval_s)
    return [(float(e["onset_s"]), float(e["end_s"])) for e in uit]


def jsonl_append(path: Path, rij: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rij, default=str) + "\n")


# ── stap unet ──────────────────────────────────────────────────────────────────────────────
def stap_unet(a) -> None:
    import torch
    from model import UNet1D
    from train import predict_night, VAL_CH
    from data import load_channels, load_psgipa, FS, MESA_CH
    from postproc import prob_to_events, gate_sleep

    model_path = Path(a.model) if a.model else MODEL
    sha = sha256(model_path)
    if model_path == MODEL and sha != MODEL_SHA:
        raise SystemExit(f"bevroren model heeft sha256 {sha}, verwacht {MODEL_SHA}")
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    if a.cpu:
        torch.set_num_threads(a.threads)
    ck = torch.load(model_path, map_location=dev, weights_only=False)
    model = UNet1D(**ck["config"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    thr = float(a.thr) if a.thr is not None else TAU
    interval = 0.0 if a.no_merge else min_interval_s()
    tag = a.tag
    d = out_dir(a.cohort)
    log = d / f"unet{tag}_log.jsonl"
    jsonl_append(log, {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "model": str(model_path),
                       "sha256": sha, "thr": thr, "min_interval_s": interval, "device": str(dev),
                       "threads": a.threads if a.cpu else None, "zero": a.zero,
                       "ck_thr": ck.get("thr"), "ck_epoch": ck.get("epoch")})
    ids = ids_for(a.cohort)[: a.limit] if a.limit else ids_for(a.cohort)
    for rec in ids:
        f_out = d / f"{rec}_unet{tag}.csv"
        if f_out.exists() and not a.force:
            continue
        t0 = time.time()
        try:
            if a.cohort == "psgipa":
                X, dur, names = load_psgipa(rec)
            elif a.cohort == "shhs":
                names = SHHS_CH
                X, dur = load_channels(edf_xml(a.cohort, rec)[0], names)
            else:
                names = [MESA_CH[i] for i in VAL_CH]
                X, dur = load_channels(edf_xml(a.cohort, rec)[0], names)
            hypno, ar, dur = load_ref(a.cohort, rec, dur)
            if a.zero:
                idx = {"eog": [1], "emg": [2], "eeg_only": [1, 2]}[a.zero]
                X = X.copy()
                X[idx] = 0
            t1 = time.time()
            p = predict_night(model, X, dev)
            t_fwd = time.time() - t1
            ev_raw = gate_sleep(prob_to_events(p, FS, thr), hypno)
            ev = merge_10s(ev_raw, interval) if interval > 0 else ev_raw
        except Exception as e:  # noqa: BLE001
            jsonl_append(log, {"rec": rec, "error": repr(e)})
            print(rec, "FOUT", repr(e), flush=True)
            continue
        write_ref(a.cohort, rec, hypno, ar, dur)
        write_events(d / f"{rec}_unet{tag}_nomerge.csv", ev_raw)
        write_events(f_out, ev)
        if not tag:
            k = FS // 10
            np.savez_compressed(d / f"{rec}_prob10hz.npz",
                                p=p[: (len(p) // k) * k].reshape(-1, k).mean(1).astype(np.float16), fs=10)
        rij = {"rec": rec, "channels": names, "n_ref": len(ar), "n_raw": len(ev_raw), "n_unet": len(ev),
               "t_total_s": round(time.time() - t0, 1), "t_forward_s": round(t_fwd, 2), "device": str(dev)}
        jsonl_append(log, rij)
        print(rec, names, "ref", len(ar), "unet", len(ev), f"{rij['t_total_s']} s", flush=True)


# ── stap baseline (+ doorwerking) ──────────────────────────────────────────────────────────
def _tel_arousalvelden(events) -> dict:
    """Aantal respiratoire events per arousal-gerelateerd veld (welke velden er zijn, telt mee)."""
    uit: dict = {}
    for e in events or []:
        for k, v in (e or {}).items():
            if "arousal" in k.lower() and v not in (None, False, 0, "", [], {}):
                uit[k] = uit.get(k, 0) + 1
    return uit


def _samenvatting(res: dict) -> dict:
    def _js(o):
        return json.loads(json.dumps(o, default=str))
    ar = res.get("arousal") or {}
    resp = res.get("respiratory") or {}
    plm = res.get("plm") or {}
    ar_ev = ar.get("events") or []
    det = (ar.get("arousals") or {}).get("summary") or {}   # detector-provenance (multi: derivations)
    eeg_used = ((res.get("meta") or {}).get("channels_used") or {}).get("eeg")
    return {
        "arousal_detector_summary": _js({k: v for k, v in det.items()
                                         if not isinstance(v, (list, dict)) or k in ("derivations", "n_per_derivation")}),
        "psgscoring_version": (res.get("meta") or {}).get("psgscoring_version"),
        "channels_used": _js((res.get("meta") or {}).get("channels_used")),
        "arousal_source": ar.get("source"),
        "arousal_error": ar.get("error"),
        "arousal_derivations": _js(det.get("derivations") or ([eeg_used] if eeg_used else None)),
        "n_arousal_events": len(ar_ev),
        "arousal_summary": _js(ar.get("summary")),
        "arousal_coupling": _js(ar.get("coupling")),
        "respiratory_summary": _js(resp.get("summary")),
        "n_resp_events": len(resp.get("events") or []),
        "resp_arousal_fields": _tel_arousalvelden(resp.get("events")),
        "resp_types": _js({t: sum(1 for e in resp.get("events") or [] if e.get("type") == t)
                           for t in sorted({e.get("type") for e in resp.get("events") or []} - {None})}),
        "plm_summary": _js(plm.get("summary")),
        "analysis_warnings": _js(res.get("analysis_warnings")),
    }


def _baseline_nacht(werk: tuple) -> dict:
    cohort, rec, tag, force = werk
    import mne
    mne.set_log_level("ERROR")
    import psgscoring
    from evaluate import lees_events
    d = out_dir(cohort)
    f_base = d / f"{rec}_base.csv"
    f_door = d / f"{rec}_door{tag}.json"
    f_unet = d / f"{rec}_unet{tag}.csv"
    rij: dict = {"rec": rec}
    edf, _ = edf_xml(cohort, rec)
    try:
        t0 = time.time()
        raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
        dur = raw.n_times / raw.info["sfreq"]
        hypno, ar, dur = load_ref(cohort, rec, dur)
        write_ref(cohort, rec, hypno, ar, dur)
        cmap = SHHS_CMAP if cohort == "shhs" else None
        if force or not f_base.exists():
            res = psgscoring.run_pneumo_analysis(raw, hypno=hypno, scoring_profile=PROFILE, channel_map=cmap)
            ev = (res.get("arousal") or {}).get("events") or []
            write_events(f_base, [(float(e["onset_s"]),
                                   float(e["end_s"]) if e.get("end_s") is not None
                                   else float(e["onset_s"]) + float(e.get("duration_s") or 0.0)) for e in ev])
            (d / f"{rec}_base.json").write_text(json.dumps(_samenvatting(res), indent=1))
            rij.update(n_base=len(ev), t_base_s=round(time.time() - t0, 1))
        if f_unet.exists() and (force or not f_door.exists()):
            t1 = time.time()
            ext = [{"onset_s": a, "duration_s": b - a} for a, b, _ in lees_events(f_unet)]
            raw2 = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
            res2 = psgscoring.run_pneumo_analysis(raw2, hypno=hypno, scoring_profile=PROFILE,
                                                  channel_map=cmap, arousal_events=ext)
            f_door.write_text(json.dumps(_samenvatting(res2), indent=1))
            rij.update(n_unet_ext=len(ext), t_door_s=round(time.time() - t1, 1))
    except Exception as e:  # noqa: BLE001
        rij["error"] = repr(e)
    return rij


def stap_baseline(a) -> None:
    from multiprocessing import get_context
    d = out_dir(a.cohort)
    ids = ids_for(a.cohort)[: a.limit] if a.limit else ids_for(a.cohort)
    if a.cohort == "mesaval":
        # De bench heeft deze baseline al (bench/eeg/baseline_mesa, psgscoring 0.34.2): overnemen.
        n = 0
        for rec in ids:
            src = BENCH / "baseline_mesa" / f"{rec}_pred.csv"
            if src.exists():
                shutil.copy(src, d / f"{rec}_base.csv"); n += 1
        print(f"mesaval: {n} baseline-bestanden overgenomen uit bench/eeg/baseline_mesa")
        return
    log = d / f"baseline{a.tag}_log.jsonl"
    jsonl_append(log, {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": a.workers, "n": len(ids)})
    werk = [(a.cohort, rec, a.tag, a.force) for rec in ids]
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for rij in pool.imap_unordered(_baseline_nacht, werk):
            jsonl_append(log, rij)
            print(rij, flush=True)


# ── stap eval ──────────────────────────────────────────────────────────────────────────────
def _metriek(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "sens": tp / (tp + fn) if tp + fn else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def _stage_of(ev, hypno):
    ep = int(ev[0] // 30.0)
    return hypno[ep] if 0 <= ep < len(hypno) else None


def _wilcoxon(d):
    from scipy.stats import wilcoxon
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.all(d == 0):
        return None
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return None


def stap_eval(a) -> None:
    from evaluate import lees_events, evalueer
    d = out_dir(a.cohort)
    tag = a.tag
    armen = {"base": "{rec}_base.csv", "unet": f"{{rec}}_unet{tag}.csv"}
    if a.extra_arm:
        armen[a.extra_arm[0]] = a.extra_arm[1]
    nachten = {}
    ontbreekt = {k: [] for k in armen}
    geen_ref = []
    for rec in ids_for(a.cohort):
        f_ref = d / f"{rec}_ref.csv"; f_h = d / f"{rec}_hypno.json"
        if not (f_ref.exists() and f_h.exists()):
            continue
        paden = {k: d / v.format(rec=rec) for k, v in armen.items()}
        mis = [k for k, p in paden.items() if not p.exists()]
        if mis:
            for k in mis:
                ontbreekt[k].append(rec)
            continue
        ref = lees_events(f_ref); meta = json.loads(f_h.read_text()); hypno = meta["hypno"]
        if not ref:
            geen_ref.append(rec)
            continue
        tst_h = float(meta["tst_h"]) or float("nan")
        rij = {"n_ref": len(ref), "tst_h": tst_h, "ref_index": len(ref) / tst_h if tst_h > 0 else None,
               "armen": {}}
        for k, p in paden.items():
            pred = lees_events(p)
            r = {"n": len(pred), "index": len(pred) / tst_h if tst_h > 0 else None,
                 "count_ratio": len(pred) / len(ref)}
            for m, t in MATCHERS:
                r[m] = evalueer(pred, ref, m, t, False)["totaal"]
            for st_lbl, st_set in (("rem", {"R"}), ("nrem", {"N1", "N2", "N3"})):
                p_s = [e for e in pred if _stage_of(e, hypno) in st_set]
                r_s = [e for e in ref if _stage_of(e, hypno) in st_set]
                tp, fp, fn = (evalueer(p_s, r_s, "project", 0.20, False)["totaal"][x] for x in ("tp", "fp", "fn"))
                r[st_lbl] = {"tp": tp, "fp": fp, "fn": fn, "n_ref": len(r_s), "n_pred": len(p_s)}
            rij["armen"][k] = r
        nachten[rec] = rij

    def f1(rij, arm, m="project"):
        v = rij["armen"][arm][m]["f1"]
        return 0.0 if v is None else float(v)

    recs = sorted(nachten)
    uit = {"cohort": a.cohort, "tag": tag, "n": len(recs), "geen_ref": geen_ref,
           "ontbreekt": {k: v for k, v in ontbreekt.items() if v}, "armen": {}, "paren": {}}
    for k in armen:
        A = {}
        for m, _ in MATCHERS:
            tp = sum(nachten[r]["armen"][k][m]["tp"] for r in recs)
            fp = sum(nachten[r]["armen"][k][m]["fp"] for r in recs)
            fn = sum(nachten[r]["armen"][k][m]["fn"] for r in recs)
            A[m] = {"gepoold": _metriek(tp, fp, fn),
                    "mediaan_f1": float(np.median([f1(nachten[r], k, m) for r in recs])) if recs else None,
                    "gemiddelde_f1": float(np.mean([f1(nachten[r], k, m) for r in recs])) if recs else None}
        for st in ("rem", "nrem"):
            tp = sum(nachten[r]["armen"][k][st]["tp"] for r in recs)
            fp = sum(nachten[r]["armen"][k][st]["fp"] for r in recs)
            fn = sum(nachten[r]["armen"][k][st]["fn"] for r in recs)
            A[st] = _metriek(tp, fp, fn)
        A["mediaan_count_ratio"] = float(np.median([nachten[r]["armen"][k]["count_ratio"] for r in recs])) if recs else None
        A["n_events"] = sum(nachten[r]["armen"][k]["n"] for r in recs)
        A["bias_index"] = float(np.mean([nachten[r]["armen"][k]["index"] - nachten[r]["ref_index"]
                                         for r in recs if nachten[r]["ref_index"] is not None])) if recs else None
        uit["armen"][k] = A
    uit["n_ref_events"] = sum(nachten[r]["n_ref"] for r in recs)

    # tertielen op referentie-arousalindex (laag -> hoog)
    orde = sorted(recs, key=lambda r: (nachten[r]["ref_index"] is None, nachten[r]["ref_index"] or 0.0))
    tertielen = [list(x) for x in np.array_split(np.array(orde, dtype=object), 3)] if recs else [[], [], []]

    for k in [x for x in armen if x != "base"]:
        d_f1 = {m: np.array([f1(nachten[r], k, m) - f1(nachten[r], "base", m) for r in recs]) for m, _ in MATCHERS}
        P = {}
        for m, _ in MATCHERS:
            dd = d_f1[m]
            P[m] = {"mean_dF1": float(dd.mean()) if len(dd) else None,
                    "median_dF1": float(np.median(dd)) if len(dd) else None,
                    "beter": int((dd > 0).sum()), "slechter": int((dd < 0).sum()), "gelijk": int((dd == 0).sum()),
                    "wilcoxon_p": _wilcoxon(dd)}
        P["tertielen"] = []
        for i, T in enumerate(tertielen):
            if not T:
                continue
            dd = np.array([f1(nachten[r], k) - f1(nachten[r], "base") for r in T])
            P["tertielen"].append({
                "tertiel": i + 1, "n": len(T),
                "ref_index_bereik": [round(min(nachten[r]["ref_index"] or 0 for r in T), 1),
                                     round(max(nachten[r]["ref_index"] or 0 for r in T), 1)],
                "mean_dF1": float(dd.mean()), "beter": int((dd > 0).sum()),
                "f1_base": float(np.mean([f1(nachten[r], "base") for r in T])),
                "f1_arm": float(np.mean([f1(nachten[r], k) for r in T])),
                "bias_base": float(np.mean([nachten[r]["armen"]["base"]["index"] - nachten[r]["ref_index"] for r in T])),
                "bias_arm": float(np.mean([nachten[r]["armen"][k]["index"] - nachten[r]["ref_index"] for r in T])),
                "count_ratio_base": float(np.median([nachten[r]["armen"]["base"]["count_ratio"] for r in T])),
                "count_ratio_arm": float(np.median([nachten[r]["armen"][k]["count_ratio"] for r in T])),
            })
        # beslisregel (vooraf): zie preregistratie §Beslisregel
        n = len(recs); pj = P["project"]
        regel = {}
        if a.cohort == "shhs":
            regel = {
                # teller op de 150 getrokken nachten; nachten zonder referentie-arousals (ΔF1
                # ongedefinieerd) tellen niet als "beter", dus de lat blijft 90.
                "dF1>0 op >=90/150": pj["beter"] >= 90,
                "wilcoxon p<0,05": pj["wilcoxon_p"] is not None and pj["wilcoxon_p"] < 0.05,
                "mediane count-ratio in [0,80;1,25]": 0.80 <= (uit["armen"][k]["mediaan_count_ratio"] or 0) <= 1.25,
                "geen tertiel mean dF1 < -0,02": all(t["mean_dF1"] >= -0.02 for t in P["tertielen"]),
            }
        elif a.cohort == "mesa":
            regel = {"dF1>0 op meerderheid": pj["beter"] > n / 2,
                     "wilcoxon p<0,05": pj["wilcoxon_p"] is not None and pj["wilcoxon_p"] < 0.05}
        elif a.cohort == "psgipa":
            regel = {"F1 niet lager op >=4/5": (pj["beter"] + pj["gelijk"]) >= 4}
        if regel:
            regel["ALLE"] = all(regel.values())
        P["beslisregel"] = regel
        uit["paren"][k] = P

    # doorwerking: base.json vs door.json (alleen rapportage)
    door = []
    for rec in recs:
        fb = d / f"{rec}_base.json"; fd = d / f"{rec}_door{tag}.json"
        if fb.exists() and fd.exists():
            b = json.loads(fb.read_text()); u = json.loads(fd.read_text())
            def g(x, *ks):
                for kk in ks:
                    x = (x or {}).get(kk) if isinstance(x, dict) else None
                return x
            door.append({"rec": rec,
                         "arousal_index": (g(b, "arousal_summary", "arousal_index"), g(u, "arousal_summary", "arousal_index")),
                         "rdi": (g(b, "arousal_summary", "rdi"), g(u, "arousal_summary", "rdi")),
                         "n_reras": (g(b, "arousal_summary", "n_reras"), g(u, "arousal_summary", "n_reras")),
                         "ahi_total": (g(b, "respiratory_summary", "ahi_total"), g(u, "respiratory_summary", "ahi_total")),
                         "n_resp_events": (b.get("n_resp_events"), u.get("n_resp_events")),
                         "resp_arousal_fields": (b.get("resp_arousal_fields"), u.get("resp_arousal_fields")),
                         "plm_arousal_index": (g(b, "plm_summary", "plm_arousal_index"), g(u, "plm_summary", "plm_arousal_index")),
                         "derivations_base": b.get("arousal_derivations"),
                         "channels_used": b.get("channels_used")})
    if door:
        S = {"n": len(door)}
        for key in ("arousal_index", "rdi", "n_reras", "ahi_total", "n_resp_events", "plm_arousal_index"):
            paren = [(x[key][0], x[key][1]) for x in door if x[key][0] is not None and x[key][1] is not None]
            if paren:
                bb = np.array([p[0] for p in paren], float); uu = np.array([p[1] for p in paren], float)
                S[key] = {"n": len(paren), "mean_base": float(bb.mean()), "mean_unet": float(uu.mean()),
                          "mean_delta": float((uu - bb).mean()), "median_delta": float(np.median(uu - bb)),
                          "wilcoxon_p": _wilcoxon(uu - bb)}
        velden = {}
        for x in door:
            for i, side in enumerate(("base", "unet")):
                for kk, v in (x["resp_arousal_fields"][i] or {}).items():
                    velden.setdefault(kk, [0, 0])[i] += v
        S["resp_arousal_fields_totaal"] = velden
        S["derivations_base_voorbeeld"] = door[0]["derivations_base"]
        S["channels_used_voorbeeld"] = door[0]["channels_used"]
        uit["doorwerking"] = S
    uit["nachten"] = nachten
    (d / f"eval{tag}.json").write_text(json.dumps(uit, indent=1, default=str))
    (d / f"eval{tag}.md").write_text(_rapport(uit))
    print(_rapport(uit))


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _rapport(uit: dict) -> str:
    L = [f"# {uit['cohort']}{uit['tag']}: n={uit['n']} nachten met referentie "
         f"({uit['n_ref_events']} NSRR/ref-arousals; {len(uit['geen_ref'])} zonder referentie overgeslagen"
         + (f"; ontbreekt: { {k: len(v) for k, v in uit['ontbreekt'].items()} }" if uit["ontbreekt"] else "") + ")", ""]
    L.append("| arm | F1 gepoold (IoU 0,20) | mediaan F1 | F1 onset ±5 s | sens | PPV | events | mediaan count-ratio | bias index/u | REM F1 | NREM F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, A in uit["armen"].items():
        g = A["project"]["gepoold"]
        L.append(f"| {k} | {_f(g['f1'])} | {_f(A['project']['mediaan_f1'])} | {_f(A['onset']['gepoold']['f1'])} | "
                 f"{_f(g['sens'])} | {_f(g['ppv'])} | {A['n_events']} | {_f(A['mediaan_count_ratio'], 2)} | "
                 f"{_f(A['bias_index'], 2)} | {_f(A['rem']['f1'])} | {_f(A['nrem']['f1'])} |")
    for k, P in uit["paren"].items():
        L += ["", f"## gepaard {k} − base"]
        for m in ("project", "onset"):
            p = P[m]
            L.append(f"- {m}: ΔF1 gemiddeld {_f(p['mean_dF1'])}, mediaan {_f(p['median_dF1'])}; "
                     f"beter/slechter/gelijk {p['beter']}/{p['slechter']}/{p['gelijk']}; Wilcoxon p={_f(p['wilcoxon_p'], 4) if p['wilcoxon_p'] is None or p['wilcoxon_p'] >= 1e-4 else f'{p['wilcoxon_p']:.1e}'}")
        L.append("")
        L.append("| tertiel ref-index | n | bereik /u | F1 base | F1 arm | ΔF1 | beter | bias base | bias arm | count-ratio base | count-ratio arm |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in P["tertielen"]:
            L.append(f"| T{t['tertiel']} | {t['n']} | {t['ref_index_bereik'][0]}–{t['ref_index_bereik'][1]} | {_f(t['f1_base'])} | "
                     f"{_f(t['f1_arm'])} | {_f(t['mean_dF1'])} | {t['beter']} | {_f(t['bias_base'], 2)} | {_f(t['bias_arm'], 2)} | "
                     f"{_f(t['count_ratio_base'], 2)} | {_f(t['count_ratio_arm'], 2)} |")
        if P["beslisregel"]:
            L += ["", "**Beslisregel (vooraf):** " + "; ".join(f"{kk}: {'JA' if v else 'NEE'}" for kk, v in P["beslisregel"].items())]
    if "doorwerking" in uit:
        S = uit["doorwerking"]
        L += ["", f"## doorwerking (psgscoring met U-Net-events als `arousal_events`, n={S['n']}; rapportage, geen criterium)"]
        L.append("| maat | n | gem. base | gem. U-Net | Δ gem. | Δ mediaan | Wilcoxon p |")
        L.append("|---|---|---|---|---|---|---|")
        for key in ("arousal_index", "rdi", "n_reras", "ahi_total", "n_resp_events", "plm_arousal_index"):
            if key in S:
                s = S[key]
                L.append(f"| {key} | {s['n']} | {_f(s['mean_base'], 2)} | {_f(s['mean_unet'], 2)} | {_f(s['mean_delta'], 2)} | "
                         f"{_f(s['median_delta'], 2)} | {_f(s['wilcoxon_p'], 4)} |")
        L.append(f"- arousalvelden op respiratoire events (base, U-Net): {S['resp_arousal_fields_totaal']}")
        L.append(f"- afleidingen baseline (voorbeeld): {S['derivations_base_voorbeeld']}; kanalen: {S['channels_used_voorbeeld']}")
    return "\n".join(L) + "\n"


# ── cli ────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort", choices=["shhs", "mesa", "psgipa", "mesaval"])
    ap.add_argument("stap", choices=["unet", "baseline", "eval"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="suffix voor varianten (bv. _cpu, _noeog, _seed28)")
    ap.add_argument("--model", default=None, help="ander checkpoint (bewaker a); default het bevroren model")
    ap.add_argument("--thr", type=float, default=None, help="werkpunt; default tau 0,35 (preregistratie)")
    ap.add_argument("--no-merge", action="store_true", help="zonder de 10 s-samenvoeging (bench-conventie)")
    ap.add_argument("--zero", choices=["eog", "emg", "eeg_only"], default=None, help="montage-ablatie (bewaker c)")
    ap.add_argument("--cpu", action="store_true", help="inferentie zonder GPU (bewaker b)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--extra-arm", nargs=2, metavar=("NAAM", "PATROON"), default=None,
                    help="extra arm in eval, patroon met {rec}, bv. unet_noeog {rec}_unet_noeog.csv")
    a = ap.parse_args()
    if a.tag and not a.tag.startswith("_"):
        a.tag = "_" + a.tag
    {"unet": stap_unet, "baseline": stap_baseline, "eval": stap_eval}[a.stap](a)


if __name__ == "__main__":
    main()
