#!/usr/bin/env python3
"""Laatste tien eigen PSG's, zeven armen in situ (container uit yasaflaskified:0.39.2, psgscoring 0.37.0).
A0 productie | SB stadiumbasislijn uit | ST strictness 0,50 | BR profiel breath | LG lgbm | ALL alles terug op breath | REC rec+SB+LG.
Schrijft L*.json ZONDER patiëntgegevens. Prereg: psgscoring docs/laatste10_humaan_vs_machine_preregistratie_20261102.md
Aanroep (in de container): python run.py [--arms A0,SB] [--suffix smoke] [L01 L02 ...]"""
import json, os, sys, time, traceback
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, "/data/slaapkliniek/myproject")
D = Path("/data/slaapkliniek/metingen/laatste10_20261102"); UP = Path("/data/slaapkliniek/uploads")
VERSIE = "0.37.0"; ENV = "PSGSCORING_AROUSAL_DETECTOR"
PROFIELEN = ("aasm_v3_breath_dual", "aasm_v3_breath", "aasm_v3_rec")
SB = {"STAGE_BASELINE_PARTIAL_EPOCHS": False}; ST = {"HYPOPNEA_STRICTNESS": 0.50}
ARMEN = {
    "A0": dict(profiel="aasm_v3_breath_dual", velden={}, detector=None),
    "SB": dict(profiel="aasm_v3_breath_dual", velden=SB, detector=None),
    "ST": dict(profiel="aasm_v3_breath_dual", velden=ST, detector=None),
    "BR": dict(profiel="aasm_v3_breath", velden={}, detector=None),
    "LG": dict(profiel="aasm_v3_breath_dual", velden={}, detector="lgbm"),
    "ALL": dict(profiel="aasm_v3_breath", velden={**SB, **ST}, detector="lgbm"),
    "REC": dict(profiel="aasm_v3_rec", velden=SB, detector="lgbm"),
}
WAKE = {"W", "Wake", "wake", "WAKE", 0, "0", "Art", "ART"}


def _tel(events):
    uit = {}
    for e in events or []:
        t = str(e.get("type")); uit[t] = uit.get(t, 0) + 1
    return uit


def _scalars(d):
    return {k: v for k, v in (d or {}).items() if not isinstance(v, (list, dict))}


def _events(evs):
    return [{k: v for k, v in e.items() if not isinstance(v, (list, dict))} for e in (evs or [])]


def _find(obj, key, diepte=0):
    if diepte > 4 or not isinstance(obj, dict):
        return None
    if isinstance(obj.get(key), dict):
        return obj[key]
    for v in obj.values():
        if isinstance(v, dict):
            w = _find(v, key, diepte + 1)
            if w is not None:
                return w
    return None


def _run(raw, hypno, cmap, art, pcm, arm):
    from pneumo_analysis import run_pneumo_analysis
    from psgscoring.constants import SCORING_PROFILES
    oud = {}
    os.environ.pop(ENV, None); os.environ.pop("PSGSCORING_AROUSAL_UNET_THRESHOLD", None)
    if arm["detector"]:
        os.environ[ENV] = arm["detector"]
    try:
        for k, v in arm["velden"].items():
            for p in PROFIELEN:
                if k in SCORING_PROFILES[p]:
                    oud[(p, k)] = SCORING_PROFILES[p][k]; SCORING_PROFILES[p][k] = v
        gelezen = {k: SCORING_PROFILES[arm["profiel"]].get(k) for k in ("STAGE_BASELINE_PARTIAL_EPOCHS", "HYPOPNEA_STRICTNESS")}
        r = run_pneumo_analysis(raw=raw.copy(), hypno=hypno, channel_map=cmap, artifact_epochs=art, scoring_profile=arm["profiel"],
                                split_night="off", split_night_breakpoint_s=None, position_code_map=pcm)
    finally:
        for (p, k), v in oud.items():
            SCORING_PROFILES[p][k] = v
        os.environ.pop(ENV, None)
    resp = r.get("respiratory") or {}; summ = resp.get("summary") or {}; ev = resp.get("events") or []
    ar = r.get("arousal") or {}; arsum = ar.get("summary") or {}; meta = r.get("meta") or {}
    dual = resp.get("dual_sensor_apnea") if isinstance(resp.get("dual_sensor_apnea"), dict) else None
    hy = [e for e in ev if "hypopnea" in str(e.get("type"))]
    return {"profiel": arm["profiel"], "velden": arm["velden"], "env_detector": arm["detector"], "gelezen": gelezen,
            "ahi_total": summ.get("ahi_total"), "oahi": summ.get("oahi"), "ahi_incl_uncertain": summ.get("ahi_incl_uncertain"),
            "severity": summ.get("severity") or summ.get("ahi_severity"), "n_events": len(ev), "per_type": _tel(ev),
            "dual": ({k: dual.get(k) for k in ("n_thermistor", "n_pressure", "n_both", "n_thermistor_only", "n_pressure_only", "n_kept")} if dual else None),
            "hyp_desat3": sum(1 for e in hy if (e.get("desaturation_pct") or 0) >= 3), "hyp_arousal_vlag": sum(1 for e in hy if e.get("rule1a_arousal")),
            "n_arousals": len(ar.get("events") or []), "arousal_index": arsum.get("arousal_index"),
            "summary": _scalars(summ), "arousal_summary": _scalars(arsum), "resp_scalars": _scalars(resp),
            "thermistor_check": _scalars(_find(r, "thermistor_check") or {}), "channels_used": meta.get("channels_used"),
            "meta_scalars": _scalars(meta), "warnings": r.get("analysis_warnings"), "resp_events": _events(ev)}


def analyse(args):
    item, armen, suffix = args; R, jid = item["R"], item["job_id"]; t0 = time.time()
    uit = {"R": R, "ahi_opgeslagen": item.get("ahi_opgeslagen"), "severity_opgeslagen": item.get("severity_opgeslagen"),
           "profiel_opgeslagen": item.get("profiel_opgeslagen"), "versie_opgeslagen": item.get("versie_opgeslagen"), "armen": {}}
    try:
        import mne; mne.set_log_level("ERROR")
        import logging; logging.getLogger().setLevel(logging.ERROR)
        from tasks import _load_edf, _pneumo_load_plan, _pneumo_channel_map, _detect_pneumo_channels, _position_code_map
        from signal_io import read_raw_signal
        import psgscoring
        if psgscoring.__version__ != VERSIE:
            raise RuntimeError(f"psgscoring {psgscoring.__version__}, verwacht {VERSIE}")
        cfg = json.load(open(UP / f"{jid}_config.json")); res = json.load(open(UP / f"{jid}_results.json"))
        hypno = res["staging"]["hypnogram"]; art = [e["epoch"] for e in (res.get("artifacts") or {}).get("artifact_epochs", [])]
        pc = cfg.get("pneumo_channels") or {}; emg_ch = cfg.get("emg_ch"); eeg_ch = cfg.get("eeg_ch"); eog_ch = cfg.get("eog_ch"); edf = cfg["edf_path"]
        pcm = _position_code_map(cfg); hdr = read_raw_signal(edf, preload=False, verbose=False); eeg_all = list(hdr.ch_names)
        lijst = _detect_pneumo_channels(edf, pc); needed = _pneumo_load_plan(lijst, eeg_ch, emg_ch, eeg_all=eeg_all, eog_ch=eog_ch)
        raw = _load_edf(edf, needed, label="PNEUMO"); cmap = _pneumo_channel_map(pc, emg_ch, raw, eog_ch=eog_ch)
        uit.update(psgscoring=psgscoring.__version__, n_epochs=len(hypno), n_artefact=len(art), sfreq=raw.info["sfreq"],
                   tst_machine_min=round(sum(1 for s in hypno if s not in WAKE) * 0.5, 1),
                   rollen={k: bool(v) for k, v in pc.items()}, eog_in_raw=bool(eog_ch and eog_ch in raw.ch_names), position_code_map=bool(pcm))
        for naam in armen:
            t1 = time.time()
            uit["armen"][naam] = _run(raw, hypno, cmap, art, pcm, ARMEN[naam]); uit["armen"][naam]["t_s"] = round(time.time() - t1, 1)
        uit["t_total_s"] = round(time.time() - t0, 1)
    except Exception:
        uit["error"] = traceback.format_exc()[-1500:]
    json.dump(uit, open(D / f"{R}{suffix}.json", "w"), indent=1, default=str)
    return R, uit.get("error") is None, uit.get("t_total_s")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    argv = sys.argv[1:]; armen = list(ARMEN); suffix = ""
    if "--arms" in argv:
        i = argv.index("--arms"); armen = argv[i + 1].split(","); del argv[i:i + 2]
    if "--suffix" in argv:
        i = argv.index("--suffix"); suffix = "_" + argv[i + 1]; del argv[i:i + 2]
    items = json.load(open(D / "mapping.json"))["sample"]
    if argv:
        items = [i for i in items if i["R"] in set(argv)]
    items = [i for i in items if not (D / f"{i['R']}{suffix}.json").exists()]
    npar = min(8, max(1, len(items)))
    print(f"start {time.strftime('%H:%M:%S')}: {len(items)} nachten, {npar} parallel, armen {armen}", flush=True)
    with get_context("spawn").Pool(npar, maxtasksperchild=1) as pool:
        for R, ok, t in pool.imap_unordered(analyse, [(i, armen, suffix) for i in items]):
            print(f"  {R}: {'ok' if ok else 'FOUT'} {t} s  {time.strftime('%H:%M:%S')}", flush=True)
    print("KLAAR", time.strftime("%H:%M:%S"), flush=True)
