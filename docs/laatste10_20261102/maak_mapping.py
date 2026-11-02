#!/usr/bin/env python3
"""Selectie van de tien meest recente diagnostische PSG-jobs + sjabloon voor de menselijke cijfers.
Draait op de HOST (python3, stdlib) vanuit /data/slaapkliniek. Schrijft mapping.json en humaan.csv
(beide mode 600, blijven op de server). Met --cijfers: leest humaan.csv en schrijft humaan_cijfers.json
ZONDER namen of job-id's (het enige bestand dat de server verlaat).
Prereg: psgscoring docs/laatste10_humaan_vs_machine_preregistratie_20261102.md"""
import csv, json, os, sys, time
from pathlib import Path
BASE = Path("/data/slaapkliniek"); UP = BASE / "uploads"; D = BASE / "metingen" / "laatste10_20261102"
N = 10
KOLOMMEN = ["L", "job_kort", "bestandsnaam", "upload_dag", "ahi_opgeslagen", "ahi_h", "n_apneu_h", "n_hypopneu_h", "tst_h_min", "regel_h"]
WAKE = {"W", "Wake", "wake", "WAKE", 0, "0", "Art", "ART"}


def _summary(res):
    pn = res.get("pneumo") or {}
    for pad in (("respiratory", "summary"), ("summary",)):
        d = pn
        for k in pad:
            d = (d or {}).get(k) or {}
        if isinstance(d, dict) and d.get("ahi_total") is not None:
            return d
    return {}


def _zoek(d, sleutels, diepte=0):
    """Eerste scalaire waarde onder een van `sleutels`, tot 3 niveaus diep."""
    if diepte > 3 or not isinstance(d, dict):
        return None
    for k in sleutels:
        if k in d and not isinstance(d[k], (dict, list)):
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            w = _zoek(v, sleutels, diepte + 1)
            if w is not None:
                return w
    return None


def _tst_min(hypno):
    return round(sum(1 for s in hypno if s not in WAKE) * 0.5, 1)


def _getal(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def selecteer():
    sample, uitgesloten, bekeken = [], {}, 0
    for cfgp in sorted(UP.glob("*_config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(sample) >= N:
            break
        bekeken += 1
        jid = cfgp.name[: -len("_config.json")]
        resp = UP / f"{jid}_results.json"
        reden = None
        cfg = res = None
        if not resp.exists():
            reden = "geen results.json"
        else:
            try:
                cfg = json.load(open(cfgp)); res = json.load(open(resp))
            except Exception:
                reden = "onleesbaar"
        if reden is None:
            st = str(cfg.get("study_type") or "diagnostic_psg").lower()
            summ = _summary(res); hypno = (res.get("staging") or {}).get("hypnogram") or []
            if "polygraph" in st or st in ("hpg", "polygraphy", "hsat"):
                reden = "polygrafie"
            elif str(cfg.get("split_night", "off")) != "off":
                reden = "split_night"
            elif not hypno:
                reden = "geen hypnogram"
            elif not summ:
                reden = "geen respiratoire samenvatting"
            elif not cfg.get("edf_path") or not os.path.exists(cfg["edf_path"]):
                reden = "EDF niet op schijf"
        if reden:
            uitgesloten[reden] = uitgesloten.get(reden, 0) + 1
            continue
        meta = (res.get("pneumo") or {}).get("meta") or {}
        sample.append({
            "R": f"L{len(sample) + 1:02d}", "job_id": jid, "bestandsnaam": os.path.basename(cfg["edf_path"]),
            "upload_dag": time.strftime("%Y-%m-%d", time.localtime(cfgp.stat().st_mtime)),
            "ahi_opgeslagen": summ.get("ahi_total"), "ahi_incl_uncertain_opgeslagen": summ.get("ahi_incl_uncertain"),
            "severity_opgeslagen": summ.get("severity") or summ.get("ahi_severity"),
            "rdi_opgeslagen": _zoek(summ, ("rdi", "rdi_total")),
            "profiel_opgeslagen": meta.get("scoring_profile") or cfg.get("scoring_profile"),
            "versie_opgeslagen": meta.get("psgscoring_version"), "study_type": cfg.get("study_type"),
            "n_epochs": len(hypno), "tst_machine_min": _tst_min(hypno),
            "tst_opgeslagen_min": _zoek(res.get("staging") or {}, ("TST", "tst", "tst_min", "total_sleep_time")),
            "eog_ch_gezet": bool(cfg.get("eog_ch")),
        })
    return {"gemaakt": time.strftime("%Y-%m-%d %H:%M"), "n_bekeken": bekeken, "uitgesloten": uitgesloten, "sample": sample}


def schrijf_sjabloon(mapping):
    pad = D / "humaan.csv"
    if pad.exists():
        print(f"{pad} bestaat al, niet overschreven")
        return
    with open(pad, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KOLOMMEN); w.writeheader()
        for it in mapping["sample"]:
            w.writerow({"L": it["R"], "job_kort": it["job_id"][:8], "bestandsnaam": it["bestandsnaam"], "upload_dag": it["upload_dag"],
                        "ahi_opgeslagen": it["ahi_opgeslagen"], "ahi_h": "", "n_apneu_h": "", "n_hypopneu_h": "", "tst_h_min": "", "regel_h": ""})
    os.chmod(pad, 0o600)
    print(f"sjabloon {pad} (600): vul ahi_h in, de rest is optioneel; regel_h = 3pct_of_arousal | 4pct | anders")


def cijfers():
    pad = D / "humaan.csv"
    uit = {}
    for rij in csv.DictReader(open(pad, newline="")):
        L = (rij.get("L") or "").strip()
        if not L:
            continue
        uit[L] = {"ahi_h": _getal(rij.get("ahi_h")), "n_apneu_h": _getal(rij.get("n_apneu_h")), "n_hypopneu_h": _getal(rij.get("n_hypopneu_h")),
                  "tst_h_min": _getal(rij.get("tst_h_min")), "regel_h": (rij.get("regel_h") or "").strip() or None}
    json.dump(uit, open(D / "humaan_cijfers.json", "w"), indent=1)
    n = sum(1 for v in uit.values() if v["ahi_h"] is not None)
    print(f"humaan_cijfers.json: {len(uit)} rijen, {n} met ahi_h (geen namen, geen job-id's)")


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    if "--cijfers" in sys.argv:
        cijfers(); sys.exit(0)
    mp = D / "mapping.json"
    if mp.exists():
        m = json.load(open(mp)); print(f"{mp} bestaat al ({len(m['sample'])} nachten), niet opnieuw getrokken")
    else:
        m = selecteer(); json.dump(m, open(mp, "w"), indent=1, default=str); os.chmod(mp, 0o600)
    print(f"bekeken {m['n_bekeken']}, uitgesloten {m['uitgesloten']}, geselecteerd {len(m['sample'])}")
    for it in m["sample"]:   # alleen getallen naar de terminal
        print(f"  {it['R']}: AHI opgeslagen {it['ahi_opgeslagen']} ({it['severity_opgeslagen']}), profiel {it['profiel_opgeslagen']}, psgscoring {it['versie_opgeslagen']}, epochs {it['n_epochs']}, TST {it['tst_machine_min']} min, EOG {it['eog_ch_gezet']}")
    schrijf_sjabloon(m)
