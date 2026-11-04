#!/usr/bin/env python
"""
compare_domino.py — één studie: de menselijke DOMINO-scoring naast psgscoring.

Leest de analyse-export van SOMNOmedics DOMINO (map, .zip of .xlsx; zie
``psgscoring.compare.domino_reader``) en de resultaten van psgscoring /
YASAFlaskified (``<job>_results.json``) van dezelfde opname, legt ze op één
tijdbasis en geeft een scorekaart: staging (κ), respiratoire events (F1 met
de koppelregel erbij), AHI op gelijke noemers, desaturaties, beenbewegingen,
arousals. Alleen getallen en labels komen eruit — geen naam, geen datum.

    PYTHONPATH=$PWD python scripts/compare_domino.py \\
        --export /pad/psg.zip --results /pad/<job>_results.json --json uit.json

DRIE DINGEN DIE JE MOET GELIJKTREKKEN, ANDERS VERGELIJK JE NIETS
----------------------------------------------------------------
1. DE TIJDBASIS. Nulpunt is de DOMINO-marker ``Start`` (``--t0`` om een andere
   kloktijd te nemen, ``--offset-s`` om te schuiven). Het script controleert
   het nulpunt op de SpO2-reeks (kruiscorrelatie, ``estimate_clock_offset``)
   en WAARSCHUWT als de klok meer dan ``--align-tolerance`` s afwijkt;
   met ``--auto-align`` past het de gevonden verschuiving toe. Zonder die
   controle rolt er een F1 uit die er echt uitziet en niets meet.
2. HET STADIUM EN DE NOEMER. De DOMINO-eventlijst bevat ook events in wake en
   vóór lights-off. Vergeleken wordt op de events in DOMINO-slaap, en de AHI
   wordt op BEIDE noemers gegeven (DOMINO-TST en psgscoring-TST). Het
   DOMINO-rapport telt zelf nog een hypopneucriterium over de lijst (een
   desaturatie binnen een venster), dus de rapport-AHI kan lager liggen dan
   "events in slaap / TST"; dat is een eigenschap van het rapport, niet van
   deze lijst.
3. DE KOPPELREGEL. Standaard IoU ≥ 0,20, één-op-één, symmetrisch
   (``psgscoring.agreement.compare_event_sets``); ``--matcher any`` is de
   elke-overlap-regel. Een F1 zonder zijn koppelregel is geen F1. Als
   ijkpunt: twaalf menselijke scoorders halen onderling F1 0,667 op PSG-IPA
   (330 paren, 07-10-2026) — rapporteer de machine-F1 daar nooit zonder.

Het staging-verschil is op de eerste kliniekexport (04-11-2026) de grootste
bron van gemiste events (YASA noemde 71 min méér wake); daarom staat de
F1 ook apart voor de epochs die BEIDE als slaap zien.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psgscoring.agreement import compare_event_sets  # noqa: E402
from psgscoring.compare.domino_reader import (  # noqa: E402
    SLEEP_STAGES, ahi_from_domino_events, estimate_clock_offset, hypnogram_kappa,
    read_domino_export,
)

PSGIPA_CEILING_F1 = 0.667   # 12 scoorders, 330 paren, respiratoire events, IoU 0,20


def _load_results(path: Path) -> dict:
    r = json.loads(path.read_text())
    block = r.get("pneumo") or r
    staging = r.get("staging") or {}
    hyp = staging.get("hypnogram") or r.get("hypnogram") or []
    resp = block.get("respiratory") or {}
    out = {
        "hypnogram": list(hyp),
        "events": list(resp.get("events") or []),
        "summary": resp.get("summary") or {},
        "spo2": block.get("spo2") or {},
        "plm": block.get("plm") or {},
        "arousal": block.get("arousal") or {},
    }
    meta = r.get("meta") or block.get("meta") or {}
    out["duration_s"] = (float(meta["duration_min"]) * 60.0 if meta.get("duration_min")
                         else len(out["hypnogram"]) * 30.0)
    if not out["hypnogram"]:
        raise SystemExit(f"{path}: geen hypnogram (staging.hypnogram) in de resultaten")
    return out


def _prf(c: dict) -> dict:
    """precisie/recall/F1 uit de telling van compare_event_sets.

    a = DOMINO (mens), b = psgscoring: ``n_shared`` zijn de gekoppelde paren,
    ``n_only_a`` de gemiste menselijke events, ``n_only_b`` de extra
    machine-events.
    """
    tp, fn, fp = c["n_shared"], c["n_only_a"], c["n_only_b"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r)) if p and r else (0.0 if (tp + fp + fn) else None)
    return {"n_domino": c["n_a"], "n_psgscoring": c["n_b"], "tp": tp, "fn": fn, "fp": fp,
            "precision": None if p is None else round(p, 3),
            "recall": None if r is None else round(r, 3),
            "f1": None if f1 is None else round(f1, 3),
            "median_iou": c.get("median_iou"), "n_type_changed": c.get("n_type_changed"),
            "type_changes": c.get("type_changes"), "kappa_1s": c.get("kappa_1s"),
            "onset_shift_median_s": c.get("onset_shift_median_s"),
            "matcher": c.get("matcher"), "iou_thresh": c.get("iou_thresh")}


def _stage_of(hyp: list[str], onset_s: float, epoch_s: float) -> str:
    i = int(onset_s // epoch_s)
    return hyp[i] if 0 <= i < len(hyp) else "?"


def _clock(t_abs: float) -> str:
    t = int(round(t_abs)) % 86400
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _idx(n: int, minutes: float) -> float | None:
    return round(n / (minutes / 60.0), 1) if minutes and minutes > 0 else None


def build_scorecard(export_path: Path, results_path: Path, *, t0=None, offset_s: float = 0.0,
                    auto_align: bool = False, align_tolerance_s: float = 2.0,
                    iou_thresh: float = 0.2, matcher: str = "iou", epoch_s: float = 30.0,
                    check_alignment: bool = True) -> dict:
    ex = read_domino_export(export_path)
    res = _load_results(results_path)
    hyp = res["hypnogram"]
    n_ep = len(hyp)
    t0_abs = ex.resolve_t0(t0) + float(offset_s)

    # 1. tijdbasis: controle op de SpO2-reeks
    align: dict = {"t0_clock": _clock(t0_abs), "offset_applied_s": float(offset_s),
                   "spo2_check": None, "warning": None}
    ts = res["spo2"].get("timeseries")
    if check_alignment and isinstance(ts, list) and ts and ex.has("SpO2"):
        dom = ex.series_1hz("SpO2", t0_abs, n_seconds=len(ts))
        try:
            lag, r, n = estimate_clock_offset(dom, ts)
            align["spo2_check"] = {"lag_s": lag, "r": r, "n_pairs": n}
            if abs(lag) > align_tolerance_s:
                if auto_align:
                    t0_abs += lag
                    align["offset_applied_s"] = float(offset_s) + lag
                    align["t0_clock"] = _clock(t0_abs)
                    align["warning"] = (f"klok week {lag:+d} s af; --auto-align heeft het "
                                        f"nulpunt verschoven naar {align['t0_clock']}")
                else:
                    align["warning"] = (f"klok wijkt {lag:+d} s af van het nulpunt (r {r}); "
                                        f"geef --offset-s {lag} of --auto-align mee — de "
                                        f"cijfers hieronder zijn op het ONGECORRIGEERDE nulpunt")
        except ValueError as e:
            align["spo2_check"] = {"error": str(e)}

    # 2. staging
    dom_ep = ex.hypnogram_epochs(t0_abs, n_epochs=n_ep, epoch_s=epoch_s)
    kap = hypnogram_kappa(dom_ep, hyp)
    tst_dom = ex.tst_minutes()
    tst_mach = res["summary"].get("tst_minutes") or res["summary"].get("sleep_minutes") \
        or round(sum(1 for s in hyp if s in SLEEP_STAGES) * epoch_s / 60.0, 1)
    both_sleep = [i for i in range(n_ep) if dom_ep[i] in SLEEP_STAGES and hyp[i] in SLEEP_STAGES]
    staging = {
        "kappa": kap["kappa"], "accuracy": kap["accuracy"], "n_epochs_compared": kap["n"],
        "n_epochs": n_ep, "epoch_s": epoch_s,
        "tst_min": {"domino": tst_dom, "psgscoring": tst_mach},
        "wake_epochs": {"domino": dom_ep.count("W"), "psgscoring": hyp.count("W")},
        "stage_counts": {"domino": dict(Counter(dom_ep)), "psgscoring": dict(Counter(hyp))},
        "n_epochs_both_sleep": len(both_sleep),
        "confusion_domino_rows_psgscoring_cols": kap["confusion"],
    }

    # 3. respiratoire events
    ev_all = ex.respiratory_events(t0_abs)
    ev_sleep = ex.events_in_sleep(ev_all)
    mach = res["events"]
    lo = ex.lights_off_s(t0_abs)
    n_before_lo = sum(1 for e in ev_all if lo is not None and e["onset_s"] < lo)
    both = set(both_sleep)
    ev_both = [e for e in ev_sleep if int(e["onset_s"] // epoch_s) in both]
    mach_both = [e for e in mach if int(float(e["onset_s"]) // epoch_s) in both]

    def cmp(a, b, m=matcher):
        return _prf(compare_event_sets(a, b, iou_thresh=iou_thresh, matcher=m,
                                       duration_s=res["duration_s"],
                                       label_a="domino", label_b="psgscoring"))

    resp = {
        "labels": ex.verify_label_mapping(),
        "n_events": {"domino_all": len(ev_all), "domino_sleep": len(ev_sleep),
                     "domino_wake": len(ev_all) - len(ev_sleep),
                     "domino_before_lights_off": n_before_lo, "psgscoring": len(mach)},
        "by_type": {"domino_sleep": dict(Counter(e["type"] for e in ev_sleep)),
                    "psgscoring": dict(Counter(e.get("type") for e in mach))},
        "ahi": {
            "domino_on_domino_tst": ahi_from_domino_events(ev_sleep, tst_dom / 60.0),
            "domino_on_psgscoring_tst": ahi_from_domino_events(ev_sleep, tst_mach / 60.0),
            "psgscoring_reported": res["summary"].get("ahi_total"),
            "psgscoring_on_domino_tst": ahi_from_domino_events(mach, tst_dom / 60.0),
            "tst_min": {"domino": tst_dom, "psgscoring": tst_mach},
        },
        "agreement_domino_sleep": cmp(ev_sleep, mach),
        "agreement_domino_sleep_any_overlap": cmp(ev_sleep, mach, "any"),
        "agreement_epochs_both_sleep": cmp(ev_both, mach_both),
        "agreement_all_domino_events": cmp(ev_all, mach),
        "missed_by_psgscoring_stage": None,
        "reference_ceiling_f1": {"value": PSGIPA_CEILING_F1,
                                 "note": "PSG-IPA, 12 scoorders, 330 paren, IoU 0,20"},
    }
    # waar liggen de gemiste menselijke events volgens de machine-staging?
    c = compare_event_sets(ev_sleep, mach, iou_thresh=iou_thresh, matcher=matcher,
                           label_a="domino", label_b="psgscoring")
    # compare_event_sets geeft geen paarlijst terug; benader de gemiste events
    # als die zonder enige overlap met een machine-event, per machine-stadium.
    def overlaps(e, lst):
        a0, a1 = e["onset_s"], e["onset_s"] + e["duration_s"]
        return any(min(a1, float(m["onset_s"]) + float(m["duration_s"])) - max(a0, float(m["onset_s"])) > 0
                   for m in lst)
    missed = [e for e in ev_sleep if not overlaps(e, mach)]
    resp["missed_by_psgscoring_stage"] = {
        "n_without_any_overlap": len(missed),
        "psgscoring_stage_at_onset": dict(Counter(_stage_of(hyp, e["onset_s"], epoch_s) for e in missed)),
        "n_shared_iou": c["n_shared"],
    }

    # 4. desaturaties
    d_all = ex.desaturations(t0_abs)
    d_sleep = ex.events_in_sleep(d_all)
    m_des = [dict(e, type="desat") for e in (res["spo2"].get("desaturations") or [])]
    desat = {
        "n": {"domino_all": len(d_all), "domino_sleep": len(d_sleep), "psgscoring": len(m_des)},
        "drop_pct_median": {"domino": _median([e["drop_pct"] for e in d_sleep if e["drop_pct"] is not None]),
                            "psgscoring": _median([e.get("drop_pct") for e in m_des if e.get("drop_pct") is not None])},
        "duration_median_s": {"domino": _median([e["duration_s"] for e in d_sleep]),
                              "psgscoring": _median([e.get("duration_s") for e in m_des if e.get("duration_s") is not None])},
        "odi": {"domino_on_domino_tst": _idx(len(d_sleep), tst_dom),
                "psgscoring_reported": (res["spo2"].get("summary") or {}).get("odi_3pct"),
                "psgscoring_on_domino_tst": _idx(len(m_des), tst_dom)},
        "agreement_any_overlap": _prf(compare_event_sets(
            [dict(e, type="desat") for e in d_sleep], m_des, matcher="any",
            duration_s=res["duration_s"], label_a="domino", label_b="psgscoring")),
    }

    # 5. beenbewegingen
    lm = ex.limb_events(t0_abs)
    lm_sleep = ex.events_in_sleep(lm)
    n_plm_dom = sum(1 for e in lm_sleep if e["is_plm"])
    plm_sum = res["plm"].get("summary") or {}
    limb = {
        "domino": {"n_lm_all": len(lm), "n_lm_sleep": len(lm_sleep), "n_plm_sleep": n_plm_dom,
                   "plm_index_on_domino_tst": _idx(n_plm_dom, tst_dom),
                   "by_type_sleep": dict(Counter(e["type"] for e in lm_sleep))},
        "psgscoring": {"n_lm_sleep": plm_sum.get("n_lm_sleep"), "n_plm": plm_sum.get("n_plm"),
                       "plm_index": plm_sum.get("plm_index")},
    }

    # 6. arousals
    ar = ex.arousals(t0_abs)
    ar_sleep = ex.events_in_sleep(ar)
    eeg = [a for a in ar_sleep if not a["autonomic"]]
    ar_sum = res["arousal"].get("summary") or {}
    m_ar = res["arousal"].get("events") or []
    arousal = {
        "domino": {"n_all": len(ar), "n_sleep": len(ar_sleep), "n_eeg_sleep": len(eeg),
                   "n_autonomic_sleep": len(ar_sleep) - len(eeg),
                   "eeg_index_on_domino_tst": _idx(len(eeg), tst_dom),
                   "kinds_eeg": dict(Counter(a["kind"] for a in eeg))},
        "psgscoring": {"n_events": len(m_ar), "arousal_index": ar_sum.get("arousal_index")},
        "agreement_eeg_any_overlap": _prf(compare_event_sets(
            [dict(a, type="arousal") for a in eeg], [dict(a, type="arousal") for a in m_ar],
            matcher="any", duration_s=res["duration_s"], label_a="domino", label_b="psgscoring"))
        if eeg and m_ar else None,
        "note": "DOMINO scoort ook autonome (pleth-)arousals; alleen de EEG-arousals zijn vergelijkbaar",
    }

    return {
        "export": {"source": ex.source, "n_signals": len(ex.signals),
                   "markers_s": ex.markers_s(t0_abs)},
        "alignment": align,
        "staging": staging,
        "respiratory": resp,
        "desaturations": desat,
        "limb": limb,
        "arousals": arousal,
    }


def _median(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 2)


def _fmt(sc: dict) -> str:
    L = []
    a = sc["alignment"]
    L.append(f"nulpunt {a['t0_clock']} (offset {a['offset_applied_s']:+.0f} s); "
             f"SpO2-controle: {a['spo2_check']}")
    if a["warning"]:
        L.append(f"  !! {a['warning']}")
    m = sc["export"]["markers_s"]
    L.append(f"markers (s t.o.v. nulpunt): {m}")
    s = sc["staging"]
    L.append(f"staging: κ {s['kappa']} acc {s['accuracy']} op {s['n_epochs_compared']} epochs; "
             f"TST DOMINO {s['tst_min']['domino']} min, psgscoring {s['tst_min']['psgscoring']} min; "
             f"wake-epochs {s['wake_epochs']}; beide slaap {s['n_epochs_both_sleep']}")
    r = sc["respiratory"]
    n = r["n_events"]
    L.append(f"events: DOMINO {n['domino_all']} (slaap {n['domino_sleep']}, wake {n['domino_wake']}, "
             f"vóór lights-off {n['domino_before_lights_off']}) | psgscoring {n['psgscoring']}")
    if r["labels"]["unmapped"]:
        L.append(f"  !! onbekende DOMINO-labels, NIET geteld: {r['labels']['unmapped']}")
    L.append(f"  per type: DOMINO-slaap {r['by_type']['domino_sleep']} | psgscoring {r['by_type']['psgscoring']}")
    ah = r["ahi"]
    L.append(f"AHI: DOMINO {ah['domino_on_domino_tst']} (eigen TST) / {ah['domino_on_psgscoring_tst']} (machine-TST) | "
             f"psgscoring {ah['psgscoring_reported']} (gerapporteerd) / {ah['psgscoring_on_domino_tst']} (DOMINO-TST)")
    for key, naam in (("agreement_domino_sleep", "koppeling, events in DOMINO-slaap"),
                      ("agreement_domino_sleep_any_overlap", "  idem, elke overlap"),
                      ("agreement_epochs_both_sleep", "  epochs die BEIDE slaap noemen"),
                      ("agreement_all_domino_events", "  alle DOMINO-events (ook wake)")):
        g = r[key]
        L.append(f"{naam}: F1 {g['f1']} (P {g['precision']} R {g['recall']}; TP {g['tp']} FN {g['fn']} FP {g['fp']}; "
                 f"{g['matcher']} {g['iou_thresh']}; mediane IoU {g['median_iou']}; typewissels {g['n_type_changed']})")
    L.append(f"  ijkpunt: menselijke scoorders onderling F1 {r['reference_ceiling_f1']['value']} "
             f"({r['reference_ceiling_f1']['note']})")
    L.append(f"  gemiste DOMINO-events zonder enige overlap: {r['missed_by_psgscoring_stage']['n_without_any_overlap']}, "
             f"machine-stadium bij onset {r['missed_by_psgscoring_stage']['psgscoring_stage_at_onset']}")
    d = sc["desaturations"]
    L.append(f"desaturaties: n {d['n']} | ODI {d['odi']} | duur mediaan {d['duration_median_s']} | "
             f"koppeling (elke overlap) F1 {d['agreement_any_overlap']['f1']}")
    li = sc["limb"]
    L.append(f"beenbewegingen: DOMINO PLM in slaap {li['domino']['n_plm_sleep']} (index {li['domino']['plm_index_on_domino_tst']}), "
             f"psgscoring n_plm {li['psgscoring']['n_plm']} (index {li['psgscoring']['plm_index']})")
    ar = sc["arousals"]
    g = ar["agreement_eeg_any_overlap"]
    L.append(f"arousals: DOMINO EEG in slaap {ar['domino']['n_eeg_sleep']} (index {ar['domino']['eeg_index_on_domino_tst']}; "
             f"autonoom {ar['domino']['n_autonomic_sleep']}), psgscoring {ar['psgscoring']['n_events']} "
             f"(index {ar['psgscoring']['arousal_index']})" + (f"; koppeling F1 {g['f1']}" if g else ""))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, type=Path, help="DOMINO-export: map, .zip of .xlsx")
    ap.add_argument("--results", required=True, type=Path, help="<job>_results.json van psgscoring/YASAFlaskified")
    ap.add_argument("--t0", default=None, help="nulpunt als kloktijd HH:MM:SS (default: marker Start)")
    ap.add_argument("--offset-s", type=float, default=0.0, help="seconden bij het nulpunt op te tellen")
    ap.add_argument("--auto-align", action="store_true", help="pas de op SpO2 gevonden verschuiving toe")
    ap.add_argument("--align-tolerance", type=float, default=2.0)
    ap.add_argument("--no-align-check", action="store_true")
    ap.add_argument("--iou", type=float, default=0.2)
    ap.add_argument("--matcher", default="iou", choices=("iou", "any"))
    ap.add_argument("--epoch-s", type=float, default=30.0)
    ap.add_argument("--json", type=Path, default=None, help="scorekaart als JSON wegschrijven")
    args = ap.parse_args(argv)

    sc = build_scorecard(args.export, args.results, t0=args.t0, offset_s=args.offset_s,
                         auto_align=args.auto_align, align_tolerance_s=args.align_tolerance,
                         iou_thresh=args.iou, matcher=args.matcher, epoch_s=args.epoch_s,
                         check_alignment=not args.no_align_check)
    print(_fmt(sc))
    if args.json:
        args.json.write_text(json.dumps(sc, indent=2, ensure_ascii=False))
        print(f"scorekaart: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
