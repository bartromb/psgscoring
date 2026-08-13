#!/usr/bin/env python
"""
compare_caisr.py — psgscoring naast CAISR-resp op dezelfde opnames.

Beide zijn regelgebaseerde respiratoire scorers. Dit script zet ze naast
elkaar en, waar een referentie meegegeven wordt, allebei tegen diezelfde
referentie. Het draait CAISR NIET: het leest de CSV's die CAISR zelf heeft
weggeschreven. CAISR-App staat onder CC BY-NC 4.0 en wordt niet
meegeleverd; installeer en draai het zelf, en alleen niet-commercieel.

    # 1. CAISR draaien (buiten psgscoring, eigen omgeving/container)
    python caisr_resp.py --input_dir /data/edf --output_csv_dir /out/caisr

    # 2. vergelijken
    PYTHONPATH=$PWD python scripts/compare_caisr.py \
        --psg-json psgipa.json \
        --caisr-dir /out/caisr/resp \
        --out caisr_vs_psgscoring.csv

DRIE DINGEN DIE JE MOET GELIJKTREKKEN, ANDERS VERGELIJK JE NIETS
----------------------------------------------------------------
1. DE NOEMER. CAISR's CSV bevat geen hypnogram en dus geen slaaptijd. Dit
   script neemt de slaaptijd van de psgscoring-kant en gebruikt die voor
   BEIDE indices. Verschillende noemers geven een AHI-verschil dat niets
   met respiratoire scoring te maken heeft — precies de val die in het
   supplement al beschreven staat voor de PSG-IPA-harnas.

2. DE TIJDBASIS. CAISR levert een labelvector van 1 Hz. Eventgrenzen zijn
   daardoor op een hele seconde afgerond en twee aangrenzende events van
   hetzelfde type zijn niet te scheiden. Dat drukt event-F1 aan beide
   kanten. Het effect wordt hier gerapporteerd als `caisr_runs_merged`, het
   aantal CAISR-events dat aan een ander event van hetzelfde type grenst.

3. DE REGELSET. CAISR's hypopneutak 4 is het 3%-of-arousal-criterium, tak 6
   het 4%-criterium. Vergelijk alleen tegen een referentie die dezelfde
   regel implementeert; een Regel 1A-profiel tegen een 4%-referentie geeft
   error-richtingen die niet interpreteerbaar zijn.

RERA's tellen niet mee in de AHI aan beide kanten; ze worden apart geteld.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from psgscoring.compare.caisr_reader import (  # noqa: E402
    ahi_from_events, labels_to_events, read_caisr_resp_csv, verify_code_mapping,
)
from validate_psgipa import (  # noqa: E402
    MATCHER_PRESETS, match_events_symmetric, type_family,
)


def _psg_events(rec: dict) -> list[dict]:
    """Events uit een psgscoring-resultaatblok, in matcher-vorm."""
    return list((rec.get("respiratory") or rec).get("events") or [])


def _to_triples(events) -> list[tuple]:
    """(onset, offset, type) — wat match_events verwacht."""
    out = []
    for e in events:
        try:
            o = float(e["onset_s"])
            d = float(e["duration_s"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append((o, o + d, e.get("type")))
    return out


def _n_adjacent_same_type(events) -> int:
    """Hoeveel events grenzen aan een gelijksoortig event.

    Op een 1 Hz-raster is "grenst aan" niet te onderscheiden van "is
    hetzelfde event". Dit getal begrenst hoeveel van het F1-verschil aan het
    formaat toegeschreven kan worden in plaats van aan het algoritme.
    """
    ev = sorted(events, key=lambda e: float(e.get("onset_s", 0.0)))
    n = 0
    for a, b in zip(ev, ev[1:]):
        end_a = float(a.get("onset_s", 0)) + float(a.get("duration_s", 0))
        if (abs(float(b.get("onset_s", 0)) - end_a) <= 1.0
                and type_family(a.get("type")) == type_family(b.get("type"))):
            n += 1
    return n


def _f1(ev_a, ev_b, matcher: dict) -> dict:
    """Neem de maten die `match_events_symmetric` zelf al berekent.

    Een eerdere versie zocht `m["n_matched"]` met een `m.get("matched", 0)` als
    terugval en rekende precisie en recall daarna zelf uit. De matcher levert
    die sleutel niet — hij heet `tp_ab` — dus die terugval gaf STIL nul, en
    daarmee F1 = 0,000 op elke opname. Dat zag eruit als "de twee systemen zijn
    het nergens eens" terwijl SN1 in werkelijkheid 25 overeenkomsten had. De
    fout bleef staan omdat CAISR tot 13-08-2026 nooit gedraaid had.

    Daarom nu: geen eigen herberekening, en een harde KeyError als de matcher
    van vorm verandert.
    """
    a, b = _to_triples(ev_a), _to_triples(ev_b)
    if not a and not b:
        return {"f1": None, "precision": None, "recall": None,
                "n_a": 0, "n_b": 0, "n_matched": 0}
    m = match_events_symmetric(a, b, **matcher)
    ontbreekt = {"f1", "precision_ab", "recall_ab", "tp_ab"} - set(m)
    if ontbreekt:
        raise KeyError(
            f"match_events_symmetric mist {sorted(ontbreekt)}; de vorm is "
            f"gewijzigd. Stil doorrekenen zou nullen opleveren die op een "
            f"meting lijken. Geleverd: {sorted(m)}")
    return {"f1": round(float(m["f1"]), 3),
            "precision": round(float(m["precision_ab"]), 3),
            "recall": round(float(m["recall_ab"]), 3),
            "n_a": len(a), "n_b": len(b), "n_matched": int(m["tp_ab"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psg-json", required=True, type=Path,
                    help="uitvoer van profile_comparison/validate_mesa: "
                         "{recording: {respiratory: {...}}}")
    ap.add_argument("--caisr-dir", required=True, type=Path,
                    help="map met CAISR-resp CSV's, één per opname")
    ap.add_argument("--reference-json", type=Path, default=None,
                    help="optioneel: {recording: {events: [...], "
                         "sleep_hours: float}}")
    ap.add_argument("--matcher", default="legacy",
                    choices=sorted(MATCHER_PRESETS))
    ap.add_argument("--out", type=Path, default=Path("caisr_vs_psgscoring.csv"))
    args = ap.parse_args(argv)

    matcher = MATCHER_PRESETS[args.matcher]
    psg = json.loads(args.psg_json.read_text())
    ref = json.loads(args.reference_json.read_text()) if args.reference_json else {}

    rows, seen_codes = [], {}
    for name, rec in sorted(psg.items()):
        csv_path = args.caisr_dir / f"{name}.csv"
        if not csv_path.exists():
            hits = list(args.caisr_dir.glob(f"*{name}*.csv"))
            if not hits:
                print(f"  overslaan {name}: geen CAISR-CSV", file=sys.stderr)
                continue
            csv_path = hits[0]

        labels, spr = read_caisr_resp_csv(csv_path)
        for code, n in verify_code_mapping(labels)["row_counts_by_code"].items():
            seen_codes[code] = seen_codes.get(code, 0) + n
        caisr_ev = labels_to_events(labels, spr)
        psg_ev = _psg_events(rec)

        # Eén noemer voor beide indices — zie kop van dit bestand.
        summary = (rec.get("respiratory") or rec).get("summary") or {}
        sleep_h = summary.get("index_denominator_h")
        if sleep_h is None and summary.get("ahi_total") and psg_ev:
            sleep_h = None  # liever niets dan een teruggerekende noemer

        row = {
            "recording": name,
            "psg_ahi": summary.get("ahi_total"),
            "caisr_ahi": ahi_from_events(caisr_ev, sleep_h) if sleep_h else None,
            "sleep_h": sleep_h,
            "psg_n_events": len(psg_ev),
            "caisr_n_events": len(caisr_ev),
            "caisr_n_rera": sum(1 for e in caisr_ev if e["caisr_code"] == 5),
            "caisr_runs_merged": _n_adjacent_same_type(caisr_ev),
        }
        row.update({f"pair_{k}": v for k, v in
                    _f1(psg_ev, caisr_ev, matcher).items()})
        if name in ref:
            r_ev = ref[name].get("events") or []
            row.update({f"psg_vs_ref_{k}": v for k, v in
                        _f1(psg_ev, r_ev, matcher).items()})
            row.update({f"caisr_vs_ref_{k}": v for k, v in
                        _f1(caisr_ev, r_ev, matcher).items()})
        rows.append(row)

    if not rows:
        print("geen overlappende opnames gevonden", file=sys.stderr)
        return 1

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({
        "n_recordings": len(rows),
        "matcher": args.matcher,
        "caisr_code_totals": dict(sorted(seen_codes.items())),
        "note": "codes: 1 OA, 2 CA, 3 MA, 4 hyp(3%), 5 RERA, 6 hyp(4%)",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
