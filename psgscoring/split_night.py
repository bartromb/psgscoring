"""Splitst een nacht in een diagnostisch deel en een deel onder therapie.

WAAROM
------
Bij een split-night wordt de eerste helft diagnostisch geregistreerd en gaat de
patiënt daarna aan CPAP. Één AHI over die hele nacht verdunt de diagnose: op de
casus die dit aanleiding gaf (Chulalongkorn, 26-08-2026) las het rapport
"Mild SAS, AHI 10,1/u" bij een patiënt die de verwijzer als ernstig kende.

WAAR HET BREEKPUNT VANDAAN KOMT -- EN WAAROM NIET UIT DE ODI
------------------------------------------------------------
Het voor de hand liggende criterium is de desaturatiedichtheid. Gemeten op die
opname zet dat het breekpunt **2,5 uur te laat**:

    ODI3 per 30 min: 66 78 52 54 44 60 34 8 42 | 6 6 6 12 0 6

De ODI zakt pas na 4:30 blijvend. Maar twee andere sporen markeren de ingreep
op 2:00, en scherper:

    flow p95        0,43-0,63  ->  0,07-0,24     (factor 3 a 4 OMLAAG)
    mediane SpO2    93 -> 83 %  ->  95-96 %      (basislijn herstelt)

Tussen 2:00 en 4:30 loopt de patiënt dus al aan CPAP terwijl de titratie nog
niet effectief is; de ODI is daar hoog. Een detector op ODI rekent die 2,5 uur
bij het diagnostische deel en verdunt precies wat hij moest scheiden.

De verwachting dat CPAP-flow "glad en groot van amplitude" is, klopt op deze
montage NIET: de gladheid verschilt niet (0,940 tegen 0,949) en de amplitude
daalt, vermoedelijk doordat de neuscanule voor een masker wijkt. Daarom telt
de GROOTTE van de stap, niet zijn richting.

WAT DEZE MODULE NIET DOET
-------------------------
Ze verandert de AHI niet. De nacht-AHI blijft staan zoals AASM hem voorschrijft;
de segmentindices komen ernaast. Een detector die stilzwijgend de hoofdindex
herdefinieert, zou een tweede soort fout introduceren om de eerste op te lossen.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: Vensterduur (s) waarop flow en saturatie worden samengevat.
WINDOW_S = 300.0

#: Beide segmenten moeten minstens zo lang zijn. Korter dan een uur is geen
#: diagnostisch deel en geen titratie, maar een signaalstoring.
MIN_SEGMENT_S = 3600.0

#: De flowamplitude moet minstens zoveel keer verschillen tussen de segmenten.
MIN_FLOW_RATIO = 2.0

#: En de saturatiebasislijn moet minstens zoveel procentpunt stijgen.
MIN_SPO2_RISE = 2.0


def _vensters(x, sf, window_s=WINDOW_S):
    n = int(window_s * sf)
    if n < 2 or len(x) < 2 * n:
        return np.array([]), np.array([])
    k = len(x) // n
    start = (np.arange(k) * n / sf)
    return start, x[: k * n].reshape(k, n)


def _flow_amplitude(flow, sf):
    """p95 van |flow − mediaan| per venster: hoe groot ademt het kanaal?"""
    _t, blok = _vensters(flow, sf)
    if not len(blok):
        return np.array([])
    med = np.median(blok, axis=1, keepdims=True)
    return np.percentile(np.abs(blok - med), 95, axis=1)


def _stil_nanmediaan(x):
    """nanmedian zonder waarschuwing op een volledig lege reeks."""
    x = np.asarray(x, dtype=float)
    geldig = x[~np.isnan(x)]
    return float(np.median(geldig)) if geldig.size else float("nan")


def _spo2_baseline(spo2, sf):
    """Mediane saturatie per venster, met onmogelijke waarden eruit."""
    _t, blok = _vensters(spo2, sf)
    if not len(blok):
        return np.array([])
    b = np.where((blok >= 50) & (blok <= 100), blok, np.nan)
    return np.array([_stil_nanmediaan(rij) for rij in b])


def detect_split_night(flow=None, sf_flow=None, spo2=None, sf_spo2=None,
                       manual_breakpoint_s=None,
                       min_segment_s=MIN_SEGMENT_S,
                       min_flow_ratio=MIN_FLOW_RATIO,
                       min_spo2_rise=MIN_SPO2_RISE) -> dict:
    """Zoek het moment waarop de therapie begon.

    Een handmatig opgegeven breekpunt wint altijd: wie weet hoe laat de CPAP
    aanging, weet het beter dan een detector.
    """
    uit = {"detected": False, "breakpoint_s": None, "method": None,
           "evidence": {}, "reason": None}

    if manual_breakpoint_s is not None:
        uit.update(detected=True, breakpoint_s=float(manual_breakpoint_s),
                   method="manual")
        return uit

    amp = _flow_amplitude(np.asarray(flow, dtype=float), sf_flow) \
        if flow is not None and sf_flow else np.array([])
    sat = _spo2_baseline(np.asarray(spo2, dtype=float), sf_spo2) \
        if spo2 is not None and sf_spo2 else np.array([])
    n = max(len(amp), len(sat))
    if n < 4:
        uit["reason"] = "te weinig signaal om een breekpunt te zoeken"
        return uit
    marge = max(1, int(round(min_segment_s / WINDOW_S)))
    if n - 2 * marge < 1:
        uit["reason"] = "opname te kort voor twee segmenten"
        return uit

    beste = None
    for k in range(marge, n - marge + 1):
        score, bewijs = 0.0, {}
        if len(amp) == n:
            v, na = np.median(amp[:k]), np.median(amp[k:])
            ratio = (max(v, na) / min(v, na)) if min(v, na) > 0 else 0.0
            bewijs["flow_ratio"] = float(ratio)
            bewijs["flow_before"] = float(v); bewijs["flow_after"] = float(na)
            score += min(ratio / min_flow_ratio, 3.0)
        if len(sat) == n:
            # Een venster kan volledig ongeldig zijn (sensor los). nanmedian
            # geeft dan NaN mét een waarschuwing; die stilte hier is bewust:
            # ontbrekende saturatie is geen bewijs vóór en geen bewijs tegen,
            # en moet dus als "geen stijging" gelden in plaats van als fout.
            v, na = _stil_nanmediaan(sat[:k]), _stil_nanmediaan(sat[k:])
            stijging = float(na - v) if not (np.isnan(v) or np.isnan(na)) else 0.0
            bewijs["spo2_rise"] = stijging
            bewijs["spo2_before"] = float(v) if not np.isnan(v) else None
            bewijs["spo2_after"] = float(na) if not np.isnan(na) else None
            score += min(max(stijging, 0.0) / min_spo2_rise, 3.0)
        if beste is None or score > beste[0]:
            beste = (score, k, bewijs)

    score, k, bewijs = beste
    uit["evidence"] = bewijs
    genoeg_flow = bewijs.get("flow_ratio", 0.0) >= min_flow_ratio
    genoeg_sat = bewijs.get("spo2_rise", 0.0) >= min_spo2_rise
    # BEIDE sporen moeten meedoen. Eén ervan alleen komt te vaak voor: een
    # canule die verschuift geeft een amplitudestap zonder klinische betekenis,
    # en een saturatiebasislijn stijgt ook als de patiënt van rug naar zij gaat.
    if genoeg_flow and genoeg_sat:
        uit.update(detected=True, breakpoint_s=float(k * WINDOW_S),
                   method="flow_amplitude+spo2_baseline")
    else:
        ontbreekt = []
        if not genoeg_flow: ontbreekt.append("flowamplitudestap")
        if not genoeg_sat: ontbreekt.append("saturatiestijging")
        uit["reason"] = "geen " + " en geen ".join(ontbreekt)
    return uit


def segment_indices(events, hypno, breakpoint_s, epoch_len_s=30.0,
                    artifact_epochs=None) -> dict:
    """AHI vóór en ná het breekpunt, in de compacte vorm die het rapport leest.

    Dit is een DUNNE LAAG op `segment_summaries()`, niet een tweede berekening.
    Tot 0.29.0 stonden hier twee implementaties naast elkaar die dezelfde
    grootheid uitrekenden; ze kwamen numeriek overeen toen ik het naliep, maar
    dat is een momentopname en geen eigenschap. Twee implementaties van
    hetzelfde getal is precies waar de stadium-AHI-reparatie over ging.

    Geeft per segment: slaaptijd, de eventtellingen, de twee AHI-varianten, of
    er genoeg slaap onder ligt, en welk deel van de events niet getypeerd kon
    worden.
    """
    sam = segment_summaries(events, hypno, breakpoint_s, epoch_len_s,
                            artifact_epochs)
    uit = {}
    for naam, s in sam.items():
        if "error" in s:
            uit[naam] = {"error": s["error"]}
            continue
        h = s.get("index_denominator_h") or 0.0
        n_ah = s.get("n_ah_total") or 0
        n_unc = s.get("n_uncertain_apnea") or 0
        uit[naam] = {
            "sleep_h": round(float(h), 3),
            "n_events": int(n_ah),
            "n_uncertain": int(n_unc),
            "ahi": s.get("ahi_total"),
            "ahi_incl_uncertain": s.get("ahi_incl_uncertain"),
            # Een half uur slaap draagt geen index: één event is dan al 2/u.
            # Hetzelfde onderscheid als `ahi_rem_reliable`.
            "reliable": bool(h >= 0.5),
            # Bij een falende effort-band is `ahi` een onvolledige telling en
            # `ahi_incl_uncertain` het eerlijke getal.
            "uncertain_fraction": s.get("uncertain_fraction") or 0.0,
        }
    return uit

def _slice(events, hypno, lo_s, hi_s, epoch_len_s=30.0, artifact_epochs=None):
    """De events, het hypnogram en de artefactlijst van één segment.

    Het hypnogram wordt niet ingekort maar BUITEN het venster op wake gezet.
    Dat houdt de epoch-indices gelijk aan die van de hele nacht, zodat elke
    index die op epochnummers steunt -- de artefactlijst, de stadiumtellingen --
    blijft kloppen. Inkorten zou alles verschuiven en stil verkeerde stadia
    opleveren.
    """
    art = set(artifact_epochs or [])
    lo_ep, hi_ep = lo_s / epoch_len_s, hi_s / epoch_len_s
    hyp = [(st if lo_ep <= i < hi_ep else "W") for i, st in enumerate(hypno)]
    ev = [e for e in events
          if lo_s <= float(e.get("onset_s", 0)) < hi_s]
    return ev, hyp, sorted(art)


def segment_summaries(events, hypno, breakpoint_s, epoch_len_s=30.0,
                      artifact_epochs=None) -> dict:
    """De VOLLEDIGE indexfamilie per segment, niet alleen de AHI.

    Bij een split-night is het diagnostische deel de meting waarop de diagnose
    rust; dan moet daar ook alles bij horen wat een diagnose draagt -- OAHI,
    stadium-AHI's, de uncertain-boekhouding -- en niet één los getal naast een
    nacht-AHI die iets anders telt.

    Hergebruikt `_compute_summary`, dus elke regel die daar geldt (welke events
    in `ahi_total` tellen, hoe stadium-AHI's dezelfde eventset gebruiken, hoe
    een lege noemer `None` oplevert in plaats van nul) geldt hier ook. Een
    tweede implementatie zou precies de tegenspraak opleveren die de
    stadium-AHI-reparatie moest wegnemen.
    """
    from .respiratory import _compute_summary

    n_s = len(hypno) * epoch_len_s
    uit = {}
    for naam, lo, hi in (("diagnostic", 0.0, float(breakpoint_s)),
                         ("therapeutic", float(breakpoint_s), n_s)):
        ev, hyp, art = _slice(events, hypno, lo, hi, epoch_len_s, artifact_epochs)
        try:
            uit[naam] = _compute_summary(ev, hyp, art)
        except Exception as e:                              # noqa: BLE001
            logger.warning("[split] samenvatting voor %s mislukt: %s", naam, e)
            uit[naam] = {"error": str(e)}
    return uit


def segment_spo2(spo2, sf_spo2, hypno, breakpoint_s, epoch_len_s=30.0,
                 artifact_epochs=None) -> dict:
    """ODI, T90 en de rest per segment, op hetzelfde venster als de AHI.

    Zonder dit zou de kop een diagnostische AHI tonen naast een ODI over de
    hele nacht -- twee getallen over verschillende stukken slaap, naast elkaar
    gepresenteerd alsof ze bij elkaar horen.
    """
    from .spo2 import analyze_spo2

    if spo2 is None or not sf_spo2:
        return {}
    n_s = len(hypno) * epoch_len_s
    uit = {}
    for naam, lo, hi in (("diagnostic", 0.0, float(breakpoint_s)),
                         ("therapeutic", float(breakpoint_s), n_s)):
        _ev, hyp, _art = _slice([], hypno, lo, hi, epoch_len_s, artifact_epochs)
        try:
            uit[naam] = (analyze_spo2(spo2, sf_spo2, hyp) or {}).get("summary") or {}
        except Exception as e:                              # noqa: BLE001
            logger.warning("[split] SpO2 voor %s mislukt: %s", naam, e)
            uit[naam] = {"error": str(e)}
    return uit
