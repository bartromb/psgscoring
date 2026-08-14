"""
psgscoring.breath_scoring — hypopnee-scoring met de ademteug als atoom.

Waarom een tweede detector naast de bestaande
----------------------------------------------
De bestaande pijplijn is een signaalverwerkingsmodel: envelope -> glijdende
drempel -> kandidaat -> valideer -> classificeer, met een reeks correcties
erbovenop. Een scorer doet iets anders: die vormt een beeld van wat bij DEZE
patiënt op DIT moment normaal ademen is, herkent een patroon, en gebruikt
saturatie en arousal als bevestiging — niet als primaire trigger.

Vrijwel elke correctie in de bestaande pijplijn repareert dat verschil.
Deze module vervangt de mismatch in plaats van hem te patchen, langs vijf
verschuivingen:

1. **De ademteug is het atoom.** De AASM spreekt van *peak signal
   excursions* — ademteugen, geen samples. Elke ademteug krijgt één
   amplitude t.o.v. de pre-event baseline; een event is een aaneengesloten
   reeks ademteugen onder de drempel; de duur is de som van ademteugduren.
   Eventgrenzen vallen daardoor op ademtransities, zoals bij mensen, en een
   herstelademteug breekt de reeks vanzelf — smoothing is niet meer nodig.

2. **Kalibratie op de patiënt, in twee passages.** Passage 1 detecteert
   alleen de onbetwistbare events en leidt daaruit een sjabloon af: typische
   duur, typische diepte, en — cruciaal — de EIGEN SpO2-vertraging van deze
   patiënt via kruiscorrelatie. Die hangt af van circulatietijd en
   probeplaats en verschilt per patiënt; een vast venster van 30-45 s voor
   iedereen is een benadering. Passage 2 beoordeelt de marginale kandidaten
   tegen dat sjabloon.

3. **Gegradeerde AASM-predicaten in plaats van harde drempels.** De regel
   blijft letterlijk de AASM-conjunctie — >=30% daling EN >=10 s EN (>=3%
   desaturatie OF arousal) — maar elk predicaat levert een gegradeerde
   waarde. Een daling van 29% met 6% desaturatie hoort te scoren; 31% met
   een twijfelachtige 3,0% niet. Elk event draagt bij hoeveel elk criterium
   bijdroeg, dus het audittrail wordt rijker, niet ondoorzichtiger.

4. **De uitkomst is een graad, geen booleaan.** Er is geen enkele juiste
   scoring: op PSG-IPA SN2 splitsten twaalf scorers 8/4. Elk event krijgt
   ``p_scored``: hoe goed het aan de AASM-conjunctie voldoet, op een schaal
   van 0 tot 1.

   **``p_scored`` is GEEN kans dat een scorer dit zou markeren, en mag niet
   zo gelezen worden.** Gemeten op PSG-IPA (163 gescoorde events, 12 scorers
   per opname, de scorerfractie per event letterlijk geteld):

   ===========================  ======
   p_scored mediaan             0,693
   werkelijke scorerfractie     0,167
   correlatie r                 0,194
   systematische afwijking      +0,333
   ===========================  ======

   De ORDENING klopt zwak — hogere ``p_scored`` gaat samen met een hogere
   scorerfractie (band 0,50-0,60 -> 0,32; band 0,90+ -> 0,58) — maar het
   NIVEAU ligt ruim 30 procentpunt te hoog. Gebruik het dus om events
   onderling te rangschikken, niet als waarschijnlijkheid.

   Kalibratie naar een echte kans is mogelijk: PSG-IPA levert de doelwaarde
   per event. Dat vraagt een gefitte afbeelding op meer dan vijf opnames en
   is hier niet gedaan.

5. **Strengheid is één as.** In plaats van drie parametercombinaties is er
   één drempel: ``strictness``. Lager = soepeler.

   **Die belofte is maar deels waargemaakt, en dat hoort hier te staan.**
   Een gevoeligheidsanalyse op PSG-IPA SN1 en SN4 laat zien dat vijf andere
   parameters het aantal gescoorde events even sterk verschuiven, over
   plausibele bereiken:

   ======================  ===============  ===============
   parameter               SN1              SN4
   ======================  ===============  ===============
   ``flow_width``          +6% .. -39%      +27% .. -68%
   ``recovery_margin``     -6% .. +42%      -27% .. +68%
   ``dur_width``           +10% .. -39%     +14% .. -50%
   ``stability_cv``        +3% .. -35%      +23% .. -36%
   ``use_template=False``  +10%             +36%
   ======================  ===============  ===============

   Alleen ``strictness`` is gekalibreerd (op PSG-IPA, met een vooraf
   vastgelegde bias-nul-regel). De rest is gekozen en niet gevalideerd. Er
   zijn dus zes assen, geen een — en wie aan de andere vijf draait,
   verlaat het gevalideerde werkpunt zonder dat iets dat signaleert.

AASM-conformiteit blijft behouden: de regelstructuur is letterlijk die van
Rule 1A, en bij afkapping op de klassieke drempels nadert het gedrag de
regelgebaseerde uitkomst. Wat verandert is dat de tolerantie ROND elke
drempel gekozen wordt in plaats van oneindig scherp te zijn.

Scope: alleen hypopneeën. Apneus worden door de bestaande detector gescoord
(F1 0,83-0,93 op PSG-IPA, met effort-gebaseerde subtypering) — daar is geen
tekort dat dit zou moeten oplossen.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "score_hypopneas_breathwise",
    "estimate_spo2_lag",
    "graded",
]


# ═══════════════════════════════════════════════════════════════
#  Gegradeerde predicaten
# ═══════════════════════════════════════════════════════════════

def graded(x: float, center: float, width: float) -> float:
    """Zachte stap: 0,5 op ``center``, oplopend over ``width``.

    Vervangt een harde drempel door een tolerantie eromheen. ``width`` is de
    breedte waarover het predicaat van ~0,12 naar ~0,88 loopt; hoe kleiner,
    hoe dichter bij het binaire gedrag van de regel.
    """
    if width <= 0:
        return 1.0 if x >= center else 0.0
    z = (float(x) - float(center)) / float(width)
    return float(1.0 / (1.0 + np.exp(-z)))


# ═══════════════════════════════════════════════════════════════
#  Ademteugreeks en pre-event baseline
# ═══════════════════════════════════════════════════════════════

def _series(breaths):
    """(onsets, ends, amplitudes) als arrays, enkel bruikbare ademteugen."""
    on, en, am = [], [], []
    for b in breaths or []:
        o, d, a = b.get("onset_s"), b.get("duration_s"), b.get("amplitude")
        if o is None or d is None or a is None:
            continue
        if not (np.isfinite(a) and a > 0):
            continue
        on.append(float(o)); en.append(float(o) + float(d)); am.append(float(a))
    return np.asarray(on), np.asarray(en), np.asarray(am)


def _pre_baseline_per_breath(onsets, amps, window_s=120.0,
                             stability_cv=0.25, n_largest=3, min_breaths=4,
                             exclude=None, robust=False):
    """Pre-event baseline vóór ELKE ademteug.

    Stabiele ademhaling (CV < ``stability_cv``) -> gemiddelde amplitude;
    anders het gemiddelde van de ``n_largest`` grootste. Dat is de
    operationalisering die de AASM aangeeft wanneer stabiele ademhaling niet
    te bepalen is. NaN waar er te weinig voorgeschiedenis is (begin van de
    opname, na een gap) — de aanroeper slaat die ademteugen over in plaats
    van een baseline te verzinnen.

    ``exclude`` is een booleaans masker over de ademteugen: gemarkeerde
    ademteugen tellen niet mee in het venster. Zo wordt "stabiele
    ademhaling" ook werkelijk stabiele ademhaling in plaats van "alles wat
    in de afgelopen twee minuten gebeurde, inclusief de events". Het masker
    wordt genegeerd wanneer er te weinig ademteugen overblijven.

    ``robust`` geeft de mediaan van het venster. Die is NIET AASM-conform en
    dient alleen als eerste passage: hij lokaliseert de events zodat de
    tweede passage ze kan uitsluiten.
    """
    n = onsets.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    lo_idx = np.searchsorted(onsets, onsets - window_s, side="left")
    for i in range(n):
        a = lo_idx[i]
        seg = amps[a:i]
        if exclude is not None and seg.size:
            kept = seg[~exclude[a:i]]
            if kept.size >= min_breaths:
                seg = kept
        if seg.size < min_breaths:
            continue
        if robust:
            out[i] = float(np.median(seg))
            continue
        m = float(seg.mean())
        if m <= 0:
            continue
        cv = float(seg.std() / m)
        if cv < stability_cv:
            out[i] = m
        else:
            k = max(1, min(int(n_largest), seg.size))
            out[i] = float(np.sort(seg)[-k:].mean())
    return out


def _candidate_runs(onsets, ends, red, floor, min_duration_s):
    """Aaneengesloten reeksen ademteugen onder de vloer, minstens zo lang."""
    below = np.nan_to_num(red, nan=-1.0) >= floor
    runs = []
    i = 0
    while i < below.size:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < below.size and below[j + 1]:
            j += 1
        if float(ends[j]) - float(onsets[i]) >= min_duration_s:
            runs.append((i, j))
        i = j + 1
    return runs


# ═══════════════════════════════════════════════════════════════
#  Patiëntsjabloon: de eigen SpO2-vertraging
# ═══════════════════════════════════════════════════════════════

def estimate_spo2_lag(event_ends, spo2, sf_spo2, lo=5.0, hi=60.0, step=1.0):
    """Vertraging tussen eventeinde en desaturatie-nadir, voor DEZE patiënt.

    Kruiscorrelatie tussen de eventeindes en het gedaalde-SpO2-signaal.
    Circulatietijd en probeplaats verschillen per patiënt; een vast venster
    van 30 of 45 s voor iedereen is een benadering die bij trage patiënten
    de nadir mist en bij snelle de verkeerde oppikt.

    Retourneert None wanneer er te weinig events zijn om iets te schatten.
    """
    if spo2 is None or len(event_ends) < 3:
        return None
    s = np.asarray(spo2, dtype=float)
    if s.size < int(sf_spo2 * 60):
        return None
    # "hoe laag staat de saturatie" als positief signaal
    base = float(np.nanpercentile(s[np.isfinite(s) & (s > 50)], 90)) if np.isfinite(s).any() else None
    if base is None:
        return None
    drop = np.clip(base - s, 0, None)
    drop[~np.isfinite(drop)] = 0.0

    best_lag, best_score = None, -np.inf
    for lag in np.arange(lo, hi + step, step):
        idx = ((np.asarray(event_ends) + lag) * sf_spo2).astype(int)
        idx = idx[(idx >= 0) & (idx < drop.size)]
        if idx.size < 3:
            continue
        score = float(np.mean(drop[idx]))
        if score > best_score:
            best_score, best_lag = score, float(lag)
    return best_lag


def _pre_event_below_local_baseline(spo2, sf_spo2, onset_s,
                                    baseline_win_s=120.0, pre_win_s=30.0):
    """Ligt de saturatie vlak vóór het event onder de lokale 2-min-baseline?

    Baseline = 90e percentiel over de 120 s ervoor; pre-event-saturatie =
    mediaan over de laatste 30 s. Dit is de conditie zoals de specificatie hem
    stelt. Let op: een 90e percentiel ligt per definitie boven het merendeel
    van zijn venster, dus deze conditie vuurt vaak — hoe vaak precies is een
    MEETVRAAG, en het antwoord bepaalt of de conditie deugt. Zie de CHANGELOG.
    """
    if spo2 is None:
        return False
    s = np.asarray(spo2, dtype=float)
    b0 = int(max(0.0, onset_s - baseline_win_s) * sf_spo2)
    p0 = int(max(0.0, onset_s - pre_win_s) * sf_spo2)
    p1 = int(onset_s * sf_spo2)
    bl = s[b0:p1]
    pre = s[p0:p1]
    bl = bl[np.isfinite(bl) & (bl > 50)]
    pre = pre[np.isfinite(pre) & (pre > 50)]
    if bl.size < 3 or pre.size < 3:
        return False
    return bool(float(np.median(pre)) < float(np.percentile(bl, 90)))


def _desat_at(spo2, sf_spo2, onset_s, end_s, lag_s, tol_s=15.0,
              pre_win_s=60.0):
    """``(diepte_pct, nadir)`` voor dit event, gezocht rond de patiëntlag.

    De nadir wordt meegegeven omdat het rapport hem toont; hem weglaten en
    ``min_spo2`` op None zetten kost informatie die hier al berekend is.
    Retourneert ``(None, None)`` wanneer er niets te meten valt.
    """
    if spo2 is None:
        return None, None
    s = np.asarray(spo2, dtype=float)
    a = int(max(0.0, onset_s - pre_win_s) * sf_spo2)
    b = int(onset_s * sf_spo2)
    pre = s[a:b]
    pre = pre[np.isfinite(pre) & (pre > 50)]
    if pre.size == 0:
        return None, None
    baseline = float(np.percentile(pre, 90))

    c = int(max(0.0, end_s + lag_s - tol_s) * sf_spo2)
    d = int((end_s + lag_s + tol_s) * sf_spo2)
    post = s[c:min(d, s.size)]
    post = post[np.isfinite(post) & (post > 50)]
    if post.size == 0:
        return None, None
    nadir = float(post.min())
    return float(baseline - nadir), nadir


# ═══════════════════════════════════════════════════════════════
#  Hoofdfunctie
# ═══════════════════════════════════════════════════════════════

def score_hypopneas_breathwise(
    breaths,
    hypno,
    spo2=None,
    sf_spo2=1.0,
    arousals=None,
    exclude_intervals=None,
    *,
    flow_reduction_threshold: float = 0.30,
    min_duration_s: float = 10.0,
    max_duration_s: float = 60.0,
    desat_threshold_pct: float = 3.0,
    desat_low_baseline_relaxation: bool = False,
    candidate_floor: float = 0.15,
    recovery_margin: float = 0.25,
    strictness: float = 0.50,
    flow_width: float = 0.08,
    desat_width: float = 0.8,
    dur_width: float = 2.0,
    baseline_window_s: float = 120.0,
    stability_cv: float = 0.25,
    arousal_window_s: float = 15.0,
    epoch_len_s: float = 30.0,
    use_template: bool = True,
    # Waarden die tot v0.13.0 hard in de functie stonden. De defaults zijn
    # exact die waarden, dus dit is byte-identiek; ze staan hier zodat ze
    # vindbaar en meetbaar zijn in plaats van verstopt.
    sure_depth: float = 0.50,
    default_lag_s: float = 25.0,
    arousal_weight: float = 0.9,
    template_floor: float = 0.65,
    template_center_frac: float = 0.60,
    template_width: float = 0.15,
    n_largest: int = 3,
    min_baseline_breaths: int = 4,
    # Passage A en B gebruikten tot v0.13.0 DEZELFDE vloer en minimumduur.
    # Die knop deed daardoor twee tegengestelde dingen tegelijk: hij bepaalt
    # welke ademteugen uit de baseline blijven EN welke kandidaat worden.
    # Verhogen maakt de baseline lager (minder uitgesloten) en de drempel
    # hoger; het netto-effect is niet-monotoon en per opname verschillend.
    # None = neem de kandidaatwaarde over, dus byte-identiek aan v0.13.0.
    exclusion_floor: float | None = None,
    exclusion_min_duration_s: float | None = None,
    # ── drie wijzigingen die de SCORING veranderen; default = v0.13.0 ──
    # Ze staan uit omdat aanzetten het gevalideerde werkpunt (strictness
    # 0,50, geijkt op PSG-IPA) verlaat. Gemeten op PSG-IPA (5 opnames):
    #
    #   configuratie          F1 med  F1 gem  pct   bias   MAE  sev
    #   default                0,434   0,512  p17  +0,17  0,29  4/5
    #   candidate_min_dur 8s   0,434   0,512  p17  +0,17  0,29  4/5
    #   arousal_latency        0,440   0,521  p17  -0,19  0,42  5/5
    #   template_use_duration  0,416   0,505  p15  +0,25  0,37  4/5
    #   alle drie              0,419   0,515  p15  -0,15  0,50  4/5
    #
    # candidate_min_duration_s doet in de praktijk NIETS: ademteugen duren
    # ~4 s, dus runs zijn gekwantiseerd op 8/12/16 s, en een run van 8 s
    # krijgt p_dur ~0,27 en haalt strictness 0,50 toch niet. De asymmetrie
    # is echt maar zonder gevolgen. template_use_duration maakt het
    # slechter. Alleen arousal_latency_grading wint (4 van 5 opnames beter,
    # geen slechter, severity 5/5) - maar de MAE verslechtert en n = 5, dus
    # aanzetten vraagt bevestiging op held-out data.
    candidate_min_duration_s: float | None = None,
    arousal_latency_grading: bool = False,
    arousal_latency_floor: float = 0.5,
    template_use_duration: bool = False,
    template_dur_width_frac: float = 0.5,
):
    """Scoor hypopneeën ademteug-voor-ademteug. Zie de moduledocstring.

    Returns ``(events, diagnostics)``. Elk event draagt ``p_scored`` — de mate
    waarin het aan de AASM-conjunctie voldoet — en een ``criteria``-dict met
    de bijdrage van elk predicaat.

    ``p_scored`` is een RANGSCHIKKING, geen kans dat een scorer het event zou
    markeren; op PSG-IPA is de correlatie met de werkelijke scorerfractie
    r = 0,194 bij een systematische afwijking van +0,33. Zie de
    moduledocstring.
    """
    onsets, ends, amps = _series(breaths)
    diag = {"n_breaths": int(onsets.size), "n_candidates": 0,
            "spo2_lag_s": None, "template": None, "n_scored": 0}
    if onsets.size < 10:
        return [], diag

    # ── baseline in twee passages ─────────────────────────────────
    # Passage A lokaliseert de events met een robuuste mediaan-baseline.
    # Passage B berekent de AASM-baseline over alléén de ademteugen die niet
    # bij een event horen — dat is wat "stabiele ademhaling" betekent.
    #
    # Zonder die uitsluiting bevat het venster de events zelf, en dan meet
    # geen enkele baselinedefinitie het juiste: het gemiddelde van de drie
    # grootste ademteugen wordt de herstelhyperpneu (waarna gewóón ademen al
    # als forse daling telt), en de mediaan wordt bij ernstige OSA het
    # eventniveau (waarna niets meer als daling telt).
    bl_a = _pre_baseline_per_breath(onsets, amps, window_s=baseline_window_s,
                                    min_breaths=min_baseline_breaths,
                                    robust=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        red_a = 1.0 - amps / bl_a
    red_a[~np.isfinite(red_a)] = np.nan
    in_event = np.zeros(onsets.size, dtype=bool)
    _ex_floor = candidate_floor if exclusion_floor is None else exclusion_floor
    _ex_dur = (min_duration_s if exclusion_min_duration_s is None
               else exclusion_min_duration_s)
    for _i, _j in _candidate_runs(onsets, ends, red_a, _ex_floor, _ex_dur):
        in_event[_i:_j + 1] = True
    # De herstelhyperpneu is net zomin stabiele ademhaling als het event
    # zelf. Blijft die in het venster, dan tilt hij de baseline op via de
    # n_largest-tak en telt gewoon ademen alsnog als daling.
    in_event |= np.nan_to_num(red_a, nan=0.0) <= -recovery_margin

    bl = _pre_baseline_per_breath(onsets, amps, window_s=baseline_window_s,
                                  stability_cv=stability_cv,
                                  n_largest=n_largest,
                                  min_breaths=min_baseline_breaths,
                                  exclude=in_event)
    with np.errstate(invalid="ignore", divide="ignore"):
        red = 1.0 - amps / bl                      # relatieve daling per ademteug
    red[~np.isfinite(red)] = np.nan

    # Invariant, zichtbaar in het audittrail: een TYPISCHE ademteug hoort
    # niet als gereduceerd te worden gemeten. Loopt dit ver van nul weg, dan
    # meet de baseline iets anders dan normaal ademen.
    _fin = np.isfinite(red)
    diag["median_breath_reduction"] = (
        round(float(np.median(red[_fin])), 4) if _fin.any() else None)
    diag["frac_breaths_excluded"] = round(float(in_event.mean()), 4)

    def _asleep(t):
        ep = int(t // epoch_len_s)
        return 0 <= ep < len(hypno) and hypno[ep] in ("N1", "N2", "N3", "R")

    # ── kandidaatreeksen: opeenvolgende ademteugen onder de vloer ──
    # De vloer ligt ONDER de AASM-drempel: marginale kandidaten moeten de
    # gegradeerde beoordeling bereiken in plaats van er binair uit te vallen.
    # De vloer op de DALING ligt onder de AASM-drempel, zodat marginale
    # dalingen de gradering halen. Voor de DUUR gold dat niet: die had een
    # harde grens op min_duration_s, waarna p_dur er nog eens omheen
    # gradeerde. Een event van 10,5 s haalde de poort en kreeg dan p_dur
    # ~0,56 - het criterium werd twee keer toegepast, en de tolerantie was
    # eenzijdig: alleen strafend, nooit toelatend. None = oude grens.
    _cand_dur = (min_duration_s if candidate_min_duration_s is None
                 else candidate_min_duration_s)
    cands = []
    for i, j in _candidate_runs(onsets, ends, red, candidate_floor,
                                _cand_dur):
        t0, t1 = float(onsets[i]), float(ends[j])
        if not _asleep(t0):
            continue
        if (t1 - t0) > max_duration_s:
            t1 = t0 + max_duration_s
            j = int(np.searchsorted(ends, t1, side="left"))
            j = min(max(j, i), onsets.size - 1)
        cands.append((i, j, t0, t1))
    diag["n_capped"] = sum(1 for _, _, a, b in cands
                           if (b - a) >= max_duration_s - 1e-6)

    if exclude_intervals:
        cands = [c for c in cands
                 if not any(a < c[3] and c[2] < b for a, b in exclude_intervals)]
    diag["n_candidates"] = len(cands)
    if not cands:
        return [], diag

    depth = np.array([float(np.nanmedian(red[i:j + 1])) for i, j, _, _ in cands])
    dur = np.array([t1 - t0 for _, _, t0, t1 in cands])

    # ── passage 1: onbetwistbare events -> patiëntsjabloon ────────
    sure = (depth >= sure_depth) & (dur >= min_duration_s)
    lag = None
    if spo2 is not None and sure.sum() >= 3:
        lag = estimate_spo2_lag([cands[k][3] for k in np.flatnonzero(sure)],
                                spo2, sf_spo2)
    if lag is None:
        lag = default_lag_s              # middenwaarde als er te
    diag["spo2_lag_s"] = lag             # weinig zekere events zijn

    if use_template and sure.sum() >= 3:
        # Periodiciteit: de mediane tijd tussen opeenvolgende zekere events.
        # Het ontwerp noemt die expliciet als sjabloonkenmerk; hij werd
        # nooit berekend. Voorlopig diagnostiek - hij stuurt de score niet,
        # want zonder validatie is dat weer een ongetoetste as erbij.
        _sure_starts = np.array([cands[k][2] for k in np.flatnonzero(sure)])
        _cycle = (float(np.median(np.diff(np.sort(_sure_starts))))
                  if _sure_starts.size >= 3 else None)
        tmpl = {"depth": float(np.median(depth[sure])),
                "dur": float(np.median(dur[sure])),
                "cycle_s": None if _cycle is None else round(_cycle, 1),
                "n": int(sure.sum())}
    else:
        tmpl = None
    diag["template"] = tmpl

    ar_onsets = sorted(float(a.get("onset_s", 0.0)) for a in (arousals or []))

    # ── passage 2: gegradeerde AASM-beoordeling ───────────────────
    events = []
    for k, (i, j, t0, t1) in enumerate(cands):
        d_red, d_dur = float(depth[k]), float(dur[k])

        p_flow = graded(d_red, flow_reduction_threshold, flow_width)
        p_dur = graded(d_dur, min_duration_s, dur_width)

        desat, nadir = (_desat_at(spo2, sf_spo2, t0, t1, lag)
                        if spo2 is not None else (None, None))
        # AFWIJKING VAN AASM REGEL 1A, alleen wanneer expliciet aangezet.
        # Bij een reeds gedaalde saturatie is de fysiologische ruimte voor een
        # 3 %-dip kleiner; het CENTRUM van de sigmoid schuift dan naar 2 %, de
        # breedte blijft gelijk. Geen nieuwe mechaniek, geen tweede drempel.
        _laag = False
        _centrum = desat_threshold_pct
        if desat_low_baseline_relaxation and spo2 is not None:
            _laag = _pre_event_below_local_baseline(spo2, sf_spo2, t0)
            if _laag:
                _centrum = 2.0
        p_desat = graded(desat, _centrum, desat_width) if desat is not None else 0.0

        near = [a for a in ar_onsets if t0 <= a <= t1 + arousal_window_s]
        if not near:
            p_arousal = 0.0
        elif not arousal_latency_grading:
            p_arousal = arousal_weight
        else:
            # Een arousal is als GEBEURTENIS binair - hij trad op of niet.
            # Wat wel gradueel is, is hoe overtuigend hij bij dit event
            # hoort: een arousal die tijdens of vlak na het event begint is
            # sterker gekoppeld dan een die tegen de rand van het
            # koppelvenster aan ligt. Lineair van 1 op latentie <=0 naar
            # arousal_latency_floor op de vensterrand.
            _lat = max(0.0, min(near) - t1)
            _f = 1.0 - (1.0 - arousal_latency_floor) * (
                _lat / arousal_window_s if arousal_window_s > 0 else 0.0)
            p_arousal = arousal_weight * float(np.clip(_f, 0.0, 1.0))

        # AASM Rule 1A, letterlijk: daling EN duur EN (desaturatie OF arousal)
        p_confirm = 1.0 - (1.0 - p_desat) * (1.0 - p_arousal)
        p = p_flow * p_dur * p_confirm

        # sjabloon: past dit bij wat deze patiënt de hele nacht doet?
        p_tmpl = 1.0
        if tmpl is not None:
            _fit = graded(d_red, template_center_frac * tmpl["depth"],
                          template_width)
            if template_use_duration:
                # Het sjabloon sloeg dur op maar gebruikte alleen depth. Een
                # event dat qua duur bij het nachtpatroon past is even
                # informatief als een dat qua diepte past; het geometrisch
                # gemiddelde laat een van beide niet domineren.
                _wd = max(1e-6, template_dur_width_frac * tmpl["dur"])
                _fit_d = graded(d_dur, template_center_frac * tmpl["dur"], _wd)
                _fit = float(np.sqrt(_fit * _fit_d))
            p_tmpl = float(np.clip(
                template_floor + (1.0 - template_floor) * _fit, 0.0, 1.0))
            p *= p_tmpl

        if p >= strictness:
            events.append({
                "type": "hypopnea",
                "onset_s": round(t0, 2),
                "duration_s": round(t1 - t0, 2),
                "stage": hypno[int(t0 // epoch_len_s)]
                         if int(t0 // epoch_len_s) < len(hypno) else "W",
                "epoch": int(t0 // epoch_len_s),
                "desaturation_pct": None if desat is None else round(desat, 2),
                "min_spo2": None if nadir is None else round(nadir, 1),
                "flow_reduction": round(100.0 * d_red, 1),
                "confidence": round(p, 3),
                "p_scored": round(p, 3),
                "n_breaths": int(j - i + 1),
                "criteria": {
                    "flow": round(p_flow, 3),
                    "duration": round(p_dur, 3),
                    "desaturation": round(p_desat, 3),
                    **({"low_baseline_relaxed": True} if _laag else {}),
                    "arousal": round(p_arousal, 3),
                    "confirmation": round(p_confirm, 3),
                    "template": round(p_tmpl, 3),
                },
                "classify_detail": {"rule": "1A_graded",
                                    "detector": "breath_scoring"},
                "rule1a_arousal": bool(near) and p_desat < 0.5,
                "rule1b": bool(near) and p_desat < 0.5,
            })

    diag["n_scored"] = len(events)
    return events, diag
