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

4. **De uitkomst is een kans, geen booleaan.** Er is geen enkele juiste
   scoring: op PSG-IPA SN2 splitsten twaalf scorers 8/4. Elk event krijgt
   ``p_scored``, te lezen als "welk deel van de scorers zou dit markeren".

5. **Strengheid is één as.** In plaats van drie parametercombinaties is er
   één drempel op die kans. Lager = soepeler.

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
                             stability_cv=0.25, n_largest=3, min_breaths=4):
    """Pre-event baseline vóór ELKE ademteug.

    Stabiele ademhaling (CV < ``stability_cv``) -> gemiddelde amplitude;
    anders het gemiddelde van de ``n_largest`` grootste. Dat is de
    operationalisering die de AASM aangeeft wanneer stabiele ademhaling niet
    te bepalen is. NaN waar er te weinig voorgeschiedenis is (begin van de
    opname, na een gap) — de aanroeper slaat die ademteugen over in plaats
    van een baseline te verzinnen.
    """
    n = onsets.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    lo_idx = np.searchsorted(onsets, onsets - window_s, side="left")
    for i in range(n):
        seg = amps[lo_idx[i]:i]
        if seg.size < min_breaths:
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


def _desat_at(spo2, sf_spo2, onset_s, end_s, lag_s, tol_s=15.0,
              pre_win_s=60.0):
    """Desaturatiediepte (%) voor dit event, gezocht rond de patiëntlag."""
    if spo2 is None:
        return None
    s = np.asarray(spo2, dtype=float)
    a = int(max(0.0, onset_s - pre_win_s) * sf_spo2)
    b = int(onset_s * sf_spo2)
    pre = s[a:b]
    pre = pre[np.isfinite(pre) & (pre > 50)]
    if pre.size == 0:
        return None
    baseline = float(np.percentile(pre, 90))

    c = int(max(0.0, end_s + lag_s - tol_s) * sf_spo2)
    d = int((end_s + lag_s + tol_s) * sf_spo2)
    post = s[c:min(d, s.size)]
    post = post[np.isfinite(post) & (post > 50)]
    if post.size == 0:
        return None
    return float(baseline - float(post.min()))


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
    candidate_floor: float = 0.15,
    strictness: float = 0.50,
    flow_width: float = 0.08,
    desat_width: float = 0.8,
    dur_width: float = 2.0,
    baseline_window_s: float = 120.0,
    stability_cv: float = 0.25,
    arousal_window_s: float = 15.0,
    epoch_len_s: float = 30.0,
    use_template: bool = True,
):
    """Scoor hypopneeën ademteug-voor-ademteug. Zie de moduledocstring.

    Returns ``(events, diagnostics)``. Elk event draagt ``p_scored`` en een
    ``criteria``-dict met de bijdrage van elk AASM-predicaat.
    """
    onsets, ends, amps = _series(breaths)
    diag = {"n_breaths": int(onsets.size), "n_candidates": 0,
            "spo2_lag_s": None, "template": None, "n_scored": 0}
    if onsets.size < 10:
        return [], diag

    bl = _pre_baseline_per_breath(onsets, amps, window_s=baseline_window_s,
                                  stability_cv=stability_cv)
    with np.errstate(invalid="ignore", divide="ignore"):
        red = 1.0 - amps / bl                      # relatieve daling per ademteug
    red[~np.isfinite(red)] = np.nan

    def _asleep(t):
        ep = int(t // epoch_len_s)
        return 0 <= ep < len(hypno) and hypno[ep] in ("N1", "N2", "N3", "R")

    # ── kandidaatreeksen: opeenvolgende ademteugen onder de vloer ──
    # De vloer ligt ONDER de AASM-drempel: marginale kandidaten moeten de
    # gegradeerde beoordeling bereiken in plaats van er binair uit te vallen.
    below = np.nan_to_num(red, nan=-1.0) >= candidate_floor
    cands = []
    i = 0
    while i < below.size:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < below.size and below[j + 1]:
            j += 1
        t0, t1 = float(onsets[i]), float(ends[j])
        if (t1 - t0) >= min_duration_s and _asleep(t0):
            # te lange reeks: knip op de diepste aaneengesloten kern
            if (t1 - t0) > max_duration_s:
                t1 = t0 + max_duration_s
                j = int(np.searchsorted(ends, t1, side="left"))
                j = min(max(j, i), below.size - 1)
            cands.append((i, j, t0, t1))
        i = j + 1

    if exclude_intervals:
        cands = [c for c in cands
                 if not any(a < c[3] and c[2] < b for a, b in exclude_intervals)]
    diag["n_candidates"] = len(cands)
    if not cands:
        return [], diag

    depth = np.array([float(np.nanmedian(red[i:j + 1])) for i, j, _, _ in cands])
    dur = np.array([t1 - t0 for _, _, t0, t1 in cands])

    # ── passage 1: onbetwistbare events -> patiëntsjabloon ────────
    sure = (depth >= 0.50) & (dur >= min_duration_s)
    lag = None
    if spo2 is not None and sure.sum() >= 3:
        lag = estimate_spo2_lag([cands[k][3] for k in np.flatnonzero(sure)],
                                spo2, sf_spo2)
    if lag is None:
        lag = 25.0                       # AASM-achtige middenwaarde als er te
    diag["spo2_lag_s"] = lag             # weinig zekere events zijn

    if use_template and sure.sum() >= 3:
        tmpl = {"depth": float(np.median(depth[sure])),
                "dur": float(np.median(dur[sure])),
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

        desat = _desat_at(spo2, sf_spo2, t0, t1, lag) if spo2 is not None else None
        p_desat = graded(desat, desat_threshold_pct, desat_width) if desat is not None else 0.0

        near = [a for a in ar_onsets if t0 <= a <= t1 + arousal_window_s]
        p_arousal = 0.9 if near else 0.0

        # AASM Rule 1A, letterlijk: daling EN duur EN (desaturatie OF arousal)
        p_confirm = 1.0 - (1.0 - p_desat) * (1.0 - p_arousal)
        p = p_flow * p_dur * p_confirm

        # sjabloon: past dit bij wat deze patiënt de hele nacht doet?
        p_tmpl = 1.0
        if tmpl is not None:
            p_tmpl = float(np.clip(
                0.65 + 0.35 * graded(d_red, 0.6 * tmpl["depth"], 0.15), 0.0, 1.0))
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
                "min_spo2": None,
                "flow_reduction": round(100.0 * d_red, 1),
                "confidence": round(p, 3),
                "p_scored": round(p, 3),
                "n_breaths": int(j - i + 1),
                "criteria": {
                    "flow": round(p_flow, 3),
                    "duration": round(p_dur, 3),
                    "desaturation": round(p_desat, 3),
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
