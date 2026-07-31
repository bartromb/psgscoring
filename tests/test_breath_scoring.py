"""Tests voor psgscoring.breath_scoring — de ademteug-gebaseerde hypopnee-detector.

De detector zit achter profiel ``aasm_v3_breath``. Deze tests dekken de vijf
verschuivingen uit de moduledocstring afzonderlijk, zodat een regressie
aanwijsbaar is naar één mechanisme in plaats van naar "de detector".
"""
import numpy as np
import pytest

from psgscoring.breath_scoring import (
    _candidate_runs,
    _desat_at,
    _pre_baseline_per_breath,
    _series,
    estimate_spo2_lag,
    graded,
    score_hypopneas_breathwise,
)

BREATH_S = 4.0
SF_SPO2 = 1.0


# ─────────────────────────────────────────────────────────────────
#  Synthetische nacht
# ─────────────────────────────────────────────────────────────────

def make_night(event_starts, n_breaths_event=4, depth=0.5, total_s=1800.0):
    """Regelmatige ademhaling met verlaagde reeksen op ``event_starts``.

    Retourneert (breaths, hypno, event_windows). Amplitude 1,0 buiten de
    events, ``1-depth`` erbinnen.
    """
    breaths, windows = [], []
    t = 0.0
    starts = sorted(event_starts)
    while t < total_s:
        in_event = False
        for s in starts:
            if s <= t < s + n_breaths_event * BREATH_S:
                in_event = True
                break
        breaths.append({"onset_s": t, "duration_s": BREATH_S,
                        "amplitude": (1.0 - depth) if in_event else 1.0})
        t += BREATH_S
    for s in starts:
        windows.append((s, s + n_breaths_event * BREATH_S))
    hypno = ["N2"] * int(np.ceil(total_s / 30.0))
    return breaths, hypno, windows


def make_spo2(windows, lag_s=20.0, drop_pct=5.0, total_s=1800.0, base=96.0):
    """SpO2 op 1 Hz met een nadir ``lag_s`` na elk eventeinde."""
    s = np.full(int(total_s * SF_SPO2), base, dtype=float)
    for _, end in windows:
        c = int((end + lag_s) * SF_SPO2)
        for k in range(-6, 7):
            i = c + k
            if 0 <= i < s.size:
                s[i] = base - drop_pct * (1.0 - abs(k) / 8.0)
    return s


# ─────────────────────────────────────────────────────────────────
#  1. Gegradeerde predicaten
# ─────────────────────────────────────────────────────────────────

def test_graded_is_half_at_center():
    assert graded(0.30, 0.30, 0.08) == pytest.approx(0.5)


def test_graded_is_monotone_increasing():
    xs = np.linspace(0.0, 0.8, 40)
    ys = [graded(x, 0.30, 0.08) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))


def test_graded_with_zero_width_is_a_hard_threshold():
    assert graded(0.31, 0.30, 0.0) == 1.0
    assert graded(0.29, 0.30, 0.0) == 0.0


def test_graded_narrower_width_is_closer_to_binary():
    """Kleinere width -> scherper. Dat is de knop die AASM-conformiteit borgt."""
    x = 0.35   # net boven de drempel
    assert graded(x, 0.30, 0.02) > graded(x, 0.30, 0.20)


# ─────────────────────────────────────────────────────────────────
#  2. Ademteugreeks
# ─────────────────────────────────────────────────────────────────

def test_series_drops_unusable_breaths():
    on, en, am = _series([
        {"onset_s": 0.0, "duration_s": 4.0, "amplitude": 1.0},
        {"onset_s": 4.0, "duration_s": 4.0, "amplitude": None},     # geen amp
        {"onset_s": 8.0, "duration_s": None, "amplitude": 1.0},     # geen duur
        {"onset_s": 12.0, "duration_s": 4.0, "amplitude": 0.0},     # amp 0
        {"onset_s": 16.0, "duration_s": 4.0, "amplitude": np.nan},  # NaN
        {"onset_s": 20.0, "duration_s": 4.0, "amplitude": 0.5},
    ])
    assert on.tolist() == [0.0, 20.0]
    assert en.tolist() == [4.0, 24.0]
    assert am.tolist() == [1.0, 0.5]


def test_series_on_empty_input():
    on, en, am = _series([])
    assert on.size == en.size == am.size == 0


# ─────────────────────────────────────────────────────────────────
#  3. Pre-event baseline
# ─────────────────────────────────────────────────────────────────

def test_baseline_is_nan_without_enough_history():
    onsets = np.arange(0.0, 40.0, 4.0)
    amps = np.ones(onsets.size)
    bl = _pre_baseline_per_breath(onsets, amps, min_breaths=4)
    assert np.all(np.isnan(bl[:4])), "eerste ademteugen hebben geen voorgeschiedenis"
    assert np.all(np.isfinite(bl[4:]))


def test_stable_breathing_uses_the_mean():
    onsets = np.arange(0.0, 200.0, 4.0)
    amps = np.full(onsets.size, 2.0)
    bl = _pre_baseline_per_breath(onsets, amps)
    assert bl[-1] == pytest.approx(2.0)


def test_unstable_breathing_uses_the_largest_excursions():
    """AASM: bij niet-stabiele ademhaling telt de grootste excursie, niet het gemiddelde."""
    onsets = np.arange(0.0, 200.0, 4.0)
    rng = np.random.default_rng(0)
    amps = rng.choice([0.2, 2.0], size=onsets.size)   # CV ver boven 0,25
    bl = _pre_baseline_per_breath(onsets, amps, n_largest=3)
    assert bl[-1] == pytest.approx(2.0), "moet de top-3 excursies pakken"
    assert bl[-1] > float(amps[:-1].mean())


def test_baseline_window_is_respected():
    """Amplitude verdubbelt; na een volledig venster telt alleen het nieuwe niveau."""
    onsets = np.arange(0.0, 600.0, 4.0)
    amps = np.where(onsets < 300.0, 1.0, 2.0)
    bl = _pre_baseline_per_breath(onsets, amps, window_s=120.0)
    assert bl[-1] == pytest.approx(2.0)


def test_exclude_mask_keeps_marked_breaths_out_of_the_window():
    onsets = np.arange(0.0, 200.0, 4.0)
    amps = np.ones(onsets.size)
    amps[10:20] = 5.0                       # zou de baseline optillen
    excl = np.zeros(onsets.size, bool)
    excl[10:20] = True
    assert _pre_baseline_per_breath(onsets, amps)[-1] > 2.0
    assert _pre_baseline_per_breath(onsets, amps, exclude=excl)[-1] == pytest.approx(1.0)


def test_exclude_mask_is_ignored_when_too_little_survives():
    """Liever een vervuilde baseline dan geen baseline."""
    onsets = np.arange(0.0, 200.0, 4.0)
    amps = np.ones(onsets.size)
    excl = np.ones(onsets.size, bool)       # alles uitgesloten
    assert np.isfinite(_pre_baseline_per_breath(onsets, amps, exclude=excl)[-1])


def test_robust_mode_returns_the_median():
    onsets = np.arange(0.0, 200.0, 4.0)
    amps = np.ones(onsets.size)
    amps[:10] = 9.0                         # uitschieters, mediaan ongevoelig
    bl = _pre_baseline_per_breath(onsets, amps, robust=True)
    assert bl[-1] == pytest.approx(1.0)


def test_candidate_runs_needs_the_minimum_duration():
    onsets = np.arange(0.0, 40.0, 4.0)
    ends = onsets + 4.0
    red = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    assert _candidate_runs(onsets, ends, red, 0.15, 10.0) == [(4, 6)]  # 12 s
    assert _candidate_runs(onsets, ends, red, 0.15, 8.0) == [(1, 2), (4, 6)]


def test_candidate_runs_treats_nan_as_not_below():
    onsets = np.arange(0.0, 40.0, 4.0)
    ends = onsets + 4.0
    red = np.array([0.5, 0.5, np.nan, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    assert _candidate_runs(onsets, ends, red, 0.15, 10.0) == [(3, 5)]


# ─────────────────────────────────────────────────────────────────
#  3b. Twee passages: de baseline mag de events niet bevatten
# ─────────────────────────────────────────────────────────────────

def make_cyclic_night(n_cycles=25, event_amp=0.4, recovery_amp=1.7,
                      n_event=5, n_recovery=3, n_normal=10):
    """Event -> herstelhyperpneu -> normaal ademen, herhaald.

    Reproduceert waar de eerste implementatie op stukliep: het venster van
    120 s bevat de events en hun herstelademteugen, waardoor geen enkele
    baselinedefinitie normaal ademen meet.
    """
    breaths, t = [], 0.0
    normal_idx = []
    for _ in range(n_cycles):
        for amp, count in ((event_amp, n_event), (recovery_amp, n_recovery),
                           (1.0, n_normal)):
            for _k in range(count):
                if amp == 1.0:
                    normal_idx.append(len(breaths))
                breaths.append({"onset_s": t, "duration_s": BREATH_S,
                                "amplitude": amp})
                t += BREATH_S
    hypno = ["N2"] * int(np.ceil(t / 30.0))
    return breaths, hypno, normal_idx


def test_two_pass_baseline_is_not_inflated_by_recovery_breaths():
    """Normaal ademen hoort niet als daling gemeten te worden.

    Dit is de regressie die de detector op PSG-IPA SN2 onderuit haalde: de
    mediane ademteug werd daar als 23,6% gereduceerd geregistreerd omdat het
    baselinevenster de herstelhyperpneu bevatte.
    """
    breaths, hypno, _ = make_cyclic_night()
    _, diag = score_hypopneas_breathwise(breaths, hypno)
    assert abs(diag["median_breath_reduction"]) < 0.10, (
        f"typische ademteug gemeten als {100*diag['median_breath_reduction']:.0f}% "
        "gereduceerd — de baseline meet niet normaal ademen")


def test_two_pass_finds_the_events_and_not_the_normal_breathing():
    breaths, hypno, _ = make_cyclic_night()
    _, diag = score_hypopneas_breathwise(breaths, hypno)
    # 25 events van 5 ademteugen = 20 s elk; het aantal kandidaten hoort daar
    # bij in de buurt te liggen, niet bij het aantal ademteugen
    assert 20 <= diag["n_candidates"] <= 30, diag["n_candidates"]
    assert diag["n_capped"] == 0, "reeksen mogen niet tegen de 60 s-cap lopen"


def test_recovery_breaths_are_excluded_from_the_baseline():
    breaths, hypno, _ = make_cyclic_night()
    _, diag = score_hypopneas_breathwise(breaths, hypno)
    # event (5/18) + herstel (3/18) = 44% van de ademteugen
    assert 0.30 <= diag["frac_breaths_excluded"] <= 0.60, diag["frac_breaths_excluded"]


def test_single_pass_baseline_would_have_failed_this():
    """Waarborgt dat de test hierboven echt iets test.

    Met de baseline over het RUWE venster — dus zonder de events uit te
    sluiten — wordt normaal ademen wel degelijk als forse daling gemeten.
    """
    breaths, _, normal_idx = make_cyclic_night()
    on, _en, am = _series(breaths)
    bl = _pre_baseline_per_breath(on, am)          # geen exclude-masker
    red = 1.0 - am / bl
    assert np.nanmedian(red[normal_idx]) > 0.25, (
        "de eenpassage-baseline hoort hier juist wél te ontsporen")


# ─────────────────────────────────────────────────────────────────
#  4. Patiënt-eigen SpO2-vertraging
# ─────────────────────────────────────────────────────────────────

def test_spo2_lag_recovers_the_injected_delay():
    _, _, windows = make_night([300.0, 600.0, 900.0, 1200.0])
    spo2 = make_spo2(windows, lag_s=20.0)
    lag = estimate_spo2_lag([w[1] for w in windows], spo2, SF_SPO2)
    assert lag == pytest.approx(20.0, abs=3.0)


def test_spo2_lag_distinguishes_a_slow_patient():
    """Het punt van verschuiving 2: de vertraging is patiënt-eigen, niet vast."""
    _, _, windows = make_night([300.0, 600.0, 900.0, 1200.0])
    fast = estimate_spo2_lag([w[1] for w in windows],
                             make_spo2(windows, lag_s=12.0), SF_SPO2)
    slow = estimate_spo2_lag([w[1] for w in windows],
                             make_spo2(windows, lag_s=45.0), SF_SPO2)
    assert slow - fast == pytest.approx(33.0, abs=6.0)


def test_spo2_lag_returns_none_with_too_few_events():
    spo2 = np.full(1800, 96.0)
    assert estimate_spo2_lag([100.0, 200.0], spo2, SF_SPO2) is None


def test_spo2_lag_returns_none_without_spo2():
    assert estimate_spo2_lag([100.0, 200.0, 300.0], None, SF_SPO2) is None


def test_desat_at_measures_the_drop():
    _, _, windows = make_night([600.0])
    spo2 = make_spo2(windows, lag_s=20.0, drop_pct=5.0)
    d = _desat_at(spo2, SF_SPO2, windows[0][0], windows[0][1], lag_s=20.0)
    assert d == pytest.approx(5.0, abs=1.0)


def test_desat_at_returns_none_without_spo2():
    assert _desat_at(None, SF_SPO2, 0.0, 10.0, 20.0) is None


# ─────────────────────────────────────────────────────────────────
#  5. Scoring: de AASM-conjunctie
# ─────────────────────────────────────────────────────────────────

EVENTS_AT = [300.0, 600.0, 900.0, 1200.0]


def test_scores_events_with_flow_drop_and_desaturation():
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0, drop_pct=5.0)
    events, diag = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.50)
    assert len(events) == len(EVENTS_AT)
    assert diag["n_candidates"] == len(EVENTS_AT)
    assert diag["n_scored"] == len(events)


def test_confirmation_is_required__no_desat_no_arousal_no_event():
    """Rule 1A is een conjunctie: zonder (desat OF arousal) telt de daling niet."""
    breaths, hypno, windows = make_night(EVENTS_AT)
    flat = np.full(1800, 96.0)          # saturatie beweegt niet
    events, diag = score_hypopneas_breathwise(
        breaths, hypno, spo2=flat, sf_spo2=SF_SPO2, strictness=0.50)
    assert diag["n_candidates"] == len(EVENTS_AT), "kandidaten worden wel gevonden"
    assert events == [], "maar zonder bevestiging niet gescoord"


def test_arousal_alone_confirms_a_hypopnea():
    """De arousal-tak van Rule 1A, zonder enige desaturatie."""
    breaths, hypno, windows = make_night(EVENTS_AT)
    flat = np.full(1800, 96.0)
    arousals = [{"onset_s": end + 5.0} for _, end in windows]
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=flat, sf_spo2=SF_SPO2,
        arousals=arousals, strictness=0.50)
    assert len(events) == len(EVENTS_AT)
    assert all(e["rule1a_arousal"] for e in events)
    assert all(e["criteria"]["desaturation"] < 0.5 for e in events)


def test_arousal_outside_the_window_does_not_confirm():
    breaths, hypno, windows = make_night(EVENTS_AT)
    flat = np.full(1800, 96.0)
    arousals = [{"onset_s": end + 60.0} for _, end in windows]   # ver buiten 15 s
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=flat, sf_spo2=SF_SPO2,
        arousals=arousals, strictness=0.50)
    assert events == []


def test_short_runs_do_not_reach_ten_seconds():
    breaths, hypno, windows = make_night(EVENTS_AT, n_breaths_event=2)  # 8 s
    spo2 = make_spo2(windows, lag_s=20.0)
    events, diag = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.50)
    assert diag["n_candidates"] == 0
    assert events == []


def test_shallow_drop_below_the_candidate_floor_is_ignored():
    breaths, hypno, windows = make_night(EVENTS_AT, depth=0.05)
    spo2 = make_spo2(windows, lag_s=20.0)
    _, diag = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.50)
    assert diag["n_candidates"] == 0


def test_wake_epochs_are_excluded():
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0)
    for start, _ in windows[:2]:
        hypno[int(start // 30.0)] = "W"
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.50)
    assert len(events) == len(EVENTS_AT) - 2


def test_exclude_intervals_are_honoured():
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0)
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2,
        exclude_intervals=[windows[0], windows[1]], strictness=0.50)
    assert len(events) == len(EVENTS_AT) - 2


# ─────────────────────────────────────────────────────────────────
#  6. Strengheid als één as
# ─────────────────────────────────────────────────────────────────

def test_strictness_is_monotone():
    """Verschuiving 5: één as, en hij moet zich als een as gedragen."""
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0, drop_pct=4.0)
    counts = []
    for s in (0.10, 0.30, 0.50, 0.70, 0.90, 0.99):
        ev, _ = score_hypopneas_breathwise(
            breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=s)
        counts.append(len(ev))
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], "de as moet daadwerkelijk iets doen"


# ─────────────────────────────────────────────────────────────────
#  7. Audittrail
# ─────────────────────────────────────────────────────────────────

def test_every_event_carries_its_criteria():
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0)
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.50)
    assert events
    for e in events:
        c = e["criteria"]
        assert set(c) == {"flow", "duration", "desaturation", "arousal",
                          "confirmation", "template"}
        assert all(0.0 <= v <= 1.0 for v in c.values())
        assert e["p_scored"] == e["confidence"]
        assert 0.0 <= e["p_scored"] <= 1.0
        assert e["type"] == "hypopnea"
        assert e["n_breaths"] >= 1
        assert e["classify_detail"]["detector"] == "breath_scoring"


def test_p_scored_discriminates_between_events():
    """Het confidence-probleem: 123 van 124 in dezelfde band betekent niets.

    Een diep event met forse desaturatie hoort een hogere kans te krijgen dan
    een marginaal event met een randdesaturatie.
    """
    breaths, hypno, _ = make_night([300.0, 900.0], depth=0.5)
    # tweede event ondieper maken
    for b in breaths:
        if 900.0 <= b["onset_s"] < 916.0:
            b["amplitude"] = 0.68          # ~32% daling: net over de drempel
    windows = [(300.0, 316.0), (900.0, 916.0)]
    spo2 = np.full(1800, 96.0)
    for (end, drop) in ((316.0, 8.0), (916.0, 3.2)):
        c = int(end + 20.0)
        for k in range(-6, 7):
            spo2[c + k] = 96.0 - drop * (1.0 - abs(k) / 8.0)
    events, _ = score_hypopneas_breathwise(
        breaths, hypno, spo2=spo2, sf_spo2=SF_SPO2, strictness=0.05)
    assert len(events) == 2
    deep, marginal = sorted(events, key=lambda e: e["onset_s"])
    assert deep["p_scored"] > marginal["p_scored"]


# ─────────────────────────────────────────────────────────────────
#  8. Degeneratiegevallen
# ─────────────────────────────────────────────────────────────────

def test_no_breaths_returns_empty():
    events, diag = score_hypopneas_breathwise([], ["N2"] * 60)
    assert events == []
    assert diag["n_breaths"] == 0


def test_too_few_breaths_returns_empty():
    breaths = [{"onset_s": i * 4.0, "duration_s": 4.0, "amplitude": 1.0}
               for i in range(5)]
    events, diag = score_hypopneas_breathwise(breaths, ["N2"] * 60)
    assert events == []
    assert diag["n_breaths"] == 5


def test_falls_back_to_a_default_lag_without_spo2():
    breaths, hypno, _ = make_night(EVENTS_AT)
    events, diag = score_hypopneas_breathwise(breaths, hypno, spo2=None)
    assert diag["spo2_lag_s"] == 25.0
    assert events == [], "zonder saturatie en zonder arousal geen bevestiging"


def test_all_wake_returns_no_events():
    breaths, hypno, windows = make_night(EVENTS_AT)
    spo2 = make_spo2(windows, lag_s=20.0)
    events, diag = score_hypopneas_breathwise(
        breaths, ["W"] * len(hypno), spo2=spo2, sf_spo2=SF_SPO2)
    assert diag["n_candidates"] == 0
    assert events == []
