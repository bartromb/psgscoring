"""PLM: `result["events"]` is afgekapt, en dat hoort zichtbaar te zijn.

`analyze_plm` zet `result["events"] = plm_eligible[:200]` zonder toelichting
in de code en zonder spoor in de samenvatting. Op PSG-IPA SN1 zijn dat 200
van 660 gedetecteerde bewegingen. Alles wat verderop `output["plm"]["events"]`
leest ziet dus hooguit de eerste 200 van de nacht:

- `pipeline.py` koppelt PLM aan arousal voor `plm_arousal_index`, dat in het
  klinische PDF-rapport staat;
- YASAFlaskified schrijft de events in de EDF+-export
  (`generate_edfplus.py:155`), dus in een viewer stopt de markering ergens
  midden in de nacht.

De afkapping zelf blijft hier staan -- ze weghalen is een gedragswijziging.
Wat hier wordt vastgelegd is dat ze niet stil mag zijn. Vergelijk de regel
die het project elders hanteert: een grens die dekking beperkt, wordt
gerapporteerd.
"""
import numpy as np

from psgscoring.plm import analyze_plm

SF = 128.0
N_MOVES = 260
GAP_S = 20.0
FIRST_S = 60.0


def _emg_with_many_movements(seed=4):
    dur_s = FIRST_S + N_MOVES * GAP_S + 120.0
    n = int(dur_s * SF)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    for k in range(N_MOVES):
        t = FIRST_S + k * GAP_S
        s, e = int(t * SF), int((t + 1.5) * SF)
        x[s:e] += rng.normal(0.0, 60.0, e - s)
    hypno = ["N2"] * int(dur_s / 30)
    return x, hypno


def test_the_cap_is_recorded_in_the_summary():
    """Hoeveel er is weggelaten, hoort uit de samenvatting af te lezen."""
    emg, hypno = _emg_with_many_movements()
    out = analyze_plm(emg, None, SF, hypno, leg_unit="uV")
    assert out["success"], out.get("error")
    s = out["summary"]
    assert s["n_plm_eligible"] > 200, (
        f"fixture levert maar {s['n_plm_eligible']} geschikte bewegingen -- "
        "te weinig om de afkapping te raken"
    )
    assert len(out["events"]) == 200, "afkapping zelf blijft ongewijzigd"
    assert "n_events_truncated" in s, (
        "de afkapping laat geen spoor na in de samenvatting; wie "
        "output['plm']['events'] leest kan niet zien dat de nacht ophoudt"
    )
    assert s["n_events_truncated"] == s["n_plm_eligible"] - 200


def test_no_truncation_marker_when_nothing_is_dropped():
    """Bij een rustige nacht hoort het getal 0 te zijn, niet afwezig."""
    n = int(1800 * SF)
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 1.0, n)
    for k in range(5):
        t = 60.0 + k * 120.0
        s, e = int(t * SF), int((t + 1.5) * SF)
        x[s:e] += rng.normal(0.0, 60.0, e - s)
    out = analyze_plm(x, None, SF, ["N2"] * int(1800 / 30), leg_unit="uV")
    assert out["success"], out.get("error")
    assert out["summary"].get("n_events_truncated") == 0


# --------------------------------------------------------------------------
# 22-08-2026: de grens mocht zichtbaar zijn, maar niet meetellen.
# --------------------------------------------------------------------------

def test_the_cap_can_be_switched_off_for_internal_use():
    """`event_list_cap=None` levert de volledige lijst.

    De pipeline heeft die nodig: een index over de nacht mag niet van een
    transportgrens afhangen.
    """
    emg, hypno = _emg_with_many_movements()
    full = analyze_plm(emg, None, SF, hypno, leg_unit="uV", event_list_cap=None)
    assert full["success"], full.get("error")
    assert len(full["events"]) == full["summary"]["n_plm_eligible"] > 200
    assert full["summary"]["n_events_truncated"] == 0, (
        "zonder afkapping hoort er niets als weggelaten geboekt te worden")

    capped = analyze_plm(emg, None, SF, hypno, leg_unit="uV")
    assert len(capped["events"]) == 200, "standaardgedrag blijft afkappen"


def test_the_plm_arousal_index_counts_the_whole_night_not_the_first_200():
    """De kern: `plm_arousal_index` telt bewegingen voorbij de grens.

    De fixture is met opzet gemeen. Alle arousals liggen op bewegingen NA de
    200e, dus in de oude volgorde -- eerst afkappen, dan koppelen -- vindt de
    koppeling er precies nul en verdwijnt de index. Vindt hij er wel, dan is
    er over de hele nacht gerekend.
    """
    from psgscoring.pipeline import _cap_plm_event_list, _compute_arousal_etiology

    emg, hypno = _emg_with_many_movements()
    plm = analyze_plm(emg, None, SF, hypno, leg_unit="uV", event_list_cap=None)
    events = plm["events"]
    assert len(events) > 210, f"fixture te klein: {len(events)}"

    # Arousals uitsluitend op de bewegingen 205..214 -- allemaal voorbij de grens.
    late = events[205:215]
    arousals = [{"onset_s": float(e["onset_s"]) + 0.5, "duration_s": 3.0}
                for e in late]

    def _fresh_output():
        return {
            "plm": {"success": True,
                    "events": [dict(e) for e in events],
                    "summary": dict(plm["summary"])},
            "arousal": {
                "events": arousals,
                "summary": {"arousal_index": 12.0,
                            "n_respiratory_arousals": 4,
                            "n_spontaneous_arousals": 26},
            },
        }

    # Geleverde volgorde: eerst koppelen, dan afkappen.
    out = _fresh_output()
    _compute_arousal_etiology(out, hypno)
    _cap_plm_event_list(out)
    goed = out["arousal"]["summary"].get("n_plm_arousals")
    assert goed == len(late), (
        f"verwacht {len(late)} gekoppelde bewegingen over de hele nacht, "
        f"kreeg {goed}")
    assert len(out["plm"]["events"]) == 200, "de payloadgrens gaat er wel af"
    assert out["plm"]["summary"]["n_events_truncated"] == len(events) - 200

    # De oude volgorde, expliciet, zodat vaststaat dat de fixture discrimineert.
    out_oud = _fresh_output()
    _cap_plm_event_list(out_oud)
    _compute_arousal_etiology(out_oud, hypno)
    fout = out_oud["arousal"]["summary"].get("n_plm_arousals")
    assert fout == 0, (
        "de fixture toont het defect niet: in de oude volgorde hoort de "
        f"koppeling nul te vinden, maar hij vond {fout}")
    assert goed != fout


def test_a_reordering_does_not_stay_silent():
    """Draait het afkappen ooit weer vóór de koppeling, dan is dat te zien."""
    import logging

    from psgscoring.pipeline import _cap_plm_event_list, _compute_arousal_etiology

    emg, hypno = _emg_with_many_movements()
    plm = analyze_plm(emg, None, SF, hypno, leg_unit="uV", event_list_cap=None)
    out = {
        "plm": {"success": True, "events": [dict(e) for e in plm["events"]],
                "summary": dict(plm["summary"])},
        "arousal": {"events": [{"onset_s": 100.0, "duration_s": 3.0}],
                     "summary": {"arousal_index": 12.0,
                                 "n_respiratory_arousals": 4,
                                 "n_spontaneous_arousals": 26}},
    }
    _cap_plm_event_list(out)

    # Niet via caplog: die hangt af van propagatie, en in de volle suite zet
    # iets anders die om -- de test viel dan om zonder dat er iets mis was.
    # Een eigen handler op de logger zelf is onafhankelijk van de rest.
    seen = []

    class _Grab(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    lg = logging.getLogger("psgscoring.pipeline")
    h = _Grab(level=logging.WARNING)
    lg.addHandler(h)
    old_level = lg.level
    lg.setLevel(logging.WARNING)
    try:
        _compute_arousal_etiology(out, hypno)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)

    assert any("AFGEKAPTE" in m for m in seen), (
        "een index over een afgekapte lijst hoort een waarschuwing te geven; "
        f"gezien: {seen}")


def test_end_to_end_the_pipeline_asks_for_the_uncapped_list(monkeypatch):
    """Door de echte pipeline heen, want alleen dat vangt een teruggedraaide
    aanroep.

    De unittests hierboven roepen de twee helpers zelf in de juiste volgorde
    aan en blijven groen als `pipeline.py` de payloadgrens weer aan
    `analyze_plm` meegeeft. Hier wordt de aanroep zelf vastgelegd: de pipeline
    MOET om de ongekapte lijst vragen, anders rekent `plm_arousal_index` over
    het begin van de nacht.

    Bewust niet via `n_plm_arousals`: die index vereist ook een gevulde
    respiratoire/spontane splitsing, en die hangt aan de respiratoire
    detectie. Een fixture die dat allemaal tegelijk moet halen toetst vooral
    zichzelf. De aanroep is het contract dat hier telt.
    """
    import pytest
    mne = pytest.importorskip("mne")
    import psgscoring
    from psgscoring import pipeline as _pl

    sf = 64.0
    n_moves, gap_s, first_s = 240, 20.0, 60.0
    dur_s = first_s + n_moves * gap_s + 120.0
    n = int(dur_s * sf)
    t = np.arange(n) / sf
    rng = np.random.default_rng(23)

    leg = rng.normal(0.0, 1e-6, n)
    for k in range(n_moves):
        t0 = first_s + k * gap_s
        a, b = int(t0 * sf), int((t0 + 1.5) * sf)
        leg[a:b] += rng.normal(0.0, 60e-6, b - a)

    info = mne.create_info(
        ["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin", "EMG LAT", "EMG RAT"],
        sf, ["misc", "misc", "eeg", "emg", "emg", "emg"])
    raw = mne.io.RawArray(
        np.vstack([np.sin(2 * np.pi * 0.25 * t), np.full(n, 97.0),
                   rng.normal(0, 20e-6, n), rng.normal(0, 5e-6, n), leg, leg]),
        info, verbose=False)

    seen = {}
    _echt = _pl.analyze_plm

    def _spy(*a, **kw):
        seen.update(kw)
        return _echt(*a, **kw)

    monkeypatch.setattr(_pl, "analyze_plm", _spy)

    out = psgscoring.run_pneumo_analysis(
        raw, hypno=["N2"] * int(np.ceil(raw.times[-1] / 30.0)),
        scoring_profile="aasm_v3_rec")

    assert "event_list_cap" in seen, (
        "de pipeline geeft geen event_list_cap mee aan analyze_plm; dan geldt "
        "de standaardgrens en telt plm_arousal_index alleen het begin van de "
        "nacht")
    assert seen["event_list_cap"] is None, (
        f"pipeline vraagt om een afgekapte lijst (cap={seen['event_list_cap']}); "
        "afgeleide indices rekenen dan over het begin van de nacht")

    # En de grens gaat er alsnog af voordat het de payload in gaat.
    ps = out["plm"]["summary"]
    assert ps["n_plm_eligible"] > 200, (
        f"fixture raakt de grens niet: {ps['n_plm_eligible']} bewegingen")
    assert len(out["plm"]["events"]) == 200, "de payloadgrens hoort er wel af"
    assert ps["n_events_truncated"] == ps["n_plm_eligible"] - 200


def test_the_cap_is_a_profile_flag_with_the_old_value_as_default():
    """De grens is instelbaar, en 200 blijft de default.

    Verhogen vergroot de opgeslagen jobpayload; dat is een beslissing voor wie
    de opslag beheert, niet iets om stil te wijzigen.
    """
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        pp = get_profile(name).post_processing
        assert pp.plm_event_list_cap == 200, (
            f"{name} wijkt af van de default-grens: {pp.plm_event_list_cap}")
        assert SCORING_PROFILES[name]["PLM_EVENT_LIST_CAP"] == 200, (
            f"{name}: registry loopt uit de pas met het profiel")


def test_the_env_override_wins_and_zero_means_no_limit(monkeypatch):
    from psgscoring.pipeline import _plm_event_cap

    prof = {"PLM_EVENT_LIST_CAP": 200}
    assert _plm_event_cap(prof) == 200

    monkeypatch.setenv("PSGSCORING_PLM_EVENT_LIST_CAP", "1500")
    assert _plm_event_cap(prof) == 1500

    # 0 = geen grens. Menselijke scoorders halen op PSG-IPA tot 1033
    # bewegingen per nacht, dus 200 kapt een echte nacht ruim af.
    monkeypatch.setenv("PSGSCORING_PLM_EVENT_LIST_CAP", "0")
    assert _plm_event_cap(prof) > 10_000

    # Onleesbaar -> profielwaarde, niet stil iets anders.
    monkeypatch.setenv("PSGSCORING_PLM_EVENT_LIST_CAP", "kaas")
    assert _plm_event_cap(prof) == 200


def test_raising_the_cap_actually_delivers_more_events():
    """Een hogere grens moet echt meer events opleveren, niet alleen een getal."""
    from psgscoring.pipeline import _cap_plm_event_list

    emg, hypno = _emg_with_many_movements()
    plm = analyze_plm(emg, None, SF, hypno, leg_unit="uV", event_list_cap=None)
    n_all = len(plm["events"])
    assert n_all > 200, f"fixture te klein: {n_all}"

    def _out():
        return {"plm": {"success": True,
                        "events": [dict(e) for e in plm["events"]],
                        "summary": dict(plm["summary"])}}

    krap = _out()
    _cap_plm_event_list(krap, 200)
    assert len(krap["plm"]["events"]) == 200
    assert krap["plm"]["summary"]["n_events_truncated"] == n_all - 200

    ruim = _out()
    _cap_plm_event_list(ruim, 10**9)
    assert len(ruim["plm"]["events"]) == n_all, (
        "met een ruime grens hoort de hele nacht in de lijst te staan")
    assert ruim["plm"]["summary"]["n_events_truncated"] == 0
