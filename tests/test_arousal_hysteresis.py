"""Arousal: hysterese op het einde van een event (opt-in).

Fase 1 bouwt haar mask per sample en labelt die direct -- er wordt geen gat
gedicht. Bandvermogen fluctueert op subseconde-schaal, dus één arousal valt
uiteen in scherven en de eis van 3 s gooit ze bijna allemaal weg. Gemeten op
MESA: 1897 ruwe regio's waarvan er 65 overblijven, mediane duur 3,6 s tegen
8,6 s (PSG-IPA) en 11,0 s (MESA) bij menselijke scoorders.

Zie docs/arousal_duration_preregistratie.md.
"""
import numpy as np

from psgscoring.arousal import AROUSAL_EXIT_RATIO, detect_arousals

SF = 100.0
DUR_S = 900
AT_S = 600.0
EVENT_S = 20.0

# De inzakkingen moeten LANGER duren dan het 2 s-vermogenvenster van
# `_bandpower_instant`, anders smoothet dat ze weg en fragmenteert de mask
# helemaal niet -- een fixture met snellere dips staat groen zonder iets te
# meten (13,3 s tegen 13,6 s). Periode 7 s met diepe inzakkingen geeft
# 6,4 s zonder en 20,8 s met hysterese.
_DIP_PERIOD_S = 7.0
_DIP_DEPTH = 0.97


def _flickering_arousal(seed=3):
    """N2-achtergrond met één arousal van 20 s die herhaald diep inzakt.

    Zo ziet echt bandvermogen eruit: de verhoging is aanwezig maar niet elke
    sample boven de drempel. Zonder hysterese valt dit uiteen en houdt de
    3 s-eis er één scherf van over.
    """
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + 4.0 * np.sin(2 * np.pi * 10.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    s, e = int(AT_S * SF), int((AT_S + EVENT_S) * SF)
    env = 1.0 + _DIP_DEPTH * np.sin(2 * np.pi * t[s:e] / _DIP_PERIOD_S)
    eeg[s:e] += env * (30.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                       + 15.0 * np.sin(2 * np.pi * 20.0 * t[s:e]))
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def _in_event(res):
    return [e for e in res["events"] if AT_S - 3 <= e["onset_s"] <= AT_S + EVENT_S]


def test_hysteresis_recovers_the_full_event_duration():
    """Zonder hysterese blijft er een scherf over, met hysterese één event."""
    eeg, hypno = _flickering_arousal()
    zonder = _in_event(detect_arousals(eeg, SF, hypno))
    met = _in_event(detect_arousals(eeg, SF, hypno, hysteresis=True))
    assert zonder, "fixture inert -- het oude pad vindt hier niets"
    d_zonder = max(e["duration_s"] for e in zonder)
    d_met = max(e["duration_s"] for e in met)
    # gemeten 6,4 s -> 20,8 s; de marge is ruim, dus een test die hier net
    # slaagt meet niets meer en hoort te breken
    assert d_met >= 2.0 * d_zonder, (
        f"hysterese verlengt het event niet genoeg: {d_zonder:.1f}s -> {d_met:.1f}s"
    )
    assert len(met) <= len(zonder), (
        f"hysterese hoort scherven samen te voegen, niet te vermeerderen: "
        f"{len(zonder)} -> {len(met)}"
    )


def test_hysteresis_does_not_create_events_below_the_entry_threshold():
    """De instapdrempel blijft ongewijzigd.

    Een verhoging die de instapdrempel nooit haalt maar wel boven exit_ratio
    ligt, hoort GEEN event te worden -- anders is de vlag stiekem een
    drempelverlaging.
    """
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(5)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + 4.0 * np.sin(2 * np.pi * 10.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    s, e = int(AT_S * SF), int((AT_S + EVENT_S) * SF)
    eeg[s:e] += 3.0 * np.sin(2 * np.pi * 10.0 * t[s:e])   # veel te klein
    eeg = eeg * 1e-6
    hypno = ["N2"] * int(DUR_S / 30)
    zonder = _in_event(detect_arousals(eeg, SF, hypno))
    met = _in_event(detect_arousals(eeg, SF, hypno, hysteresis=True))
    assert not zonder
    assert not met, f"hysterese schiep een event uit het niets: {met}"


def test_flag_off_ignores_exit_ratio():
    eeg, hypno = _flickering_arousal()
    base = detect_arousals(eeg, SF, hypno)
    for er in (1.0, 1.9):
        other = detect_arousals(eeg, SF, hypno, hysteresis=False, exit_ratio=er)
        assert other["events"] == base["events"]
        assert other["summary"] == base["summary"]


def test_exit_ratio_matches_the_preregistration():
    """Vastgelegd in docs/arousal_duration_preregistratie.md vóór de meting."""
    assert AROUSAL_EXIT_RATIO == 1.2


def test_profile_flag_reaches_the_profile_dict():
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_HYSTERESIS" in d, name
        assert isinstance(d["AROUSAL_HYSTERESIS"], bool), name
    aan = [n for n, d in SCORING_PROFILES.items() if d["AROUSAL_HYSTERESIS"]]
    assert not aan, f"vlag hoort nergens default aan te staan: {aan}"
