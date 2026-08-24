"""Arousal: het hybride pad zonder kin-EMG.

WAAROM DEZE TESTS BESTAAN
-------------------------
Het gebundelde model ``arousal_classifier_v3.txt`` splitst 486 keer op
``emg_var_ratio`` (in 279 van de 500 bomen), en ALLE drempels liggen boven nul
(min 0,0157; mediaan 1,86). Op gain is het feature nummer vier.

Zonder EMG-kanaal zet ``_arousal_lgbm_features()`` dat feature op een
constante 0,0. Elke kandidaat gaat dan in alle 486 splits dezelfde
"geen EMG-burst"-kant op; de kansverdeling schuift omlaag en op een VASTE
cutoff (0,80 op de wired profielen) blijft er een fractie over van wat er met
EMG overblijft.

Dat is precies wat er klinisch gebeurde: de MESA-meting die het werkpunt
0,80 koos draaide op de volle montage MET chin-EMG, de productieketen van
YASAFlaskified leverde het kanaal nooit aan. Twee AZORG-opnames gingen van
AI 23,0 naar 4,9 en van 11,0 naar 3,5 — die laatste bij AHI 42.

De guard hieronder is dezelfde vorm als de bestaande ``AROUSAL_LGBM_MIN_SF``:
kan de classifier niet zinvol draaien, dan mogen de kandidaatdrempels ook niet
verruimd worden en is regelgebaseerd het juiste antwoord.
"""
import numpy as np

from psgscoring.arousal import detect_arousals

SF = 100.0
DUR_S = 900


def _recording(seed=11):
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + 4.0 * np.sin(2 * np.pi * 10.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    for at in (120.0, 300.0, 480.0, 660.0):
        s, e = int(at * SF), int((at + 6.0) * SF)
        eeg[s:e] = (28.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                    + 28.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                    + rng.normal(0.0, 1.0, e - s))
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def _emg(seed=12):
    rng = np.random.default_rng(seed)
    emg = rng.normal(0.0, 5.0, int(DUR_S * SF))
    for at in (120.0, 300.0, 480.0, 660.0):
        s, e = int(at * SF), int((at + 4.0) * SF)
        emg[s:e] += rng.normal(0.0, 40.0, e - s)
    return emg * 1e-6


# ══════════════════════════════════════════════════════════════
# T1 — de guard zelf
# ══════════════════════════════════════════════════════════════

def test_without_emg_the_hybrid_is_skipped_with_a_reason(monkeypatch):
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    out = detect_arousals(eeg, SF, hypno, emg_data=None, lgbm_threshold=0.80)
    s = out["summary"]
    assert s.get("lgbm_available") is False, (
        "zonder kin-EMG draait de classifier op een constant-0 feature waar hij "
        "486 keer op splitst; de samenvatting hoort te melden dat hij oversloeg"
    )
    assert str(s.get("lgbm_skipped_reason", "")).startswith("no_emg_channel"), (
        f"reden ontbreekt of is verkeerd: {s.get('lgbm_skipped_reason')!r}"
    )


def test_without_emg_the_result_equals_the_rule_based_path(monkeypatch):
    """De kern: geen kandidatenlijst, geen gedecimeerde lijst — de
    regelgebaseerde uitkomst, exact zoals v0.22 die gaf."""
    eeg, hypno = _recording()
    regels = detect_arousals(eeg, SF, hypno, emg_data=None)

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    hybride = detect_arousals(eeg, SF, hypno, emg_data=None,
                              lgbm_threshold=0.80)
    assert hybride["events"] == regels["events"], (
        f"{len(hybride['events'])} events zonder EMG tegen "
        f"{len(regels['events'])} regelgebaseerd"
    )
    assert "lgbm_n_post" not in hybride["summary"], (
        "lgbm_n_post suggereert dat er gefilterd is"
    )


def test_with_emg_the_hybrid_still_runs(monkeypatch):
    """De guard mag het normale pad niet uitschakelen — anders meet test 2
    niets en is de classifier stilzwijgend dood."""
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    out = detect_arousals(eeg, SF, hypno, emg_data=_emg(),
                          lgbm_threshold=0.80)
    s = out["summary"]
    assert s.get("lgbm_available") is True, s.get("lgbm_skipped_reason")
    assert "lgbm_n_post" in s, "de classifier heeft niet gefilterd"


def test_an_all_zero_emg_channel_is_not_an_emg_channel(monkeypatch):
    """Een aanwezig maar dood kanaal geeft var 0 in pre EN cand, dus
    emg_var_ratio 0/eps = 0 — exact dezelfde degeneratie als geen kanaal."""
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    out = detect_arousals(eeg, SF, hypno,
                          emg_data=np.zeros(int(DUR_S * SF)),
                          lgbm_threshold=0.80)
    assert out["summary"].get("lgbm_available") is False
    assert str(out["summary"].get("lgbm_skipped_reason", "")).startswith(
        "no_emg_channel")


# ══════════════════════════════════════════════════════════════
# T2 — emg_confirmed liegt niet meer
# ══════════════════════════════════════════════════════════════

def test_emg_confirmed_is_false_when_no_emg_test_ran():
    """``emg_confirmed`` stond op True als DEFAULT, ook op NREM-events waar de
    test niet eens van toepassing is en op montages zonder kin-EMG. De
    samenvatting meldde dan ``n_emg_confirmed == n_arousals``: een bevestiging
    die nooit heeft plaatsgevonden."""
    eeg, hypno = _recording()
    out = detect_arousals(eeg, SF, hypno, emg_data=None)
    assert out["events"], "geen events — deze fixture meet niets"
    assert not any(e["emg_confirmed"] for e in out["events"]), (
        "events melden EMG-bevestiging terwijl er geen EMG-kanaal is"
    )
    assert out["summary"]["n_emg_confirmed"] == 0
