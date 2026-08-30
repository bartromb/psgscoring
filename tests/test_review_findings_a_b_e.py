"""
Drie bevindingen uit de scoringsketen-review, elk met een eigen valkuil.

A. Het rapport noemde een criterium dat de software niet toepast.
B. Te lange arousalregio's verdwenen zonder spoor.
E. In REM meten teller en noemer verschillende grootheden.
"""
import numpy as np

SF, DUUR = 128.0, 900


# ══════════════════════════════════════════════════════════════════════
#  A. Het gerapporteerde criterium moet waar zijn
# ══════════════════════════════════════════════════════════════════════

def test_de_arousaltak_geldt_alleen_waar_hij_werkelijk_draait():
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.pipeline import arousal_limb_is_effective

    # envelope-detector zonder bedrading -> tak draait NIET
    assert arousal_limb_is_effective(SCORING_PROFILES["aasm_v3_rec"]) is False
    # breath_graded leest de arousals rechtstreeks in stap 7b -> WEL
    assert arousal_limb_is_effective(SCORING_PROFILES["aasm_v3_breath"]) is True
    # desaturatie-only profiel -> nooit
    assert arousal_limb_is_effective(SCORING_PROFILES["cms_medicare"]) is False


def test_het_criterium_belooft_geen_arousaltak_die_uitstaat():
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.pipeline import _hypopnea_criterion_str, arousal_limb_is_effective

    for naam, d in SCORING_PROFILES.items():
        tekst = _hypopnea_criterion_str(d) or ""
        belooft = "OR arousal" in tekst
        assert belooft == arousal_limb_is_effective(d), (
            f"{naam}: het criterium zegt {'wel' if belooft else 'niet'} "
            f"'OR arousal', maar de tak draait "
            f"{'wel' if arousal_limb_is_effective(d) else 'niet'} -- {tekst}")


def test_een_toegestane_maar_uitgeschakelde_tak_wordt_benoemd():
    """Zwijgen zou net zo misleidend zijn als liegen."""
    from psgscoring.constants import SCORING_PROFILES
    from psgscoring.pipeline import _hypopnea_criterion_str

    tekst = _hypopnea_criterion_str(SCORING_PROFILES["aasm_v3_rec"])
    assert "not enabled in this profile" in tekst


# ══════════════════════════════════════════════════════════════════════
#  B. Weggegooide lange regio's zijn telbaar
# ══════════════════════════════════════════════════════════════════════

def _eeg_met_lange_burst(burst_s):
    t = np.arange(int(SF * DUUR)) / SF
    rng = np.random.default_rng(3)
    eeg = 40 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 8, t.size)
    a, b = int(200 * SF), int((200 + burst_s) * SF)
    tt = t[a:b]
    eeg[a:b] = (55 * np.sin(2 * np.pi * 9.5 * tt)
                + 35 * np.sin(2 * np.pi * 20.0 * tt) + rng.normal(0, 8, b - a))
    return eeg


def test_een_te_lange_regio_wordt_geteld_in_plaats_van_stil_verdwenen():
    from psgscoring.arousal import AROUSAL_MAX_DUR_S, detect_arousals

    r = detect_arousals(_eeg_met_lange_burst(40.0), SF, ["N2"] * (DUUR // 30))
    s = r["summary"]
    assert s["n_too_long_discarded"] >= 1, (
        "een regio van 40 s hoort boven de grens te vallen en geteld te worden")
    assert s["too_long_discarded_s"] > AROUSAL_MAX_DUR_S
    assert s["max_duration_s"] == AROUSAL_MAX_DUR_S, (
        "de grens zelf hoort erbij te staan, anders is de telling niet te duiden")


def test_nul_is_ook_zichtbaar():
    """Het verschil tussen 'niets weggegooid' en 'niet gekeken'."""
    from psgscoring.arousal import detect_arousals

    s = detect_arousals(_eeg_met_lange_burst(4.0), SF, ["N2"] * (DUUR // 30))["summary"]
    assert s["n_too_long_discarded"] == 0
    assert "too_long_discarded_s" in s


def test_de_teller_overleeft_de_multi_wrapper():
    from psgscoring.arousal import detect_arousals_multi

    eeg = _eeg_met_lange_burst(40.0)
    r = detect_arousals_multi([("C4-M1", eeg, SF), ("O2-M1", eeg.copy(), SF)],
                              SF, ["N2"] * (DUUR // 30))
    assert r["summary"]["n_too_long_discarded"] >= 2, (
        "elke afleiding gooit zijn eigen regio weg; de union ziet die nooit")


# ══════════════════════════════════════════════════════════════════════
#  E. REM: teller en noemer dezelfde grootheid
# ══════════════════════════════════════════════════════════════════════

def _rem_eeg():
    t = np.arange(int(SF * DUUR)) / SF
    rng = np.random.default_rng(9)
    eeg = 30 * np.sin(2 * np.pi * 6.0 * t) + rng.normal(0, 8, t.size)   # theta
    for st in range(120, DUUR - 120, 150):
        a, b = int(st * SF), int((st + 4) * SF)
        eeg[a:b] += 45 * np.sin(2 * np.pi * 9.5 * t[a:b])               # alpha
    return eeg


def test_elk_profiel_houdt_de_rem_basislijn_ongewijzigd():
    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert d["AROUSAL_REM_BASELINE_ALPHA"] is False, (
            f"{naam} verandert de REM-telling; dat vraagt eerst een meting")


def test_de_vlag_verandert_de_rem_detectie():
    from psgscoring.arousal import detect_arousals

    hyp = ["R"] * (DUUR // 30)
    uit = detect_arousals(_rem_eeg(), SF, hyp)["summary"]["n_arousals"]
    aan = detect_arousals(_rem_eeg(), SF, hyp, rem_alpha_baseline=True)["summary"]["n_arousals"]
    assert aan > uit, (
        f"de alpha-basislijn hoort in REM gevoeliger te zijn (theta is daar de "
        f"achtergrond en drukt de noemer op); kreeg uit={uit} aan={aan}")


def test_de_vlag_raakt_nrem_niet():
    from psgscoring.arousal import detect_arousals

    eeg = _eeg_met_lange_burst(4.0)
    hyp = ["N2"] * (DUUR // 30)
    uit = detect_arousals(eeg, SF, hyp)["summary"]["n_arousals"]
    aan = detect_arousals(eeg, SF, hyp, rem_alpha_baseline=True)["summary"]["n_arousals"]
    assert uit == aan, "de vlag hoort alleen REM te raken"
