"""De duurbepaling van een beenbeweging volgt AASM regel 4.A.

Onset bij een stijging van 8 uV boven rust; het EINDE is het begin van een
periode van minstens 0,5 s waarin het EMG NIET boven 2 uV boven rust komt.
Twee verschillende drempels dus.

Deze module gebruikte 8 uV voor allebei en mat de duur als de tijd boven
8 uV -- stelselmatig korter. Een beweging die net boven de drempel uitkomt
zakt daardoor onder het minimum van 0,5 s en verdwijnt. Op PSG-IPA SN5 haalt
90 % van de gemiste slaapbewegingen de drempel wel, met een mediane piek van
12,6 uV boven rust: precies het marginale geval.

De vlag staat default UIT; deze tests leggen beide gedragingen vast.
"""
import numpy as np

from psgscoring.plm import _detect_lm_channel

SF = 100.0


def _signaal(piek_uv, plateau_s=0.30, staart_uv=4.0, staart_s=1.2, n_s=60):
    """Een marginale beweging: kort boven 8 uV, daarna een trage staart.

    EMG is ruis, geen gelijkspanning -- de detector bandfiltert vóór de RMS,
    dus een DC-sprong verdwijnt volledig. De burst is daarom een verhoogde
    ruisamplitude, zoals echt spier-EMG.

    De staart blijft boven de AASM-einde-drempel van 2 uV, dus onder die regel
    loopt de beweging door; onder de oude regel stopt hij bij het plateau.
    """
    n = int(SF * n_s)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.5, n)                     # rust ~0,5 uV RMS
    t0 = int(SF * 20)
    a, b = t0, t0 + int(SF * plateau_s)
    x[a:b] = rng.normal(0, piek_uv, b - a)
    if staart_s > 0:
        c = b + int(SF * staart_s)
        x[b:c] = rng.normal(0, staart_uv, c - b)
    return x


def test_the_old_rule_loses_a_marginal_movement():
    """Legt het defect vast: zonder dit meet de volgende test niets."""
    lms = _detect_lm_channel(_signaal(piek_uv=12.0), SF, unit="uV",
                             offset_aasm=False)
    assert len(lms) == 0, (
        f"de oude regel vindt hem al ({lms}); dan toont deze fixture het "
        "probleem niet")


def test_the_aasm_rule_keeps_it():
    lms = _detect_lm_channel(_signaal(piek_uv=12.0), SF, unit="uV",
                             offset_aasm=True)
    assert len(lms) == 1, f"AASM-regel vindt hem niet: {lms}"
    d = lms[0]["duration_s"]
    assert 0.5 <= d <= 10.0, f"duur buiten de AASM-grenzen: {d}"
    assert d > 0.30, (
        f"duur {d} is niet langer dan het plateau van 0,30 s -- het einde "
        "wordt kennelijk nog op 8 uV bepaald")


def test_a_clear_movement_is_found_by_both():
    """De nieuwe regel mag geen gewone bewegingen anders behandelen."""
    x = _signaal(piek_uv=40.0, plateau_s=1.5, staart_uv=0.0, staart_s=0.0)
    oud = _detect_lm_channel(x, SF, unit="uV", offset_aasm=False)
    nieuw = _detect_lm_channel(x, SF, unit="uV", offset_aasm=True)
    assert len(oud) == 1 and len(nieuw) == 1, (oud, nieuw)


def test_nothing_is_found_in_quiet_signal():
    x = np.random.default_rng(7).normal(0, 0.4, int(SF * 60))
    for flag in (False, True):
        assert _detect_lm_channel(x, SF, unit="uV", offset_aasm=flag) == [], flag


def test_the_flag_defaults_to_the_old_behaviour():
    from psgscoring.plm import analyze_plm
    import inspect
    assert inspect.signature(analyze_plm).parameters["offset_aasm"].default is False


def test_the_env_override_exists_so_arms_can_be_separated(monkeypatch):
    """Zonder override meet een vergelijking van beide armen nul verschil.

    Dat is op 23-08-2026 al een keer gebeurd met AROUSAL_LGBM: de pipeline gaf
    een expliciete bool door, de env kwam nooit aan bod, en 30/30 kwam
    identiek uit -- wat eruitzag als een resultaat.
    """
    from psgscoring.pipeline import _plm_offset_aasm

    prof = {"PLM_OFFSET_AASM": False}
    assert _plm_offset_aasm(prof) is False
    monkeypatch.setenv("PSGSCORING_PLM_OFFSET_AASM", "1")
    assert _plm_offset_aasm(prof) is True
    monkeypatch.setenv("PSGSCORING_PLM_OFFSET_AASM", "0")
    assert _plm_offset_aasm({"PLM_OFFSET_AASM": True}) is False
