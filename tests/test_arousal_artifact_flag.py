"""De arousalstap mag de artefactlijst negeren, en dat moet echt doorwerken.

Gemeten op MESA n=30, gepaard, arousal-F1 tegen de NSRR-annotatie:

    geen lijst              0,421   (0 % gevlagd)
    huidige regel 500 uV    0,338   (19,9 %)
    yasa.art_detect         0,356   (2,1 %)

Geen lijst wint van beide op 30 van de 30, p = 1,7e-06, en het teken
repliceert op PSG-IPA. Het probleem is de onderdrukking zelf: art_detect gooit
tien keer minder weg en scoort toch niet beter, want hij selecteert de epochs
met de grootste variantie -- precies waar arousals zitten.

Default blijft True (huidig gedrag); omzetten verandert de arousal-index en via
de RERA-koppeling de RDI.
"""
import numpy as np
import pytest


def test_the_list_is_ignored_everywhere_except_on_the_pinned_profiles():
    """Sinds 23-08-2026 (gebruikersbeslissing) negeert de arousalstap de lijst.

    Behalve op de vijf profielen die een externe regelset of een gepubliceerde
    dataset-analyse reproduceren: daar zou het de gereproduceerde uitkomst
    verschuiven.
    """
    from psgscoring.profiles import get_profile, list_profiles

    gepind = {"mesa_shhs", "chicago_1999", "cms_medicare", "aasm_v1_rec",
           "aasm_v2_rec"}
    for name in list_profiles():
        got = get_profile(name).post_processing.arousal_uses_artifact_epochs
        verwacht = name in gepind
        assert got is verwacht, (
            f"{name}: artefactlijst gebruiken={got}, verwacht {verwacht}")


def test_the_registry_carries_the_flag():
    from psgscoring.constants import SCORING_PROFILES

    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_USES_ARTIFACT_EPOCHS" in d, name
        assert isinstance(d["AROUSAL_USES_ARTIFACT_EPOCHS"], bool), name


def _raw_with_arousals():
    mne = pytest.importorskip("mne")
    sf, minutes = 64.0, 30
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(4)
    eeg = rng.normal(0, 20e-6, n)
    # arousals elke 90 s, allemaal in de eerste helft van de nacht
    for start in range(60, 60 * minutes // 2, 90):
        a, b = int(start * sf), int((start + 5) * sf)
        eeg[a:b] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    info = mne.create_info(["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin"],
                           sf, ["misc", "misc", "eeg", "emg"])
    raw = mne.io.RawArray(
        np.vstack([np.sin(2 * np.pi * 0.25 * t), np.full(n, 97.0), eeg,
                   rng.normal(0, 5e-6, n)]), info, verbose=False)
    return raw, ["N2"] * int(np.ceil(raw.times[-1] / 30.0))


def test_suppressing_the_arousal_rich_half_costs_events_and_the_flag_prevents_it(monkeypatch):
    """Fixture die discrimineert: de artefact-epochs bedekken juist de arousals."""
    import psgscoring

    raw, hypno = _raw_with_arousals()
    helft = len(hypno) // 2
    art = list(range(helft))          # exact de epochs met arousals erin

    # Regelpad: de op MESA getrainde classifier verwerpt synthetische
    # 10 Hz-bursts, en dan meet de fixture niets meer.
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")

    def n_ar(**kw):
        out = psgscoring.run_pneumo_analysis(
            raw.copy(), hypno=hypno, scoring_profile="aasm_v3_rec",
            artifact_epochs=art, **kw)
        return len(out["arousal"].get("events", []))

    monkeypatch.setenv("PSGSCORING_AROUSAL_USES_ARTIFACT_EPOCHS", "1")
    met = n_ar()
    monkeypatch.setenv("PSGSCORING_AROUSAL_USES_ARTIFACT_EPOCHS", "0")
    zonder = n_ar()

    assert zonder > met, (
        "de vlag doet niets: met onderdrukking "
        f"{met} arousals, zonder {zonder} -- terwijl de artefact-epochs "
        "precies de arousals bedekken")


def test_the_flag_is_inert_when_there_are_no_artifact_epochs(monkeypatch):
    import psgscoring

    raw, hypno = _raw_with_arousals()

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")

    def n_ar():
        out = psgscoring.run_pneumo_analysis(
            raw.copy(), hypno=hypno, scoring_profile="aasm_v3_rec",
            artifact_epochs=[])
        return len(out["arousal"].get("events", []))

    monkeypatch.setenv("PSGSCORING_AROUSAL_USES_ARTIFACT_EPOCHS", "1")
    a = n_ar()
    monkeypatch.setenv("PSGSCORING_AROUSAL_USES_ARTIFACT_EPOCHS", "0")
    assert n_ar() == a, "zonder artefact-epochs hoort de vlag niets te doen"
