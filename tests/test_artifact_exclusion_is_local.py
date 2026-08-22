"""Een uitgesloten epoch mag de detectie ERBUITEN niet raken.

De rollende basislijn kijkt 120 s terug over het slaapmasker, en dat masker
sluit artefact-epochs uit. Een uitgesloten blok zou dus het venster van zijn
buren kunnen uithollen, en dan zou de schade verder reiken dan de uitgesloten
epochs zelf.

Dat was de openstaande verklaring voor een oncomfortabel cijfer: op MESA kost
2,1 % van de epochs uitsluiten 19 % van de recall, een verrijking van bijna
een factor tien. Gemeten op 23-08-2026 en WEERLEGD -- het effect is lokaal.
Wat de verrijking dan wel verklaart, is dat een variantie-gebaseerde
artefactdetector juist de epochs kiest waar arousals zitten.

Deze test legt de eigenschap vast: gaat iemand aan de basislijn sleutelen en
begint uitsluiting wél te lekken, dan valt dit om.
"""

import numpy as np
import pytest

SF, MIN = 64.0, 40
SPLIT = 60 * MIN // 2


def _signaal():
    n = int(SF * 60 * MIN)
    t = np.arange(n) / SF
    rng = np.random.default_rng(11)
    # niet-stationair: achtergrondvermogen loopt op met een factor 3, zodat een
    # uitgehongerd basislijnvenster op een verkeerde waarde zou landen
    schaal = np.linspace(1.0, 3.0, n)
    eeg = rng.normal(0, 20e-6, n) * schaal
    onsets = list(range(SPLIT + 60, 60 * MIN - 60, 90))
    for s0 in onsets:
        a, b = int(s0 * SF), int((s0 + 5) * SF)
        eeg[a:b] += 70e-6 * schaal[a:b] * np.sin(2 * np.pi * 10.0 * t[a:b])
    return eeg, ["N2"] * int(np.ceil(n / SF / 30)), onsets


def _na_de_splitsing(eeg, hypno, art, monkeypatch):
    from psgscoring.arousal import detect_arousals

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")   # regelpad
    r = detect_arousals(eeg, SF, hypno, emg_data=None, artifact_epochs=art)
    return len([e["onset_s"] for e in (r.get("events") or [])
                if e["onset_s"] >= SPLIT])


def test_the_fixture_can_show_an_effect(monkeypatch):
    """Zonder deze controle zegt "geen lek" niets."""
    eeg, hypno, _ = _signaal()
    basis = _na_de_splitsing(eeg, hypno, None, monkeypatch)
    assert basis >= 8, f"te weinig arousals gedetecteerd om iets te tonen: {basis}"

    met_arousals = list(range((SPLIT + 60) // 30, (SPLIT + 60) // 30 + 10))
    minder = _na_de_splitsing(eeg, hypno, met_arousals, monkeypatch)
    assert minder < basis, (
        "epochs MET arousals uitsluiten verandert de telling niet; deze "
        f"fixture meet niets ({minder} tegen {basis})")


@pytest.mark.parametrize("naam,blok", [
    ("vlak voor de arousals", list(range(1000 // 30, 1200 // 30))),
    ("ver weg",               list(range(120 // 30, 320 // 30))),
    ("verspreid",             [e for e in range(SPLIT // 30) if e % 5 == 0]),
])
def test_excluding_epochs_without_arousals_does_not_change_detection_elsewhere(
        naam, blok, monkeypatch):
    eeg, hypno, _ = _signaal()
    basis = _na_de_splitsing(eeg, hypno, None, monkeypatch)
    met = _na_de_splitsing(eeg, hypno, blok, monkeypatch)
    assert met == basis, (
        f"uitsluiting lekt ({naam}): {len(blok)} epochs zonder arousals "
        f"uitgesloten verandert de telling erbuiten van {basis} naar {met}")
