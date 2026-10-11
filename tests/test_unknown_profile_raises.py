"""Een onbekende profielnaam is een fout, geen stille terugval.

`run_pneumo_analysis` deed `SCORING_PROFILES.get(naam, SCORING_PROFILES
["aasm_v3_rec"])`: een typefout in de profielnaam scoorde stil op het
defaultprofiel en schreef de FOUTE naam in `meta["scoring_profile"]`. Een
meting die "aasm_v3_breath_daul" vroeg kreeg zo een rec-uitslag met een
breath_dual-etiket -- precies het soort fout dat in een vergelijking als
"geen verschil" leest.

`profiles.get_profile` gooit al een KeyError; de pipeline hoort hetzelfde
te doen.
"""
import warnings

import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 300
N = int(DUR_S * SF)


def _raw():
    t = np.arange(N) / SF
    flow = np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(N, 96.0)
    info = mne.create_info(["Resp nasal", "SaO2"], SF, ["misc", "misc"])
    return mne.io.RawArray(np.vstack([flow, spo2]), info, verbose="ERROR")


HYPNO = ["N2"] * (DUR_S // 30)


def test_een_typefout_in_de_profielnaam_gooit_een_keyerror():
    import psgscoring
    with pytest.raises(KeyError, match="aasm_v3_breath_daul"):
        psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                       scoring_profile="aasm_v3_breath_daul")


def test_de_foutmelding_noemt_de_bestaande_profielen():
    import psgscoring
    with pytest.raises(KeyError, match="aasm_v3_rec"):
        psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                       scoring_profile="bestaat_niet")


def test_een_legacy_alias_blijft_werken():
    """De aliasresolutie (strict/standard/sensitive) is niet wat verandert."""
    import psgscoring
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                             scoring_profile="standard")
    assert out["meta"]["scoring_profile"] == "aasm_v3_rec"
    assert out["meta"]["scoring_input_name"] == "standard"


def test_de_pipeline_en_get_profile_zijn_het_eens():
    """Dezelfde naam, hetzelfde oordeel -- anders kan een aanroeper de ene
    laag niet gebruiken om de andere te controleren."""
    import psgscoring
    from psgscoring.profiles import get_profile
    with pytest.raises(KeyError):
        get_profile("aasm_v3_breath_daul")
    with pytest.raises(KeyError):
        psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                       scoring_profile="aasm_v3_breath_daul")
