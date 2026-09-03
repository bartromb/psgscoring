"""Het generieke afleidingspad draagt zijn EIGEN werkpunt: 0,80.

DE KOPPELING, GEMETEN
=====================
Drie afleidingen door de hele keten, met de 10 s-regel aan zoals in productie:

  MESA set 3 (87)   drie op 0,70: index 25,2 / bias +3,3  -- overtelling
                    drie op 0,80: index 20,8 / mens 20,9, dF1 +0,0943 (85/87)
  MESA set 5 (89)   drie op 0,80: dF1 +0,1079, beter op 89/89, |bias| p<1e-4

Maar op PSG-IPA -- echte afleidingsnamen, twaalf scoorders -- zegt de meting
van 30-08 MET de regel: 0,70 -> count-ratio 1,000 en 0,80 -> 0,810. De twee
paden vragen dus een verschillende drempel, en de koppeling hoort bij het PAD
en niet bij het profiel: een globale 0,80 zou de kliniek 19 % te laag laten
tellen.

(Mijn sweep van vanochtend die 0,80 ook voor PSG-IPA aanwees, draaide zonder
de 10 s-regel -- `min_interval_s` default 0,0 bij directe aanroep -- en is
als onderbouwing ongeldig. Zelfde fout als het 32 Hz-snurkkanaal.)
"""
import mne
import numpy as np

from psgscoring.constants import _profile_to_legacy_dict as _L
from psgscoring.pipeline import arousal_derivation_channels
from psgscoring.profiles import PROFILES

MESA = ["EKG", "EOG-L", "EOG-R", "EMG", "EEG1", "EEG2", "EEG3",
        "EEG1_Off", "EEG2_Off", "EEG3_Off"]
KLINISCH = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EMG chin"]


def test_de_defaults_dragen_de_uitrol():
    d = _L(PROFILES["aasm_v3_rec"])
    assert d["AROUSAL_GENERIC_DERIVATIONS"] is True, (
        "uitgerold 2026-09-03: generieke afleidingen aan")
    assert d["AROUSAL_LGBM_THRESHOLD"] == 0.70, (
        "de klinische drempel blijft het besluit van 30-08")
    assert d["AROUSAL_LGBM_THRESHOLD_GENERIC"] == 0.80


def test_mesa_shhs_en_chicago_blijven_byte_identiek():
    for naam in ("mesa_shhs", "chicago_1999"):
        assert _L(PROFILES[naam])["AROUSAL_GENERIC_DERIVATIONS"] is False, naam


def test_generiek_pad_wordt_herkend_klinisch_pad_niet():
    """De koppeling hangt aan de vraag of de TERUGVAL iets toevoegde."""
    basis_m = arousal_derivation_channels(MESA)
    gen_m = arousal_derivation_channels(MESA, include_generic=True)
    assert len(basis_m) == 1 and len(gen_m) == 3, "generiek pad actief op MESA"
    basis_k = arousal_derivation_channels(KLINISCH)
    gen_k = arousal_derivation_channels(KLINISCH, include_generic=True)
    assert basis_k == gen_k == KLINISCH[:3], (
        "op een klinische montage mag de terugval niets doen")


def test_de_koppeling_bereikt_de_LEVERING():
    """summary["lgbm_threshold"] moet 0,80 zijn op een generieke montage en
    0,70 op een klinische — anders is de koppeling decoratie.

    Dit is in dit project drie keer eerder misgegaan (topografiewaarschuwing,
    wake-vlag, en de sweep zonder 10 s-regel), dus expliciet op het
    leveringsoppervlak getoetst.
    """
    import psgscoring

    sf, n_s = 128.0, 480.0
    n = int(sf * n_s)
    rng = np.random.default_rng(7)

    def _raw(namen):
        info = mne.create_info(list(namen), sf, ch_types="misc", verbose=False)
        data = rng.normal(0, 2e-5, (len(namen), n))
        return mne.io.RawArray(data, info, verbose=False)

    hypno = ["N2"] * int(n_s // 30)
    uit_gen = psgscoring.run_pneumo_analysis(
        _raw(["EEG1", "EEG2", "EEG3", "EMG", "Pres"]), hypno=hypno,
        scoring_profile="aasm_v3_rec")
    uit_kli = psgscoring.run_pneumo_analysis(
        _raw(["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EMG chin", "Pres"]),
        hypno=hypno, scoring_profile="aasm_v3_rec")

    s_gen = (uit_gen.get("arousal") or {}).get("summary") or {}
    s_kli = (uit_kli.get("arousal") or {}).get("summary") or {}
    if "lgbm_threshold" not in s_gen or "lgbm_threshold" not in s_kli:
        import pytest
        pytest.skip("LGBM-model niet beschikbaar in deze omgeving")
    assert s_gen["lgbm_threshold"] == 0.80, s_gen["lgbm_threshold"]
    assert s_kli["lgbm_threshold"] == 0.70, s_kli["lgbm_threshold"]
