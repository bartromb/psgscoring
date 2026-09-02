"""Opnames met generieke EEG-namen krijgen maar één afleiding.

DE VONDST, GEMETEN OP 24 MESA-OPNAMES
=====================================
`arousal_derivation_channels` loopt na de eerste pick een regiovolgorde af met
sleutels als C4-M1, O2 en F4. MESA noemt zijn kanalen EEG1, EEG2 en EEG3, en
die dragen geen enkele regiosleutel. Resultaat: één afleiding waar er drie
liggen.

Kandidaatdekking van menselijke arousals, 24 opnames:

    EEG1 (wat er nu draait)   47,5 %   pool  908
    EEG2                      42,5 %   pool  875
    EEG3                      47,8 %   pool  820
    ALLE DRIE                 61,5 %   pool 1366

    +14,3 procentpunt, beter op 24 van de 24, p < 0,0001

De multi-afleidingsunie bracht de PSG-IPA-arousal-F1 van 0,442 naar 0,514.
Op MESA heeft die wijziging nooit iets gedaan -- er viel niets te unieren.

DE VALKUIL DIE HIER IN ZIT
==========================
MESA draagt ook EEG1_Off, EEG2_Off en EEG3_Off: offsetkanalen op 1 Hz.
`_NOT_EEG_TOKENS` sluit die niet uit, dus "voeg alle EEG-achtige kanalen toe"
trekt ze mee en dan meet de arousalstap een gelijkstroomlijn.
"""
import numpy as np

from psgscoring.pipeline import arousal_derivation_channels

MESA = ["EKG", "EOG-L", "EOG-R", "EMG", "EEG1", "EEG2", "EEG3",
        "EEG1_Off", "EEG2_Off", "EEG3_Off", "Pleth", "SpO2"]
KLINISCH = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EMG chin", "EOG E1-M2"]


def test_zonder_de_vlag_verandert_er_NIETS():
    """Bestaand gedrag is de default -- ook op MESA."""
    assert arousal_derivation_channels(MESA) == ["EEG1"]
    assert arousal_derivation_channels(KLINISCH) == [
        "EEG F4-M1", "EEG C4-M1", "EEG O2-M1"]


def test_met_de_vlag_komen_de_drie_MESA_kanalen_mee():
    uit = arousal_derivation_channels(MESA, include_generic=True)
    assert uit == ["EEG1", "EEG2", "EEG3"], uit


def test_de_offsetkanalen_blijven_ERBUITEN():
    """EEG1_Off is een 1 Hz gelijkstroomlijn, geen afleiding."""
    uit = arousal_derivation_channels(MESA, include_generic=True)
    assert not any("OFF" in c.upper() for c in uit), uit


def test_de_vlag_raakt_een_opname_met_ECHTE_afleidingen_NIET():
    """Waar de regiovolgorde al drie regio's vindt, mag de terugval zwijgen.

    Anders zou de vlag op een klinische montage stilletjes extra kanalen
    binnenhalen die de picker bewust niet koos.
    """
    met = ["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2", "EEG Fpz"]
    assert (arousal_derivation_channels(met, include_generic=True)
            == arousal_derivation_channels(met))


def test_niet_meer_dan_drie():
    veel = ["EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6"]
    assert len(arousal_derivation_channels(veel, include_generic=True)) == 3


def test_geen_EEG_kanaal_blijft_leeg():
    assert arousal_derivation_channels(["SpO2", "Pleth"],
                                       include_generic=True) == []


def test_de_vlag_bereikt_de_pipeline():
    """Leveringsoppervlak: een profielveld dat de picker nooit bereikt is geen
    vlag maar een decoratie. Dat is in dit project al twee keer gebeurd."""
    from psgscoring.constants import _profile_to_legacy_dict as L
    from psgscoring.profiles import PROFILES

    d = L(PROFILES["aasm_v3_rec"])
    assert "AROUSAL_GENERIC_DERIVATIONS" in d
    assert d["AROUSAL_GENERIC_DERIVATIONS"] is False, (
        "default moet bestaand gedrag zijn")
    assert L(PROFILES["mesa_shhs"])["AROUSAL_GENERIC_DERIVATIONS"] is False, (
        "mesa_shhs moet byte-identiek blijven voor paper v31/v37")


def test_de_vlag_bereikt_de_KANAALKEUZE_niet_alleen_het_profiel():
    """Een profielveld dat de picker niet haalt is decoratie.

    Dit is in dit project twee keer eerder gebeurd: `_topography_warning`
    kreeg een naam die niet bestond, en `score_wake_arousals` bereikte maar
    één van de drie lagen. Daarom hier de KEUZE zelf, niet het veld.
    """
    import mne

    from psgscoring.pipeline import _pick_eeg_multi

    sf, n = 100.0, 3000
    namen = ["EEG1", "EEG2", "EEG3", "EEG1_Off"]
    info = mne.create_info(namen, sf, ch_types="eeg", verbose=False)
    rng = np.random.default_rng(3)
    raw = mne.io.RawArray(rng.normal(0, 1e-5, (len(namen), n)), info,
                          verbose=False)

    zonder = _pick_eeg_multi(raw, {}, include_generic=False)
    met = _pick_eeg_multi(raw, {}, include_generic=True)
    assert [nm for nm, _d, _s in zonder] == ["EEG1"]
    assert [nm for nm, _d, _s in met] == ["EEG1", "EEG2", "EEG3"]
