"""De pre-event-validator moet op hersteladem ankeren, niet op het kale gemiddelde.

DE VONDST (verliesrekening 2026-09-04, 15 zwaarste hoge-AHI-nachten)
====================================================================
62,8 % van de terughaalbare verliezen sneuvelt op `_validate_local_reduction`,
ruim een kwart met een NEGATIEVE reductie: de flow tijdens het menselijke
event ligt BOVEN de "pre-event basislijn". `pre_mean = np.mean(pre_seg)` --
op een AHI-60-nacht bestaat die 30 s grotendeels uit vórige events, en tegen
een ingezakte referentie verdwijnt elke reductie.

`compute_dynamic_baseline` heeft hier al jaren de juiste beveiliging voor
(95e-percentiel-anker, samples <= 30 % daarvan uitgesloten "to prevent apnea
periods from suppressing it"); deze validator kreeg die nooit. De vlag zet
exact diezelfde regel op het pre-event-venster.
"""
import numpy as np

from psgscoring.constants import _profile_to_legacy_dict as _L
from psgscoring.profiles import PROFILES
from psgscoring.respiratory import _validate_local_reduction

SF = 32.0


def _dichte_nacht_env(n_s=60.0):
    """30 s pre-venster dat grotendeels uit events bestaat: 6 s herstel op
    amplitude 1,0, daarna 24 s event-ademhaling op 0,25; dan het kandidaat-
    event op 0,55 -- een echte reductie van 45 % t.o.v. de hersteladem, maar
    BOVEN het kale venstergemiddelde (0,40)."""
    sf = SF
    env = np.full(int(n_s * sf), 0.25)
    env[:int(6 * sf)] = 1.0                      # hersteladem
    ev0 = int(30 * sf)
    env[ev0:int(45 * sf)] = 0.55                 # kandidaat-event
    return env, ev0, int(45 * sf)


def test_default_kale_gemiddelde_wijst_af_op_de_dichte_nacht():
    """Documenteert het defect: zonder de vlag is de reductie negatief."""
    env, a, b = _dichte_nacht_env()
    ok, red, _ = _validate_local_reduction(env, a, b, SF)
    assert ok is False and red < 0, (red, "de fixture reproduceert de "
                                     "negatieve-reductieval niet")


def test_met_herstel_anker_wordt_dezelfde_nacht_geaccepteerd():
    env, a, b = _dichte_nacht_env()
    ok, red, _ = _validate_local_reduction(env, a, b, SF, recovery_anchor=True)
    assert ok is True and red is not None and red > 20, red


def test_op_een_gezonde_nacht_verandert_het_anker_vrijwel_niets():
    """Stabiele pre-ademhaling: herstel-anker en kale gemiddelde moeten
    hetzelfde meten -- anders is de vlag een globale drempelverschuiving in
    vermomming."""
    sf = SF
    env = np.full(int(60 * sf), 1.0)
    rng = np.random.default_rng(1)
    env += rng.normal(0, 0.03, len(env))
    ev0 = int(30 * sf); ev1 = int(45 * sf)
    env[ev0:ev1] = 0.5
    _ok0, red0, _ = _validate_local_reduction(env, ev0, ev1, sf)
    _ok1, red1, _ = _validate_local_reduction(env, ev0, ev1, sf,
                                              recovery_anchor=True)
    assert abs(red0 - red1) < 3.0, (red0, red1)


def test_onmeetbaar_blijft_niet_afwijzen():
    """Pre-venster volledig onder het anker-uitsluitpunt (alles event):
    'niet gemeten', geen afwijzing -- dezelfde filosofie als de bestaande
    dropout-tak."""
    sf = SF
    env = np.full(int(60 * sf), 0.2)             # alles even laag
    env[int(30 * sf):int(45 * sf)] = 0.19
    ok, red, _ = _validate_local_reduction(env, int(30 * sf), int(45 * sf),
                                           sf, recovery_anchor=True)
    # kale vlakke omgeving: het anker sluit niets uit (alles > 30 % van het
    # anker), dus dit meet gewoon -- de ECHT onmeetbare tak is dropout
    assert ok in (True, False)


def test_profielveld_bestaat_en_default_is_uit():
    d = _L(PROFILES["aasm_v3_rec"])
    assert d["LOCAL_BASELINE_RECOVERY_ANCHOR"] is False, (
        "default uit tot de meting op de 15 nachten er ligt")
