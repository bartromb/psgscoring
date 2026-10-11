"""Het hypopneemasker snijdt de apneuband eruit, en dat knipt kandidaten stuk.

WAT ER GEBEURT
--------------
De gecorrigeerde pas bouwt zijn envelope-masker als

    (norm < HYPOPNEA_THRESHOLD) & ~(norm < APNEA_THRESHOLD)

en het ademteugmasker met dezelfde uitsluiting (`ratio >= APNEA_THRESHOLD`).
Bevestigde apneus zijn op dat moment al uitgesloten via `excl` (±5 s), dus
de uitsluiting raakt alleen samples onder 10 % die GEEN apneu werden: een
korte bijna-stilstand (<10 s) midden in een hypopneu. Die valt uit het
masker, de hypopneu breekt in twee stukken, en twee stukken van 7 s halen
de >=10 s-eis geen van beide. Een kandidaat van 18 s met een 30 %-daling
verdwijnt.

DE STAND
--------
`hypopnea_mask_excludes_apnea_band` staat op ALLE profielen op True (het
oude gedrag), ook de klinische. Dit verandert kandidaten op elke opname en
moet eerst gemeten worden. Deze tests documenteren beide standen.
"""
import numpy as np
from scipy.ndimage import find_objects, label

from psgscoring.constants import APNEA_THRESHOLD, HYPOPNEA_THRESHOLD
from psgscoring.respiratory import _breath_is_reduced, _hypopnea_envelope_mask

SF = 32.0


def _norm_7_4_7():
    """Genormaliseerde, gladde envelope: 1,0, dan 0,5 / 0,05 / 0,5 gedurende
    7 / 4 / 7 s, dan weer 1,0."""
    delen = [(60.0, 1.0), (7.0, 0.5), (4.0, 0.05), (7.0, 0.5), (60.0, 1.0)]
    return np.concatenate([np.full(int(d * SF), v) for d, v in delen])


def _runs_van_minstens(mask, min_s):
    lab, _ = label(mask)
    return [sl[0] for sl in find_objects(lab)
            if sl is not None and (sl[0].stop - sl[0].start) / SF >= min_s]


def test_zonder_uitsluiting_een_kandidaat_van_18_s():
    m = _hypopnea_envelope_mask(_norm_7_4_7(), HYPOPNEA_THRESHOLD, APNEA_THRESHOLD,
                                exclude_apnea_band=False)
    runs = _runs_van_minstens(m, 10.0)
    assert len(runs) == 1, runs
    assert (runs[0].stop - runs[0].start) / SF == 18.0


def test_met_uitsluiting_geen_enkele_kandidaat_van_10_s():
    """Het OUDE gedrag, gedocumenteerd: 7 + 7 s, allebei te kort."""
    m = _hypopnea_envelope_mask(_norm_7_4_7(), HYPOPNEA_THRESHOLD, APNEA_THRESHOLD,
                                exclude_apnea_band=True)
    assert _runs_van_minstens(m, 10.0) == []
    assert len(_runs_van_minstens(m, 7.0)) == 2


def test_de_ademteugregel_volgt_dezelfde_schakelaar():
    assert _breath_is_reduced(0.5, HYPOPNEA_THRESHOLD, APNEA_THRESHOLD, True)
    assert not _breath_is_reduced(0.05, HYPOPNEA_THRESHOLD, APNEA_THRESHOLD, True)
    assert _breath_is_reduced(0.05, HYPOPNEA_THRESHOLD, APNEA_THRESHOLD, False)
    assert not _breath_is_reduced(0.8, HYPOPNEA_THRESHOLD, APNEA_THRESHOLD, False)


def test_veld_staat_overal_op_het_oude_gedrag():
    from psgscoring.profiles import PROFILES, PostProcessingRules
    assert PostProcessingRules().hypopnea_mask_excludes_apnea_band is True
    aan = [n for n, p in PROFILES.items()
           if p.post_processing.hypopnea_mask_excludes_apnea_band is not True]
    assert aan == [], f"nog niet gemeten; hoort overal uit te staan: {aan}"


def test_veld_bereikt_de_legacy_dict_en_de_detector():
    import inspect

    from psgscoring import respiratory
    from psgscoring.constants import SCORING_PROFILES
    assert all(d["HYPOPNEA_MASK_EXCLUDES_APNEA_BAND"] is True
               for d in SCORING_PROFILES.values())
    bron = inspect.getsource(respiratory.detect_respiratory_events)
    assert "HYPOPNEA_MASK_EXCLUDES_APNEA_BAND" in bron
