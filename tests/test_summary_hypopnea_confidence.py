"""De samenvatting stratificeert ALLEEN apneus op betrouwbaarheid.

WAT ER MIS WAS
--------------
`_compute_summary` liet hypopneeën meetellen in `oahi_conf60`,
`oahi_thresholds`, `oahi_sweep` en `ahi_excl_noise` zolang hun `confidence`
boven de drempel lag, met als toelichting "hypopneeën krijgen altijd
conf=0,70 en tellen dus altijd mee". Dat was de bedoeling: de betrouwbaarheid
van een hypopneu is geen classificatiezekerheid (obstructief/centraal/
gemengd) maar — op de ademteugdetector — `p_scored`, hoe goed het event aan
de AASM-conjunctie voldoet, een getal tussen `strictness` (0,30) en 1,0.

Gevolg op `aasm_v3_breath` en zijn familie: elke hypopneu met p_scored onder
0,60 viel uit de officiële OAHI, uit de drempelgevoeligheidstabel en uit de
robuustheidsgraad, terwijl diezelfde hypopneu wél in `ahi_total` stond. Twee
indices die dezelfde events horen te tellen, telden er verschillende.

De reparatie: hypopneeën tellen altijd, in elke confidence-stratificatie; de
stratificatie gaat alleen over apneus. `p_scored` blijft op de events staan.
"""
import pytest

from psgscoring.respiratory import _compute_summary

HYPNO = ["N2"] * 240          # 2 h slaap


def _breath_profile_events():
    """Apneus met hoge zekerheid, hypopneeën met p_scored over [0,3, 1,0]."""
    ev = []
    for i, p in enumerate([0.31, 0.45, 0.55, 0.59, 0.60, 0.72, 0.88, 1.0]):
        ev.append({"type": "hypopnea", "onset_s": 100.0 + i * 300, "duration_s": 20.0,
                   "stage": "N2", "epoch": int((100 + i * 300) // 30),
                   "confidence": p, "p_scored": p})
    # Strikt boven 0,85: `_oahi_at` hanteert voor apneus `>`, niet `>=`.
    for i, c in enumerate([0.86, 0.90, 0.95]):
        ev.append({"type": "obstructive", "onset_s": 3000.0 + i * 300,
                   "duration_s": 15.0, "stage": "N2",
                   "epoch": int((3000 + i * 300) // 30), "confidence": c})
    return ev


def test_hypopneeen_tellen_op_elke_confidence_drempel_mee():
    s = _compute_summary(_breath_profile_events(), HYPNO)
    assert s["oahi_thresholds"]["0.60"] == s["oahi_all"], s["oahi_thresholds"]
    assert s["oahi_conf60"] == s["oahi_all"]
    # Alle apneus zitten boven 0,85, dus ook de strengste rij telt alles.
    assert s["oahi_thresholds"]["0.85"] == s["oahi_all"]


def test_de_sweep_is_vlak_als_alleen_hypopneeen_varieren():
    """De robuustheidsgraad hoort over de classifier te gaan, niet over
    `p_scored`: drie zekere apneus plus hypopneeën geven breedte 0 en graad A."""
    s = _compute_summary(_breath_profile_events(), HYPNO)
    assert s["oahi_sweep"]["lenient"] == s["oahi_sweep"]["strict"]
    assert s["oahi_sweep_width"] == 0
    assert s["robustness_grade"] == "A"


def test_ahi_excl_noise_telt_hypopneeen_altijd():
    s = _compute_summary(_breath_profile_events(), HYPNO)
    assert s["ahi_excl_noise"] == pytest.approx(s["ahi_total"], abs=0.01)


def test_apneus_worden_nog_wel_gestratificeerd():
    """De reparatie mag de apneustratificatie niet meenemen in haar val."""
    ev = _breath_profile_events()
    ev.append({"type": "obstructive", "onset_s": 5000.0, "duration_s": 15.0,
               "stage": "N2", "epoch": 166, "confidence": 0.30})
    s = _compute_summary(ev, HYPNO)
    assert s["oahi_thresholds"]["0.60"] < s["oahi_all"]
    assert s["ahi_excl_noise"] < s["ahi_total"]
    assert s["confidence_bands"]["low"] == 1


def test_p_scored_blijft_op_het_event_staan():
    """De samenvatting repareert de telling; het event verliest zijn graad niet."""
    ev = _breath_profile_events()
    _compute_summary(ev, HYPNO)
    assert all("p_scored" in e for e in ev if e["type"] == "hypopnea")
