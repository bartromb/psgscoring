"""
Een knop die de gekozen detector niet leest, is op dat profiel inert.

De registry draagt twee hypopneedetectors (`hypopnea_detector`: 'envelope' of
'breath_graded') en een aantal velden die maar één van beide leest. Zet een
envelope-profiel `hypopnea_strictness=0.30`, dan verandert er niets -- en in
een meting ziet dat eruit als 'geen effect'. `FIELD_DETECTOR_PATH` legt per
veld vast welk pad het leest; `profile_inert_fields()` meldt de
niet-default waarden die het gekozen pad niet kan zien.

De tabel volgt de CODE, niet de bedoeling. Concreet: de Rule 1A-herstelpas
(pipeline.py, stap 8) heeft geen detectorpoort en draait ook onder
`breath_graded`, op de kandidaten die de ENVELOPE-detector afwees. De
`rule1a_*`/`rule1b_*`-knoppen zijn daar dus niet inert -- ze doen iets, al is
het iets dat op dat pad weinig zin heeft. Dat staat als 'both' in de tabel,
met die reden; wie de poort in pipeline.py toevoegt, zet de tabel om.
"""
from __future__ import annotations

from dataclasses import fields, replace

BREATH_ALLEEN = {
    "hypopnea_strictness",
    "hypopnea_arousal_weight",
    "hypopnea_desat_width",
    "hypopnea_arousal_latency_grading",
    "desat_low_baseline_relaxation",
}

#: Door de opdracht als 'envelope' of 'alleen-provenance' bestempeld, door de
#: code op beide paden gelezen (zie de moduledocstring).
BEIDE_EXPLICIET = {
    "rule1a_arousal_enabled", "rule1a_gap_max_breaths",
    "rule1a_arousal_min_flow_reduction_pct", "rule1a_arousal_max_duration_s",
    "rule1a_arousal_min_local_reduction_pct",
    "rule1b_graded", "rule1b_min_proba", "rule1b_arousal_window_s",
    "arousal_limb_wired", "desat_or_arousal", "desat_required",
}


def test_tabel_dekt_de_bekende_padgebonden_velden():
    from psgscoring.profiles import FIELD_DETECTOR_PATH
    for k in BREATH_ALLEEN:
        assert FIELD_DETECTOR_PATH[k] == "breath_graded", k
    for k in BEIDE_EXPLICIET:
        assert FIELD_DETECTOR_PATH[k] == "both", k


def test_tabel_noemt_alleen_bestaande_velden_en_geldige_paden():
    from psgscoring.profiles import (FIELD_DETECTOR_PATH, HypopneaRules,
                                     PostProcessingRules)
    namen = ({f.name for f in fields(PostProcessingRules)}
             | {f.name for f in fields(HypopneaRules)})
    for k, v in FIELD_DETECTOR_PATH.items():
        assert k in namen, f"{k} is geen profielveld"
        assert v in ("breath_graded", "envelope", "both"), (k, v)


def test_envelope_profiel_met_breath_knop_is_inert():
    from psgscoring.profiles import get_profile, profile_inert_fields
    p = get_profile("aasm_v3_rec")
    q = replace(p, post_processing=replace(p.post_processing,
                                           hypopnea_strictness=0.30,
                                           hypopnea_arousal_weight=0.70))
    assert profile_inert_fields(q) == ["hypopnea_arousal_weight",
                                       "hypopnea_strictness"]


def test_breath_profiel_leest_zijn_eigen_knoppen():
    from psgscoring.profiles import get_profile, profile_inert_fields
    p = get_profile("aasm_v3_breath")
    assert p.post_processing.hypopnea_strictness != 0.50, "premisse: niet-default"
    assert "hypopnea_strictness" not in profile_inert_fields(p)


def test_default_waarde_is_nooit_inert():
    from psgscoring.profiles import get_profile, profile_inert_fields
    p = get_profile("aasm_v3_rec")
    q = replace(p, post_processing=replace(p.post_processing,
                                           hypopnea_strictness=0.50))
    assert profile_inert_fields(q) == []


def test_accepteert_ook_een_profielnaam():
    from psgscoring.profiles import profile_inert_fields
    assert profile_inert_fields("aasm_v3_rec") == []


def test_geen_geregistreerd_profiel_heeft_inerte_knoppen():
    """Meld het, verander het profiel niet: een inerte knop in de registry is
    een beslissing voor de eigenaar van dat profiel."""
    from psgscoring.profiles import PROFILES, profile_inert_fields
    inert = {n: profile_inert_fields(p) for n, p in PROFILES.items()}
    inert = {n: v for n, v in inert.items() if v}
    assert not inert, (
        "deze profielen zetten een niet-default waarde op een knop die hun "
        f"hypopneedetector niet leest: {inert}")
