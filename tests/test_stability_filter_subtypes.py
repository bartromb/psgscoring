"""Het stabiele-ademhalingsfilter vergeleek het eventtype exact.

`hypopnea_central`, `hypopnea_mixed` en `hypopnea_uncertain` ontsnapten er dus
aan. Hoeveel dat scheelt hangt af van of de effortclassificatie draaide en dus
van de RIP-poort: op PSG-IPA draagt 96 % van de hypopneus het kale label en
wijst het filter er 190 af, op MESA met dichte poort nul.

Zie `docs/rip_poort_reparatie_20260812.md`.
"""
from __future__ import annotations

import numpy as np

from psgscoring.profiles import PROFILES, PostProcessingRules

SUBTYPES = ["hypopnea", "hypopnea_central", "hypopnea_mixed",
            "hypopnea_uncertain"]


def _is_hypopnea(ev_type: str, alle_subtypes: bool) -> bool:
    """De beslissing zoals `detect_respiratory_events` hem neemt."""
    return ("hypopnea" in ev_type) if alle_subtypes else (ev_type == "hypopnea")


# ── het veld ─────────────────────────────────────────────────────────

GEPIND = ("mesa_shhs", "chicago_1999")


def test_default_is_aan():
    """Sinds 13-08-2026 dekt het filter alle hypopneus, ongeacht subtype."""
    assert PostProcessingRules().stability_filter_all_hypopnea_subtypes is True


def test_precies_twee_profielen_zijn_gepind():
    uit = {n for n, p in PROFILES.items()
           if not p.post_processing.stability_filter_all_hypopnea_subtypes}
    assert uit == set(GEPIND), f"onverwachte pinning: {uit}"


def test_veld_bereikt_de_legacy_dict():
    import psgscoring.constants as C
    for naam, d in C.SCORING_PROFILES.items():
        verwacht = naam not in GEPIND
        assert d["STABILITY_FILTER_ALL_HYPOPNEA_SUBTYPES"] is verwacht, naam


# ── het defect ───────────────────────────────────────────────────────

def test_oud_gedrag_laat_drie_van_de_vier_subtypes_ontsnappen():
    gedekt = [t for t in SUBTYPES if _is_hypopnea(t, alle_subtypes=False)]
    assert gedekt == ["hypopnea"], (
        "als dit meer dekt, reproduceert de test het defect niet")


def test_nieuw_gedrag_dekt_alle_vier():
    gedekt = [t for t in SUBTYPES if _is_hypopnea(t, alle_subtypes=True)]
    assert gedekt == SUBTYPES


def test_apneus_blijven_buiten_schot_in_beide_standen():
    for t in ("obstructive", "central", "mixed", "uncertain"):
        assert not _is_hypopnea(t, alle_subtypes=False)
        assert not _is_hypopnea(t, alle_subtypes=True), (
            f"{t!r} is een apneu, niet een hypopnee — de substringmatch mag "
            f"'uncertain' niet met 'hypopnea_uncertain' verwarren")


def test_mesa_situatie_nul_dekking_oud_volledige_dekking_nieuw():
    """Met dichte RIP-poort heet elke hypopnee `hypopnea_uncertain`."""
    events = [{"type": "hypopnea_uncertain"} for _ in range(40)]
    oud = sum(1 for e in events if _is_hypopnea(e["type"], False))
    nieuw = sum(1 for e in events if _is_hypopnea(e["type"], True))
    assert oud == 0, "dit is waarom het filter op MESA nooit draaide"
    assert nieuw == 40


def test_psgipa_situatie_bijna_volledige_dekking_in_beide_standen():
    """Gemeten verdeling: 200 kaal, 5 gemengd, 3 centraal."""
    events = ([{"type": "hypopnea"}] * 200
              + [{"type": "hypopnea_mixed"}] * 5
              + [{"type": "hypopnea_central"}] * 3)
    oud = sum(1 for e in events if _is_hypopnea(e["type"], False))
    nieuw = sum(1 for e in events if _is_hypopnea(e["type"], True))
    assert oud == 200 and nieuw == 208
    assert oud / nieuw > 0.96, (
        "op dit cohort is het verschil klein — daarom viel het bij de "
        "PSG-IPA-validatie van v0.2.8 niet op")


# ── het filter zelf ──────────────────────────────────────────────────
#
# `reject_stable_breathing` staat los van `detect_respiratory_events` omdat een
# end-to-end-fixture die geen events oplevert de test LEEG laat slagen. Een
# eerdere versie van deze suite deed precies dat: nul events, nul afwijzingen,
# groen. Deze tests voeren daarom echte eventdicts in.

def _stabiele_ademhaling(n=200, amplitude=1.0):
    """Constante amplitude = variatiecoëfficiënt 0, dus het filter slaat toe."""
    return [{"onset_s": 3.0 * i, "amplitude": amplitude} for i in range(n)]


def _ev(type_, onset=300.0):
    return {"type": type_, "onset_s": onset, "duration_s": 15.0,
            "stage": "N2", "epoch": int(onset // 30)}


def test_fixture_wijst_werkelijk_af():
    """Zonder deze controle meet de rest van dit blok niets."""
    from psgscoring.respiratory import reject_stable_breathing
    _keep, _rej, n = reject_stable_breathing(
        [_ev("hypopnea")], [], _stabiele_ademhaling(),
        stability_cv=0.45, all_hypopnea_subtypes=False)
    assert n == 1, "de fixture moet het filter laten vuren"


def test_gesubtypeerde_hypopneus_ontsnappen_in_de_oude_stand():
    from psgscoring.respiratory import reject_stable_breathing
    events = [_ev(t, 300.0 + 60 * i) for i, t in enumerate(SUBTYPES)]
    keep, rej, n = reject_stable_breathing(
        events, [], _stabiele_ademhaling(),
        stability_cv=0.45, all_hypopnea_subtypes=False)
    assert n == 1 and len(keep) == 3
    assert rej[0]["type"] == "hypopnea"
    assert {e["type"] for e in keep} == {
        "hypopnea_central", "hypopnea_mixed", "hypopnea_uncertain"}


def test_nieuwe_stand_pakt_alle_vier():
    from psgscoring.respiratory import reject_stable_breathing
    events = [_ev(t, 300.0 + 60 * i) for i, t in enumerate(SUBTYPES)]
    keep, rej, n = reject_stable_breathing(
        events, [], _stabiele_ademhaling(),
        stability_cv=0.45, all_hypopnea_subtypes=True)
    assert n == 4 and keep == []
    assert all(r["reject_reason"].startswith("stable_breathing_cv_")
               for r in rej)


def test_apneus_overleven_beide_standen():
    from psgscoring.respiratory import reject_stable_breathing
    events = [_ev(t, 300.0 + 60 * i)
              for i, t in enumerate(("obstructive", "central", "uncertain"))]
    for vlag in (False, True):
        keep, _rej, n = reject_stable_breathing(
            events, [], _stabiele_ademhaling(),
            stability_cv=0.45, all_hypopnea_subtypes=vlag)
        assert n == 0 and len(keep) == 3, (
            "'uncertain' is een apneu en mag niet met 'hypopnea_uncertain' "
            "verward worden")


def test_onstabiele_ademhaling_wijst_niets_af():
    from psgscoring.respiratory import reject_stable_breathing
    # Afwisselend 0,2 en 2,0 geeft CV ~0,82 — ruim boven de drempel. Een
    # uniforme trekking uit hetzelfde bereik landt op ~0,47 en dus zó dicht
    # tegen 0,45 aan dat de steekproef in het venster de uitkomst bepaalt.
    wisselend = [{"onset_s": 3.0 * i, "amplitude": 0.2 if i % 2 else 2.0}
                 for i in range(200)]
    _keep, _rej, n = reject_stable_breathing(
        [_ev(t) for t in SUBTYPES], [], wisselend,
        stability_cv=0.45, all_hypopnea_subtypes=True)
    assert n == 0, "hoge CV betekent echte variatie, dus geen afwijzing"


def test_te_weinig_ademhalingen_in_het_venster():
    from psgscoring.respiratory import reject_stable_breathing
    ver_weg = [{"onset_s": 5000.0 + 3.0 * i, "amplitude": 1.0}
               for i in range(200)]
    _keep, _rej, n = reject_stable_breathing(
        [_ev("hypopnea")], [], ver_weg,
        stability_cv=0.45, all_hypopnea_subtypes=True)
    assert n == 0


def test_gedegradeerde_records_dragen_wat_consumenten_lezen():
    from psgscoring.respiratory import reject_stable_breathing
    _keep, rej, _n = reject_stable_breathing(
        [_ev("hypopnea_central")], [], _stabiele_ademhaling(),
        stability_cv=0.45, all_hypopnea_subtypes=True)
    for sleutel in ("type", "onset_s", "duration_s", "stage", "epoch"):
        assert sleutel in rej[0], (
            f"{sleutel} ontbreekt — ML-promotie en Regel 1B lezen die")
    assert rej[0]["type"] == "hypopnea_central", "subtype hoort behouden"


def test_bestaande_rejected_blijven_staan():
    from psgscoring.respiratory import reject_stable_breathing
    bestaand = [{"type": "hypopnea", "reject_reason": "iets anders"}]
    _keep, rej, _n = reject_stable_breathing(
        [_ev("hypopnea")], bestaand, _stabiele_ademhaling(),
        stability_cv=0.45, all_hypopnea_subtypes=True)
    assert len(rej) == 2 and rej[0]["reject_reason"] == "iets anders"
