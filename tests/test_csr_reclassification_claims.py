"""
De CSR-herclassificatie claimde meer dan ze kon waarmaken.

`reclassify_csr_events` maakt van elke CSR-gevlagde obstructieve of gemengde
apneu een centrale, en zette daarbij `confidence = max(conf, 0.80)` met als
toelichting dat de CSR-context goed bewijs levert.

Die aanname is gemeten en weerlegd. `docs/subtypering_mesa_20260814.md`, MESA
n=52: kappa **0,091** op CSR-nachten tegen 0,311 zonder, en de dominante fout
is juist obstructief -> centraal (230 van de 235). De regel verhoogde dus het
vertrouwen in precies de beslissing die daar het zwakst is.

Het is zichtbaar in het klinische rapport: de sterrenkolom leest deze
confidence, en 0,80 valt in de band 0,60-0,84. Op een opname met 29
herclassificaties stonden er 26 op twee sterren, uitsluitend omdat deze regel
dat getal zette.

Drie ingrepen, geen ervan raakt de AHI:
  1. geen confidence-verhoging meer (profielvlag herstelt het oude gedrag)
  2. de therapiehelft van een split-night als onafhankelijke tegenspraak
  3. de stap zelf uitzetbaar, zodat hij meetbaar wordt
"""

from psgscoring.postprocess import (
    csr_therapy_contradiction,
    postprocess_respiratory_events,
    reclassify_csr_events,
)

CSR = {"csr_detected": True}


def _ev(t="obstructive", conf=0.42, flagged=True):
    return {"type": t, "confidence": conf, "csr_flagged": flagged,
            "onset_s": 100.0, "duration_s": 20.0}


# ══════════════════════════════════════════════════════════════════════
#  1. Geen vertrouwen dat niet verdiend is
# ══════════════════════════════════════════════════════════════════════

def test_de_herclassificatie_verhoogt_de_confidence_niet_meer():
    uit = reclassify_csr_events([_ev(conf=0.42)], CSR)
    assert uit[0]["type"] == "central", "de herclassificatie zelf hoort te blijven"
    assert uit[0]["confidence"] == 0.42, (
        f"confidence opgetrokken naar {uit[0]['confidence']} op een beslissing "
        "met kappa 0,09 op deze nachten")


def test_de_sterrenband_verandert_daardoor_mee():
    """0,80 valt in de band 0,60-0,84 (twee sterren); 0,42 in 0,40-0,59."""
    uit = reclassify_csr_events([_ev(conf=0.42)], CSR)
    assert not (0.60 <= uit[0]["confidence"] <= 0.84), (
        "het event landt nog steeds in de tweesterrenband door toedoen van de regel")


def test_het_oude_gedrag_is_herstelbaar_voor_een_reproductie():
    uit = reclassify_csr_events([_ev(conf=0.42)], CSR, confidence_floor=0.80)
    assert uit[0]["confidence"] == 0.80


def test_een_hogere_eigen_confidence_wordt_nooit_verlaagd():
    uit = reclassify_csr_events([_ev(conf=0.93)], CSR, confidence_floor=0.80)
    assert uit[0]["confidence"] == 0.93


def test_elk_profiel_laat_de_ondergrens_ongezet():
    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert d["CSR_CONFIDENCE_FLOOR"] is None, f"{naam} trekt de confidence op"


# ══════════════════════════════════════════════════════════════════════
#  3. De stap is meetbaar geworden
# ══════════════════════════════════════════════════════════════════════

def test_de_stap_is_uit_te_zetten():
    aan = postprocess_respiratory_events([_ev()], csr_info=CSR)
    uit = postprocess_respiratory_events([_ev()], csr_info=CSR,
                                         csr_reclassification=False)
    assert aan["n_csr_reclassified"] == 1
    assert uit["n_csr_reclassified"] == 0
    assert uit["events"][0]["type"] == "obstructive"


def test_elk_profiel_houdt_de_stap_aan():
    """Default = bestaand gedrag; de vlag bestaat om te MÉTEN, niet om te wijzigen."""
    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert d["CSR_RECLASSIFICATION"] is True, naam


# ══════════════════════════════════════════════════════════════════════
#  2. De therapiehelft als onafhankelijke tegenspraak
# ══════════════════════════════════════════════════════════════════════

def _split(diag=83.5, ther=1.1, betrouwbaar=True):
    return {"detected": True, "segments": {
        "diagnostic": {"ahi": diag, "reliable": betrouwbaar},
        "therapeutic": {"ahi": ther, "reliable": betrouwbaar}}}


def _centraal(n=29, herklas=29):
    return ([{"type": "central", "csr_reclassified": True} for _ in range(herklas)]
            + [{"type": "central"} for _ in range(n - herklas)]
            + [{"type": "obstructive"} for _ in range(39)])


def test_de_tegenspraak_wordt_gemeld():
    r = csr_therapy_contradiction(_centraal(), _split())
    assert r is not None
    assert r["diagnostic_ahi"] == 83.5 and r["therapeutic_ahi"] == 1.1
    assert r["n_central_from_csr_reclassification"] == 29
    assert "kappa 0,09" in r["message"]


def test_geen_melding_zonder_split_night():
    assert csr_therapy_contradiction(_centraal(), {"detected": False}) is None
    assert csr_therapy_contradiction(_centraal(), None) is None


def test_geen_melding_als_de_therapie_niet_aanslaat():
    """Blijft de AHI onder therapie hoog, dan is er geen tegenspraak."""
    assert csr_therapy_contradiction(_centraal(), _split(ther=28.0)) is None


def test_geen_melding_bij_een_milde_diagnostische_helft():
    """Een vrijwel-volledige respons is pas informatief vanaf matig-ernstig."""
    assert csr_therapy_contradiction(_centraal(), _split(diag=9.0)) is None


def test_geen_melding_bij_echt_gedetecteerde_centrale_events():
    """Komt de minderheid uit de herclassificatie, dan zegt de respons niets
    over die beslissing."""
    assert csr_therapy_contradiction(_centraal(n=29, herklas=5), _split()) is None


def test_geen_melding_bij_een_onbetrouwbaar_segment():
    assert csr_therapy_contradiction(_centraal(), _split(betrouwbaar=False)) is None


def test_de_melding_verandert_niets():
    """Een OBSERVATIE: geen event wordt teruggezet, geen index beweegt."""
    ev = _centraal()
    voor = [dict(e) for e in ev]
    csr_therapy_contradiction(ev, _split())
    assert ev == voor


def test_de_vlag_wordt_ook_werkelijk_gelezen():
    """`csr_reclassification` bestond sinds v0.4.x en deed NIETS.

    Het veld stond in de dataclass, twee profielen zetten hem expliciet, en
    `pipeline.py` gaf hem nooit door. Uitzetten had geen effect. Dat is
    dezelfde dode-knopklasse als `desat_global_baseline_min_local_pct`, en hij
    kwam boven doordat een duplicaat-lint aansloeg -- niet doordat een test
    hem miste. Deze test dekt dat gat.
    """
    import inspect

    from psgscoring import pipeline
    bron = inspect.getsource(pipeline)
    assert "CSR_RECLASSIFICATION" in bron, (
        "pipeline.py leest de vlag niet; hij staat dan in de registry zonder "
        "consument, precies zoals hij jarenlang stond")
    assert "csr_reclassification=" in bron


def test_een_profiel_dat_de_stap_uitzet_verandert_de_uitkomst():
    """De doorgifte end-to-end, niet alleen de aanwezigheid van een string."""
    aan = postprocess_respiratory_events([_ev()], csr_info=CSR,
                                         csr_reclassification=True)
    uit = postprocess_respiratory_events([_ev()], csr_info=CSR,
                                         csr_reclassification=False)
    assert aan["events"][0]["type"] != uit["events"][0]["type"]
