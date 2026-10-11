"""De gemengde-apneu-decompositie meet effort op de ENVELOPE, niet op |ruw|.

WAT ER MIS WAS
--------------
`decompose_mixed_apneas` sommeerde `|thorax| + |abdomen|` op de ruwe banden
en noemde samples onder 20 % van het segmentmaximum "geen effort". Een
RIP-band met een gelijkspanningsoffset (bij EDF-export heel gewoon) staat
nooit onder die drempel: |0,1 V| is altijd groter dan 20 % van 0,15 V, dus
het centrale deel werd 0,0 s en geen enkele gemengde apneu werd ooit
gedecomponeerd. Een band zónder offset kruist elke halve ademteug de nul,
dus daar hield het "centrale deel" op bij de eerste nuldoorgang.

De pijplijn heeft de goede grootheid al: `preprocess_effort` (bandfilter
0,03-2 Hz, amplitude-envelope, 2 s gladgestreken). Daarop is afwezige effort
klein en aanwezige effort groot, onafhankelijk van offset of fase.

Daarnaast zette de oude code de confidence van een gedecomponeerd event op
minstens 0,85/0,80 -- een vloer die niets met de meting te maken had.

`mixed_decomposition_on_envelope`: default True (klinisch), False gepind op
`mesa_shhs` en `chicago_1999`.
"""
import numpy as np
import pytest

from psgscoring.postprocess import (
    decompose_mixed_apneas,
    postprocess_respiratory_events,
)
from psgscoring.signal import preprocess_effort

SF = 32.0
DUR_S = 160.0
EV_ON, EV_OFF = 40.0, 65.0          # gemengde apneu van 25 s
CENTRAAL_S = 12.0                   # eerste 12 s zonder effort


def _band(offset=0.1, amp=0.05):
    """RIP-band: `offset` + `amp`*sinus, vlak tijdens het centrale deel."""
    n = int(SF * DUR_S)
    t = np.arange(n) / SF
    x = offset + amp * np.sin(2 * np.pi * 0.25 * t)
    a, b = int(EV_ON * SF), int((EV_ON + CENTRAAL_S) * SF)
    x[a:b] = offset
    return x


def _event(conf=0.6):
    return [{"type": "mixed", "onset_s": EV_ON, "duration_s": EV_OFF - EV_ON,
             "stage": "N2", "epoch": 1, "confidence": conf}]


def test_met_offset_wordt_het_centrale_deel_gemeten():
    th, ab = _band(), _band()
    out = decompose_mixed_apneas(_event(), th, ab, SF)
    e = out[0]
    assert e["mixed_decomposed"] is True
    assert e["central_duration_s"] == pytest.approx(CENTRAAL_S, abs=1.5), e
    assert e["type"] == "central" and e["original_type"] == "mixed"


def test_de_confidence_krijgt_geen_vloer_meer():
    out = decompose_mixed_apneas(_event(conf=0.6), _band(), _band(), SF)
    assert out[0]["confidence"] == 0.6


def test_het_oude_pad_zag_niets_door_de_offset():
    """Gedocumenteerd, niet gerepareerd: dit is wat `on_envelope=False` doet."""
    out = decompose_mixed_apneas(_event(conf=0.6), _band(), _band(), SF,
                                 on_envelope=False)
    e = out[0]
    assert e["central_duration_s"] == 0.0
    assert e["type"] == "mixed"


def test_het_oude_pad_houdt_zijn_vloer():
    th, ab = _band(offset=0.0), _band(offset=0.0)
    out = decompose_mixed_apneas(_event(conf=0.6), th, ab, SF, on_envelope=False)
    e = out[0]
    if e["type"] == "central":
        assert e["confidence"] >= 0.80


def test_meegegeven_envelopes_geven_hetzelfde_als_ruwe_banden():
    th, ab = _band(), _band()
    via_raw = decompose_mixed_apneas(_event(), th, ab, SF)
    via_env = decompose_mixed_apneas(
        _event(), None, None, SF,
        thorax_env=preprocess_effort(th, SF), abdomen_env=preprocess_effort(ab, SF))
    assert via_env[0]["central_duration_s"] == via_raw[0]["central_duration_s"]


def test_zonder_banden_blijft_het_event_ongemoeid():
    out = decompose_mixed_apneas(_event(), None, None, SF)
    assert out == _event()


def test_de_wrapper_leest_het_profiel():
    th, ab = _band(), _band()
    aan = postprocess_respiratory_events(
        events=_event(), thorax_data=th, abdomen_data=ab, sf_effort=SF,
        scoring_profile={"MIXED_DECOMPOSITION_ON_ENVELOPE": True})
    uit = postprocess_respiratory_events(
        events=_event(), thorax_data=th, abdomen_data=ab, sf_effort=SF,
        scoring_profile={"MIXED_DECOMPOSITION_ON_ENVELOPE": False})
    assert aan["n_mixed_to_central"] == 1
    assert uit["n_mixed_to_central"] == 0
    # Expliciet argument wint van het profiel.
    expl = postprocess_respiratory_events(
        events=_event(), thorax_data=th, abdomen_data=ab, sf_effort=SF,
        scoring_profile={"MIXED_DECOMPOSITION_ON_ENVELOPE": True},
        mixed_decomposition_on_envelope=False)
    assert expl["n_mixed_to_central"] == 0


# ── het profielveld ──────────────────────────────────────────────────────

def test_veld_default_en_pins():
    from psgscoring.profiles import PROFILES, PostProcessingRules
    assert PostProcessingRules().mixed_decomposition_on_envelope is True
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.mixed_decomposition_on_envelope is False, naam
    assert PROFILES["aasm_v3_rec"].post_processing.mixed_decomposition_on_envelope is True


def test_veld_bereikt_de_legacy_dict():
    from psgscoring.constants import SCORING_PROFILES
    assert SCORING_PROFILES["aasm_v3_rec"]["MIXED_DECOMPOSITION_ON_ENVELOPE"] is True
    assert SCORING_PROFILES["chicago_1999"]["MIXED_DECOMPOSITION_ON_ENVELOPE"] is False
