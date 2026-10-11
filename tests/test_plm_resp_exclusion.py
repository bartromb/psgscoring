"""PLM: de respiratoire uitsluiting beslaat het HELE event, niet alleen het einde.

AASM: een beenbeweging wordt niet gescoord als ze valt in de periode van
0,5 s vóór het begin van een apneu/hypopneu tot 0,5 s na het einde. De module
toetste alleen `einde +/- 0,5 s`. Een beweging midden in een apneu van 30 s
telde dus gewoon mee in de PLM-index.

`plm_resp_exclusion="aasm"` (default) toetst het venster [onset - 0,5,
einde + 0,5] op OVERLAP met de beweging; `"end"` is het oude gedrag, dat de
bevroren profielen houden.

Hieronder ook: het einde van een samengevoegde bilaterale beweging is het
laatste einde van de twee, niet de langste duur vanaf de vroegste onset.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.plm import (
    _exclude_resp_associated,
    _merge_bilateral,
    analyze_plm,
    plm_resp_exclusion_from_profile,
)
from psgscoring.profiles import PROFILES

SF = 64.0
DUR_S = 1800
N = int(DUR_S * SF)
HYPNO = ["N2"] * (DUR_S // 30)
APNEU = [{"onset_s": 1000.0, "duration_s": 30.0}]


def _emg(bursts, amp_uv=40.0, rust_uv=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, rust_uv, N)
    for onset, dur in bursts:
        a, b = int(onset * SF), int((onset + dur) * SF)
        x[a:b] += rng.normal(0, amp_uv, b - a)
    return x


def _lm(onset, dur=1.5):
    return {"onset_s": float(onset), "duration_s": dur}


# ── de helper ───────────────────────────────────────────────────────────

def test_een_beweging_midden_in_een_apneu_is_respiratoir():
    lms = [_lm(1012.0)]
    elig, n = _exclude_resp_associated(lms, resp_spans=[(1000.0, 1030.0)], mode="aasm")
    assert n == 1 and elig == []
    lms = [_lm(1012.0)]
    elig, n = _exclude_resp_associated(lms, resp_spans=[(1000.0, 1030.0)], mode="end")
    assert n == 0 and len(elig) == 1


@pytest.mark.parametrize("onset,uitgesloten", [
    (999.4, False),    # eindigt op 1000,9: overlapt het venster vanaf 999,5 -- zie onder
    (998.0, False),    # eindigt op 999,5 -> raakt het venster net niet
    (999.6, True),     # binnen 0,5 s vóór de onset
    (1030.4, True),    # binnen 0,5 s na het einde
    (1030.6, False),
])
def test_de_grenzen_van_het_aasm_venster(onset, uitgesloten):
    lm = _lm(onset, dur=1.0)
    _, n = _exclude_resp_associated([lm], resp_spans=[(1000.0, 1030.0)], mode="aasm")
    # Een beweging die op 999,4 BEGINT en 1,0 s duurt, overlapt wel degelijk
    # met [999,5, 1030,5]: de regel toetst overlap, niet alleen de onset.
    verwacht = uitgesloten or (onset + 1.0 > 999.5 and onset < 1030.5)
    assert bool(n) is verwacht, (onset, n)


def test_de_oude_modus_toetst_alleen_het_einde():
    lms = [_lm(100.0), _lm(110.0)]
    elig, n = _exclude_resp_associated(lms, resp_ends=[105.0], mode="end")
    assert n == 0 and len(elig) == 2
    _, n = _exclude_resp_associated([_lm(105.3)], resp_ends=[105.0], mode="end")
    assert n == 1


def test_aasm_modus_zonder_spans_valt_terug_op_de_eindes():
    _, n = _exclude_resp_associated([_lm(105.3)], resp_ends=[105.0], mode="aasm")
    assert n == 1


# ── end-to-end ──────────────────────────────────────────────────────────

def test_analyze_plm_volgt_de_modus():
    x = _emg([(1012.0, 1.5)])
    aasm = analyze_plm(x, None, SF, HYPNO, resp_events=APNEU, leg_unit="uV",
                       resp_exclusion="aasm")
    end = analyze_plm(x, None, SF, HYPNO, resp_events=APNEU, leg_unit="uV",
                      resp_exclusion="end")
    assert aasm["summary"]["n_lm_sleep"] == 1 == end["summary"]["n_lm_sleep"]
    assert aasm["summary"]["n_resp_associated"] == 1
    assert end["summary"]["n_resp_associated"] == 0
    assert aasm["summary"]["resp_exclusion"] == "aasm"


def test_de_default_is_aasm():
    x = _emg([(1012.0, 1.5)])
    r = analyze_plm(x, None, SF, HYPNO, resp_events=APNEU, leg_unit="uV")
    assert r["summary"]["n_resp_associated"] == 1


# ── profielveld en lezer ────────────────────────────────────────────────

def test_veld_aasm_op_klinisch_end_op_de_bevroren_profielen():
    assert PROFILES["aasm_v3_rec"].post_processing.plm_resp_exclusion == "aasm"
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.plm_resp_exclusion == "end", naam
        assert SCORING_PROFILES[naam]["PLM_RESP_EXCLUSION"] == "end"


def test_de_lezer_volgt_de_legacy_sleutel():
    assert plm_resp_exclusion_from_profile(SCORING_PROFILES["aasm_v3_rec"]) == "aasm"
    assert plm_resp_exclusion_from_profile(SCORING_PROFILES["mesa_shhs"]) == "end"
    assert plm_resp_exclusion_from_profile(None) == "aasm"
    with pytest.raises(ValueError):
        plm_resp_exclusion_from_profile({"PLM_RESP_EXCLUSION": "halverwege"})


# ── bilateraal: einde = laatste einde ───────────────────────────────────

def test_bilaterale_samenvoeging_eindigt_bij_het_laatste_einde():
    links = [{"onset_s": 100.0, "duration_s": 1.0, "amplitude_uv": 20.0}]
    rechts = [{"onset_s": 100.4, "duration_s": 1.0, "amplitude_uv": 25.0}]
    uit = _merge_bilateral(links, rechts, window_s=0.5)
    assert len(uit) == 1 and uit[0]["onset_s"] == 100.0
    assert uit[0]["duration_s"] == pytest.approx(1.4)
