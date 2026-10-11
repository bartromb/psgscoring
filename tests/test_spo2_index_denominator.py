"""`analyze_spo2`: dezelfde noemer als de AHI, en percentages over geldige tijd.

De ODI en de T90-percentages rekenden over ALLE slaapsamples, inclusief
artefact-epochs en sensoruitval, terwijl de AHI de artefact-epochs uit zijn
noemer houdt. Twee indices op dezelfde rapportpagina, twee verschillende
nachten.

Met `artifact_epochs` en `index_denominator="artifact_free_sleep"`:
  * de noemer van de ODI is de artefactvrije slaaptijd, en desaturaties in
    een artefact-epoch tellen niet mee (zoals events daar niet meetellen);
  * de T90/T80/T70- en bandpercentages delen door het aantal GELDIGE
    samples, zodat uitval het percentage niet verdunt.
Zonder `artifact_epochs` (de huidige pipeline-aanroep) of met `"sleep"` (de
bevroren profielen) is de uitkomst byte-identiek aan vroeger.

Ook hier: NaN in `timeseries` wordt None, want NaN is geen JSON.
"""
import json
import math

import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import PROFILES
from psgscoring.spo2 import analyze_spo2, index_denominator_from_profile

SF = 1.0
N_EP = 40                      # 20 min, alles N2
HYPNO = ["N2"] * N_EP


def _spo2():
    s = np.full(N_EP * 30, 96.0)
    s[305:325] = 90.0          # desaturatie van 6 % in epoch 10
    s[905:925] = 90.0          # en één in epoch 30
    return s


def test_zonder_artefactlijst_verandert_er_niets():
    s = _spo2()
    a = analyze_spo2(s, SF, HYPNO)["summary"]
    b = analyze_spo2(s, SF, HYPNO, artifact_epochs=None)["summary"]
    assert a == b
    assert a["n_desat_3pct"] == 2
    assert a["odi_3pct"] == pytest.approx(2 / (N_EP * 30 / 3600), abs=0.05)


def test_artefact_epochs_vallen_uit_teller_en_noemer():
    r = analyze_spo2(_spo2(), SF, HYPNO, artifact_epochs=[10])
    s = r["summary"]
    assert s["n_desat_3pct"] == 1, "de desaturatie in het artefact-epoch telt niet"
    assert s["total_sleep_s"] == (N_EP - 1) * 30
    assert s["odi_3pct"] == pytest.approx(1 / ((N_EP - 1) * 30 / 3600), abs=0.05)
    assert s["index_denominator"] == "artifact_free_sleep"
    assert s["n_artifact_epochs_excluded"] == 1


def test_de_bevroren_modus_negeert_de_artefactlijst():
    oud = analyze_spo2(_spo2(), SF, HYPNO)["summary"]
    s = analyze_spo2(_spo2(), SF, HYPNO, artifact_epochs=[10],
                     index_denominator="sleep")["summary"]
    for k in ("n_desat_3pct", "odi_3pct", "total_sleep_s", "pct_below_90"):
        assert s[k] == oud[k], k


def test_uitval_verdunt_de_percentages_niet_meer():
    s = np.full(N_EP * 30, 96.0)
    s[100:220] = 88.0          # 120 s onder de 90
    s[600:660] = 0.0           # 60 s sensoruitval
    oud = analyze_spo2(s, SF, HYPNO)["summary"]
    nieuw = analyze_spo2(s, SF, HYPNO, artifact_epochs=[])["summary"]
    assert oud["pct_below_90"] == pytest.approx(120 / 1200 * 100, abs=0.05)
    assert nieuw["pct_below_90"] == pytest.approx(120 / 1140 * 100, abs=0.05)
    assert nieuw["pct_90_95"] + nieuw["pct_95_100"] + nieuw["pct_80_90"] == pytest.approx(100.0, abs=0.2)


def test_timeseries_bevat_geen_nan():
    s = np.full(N_EP * 30, 96.0)
    s[600:660] = 0.0
    r = analyze_spo2(s, SF, HYPNO)
    ts = r["timeseries"]
    assert not any(isinstance(v, float) and math.isnan(v) for v in ts)
    assert None in ts
    json.dumps(ts, allow_nan=False)


# ── profielveld en lezer ────────────────────────────────────────────────

def test_veld_artefactvrij_op_klinisch_sleep_op_de_bevroren_profielen():
    assert PROFILES["aasm_v3_rec"].post_processing.spo2_index_denominator == "artifact_free_sleep"
    for naam in ("mesa_shhs", "chicago_1999"):
        assert PROFILES[naam].post_processing.spo2_index_denominator == "sleep", naam
        assert SCORING_PROFILES[naam]["SPO2_INDEX_DENOMINATOR"] == "sleep"


def test_de_lezer_volgt_de_legacy_sleutel():
    assert index_denominator_from_profile(SCORING_PROFILES["aasm_v3_rec"]) == "artifact_free_sleep"
    assert index_denominator_from_profile(SCORING_PROFILES["mesa_shhs"]) == "sleep"
    assert index_denominator_from_profile(None) == "artifact_free_sleep"
    with pytest.raises(ValueError):
        index_denominator_from_profile({"SPO2_INDEX_DENOMINATOR": "nacht"})
