"""Een gefaalde ademteug-detector mag niet stil in een envelope-uitslag eindigen.

Stap 7b verving de hypopneeën door de gegradeerde set; gooide
`score_hypopneas_breathwise` een exception, dan bleef de envelope-uitslag
staan met alleen een regel in het log. De uitvoer droeg daarna een
breath-profielnaam, een breath-conform `ahi_rule` en GEEN `breath_detector`-
blok -- voor een lezer niet te onderscheiden van een geslaagde run waarin
de gradering toevallig niets veranderde.

Nu staat de terugval in de uitvoer zelf: in het detectorblok en in het
herkomstblok.
"""
import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 1200
N = int(DUR_S * SF)


def _raw():
    t = np.arange(N) / SF
    rng = np.random.default_rng(3)
    flow = np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.normal(size=N)
    spo2 = np.full(N, 96.0)
    for t0 in range(120, DUR_S - 120, 90):
        a, b = int(t0 * SF), int((t0 + 20) * SF)
        flow[a:b] *= 0.3
        spo2[b:b + int(15 * SF)] -= 5.0
    info = mne.create_info(["Resp nasal", "SaO2", "Thorax", "Abdomen"],
                           SF, ["misc"] * 4)
    data = np.vstack([flow, spo2, 0.6 * np.sin(2 * np.pi * 0.25 * t),
                      0.6 * np.sin(2 * np.pi * 0.25 * t)])
    return mne.io.RawArray(data, info, verbose="ERROR")


HYPNO = ["N2"] * (DUR_S // 30)


@pytest.fixture
def kapot(monkeypatch):
    import psgscoring.breath_scoring as bs

    def _boom(**kw):
        raise RuntimeError("ademteugdetector kapot")

    monkeypatch.setattr(bs, "score_hypopneas_breathwise", _boom)


def test_de_terugval_staat_in_het_detectorblok(kapot):
    import psgscoring
    out = psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                         scoring_profile="aasm_v3_breath")
    bd = out["respiratory"].get("breath_detector")
    assert bd, "geen breath_detector-blok: de terugval is onzichtbaar"
    assert bd["status"] == "exception"
    assert "ademteugdetector kapot" in bd["error"]
    assert bd["fallback"] == "envelope"


def test_de_terugval_staat_in_het_herkomstblok(kapot):
    import psgscoring
    out = psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                         scoring_profile="aasm_v3_breath")
    assert out["meta"].get("detector_fallback") == "envelope"


def test_zonder_fout_is_er_geen_terugval():
    import psgscoring
    out = psgscoring.run_pneumo_analysis(_raw(), hypno=HYPNO,
                                         scoring_profile="aasm_v3_breath")
    assert "detector_fallback" not in out["meta"]
    bd = out["respiratory"].get("breath_detector") or {}
    assert bd.get("status") != "exception"
