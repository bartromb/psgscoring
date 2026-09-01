"""De AHI heet AHI, ook wanneer hij geen enkele v3-regel volgt.

WAT ER AAN DE HAND IS
---------------------
AASM v3 verplaatst het 4 %-criterium van ACCEPTABLE naar OPTIONAL. Daarmee is
"≥30 % flowreductie + (≥3 % desaturatie OF arousal)" de enige AANBEVOLEN regel,
en Rule 1B (≥4 % desaturatie) de enige optionele.

Op `aasm_v3_rec` -- het defaultprofiel -- staat de arousal-tak uit. De uitvoer
is dan "≥30 % flowreductie + ≥3 % desaturatie zonder arousal-alternatief", en
dat is geen van beide: niet de aanbevolen regel, niet de optionele. Onder v2.6
was dit een keuze binnen twee toegestane definities; onder v3 is het een derde
definitie die de manual niet kent.

Het getal heet niettemin `ahi_total`, en een consument die alleen dat veld leest
kan niet weten dat het een andere grootheid is dan de AHI in de literatuur.

De tekstuele `hypopnea_criterion` bestaat al en zegt het correct -- maar hij
staat in het herkomstblok, en YASAFlaskified las hem tot nu toe nergens. Een
losse noot onderaan een PDF telt niet: het label moet meereizen met het getal.

Zie docs/AASM_v3_conformiteit.md §1.1.
"""
import pytest

from psgscoring.constants import _profile_to_legacy_dict
from psgscoring.pipeline import ahi_rule_conformity, arousal_limb_is_effective
from psgscoring.profiles import get_profile


def _d(naam):
    return _profile_to_legacy_dict(get_profile(naam))


# ── Het oordeel zelf ──────────────────────────────────────────────────────

def test_de_aanbevolen_regel_wordt_als_conform_gemeld():
    """`aasm_v3_breath` heeft de arousal-tak wél effectief."""
    d = _d("aasm_v3_breath")
    assert arousal_limb_is_effective(d) is True
    c = ahi_rule_conformity(d)
    assert c["verdict"] == "v3_recommended"
    assert c["conform"] is True


def test_desaturatie_zonder_arousal_is_geen_v3_regel():
    """Dit is de kern: het defaultprofiel volgt geen enkele v3-regel."""
    d = _d("aasm_v3_rec")
    assert arousal_limb_is_effective(d) is False
    c = ahi_rule_conformity(d)
    assert c["verdict"] == "desat_only"
    assert c["conform"] is False
    assert "3" in c["label"], c["label"]
    # Het label moet zelfstandig leesbaar zijn: wie alleen dit veld ziet, moet
    # weten dat het geen gewone AHI is.
    assert "AHI" in c["label"]
    assert c["reason"], "een niet-conform oordeel zonder reden is een oordeel"


@pytest.mark.parametrize("naam", ["aasm_v3_rec", "aasm_v3_breath",
                                  "aasm_v3_dual", "mesa_shhs",
                                  "chicago_1999"])
def test_elk_profiel_krijgt_een_oordeel(naam):
    """Geen enkel profiel mag stil zonder label blijven."""
    c = ahi_rule_conformity(_d(naam))
    assert set(c) >= {"verdict", "conform", "label", "reason", "aasm_version"}
    assert isinstance(c["conform"], bool)
    assert c["label"].strip()


def test_het_label_verschilt_tussen_conform_en_niet_conform():
    """Zou het label gelijk zijn, dan draagt het geen informatie."""
    a = ahi_rule_conformity(_d("aasm_v3_breath"))["label"]
    b = ahi_rule_conformity(_d("aasm_v3_rec"))["label"]
    assert a != b


def test_de_versie_van_de_manual_staat_erbij():
    """§3.7: het pakket stond als 2.6-conform geboekt zonder dat ergens te
    zeggen. Welke manualversie geldt, hoort naast elk oordeel."""
    c = ahi_rule_conformity(_d("aasm_v3_rec"))
    assert "v3" in c["aasm_version"]
    c26 = ahi_rule_conformity(_d("chicago_1999"))
    assert c26["aasm_version"], "ook een historisch profiel noemt zijn regelset"


# ── Bereikt het de uitvoer? ───────────────────────────────────────────────

def test_het_oordeel_staat_in_de_samenvatting_naast_de_ahi():
    """Het label moet meereizen met het getal, niet in een apart blok staan.

    Dit is de reparatie: `hypopnea_criterion` stond in het herkomstblok en werd
    door geen enkele consument gelezen.
    """
    import numpy as np

    import psgscoring

    mne = pytest.importorskip("mne")
    sf, minuten = 32.0, 25
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    flow = np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 96.0)
    for start in range(60, 60 * minuten - 60, 100):
        a, b = int(start * sf), int((start + 20) * sf)
        flow[a:b] *= 0.3
        spo2[b:b + int(15 * sf)] -= 5.0
    info = mne.create_info(["Resp nasal", "SaO2", "Thorax", "Abdomen"],
                           sf, ["misc"] * 4)
    raw = mne.io.RawArray(np.vstack([flow, spo2,
                                     np.sin(2 * np.pi * 0.25 * t),
                                     np.sin(2 * np.pi * 0.25 * t)]),
                          info, verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_rec")
    summ = out["respiratory"]["summary"]
    assert "ahi_rule" in summ, (
        "het oordeel staat niet naast de AHI; een consument die alleen de "
        "samenvatting leest kan niet weten welke regel gold")
    assert summ["ahi_rule"]["conform"] is False
    assert summ["ahi_rule"]["verdict"] == "desat_only"
