"""Eén definitie van "slaap", voor elk hypnogramlabel.

Bij de codelezing van 10-11-2026 stonden er drie naast elkaar:

* `utils.is_sleep`: alles behalve 0, -1 en "W" -- dus "ART", "?" en None
  telden als slaap;
* `indices.SLEEP_STAGES`: alleen de vier tekstlabels -- een numeriek hypnogram
  gaf een slaaptijd van nul;
* `!= "W"` in `_compute_summary`, de PLM-stap en de houdingsanalyse -- in een
  numeriek hypnogram is waak 0, en dat telde dan als slaap.

Op de labels die YASA levert (W, N1, N2, N3, R) geven de drie hetzelfde; ze
lopen uiteen zodra een hypnogram iets anders draagt. Nu geldt overal: slaap is
N1, N2, N3 of R (of 1-4), en al het andere niet.
"""
import pathlib
import re

import pytest

import psgscoring
from psgscoring.indices import tst_hours
from psgscoring.respiratory import _compute_summary
from psgscoring.utils import build_sleep_mask, hypno_to_numeric, is_sleep

E_H = 30.0 / 3600.0


@pytest.mark.parametrize("label", ["N1", "N2", "N3", "R", 1, 2, 3, 4])
def test_de_vier_stadia_zijn_slaap(label):
    assert is_sleep(label) is True


@pytest.mark.parametrize("label", ["W", 0, -1, "ART", "?", "UNS", None, "", 9])
def test_al_het_andere_is_geen_slaap(label):
    assert is_sleep(label) is False


def test_een_artefactlabel_telt_niet_mee_in_de_noemer_van_de_ahi():
    hypno = ["N2"] * 100 + ["ART"] * 20
    s = _compute_summary([], hypno, None, denominator="sleep")
    assert s["index_denominator_h"] == pytest.approx(100 * E_H, abs=1e-3)


def test_een_numeriek_hypnogram_telt_waak_niet_als_slaap():
    hypno = [0] * 20 + [2] * 100
    s = _compute_summary([], hypno, None, denominator="sleep")
    assert s["index_denominator_h"] == pytest.approx(100 * E_H, abs=1e-3)


def test_tst_hours_leest_ook_een_numeriek_hypnogram():
    assert tst_hours([0, 2, 2, 4]) == pytest.approx(3 * E_H)
    assert tst_hours(["W", "N2", "ART", "R"]) == pytest.approx(2 * E_H)


def test_het_samplemasker_kent_dezelfde_labels():
    """`build_sleep_mask` loopt via `hypno_to_numeric`; een numeriek hypnogram
    gaf daar -1 voor elk label en dus geen enkele slaapsample, terwijl
    `tst_hours` er wel slaaptijd in zag: een noemer zonder detectiemasker."""
    assert list(hypno_to_numeric([0, 2, 4, 9, -1, "N3", "ART", None])) == [
        0, 2, 4, -1, -1, 3, -1, -1]
    masker = build_sleep_mask([0, 2, 2, 4], 1.0, 120)
    assert masker.sum() == 90 and not masker[:30].any()
    assert build_sleep_mask(["W", "N2", "ART", "R"], 1.0, 120).sum() == 60


def test_plm_en_houding_tellen_een_artefactlabel_niet_als_slaap():
    import numpy as np

    from psgscoring.ancillary import analyze_position
    from psgscoring.plm import analyze_plm
    sf, n_ep = 32.0, 120
    hypno = ["N2"] * 60 + ["ART"] * 60
    rng = np.random.default_rng(3)
    been = rng.normal(0, 1e-6, int(sf * 30 * n_ep))
    plm = analyze_plm(been, None, sf, hypno)
    assert plm["summary"]["total_sleep_h"] == pytest.approx(0.5, abs=0.06)
    pos = analyze_position(np.full(int(n_ep * 30), 2.0), 1.0, hypno, [])
    assert sum(v or 0 for v in pos["summary"]["sleep_time_min"].values()) == pytest.approx(30.0)


def test_de_arousalmodule_heeft_geen_eigen_definitie():
    from psgscoring import arousal
    assert arousal._is_sleep is is_sleep


def test_geen_module_leidt_slaap_af_uit_ongelijk_aan_w():
    """`!= "W"` als slaaptoets komt in de bibliotheek niet meer voor."""
    patroon = re.compile(r"""!=\s*["']W["']""")
    treffers = []
    for pad in sorted(pathlib.Path(psgscoring.__file__).parent.rglob("*.py")):
        for nr, regel in enumerate(pad.read_text(encoding="utf-8").splitlines(), 1):
            code = regel.split("#", 1)[0]
            if patroon.search(code):
                treffers.append(f"{pad.name}:{nr}: {regel.strip()}")
    assert treffers == []
