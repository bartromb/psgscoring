"""Eén definitie van slaaptijd in de pipeline, en geen letterlijke 30 meer.

`pipeline.py` rekende de totale slaaptijd op vier plaatsen uit, met drie
definities: met en zonder artefact-epochs, en met `* 30 / 3600` naast
`EPOCH_LEN_S`. De arousal-index op extern aangeleverde arousals deelde door
de VOLLEDIGE slaaptijd terwijl `ahi_total` de artefact-epochs uitsluit --
twee indices op dezelfde nacht met een andere noemer.

`indices.tst_hours(hypno, artifact_epochs)` is nu de ene plek. Waar de
artefactset leeg is, is dit byte-identiek.
"""
import pathlib
import re

import numpy as np
import pytest

from psgscoring.constants import EPOCH_LEN_S
from psgscoring.indices import tst_hours

SRC = (pathlib.Path(__file__).resolve().parent.parent
       / "psgscoring" / "pipeline.py")


def test_slaapstadia_tellen_wake_en_onbekende_labels_niet():
    assert tst_hours(["W", "N1", "N2", "R", "N3"]) == 4 * EPOCH_LEN_S / 3600.0
    assert tst_hours(["W", "ART", "?", None, "N2"]) == 1 * EPOCH_LEN_S / 3600.0
    assert tst_hours([]) == 0.0
    assert tst_hours(None) == 0.0


def test_artefact_epochs_vallen_uit_de_noemer():
    hypno = ["N2"] * 120
    assert tst_hours(hypno, [0, 1, 2]) == 117 * EPOCH_LEN_S / 3600.0
    assert tst_hours(hypno, None) == 120 * EPOCH_LEN_S / 3600.0
    # Een artefact in wake verandert niets: die epoch telde al niet.
    assert tst_hours(["W"] + hypno, [0]) == 120 * EPOCH_LEN_S / 3600.0


def test_de_externe_arousalindex_deelt_door_dezelfde_noemer_als_de_ahi():
    mne = pytest.importorskip("mne")
    import psgscoring

    sf, dur = 32.0, 600
    n = int(sf * dur)
    t = np.arange(n) / sf
    info = mne.create_info(["Resp nasal", "SaO2"], sf, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(n, 96.0)]), info, verbose="ERROR")
    hypno = ["N2"] * (dur // 30)                      # 20 epochs
    art = [0, 1, 2, 3, 4, 5]                          # 6 eruit -> 14 over
    arousals = [{"onset_s": 30.0 * k + 5.0, "duration_s": 3.0}
                for k in range(12)]
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno, artifact_epochs=art,
                                         arousal_events=arousals,
                                         scoring_profile="aasm_v3_rec")
    tst = tst_hours(hypno, art)
    assert tst == 14 * EPOCH_LEN_S / 3600.0
    assert out["arousal"]["summary"]["arousal_index"] == round(12 / tst, 2)
    # en dat is ook de noemer van de AHI-familie (`tst_hours` in de
    # samenvatting is op 1 decimaal afgerond, `index_denominator_h` op 3 of 4
    # naargelang welke schrijver het laatst was)
    assert out["respiratory"]["summary"]["index_denominator_h"] == pytest.approx(
        tst, abs=1e-3)


def test_de_csr_dichtheid_gebruikt_dezelfde_noemer():
    from psgscoring.pipeline import _annotate_csr_density
    events = [{"type": "central", "duration_s": 20}] * 12
    hypno = ["N2"] * 240                              # 2 u
    art = list(range(120))                            # waarvan 1 u artefact
    out = {"cheyne_stokes": {"csr_detected": True},
           "respiratory": {"summary": {}, "events": events}}
    _annotate_csr_density(out, hypno, artifact_epochs=art)
    assert out["cheyne_stokes"]["central_events_per_h"] == 12.0
    assert out["cheyne_stokes"]["monitoring_hours"] == 2.0   # bewaking, niet slaap


def test_de_rera_noemer_is_tst_hours():
    from psgscoring.pipeline import _compute_rera_rdi
    from psgscoring.respiratory import _compute_summary
    hypno = ["N2"] * 120
    art = [0, 1, 2, 3]
    out = {"respiratory": {"success": True, "events": [],
                           "summary": _compute_summary([], hypno, art),
                           "rejected_hypopneas": [], "_breaths": []}}
    _compute_rera_rdi(out, hypno, arousals=[], artifact_epochs=art)
    assert out["respiratory"]["summary"]["index_denominator_h"] == round(
        tst_hours(hypno, art), 4)


def test_geen_eigen_slaaptijdberekening_en_geen_letterlijke_30_meer():
    src = SRC.read_text()
    assert "tst_hours(" in src
    assert "n_tst_eps" not in src
    assert re.search(r"\* 30 / 3600", src) is None
    assert re.search(r"\* 30\.0 \*", src) is None
    assert re.search(r"\bsf_hypop \* 30\)", src) is None
    # De drie plekken die zelf slaapepochs telden zijn weg.
    assert src.count('in ("N1", "N2", "N3", "R")') + src.count(
        'in ("N1","N2","N3","R")') == 0, "er telt nog een plek zelf slaapepochs"
