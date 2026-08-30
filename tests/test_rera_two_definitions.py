"""
Er bestaan TWEE RERA-definities. Precies één ervan wordt gerapporteerd.

  * `arousal.detect_reras()` — flow-limitatie op de envelope, met een eigen
    koppelvenster. Landt op `output["arousal"]["reras"]`.
  * `pipeline._compute_rera_rdi()` — FRI-kandidaten plus flattening-reeksen.
    Landt op `output["respiratory"]["summary"]["n_rera"]` en voedt de RDI en
    het PDF-rapport.

Ze stonden onder bijna dezelfde naam -- `n_reras` tegen `n_rera` -- en leveren
verschillende getallen. `generate_psg_report.py` in YASAFlaskified leest de
eerste; die module staat uitgecommentarieerd in `tasks.py` en is dus dood, maar
wie hem terughaalt rapporteert stilzwijgend de andere definitie zonder dat er
iets omvalt.

Deze module pint het onderscheid vast in de UITVOER, zodat een consument het
kan zien in plaats van moeten weten.
"""
import numpy as np


def test_de_diagnostische_telling_noemt_zichzelf_niet_gezaghebbend():
    from psgscoring.arousal import detect_reras

    sf, dur = 32.0, 600
    t = np.arange(int(sf * dur)) / sf
    flow = np.sin(2 * np.pi * 0.25 * t)
    r = detect_reras(flow, np.clip(np.abs(flow), 0, 2), sf,
                     arousals=[{"onset_s": 100.0, "duration_s": 3.0}],
                     resp_events=[], hypno=["N2"] * (dur // 30))
    s = r["summary"]
    assert s["authoritative"] is False, (
        "deze telling presenteert zich als gezaghebbend terwijl het rapport "
        "een andere definitie gebruikt")
    assert "respiratory.summary.n_rera" in s["reported_by"], (
        "de uitvoer wijst niet naar de telling die WEL gerapporteerd wordt")


def test_de_twee_namen_verschillen_zodat_ze_niet_te_verwisselen_zijn():
    """`n_reras` (diagnostisch) tegen `n_rera` (gerapporteerd)."""
    from psgscoring.arousal import detect_reras

    sf, dur = 32.0, 300
    t = np.arange(int(sf * dur)) / sf
    flow = np.sin(2 * np.pi * 0.25 * t)
    diag = detect_reras(flow, np.clip(np.abs(flow), 0, 2), sf, [], [],
                        ["N2"] * (dur // 30))["summary"]
    assert "n_reras" in diag and "n_rera" not in diag, (
        "de diagnostische samenvatting draagt de gerapporteerde naam -- dan is "
        "verwisseling een kwestie van tijd")


def test_het_koppelvenster_is_niet_langer_hardgecodeerd():
    """Stond op 10 s hier en op 15 s in het geleverde pad."""
    import inspect

    from psgscoring.arousal import RERA_AROUSAL_WINDOW_S, detect_reras
    assert RERA_AROUSAL_WINDOW_S == 15.0
    sig = inspect.signature(detect_reras)
    assert sig.parameters["arousal_window_s"].default == RERA_AROUSAL_WINDOW_S

    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert d["RERA_AROUSAL_WINDOW_S"] == 15.0, (
            f"{naam} wijkt af; dat is een gedragswijziging op de RDI")
