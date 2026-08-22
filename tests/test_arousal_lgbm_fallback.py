"""Arousal: wat er gebeurt als de hybride niet kan draaien.

In hybride modus (`PSGSCORING_AROUSAL_LGBM=1`) zet `detect_arousals` de
drempels op de RUIME kandidaatwaarden (ratio 1,2 / abrupt 1,0) en laat een
LightGBM-model daarna wegfilteren. Faalt dat model -- bestand ontbreekt,
lightgbm niet geinstalleerd, corrupte booster -- dan logt de code
"falling back to rule-based output", maar `result["events"]` bevat op dat
moment de KANDIDATEN, niet de regelgebaseerde uitkomst.

Dat is het gevaarlijke geval: een installatie zonder model levert dan stil een
veel te hoge arousal-index, met een logregel die het tegendeel beweert. Zolang
de vlag opt-in is, is het een voetangel; wordt hij ooit default, dan is het
een productiedefect.
"""
import numpy as np
import pytest

from psgscoring.arousal import AROUSAL_RATIO_THRESH, detect_arousals

SF = 100.0
DUR_S = 900


def _recording(seed=11):
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + 4.0 * np.sin(2 * np.pi * 10.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    # een handvol echte verschuivingen
    for at in (120.0, 300.0, 480.0, 660.0):
        s, e = int(at * SF), int((at + 6.0) * SF)
        eeg[s:e] = (28.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                    + 28.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                    + rng.normal(0.0, 1.0, e - s))
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def test_a_missing_model_falls_back_to_the_rule_based_result(monkeypatch):
    """Met een onvindbaar model hoort de uitkomst gelijk te zijn aan het
    regelgebaseerde pad -- niet aan de ruime kandidatenlijst."""
    eeg, hypno = _recording()
    regels = detect_arousals(eeg, SF, hypno)

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    monkeypatch.setattr("psgscoring.arousal.AROUSAL_LGBM_MODEL_PATH",
                        "/nonexistent/arousal_classifier_v3.txt")
    monkeypatch.setattr("psgscoring.arousal._AROUSAL_LGBM_BOOSTER", None)
    kapot = detect_arousals(eeg, SF, hypno)

    assert kapot["success"], kapot.get("error")
    assert len(kapot["events"]) == len(regels["events"]), (
        f"model ontbreekt maar er komen {len(kapot['events'])} events uit tegen "
        f"{len(regels['events'])} regelgebaseerd -- dit is de kandidatenlijst "
        f"op ratio 1,2 in plaats van {AROUSAL_RATIO_THRESH}"
    )
    assert kapot["events"] == regels["events"]


def test_the_summary_says_the_model_did_not_run(monkeypatch):
    """Een consument moet kunnen zien dat de hybride niet gedraaid heeft."""
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    monkeypatch.setattr("psgscoring.arousal.AROUSAL_LGBM_MODEL_PATH",
                        "/nonexistent/arousal_classifier_v3.txt")
    monkeypatch.setattr("psgscoring.arousal._AROUSAL_LGBM_BOOSTER", None)
    out = detect_arousals(eeg, SF, hypno)
    s = out["summary"]
    assert s.get("lgbm_available") is False, (
        "de samenvatting zegt niet dat de classifier niet beschikbaar was"
    )
    assert "lgbm_n_post" not in s, (
        "lgbm_n_post suggereert dat er gefilterd is terwijl dat niet gebeurd is"
    )


# ══════════════════════════════════════════════════════════════
# Het profielveld — `arousal_lgbm`
# ══════════════════════════════════════════════════════════════

def _lgbm_installed() -> bool:
    """Kan de classifier hier uberhaupt draaien?

    CI installeert `.[test]`, niet `[ml]`, dus daar ontbreekt lightgbm. Dat is
    geen tekortkoming van de test maar de omgeving waarin het terugvalpad moet
    werken -- en de enige plek waar dat pad ECHT getoetst wordt, want lokaal
    is lightgbm geinstalleerd. De eerste versie van deze test eiste
    `lgbm_available is True` en viel daardoor terecht om in CI.
    """
    try:
        import lightgbm  # noqa: F401
    except Exception:      # noqa: BLE001
        return False
    from pathlib import Path
    from psgscoring.arousal import AROUSAL_LGBM_MODEL_PATH
    return Path(AROUSAL_LGBM_MODEL_PATH).exists()


def test_the_profile_field_reaches_the_detector(monkeypatch):
    """`lgbm=True` bereikt de detector zonder env-variabele.

    Tot v0.23.0 was het hybride pad ALLEEN via PSGSCORING_AROUSAL_LGBM te
    bereiken. Dat maakte de keuze installatiebreed: `mesa_shhs` kon niet
    gepind blijven terwijl de klinische profielen hem gebruikten.

    Getoetst wordt of de VLAG aankomt -- de sleutel verschijnt in de
    samenvatting -- niet of de classifier kan draaien. Dat tweede hangt van de
    omgeving af, en de waarde van de sleutel zegt precies welke van de twee je
    voor je hebt.
    """
    monkeypatch.delenv("PSGSCORING_AROUSAL_LGBM", raising=False)
    monkeypatch.delenv("YASAFLASKIFIED_AROUSAL_LGBM", raising=False)
    eeg, hypno = _recording()

    uit = detect_arousals(eeg, SF, hypno)
    aan = detect_arousals(eeg, SF, hypno, lgbm=True)

    assert "lgbm_available" not in uit["summary"], (
        "zonder de vlag hoort er geen lgbm-sleutel in de samenvatting te staan")
    assert "lgbm_available" in aan["summary"], (
        "profielveld bereikt de detector niet")
    assert aan["summary"]["lgbm_available"] is _lgbm_installed(), (
        "de sleutel hoort te zeggen of de classifier werkelijk gedraaid heeft")
    if not _lgbm_installed():
        # zonder classifier hoort het resultaat gelijk te zijn aan de regels,
        # niet aan de ruime kandidatenlijst
        assert aan["events"] == uit["events"]


def test_the_env_variable_still_wins(monkeypatch):
    """De env blijft werken en overschrijft het profiel — in beide richtingen.

    Een installatie moet hem kunnen forceren of uitzetten, en een meting moet
    kunnen aantonen dat hij niet actief was.
    """
    eeg, hypno = _recording()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")
    assert "lgbm_available" not in detect_arousals(
        eeg, SF, hypno, lgbm=True)["summary"], (
        "env=0 hoort het profielveld te overrulen")
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    assert "lgbm_available" in detect_arousals(
        eeg, SF, hypno, lgbm=False)["summary"], (
        "env=1 hoort het profielveld te overrulen")


def test_default_is_on_except_where_a_ruleset_is_reproduced():
    """Default AAN sinds 22-08-2026, behalve waar een externe regelset of een
    gepubliceerde dataset-analyse gereproduceerd wordt.

    Een ML-classifier maakt geen deel uit van AASM v1/v2 of de CMS-regels, en
    `mesa_shhs`/`chicago_1999` moeten paper v31/v37 reproduceren. Verandert
    een van die vijf, dan verschuiven gepubliceerde of regulatoire cijfers en
    hoort dit om te vallen.
    """
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_LGBM" in d, name
        assert isinstance(d["AROUSAL_LGBM"], bool), name
    uit = {n for n, d in SCORING_PROFILES.items() if not d["AROUSAL_LGBM"]}
    assert uit == {"aasm_v2_rec", "aasm_v1_rec", "cms_medicare",
                   "mesa_shhs", "chicago_1999"}, (
        f"verwacht alleen de vijf gepinde profielen uit, kreeg {sorted(uit)}")


def test_low_sample_rate_falls_back_instead_of_rejecting_everything(monkeypatch):
    """Onder 64 Hz draait de classifier niet, want hij zou alles verwerpen.

    Het model gebruikt bandvermogens tot beta (16-30 Hz). Ligt Nyquist daar
    onder, dan bestaan die kenmerken niet in het signaal en krijgt een op
    256 Hz getraind model een gedegenereerde vector. Gemeten op de
    golden-fixture met EEG op 32 Hz: 23 kandidaten, kansen 0,012 tot 0,066,
    drempel 0,60, dus nul events. Op een profiel waar hypopneus alleen via een
    arousal kwalificeren werd de AHI daarmee 18,9 -> 0,0.

    Dit is dezelfde klasse als het ontbrekende model: een voorwaarde die vóór
    de gedragswijziging getoetst moet worden, niet erna.
    """
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    import numpy as np

    from psgscoring.arousal import AROUSAL_LGBM_MIN_SF

    # zelfde signaal, twee samplefrequenties
    for sf, verwacht_hybride in ((32.0, False), (128.0, True)):
        n = int(900 * sf)
        t = np.arange(n) / sf
        rng = np.random.default_rng(4)
        eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
               + 6.0 * np.sin(2 * np.pi * 6.0 * t)
               + rng.normal(0.0, 1.0, t.size))
        for at in (200.0, 400.0, 600.0):
            s, e = int(at * sf), int((at + 6.0) * sf)
            eeg[s:e] = (28.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                        + rng.normal(0.0, 1.0, e - s))
        out = detect_arousals(eeg * 1e-6, sf, ["N2"] * 30)
        reden = out["summary"].get("lgbm_skipped_reason")
        sf_reden = f"sample_rate_below_{AROUSAL_LGBM_MIN_SF:.0f}"
        # Toets de REDEN, niet de afwezigheid van de sleutel. Zonder lightgbm
        # -- zoals in CI, dat `.[test]` installeert -- slaat het model-pad ook
        # over, met reden "model_unavailable". Een test die alleen op
        # afwezigheid toetst, meet dan de omgeving in plaats van de poort.
        if verwacht_hybride:
            assert reden != sf_reden, (
                f"sf={sf} ligt boven {AROUSAL_LGBM_MIN_SF:.0f} Hz, de "
                "samplefrequentie-poort hoort niet te vuren")
        else:
            assert out["summary"].get("lgbm_available") is False
            assert reden == sf_reden, (
                f"sf={sf}: verwacht {sf_reden!r}, kreeg {reden!r}")
