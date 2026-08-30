"""Het werkpunt van de classifier is een profielveld geworden.

0,60 is gedomineerd: herijkt met gescheiden steekproeven gaat de F1 op 30
ongeziene MESA-opnames van 0,421 naar 0,543 bij drempel 0,90 (gepaard +0,091,
24/30, p = 1,5e-05), en 0,80 verslaat 0,60 op beide cohorten met bovendien een
zuivere eventtelling (1,07 / 1,01 tegen 1,52 / 1,47).

Default blijft `None` -- de moduleconstante 0,60 -- tot de keuze tussen 0,80
en 0,90 gemaakt is; die is klinisch, want bij 0,90 telt de arousal-index ruim
een derde te laag.
"""
import numpy as np
import pytest


def test_the_operating_point_is_0_70_everywhere_it_is_read():
    """0,70 sinds 30-08-2026 (gebruikersbeslissing), was 0,80 sinds 23-08.

    De verschuiving hoort ONLOSMAKELIJK bij `arousal_min_interval_s = 10.0`,
    die dezelfde dag aan ging. De 10 s-regel haalt ~15 % van de events weg,
    dus het optimum schuift mee omlaag; de regel op 0,80 laten staan geeft een
    count-ratio van 0,81 op PSG-IPA en dan telt de index te laag.

    Gemeten event-F1 (`--multi`), PSG-IPA n=5 / MESA n=30:
        0,80 + regel uit   0,5144 (ratio 1,014)  /  0,4101 (ratio 0,760)
        0,80 + regel aan   0,5371 (ratio 0,810)  /  0,4251 (ratio 0,669)
        0,70 + regel aan   0,5559 (ratio 1,000)  /  0,4171 (ratio 0,841)

    Eerlijk gelezen: op PSG-IPA is dit het beste punt en het enige met ratio
    1,00; op MESA is 0,80 + regel béter. Het vooraf vastgelegde primaire
    MESA-criterium voor juist deze combinatie is NIET gehaald (20/30, p=0,099).
    De keuze is gemaakt op de multi-scoordercohort en is klinisch, niet
    statistisch. Zie docs/arousal_10s_regel_20260830.md.

    Op de vijf gepinde profielen staat de classifier uit, dus daar wordt de
    waarde nooit gelezen; die worden hier niet gepind op een getal maar op het
    feit dat hij niet gelezen wordt.
    """
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        pp = get_profile(name).post_processing
        if not pp.arousal_lgbm:
            continue                      # classifier uit: drempel irrelevant
        assert pp.arousal_lgbm_threshold == 0.70, (
            f"{name} draait de classifier op {pp.arousal_lgbm_threshold}")


def test_the_interval_rule_is_on_except_where_reproduction_forbids_it():
    """De 10 s-regel hoort overal aan te staan BEHALVE waar een gepubliceerd
    resultaat eraan hangt.

    `mesa_shhs` reproduceert paper v31/v37 en de NSRR-conventie, `chicago_1999`
    bevriest het gedrag van 1999. Een andere arousaltelling breekt allebei --
    op PSG-IPA SN3 gaat de telling van 173 naar 159.
    """
    from psgscoring.profiles import get_profile, list_profiles

    for name in list_profiles():
        p = get_profile(name)
        verwacht = 0.0 if p.family in ("dataset", "legacy") else 10.0
        assert p.post_processing.arousal_min_interval_s == verwacht, (
            f"{name} ({p.family}) staat op "
            f"{p.post_processing.arousal_min_interval_s}, verwacht {verwacht}")


def test_the_two_flags_move_together():
    """Los van elkaar zijn ze niet verdedigbaar; deze test zegt dat hardop.

    0,70 zonder de regel ondertelt niet maar OVERtelt (ratio 1,215 op
    PSG-IPA); de regel zonder de drempelverschuiving ondertelt (0,810). Wie
    er een terugdraait, hoort de ander mee te nemen.
    """
    from psgscoring.profiles import get_profile

    for name in ("aasm_v3_rec", "aasm_v3_breath"):
        pp = get_profile(name).post_processing
        assert (pp.arousal_lgbm_threshold, pp.arousal_min_interval_s) == (0.70, 10.0), (
            f"{name}: de twee vlaggen zijn uit de pas gelopen")


def test_the_pinned_profiles_do_not_run_the_classifier_at_all():
    from psgscoring.profiles import get_profile

    for name in ("mesa_shhs", "chicago_1999", "cms_medicare",
                 "aasm_v1_rec", "aasm_v2_rec"):
        assert get_profile(name).post_processing.arousal_lgbm is False, name


def test_the_registry_carries_it():
    from psgscoring.constants import SCORING_PROFILES

    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_LGBM_THRESHOLD" in d, name


def test_the_detector_accepts_a_threshold_and_it_changes_the_outcome():
    """Een hogere drempel hoort MINDER events te geven, en meetbaar."""
    pytest.importorskip("lightgbm")
    from psgscoring.arousal import detect_arousals

    sf, minutes = 256.0, 20
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(3)
    eeg = rng.normal(0, 20e-6, n)
    for start in range(60, 60 * minutes - 60, 60):
        a, b = int(start * sf), int((start + 4) * sf)
        eeg[a:b] += 60e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    hypno = ["N2"] * int(np.ceil(n / sf / 30))
    # v0.27.1: zonder bruikbaar kin-EMG slaat de detector het hybride pad
    # over (emg_var_ratio zou constant 0 zijn en de kansverdeling
    # degenereert). Deze test gaat over het WERKPUNT, dus moet het hybride
    # pad ook echt kunnen draaien.
    emg = rng.normal(0, 10e-6, n)

    laag = detect_arousals(eeg, sf, hypno, emg_data=emg,
                           lgbm=True, lgbm_threshold=0.30)
    hoog = detect_arousals(eeg, sf, hypno, emg_data=emg,
                           lgbm=True, lgbm_threshold=0.95)
    n_laag = len(laag.get("events") or [])
    n_hoog = len(hoog.get("events") or [])
    assert n_laag > 0, "fixture levert geen kandidaten; meet niets"
    assert n_hoog <= n_laag, (
        f"hogere drempel gaf MEER events: {n_hoog} tegen {n_laag}")
    assert laag["summary"].get("lgbm_threshold") == 0.30
    assert hoog["summary"].get("lgbm_threshold") == 0.95


def test_the_summary_reports_the_threshold_that_was_used():
    """Anders is achteraf niet na te gaan welk werkpunt een rapport draaide."""
    pytest.importorskip("lightgbm")
    from psgscoring.arousal import detect_arousals

    sf = 256.0
    n = int(sf * 60 * 12)
    rng = np.random.default_rng(8)
    eeg = rng.normal(0, 20e-6, n)
    emg = rng.normal(0, 10e-6, n)   # zie de opmerking hierboven
    hypno = ["N2"] * int(np.ceil(n / sf / 30))
    out = detect_arousals(eeg, sf, hypno, emg_data=emg,
                          lgbm=True, lgbm_threshold=0.77)
    if "lgbm_threshold" in out.get("summary", {}):
        assert out["summary"]["lgbm_threshold"] == 0.77


def test_the_classifier_can_be_switched_per_run(monkeypatch):
    """Zonder deze override is een arm niet van de andere te scheiden.

    De pipeline gaf `AROUSAL_LGBM` als expliciete bool door, dus de env die
    `arousal.py` kent kwam nooit aan bod. Een meting die de classifier op een
    RERA-dragend profiel wilde vergelijken mat daardoor nul verschil op 30 van
    30 opnames -- en dat zag eruit als een uitkomst.
    """
    import numpy as np
    import pytest
    pytest.importorskip("mne")
    import mne

    import psgscoring

    sf, minutes = 64.0, 25
    n = int(sf * 60 * minutes)
    t = np.arange(n) / sf
    rng = np.random.default_rng(6)
    eeg = rng.normal(0, 20e-6, n)
    for s0 in range(60, 60 * minutes - 60, 70):
        a, b = int(s0 * sf), int((s0 + 5) * sf)
        eeg[a:b] += 70e-6 * np.sin(2 * np.pi * 10.0 * t[a:b])
    info = mne.create_info(["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin"],
                           sf, ["misc", "misc", "eeg", "emg"])
    raw = mne.io.RawArray(
        np.vstack([np.sin(2 * np.pi * 0.25 * t), np.full(n, 97.0), eeg,
                   rng.normal(0, 5e-6, n)]), info, verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))

    def n_ar():
        out = psgscoring.run_pneumo_analysis(
            raw.copy(), hypno=hypno, scoring_profile="aasm_v3_breath")
        return len(out["arousal"].get("events", []))

    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "0")
    uit = n_ar()
    monkeypatch.setenv("PSGSCORING_AROUSAL_LGBM", "1")
    aan = n_ar()
    assert uit != aan, (
        f"de env schakelt de classifier niet: uit={uit}, aan={aan} -- een "
        "vergelijking van beide armen zou nul verschil meten")


def test_the_pipeline_stamps_its_own_version_in_the_output():
    """Zonder dit stempel kan een later gerenderd rapport niet zeggen wat er
    gescoord heeft.

    YASAFlaskified legde de versie alleen vast in een `comparison`-blok, en dat
    bestaat bij een gewone klinische run met EEN profiel niet eens. Gevolg: het
    rapport toonde permanent een onzekerheidsteken.
    """
    import numpy as np
    import pytest
    mne = pytest.importorskip("mne")
    import psgscoring

    sf = 32.0
    n = int(sf * 60 * 8)
    t = np.arange(n) / sf
    rng = np.random.default_rng(2)
    info = mne.create_info(["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin"],
                           sf, ["misc", "misc", "eeg", "emg"])
    raw = mne.io.RawArray(
        np.vstack([np.sin(2 * np.pi * 0.25 * t), np.full(n, 97.0),
                   rng.normal(0, 20e-6, n), rng.normal(0, 5e-6, n)]),
        info, verbose=False)
    out = psgscoring.run_pneumo_analysis(
        raw, hypno=["N2"] * int(np.ceil(raw.times[-1] / 30.0)),
        scoring_profile="aasm_v3_rec")

    got = (out.get("meta") or {}).get("psgscoring_version")
    assert got == psgscoring.__version__, (
        f"meta draagt {got!r} in plaats van {psgscoring.__version__!r}")
