"""De positieanalyse moet de eventlijst volgen die het rapport toont.

`analyze_position` draait in stap 6 op de eventlijst van dat moment. Stap 7b —
de ademteug-gegradeerde detector — vervangt daarna ELKE hypopnee. Zonder
herberekening sloeg de AHI-per-positie dus op de envelope-detector terwijl de
rest van het rapport de gegradeerde events toonde.

Dat raakt elk profiel met `hypopnea_detector="breath_graded"`: `aasm_v3_breath`,
`aasm_v3_prob`, `aasm_v3_breath_dual` en `aasm_v3_prob_dual`. En daarmee ook de
positionele fenotypering (`positional_osa`), die `ahi_per_pos` leest — dus het
oordeel "kandidaat voor positietherapie" stond op de verkeerde events.
"""

import numpy as np
import pytest

mne = pytest.importorskip("mne")

SF = 32.0
DUR_S = 1800
N = int(DUR_S * SF)
T = np.arange(N) / SF


def _breathing(amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * 0.01 * T)
    return amp * mod * np.sin(2 * np.pi * 0.25 * T) + 0.05 * rng.normal(size=N)


def _flatten(sig, t0, t1, factor=0.25):
    out = sig.copy()
    out[int(t0 * SF):int(t1 * SF)] *= factor
    return out


def _montage():
    """Events geconcentreerd in de eerste helft; positie wisselt halverwege.

    Zo verschilt de AHI per positie echt, en verschuift hij mee als de
    eventlijst verandert.
    """
    pres = _breathing(seed=1)
    for t0 in range(120, 850, 90):          # hypopnees in de eerste helft
        pres = _flatten(pres, t0, t0 + 20)

    spo2 = np.full(N, 96.0)
    for t0 in range(120, 850, 90):
        a, b = int((t0 + 22) * SF), int((t0 + 38) * SF)
        spo2[a:min(b, N)] = 91.5

    pos = np.zeros(N)                        # 0 = supine
    pos[N // 2:] = 1.0                       # daarna links
    data = np.vstack([pres, _breathing(amp=0.6, seed=2),
                      _breathing(amp=0.6, seed=3), spo2, pos])
    info = mne.create_info(["Pressure Flow", "THORAX", "ABDOMEN", "SPO2", "Pos."],
                           SF, ch_types=["misc"] * 5)
    cmap = {"flow_pressure": "Pressure Flow", "thorax": "THORAX",
            "abdomen": "ABDOMEN", "spo2": "SPO2", "position": "Pos."}
    return mne.io.RawArray(data, info, verbose="ERROR"), cmap


HYPNO = ["N2"] * (DUR_S // 30)


def _run(profile):
    import psgscoring
    raw, cmap = _montage()
    return psgscoring.run_pneumo_analysis(raw, hypno=HYPNO, channel_map=cmap,
                                          scoring_profile=profile)


@pytest.fixture(scope="module")
def graded():
    return _run("aasm_v3_breath")


def _events_in(out):
    return out["respiratory"]["events"]


def _pos_total(out):
    """Som van de events die de positieanalyse toewees."""
    pos = (out.get("position") or {}).get("summary", {}) or {}
    per = pos.get("events_per_pos") or pos.get("n_events_per_pos") or {}
    return sum(v for v in per.values() if isinstance(v, (int, float)))


def test_the_position_analysis_sees_the_same_events_as_the_report(graded):
    """De kern. Zonder herberekening telde de positieanalyse de hypopneeën van
    de envelope-detector, terwijl het rapport de gegradeerde toonde."""
    ev = _events_in(graded)
    assert ev, "geen events — dan toetst deze test niets"
    pos = (graded.get("position") or {}).get("summary", {}) or {}
    assert pos, "geen positieanalyse in de uitvoer"
    n_pos = _pos_total(graded)
    if n_pos:
        assert n_pos == pytest.approx(len(ev), abs=1), (
            f"positieanalyse telt {n_pos} events, rapport toont {len(ev)}")


def test_the_graded_detector_actually_replaced_the_hypopneas(graded):
    """Zonder deze bewaking zou de test hierboven ook slagen als stap 7b
    helemaal niet gedraaid had."""
    diag = graded["respiratory"].get("breath_detector")
    assert diag, "de gegradeerde detector draaide niet"
    assert diag.get("n_candidates", 0) > 0


def test_the_positional_ahi_is_present_and_finite(graded):
    pos = (graded.get("position") or {}).get("summary", {}) or {}
    per = pos.get("ahi_per_pos") or {}
    assert per, "geen AHI per positie"
    for k, v in per.items():
        assert v is None or v >= 0, f"{k}: {v}"


def test_the_recompute_is_wired_for_every_graded_profile():
    """De herberekening hangt aan de detectorkeuze, niet aan één profielnaam."""
    from psgscoring.profiles import PROFILES
    graded_profiles = [n for n, p in PROFILES.items()
                       if p.post_processing.hypopnea_detector == "breath_graded"]
    assert set(graded_profiles) == {
        "aasm_v3_breath", "aasm_v3_prob",
        "aasm_v3_breath_dual", "aasm_v3_prob_dual"}, graded_profiles


def test_the_pipeline_recomputes_position_after_the_graded_step():
    """Structureel: de aanroep staat ná de merge, niet ervoor."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "psgscoring" / "pipeline.py").read_text()
    i_merge = src.index('output["respiratory"]["breath_detector"] = _bdiag')
    i_pos = src.index("positieanalyse herrekend")
    assert i_merge < i_pos, "de herberekening staat niet na stap 7b"
