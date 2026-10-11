"""De drie reparaties uit de pipeline-review die getallen verschuiven, staan
achter een profielveld -- default aan, `mesa_shhs` en `chicago_1999` op het
oude gedrag. Zelfde idioom als `plm_time_base` en `rip_quality_scale_free`.

  * `indices_on_final_event_list`   -- positie/PLM op de definitieve lijst
  * `ancillary_tst_excludes_artifacts` -- externe arousal-index en
                                       CSR-dichtheid op de AHI-noemer
  * `rem_ahi_alias_excludes_uncertain` -- alias op het `ahi_total`-paar

Hier staat vast WAT gepind is en dat de pin het oude getal werkelijk
oplevert; de reparaties zelf worden in hun eigen testbestanden gemeten.
"""
import numpy as np
import pytest

from psgscoring.constants import SCORING_PROFILES
from psgscoring.profiles import get_profile, list_profiles

VELDEN = ("indices_on_final_event_list", "ancillary_tst_excludes_artifacts",
          "rem_ahi_alias_excludes_uncertain")
SLEUTELS = ("INDICES_ON_FINAL_EVENT_LIST", "ANCILLARY_TST_EXCLUDES_ARTIFACTS",
            "REM_AHI_ALIAS_EXCLUDES_UNCERTAIN")
BEVROREN = ("mesa_shhs", "chicago_1999")


@pytest.mark.parametrize("veld,sleutel", list(zip(VELDEN, SLEUTELS)))
def test_default_aan_en_bevroren_profielen_uit(veld, sleutel):
    for naam in list_profiles():
        waarde = getattr(get_profile(naam).post_processing, veld)
        verwacht = naam not in BEVROREN
        assert waarde is verwacht, f"{naam}.{veld} = {waarde}"
        assert SCORING_PROFILES[naam][sleutel] is verwacht, (
            f"{naam}: registry loopt uit de pas met het profiel voor {sleutel}")


# ── de pin levert het oude getal ────────────────────────────────────────

def test_de_rem_ahi_pin_is_het_incl_uncertain_paar():
    """Op de reproductieprofielen is de alias het oude getal: alle events in
    REM, inclusief de kale uncertain."""
    from psgscoring.pipeline import _compute_rera_rdi
    from psgscoring.respiratory import _compute_summary

    hypno = ["R"] * 20 + ["N2"] * 100
    ev = [{"type": "obstructive", "onset_s": 30.0, "duration_s": 15.0,
           "stage": "R", "epoch": 1, "confidence": 0.9, "desaturation_pct": 4.0},
          {"type": "uncertain", "onset_s": 120.0, "duration_s": 15.0,
           "stage": "R", "epoch": 4, "confidence": 0.9, "desaturation_pct": 4.0}]
    out = {"respiratory": {"success": True, "events": ev,
                           "summary": _compute_summary(ev, hypno),
                           "rejected_hypopneas": [], "_breaths": []}}
    _compute_rera_rdi(out, hypno, arousals=[], rem_alias_excludes_uncertain=False)
    s = out["respiratory"]["summary"]
    assert s["rem_ahi"] == s["ahi_rem_incl_uncertain"] == 12.0
    assert s["ahi_rem"] == 6.0


def test_de_tst_pin_deelt_door_de_volledige_slaaptijd(monkeypatch):
    mne = pytest.importorskip("mne")
    import psgscoring
    from psgscoring.indices import tst_hours

    monkeypatch.setitem(SCORING_PROFILES["aasm_v3_rec"],
                        "ANCILLARY_TST_EXCLUDES_ARTIFACTS", False)
    sf, dur = 32.0, 600
    n = int(sf * dur)
    t = np.arange(n) / sf
    info = mne.create_info(["Resp nasal", "SaO2"], sf, ["misc", "misc"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(n, 96.0)]), info, verbose="ERROR")
    hypno = ["N2"] * (dur // 30)
    art = [0, 1, 2, 3, 4, 5]
    arousals = [{"onset_s": 30.0 * k + 5.0, "duration_s": 3.0} for k in range(12)]
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno, artifact_epochs=art,
                                         arousal_events=arousals,
                                         scoring_profile="aasm_v3_rec")
    assert out["arousal"]["summary"]["arousal_index"] == round(
        12 / tst_hours(hypno, None), 2)


def test_de_eventlijst_pin_rekent_op_de_momentopname_van_stap_1(monkeypatch):
    """Met de vlag uit ziet de PLM-stap de lijst van stap 1 -- hier leeg,
    want de envelope-detector vindt niets op een schone sinus -- en niet de
    tien gegradeerde hypopneeën van stap 7b."""
    mne = pytest.importorskip("mne")
    import psgscoring
    import psgscoring.breath_scoring as bs

    monkeypatch.setitem(SCORING_PROFILES["aasm_v3_breath"],
                        "INDICES_ON_FINAL_EVENT_LIST", False)
    sf, first, gap, n_moves = 64.0, 60.0, 20.0, 40
    dur = first + n_moves * gap + 120.0
    n = int(dur * sf)
    t = np.arange(n) / sf
    rng = np.random.default_rng(23)
    leg = rng.normal(0.0, 1e-6, n)
    for k in range(n_moves):
        a, b = int((first + k * gap) * sf), int((first + k * gap + 1.5) * sf)
        leg[a:b] += rng.normal(0.0, 60e-6, b - a)
    hyps = [{"type": "hypopnea", "onset_s": first + k * gap - 15.0,
             "duration_s": 15.0, "stage": "N2",
             "epoch": int((first + k * gap - 15.0) // 30), "confidence": 0.8,
             "desaturation_pct": 3.5} for k in range(10)]
    monkeypatch.setattr(bs, "score_hypopneas_breathwise",
                        lambda **kw: (list(hyps), {"n_scored": 10,
                                                   "n_candidates": 10}))
    info = mne.create_info(["Resp nasal", "SaO2", "EMG LAT", "EMG RAT"], sf,
                           ["misc", "misc", "emg", "emg"])
    raw = mne.io.RawArray(np.vstack([np.sin(2 * np.pi * 0.25 * t),
                                     np.full(n, 97.0), leg, leg]),
                          info, verbose="ERROR")
    hypno = ["N2"] * int(np.ceil(dur / 30.0))
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_breath")
    assert len(out["respiratory"]["events"]) == 10      # 7b heeft vervangen
    assert out["plm"]["summary"]["n_resp_associated"] == 0   # maar de pin kijkt naar stap 1
