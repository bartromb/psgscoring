"""Event-gebonden hartslagstijging (heart rate surge) als rapportmaat.

Huang e.a., *JACC Asia* 2026 (10.1016/j.jacasi.2026.08.040) definiëren de
heart rate surge als "peak heart rate within the event-aligned window minus
mean heart rate during the 60-second pre-event baseline" en tonen op 4 790
deelnemers dat de maat het cardiovasculaire risico stratificeert bóven de
hypoxic burden. De bibliotheek berekende die hartslagrespons al per
arousalkandidaat (autonome re-ranker); hier staat ze per respiratoir event en
als nachtmaat in het resultaat. Geen index verandert erdoor.

De toetsen leggen vast wat het getal betekent: een bekende stijging op een
bekend event komt er precies uit, een stijging buiten het venster telt niet,
sensoruitval in de basislijn maakt het event ongeldig (en niet nul), een ruw
ECG is geen hartfrequentie en levert geen getal, en de pipeline draagt het blok
mét reden wanneer er niets te berekenen valt.
"""
from __future__ import annotations

import numpy as np
import pytest

from psgscoring.ancillary import compute_event_hr_surge

SF = 1.0
N = 3600                      # één uur op 1 Hz
HYPNO = ["N2"] * (N // 30)
DUR = 20.0


def _pulse(bpm: float = 60.0, jitter: float = 0.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return bpm + jitter * rng.standard_normal(N)


def _events(onsets, dur: float = DUR, typ: str = "hypopnea") -> list[dict]:
    return [{"onset_s": float(o), "duration_s": dur, "type": typ} for o in onsets]


def _bump(hr: np.ndarray, start_s: float, delta: float, len_s: int = 6) -> None:
    a = int(start_s)
    hr[a:a + len_s] += delta


# ── 1. De definitie: piek in het eventvenster min gemiddelde van de 60 s ervoor ──

def test_known_surges_are_recovered_per_event_and_as_night_measure():
    hr = _pulse()
    onsets = [300, 600, 900, 1200]
    deltas = [20.0, 10.0, 30.0, 20.0]
    for o, d in zip(onsets, deltas):
        _bump(hr, o + DUR + 3, d)          # piek 3 s na het eventeinde
    r = compute_event_hr_surge(hr, SF, _events(onsets), HYPNO, source="pulse")

    assert r["n_events_total"] == 4
    assert r["n_events_valid"] == 4
    assert [round(x) for x in r["per_event_bpm"]] == [20, 10, 30, 20]
    assert r["hr_surge_mean_bpm"] == pytest.approx(20.0, abs=0.2)
    assert r["hr_surge_median_bpm"] == pytest.approx(20.0, abs=0.2)
    assert r["unit"] == "bpm"
    assert r["method"] == "huang2026"
    assert r["source"] == "pulse"
    assert r["baseline_s"] == 60.0
    assert r["window"]["start"] == "event_onset"
    assert r["window"]["post_event_s"] == 15.0
    assert r["reason"] is None


def test_the_baseline_is_the_mean_of_the_sixty_seconds_before_onset():
    """Een hogere basislijn vóór het event verlaagt de stijging met hetzelfde
    bedrag: de maat is relatief, niet absoluut."""
    hr = _pulse()
    o = 600.0
    hr[int(o - 60):int(o)] += 10.0      # basislijn 70 i.p.v. 60
    _bump(hr, o + DUR + 2, 25.0)        # piek 85
    r = compute_event_hr_surge(hr, SF, _events([o]), HYPNO, source="pulse")
    assert r["per_event_bpm"][0] == pytest.approx(15.0, abs=0.2)


def test_a_rise_outside_the_event_window_does_not_count():
    hr = _pulse()
    o = 600.0
    _bump(hr, o + DUR + 15.0 + 10, 30.0)   # 25 s na het eventeinde: buiten het venster
    r = compute_event_hr_surge(hr, SF, _events([o]), HYPNO, source="pulse")
    assert r["n_events_valid"] == 1
    assert abs(r["per_event_bpm"][0]) < 0.5


def test_the_window_covers_the_event_itself_and_the_post_event_seconds():
    """Een piek tijdens het event (bv. een korte arousal halverwege) telt mee,
    net als een piek tot 15 s na het einde."""
    hr = _pulse()
    o = 600.0
    _bump(hr, o + 5, 12.0)                  # tijdens het event
    r1 = compute_event_hr_surge(hr, SF, _events([o]), HYPNO, source="pulse")
    hr = _pulse()
    _bump(hr, o + DUR + 14, 18.0, len_s=1)  # laatste seconde van het venster
    r2 = compute_event_hr_surge(hr, SF, _events([o]), HYPNO, source="pulse")
    assert r1["per_event_bpm"][0] == pytest.approx(12.0, abs=0.2)
    assert r2["per_event_bpm"][0] == pytest.approx(18.0, abs=0.2)


# ── 2. Ongeldige events zijn geen nul ────────────────────────────────────────

def test_sensor_dropout_in_the_baseline_invalidates_the_event_not_the_night():
    hr = _pulse()
    onsets = [300, 600, 900]
    for o in onsets:
        _bump(hr, o + DUR + 3, 20.0)
    hr[600 - 60:600 - 10] = 0.0            # 50 van 60 basislijnseconden onmogelijk
    r = compute_event_hr_surge(hr, SF, _events(onsets), HYPNO, source="pulse")
    assert r["n_events_total"] == 3
    assert r["n_events_valid"] == 2
    assert r["n_events_invalid_baseline"] == 1
    assert r["hr_surge_mean_bpm"] == pytest.approx(20.0, abs=0.2)


def test_implausible_samples_in_the_window_are_ignored_not_peaks():
    hr = _pulse()
    o = 600.0
    _bump(hr, o + DUR + 3, 20.0)
    hr[int(o + 8)] = 300.0                  # één onmogelijk sample in het venster
    r = compute_event_hr_surge(hr, SF, _events([o]), HYPNO, source="pulse")
    assert r["per_event_bpm"][0] == pytest.approx(20.0, abs=0.2)


def test_an_event_in_wake_is_counted_but_not_measured():
    hr = _pulse()
    o = 600.0
    _bump(hr, o + DUR + 3, 20.0)
    hypno = list(HYPNO)
    hypno[int(o // 30)] = "W"
    r = compute_event_hr_surge(hr, SF, _events([o]), hypno, source="pulse")
    assert r["n_events_total"] == 1
    assert r["n_events_valid"] == 0
    assert r["n_events_wake"] == 1
    assert r["hr_surge_mean_bpm"] is None


def test_a_pulse_channel_at_four_hertz_gives_the_same_answer():
    hr = _pulse()
    o = 600.0
    _bump(hr, o + DUR + 3, 20.0)
    hr4 = np.repeat(hr, 4)
    r1 = compute_event_hr_surge(hr, 1.0, _events([o]), HYPNO, source="pulse")
    r4 = compute_event_hr_surge(hr4, 4.0, _events([o]), HYPNO, source="pulse")
    assert r4["per_event_bpm"][0] == pytest.approx(r1["per_event_bpm"][0], abs=0.2)


# ── 3. Geen getal zonder hartfrequentie ──────────────────────────────────────

def test_a_raw_ecg_waveform_gives_no_number_but_a_reason():
    rng = np.random.default_rng(5)
    ecg = 120.0 * np.sin(2 * np.pi * 1.2 * np.arange(N) / SF) + 30 * rng.standard_normal(N)
    r = compute_event_hr_surge(ecg, SF, _events([600.0]), HYPNO, source="ecg")
    assert r["hr_surge_mean_bpm"] is None
    assert r["n_events_valid"] == 0
    assert r["source"] == "ecg"
    assert "R-piek" in r["reason"]


def test_missing_heart_rate_gives_no_number_but_a_reason():
    r = compute_event_hr_surge(None, None, _events([600.0]), HYPNO, source="pulse")
    assert r["hr_surge_mean_bpm"] is None
    assert r["n_events_total"] == 1
    assert "hartslagreeks" in r["reason"]


def test_no_respiratory_events_gives_no_number_but_a_reason():
    r = compute_event_hr_surge(_pulse(), SF, [], HYPNO, source="pulse")
    assert r["hr_surge_mean_bpm"] is None
    assert r["n_events_total"] == 0
    assert "geen respiratoire events" in r["reason"]


# ── 4. De pipeline draagt het blok ───────────────────────────────────────────

mne = pytest.importorskip("mne")

PSF = 32.0
PDUR_S = 1800
PN = int(PDUR_S * PSF)
PHYPNO = ["N2"] * (PDUR_S // 30)
PONSETS = list(range(120, PDUR_S - 120, 90))


def _raw(pulse: str | None = "pulse"):
    t = np.arange(PN) / PSF
    rng = np.random.default_rng(5)
    flow = np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.normal(size=PN)
    spo2 = np.full(PN, 96.0)
    hr = np.full(PN, 60.0)
    for t0 in PONSETS:
        a, b = int(t0 * PSF), int((t0 + 20) * PSF)
        flow[a:b] *= 0.3
        spo2[b:b + int(15 * PSF)] -= 5.0
        hr[b + int(3 * PSF):b + int(9 * PSF)] += 20.0     # +20 bpm na elk event
    names = ["Resp nasal", "SaO2", "Thorax", "Abdomen", "EEG C4-M1", "EMG chin"]
    types = ["misc", "misc", "misc", "misc", "eeg", "emg"]
    chans = [flow, spo2,
             0.6 * np.sin(2 * np.pi * 0.25 * t),
             0.6 * np.sin(2 * np.pi * 0.25 * t),
             rng.normal(0, 20e-6, PN), rng.normal(0, 5e-6, PN)]
    cmap = {"flow": "Resp nasal", "spo2": "SaO2", "thorax": "Thorax",
            "abdomen": "Abdomen"}
    if pulse == "pulse":
        names.append("Pulse"); types.append("misc"); chans.append(hr)
        cmap["pulse"] = "Pulse"
    elif pulse == "ecg":
        ecg = 1e-3 * np.sin(2 * np.pi * 1.0 * t)
        names.append("ECG"); types.append("ecg"); chans.append(ecg)
        cmap["ecg"] = "ECG"
    info = mne.create_info(names, PSF, types)
    return mne.io.RawArray(np.vstack(chans), info, verbose="ERROR"), cmap


def _run(pulse):
    import psgscoring
    raw, cmap = _raw(pulse)
    # Zelfde opbouw als tests/test_ahi_rule_as_run.py: onder `aasm_v3_breath`
    # scoort de ademteugdetector de 30 %-flowreducties mét desaturatie.
    return psgscoring.run_pneumo_analysis(raw, hypno=PHYPNO, channel_map=cmap,
                                          scoring_profile="aasm_v3_breath")


def test_pipeline_carries_the_block_and_mirrors_it_in_the_heart_rate_summary():
    out = _run("pulse")
    assert out["respiratory"]["success"]
    assert len(out["respiratory"]["events"]) >= 5, "de toets vraagt gescoorde events"
    hs = out["hr_surge"]
    assert hs["method"] == "huang2026"
    assert hs["n_events_valid"] >= 5
    assert hs["hr_surge_mean_bpm"] == pytest.approx(20.0, abs=1.0)
    hr_sum = out["heart_rate"]["summary"]
    assert hr_sum["event_hr_surge_bpm"] == hs["hr_surge_mean_bpm"]
    assert hr_sum["event_hr_surge_method"] == "huang2026"
    assert hr_sum["event_hr_surge_n"] == hs["n_events_valid"]


def test_pipeline_without_a_pulse_channel_carries_the_block_with_a_reason():
    out = _run(None)
    hs = out["hr_surge"]
    assert hs["hr_surge_mean_bpm"] is None
    assert hs["reason"]
    assert "event_hr_surge_bpm" not in (out["heart_rate"].get("summary") or {})


def test_pipeline_with_only_a_raw_ecg_never_reports_a_surge():
    out = _run("ecg")
    hs = out["hr_surge"]
    assert hs["hr_surge_mean_bpm"] is None
    assert hs["source"] == "ecg"
    assert "R-piek" in hs["reason"]
