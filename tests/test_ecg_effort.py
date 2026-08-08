"""ECG-afgeleide ademinspanning — dekking was 35 % (104 van 159 regels).

Deze module beslist mee of een apneu **centraal** of **obstructief** heet, en
dat is het verschil tussen CPAP en een cardiologische verwijzing. Twee methodes
naast elkaar, beide uit Berry et al.:

1. **TECG** (JCSM 2019) — hoogdoorlaat, QRS wegblanken, gelijkrichten,
   integreren. Wat overblijft is intercostale EMG: inspiratoire bursts tijdens
   een apneu betekenen dat de patiënt wél ademt tegen een gesloten keel.
2. **Spectrale classificatie** — hoeveel van het vermogen op de RIP-band zit in
   de hartband tegenover de ademband. Overheerst de hartslag, dan is de
   "beweging" van de band cardiogeen artefact en geen ademinspanning.

Dat tweede is precies de valkuil waarvoor deze module bestaat: cardiogene
pulsatie op de effortbanden laat een centrale apneu eruitzien alsof er
inspanning is.

Er zit één patiëntgroep in waar de standaardbanden botsen. Bij bradycardie —
atleten, bètablokkers — zakt de hartfundamentele naar 0,5–0,6 Hz en overlapt hij
de ademband (0,1–0,5 Hz). `compute_adaptive_cardiac_band` schuift de band dan
mee met de werkelijke hartslag.
"""

import numpy as np
import pytest

from psgscoring.ecg_effort import (
    CARDIAC_BAND_HZ,
    CARDIAC_DOMINANCE_THR,
    RESPIRATORY_BAND_HZ,
    compute_adaptive_cardiac_band,
    compute_tecg,
    detect_r_peaks,
    qrs_blanking,
    spectral_effort_classifier,
)

SF = 200.0


# ─────────────────────────────────────────────────────────────
#  Signaalopbouw
# ─────────────────────────────────────────────────────────────

def _ecg(dur_s=60.0, hr_bpm=60.0, sf=SF, amp=1.0, seed=0):
    """ECG met scherpe R-toppen op een vaste hartslag."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sf)
    x = rng.normal(0, 0.02, n)
    rr = 60.0 / hr_bpm
    for k in range(int(dur_s / rr)):
        i = int((0.3 + k * rr) * sf)
        if 0 <= i < n - 3:
            x[i] += amp                      # R
            x[i - 2] -= amp * 0.15           # Q
            x[i + 2] -= amp * 0.25           # S
    return x


def _band(freq_hz, dur_s=60.0, sf=SF, amp=1.0, seed=1):
    """Een zuivere oscillatie met wat ruis — een effortband die één ding doet."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * sf)) / sf
    return amp * np.sin(2 * np.pi * freq_hz * t) + 0.02 * rng.normal(size=len(t))


# ─────────────────────────────────────────────────────────────
#  R-piekdetectie
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hr", [45.0, 60.0, 90.0])
def test_the_r_peaks_are_found_at_the_imposed_heart_rate(hr):
    peaks = detect_r_peaks(_ecg(hr_bpm=hr), SF)
    verwacht = int(60.0 / hr * 1)      # slagen per seconde
    gevonden = len(peaks) / 60.0
    assert gevonden == pytest.approx(hr / 60.0, rel=0.10), (
        f"{len(peaks)} pieken over 60 s bij {hr} bpm")
    assert verwacht >= 0


def test_the_peaks_are_ordered_and_respect_the_refractory_period():
    """QRS_REFRACTORY_MS = 200 — twee R-toppen binnen 200 ms bestaan niet."""
    peaks = detect_r_peaks(_ecg(hr_bpm=90.0), SF)
    assert len(peaks) > 1
    d = np.diff(peaks) / SF
    assert np.all(d > 0), "pieken niet oplopend"
    assert d.min() >= 0.2 - 1e-9, f"kortste R-R is {d.min():.3f}s"


def test_a_sampling_rate_below_the_qrs_band_is_refused_not_guessed():
    """De 5-30 Hz QRS-bandpass heeft Nyquist boven 30 Hz nodig. Bij 50 Hz
    sampling kan dat niet, en dan is een leeg antwoord eerlijker dan ruis."""
    peaks = detect_r_peaks(_ecg(sf=50.0), 50.0)
    assert len(peaks) == 0


def test_a_flat_signal_yields_no_peaks():
    assert len(detect_r_peaks(np.zeros(int(60 * SF)), SF)) == 0


# ─────────────────────────────────────────────────────────────
#  QRS-blanking en TECG
# ─────────────────────────────────────────────────────────────

def test_blanking_removes_the_qrs_spikes():
    """Dat is het hele punt: het QRS-complex overstemt de EMG eronder."""
    ecg = _ecg(hr_bpm=60.0, amp=5.0)
    blanked = qrs_blanking(ecg, SF)
    assert len(blanked) == len(ecg)
    assert blanked.max() < ecg.max(), "de R-toppen staan er nog"


def test_blanking_leaves_the_signal_between_beats_alone():
    """Alleen rond de R-top wordt vervangen; de rest moet ongemoeid blijven."""
    ecg = _ecg(hr_bpm=60.0, amp=5.0)
    blanked = qrs_blanking(ecg, SF)
    # midden tussen twee slagen: 0,3 s na de eerste R bij 60 bpm
    i = int(0.8 * SF)
    assert blanked[i] == pytest.approx(ecg[i], abs=1e-9)


def test_the_tecg_is_non_negative_and_the_same_length():
    """Gelijkgericht en geïntegreerd — negatieve waarden horen niet te bestaan."""
    tecg = compute_tecg(_ecg(), SF)
    assert len(tecg) == int(60 * SF)
    assert tecg.min() >= -1e-9, f"negatieve TECG-waarde {tecg.min()}"


def test_the_tecg_is_larger_when_there_is_muscle_activity_under_the_ecg():
    """Een ECG met hoogfrequente EMG erop hoort een hogere TECG te geven dan
    een schoon ECG. Dat is de grootheid waar de burstdetectie op leunt."""
    rng = np.random.default_rng(4)
    schoon = _ecg(seed=4)
    met_emg = schoon + rng.normal(0, 0.05, len(schoon))
    assert compute_tecg(met_emg, SF).mean() > compute_tecg(schoon, SF).mean()


# ─────────────────────────────────────────────────────────────
#  De spectrale classificatie
# ─────────────────────────────────────────────────────────────

def test_a_band_that_only_pulses_with_the_heart_reads_as_cardiac():
    """1,2 Hz = 72 bpm, midden in de hartband en ver boven de ademband."""
    sig = _band(1.2)
    r = spectral_effort_classifier(sig, SF, 0, len(sig))
    assert r["cardiac_fraction"] > CARDIAC_DOMINANCE_THR
    assert r["cardiac_dominant"] is True
    assert r["classification_hint"] == "probable_central"


def test_a_band_that_moves_with_the_breathing_reads_as_effort():
    """0,25 Hz = 15 ademhalingen per minuut."""
    sig = _band(0.25)
    r = spectral_effort_classifier(sig, SF, 0, len(sig))
    assert r["respiratory_fraction"] > 0.5
    assert r["cardiac_dominant"] is False
    assert r["classification_hint"] == "effort_present"


def test_the_two_fractions_sum_to_one():
    """Ze delen dezelfde noemer; anders is een van de twee niet te lezen."""
    r = spectral_effort_classifier(_band(0.25), SF, 0, int(60 * SF))
    assert r["cardiac_fraction"] + r["respiratory_fraction"] == pytest.approx(1.0, abs=0.01)


def test_a_segment_too_short_to_analyse_says_so():
    """Minder dan 4 s geeft geen bruikbaar spectrum. Dan hoort er geen
    classificatie uit te komen, ook geen voorzichtige."""
    sig = _band(0.25, dur_s=2.0)
    r = spectral_effort_classifier(sig, SF, 0, len(sig))
    assert r["classification_hint"] == "insufficient_data"
    assert r["cardiac_dominant"] is False


def test_a_silent_band_is_not_called_cardiac():
    """Een dood kanaal mag geen "waarschijnlijk centraal" opleveren — dat zou
    een apneu van type doen wisselen op grond van niets."""
    r = spectral_effort_classifier(np.zeros(int(60 * SF)), SF, 0, int(60 * SF))
    assert r["cardiac_dominant"] is False
    assert r["classification_hint"] in ("no_signal", "insufficient_data")


def test_mixed_content_is_not_forced_into_a_verdict():
    """Half ademhaling, half hartslag: geen van beide domineert, en dan hoort
    de classificatie niet naar 'centraal' te neigen."""
    sig = _band(0.25, seed=2) + _band(1.2, seed=3)
    r = spectral_effort_classifier(sig, SF, 0, len(sig))
    assert r["cardiac_dominant"] is False


# ─────────────────────────────────────────────────────────────
#  De adaptieve hartband — de bradycardiepatiënt
# ─────────────────────────────────────────────────────────────

def test_without_r_peaks_it_falls_back_to_the_default_band():
    assert compute_adaptive_cardiac_band(None, SF) == CARDIAC_BAND_HZ


def test_a_normal_heart_rate_keeps_the_band_clear_of_the_respiratory_range():
    lo, hi = compute_adaptive_cardiac_band(detect_r_peaks(_ecg(hr_bpm=72.0), SF), SF)
    assert lo > RESPIRATORY_BAND_HZ[1], "hartband overlapt de ademband"
    assert lo < 1.2 < hi, f"72 bpm = 1,2 Hz valt buiten [{lo}, {hi}]"


def test_a_bradycardic_patient_moves_the_band_down():
    """Bij 42 bpm ligt de fundamentele op 0,7 Hz. De standaardband begint pas
    op 0,8 en zou de hartslag dus missen — precies de atleet of de patiënt op
    bètablokkers."""
    traag = compute_adaptive_cardiac_band(detect_r_peaks(_ecg(hr_bpm=42.0), SF), SF)
    normaal = compute_adaptive_cardiac_band(detect_r_peaks(_ecg(hr_bpm=72.0), SF), SF)
    assert traag[0] < normaal[0], f"band verschoof niet mee: {traag} vs {normaal}"
    assert traag[0] <= 0.7 <= traag[1], f"0,7 Hz valt buiten {traag}"


def test_the_band_stays_a_valid_interval():
    for hr in (40.0, 60.0, 100.0):
        lo, hi = compute_adaptive_cardiac_band(detect_r_peaks(_ecg(hr_bpm=hr), SF), SF)
        assert 0 < lo < hi, f"{hr} bpm gaf een ongeldige band [{lo}, {hi}]"


def test_too_few_beats_fall_back_rather_than_extrapolate():
    """Twee R-toppen zijn geen hartslagschatting."""
    assert compute_adaptive_cardiac_band(np.array([100, 300]), SF) == CARDIAC_BAND_HZ
