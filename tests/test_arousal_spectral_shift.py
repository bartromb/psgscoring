"""Arousal: het schaalvrije spectrale criterium (opt-in) en de byte-identiteit
van het bestaande pad.

Twee dingen worden hier vastgelegd.

1. **Wat het oude criterium meet.** `detect_arousals` toetst het VERMOGEN in
   alpha+theta+beta tegen een basislijn uit de opname zelf. Een stuk EEG dat
   alleen in amplitude toeneemt -- zelfde spectrum, alles maal 1,6 -- haalt die
   drempel en wordt als AASM-arousal gescoord. De AASM beschrijft een
   verschuiving van de FREQUENTIE, niet van de amplitude. `test_pure_amplitude_*`
   legt beide kanten vast: het oude pad scoort de amplitudestap, het nieuwe niet.

2. **Een asymmetrie die er per ongeluk uit gepoetst wordt.** In REM toetst
   fase 1 alleen `alpha_pow`, maar vergelijkt dat met een basislijn over
   alpha+theta+beta. Dat ziet er als een slordigheid uit en is bij het inbouwen
   van `spectral_shift` dan ook prompt "opgeruimd" -- waarna het aantal
   REM-arousals stil veranderde (PSG-IPA SN3: 61 -> 73 events, met alle 934
   tests groen). `test_rem_baseline_*` maakt de asymmetrie expliciet.
"""
import numpy as np

from psgscoring.arousal import (
    AROUSAL_SHIFT_ABRUPT,
    AROUSAL_SHIFT_DELTA,
    detect_arousals,
)

SF = 100.0
DUR_S = 900          # 15 min
AT_S = 600.0         # moment van de gebeurtenis
LEN_S = 6.0


def _nrem_background(seed=11):
    """N2/N3-achtergrond: delta-gedomineerd, weinig snelle banden.

    Delta hoort NIET bij `arousal_pow` maar wel bij de noemer van `r`, dus dit
    is de achtergrond waarop de twee criteria uiteenlopen.
    """
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    return t, (60.0 * np.sin(2 * np.pi * 1.5 * t)
               + 6.0 * np.sin(2 * np.pi * 6.0 * t)
               + 4.0 * np.sin(2 * np.pi * 10.0 * t)
               + 2.0 * np.sin(2 * np.pi * 20.0 * t)
               + rng.normal(0.0, 1.0, t.size))


def _rem_background(seed=20260821):
    """REM-achtergrond: theta-gedomineerd.

    Theta zit in `arousal_pow`, dus hier is de GECOMBINEERDE basislijn veel
    hoger dan de alpha-basislijn -- precies het verschil dat
    `test_rem_baseline_*` vastlegt.
    """
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    return t, (40.0 * np.sin(2 * np.pi * 6.0 * t)
               + 4.0 * np.sin(2 * np.pi * 10.0 * t)
               + 2.0 * np.sin(2 * np.pi * 20.0 * t)
               + rng.normal(0.0, 1.0, t.size))


def _amplitude_step(gain=1.8):
    """PURE amplitudetoename: hetzelfde signaal maal `gain`, spectrum identiek.

    Geen frequentieverschuiving, dus per AASM geen arousal.
    """
    t, eeg = _nrem_background()
    s, e = int(AT_S * SF), int((AT_S + LEN_S) * SF)
    eeg = eeg.copy()
    eeg[s:e] *= gain
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def _rem_amplitude_step(gain=1.8):
    """Dezelfde amplitudestap, maar op een REM-achtergrond."""
    t, eeg = _rem_background()
    s, e = int(AT_S * SF), int((AT_S + LEN_S) * SF)
    eeg = eeg.copy()
    eeg[s:e] *= gain
    return eeg * 1e-6, ["R"] * int(DUR_S / 30)


def _frequency_shift():
    """Echte verschuiving van het spectrale zwaartepunt bij gelijke amplitude."""
    t, eeg = _nrem_background()
    rng = np.random.default_rng(7)
    s, e = int(AT_S * SF), int((AT_S + LEN_S) * SF)
    eeg = eeg.copy()
    eeg[s:e] = (28.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                + 28.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                + rng.normal(0.0, 1.0, e - s))
    return eeg * 1e-6, ["N2"] * int(DUR_S / 30)


def _hits(res):
    return [e["onset_s"] for e in res["events"] if AT_S - 4 <= e["onset_s"] <= AT_S + 8]


# ── 1. wat de twee criteria meten ────────────────────────────────────────

def test_pure_amplitude_step_is_scored_by_the_power_criterion():
    """Karakterisering, geen goedkeuring: het huidige pad scoort een
    amplitudestap zonder frequentieverschuiving als arousal.

    Deze test hoort te BREKEN wanneer het vermogenscriterium ooit vervangen
    wordt -- dan is dit precies de winst die geboekt is, en hoort de
    verwachting hier mee te veranderen.
    """
    eeg, hypno = _amplitude_step()
    res = detect_arousals(eeg, SF, hypno)
    assert res["success"], res.get("error")
    assert _hits(res), "fixture is inert geworden -- de amplitudestap haalt de drempel niet meer"


def test_pure_amplitude_step_is_rejected_by_the_spectral_criterion():
    """De ingreep: een amplitudestap verplaatst het spectrale zwaartepunt niet."""
    eeg, hypno = _amplitude_step()
    res = detect_arousals(eeg, SF, hypno, spectral_shift=True)
    assert res["success"], res.get("error")
    assert not _hits(res), f"amplitudestap toch gescoord: {_hits(res)}"


def test_real_frequency_shift_is_scored_by_the_spectral_criterion():
    """En een echte frequentieverschuiving wordt wel gezien -- anders zou de
    vorige test ook slagen met een detector die niets detecteert.

    Beide criteria zien deze; de winst zit niet in wat er extra gevonden wordt
    maar in wat er NIET meer meetelt.
    """
    eeg, hypno = _frequency_shift()
    assert _hits(detect_arousals(eeg, SF, hypno)), "fixture inert voor het oude pad"
    res = detect_arousals(eeg, SF, hypno, spectral_shift=True)
    assert res["success"], res.get("error")
    assert _hits(res), (
        "frequentieverschuiving gemist; "
        f"onsets={[e['onset_s'] for e in res['events']]}"
    )


def test_spectral_criterion_is_invariant_under_amplitude_scaling():
    """De hele opname maal tien is dezelfde opname.

    `shift_delta` is een absoluut increment op een FRACTIE, niet op een
    vermogen -- juist daarom mag het over opnames heen dezelfde betekenis
    houden. Zou het op vermogen werken, dan faalt deze test onmiddellijk.
    """
    eeg, hypno = _frequency_shift()
    a = detect_arousals(eeg, SF, hypno, spectral_shift=True)
    b = detect_arousals(eeg * 10.0, SF, hypno, spectral_shift=True)
    assert [e["onset_s"] for e in a["events"]] == [e["onset_s"] for e in b["events"]]


# ── 2. byte-identiteit van het bestaande pad ─────────────────────────────

def test_rem_baseline_uses_combined_power_not_alpha():
    """In REM hoort `alpha_pow` vergeleken te worden met een basislijn over
    alpha+theta+beta -- niet met een alpha-basislijn.

    Bij een alpha-basislijn ligt de drempel een factor ~100 lager en verschijnt
    de amplitudestap hieronder wel als REM-arousal. Zie de moduledocstring.
    """
    eeg, hypno = _rem_amplitude_step()
    res = detect_arousals(eeg, SF, hypno)
    assert res["success"], res.get("error")
    assert not _hits(res), (
        "amplitudestap in REM gescoord -- de REM-basislijn draait nu "
        f"waarschijnlijk op alpha_pow in plaats van arousal_pow. {_hits(res)}"
    )


def test_flag_off_ignores_the_spectral_parameters():
    """Met de vlag uit mogen shift_delta/shift_abrupt niets uithalen."""
    eeg, hypno = _amplitude_step()
    base = detect_arousals(eeg, SF, hypno)
    for dlt, abr in ((0.01, 0.01), (0.99, 0.99)):
        other = detect_arousals(eeg, SF, hypno, spectral_shift=False,
                                shift_delta=dlt, shift_abrupt=abr)
        assert other["events"] == base["events"]
        assert other["summary"] == base["summary"]


# ── 3. doorvoer en preregistratie ────────────────────────────────────────

def test_thresholds_match_the_preregistration():
    """Vastgelegd in docs/arousal_spectral_shift_preregistratie.md vóór enige
    meting. Wie ze verandert, verandert een preregistratie."""
    assert AROUSAL_SHIFT_DELTA == 0.15
    assert AROUSAL_SHIFT_ABRUPT == 0.10


def test_profile_flag_reaches_the_profile_dict():
    from psgscoring.constants import SCORING_PROFILES
    for name, d in SCORING_PROFILES.items():
        assert "AROUSAL_SPECTRAL_SHIFT" in d, name
        assert isinstance(d["AROUSAL_SPECTRAL_SHIFT"], bool), name
    aan = [n for n, d in SCORING_PROFILES.items() if d["AROUSAL_SPECTRAL_SHIFT"]]
    assert not aan, f"vlag hoort nergens default aan te staan: {aan}"
