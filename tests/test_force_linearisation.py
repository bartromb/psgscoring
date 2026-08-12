"""De wortellinearisatie hing aan de MONTAGE in plaats van aan het profiel.

`_setup_hypop_channel` past AASM Regel 3 alleen toe wanneer het hypopnee-kanaal
een ánder kanaal is dan het apneukanaal. Op een montage met twee flowkanalen
(MESA: `Pres` + `Therm`) wordt de hypopnee-envelope dus gelineariseerd; op een
montage met één kanaal (PSG-IPA) hergebruikt de code de apneu-envelope, die
zonder linearisatie is berekend.

Zonder linearisatie meet een werkelijke flowreductie van 50 % als 75 %
amplitudereductie. Reducties worden stelselmatig overschat, meer kandidaten
halen het 30 %-criterium, en `hypopnea_strictness` is op die conventie geijkt.
Dat is de vermoedelijke oorzaak van de tegengestelde bias-richting: +1,77 op
PSG-IPA tegen −11 tot −15 op MESA.

`hypopnea_force_linearisation` maakt de keuze een profielbeslissing in plaats
van een montage-eigenschap. Default `False` = bestaand gedrag.

Deze toetsen leggen twee dingen vast die allebei nodig zijn:
  1. met de default verandert er NIETS — geen enkel bestaand profiel beweegt;
  2. met het veld aan verandert er WEL iets, en op de juiste tak.
Zonder (2) zou (1) triviaal slagen met een veld dat nergens is aangesloten.
"""

import numpy as np
import pytest

from psgscoring.constants import _profile_to_legacy_dict
from psgscoring.profiles import PROFILES, get_profile
from psgscoring.respiratory import _setup_hypop_channel, preprocess_flow

SF = 32.0
DUUR_S = 600


def _flow(amp=100.0):
    """Piekerige ademhaling met één duidelijke reductie.

    Piekerig en niet sinusvormig: de linearisatie werkt op de amplitude, en
    op een zuivere sinus is het verschil tussen wel en niet lineariseren
    kleiner dan de ruis waarmee je het meet.
    """
    t = np.arange(int(DUUR_S * SF)) / SF
    s = np.sin(2 * np.pi * 0.25 * t)
    x = amp * np.sign(s) * np.abs(s) ** 3
    x[(t >= 300) & (t < 320)] *= 0.5          # halvering van de amplitude
    return np.abs(x) + 1.0                     # positief: drukopnemer-achtig


# ──────────────────────────────────────────────────────────────
#  1. Default verandert niets
# ──────────────────────────────────────────────────────────────

def test_the_default_is_current_behaviour():
    from psgscoring.profiles import PostProcessingRules
    assert PostProcessingRules().hypopnea_force_linearisation is False


@pytest.mark.parametrize("naam", sorted(PROFILES))
def test_no_shipped_profile_enables_it(naam):
    """Elk bestaand profiel blijft byte-identiek zolang niemand het aanzet."""
    p = get_profile(naam)
    assert p.post_processing.hypopnea_force_linearisation is False, naam


@pytest.mark.parametrize("naam", sorted(PROFILES))
def test_the_field_reaches_the_legacy_dict(naam):
    """Een veld dat de legacy-dict niet haalt, wordt door de detector nooit
    gelezen — dan slaagt de byte-identiteitstest omdat er niets is aangesloten."""
    d = _profile_to_legacy_dict(get_profile(naam))
    assert "HYPOPNEA_FORCE_LINEARISATION" in d
    assert d["HYPOPNEA_FORCE_LINEARISATION"] is False


# ──────────────────────────────────────────────────────────────
#  2. Aangezet verandert het wél, en op de juiste tak
# ──────────────────────────────────────────────────────────────

def _run(force, gedeeld=True):
    flow = _flow()
    flow_env = preprocess_flow(flow, SF, is_nasal_pressure=False)
    baseline = np.full_like(flow_env, float(np.percentile(flow_env, 95)))
    flow_norm = np.clip(flow_env / baseline, 0, 2)
    result = {}
    env, norm, bl, sf_hy = _setup_hypop_channel(
        flow, SF, flow_env, baseline, flow_norm, SF,
        hypno=["N2"] * (DUUR_S // 30), artifact_epochs=[], pos_changes=[],
        pos_data=None, sf_pos=None, result=result,
        precomputed_hypop_baseline=baseline,
        hypop_is_same_channel=gedeeld,
        force_linearisation=force,
    )
    return env, bl, result


def test_a_shared_channel_is_not_linearised_by_default():
    env, _bl, result = _run(force=False)
    assert result["hypopnea_channel_shared"] is True
    assert result["hypopnea_linearised"] is False
    assert result["hypopnea_linearisation_forced"] is False


def test_forcing_it_linearises_the_shared_channel():
    env, _bl, result = _run(force=True)
    assert result["hypopnea_channel_shared"] is True, "nog steeds hetzelfde kanaal"
    assert result["hypopnea_linearised"] is True
    assert result["hypopnea_linearisation_forced"] is True


def test_forcing_it_actually_changes_the_envelope():
    """De vlag mag niet alleen metadata verzetten."""
    env_uit, _, _ = _run(force=False)
    env_aan, _, _ = _run(force=True)
    assert env_uit.shape == env_aan.shape
    assert not np.allclose(env_uit, env_aan), \
        "envelope identiek — de linearisatie is niet toegepast"


def test_the_baseline_follows_the_new_envelope():
    """De kern van de implementatie. De voorberekende basislijn staat op de
    ONGELINEARISEERDE schaal; hergebruiken zou een gelineariseerde teller door
    een ongelineariseerde noemer delen, en dan meet de verhouding iets anders
    dan een flowreductie."""
    _env_uit, bl_uit, _ = _run(force=False)
    _env_aan, bl_aan, _ = _run(force=True)
    assert not np.allclose(np.asarray(bl_uit, dtype=float),
                           np.asarray(bl_aan, dtype=float)), \
        "basislijn hergebruikt terwijl de envelope veranderde"


def test_a_reduction_measures_smaller_after_linearisation():
    """De richting die het hele argument draagt: linearisatie VERKLEINT de
    gemeten reductie, dus hetzelfde operatiepunt wordt strenger."""
    env_uit, bl_uit, _ = _run(force=False)
    env_aan, bl_aan, _ = _run(force=True)
    n = len(env_uit)
    ev = slice(int(305 * SF), int(315 * SF))       # midden van de reductie
    ref = slice(int(100 * SF), int(200 * SF))      # rustige ademhaling ervoor
    red_uit = 1 - env_uit[ev].mean() / env_uit[ref].mean()
    red_aan = 1 - env_aan[ev].mean() / env_aan[ref].mean()
    assert red_aan < red_uit, (
        f"gelineariseerd {red_aan:.3f} niet kleiner dan ongelineariseerd "
        f"{red_uit:.3f} — de richting van het argument klopt niet")


def test_a_separate_channel_is_unaffected_by_the_flag():
    """Op een montage met twee kanalen werd al gelineariseerd; het veld mag
    daar niets veranderen."""
    _e1, _b1, r_uit = _run(force=False, gedeeld=False)
    _e2, _b2, r_aan = _run(force=True, gedeeld=False)
    assert r_uit["hypopnea_linearised"] is True
    assert r_aan["hypopnea_linearised"] is True
    assert r_uit["hypopnea_linearisation_forced"] is False
    assert r_aan["hypopnea_linearisation_forced"] is False
