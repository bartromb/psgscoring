"""PLM: `result["events"]` is afgekapt, en dat hoort zichtbaar te zijn.

`analyze_plm` zet `result["events"] = plm_eligible[:200]` zonder toelichting
in de code en zonder spoor in de samenvatting. Op PSG-IPA SN1 zijn dat 200
van 660 gedetecteerde bewegingen. Alles wat verderop `output["plm"]["events"]`
leest ziet dus hooguit de eerste 200 van de nacht:

- `pipeline.py` koppelt PLM aan arousal voor `plm_arousal_index`, dat in het
  klinische PDF-rapport staat;
- YASAFlaskified schrijft de events in de EDF+-export
  (`generate_edfplus.py:155`), dus in een viewer stopt de markering ergens
  midden in de nacht.

De afkapping zelf blijft hier staan -- ze weghalen is een gedragswijziging.
Wat hier wordt vastgelegd is dat ze niet stil mag zijn. Vergelijk de regel
die het project elders hanteert: een grens die dekking beperkt, wordt
gerapporteerd.
"""
import numpy as np

from psgscoring.plm import analyze_plm

SF = 128.0
N_MOVES = 260
GAP_S = 20.0
FIRST_S = 60.0


def _emg_with_many_movements(seed=4):
    dur_s = FIRST_S + N_MOVES * GAP_S + 120.0
    n = int(dur_s * SF)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    for k in range(N_MOVES):
        t = FIRST_S + k * GAP_S
        s, e = int(t * SF), int((t + 1.5) * SF)
        x[s:e] += rng.normal(0.0, 60.0, e - s)
    hypno = ["N2"] * int(dur_s / 30)
    return x, hypno


def test_the_cap_is_recorded_in_the_summary():
    """Hoeveel er is weggelaten, hoort uit de samenvatting af te lezen."""
    emg, hypno = _emg_with_many_movements()
    out = analyze_plm(emg, None, SF, hypno, leg_unit="uV")
    assert out["success"], out.get("error")
    s = out["summary"]
    assert s["n_plm_eligible"] > 200, (
        f"fixture levert maar {s['n_plm_eligible']} geschikte bewegingen -- "
        "te weinig om de afkapping te raken"
    )
    assert len(out["events"]) == 200, "afkapping zelf blijft ongewijzigd"
    assert "n_events_truncated" in s, (
        "de afkapping laat geen spoor na in de samenvatting; wie "
        "output['plm']['events'] leest kan niet zien dat de nacht ophoudt"
    )
    assert s["n_events_truncated"] == s["n_plm_eligible"] - 200


def test_no_truncation_marker_when_nothing_is_dropped():
    """Bij een rustige nacht hoort het getal 0 te zijn, niet afwezig."""
    n = int(1800 * SF)
    rng = np.random.default_rng(9)
    x = rng.normal(0.0, 1.0, n)
    for k in range(5):
        t = 60.0 + k * 120.0
        s, e = int(t * SF), int((t + 1.5) * SF)
        x[s:e] += rng.normal(0.0, 60.0, e - s)
    out = analyze_plm(x, None, SF, ["N2"] * int(1800 / 30), leg_unit="uV")
    assert out["success"], out.get("error")
    assert out["summary"].get("n_events_truncated") == 0
