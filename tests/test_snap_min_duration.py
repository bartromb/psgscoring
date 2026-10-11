"""Snappen naar ademteuggrenzen mag een event niet onder zijn minimumduur
trekken.

`snap_events_to_breaths` (blok 1B, opt-in via `event_boundaries="breath"`)
verschuift beide grenzen naar de dichtstbijzijnde ademteugrand. Ligt de
onset net vóór een rand en de offset net ná een rand, dan schuiven de grenzen
naar elkaar toe: een hypopneu van 10,5 s wordt er een van 8,0 s. De
>=10 s-eis is dan al getoetst -- op de envelope-duur -- en het event blijft
staan met een duur die hij volgens de regel niet mag hebben.

Reparatie: komt het gesnapte event onder de minimumduur terwijl het
envelope-event die haalde, dan blijven de envelope-grenzen staan en telt
`n_kept_min_duration` het. Opt-in pad; geen profielveld.
"""
from psgscoring.constants import APNEA_MIN_DUR_S, HYPOPNEA_MIN_DUR_S
from psgscoring.respiratory import snap_events_to_breaths


def _breaths(n=200, duur=4.0):
    return [{"onset_s": i * duur, "duration_s": duur, "amplitude": 1.0}
            for i in range(n)]


def _ev(onset, duur, type_="hypopnea"):
    return {"type": type_, "onset_s": onset, "duration_s": duur,
            "stage": "N2", "epoch": int(onset // 30)}


def test_een_event_dat_onder_de_minimumduur_zou_zakken_houdt_zijn_envelope_grenzen():
    """39,0 -> 49,5 (10,5 s). Dichtstbijzijnde randen: 40 en 48 -> 8 s."""
    uit, st = snap_events_to_breaths(
        [_ev(39.0, 10.5)], _breaths(),
        hypopnea_min_duration_s=HYPOPNEA_MIN_DUR_S,
        apnea_min_duration_s=APNEA_MIN_DUR_S)
    assert uit[0]["onset_s"] == 39.0 and uit[0]["duration_s"] == 10.5, uit[0]
    assert st["n_kept_min_duration"] == 1
    assert st["n_snapped"] == 0
    assert uit[0].get("classify_detail", {}).get("boundaries") != "breath"


def test_zonder_minimumduur_blijft_het_oude_gedrag():
    """Aanroepers die geen minimum meegeven krijgen wat ze kregen."""
    uit, st = snap_events_to_breaths([_ev(39.0, 10.5)], _breaths())
    assert uit[0]["onset_s"] == 40.0 and uit[0]["duration_s"] == 8.0
    assert st["n_snapped"] == 1
    assert st.get("n_kept_min_duration", 0) == 0


def test_een_event_dat_boven_het_minimum_blijft_snapt_gewoon():
    """38,6 -> 60,0 (21,4 s) -> 40,0 -> 60,0 (20 s): ruim boven 10 s."""
    uit, st = snap_events_to_breaths(
        [_ev(38.6, 21.4)], _breaths(),
        hypopnea_min_duration_s=HYPOPNEA_MIN_DUR_S,
        apnea_min_duration_s=APNEA_MIN_DUR_S)
    assert uit[0]["onset_s"] == 40.0 and uit[0]["duration_s"] == 20.0
    assert st["n_snapped"] == 1 and st["n_kept_min_duration"] == 0


def test_apneus_toetsen_tegen_hun_eigen_minimum():
    uit, st = snap_events_to_breaths(
        [_ev(39.0, 10.5, type_="obstructive")], _breaths(),
        hypopnea_min_duration_s=HYPOPNEA_MIN_DUR_S,
        apnea_min_duration_s=APNEA_MIN_DUR_S)
    assert uit[0]["duration_s"] == 10.5
    assert st["n_kept_min_duration"] == 1
