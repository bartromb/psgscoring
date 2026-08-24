"""Dezelfde ruwe waarde hoort altijd dezelfde houding te zijn.

`_map_position_signal` quantiseerde een ruw ADC-signaal op **percentielen van
de nachtverdeling**: `np.percentile(valid, [0,20,40,60,80,100])`. De bingrenzen
schuiven dus mee met hoe lang de patiënt in elke houding lag. Vier vaste
plateauwaarden kregen daardoor drie verschillende labelsets:

    gelijk verdeeld   120→Left   400→Supine  680→Right    950→Upright
    veel op de rug    120→Prone  400→Prone   680→Upright  950→Upright
    veel op links     120→Prone  400→Upright 680→Upright  950→Upright

Twee dingen gaan daar mis. Het label van een waarde hangt af van de DUUR van
andere houdingen — twee nachten van dezelfde recorder krijgen andere labels —
en twee onderscheiden houdingen vallen samen op één code, waarna hun events
en minuten op één hoop komen.

De positie-AHI, het POSA-fenotype en de therapieaanbeveling hangen hieraan.
"""
import numpy as np

from psgscoring.ancillary import _map_position_signal

SF = 4.0
NIVEAUS = [120.0, 400.0, 680.0, 950.0, 1200.0]


def _signaal(minuten, waarden=NIVEAUS):
    return np.concatenate([np.full(int(m * 60 * SF), float(v))
                           for m, v in zip(minuten, waarden)])


def _codes_per_niveau(minuten, waarden=NIVEAUS):
    mapped, _method = _map_position_signal(_signaal(minuten, waarden))
    uit, i = {}, 0
    for m, v in zip(minuten, waarden):
        n = int(m * 60 * SF)
        uit[v] = int(np.bincount(mapped[i:i + n]).argmax())
        i += n
    return uit


def test_the_labels_do_not_depend_on_how_long_the_patient_lay_there():
    gelijk = _codes_per_niveau([80, 80, 80, 80, 80])
    ruglig = _codes_per_niveau([20, 20, 340, 20, 20])
    links  = _codes_per_niveau([20, 340, 20, 20, 20])
    assert gelijk == ruglig == links, (
        f"labels verschuiven met de duurverdeling:\n  gelijk {gelijk}\n"
        f"  ruglig {ruglig}\n  links  {links}")


def test_distinct_positions_never_collapse_onto_one_code():
    codes = _codes_per_niveau([20, 20, 340, 20, 20])
    assert len(set(codes.values())) == len(NIVEAUS), (
        f"houdingen samengevallen: {codes}")


def test_the_order_of_the_raw_values_is_preserved():
    codes = _codes_per_niveau([80, 80, 80, 80, 80])
    waarden = sorted(codes)
    assert [codes[v] for v in waarden] == sorted(codes[v] for v in waarden)


def test_a_precoded_signal_is_still_used_as_is():
    ruw = np.repeat(np.array([0, 1, 2, 3, 4], dtype=float), 100)
    mapped, method = _map_position_signal(ruw)
    np.testing.assert_array_equal(mapped, ruw.astype(int))
    assert method == "coded"


def test_scale_and_offset_do_not_change_anything():
    basis = _signaal([80] * 5)
    a, _ = _map_position_signal(basis)
    b, _ = _map_position_signal(basis * 2.0 + 50.0)
    np.testing.assert_array_equal(a, b)


def test_the_method_is_reported():
    """Zonder deze aantekening is achteraf niet te zien of een positielabel uit
    de recordercodering komt of uit een gok op de rangorde."""
    _m, method = _map_position_signal(_signaal([80] * 5))
    assert method in ("coded", "levels", "percentile")
    assert method == "levels", method


def test_a_continuous_signal_falls_back_and_says_so():
    rng = np.random.default_rng(2)
    _m, method = _map_position_signal(rng.normal(0, 1, 20000))
    assert method == "percentile"


def test_fewer_positions_than_codes_is_not_padded_out():
    """Een nacht met drie houdingen mag er geen vijf verzinnen."""
    codes = _codes_per_niveau([100, 100, 100], waarden=[120.0, 680.0, 1200.0])
    assert len(set(codes.values())) == 3, codes


def test_the_analysis_publishes_the_method():
    from psgscoring.ancillary import analyze_position
    sig = _signaal([80] * 5)
    hypno = ["N2"] * int(len(sig) / SF / 30)
    out = analyze_position(sig, SF, hypno, [])
    assert out["summary"]["position_mapping_method"] == "levels"
