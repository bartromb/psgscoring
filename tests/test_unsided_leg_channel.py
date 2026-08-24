"""Eén ongezijderd beenkanaal is nog steeds een beenkanaal.

MESA-EDF's dragen één kaal kanaal `Leg`. `CHANNEL_PATTERNS` kent alleen
`leg_l` en `leg_r`, met patronen die allemaal een zijde noemen ("leg l",
"lleg", "emg tib l", ...). Een kanaal dat `Leg` heet matcht geen van beide,
dus `detect_channels` levert niets, `_pick` geeft None, en de pijplijn neemt
de tak `"No leg-EMG channels"`. Geen fout, geen waarschuwing, lege
samenvatting — de PLM-stap draait op MESA **helemaal niet**.

Daardoor is elke PLM-wijziging op MESA onmeetbaar: `plm_offset_aasm` moest op
een expliciete `channel_map` gemeten worden en de gecombineerde-impactvraag
kon niet beantwoord worden.

Waarom er een DERDE rol bijkomt in plaats van `leg_l` te verruimen: een kaal
"leg" in `leg_l` maakt van een ongezijderd kanaal het linkerbeen, en dat is
niet waar. Het verschil is klinisch: `_merge_bilateral` ontdubbelt bewegingen
die beide benen zien, en die regel kan niet toegepast worden op één kanaal.
Een LM-telling uit één kanaal is dus niet zonder meer vergelijkbaar met een
telling uit twee, en dat hoort in de provenance te staan in plaats van
verzwegen te worden.
"""
import numpy as np
import pytest

from psgscoring.utils import detect_channels

# ══════════════════════════════════════════════════════════════
# Kanaaldetectie
# ══════════════════════════════════════════════════════════════

def test_the_bare_mesa_leg_channel_is_recognised():
    ch = detect_channels(["EKG", "EOG-L", "EMG", "EEG1", "Pres", "Thor",
                          "Abdo", "Leg", "Therm", "Pos", "SpO2"])
    assert ch.get("leg") == "Leg", ch
    assert ch.get("leg_l") is None
    assert ch.get("leg_r") is None


@pytest.mark.parametrize("naam", ["Leg", "EMG Leg", "Tibialis", "Tib",
                                  "PLM", "Beenbeweging"])
def test_the_usual_unsided_labels_are_recognised(naam):
    assert detect_channels(["EEG1", naam, "Pres"]).get("leg") == naam


def test_a_sided_montage_is_untouched():
    """De bestaande rollen mogen niet verschuiven, en de nieuwe rol mag hun
    kanaal niet inpikken."""
    ch = detect_channels(["C4:A1", "Pressure Flow", "PLMl", "PLMr", "SpO2"])
    assert ch.get("leg_l") == "PLMl"
    assert ch.get("leg_r") == "PLMr"
    assert ch.get("leg") is None, (
        f"de ongezijderde rol pikt een gezijderd kanaal in: {ch.get('leg')}")


def test_one_sided_leg_channel_stays_where_it_belongs():
    ch = detect_channels(["C4:A1", "EMG Tib L", "SpO2"])
    assert ch.get("leg_l") == "EMG Tib L"
    assert ch.get("leg") is None


def test_the_chin_emg_is_still_not_a_leg_and_the_leg_not_a_chin():
    ch = detect_channels(["C4:A1", "Chin EMG", "Leg", "SpO2"])
    assert ch.get("emg") == "Chin EMG"
    assert ch.get("leg") == "Leg"


# ══════════════════════════════════════════════════════════════
# De pijplijn draait er ook werkelijk op
# ══════════════════════════════════════════════════════════════

def _raw(mne, beenkanalen: list[str]):
    sf, minuten = 64.0, 30
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(7)
    namen = ["Pres", "SpO2", "EEG1"] + beenkanalen
    data = [np.sin(2 * np.pi * 0.25 * t), np.full(n, 97.0),
            rng.normal(0, 20e-6, n)]
    for _ in beenkanalen:
        been = rng.normal(0, 2e-6, n)
        # periodieke beenbewegingen: 1 s burst elke 25 s
        for start in range(60, 60 * minuten - 60, 25):
            a, b = int(start * sf), int((start + 1.0) * sf)
            been[a:b] += rng.normal(0, 60e-6, b - a)
        data.append(been)
    info = mne.create_info(namen, sf, ["misc"] * 2 + ["eeg"] + ["emg"] * len(beenkanalen))
    raw = mne.io.RawArray(np.vstack(data), info, verbose=False)
    return raw, ["N2"] * int(np.ceil(raw.times[-1] / 30.0))


def _run(beenkanalen):
    mne = pytest.importorskip("mne")
    import psgscoring
    raw, hypno = _raw(mne, beenkanalen)
    return psgscoring.run_pneumo_analysis(
        raw, hypno=hypno, scoring_profile="aasm_v3_breath")


def test_the_plm_step_runs_on_a_single_unsided_channel():
    out = _run(["Leg"])
    plm = out["plm"]
    assert plm.get("error") != "No leg-EMG channels", (
        "de PLM-stap slaat een aanwezig beenkanaal over")
    assert plm.get("success") is True, plm.get("error")
    assert (plm.get("summary") or {}).get("n_lm_total", 0) > 0, plm["summary"]


def test_the_provenance_names_the_leg_channels_it_used():
    meta = _run(["Leg"])["meta"]["plm_channels"]
    assert meta["unsided"] == "Leg"
    assert meta["leg_l"] is None and meta["leg_r"] is None
    assert meta["bilateral"] is False, (
        "een telling uit één kanaal is niet zonder meer vergelijkbaar met een "
        "telling uit twee -- _merge_bilateral kan hier niet draaien")


def test_a_bilateral_montage_says_so():
    meta = _run(["PLMl", "PLMr"])["meta"]["plm_channels"]
    assert meta["leg_l"] == "PLMl" and meta["leg_r"] == "PLMr"
    assert meta["unsided"] is None
    assert meta["bilateral"] is True


def test_no_leg_channel_at_all_still_reports_that():
    out = _run([])
    assert out["plm"].get("error") == "No leg-EMG channels"
    assert out["meta"]["plm_channels"]["bilateral"] is False
