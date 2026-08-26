"""Vaste verschuiving van de arousal-onsets (`arousal_onset_offset_s`).

WAT DE VLAG DOET EN WAAROM HIJ BESTAAT
--------------------------------------
De kandidaatdekking op PSG-IPA piekt op vier van de vijf opnames bij +2 s in
plaats van 0. Daarop is +2 s vooraf vastgelegd en op event-F1 gemeten:

    PSG-IPA (12 scoorders, 3 afleidingen, n=5) : +0,0123, beter op 5/5
    MESA    ( 1 scoorder,  1 afleiding,  n=30) : +0,0140, beter op 22/30,
                                                 tekentoets p = 0,00053

Beide reeksen zijn glad en eentoppig met hun maximum op +2 s. Klinische prijs
op MESA n=30 gepaard: AHI-ernstklasse 0/30, RDI-ernstklasse 0/30.

WAAR DE TESTEN OP LETTEN
------------------------
Niet "schuift er iets op" -- dat haalt elke implementatie. Wel: schuift hij
VOOR de koppeling en de RERA-stap. Een implementatie die de onsets pas na
afloop opschuift, verandert alleen wat er GERAPPORTEERD wordt en niet wat de
AHI en de RDI voedt, en ziet er in een naïeve test precies hetzelfde uit.
`test_verschuiving_gebeurt_voor_de_koppeling` valt om op zo'n implementatie.
"""
import numpy as np
import pytest

from psgscoring.arousal import run_arousal_respiratory_analysis
from psgscoring.constants import SCORING_PROFILES

# Het koppelvenster is [event_einde - 5 s, event_einde + 15 s]. Een event dat
# op t=100 eindigt en een arousal op t=114 geven latentie 14 s: respiratoir.
# Twee seconden later is de latentie 16 s en valt hij erbuiten. Dat randgeval
# is met opzet zo gekozen: het onderscheidt "voor" van "na" de koppeling.
RESP = [{"onset_s": 80.0, "duration_s": 20.0, "type": "hypopnea"}]
AROUSAL = {"onset_s": 114.0, "end_s": 117.0, "duration_s": 3.0,
           "stage": "N2", "dominant_band": "alpha"}


def _draai(monkeypatch, offset):
    """Detectie vervangen door één vast event, zodat alleen de vlag varieert."""
    import psgscoring.arousal as A

    def _vast(*_a, **_k):
        return {"success": True, "events": [dict(AROUSAL)],
                "summary": {"n_arousals": 1}}

    monkeypatch.setattr(A, "detect_arousals", _vast)
    return run_arousal_respiratory_analysis(
        eeg_data=np.zeros(256 * 200, dtype=float), sf_eeg=256.0,
        flow_data=None, flow_norm=None, sf_flow=None,
        resp_events=RESP, hypno=["N2"] * 7,
        onset_offset_s=offset,
    )


def test_verschuiving_gebeurt_voor_de_koppeling(monkeypatch):
    """De koppeling moet de VERSCHOVEN onset zien, niet de oorspronkelijke.

    Dit is de test die de ontwerpkeuze vastlegt. Schuift de implementatie pas
    na `correlate_arousals_to_respiratory`, dan blijft de arousal in beide
    armen respiratoir en valt deze test om.
    """
    zonder = _draai(monkeypatch, 0.0)
    assert zonder["summary"]["n_respiratory_arousals"] == 1, (
        "opzet klopt niet: zonder verschuiving hoort de arousal binnen het "
        "venster van 15 s te vallen")

    met = _draai(monkeypatch, 2.0)
    assert met["summary"]["n_respiratory_arousals"] == 0, (
        "de koppeling zag de oorspronkelijke onset, niet de verschoven — de "
        "verschuiving grijpt te laat aan en raakt AHI noch RDI")
    assert met["summary"]["n_spontaneous_arousals"] == 1


def test_verschuift_precies_een_keer(monkeypatch):
    """Onset en einde schuiven met exact de offset, niet met een veelvoud.

    `detect_arousals_multi` roept intern `detect_arousals` aan; een
    verschuiving op beide niveaus zou +4 opleveren in plaats van +2.
    """
    ev = _draai(monkeypatch, 2.0)["arousals"]["events"][0]
    assert ev["onset_s"] == pytest.approx(116.0)
    assert ev["end_s"] == pytest.approx(119.0)
    assert ev["duration_s"] == pytest.approx(3.0), "de duur mag niet meebewegen"


def test_negatieve_offset_schuift_terug(monkeypatch):
    ev = _draai(monkeypatch, -1.5)["arousals"]["events"][0]
    assert ev["onset_s"] == pytest.approx(112.5)


def test_nul_laat_alles_ongemoeid(monkeypatch):
    ev = _draai(monkeypatch, 0.0)["arousals"]["events"][0]
    assert ev["onset_s"] == pytest.approx(114.0)
    assert ev["end_s"] == pytest.approx(117.0)


def test_offset_staat_in_de_provenance(monkeypatch):
    """Een verschuiving die nergens gemeld wordt, is niet te herleiden."""
    assert _draai(monkeypatch, 2.0)["arousals"]["summary"]["onset_offset_s"] == 2.0
    assert "onset_offset_s" not in _draai(monkeypatch, 0.0)["arousals"]["summary"]


def test_alle_profielen_staan_default_uit():
    """Geen enkel profiel mag de verschuiving aan hebben zonder beslissing.

    De gerapporteerde arousal-onsets in het klinische rapport schuiven mee,
    ook al blijft de index onveranderd; dat maakt aanzetten een aparte keuze.
    """
    # EERST: het veld moet er zijn. Alleen op waarheid toetsen met .get()
    # slaagt ook als de vlag helemaal niet in de registry zit -- dan meet de
    # test niets en blijft hij groen terwijl de vlag onbereikbaar is.
    ontbreekt = [n for n, p in SCORING_PROFILES.items()
                 if "AROUSAL_ONSET_OFFSET_S" not in p]
    assert not ontbreekt, (
        f"vlag niet in de registry voor: {ontbreekt}")

    aan = {n: p["AROUSAL_ONSET_OFFSET_S"] for n, p in SCORING_PROFILES.items()
           if p["AROUSAL_ONSET_OFFSET_S"]}
    assert aan == {}, f"profielen met een verschuiving aan: {aan}"


def test_env_override_en_terugval(monkeypatch):
    """De env-override bestaat om beide armen te meten zonder registrymutatie."""
    from psgscoring.pipeline import _arousal_onset_offset_s
    prof = {"AROUSAL_ONSET_OFFSET_S": 0.0}

    assert _arousal_onset_offset_s(prof) == 0.0

    monkeypatch.setenv("PSGSCORING_AROUSAL_ONSET_OFFSET_S", "2")
    assert _arousal_onset_offset_s(prof) == 2.0

    # Onleesbaar: profielwaarde aanhouden en melden. Stilletjes het verkeerde
    # getal draaien is erger dan de default aanhouden.
    monkeypatch.setenv("PSGSCORING_AROUSAL_ONSET_OFFSET_S", "twee")
    assert _arousal_onset_offset_s({"AROUSAL_ONSET_OFFSET_S": 1.0}) == 1.0
