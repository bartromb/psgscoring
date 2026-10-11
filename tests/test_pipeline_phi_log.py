"""De EDF-kop draagt naam en geboortedatum; het log mag die nooit dragen.

`_parse_edf_patient_info` las de 80-byte patiëntregel van de EDF-kop en
schreef naam, geslacht en geboortedatum met `logger.info` weg. Logs reizen
verder dan de uitslag: ze staan in docker-logs, in de RQ-worker-uitvoer en in
elke bugmelding die iemand plakt. Het handboek zegt het al: PHI nooit in
git/PyPI/logs/chat.

Wat het log WEL mag zeggen: óf er een naam en een geboortedatum in de kop
stonden (dat is een montage-eigenschap, nuttig om een anonieme export te
herkennen) en welk apparaat de opname maakte.

De kop hieronder is het voorbeeld uit de EDF+-specificatie zelf
("Haagse_Harry"), geen echte patiënt.
"""
import logging
from types import SimpleNamespace

from psgscoring.pipeline import _parse_edf_patient_info

PATIENT = "MCH-0234567 F 02-MAY-1951 Haagse_Harry"
RECORDING = "Startdate 02-MAR-2002 PSG-1234/2002 NN Telemetry03"


def _edf_header(path):
    """Alleen de vaste 256-byte kop; de parser leest 168 bytes ervan."""
    hdr = (b"0".ljust(8) + PATIENT.encode().ljust(80)
           + RECORDING.encode().ljust(80) + b"02.03.02" + b"22.00.00"
           + b"256".ljust(8) + b"".ljust(44) + b"0".ljust(8)
           + b"1".ljust(8) + b"0".ljust(4))
    assert len(hdr) == 256
    path.write_bytes(hdr)
    return SimpleNamespace(filenames=(str(path),))


def test_de_parser_leest_de_kop_nog_steeds(tmp_path):
    """De reparatie zit in het log, niet in de uitvoer: het rapport heeft de
    velden nodig (patient_info in meta)."""
    info = _parse_edf_patient_info(_edf_header(tmp_path / "k.edf"))
    assert info["name"] == "Haagse Harry"
    assert info["birthday_str"] == "02-MAY-1951"
    assert info["sex"] == "F"
    assert info["equipment"] == "Telemetry03"


def test_naam_en_geboortedatum_bereiken_het_log_niet(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger="psgscoring.pipeline"):
        _parse_edf_patient_info(_edf_header(tmp_path / "k.edf"))
    tekst = caplog.text
    for geheim in ("Haagse", "Harry", "02-MAY-1951", "1951", "MCH-0234567"):
        assert geheim not in tekst, f"{geheim!r} staat in het log: {tekst}"


def test_het_log_zegt_wel_dat_er_een_naam_en_datum_in_de_kop_stonden(tmp_path, caplog):
    """Aanwezigheid is geen PHI en wel informatie: een export zonder naam is
    anders, en dat hoort zichtbaar te zijn."""
    with caplog.at_level(logging.INFO, logger="psgscoring.pipeline"):
        _parse_edf_patient_info(_edf_header(tmp_path / "k.edf"))
    tekst = caplog.text
    assert "name_present=True" in tekst, tekst
    assert "dob_present=True" in tekst, tekst
    assert "Telemetry03" in tekst, "het apparaat mag wel genoemd worden"
