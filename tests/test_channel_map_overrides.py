"""Een handmatige kanaalkeuze die niet bestaat werd stil genegeerd.

`channel_map_from_user` voegt de automatische detectie samen met de keuzes
van de gebruiker. Een override naar een kanaalnaam die niet in het bestand
voorkomt -- een typefout, een andere montage dan gedacht -- viel weg zonder
melding, en de analyse draaide op het automatisch gekozen kanaal alsof de
gebruiker dat zo had gewild. In de uitvoer is dat niet te zien: de map
bevat gewoon een geldige naam.

Nu wordt het gemeld (WARNING) en geregistreerd op de teruggegeven map
(`.ignored`), zodat de pijplijn het in `meta` kan zetten.
"""
import json
import logging

from psgscoring.utils import channel_map_from_user, detect_channels

NAMEN = ["Flow Patient", "Thorax", "Abdomen", "SpO2", "EEG C4-M1", "Chin EMG"]


def test_een_ongeldige_override_wordt_gemeld_en_geregistreerd(caplog):
    with caplog.at_level(logging.WARNING, logger="psgscoring.utils"):
        m = channel_map_from_user({"flow": "Kanaal Dat Niet Bestaat"}, NAMEN)
    # Terugval op de automatische keuze, zoals voorheen ...
    assert m["flow"] == detect_channels(NAMEN)["flow"]
    # ... maar nu met een spoor.
    assert m.ignored == [{
        "role": "flow",
        "channel": "Kanaal Dat Niet Bestaat",
        "fallback": detect_channels(NAMEN)["flow"],
        "reason": "kanaal staat niet in het bestand",
    }], m.ignored
    assert any("Kanaal Dat Niet Bestaat" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_een_geldige_override_wint_en_laat_geen_spoor(caplog):
    with caplog.at_level(logging.WARNING, logger="psgscoring.utils"):
        m = channel_map_from_user({"flow": "Thorax"}, NAMEN)
    assert m["flow"] == "Thorax"
    assert m.ignored == []
    assert not caplog.records


def test_een_lege_override_is_geen_override():
    """`None` of "" betekent "geen keuze", niet "verkeerde keuze"."""
    m = channel_map_from_user({"flow": None, "thorax": ""}, NAMEN)
    assert m["flow"] == detect_channels(NAMEN)["flow"]
    assert m.ignored == []


def test_een_ongeldige_override_voor_een_rol_zonder_autodetectie():
    """Dan is er geen terugval; de rol ontbreekt gewoon, en dat staat erbij."""
    m = channel_map_from_user({"leg_l": "Been Links"}, NAMEN)
    assert "leg_l" not in m
    assert m.ignored[0]["role"] == "leg_l"
    assert m.ignored[0]["fallback"] is None


def test_de_map_blijft_een_gewone_dict():
    """De pijplijn leest hem als dict en serialiseert hem naar meta."""
    m = channel_map_from_user({"flow": "Nee"}, NAMEN)
    assert isinstance(m, dict)
    assert json.loads(json.dumps(m)) == dict(m)
    assert channel_map_from_user(None, NAMEN).ignored == []
