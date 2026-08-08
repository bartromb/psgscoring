"""Gepubliceerde indices dragen één decimaal.

Deze toets bestaat door een concrete fout. In 0.14.7 werd twaalf keer
``max(uren, 0.001)`` vervangen door de gedeelde helper ``per_hour``. Bij die
vervanging kregen negen aanroepen in ``arousal.py`` en twee in ``spo2.py`` een
expliciete ``ndigits`` mee die niet overeenkwam met wat de oude helpers deden:
``_safe`` en ``safe_r`` rondden allebei af op **één** decimaal.

Het gevolg stond maanden later in een klinisch rapport::

    Arousal index (AI)          57.906 /u      naast   Respiratoire  16.8 /u
    ODI 3%                      24.05  /u      was     24.0 /u

Waarom de bestaande toetsen dit niet zagen: de golden-harnas vergelijkt een
digest die zélf op één decimaal afrondt (``_r(x, nd=1)`` in
``test_golden_output.py``). Een verandering ván 1 decimaal náár 2 of 3 is
daarin per constructie onzichtbaar — de digest gooit precies het verschil weg.
De golden bewaakt de wáárde; deze toets bewaakt de weergave.

Waarom één decimaal het juiste antwoord is: een arousal-index is niet tot op
0,001/u te kennen. Waar de grens van een arousal ligt is een scorersoordeel met
een spreiding van seconden; drie decimalen suggereren een nauwkeurigheid die de
meting niet heeft, en dat is in een klinisch document misleidend.
"""

import re
from pathlib import Path

import pytest

from psgscoring.arousal import _recompute_arousal_summary
from psgscoring.indices import per_hour

PKG = Path(__file__).resolve().parent.parent / "psgscoring"


def _decimals(x: float) -> int:
    """Aantal decimalen in de decimale weergave van een afgerond getal."""
    s = repr(float(x))
    return len(s.split(".")[1].rstrip("0")) if "." in s and not s.endswith(".0") else 0


# ──────────────────────────────────────────────────────────────
#  De helper zelf
# ──────────────────────────────────────────────────────────────

def test_per_hour_defaults_to_one_decimal():
    assert per_hour(57, 0.984) == 57.9
    assert _decimals(per_hour(57, 0.984)) <= 1


def test_the_default_is_what_the_old_helpers_did():
    """`_safe(val, dec=1)` en `safe_r(val, dec=1)` — beide één decimaal."""
    from psgscoring.utils import safe_r
    n, h = 213, 3.6789
    assert per_hour(n, h) == safe_r(n / h)


# ──────────────────────────────────────────────────────────────
#  De arousal-index, het geval dat in het rapport stond
# ──────────────────────────────────────────────────────────────

def _hypno(n_sleep_epochs: int) -> list:
    return ["N2"] * n_sleep_epochs


def test_the_arousal_index_carries_one_decimal():
    """118 arousals over 122 min slaap gaf ooit 57.906/u."""
    arousals = [{"stage": "N2", "duration_s": 5.0, "dominant_band": "theta"}
                for _ in range(118)]
    s = _recompute_arousal_summary(arousals, _hypno(244), set())
    assert _decimals(s["arousal_index"]) <= 1, s["arousal_index"]


@pytest.mark.parametrize("key", ["arousal_index", "nrem_arousal_index",
                                 "rem_arousal_index"])
def test_every_arousal_index_carries_one_decimal(key):
    arousals = ([{"stage": "N2", "duration_s": 5.0, "dominant_band": "theta"}] * 37
                + [{"stage": "R", "duration_s": 5.0, "dominant_band": "alpha"}] * 11)
    s = _recompute_arousal_summary(arousals, ["N2"] * 300 + ["R"] * 91, set())
    assert s[key] is not None
    assert _decimals(s[key]) <= 1, f"{key}={s[key]}"


def test_an_index_without_sleep_is_none_not_a_number():
    """De reparatie van 0.14.7 zelf mag niet sneuvelen: geen ondergrens op de
    noemer, dus geen index maal duizend."""
    s = _recompute_arousal_summary(
        [{"stage": "N2", "duration_s": 5.0, "dominant_band": "theta"}], [], set())
    assert s["arousal_index"] is None


# ──────────────────────────────────────────────────────────────
#  De regel, niet alleen de gevallen
# ──────────────────────────────────────────────────────────────

_CALL = re.compile(r"per_hour\(\s*[^()]*?(?:\([^()]*\))?[^()]*?,\s*[^,()]+,\s*(\d+)\s*\)")


def test_no_call_site_overrides_the_rounding():
    """Elke afwijking hier is een presentatiebeslissing en hoort zichtbaar te
    zijn in de review, niet als derde argument in een regel over noemers."""
    offenders = []
    for path in sorted(PKG.glob("*.py")):
        if path.name == "indices.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "per_hour(" in line and re.search(r",\s*\d+\s*\)", line):
                if _CALL.search(line):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, (
        "per_hour krijgt een expliciete ndigits mee:\n  " + "\n  ".join(offenders))
