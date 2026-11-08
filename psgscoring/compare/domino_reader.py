"""
psgscoring.compare.domino_reader
================================
Lees de analyse-export van SOMNOmedics DOMINO ("File → Export analysis
data", als ASCII of als Excel) en vertaal die naar het event- en
hypnogramschema van psgscoring, zodat een menselijke scoring uit de kliniek
per studie naast de machine gelegd kan worden.

Deze module is een LEZER. Zij zit niet in het scoringspad, importeert
niets van SOMNOmedics en levert niets van hen mee. Zij leest tekst- of
Excel-bestanden die DOMINO zelf heeft weggeschreven.

WAT DE EXPORT IS
----------------
Eén bestand (of één Excel-blad) per signaal, 45 in de kliniekexport. Elk
bestand draagt een kopblok en daarna rijen::

    Signal ID: FlowD\\flow
    Start Time: 01-01-2026 22:00:00
    Unit: s
    Signal Type: Impuls

    22:00:14,246-22:00:29,121; 15;Hypopnea

Drie signaaltypen:

``Impuls``   eventlijst: ``begin-einde; waarde;label``. Respiratoire events
             (``Flow Events``), desaturaties (``SpO2 Events``), beenbewegingen
             (``PLM Events``), arousals (``Classification Arousals``), snurken,
             blinks, …
``Discret``  toestand per tijdstip: het slaapprofiel (``Sleep profile``,
             ``Rate: 30 s``, waarden ``Wake/N1/N2/N3/REM/A``) en de
             lichaamspositie (alleen wissels, geen rate).
``Analog``   reeks: SpO2, hartfrequentie, … ASCII schrijft een rij per
             monster of per verandering (SpO2 ≈ 4 300 rijen per nacht,
             hartfrequentie 1 Hz); een waarde geldt tot de volgende rij,
             ``A`` is een artefactrij. De Excel-export vlakt alles af tot
             30 s en laat de laatste epoch weg.

De ``Markers``-export (``Signal ID: User markers``) draagt ``Start``,
``Lights Off`` en ``End``. Decimale komma, CRLF, cp1252 (``µV`` is één
byte). Geen patiëntgegevens in koppen of bestandsnamen, wel de opnamedatum
in ``Start Time``: log die niet.

De Excel-vorm heeft hetzelfde kopblok in rijen 1–6, een kolomkop in rij 8
en per event een extra kolom ``Sleep stage`` (het DOMINO-stadium van de
epoch waarin het event begint — hier opnieuw afgeleid uit het profiel, zodat
ASCII en Excel hetzelfde geven). Alle ``Start Time``-cellen staan in Excel op
het 30-s-raster; de ASCII-eventbestanden dragen de echte start. Het
referentiepunt is daarom de marker ``Start``, niet een kopregel.

DRIE DINGEN DIE JE MOET GELIJKTREKKEN VÓÓR JE VERGELIJKT
--------------------------------------------------------
1. DE TIJDBASIS. Alle tijden zijn kloktijden; de lezer rekent ze om naar
   seconden t.o.v. een nulpunt ``t0`` (standaard de marker ``Start``, die op
   de gecontroleerde kliniekexport van 04-11-2026 tot op 1 s gelijk is aan de
   EDF-kop). Geef ``t0`` expliciet mee als de EDF ergens anders begint, en
   controleer het met ``estimate_clock_offset`` op de SpO2-reeks: dat is
   goedkoop en vangt een verkeerd nulpunt op vóór er een F1 uit rolt.
2. HET RASTER. Het DOMINO-slaapprofiel ligt op :00/:30 van de klok, de
   machine-epochs beginnen bij de EDF-start. ``hypnogram_epochs`` herbemonstert
   het profiel op het machine-raster (midden van elke epoch);
   ``hypnogram_1s`` geeft de seconde-reeks. Buiten het profiel staat ``"?"``,
   nooit een verzonnen stadium.
3. DE REGEL EN DE NOEMER. De eventlijst is de volledige lijst, óók events in
   wake en vóór lights-off; het DOMINO-rapport telt daar zelf nog een
   hypopneucriterium overheen (desaturatie binnen een venster), dus een
   rapport-AHI is niet "aantal events / TST" uit deze lijst. Filter met
   ``events_in_sleep`` en deel door dezelfde slaaptijd als aan de machinekant
   (``ahi_from_domino_events``), anders vergelijk je noemers.

LABELS
------
``DOMINO_RESP_LABELS`` vertaalt de DOMINO-eventnamen naar de typenamen die
psgscoring zelf gebruikt (``obstructive``, ``central``, ``mixed``,
``hypopnea``, …). Een label dat er niet in staat wordt NIET stil als
hypopneu meegeteld en NIET stil weggelaten: ``respiratory_events`` laat hem
weg en ``verify_label_mapping`` meldt hem. Kale ``Hypopnea`` (zonder
``Obstructive``) blijkt in de kliniekexport het onbewerkte auto-event te zijn
dat de scoorder niet heeft aangeraakt; het blijft gewoon ``hypopnea`` — of
het meetelt is een keuze van de vergelijking, niet van de lezer.
"""

from __future__ import annotations

import re
import zipfile
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "DOMINO_RESP_LABELS",
    "DOMINO_STAGE_LABELS",
    "DOMINO_LIMB_LABELS",
    "DominoRow",
    "DominoSignal",
    "DominoExport",
    "read_domino_export",
    "parse_domino_txt",
    "parse_domino_sheet",
    "NotDominoFile",
    "ahi_from_domino_events",
    "estimate_clock_offset",
    "hypnogram_kappa",
]

# DOMINO-stadium -> psgscoring-stadium. "A" is het artefact-epoch van DOMINO
# (in de Events list "Artefact"); N4 komt uit de R&K-lijst van oudere
# profielen en valt op N3.
DOMINO_STAGE_LABELS: dict[str, str] = {
    "Wake": "W", "W": "W",
    "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N3",
    "REM": "R", "R": "R",
    "A": "A", "Artefact": "A", "Artifact": "A",
}

# DOMINO-eventnaam -> psgscoring-typenaam. Bewust de namen die psgscoring
# zelf gebruikt ("obstructive", niet "obstructive_apnea"): validate_psgipa
# .type_family() werkt met een exacte woordenlijst en agreement
# .event_category() met de prefix "hypopnea". None = geen respiratoir event.
DOMINO_RESP_LABELS: dict[str, str | None] = {
    "Obstructive Apnea":    "obstructive",
    "Central Apnea":        "central",
    "Mixed Apnea":          "mixed",
    "Apnea":                "apnea",            # niet onderverdeeld
    "Obstructive Hypopnea": "hypopnea",
    "Central Hypopnea":     "hypopnea_central",
    "Mixed Hypopnea":       "hypopnea_mixed",
    "Hypopnea":             "hypopnea",         # onbewerkt auto-event
    "RERA":                 "rera",
    "Body event":           None,               # positie-/bewegingsmarkering
}

DOMINO_LIMB_LABELS: dict[str, str] = {
    "P-Marker":         "plm",          # periodieke beenbeweging (PLM-reeks)
    "I-Marker":         "lm",           # geïsoleerde beenbeweging
    "LM Resp":          "lm_resp",      # beenbeweging bij een resp. event
    "LM Body Position": "lm_position",  # beenbeweging bij een positiewissel
}

SLEEP_STAGES = frozenset({"N1", "N2", "N3", "R"})

# Bestandsnamen (zonder .txt) / bladnamen zoals DOMINO ze schrijft.
SIG_PROFILE = "Sleep profile"
SIG_FLOW = "Flow Events"
SIG_SPO2_EVENTS = "SpO2 Events"
SIG_PLM = "PLM Events"
SIG_AROUSAL = "Classification Arousals"
SIG_MARKERS = "Markers"
SIG_SPO2 = "SpO2"
SIG_HR = "Heart Rate"
SIG_POSITION = "Position"

_MIN_OVERLAP_S = 600

_TIME = r"(\d{1,2}):(\d{2}):(\d{2})(?:[,.](\d{1,6}))?"
_RE_IMPULS = re.compile(rf"^\s*{_TIME}\s*-\s*{_TIME}\s*;\s*(.*?)\s*$")
_RE_POINT = re.compile(rf"^\s*{_TIME}\s*;\s*(.*?)\s*$")
_RE_START = re.compile(r"(\d{1,2})[-./](\d{1,2})[-./](\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})")
_RE_RATE = re.compile(r"([\d.,]+)\s*(ms|s)?", re.I)
_RE_NUM = re.compile(r"^[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?$")
_RE_ROWSTART = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}")

# Een sprong TERUG in de kloktijd van meer dan dit is middernacht; kleiner is
# jitter (de bron is soms 1 s niet-monotoon) of het profiel dat op :30 vóór de
# start begint. Een sprong vooruit is nooit een nieuwe dag: een ijl
# eventbestand mag uren leeg zijn.
_MIDNIGHT_BACKSTEP_S = 3600.0


class NotDominoFile(ValueError):
    """Geen DOMINO-exportbestand (kopregel ``Signal ID`` ontbreekt). Wordt bij
    het lezen van een map of zip stil overgeslagen; elke andere leesfout niet."""


def _norm(name: str) -> str:
    """Sleutel voor het opzoeken van een signaal: 'Spindle & K', 'Spindle  K'
    en 'spindle_k.txt' zijn hetzelfde signaal."""
    name = re.sub(r"\.(txt|csv)$", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _tod(h, m, s, frac=None) -> float:
    ms = 0.0
    if frac:
        ms = float("0." + frac)
    return int(h) * 3600 + int(m) * 60 + int(s) + ms


def _num(text: str):
    """Decimale komma -> float; 'A' en andere tekst blijven tekst."""
    t = text.strip()
    if _RE_NUM.match(t):
        return float(t.replace(",", "."))
    return t


@dataclass(frozen=True)
class DominoRow:
    """Eén rij. Tijden in seconden sinds middernacht van de referentiedatum
    van de export, doorlopend over middernacht (kunnen > 86400 zijn)."""
    t_start: float
    t_end: float | None
    value: Any
    label: str | None


@dataclass
class DominoSignal:
    name: str
    signal_id: str
    start_time: datetime | None
    unit: str
    kind: str                       # "Impuls" | "Discret" | "Analog" | "Marker"
    events_list: tuple[str, ...]
    rate_s: float | None
    rows: list[DominoRow] = field(default_factory=list)

    @property
    def start_s(self) -> float | None:
        """Start Time als seconden sinds middernacht van de eigen datum."""
        if self.start_time is None:
            return None
        return (self.start_time.hour * 3600 + self.start_time.minute * 60
                + self.start_time.second + self.start_time.microsecond / 1e6)


def _unwrap(tods: Iterable[tuple[float, float | None]], ref_tod: float,
            ref_abs: float) -> list[tuple[float, float | None]]:
    """Kloktijden (seconden van de dag) -> doorlopende seconden.

    Elke rij wordt t.o.v. de vorige gelegd; alleen een sprong TERUG van meer
    dan ``_MIDNIGHT_BACKSTEP_S`` is een nieuwe dag (tot 04-11-2026 gold een
    symmetrische drempel van twaalf uur, waardoor een gat van meer dan twaalf
    uur tussen twee rijen een dag te vroeg viel). Een einde dat vóór zijn
    begin ligt, ligt over middernacht.
    """
    out = []
    prev_tod, prev_abs = ref_tod, ref_abs
    for t0, t1 in tods:
        d = t0 - prev_tod
        if d < -_MIDNIGHT_BACKSTEP_S:
            d += 86400
        a0 = prev_abs + d
        a1 = None
        if t1 is not None:
            e = t1 - t0
            if e < 0:
                e += 86400
            a1 = a0 + e
        out.append((a0, a1))
        prev_tod, prev_abs = t0, a0
    return out


def parse_domino_txt(text: str, name: str) -> DominoSignal:
    """Eén ASCII-exportbestand -> ``DominoSignal``.

    De rijtijden staan in seconden sinds middernacht van de datum in de
    eigen ``Start Time``-regel; ``DominoExport`` legt alle signalen daarna op
    één datum.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header: dict[str, str] = {}
    i = 0
    # kopblok tot de lege regel — of tot de eerste rij die met een kloktijd
    # begint, voor een export zonder die lege regel
    while i < len(lines) and lines[i].strip() and not _RE_ROWSTART.match(lines[i]):
        key, _, val = lines[i].partition(":")
        header[key.strip().lower()] = val.strip()
        i += 1
    if "signal id" not in header:
        raise NotDominoFile(f"{name}: geen DOMINO-exportbestand (kopregel 'Signal ID' ontbreekt)")
    if i < len(lines) and lines[i].strip():
        i -= 1          # geen lege regel: de rijen beginnen op regel i
    start_time = None
    m = _RE_START.search(header.get("start time", ""))
    if m:
        d, mo, y, h, mi, s = (int(g) for g in m.groups())
        start_time = datetime(y, mo, d, h, mi, s)
    kind = header.get("signal type") or ("Marker" if "marker" in header["signal id"].lower() else "")
    events_list = tuple(x.strip() for x in header.get("events list", "").split(",") if x.strip())
    rate_s = None
    if header.get("rate"):
        mr = _RE_RATE.search(header["rate"])
        if mr:
            rate_s = float(mr.group(1).replace(",", "."))
            if (mr.group(2) or "s").lower() == "ms":
                rate_s /= 1000.0

    raw: list[tuple[float, float | None, Any, str | None]] = []
    for ln in lines[i + 1:]:
        if not ln.strip():
            continue
        mi_ = _RE_IMPULS.match(ln)
        if mi_:
            g = mi_.groups()
            rest = g[8]
            val, _, lab = rest.partition(";")
            raw.append((_tod(*g[0:4]), _tod(*g[4:8]), _num(val), lab.strip() or None))
            continue
        mp = _RE_POINT.match(ln)
        if mp:
            g = mp.groups()
            rest = g[4]
            if kind == "Marker":
                raw.append((_tod(*g[0:4]), None, None, rest.strip() or None))
            else:
                raw.append((_tod(*g[0:4]), None, _num(rest), None))
            continue
        raise ValueError(f"{name}: onbegrepen rij {ln[:60]!r}")

    if start_time is not None:
        ref_tod = ref_abs = float(start_time.hour * 3600 + start_time.minute * 60 + start_time.second)
    elif raw:
        ref_tod = ref_abs = float(raw[0][0])
    else:
        ref_tod = ref_abs = 0.0
    times = _unwrap(((r[0], r[1]) for r in raw), ref_tod, ref_abs)
    rows = [DominoRow(a0, a1, r[2], r[3]) for (a0, a1), r in zip(times, raw)]
    rows.sort(key=lambda r: r.t_start)      # de bron is soms 1 s niet-monotoon
    return DominoSignal(name=name, signal_id=header["signal id"], start_time=start_time,
                        unit=header.get("unit", ""), kind=kind, events_list=events_list,
                        rate_s=rate_s, rows=rows)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def _is_time_cell(c: Any) -> bool:
    return isinstance(c, (datetime, time))


def _tod_cell(c: Any) -> float:
    return c.hour * 3600 + c.minute * 60 + c.second + c.microsecond / 1e6


def parse_domino_sheet(name: str, rows: list[tuple]) -> DominoSignal | None:
    """Eén blad van de Excel-export (als rijen van celwaarden) -> ``DominoSignal``
    in dezelfde vorm als ASCII; ``None`` als het blad geen DOMINO-signaal is.
    Tijdcellen mogen ``datetime`` of ``time`` zijn; ``Rate [ms]`` wordt naar
    seconden omgerekend."""
    header: dict[str, Any] = {}
    k = 0
    for k, r in enumerate(rows):
        c0 = r[0] if r else None
        if _is_time_cell(c0):
            break
        if isinstance(c0, str) and len(r) > 1:
            key = c0.strip().lower()
            mu = re.search(r"\[\s*(\w+)\s*\]\s*$", key)
            key = re.sub(r"\s*\[.*\]$", "", key)
            header[key] = r[1]
            if mu:
                header[key + " unit"] = mu.group(1)
    else:
        k = len(rows)
    data = rows[k:]
    is_marker = "marker" in {str(c).strip().lower() for c in (rows[0] if rows else ()) if c}
    if "signal id" not in header and not is_marker:
        return None
    start_time = header.get("start time")
    if isinstance(start_time, str):
        m = _RE_START.search(start_time)
        start_time = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)),
                              int(m.group(4)), int(m.group(5)), int(m.group(6))) if m else None
    elif not isinstance(start_time, datetime):
        start_time = None
    if is_marker:
        kind = "Marker"
        if start_time is None and data and isinstance(data[0][0], datetime):
            start_time = data[0][0].replace(microsecond=0)
    else:
        kind = str(header.get("signal type") or "")
    ev = header.get("events list")
    events_list = tuple(x.strip() for x in str(ev).split(",") if x.strip()) if ev else ()
    rate = header.get("rate")
    rate_s = float(rate) if isinstance(rate, (int, float)) else None
    if rate_s is not None and str(header.get("rate unit", "s")).lower() == "ms":
        rate_s /= 1000.0
    tod = _tod_cell

    raw: list[tuple[float, float | None, Any, str | None]] = []
    for r in data:
        if not r or not _is_time_cell(r[0]):
            continue
        if kind == "Impuls":
            t1 = tod(r[1]) if len(r) > 1 and _is_time_cell(r[1]) else None
            val = r[2] if len(r) > 2 else None
            lab = r[3] if len(r) > 3 else None
            raw.append((tod(r[0]), t1, _num(val) if isinstance(val, str) else val,
                        str(lab).strip() if lab is not None else None))
        elif kind == "Marker":
            raw.append((tod(r[0]), None, None, str(r[1]).strip() if len(r) > 1 and r[1] else None))
        else:
            val = r[1] if len(r) > 1 else None
            raw.append((tod(r[0]), None, _num(val) if isinstance(val, str) else val, None))
    if start_time is not None:
        ref = float(start_time.hour * 3600 + start_time.minute * 60 + start_time.second)
    else:
        ref = raw[0][0] if raw else 0.0
    times = _unwrap(((r[0], r[1]) for r in raw), ref, ref)
    rows_ = [DominoRow(a0, a1, r[2], r[3]) for (a0, a1), r in zip(times, raw)]
    rows_.sort(key=lambda r: r.t_start)
    return DominoSignal(name=name, signal_id=str(header.get("signal id", "User markers")),
                        start_time=start_time, unit=str(header.get("unit") or ""),
                        kind=kind, events_list=events_list, rate_s=rate_s, rows=rows_)


def read_domino_xlsx(path: str | Path) -> dict[str, DominoSignal]:
    """Alle bladen van een DOMINO-Excel-export. Vereist ``openpyxl``."""
    try:
        import openpyxl  # noqa: WPS433  (optioneel; alleen voor .xlsx)
    except ImportError as e:  # pragma: no cover
        raise ImportError("de Excel-export vraagt openpyxl (pip install openpyxl); "
                          "de ASCII-export heeft het niet nodig") from e
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    out: dict[str, DominoSignal] = {}
    for ws in wb.worksheets:
        sig = parse_domino_sheet(ws.title, [tuple(r) for r in ws.iter_rows(values_only=True)])
        if sig is not None:
            out[ws.title] = sig
    return out


# ---------------------------------------------------------------------------
# De export als geheel
# ---------------------------------------------------------------------------

@dataclass
class DominoExport:
    """Alle signalen van één export op één tijdbasis.

    ``start_s``: nulpunt in seconden sinds middernacht van ``reference_date``
    — de marker ``Start`` als die er is, anders de meest voorkomende
    ``Start Time`` van de eventbestanden. Elke methode met ``t0`` rekent
    t.o.v. dat nulpunt tenzij een ander ``t0`` wordt meegegeven (seconden in
    hetzelfde frame, een kloktijd ``"HH:MM:SS"`` of een ``datetime``).
    """
    signals: dict[str, DominoSignal]
    reference_date: date | None
    start_s: float
    source: str
    _index: dict[str, str] = field(default_factory=dict, repr=False)
    _profile: list[tuple[float, str]] = field(default_factory=list, repr=False)
    _profile_starts: list[float] = field(default_factory=list, repr=False)
    _profile_rate: float = 30.0

    # -- opbouw -------------------------------------------------------------
    @classmethod
    def from_signals(cls, signals: dict[str, DominoSignal], source: str) -> "DominoExport":
        if not signals:
            raise ValueError("geen DOMINO-exportbestanden gevonden")
        dates = [s.start_time.date() for s in signals.values() if s.start_time is not None]
        ref_date = min(dates) if dates else None
        # alle signalen op het frame van de referentiedatum
        if ref_date is not None:
            for sig in signals.values():
                if sig.start_time is None:
                    continue
                shift = (sig.start_time.date() - ref_date).days * 86400.0
                if shift:
                    sig.rows = [DominoRow(r.t_start + shift,
                                          None if r.t_end is None else r.t_end + shift,
                                          r.value, r.label) for r in sig.rows]
        ex = cls(signals=signals, reference_date=ref_date, start_s=0.0, source=source)
        ex._index = {_norm(k): k for k in signals}
        ex.start_s = ex._reference_start()
        ex._build_profile()
        return ex

    def _reference_start(self) -> float:
        mk = self._get(SIG_MARKERS)
        if mk is not None:
            for r in mk.rows:
                if r.label and r.label.strip().lower() == "start":
                    return r.t_start
        starts: Counter[float] = Counter()
        for sig in self.signals.values():
            s0 = sig.start_s
            if sig.rate_s is None and sig.start_time is not None and s0 is not None and sig.kind != "Marker":
                shift = 0.0
                if self.reference_date is not None:
                    shift = (sig.start_time.date() - self.reference_date).days * 86400.0
                starts[s0 + shift] += 1
        if starts:
            best = max(starts.values())
            return max(t for t, n in starts.items() if n == best)
        sig = next(iter(self.signals.values()))
        return sig.start_s or 0.0

    def _build_profile(self) -> None:
        prof = self._get(SIG_PROFILE)
        self._profile, self._profile_starts = [], []
        if prof is None:
            return
        self._profile_rate = prof.rate_s or 30.0
        for r in prof.rows:
            lab = str(r.value).strip()
            self._profile.append((r.t_start, DOMINO_STAGE_LABELS.get(lab, "?")))
        self._profile.sort(key=lambda x: x[0])
        self._profile_starts = [t for t, _ in self._profile]

    # -- opzoeken -----------------------------------------------------------
    def _get(self, name: str) -> DominoSignal | None:
        key = self._index.get(_norm(name))
        return self.signals[key] if key is not None else None

    def signal(self, name: str) -> DominoSignal:
        sig = self._get(name)
        if sig is None:
            raise KeyError(f"signaal {name!r} niet in de export; aanwezig: {sorted(self.signals)}")
        return sig

    def has(self, name: str) -> bool:
        return self._get(name) is not None

    # -- tijdbasis ----------------------------------------------------------
    def resolve_t0(self, t0: float | str | datetime | None = None) -> float:
        """``t0`` -> seconden in het frame van de export (zie klasse-docstring)."""
        if t0 is None:
            return self.start_s
        if isinstance(t0, datetime):
            if self.reference_date is None:
                raise ValueError("export zonder datum: geef t0 als seconden of kloktijd")
            return (t0 - datetime.combine(self.reference_date, datetime.min.time())).total_seconds()
        if isinstance(t0, str):
            m = re.match(rf"^\s*{_TIME}\s*$", t0)
            if not m:
                raise ValueError(f"t0 {t0!r} is geen kloktijd HH:MM:SS[,mmm]")
            tod = _tod(*m.groups())
            # de dag die het dichtst bij de start ligt
            cands = [tod + k * 86400 for k in (-1, 0, 1)]
            return min(cands, key=lambda t: abs(t - self.start_s))
        return float(t0)

    def rel(self, t_abs: float, t0: float | str | datetime | None = None) -> float:
        return t_abs - self.resolve_t0(t0)

    def markers_s(self, t0=None) -> dict[str, float]:
        mk = self._get(SIG_MARKERS)
        if mk is None:
            return {}
        z = self.resolve_t0(t0)
        return {r.label: round(r.t_start - z, 3) for r in mk.rows if r.label}

    def marker_s(self, label: str, t0=None) -> float | None:
        for k, v in self.markers_s(t0).items():
            if k.strip().lower() == label.strip().lower():
                return v
        return None

    def lights_off_s(self, t0=None) -> float | None:
        return self.marker_s("Lights Off", t0)

    def lights_on_s(self, t0=None) -> float | None:
        return self.marker_s("Lights On", t0)

    def end_s(self, t0=None) -> float | None:
        return self.marker_s("End", t0)

    # -- hypnogram ----------------------------------------------------------
    def sleep_profile(self, t0=None) -> list[tuple[float, str]]:
        """(relatieve tijd, stadium) per DOMINO-epoch, in psgscoring-labels."""
        z = self.resolve_t0(t0)
        return [(round(t - z, 3), s) for t, s in self._profile]

    @property
    def profile_rate_s(self) -> float:
        return self._profile_rate

    def stage_at(self, t_abs: float) -> str:
        """DOMINO-stadium op een absoluut tijdstip; ``"?"`` buiten het profiel."""
        if not self._profile:
            return "?"
        i = bisect_right(self._profile_starts, t_abs) - 1
        if i < 0:
            return "?"
        t, s = self._profile[i]
        return s if t_abs < t + self._profile_rate else "?"

    def hypnogram_1s(self, t0=None, n_seconds: int | None = None) -> list[str]:
        z = self.resolve_t0(t0)
        if n_seconds is None:
            end = (self._profile[-1][0] + self._profile_rate) if self._profile else z
            n_seconds = max(0, int(round(end - z)))
        return [self.stage_at(z + i) for i in range(int(n_seconds))]

    def hypnogram_epochs(self, t0=None, n_epochs: int | None = None,
                         epoch_s: float = 30.0) -> list[str]:
        """Het profiel herbemonsterd op het machine-raster (midden van elke epoch)."""
        z = self.resolve_t0(t0)
        if n_epochs is None:
            end = (self._profile[-1][0] + self._profile_rate) if self._profile else z
            n_epochs = max(0, int((end - z) // epoch_s))
        return [self.stage_at(z + (k + 0.5) * epoch_s) for k in range(int(n_epochs))]

    def stage_counts(self) -> dict[str, int]:
        return dict(Counter(s for _, s in self._profile))

    def tst_minutes(self) -> float:
        n = sum(1 for _, s in self._profile if s in SLEEP_STAGES)
        return round(n * self._profile_rate / 60.0, 3)

    # -- eventlijsten -------------------------------------------------------
    def _events(self, name: str, t0) -> list[tuple[DominoRow, float, float, str]]:
        sig = self._get(name)
        if sig is None:
            return []
        z = self.resolve_t0(t0)
        out = []
        for r in sig.rows:
            end = r.t_end if r.t_end is not None else r.t_start
            out.append((r, round(r.t_start - z, 3), round(end - r.t_start, 3), self.stage_at(r.t_start)))
        return out

    def respiratory_events(self, t0=None, *, include_unmapped: bool = False) -> list[dict]:
        """``Flow Events`` in het eventschema van psgscoring.

        Elk event: ``type`` (psgscoring-naam), ``onset_s``, ``duration_s``,
        ``label`` (DOMINO-naam), ``stage`` (DOMINO-stadium bij de onset),
        ``value`` (DOMINO's eigen waarde, de duur in s), ``source``,
        ``confidence`` (None: de export draagt er geen).
        """
        out = []
        for r, onset, dur, stage in self._events(SIG_FLOW, t0):
            lab = r.label or ""
            if lab not in DOMINO_RESP_LABELS:
                if not include_unmapped:
                    continue
                typ = None
            else:
                typ = DOMINO_RESP_LABELS[lab]
                if typ is None:
                    continue
            out.append({"type": typ, "onset_s": onset, "duration_s": dur, "label": lab,
                        "stage": stage, "value": r.value, "source": "domino",
                        "confidence": None})
        return out

    @staticmethod
    def events_in_sleep(events: Iterable[dict]) -> list[dict]:
        return [e for e in events if e.get("stage") in SLEEP_STAGES]

    def verify_label_mapping(self) -> dict:
        """Tel de labels in ``Flow Events``: gekend, uitgesloten, onbekend."""
        sig = self._get(SIG_FLOW)
        labels = Counter((r.label or "") for r in (sig.rows if sig else []))
        return {
            "flow_labels": dict(labels),
            "mapped": {l: DOMINO_RESP_LABELS[l] for l in labels
                       if DOMINO_RESP_LABELS.get(l) is not None},
            "excluded": {l: n for l, n in labels.items()
                         if l in DOMINO_RESP_LABELS and DOMINO_RESP_LABELS[l] is None},
            "unmapped": {l: n for l, n in labels.items() if l not in DOMINO_RESP_LABELS},
        }

    def desaturations(self, t0=None) -> list[dict]:
        out = []
        for r, onset, dur, stage in self._events(SIG_SPO2_EVENTS, t0):
            out.append({"onset_s": onset, "duration_s": dur,
                        "drop_pct": float(r.value) if isinstance(r.value, (int, float)) else None,
                        "label": r.label, "stage": stage, "source": "domino"})
        return out

    def limb_events(self, t0=None) -> list[dict]:
        out = []
        for r, onset, dur, stage in self._events(SIG_PLM, t0):
            lab = r.label or ""
            out.append({"onset_s": onset, "duration_s": dur,
                        "type": DOMINO_LIMB_LABELS.get(lab, "unknown"), "label": lab,
                        "is_plm": lab == "P-Marker", "value": r.value, "stage": stage,
                        "source": "domino"})
        return out

    def arousals(self, t0=None) -> list[dict]:
        out = []
        for r, onset, dur, stage in self._events(SIG_AROUSAL, t0):
            lab = r.label or ""
            auto = "(autonomic)" in lab.lower()
            kind = re.sub(r"\s*\(autonomic\)\s*", "", lab, flags=re.I).strip()
            out.append({"onset_s": onset, "duration_s": dur, "label": lab, "kind": kind,
                        "autonomic": auto, "stage": stage, "source": "domino"})
        return out

    def eeg_arousals(self, t0=None) -> list[dict]:
        return [a for a in self.arousals(t0) if not a["autonomic"]]

    # -- reeksen ------------------------------------------------------------
    def series(self, name: str, t0=None) -> list[tuple[float, Any]]:
        """(relatieve tijd, waarde) van een Analog/Discret-signaal; artefact
        (``A``) en andere niet-numerieke waarden van een Analog-signaal worden
        ``None``, toestanden van een Discret-signaal blijven tekst."""
        sig = self.signal(name)
        z = self.resolve_t0(t0)
        out = []
        for r in sig.rows:
            v = r.value
            if sig.kind == "Analog" and not isinstance(v, (int, float)):
                v = None
            elif isinstance(v, bool):
                v = float(v)
            out.append((round(r.t_start - z, 3), v))
        return out

    def series_1hz(self, name: str, t0=None, n_seconds: int | None = None,
                   max_gap_s: float | None = None) -> list[float | None]:
        """Analog-signaal op 1 Hz vanaf ``t0``: het laatste monster ≤ t geldt
        (DOMINO schrijft een rij per verandering), tot een artefactrij
        (``None``). Met ``max_gap_s`` stopt het vooruitvullen na zoveel
        seconden; standaard onbegrensd."""
        pts = self.series(name, t0)
        if n_seconds is None:
            n_seconds = int(max((t for t, _ in pts), default=0)) + 1
        ts = [t for t, _ in pts]
        out: list[float | None] = []
        for i in range(int(n_seconds)):
            j = bisect_right(ts, float(i)) - 1
            if j < 0:
                out.append(None)
                continue
            t, v = pts[j]
            out.append(None if (v is None or (max_gap_s is not None and i - t > max_gap_s))
                       else float(v))
        return out


def _signals_from_dir(d: Path) -> dict[str, DominoSignal]:
    out: dict[str, DominoSignal] = {}
    for p in sorted(d.glob("*.txt")):
        try:
            out[p.stem] = parse_domino_txt(p.read_bytes().decode("cp1252", errors="replace"), p.stem)
        except NotDominoFile:
            continue        # een los tekstbestand in de map; elke andere fout komt eruit
    return out


def _signals_from_zip(z: Path) -> dict[str, DominoSignal]:
    out: dict[str, DominoSignal] = {}
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if info.is_dir() or not base.lower().endswith(".txt") or "__macosx" in info.filename.lower():
                continue
            try:
                out[base[:-4]] = parse_domino_txt(zf.read(info).decode("cp1252", errors="replace"), base[:-4])
            except NotDominoFile:
                continue
    return out


def read_domino_export(path: str | Path) -> DominoExport:
    """Lees een DOMINO-export: een map met .txt-bestanden, een .zip daarvan,
    of de .xlsx-vorm."""
    p = Path(path)
    if p.is_dir():
        sigs, source = _signals_from_dir(p), "ascii"
    elif p.suffix.lower() == ".zip":
        sigs, source = _signals_from_zip(p), "ascii"
    elif p.suffix.lower() in (".xlsx", ".xlsm"):
        sigs, source = read_domino_xlsx(p), "xlsx"
    elif p.suffix.lower() == ".txt":
        sigs, source = {p.stem: parse_domino_txt(p.read_bytes().decode("cp1252", errors="replace"), p.stem)}, "ascii"
    else:
        raise ValueError(f"{p}: geen map, .zip, .xlsx of .txt")
    if not sigs:
        raise ValueError(f"{p}: geen DOMINO-exportbestanden gevonden (kopregel 'Signal ID' ontbreekt)")
    return DominoExport.from_signals(sigs, source)


# ---------------------------------------------------------------------------
# Index, uitlijning, hypnogram-overeenstemming
# ---------------------------------------------------------------------------

def _counts_for_ahi(e: dict) -> bool:
    t = str(e.get("type") or "")
    return t.startswith("hypopnea") or t in ("obstructive", "central", "mixed", "apnea")


def ahi_from_domino_events(events: Iterable[dict], sleep_hours: float) -> float | None:
    """AHI uit DOMINO-events over een OPGEGEVEN slaaptijd; RERA's tellen niet.

    De noemer komt bewust van buiten: deel door dezelfde slaaptijd als aan de
    machinekant, of door DOMINO's eigen ``tst_minutes()/60`` — maar zeg erbij
    welke.
    """
    if not sleep_hours or sleep_hours <= 0:
        return None
    return round(sum(1 for e in events if _counts_for_ahi(e)) / float(sleep_hours), 1)


def estimate_clock_offset(domino_1hz: Sequence[float | None], machine_1hz: Sequence[float | None],
                          *, max_lag_s: int = 1200, coarse_step_s: int = 10,
                          min_overlap_s: int = _MIN_OVERLAP_S) -> tuple[int, float, int]:
    """Schat hoeveel seconden bij ``t0`` OPGETELD moeten worden.

    ``domino_1hz`` is een reeks van de export op 1 Hz, berekend met het
    aangenomen ``t0`` (``series_1hz``); ``machine_1hz`` dezelfde grootheid uit
    de machine-uitvoer vanaf de EDF-start. Gezocht wordt de verschuiving
    ``L`` waarvoor ``domino[i] ≈ machine[i - L]`` de hoogste correlatie geeft:
    dan begint de EDF op ``t0 + L``. Grof met stap ``coarse_step_s``, daarna
    fijn op de seconde. Retourneert ``(L, r, n_paren)``.
    """
    d = np.array([np.nan if v is None else float(v) for v in domino_1hz], dtype=float)
    m = np.array([np.nan if v is None else float(v) for v in machine_1hz], dtype=float)
    n = min(len(d), len(m))
    d, m = d[:n], m[:n]

    def corr(lag: int) -> tuple[float, int]:
        if lag >= 0:
            x, y = d[lag:], m[:n - lag]
        else:
            x, y = d[:n + lag], m[-lag:]
        ok = np.isfinite(x) & np.isfinite(y)
        k = int(ok.sum())
        if k < min_overlap_s:
            return float("nan"), k
        xs, ys = x[ok], y[ok]
        if xs.std() == 0 or ys.std() == 0:
            return float("nan"), k
        return float(np.corrcoef(xs, ys)[0, 1]), k

    lags = list(range(-max_lag_s, max_lag_s + 1, max(1, coarse_step_s)))
    coarse = {L: corr(L) for L in lags}
    valid = {L: r for L, (r, _) in coarse.items() if np.isfinite(r)}
    if not valid:
        raise ValueError(f"te weinig overlap tussen de reeksen (< {min_overlap_s} s) om een "
                         "klokverschuiving te schatten")
    best = max(valid, key=lambda L: valid[L])
    fine = {L: corr(L) for L in range(best - coarse_step_s - 5, best + coarse_step_s + 6)}
    fine_valid = {L: r for L, (r, _) in fine.items() if np.isfinite(r)}
    best = max(fine_valid, key=lambda L: fine_valid[L])
    r, k = fine[best]
    return int(best), round(float(r), 4), int(k)


def hypnogram_kappa(a: Sequence[str], b: Sequence[str], *,
                    ignore: Iterable[str] = ("?",)) -> dict:
    """Cohen's κ, nauwkeurigheid en verwarringsmatrix van twee gelijk
    bemonsterde stadiumreeksen; paren met een label uit ``ignore`` tellen niet."""
    ign = set(ignore)
    pairs = [(x, y) for x, y in zip(a, b) if x not in ign and y not in ign]
    n = len(pairs)
    conf: dict[str, dict[str, int]] = {}
    for x, y in pairs:
        conf.setdefault(x, {}).setdefault(y, 0)
        conf[x][y] += 1
    if n == 0:
        return {"kappa": None, "accuracy": None, "n": 0, "confusion": conf}
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    kappa = 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)
    return {"kappa": round(kappa, 4), "accuracy": round(po, 4), "n": n, "confusion": conf}
