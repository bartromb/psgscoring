"""Bij een split-night hoort ELKE index over het juiste stuk nacht te gaan.

WAAROM DIT BESTAAT
------------------
De AHI werd in 0.29.0 gesplitst, de rest niet. Daardoor stond in het rapport
een diagnostische AHI naast een arousalindex, een RDI en een PLM-index over de
HELE nacht -- inclusief de uren onder CPAP. Dat verdunt precies zoals de
nacht-AHI verdunde: een patiënt met 60 arousals in twee diagnostische uren en
5 in vijf uur titratie leest als 9,3/u, terwijl het diagnostische deel op
30/u ligt.

De richting van de fout is niet neutraal. De therapie werkt, dus het tweede
deel drukt elk getal omlaag; wat overblijft ziet er milder uit dan de meting
waarop de diagnose rust.

DE NOEMER
---------
Alles deelt door DEZELFDE segmentnoemer als de segment-AHI, opgehaald uit
`_compute_summary`. Een tweede definitie van slaaptijd is hier al eerder
misgegaan: één rapport toonde 44,3/u en 43,2/u voor dezelfde teller.
"""
import pytest

from psgscoring.split_night import (
    segment_arousals,
    segment_denominator_h,
    segment_indices,
    segment_plm,
    segment_rdi,
)

BREUK = 2 * 3600.0
HYP = ["N2"] * (7 * 120)          # 7 uur slaap, epochs van 30 s


def _arousals(n_voor, n_na, n_resp_voor=0, n_resp_na=0):
    """Arousals zoals de correlatiestap ze publiceert: mét `type`."""
    ev = []
    for i in range(n_voor):
        ev.append({"onset_s": 60.0 + 100 * i, "duration_s": 5.0,
                   "type": "respiratory" if i < n_resp_voor else "spontaneous"})
    for i in range(n_na):
        ev.append({"onset_s": BREUK + 60.0 + 100 * i, "duration_s": 5.0,
                   "type": "respiratory" if i < n_resp_na else "spontaneous"})
    return ev


# ── De noemer ─────────────────────────────────────────────────────────────

def test_de_noemer_is_dezelfde_als_die_van_de_segment_ahi():
    """Niet ongeveer gelijk: hetzelfde getal, uit dezelfde functie."""
    d = segment_denominator_h(HYP, BREUK)
    s = segment_indices([], HYP, BREUK)
    assert d["diagnostic"] == s["diagnostic"]["sleep_h"]
    assert d["therapeutic"] == s["therapeutic"]["sleep_h"]
    assert d["diagnostic"] == pytest.approx(2.0, abs=0.05)


def test_artefactepochs_verkleinen_de_noemer():
    """Zonder dit deelt de arousalindex door meer slaap dan de AHI."""
    art = list(range(120))                 # eerste uur volledig artefact
    d = segment_denominator_h(HYP, BREUK, artifact_epochs=art)
    assert d["diagnostic"] == pytest.approx(1.0, abs=0.05)


# ── Arousals ──────────────────────────────────────────────────────────────

def test_de_arousalindex_wordt_per_deel_gerekend():
    """De casus uit de docstring: 60 in twee uur, 5 in vijf uur."""
    a = segment_arousals(_arousals(60, 5), HYP, BREUK)
    assert a["diagnostic"]["n_arousals"] == 60
    assert a["therapeutic"]["n_arousals"] == 5
    assert a["diagnostic"]["arousal_index"] == pytest.approx(30.0, abs=0.5)
    assert a["therapeutic"]["arousal_index"] == pytest.approx(1.0, abs=0.3)
    # En niet het verdunde nachtgetal van 9,3/u.
    assert a["diagnostic"]["arousal_index"] > 20


def test_respiratoir_en_spontaan_worden_apart_geteld_per_deel():
    """Onder CPAP hoort het respiratoire deel weg te vallen; dat is juist het
    klinisch interessante verschil en het verdwijnt in een nachtgemiddelde."""
    a = segment_arousals(_arousals(60, 5, n_resp_voor=48, n_resp_na=0), HYP, BREUK)
    d, t = a["diagnostic"], a["therapeutic"]
    assert d["n_respiratory"] == 48 and d["n_spontaneous"] == 12
    assert t["n_respiratory"] == 0 and t["n_spontaneous"] == 5
    assert d["respiratory_arousal_index"] == pytest.approx(24.0, abs=0.5)
    assert t["respiratory_arousal_index"] == pytest.approx(0.0, abs=0.01)
    # De twee deelindices tellen op tot de totale index van dat segment.
    assert (d["respiratory_arousal_index"] + d["spontaneous_arousal_index"]
            == pytest.approx(d["arousal_index"], abs=0.15))


def test_de_arousals_van_beide_delen_zijn_samen_de_hele_nacht():
    """Geen event mag tussen de segmenten wegvallen of dubbel geteld worden."""
    ev = _arousals(60, 5)
    a = segment_arousals(ev, HYP, BREUK)
    assert a["diagnostic"]["n_arousals"] + a["therapeutic"]["n_arousals"] == len(ev)


def test_zonder_typeveld_blijft_de_totaalindex_staan():
    """Een externe arousallijst draagt geen etiologie; dan hoort de index er
    nog steeds te zijn en de onderverdeling afwezig -- niet nul."""
    kaal = [{"onset_s": 60.0 + 100 * i, "duration_s": 5.0} for i in range(60)]
    a = segment_arousals(kaal, HYP, BREUK)
    assert a["diagnostic"]["arousal_index"] == pytest.approx(30.0, abs=0.5)
    assert a["diagnostic"]["respiratory_arousal_index"] is None


def test_een_te_kort_segment_is_niet_betrouwbaar():
    """Zelfde regel als bij de AHI: een half uur draagt geen index."""
    korte_hyp = ["N2"] * (3 * 120)
    a = segment_arousals(_arousals(2, 40), korte_hyp, 900.0)   # 15 min diagnostisch
    assert a["diagnostic"]["reliable"] is False
    assert a["therapeutic"]["reliable"] is True


# ── RDI ───────────────────────────────────────────────────────────────────

def _resp_events(n_voor, n_na):
    ev = [{"onset_s": 60.0 + 120 * i, "duration_s": 15.0,
           "type": "obstructive", "stage": "N2"} for i in range(n_voor)]
    ev += [{"onset_s": BREUK + 60.0 + 120 * i, "duration_s": 15.0,
            "type": "hypopnea", "stage": "N2"} for i in range(n_na)]
    return ev


def test_de_rdi_wordt_per_deel_gerekend():
    """RDI = AHI + RERA-index, per segment op dezelfde noemer."""
    rera_onsets = [30.0 + 200 * i for i in range(20)]          # 20 diagnostisch
    rera_onsets += [BREUK + 30.0 + 200 * i for i in range(2)]  # 2 onder therapie
    r = segment_rdi(_resp_events(60, 6), rera_onsets, HYP, BREUK)
    assert r["diagnostic"]["n_rera"] == 20
    assert r["therapeutic"]["n_rera"] == 2
    assert r["diagnostic"]["rera_index"] == pytest.approx(10.0, abs=0.3)
    # AHI 30 + RERA 10
    assert r["diagnostic"]["rdi"] == pytest.approx(40.0, abs=0.6)
    assert r["therapeutic"]["rdi"] == pytest.approx(1.6, abs=0.4)


def test_zonder_reras_is_de_rdi_de_ahi():
    r = segment_rdi(_resp_events(60, 6), [], HYP, BREUK)
    assert r["diagnostic"]["rdi"] == pytest.approx(30.0, abs=0.5)


# ── PLM ───────────────────────────────────────────────────────────────────

def test_de_plm_index_wordt_per_deel_gerekend():
    plm = [{"onset_s": 60.0 + 90 * i, "duration_s": 2.0, "is_plm": True}
           for i in range(40)]
    plm += [{"onset_s": BREUK + 60.0 + 90 * i, "duration_s": 2.0, "is_plm": True}
            for i in range(5)]
    p = segment_plm(plm, HYP, BREUK)
    assert p["diagnostic"]["n_plm"] == 40
    assert p["therapeutic"]["n_plm"] == 5
    assert p["diagnostic"]["plm_index"] == pytest.approx(20.0, abs=0.5)
    assert p["therapeutic"]["plm_index"] == pytest.approx(1.0, abs=0.3)


def test_lege_lijsten_leveren_nul_en_geen_fout():
    for fn, arg in ((segment_arousals, []), (segment_plm, [])):
        uit = fn(arg, HYP, BREUK)
        assert uit["diagnostic"]["reliable"] is True
        assert uit["therapeutic"]["reliable"] is True


def test_geen_slaap_geeft_geen_getal_maar_None():
    """De 81000/u-les: een index zonder noemer is geen index."""
    wakker = ["W"] * (7 * 120)
    a = segment_arousals(_arousals(60, 5), wakker, BREUK)
    assert a["diagnostic"]["arousal_index"] is None
    p = segment_plm([{"onset_s": 60.0, "duration_s": 2.0}], wakker, BREUK)
    assert p["diagnostic"]["plm_index"] is None


def test_alleen_reeksdeelnemers_tellen_mee_in_de_plm_index():
    """`analyze_plm` publiceert ALLE in aanmerking komende beenbewegingen; de
    index rekent alleen die in een gekwalificeerde reeks.

    De eerste versie telde de lijst zelf. Op een echte opname gaf dat PLMI 91,3
    en 29,2 per helft naast 12,9 over de nacht -- twee helften die allebei ver
    boven hun eigen gemiddelde liggen. Een synthetische lijst kon dat niet
    laten zien, want die bevatte per constructie alleen reeksdeelnemers.
    """
    plm = [{"onset_s": 60.0 + 90 * i, "duration_s": 2.0, "is_plm": i < 20}
           for i in range(40)]
    p = segment_plm(plm, HYP, BREUK)
    assert p["diagnostic"]["n_plm"] == 20, "losse bewegingen tellen mee"
    assert p["diagnostic"]["n_lm_eligible"] == 40
    assert p["diagnostic"]["plm_index"] == pytest.approx(10.0, abs=0.3)
