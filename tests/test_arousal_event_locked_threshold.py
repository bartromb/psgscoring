"""Event-locked werkpunt: een lagere cutoff waar de prior hoog is.

WAAROM DIT, EN NIET WAT HET DIAGNOSEDOCUMENT VOORSTELDE
-------------------------------------------------------
D3 stelde voor de KANDIDAATdrempels te verlagen rond een respiratoir
event-einde. Gemeten op PSG-IPA SN3 (327 menselijke events):

    kandidaat in het venster (ruime drempels) : 283  (87 %)
    overleeft de filter op 0,80               :  89  (27 %)
    events zonder ENIGE kandidaat             :  44   <- ruimte voor D3
    events met kandidaat, weggefilterd        : 194   <- filterverlies

Het filterverlies is ruim vier keer zo groot. Kandidaten toevoegen waar er al
een ligt die weggegooid wordt, verandert niets; de ingreep hoort bij de FILTER.

WAAROM DIT VERDEDIGBAAR IS EN GEEN TRUC
---------------------------------------
Het model levert een kans, geen beslissing. De drempel waarop je die kans
afkapt hoort van de PRIOR af te hangen, en die prior is vlak na het einde van
een respiratoir event aantoonbaar anders: mensen koppelen daar 60,4 % van hun
events aan een arousal (gepoold over PSG-IPA), tegen een nachtgemiddelde dat
veel lager ligt. Eén vaste cutoff over de hele nacht negeert dat.

Het venster is exact dat van `correlate_arousals_to_respiratory`
(event-onset tot 15 s na het einde), zodat detectie en koppeling dezelfde
geometrie delen -- anders vind je events die de koppeling daarna niet erkent.

Default UIT tot gemeten, zoals elke gedragswijziging hier.
"""
import numpy as np
import pytest

from psgscoring.arousal import detect_arousals

SF = 100.0
DUR_S = 1200


def _recording(seed=17):
    """EEG met twee ZWAKKE verschuivingen: één in een eventvenster, één ver
    daarbuiten. Beide net te zwak voor de gewone cutoff."""
    t = np.arange(int(DUR_S * SF)) / SF
    rng = np.random.default_rng(seed)
    eeg = (60.0 * np.sin(2 * np.pi * 1.5 * t)
           + 6.0 * np.sin(2 * np.pi * 6.0 * t)
           + rng.normal(0.0, 1.0, t.size))
    for at in (305.0, 905.0):          # in venster / buiten venster
        s, e = int(at * SF), int((at + 5.0) * SF)
        eeg[s:e] = (16.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                    + 16.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                    + rng.normal(0.0, 1.0, e - s))
    # sterke verschuivingen zodat de fixture uberhaupt kandidaten oplevert
    for at in (120.0, 480.0, 660.0):
        s, e = int(at * SF), int((at + 6.0) * SF)
        eeg[s:e] = (30.0 * np.sin(2 * np.pi * 10.0 * t[s:e])
                    + 30.0 * np.sin(2 * np.pi * 20.0 * t[s:e])
                    + rng.normal(0.0, 1.0, e - s))
    emg = rng.normal(0.0, 5.0, t.size)
    for at in (120.0, 305.0, 480.0, 660.0, 905.0):
        s, e = int(at * SF), int((at + 4.0) * SF)
        emg[s:e] += rng.normal(0.0, 40.0, e - s)
    return eeg * 1e-6, emg * 1e-6, ["N2"] * int(DUR_S / 30)


def _resp_fixture(mne):
    """Flowreducties MET desaturatie. Zonder de desat kwalificeert er niets
    als hypopneu en levert de fixture nul respiratoire events -- dan meet een
    test over de event-eindes niets, wat precies het punt is."""
    sf, minuten = 64.0, 40
    n = int(sf * 60 * minuten)
    t = np.arange(n) / sf
    rng = np.random.default_rng(4)
    eeg = rng.normal(0, 20e-6, n)
    flow = np.sin(2 * np.pi * 0.25 * t)
    spo2 = np.full(n, 96.0)
    for start in range(90, 60 * minuten - 90, 120):
        a, b = int(start * sf), int((start + 15) * sf)
        flow[a:b] *= 0.02          # APNEUS, niet hypopneeen
        spo2[b:b + int(15 * sf)] = 90.0
        eeg[b:b + int(5 * sf)] += 70e-6 * np.sin(
            2 * np.pi * 10.0 * t[b:b + int(5 * sf)])
    info = mne.create_info(
        ["Resp nasal", "SaO2", "EEG C4-M1", "EMG chin", "Thorax", "Abdomen"],
        sf, ["misc", "misc", "eeg", "misc", "misc", "misc"])
    effort = np.sin(2 * np.pi * 0.25 * t) * 0.02
    raw = mne.io.RawArray(
        np.vstack([flow, spo2, eeg, rng.normal(0, 5e-6, n), effort, effort]),
        info, verbose=False)
    hypno = ["N2"] * int(np.ceil(raw.times[-1] / 30.0))
    return raw, hypno


def _n(eeg, emg, hypno, **kw):
    out = detect_arousals(eeg, SF, hypno, emg_data=emg, lgbm=True,
                          lgbm_threshold=0.80, **kw)
    return out, len(out.get("events") or [])


def test_the_flag_is_inert_without_event_ends():
    eeg, emg, hypno = _recording()
    _a, zonder = _n(eeg, emg, hypno)
    _b, met = _n(eeg, emg, hypno, event_locked_threshold=0.40)
    assert met == zonder, (
        "zonder eventlijst mag een venster-werkpunt niets doen")


def test_the_flag_is_inert_without_a_relaxed_threshold():
    eeg, emg, hypno = _recording()
    _a, zonder = _n(eeg, emg, hypno)
    _b, met = _n(eeg, emg, hypno, resp_event_ends=[300.0])
    assert met == zonder, "een eventlijst zonder drempel mag niets doen"


def test_a_relaxed_window_can_only_add_never_remove():
    """Een lagere drempel in een venster mag buiten dat venster niets
    veranderen, en binnen het venster nooit iets wegnemen."""
    eeg, emg, hypno = _recording()
    basis, n_basis = _n(eeg, emg, hypno)
    _r, n_relaxed = _n(eeg, emg, hypno, resp_event_ends=[300.0],
                       event_locked_threshold=0.20)
    assert n_relaxed >= n_basis, (
        f"venster-werkpunt verwijderde events: {n_relaxed} < {n_basis}")
    basis_onsets = {round(e["onset_s"], 1) for e in basis["events"]}
    relaxed_onsets = {round(e["onset_s"], 1) for e in _r["events"]}
    assert basis_onsets <= relaxed_onsets, (
        "een event uit de strengere arm ontbreekt in de ruimere")


def test_a_stricter_window_threshold_is_refused():
    """Het venster mag alleen VERSOEPELEN. Een hogere drempel daar zou de
    prior omgekeerd toepassen."""
    eeg, emg, hypno = _recording()
    with pytest.raises(ValueError, match="strenger|lager"):
        detect_arousals(eeg, SF, hypno, emg_data=emg, lgbm=True,
                        lgbm_threshold=0.50, resp_event_ends=[300.0],
                        event_locked_threshold=0.90)


def test_events_carry_the_threshold_that_admitted_them():
    """Zonder dat is achteraf niet te zien welke events aan het venster te
    danken zijn -- en dat is precies wat een lezer moet kunnen nagaan."""
    eeg, emg, hypno = _recording()
    out, _n2 = _n(eeg, emg, hypno, resp_event_ends=[300.0],
                  event_locked_threshold=0.20)
    for e in out["events"]:
        assert "lgbm_threshold_used" in e, e
    binnen = [e for e in out["events"] if e.get("event_locked")]
    assert all(0 <= e["onset_s"] <= 320.0 for e in binnen), binnen


def test_the_summary_counts_the_window_admissions():
    eeg, emg, hypno = _recording()
    out, _n3 = _n(eeg, emg, hypno, resp_event_ends=[300.0],
                  event_locked_threshold=0.20)
    s = out["summary"]
    assert "n_event_locked" in s
    assert s["n_event_locked"] == sum(
        1 for e in out["events"] if e.get("event_locked"))


def test_the_window_matches_the_coupling_definition():
    """Detectie en koppeling moeten dezelfde geometrie delen: een event dat
    het venster toelaat maar de koppeling niet erkent, is een event dat
    nergens in terugkomt."""
    from psgscoring.arousal import POST_RESP_WINDOW_S, arousal_couples_to_event
    eeg, emg, hypno = _recording()
    out, _n4 = _n(eeg, emg, hypno, resp_event_ends=[300.0],
                  event_locked_threshold=0.20)
    for e in out["events"]:
        if not e.get("event_locked"):
            continue
        assert arousal_couples_to_event(
            e["onset_s"], 300.0 - 5.0, 300.0, POST_RESP_WINDOW_S), (
            f"venster liet {e['onset_s']} toe maar de koppeling erkent hem niet")


# ══════════════════════════════════════════════════════════════
# Profielvlag
# ══════════════════════════════════════════════════════════════

def test_the_flag_is_off_on_every_profile():
    from psgscoring.profiles import get_profile, list_profiles
    for naam in list_profiles():
        v = get_profile(naam).post_processing.arousal_event_locked_threshold
        assert v is None, f"{naam} heeft het venster-werkpunt aan staan: {v}"


def test_the_registry_carries_the_field():
    from psgscoring.constants import SCORING_PROFILES
    for naam, d in SCORING_PROFILES.items():
        assert "AROUSAL_EVENT_LOCKED_THRESHOLD" in d, naam


# ══════════════════════════════════════════════════════════════
# Doorgifte: de vlag moet de detector BEREIKEN
# ══════════════════════════════════════════════════════════════
#
# De les van v0.27.0: een override die stil niet aankomt lijkt een resultaat.
# Die meting was toen 30/30 identiek omdat de pijplijn het profielveld niet
# doorgaf. Deze tests falen als dat opnieuw gebeurt.

def test_the_pipeline_passes_the_flag_and_the_event_ends(monkeypatch):
    mne = pytest.importorskip("mne")
    import psgscoring
    import psgscoring.arousal as A

    gezien = {}
    echt = A.detect_arousals

    def spion(*args, **kw):
        gezien.setdefault("calls", []).append(
            (kw.get("event_locked_threshold"), kw.get("resp_event_ends")))
        return echt(*args, **kw)

    monkeypatch.setattr(A, "detect_arousals", spion)

    raw, hypno = _resp_fixture(mne)

    monkeypatch.setenv("PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD", "0.30")
    # `aasm_v3_rec`, niet `aasm_v3_breath`: zie de test hieronder over de
    # volgorde. Op de breath-profielen bestaan de hypopneeen nog niet wanneer
    # de arousalstap draait.
    psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                   scoring_profile="aasm_v3_rec")
    assert gezien.get("calls"), "detect_arousals is niet aangeroepen"
    drempels = {d for d, _e in gezien["calls"]}
    assert 0.30 in drempels, (
        f"het venster-werkpunt bereikt de detector niet: {drempels}")
    eindes = [e for _d, e in gezien["calls"] if e]
    assert eindes, ("de respiratoire event-eindes bereiken de detector niet "
                    "(of de fixture levert geen events -- dan meet deze test "
                    "niets)")
    assert len(eindes[0]) >= 5, eindes[0]


def test_the_env_override_works_in_both_directions(monkeypatch):
    """Meten zonder de registry te muteren, en hem ook UIT kunnen zetten op
    een profiel dat hem aan heeft."""
    from psgscoring.pipeline import _arousal_event_locked_threshold
    prof = {"AROUSAL_EVENT_LOCKED_THRESHOLD": 0.40}
    assert _arousal_event_locked_threshold(prof) == 0.40
    monkeypatch.setenv("PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD", "0.25")
    assert _arousal_event_locked_threshold(prof) == 0.25
    monkeypatch.setenv("PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD", "")
    assert _arousal_event_locked_threshold(prof) is None


def test_the_multi_wrapper_propagates_it():
    """Multi is de klinische default. Kwam de vlag daar niet doorheen, dan
    deed hij in productie niets terwijl de tests groen stonden."""
    from psgscoring.arousal import detect_arousals_multi
    eeg, emg, hypno = _recording()
    derivs = [("A", eeg, SF), ("B", eeg * 0.98, SF)]
    zonder = detect_arousals_multi(derivs, SF, hypno, emg_data=emg,
                                   lgbm=True, lgbm_threshold=0.80)
    met = detect_arousals_multi(derivs, SF, hypno, emg_data=emg,
                                lgbm=True, lgbm_threshold=0.80,
                                resp_event_ends=[300.0],
                                event_locked_threshold=0.20)
    assert len(met.get("events") or []) >= len(zonder.get("events") or [])
    assert met["summary"].get("n_event_locked") is not None, (
        "de multi-wrapper laat de venstertelling vallen")


def test_on_breath_profiles_only_the_apneas_are_in_the_window(monkeypatch):
    """DE BEPERKING, expliciet vastgelegd in plaats van verzwegen.

    Op de `breath_graded`-profielen -- de klinische defaults -- vervangt stap
    7b de hypopneeen NA de arousalstap. De event-eindes die het venster ziet
    zijn dan alleen die van de apneus. Op de klinische opname die dit traject
    motiveerde is dat 235 van de 377 events (63 %); op een fixture met
    uitsluitend hypopneeen is het nul.

    Dat halveert het bereik van het venster op precies de profielen waar het
    het meest nodig is. Repareren vraagt een tweede detectiepas NA stap 7b, en
    dat is een aparte ingreep: de breath-detector gebruikt de arousals zelf
    (HYPOPNEA_AROUSAL_WEIGHT), dus er is een echte cyclische afhankelijkheid.

    Deze test faalt zodra dat verandert -- dan hoort de documentatie mee te
    veranderen.
    """
    mne = pytest.importorskip("mne")
    import psgscoring
    import psgscoring.arousal as A

    gezien = {}
    echt = A.detect_arousals

    def spion(*args, **kw):
        gezien["ends"] = kw.get("resp_event_ends")
        return echt(*args, **kw)

    monkeypatch.setattr(A, "detect_arousals", spion)
    raw, hypno = _resp_fixture(mne)
    monkeypatch.setenv("PSGSCORING_AROUSAL_EVENT_LOCKED_THRESHOLD", "0.30")
    out = psgscoring.run_pneumo_analysis(raw, hypno=hypno,
                                         scoring_profile="aasm_v3_breath")
    ev = out["respiratory"]["events"]
    assert ev, "fixture levert geen events"
    n_apneu = sum(1 for e in ev if e.get("type") in
                  ("obstructive", "central", "mixed", "uncertain"))
    n_hyp = len(ev) - n_apneu
    # De fixture is bewust apneu-only, dus het venster ziet ze allemaal. Wat
    # deze test vastlegt is de REGEL: het venster ziet exact de events die bij
    # de arousalstap al bestaan, en op breath-profielen zijn de hypopneeen dat
    # niet.
    assert len(gezien.get("ends") or []) == n_apneu, (
        f"venster zag {len(gezien.get('ends') or [])} eindes bij {n_apneu} "
        f"apneus en {n_hyp} hypopneeen -- als dit aantal de hypopneeen "
        f"meetelt, is de volgorde veranderd en klopt de beperking niet meer")
