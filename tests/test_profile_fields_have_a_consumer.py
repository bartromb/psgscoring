"""
Een profielveld dat niemand leest is geen instelling maar een belofte.

`profiles.py` heet in de code "the single source of truth" voor het
scoringsgedrag. Op 31-08-2026 bleken twaalf van de 101 velden dat niet te zijn:
ze staan in de dataclass, profielen zetten ze, en geen enkele consument leest
ze. Twee ervan VARIEREN bovendien over de profielen, dus die beweren vandaag al
iets dat niet gebeurt.

Hoe dat kon blijven staan: `csr_reclassification` was er zo een, en hij kwam
niet boven door een test maar doordat een duplicaat-lint aansloeg toen hij per
ongeluk een tweede keer gedeclareerd werd. Zonder die vergissing was hij nog
jaren blijven liggen. `tests/test_env_overrides_reachable.py` dekt hetzelfde
gat voor omgevingsvariabelen; dit is de profielkant.

DE POORT IS EEN ALLOW-LIST, GEEN VERBOD. Wie een veld toevoegt dat (nog) geen
consument heeft, zet het hier neer met een reden. Dat dwingt een BESLISSING af
in plaats van een ontdekking tijdens een meting.

SLEUTEL IS (DATACLASS, VELD), NIET DE KALE VELDNAAM. Tot 11-10-2026 zocht de
poort op `\\.{veld}\\b` in constants.py, ongeacht welke dataclass het veld
droeg. Daardoor dekte `h.flow_reduction_threshold` (HypopneaRules) ook
`ApneaRules.flow_reduction_threshold` af, en `pp.baseline_window_s`
(PostProcessingRules) ook `SpO2Rules.baseline_window_s` -- twee velden die
profielen expliciet pinnen (`aasm_v3_sensitive` 0.90, `aasm_v3_strict` 120.0)
terwijl de detector een moduleconstante leest. De renderer wordt nu met de
AST gelezen en de attribuutketen naar de dataclass herleid.
"""
import ast
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "psgscoring"

#: Welk Profile-attribuut welke geneste dataclass draagt. De renderer in
#: constants.py leest `profile.<attr>.<veld>` of een lokale alias daarvan.
_PROFILE_ATTR_KLASSE = {
    "hypopnea":        "HypopneaRules",
    "apnea":           "ApneaRules",
    "spo2":            "SpO2Rules",
    "post_processing": "PostProcessingRules",
}

#: Velden die per definitie geen gedragsconsument hebben: naamgeving,
#: documentatie, herkomst, en de dataclasses die de andere dragen.
BESCHRIJVEND = {
    "name", "display_name", "description", "citation", "aasm_rule",
    "aasm_version", "family", "deprecated", "deprecated_alias_for",
    "output_variants", "hypopnea", "apnea", "spo2", "post_processing",
}

#: BEKENDE SCHULD, geïnventariseerd 31-08-2026 en hersleuteld 11-10-2026.
#: Elk veld hier staat in de registry zonder dat iets het leest. Ze staan er
#: niet omdat het goed is.
#:
#: Notatie: "Dataclass.veld". Een kale veldnaam zonder punt wordt nog
#: geaccepteerd en slaat dan op ELKE dataclass die dat veld draagt -- dat is
#: de oude, grovere notatie; gebruik hem alleen als dat echt bedoeld is.
ZONDER_CONSUMENT = {
    # `unsure_as_hypopnea` en `HypopneaRules.sensor` STONDEN hier. Ze zijn op
    # 31-08-2026 VERWIJDERD in plaats van bedraad, want ze beweerden allebei
    # iets onwaars in plaats van iets ongebruikts:
    #   * `sensor` dupliceerde `flow_fallback_strategy`, en bij chicago_1999
    #     spraken de twee elkaar tegen (`nasal_pressure_or_flow` naast
    #     strategie `none`);
    #   * `unsure_as_hypopnea` beschreef hoe de NSRR-ANNOTATIE gelezen moet
    #     worden, niet hoe wij scoren -- dat hoort in het meetharnas, waar het
    #     ook al geïmplementeerd is.
    # Wat overblijft is `ApneaRules.sensor`, die niet varieert.
    "ApneaRules.sensor",
    # De >=10 s-eis, in BEIDE dataclasses. De detectoren lezen de
    # MODULECONSTANTE; `pipeline.py` doet `profile.get("HYPOPNEA_MIN_DUR_S",
    # 10.0)` op een sleutel die niet in de dict zit, dus die get faalt altijd
    # naar de default.
    "HypopneaRules.min_duration_s",
    "ApneaRules.min_duration_s",
    "HypopneaRules.arousal_required",
    "HypopneaRules.nasal_pressure_fallback",
    # Superseded door `hypopnea_force_linearisation`, die wel bedraad is.
    "HypopneaRules.square_root_linearisation",
    "SpO2Rules.global_p95_fallback",
    # Het filter is alleen uit te zetten via cv=0, niet via deze vlag.
    "PostProcessingRules.stability_filter_enabled",
    "PostProcessingRules.local_baseline_validation",
    # Draait op `sf_effort > 0`, niet op deze vlag.
    "PostProcessingRules.mixed_apnea_decomposition",
    # 11-10-2026: de twee die zich achter een naamgenoot verscholen (zie de
    # moduledocstring). Allebei expliciet gepind door een profiel op precies
    # de waarde die de code toch al hard aanhoudt, zodat de pin niets deed en
    # ook niets kon doen. De pins zijn verwijderd; de velden blijven staan tot
    # hun consument ze leest.
    #
    # TODO consument: respiratory.py leest de moduleconstante
    # `APNEA_THRESHOLD = 0.10` (= 1 - 0.90). Renderen als `APNEA_THRESHOLD`
    # in constants.py en daar lezen, naast de al bestaande
    # `APNEA_THRESHOLD_THERMISTOR`. Bij de defaults byte-identiek.
    "ApneaRules.flow_reduction_threshold",
    # TODO consument: spo2.py hard-codeert het pre-event-venster van 120 s op
    # twee plekken (`get_desaturation`, regel ~87, en de ODI-lus, regel ~430).
    # Renderen als `SPO2_BASELINE_WINDOW_S` en daar doorgeven. Bij de default
    # byte-identiek.
    "SpO2Rules.baseline_window_s",
}


def _velden_per_klasse() -> dict:
    """{(klasse, veld)} voor elke geannoteerde dataclass-attribuut in profiles.py."""
    velden = set()
    boom = ast.parse((PKG / "profiles.py").read_text(encoding="utf-8"))
    for n in ast.walk(boom):
        if isinstance(n, ast.ClassDef):
            for x in n.body:
                if isinstance(x, ast.AnnAssign) and isinstance(x.target, ast.Name):
                    velden.add((n.name, x.target.id))
    return velden


def _keten(node):
    """Attribuutketen als lijst van namen, of None als het geen keten is."""
    delen = []
    while isinstance(node, ast.Attribute):
        delen.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        delen.append(node.id)
        return list(reversed(delen))
    return None


def _sleutels_per_veld() -> dict:
    """{(klasse, veld): {LEGACY_SLEUTEL, ...}} uit de renderer in constants.py.

    Leest `_profile_to_legacy_dict` met de AST: lokale aliassen
    (`h = profile.hypopnea`, `pp = profile.post_processing`, of een lokale die
    rechtstreeks een veld vasthoudt) worden herleid, en elke attribuutketen in
    een waarde van de teruggegeven dict wordt op zijn dataclass gesleuteld.
    """
    boom = ast.parse((PKG / "constants.py").read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(boom)
               if isinstance(n, ast.FunctionDef)
               and n.name == "_profile_to_legacy_dict"), None)
    assert fn is not None, "constants.py heeft geen _profile_to_legacy_dict meer"
    profiel_param = fn.args.args[0].arg

    # Alias -> dataclass-object ("pp" -> "PostProcessingRules"), en
    # alias -> (klasse, veld) voor een lokale die al een veld vasthoudt.
    object_alias = {profiel_param: "Profile"}
    veld_alias: dict = {}

    def herleid(node):
        """(klasse, veld) voor een attribuutketen of een veld-alias, anders None."""
        if isinstance(node, ast.Name):
            return veld_alias.get(node.id)
        k = _keten(node)
        if not k or k[0] not in object_alias:
            return None
        basis = object_alias[k[0]]
        rest = k[1:]
        if basis == "Profile" and len(rest) >= 2 and rest[0] in _PROFILE_ATTR_KLASSE:
            return (_PROFILE_ATTR_KLASSE[rest[0]], rest[1])
        if len(rest) >= 1:
            return (basis, rest[0])
        return None

    for st in fn.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                and isinstance(st.targets[0], ast.Name):
            doel = st.targets[0].id
            k = _keten(st.value)
            if k and k[0] in object_alias and len(k) == 2 \
                    and object_alias[k[0]] == "Profile" \
                    and k[1] in _PROFILE_ATTR_KLASSE:
                object_alias[doel] = _PROFILE_ATTR_KLASSE[k[1]]
                continue
            kv = herleid(st.value)
            if kv is not None:
                veld_alias[doel] = kv

    uit: dict = {}
    for ret in (n for n in ast.walk(fn) if isinstance(n, ast.Return)):
        if not isinstance(ret.value, ast.Dict):
            continue
        for sleutel, waarde in zip(ret.value.keys, ret.value.values):
            if not (isinstance(sleutel, ast.Constant) and isinstance(sleutel.value, str)):
                continue
            for sub in ast.walk(waarde):
                kv = herleid(sub) if isinstance(sub, (ast.Attribute, ast.Name)) else None
                if kv is not None:
                    uit.setdefault(kv, set()).add(sleutel.value)
    return uit


def _ontvouw(entry: str, velden: set) -> set:
    """"Klasse.veld" -> {(Klasse, veld)}; kale "veld" -> alle klassen die het dragen."""
    if "." in entry:
        kl, v = entry.split(".", 1)
        return {(kl, v)}
    return {kv for kv in velden if kv[1] == entry}


@pytest.fixture(scope="module")
def analyse():
    velden = _velden_per_klasse()
    sleutels = _sleutels_per_veld()

    bron = {p.name: p.read_text(encoding="utf-8") for p in PKG.glob("*.py")}

    def consument(kv):
        klasse, veld = kv
        ks = sleutels.get(kv, set())
        for f, txt in bron.items():
            if f in ("profiles.py", "constants.py"):
                continue
            # Attribuuttoegang buiten de renderer. Hier is de dataclass niet
            # uit een regex te herleiden; dit blijft dus op veldnaam, en is
            # daarmee de MILDE kant van de poort.
            if re.search(rf"\.{veld}\b", txt):
                return f
            for k in ks:
                if f'"{k}"' in txt or f"'{k}'" in txt:
                    return f
        return None

    zonder = {kv for kv in velden
              if not kv[1].startswith("_") and kv[1] not in BESCHRIJVEND
              and consument(kv) is None}
    toegestaan = set()
    for e in ZONDER_CONSUMENT:
        toegestaan |= _ontvouw(e, velden)
    return velden, zonder, toegestaan, sleutels


def _fmt(kvs):
    return sorted(f"{k}.{v}" for k, v in kvs)


def test_geen_nieuw_veld_zonder_consument(analyse):
    _velden, zonder, toegestaan, _ = analyse
    nieuw = zonder - toegestaan
    assert not nieuw, (
        "deze profielvelden worden door niets gelezen en staan niet in "
        "ZONDER_CONSUMENT. Een veld dat niemand leest is geen instelling maar "
        "een belofte: het zetten verandert niets, en dat ziet er in een meting "
        f"precies zo uit als 'geen effect'. Beslis expliciet: {_fmt(nieuw)}")


def test_de_lijst_bevat_geen_velden_die_inmiddels_wel_gelezen_worden(analyse):
    """Anders groeit de schuldenlijst en krimpt hij nooit."""
    _velden, zonder, toegestaan, _ = analyse
    opgelost = toegestaan - zonder
    assert not opgelost, (
        f"deze staan als schuld genoteerd maar hebben nu een consument; "
        f"haal ze uit ZONDER_CONSUMENT: {_fmt(opgelost)}")


def test_de_lijst_verwijst_alleen_naar_bestaande_velden(analyse):
    velden, _zonder, _toegestaan, _ = analyse
    spook = [e for e in ZONDER_CONSUMENT if not _ontvouw(e, velden)]
    assert not spook, f"ZONDER_CONSUMENT noemt niet-bestaande velden: {sorted(spook)}"


def test_de_gewired_velden_van_vannacht_staan_er_niet_meer_in(analyse):
    """`csr_reclassification` en `desat_global_baseline_min_local_pct` waren
    dezelfde fout en zijn op 31-08 bedraad."""
    _velden, zonder, _toegestaan, _ = analyse
    namen = {v for _k, v in zonder}
    for veld in ("csr_reclassification", "desat_global_baseline_min_local_pct",
                 "arousal_min_interval_s", "event_gap_tolerance_breaths"):
        assert veld not in namen, f"{veld} heeft weer geen consument"


# ══════════════════════════════════════════════════════════════════════
#  De poort sleutelt op de dataclass, niet op de naam (11-10-2026)
# ══════════════════════════════════════════════════════════════════════

def test_de_poort_onderscheidt_naamgenoten_in_verschillende_dataclasses(analyse):
    """Twee velden met dezelfde naam zijn twee velden.

    `HypopneaRules.flow_reduction_threshold` voedt HYPOPNEA_THRESHOLD en heeft
    een consument; `ApneaRules.flow_reduction_threshold` voedt niets -- de
    apneudrempel is de moduleconstante APNEA_THRESHOLD in respiratory.py.
    Idem `PostProcessingRules.baseline_window_s` (BASELINE_WINDOW_S, de
    flow-basislijn) tegenover `SpO2Rules.baseline_window_s` (spo2.py houdt
    120 s hard aan). Vóór 11-10-2026 zag de poort het verschil niet.
    """
    _velden, zonder, _toegestaan, sleutels = analyse
    assert ("HypopneaRules", "flow_reduction_threshold") not in zonder
    assert ("ApneaRules", "flow_reduction_threshold") in zonder
    assert ("PostProcessingRules", "baseline_window_s") not in zonder
    assert ("SpO2Rules", "baseline_window_s") in zonder
    # En de renderer-analyse zelf: de hypopneedrempel is herleid tot zijn
    # eigen dataclass, niet tot "het eerste veld met die naam".
    assert "HYPOPNEA_THRESHOLD" in sleutels[("HypopneaRules", "flow_reduction_threshold")]
    assert ("ApneaRules", "flow_reduction_threshold") not in sleutels
    assert "BASELINE_WINDOW_S" in sleutels[("PostProcessingRules", "baseline_window_s")]
    assert ("SpO2Rules", "baseline_window_s") not in sleutels


def test_de_twee_max_duration_velden_hebben_elk_een_eigen_consument(analyse):
    """`HypopneaRules.max_duration_s` -> HYPOPNEA_MAX_DUR_S en
    `ApneaRules.max_duration_s` -> APNEA_MAX_DUR_S; allebei gelezen door
    respiratory.py (splitsing van te lange events). Dit is de tegenhanger van
    de vorige test: naamgenoten die WEL allebei leven."""
    _velden, zonder, _toegestaan, sleutels = analyse
    assert "HYPOPNEA_MAX_DUR_S" in sleutels[("HypopneaRules", "max_duration_s")]
    assert "APNEA_MAX_DUR_S" in sleutels[("ApneaRules", "max_duration_s")]
    assert ("HypopneaRules", "max_duration_s") not in zonder
    assert ("ApneaRules", "max_duration_s") not in zonder


def test_de_misleidende_pins_op_dode_velden_zijn_weg():
    """`aasm_v3_sensitive` pinde `apnea.flow_reduction_threshold=0.90` en
    `aasm_v3_strict` pinde `spo2.baseline_window_s=120.0`: allebei exact de
    default, allebei door niets gelezen. Een pin op een dode knop suggereert
    een keuze die niet bestaat. De bron mag ze niet meer expliciet zetten;
    de WAARDE blijft de default, dus het profiel rendert byte-identiek."""
    bron = (PKG / "profiles.py").read_text(encoding="utf-8")
    sens = bron[bron.index("_aasm_v3_sensitive = Profile("):]
    sens = sens[:sens.index("\n)\n")]
    strict = bron[bron.index("_aasm_v3_strict = Profile("):]
    strict = strict[:strict.index("\n)\n")]
    assert not re.search(r"^\s*flow_reduction_threshold=0\.90?,", sens, re.M), (
        "aasm_v3_sensitive pint nog apnea.flow_reduction_threshold")
    assert not re.search(r"^\s*baseline_window_s=120\.0?,", strict, re.M), (
        "aasm_v3_strict pint nog spo2.baseline_window_s")
    from psgscoring.profiles import get_profile
    assert get_profile("aasm_v3_sensitive").apnea.flow_reduction_threshold == 0.90
    assert get_profile("aasm_v3_strict").spo2.baseline_window_s == 120.0


# ══════════════════════════════════════════════════════════════════════
#  De twee die niet sliepen maar logen
# ══════════════════════════════════════════════════════════════════════

def test_hypopnea_regels_dragen_geen_sensorveld_meer():
    """`HypopneaRules.sensor` dupliceerde `flow_fallback_strategy`.

    Erger dan ongebruikt: bij `chicago_1999` spraken ze elkaar tegen. Het veld
    zei `nasal_pressure_or_flow` -- dus met terugval -- terwijl de bedrade
    strategie `none` was. Eén profiel, twee tegengestelde uitspraken over
    dezelfde vraag, en alleen de tweede deed iets.
    """
    from psgscoring.profiles import get_profile
    assert not hasattr(get_profile("aasm_v3_rec").hypopnea, "sensor")


def test_de_sensorkeuze_leeft_in_de_bedrade_vlag():
    """Wat er wél is, en dat het onderscheid nog steeds draagt."""
    from psgscoring.profiles import get_profile
    assert get_profile("mesa_shhs").post_processing.flow_fallback_strategy == \
        "ripsum_on_nasal_failure"
    assert get_profile("aasm_v3_rec").post_processing.flow_fallback_strategy == "none"


def test_unsure_as_hypopnea_is_geen_profielveld_meer():
    """Het beschreef de NSRR-ANNOTATIE, niet onze scoring.

    Een profiel zegt hoe wíj scoren; hoe een dataset zijn labels bedoelt hoort
    in het harnas dat die dataset leest. `validate_mesa.py` implementeert het
    daar ook al.
    """
    from psgscoring.profiles import get_profile
    assert not hasattr(get_profile("mesa_shhs").post_processing,
                       "unsure_as_hypopnea")


def test_de_nsrr_conventie_is_niet_verloren():
    """Verwijderen mag geen kennis weggooien -- inclusief de open tegenspraak."""
    import inspect

    from psgscoring import profiles
    bron = inspect.getsource(profiles)
    assert "HYPOPNEA_CONCEPTS" in bron, "de verwijzing naar het harnas ontbreekt"
    assert ">50 %" in bron and ">=30 %" in bron, (
        "de openstaande tegenspraak over wat `Unsure` betekent is niet vastgelegd")
