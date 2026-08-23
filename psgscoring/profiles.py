"""
psgscoring.profiles
====================

Scoring profile definitions for psgscoring v0.4.0.

This module provides a unified framework for running PSG respiratory
event scoring under multiple historical and dataset-specific rule sets.

Profile families:
    - clinical : AASM-compliant profiles (v1/v2/v3 + CMS + exploratory)
    - dataset  : Faithful reproduction of NSRR scoring conventions
    - legacy   : Pre-AASM Chicago criteria

Usage:
    >>> from psgscoring.profiles import get_profile, list_profiles
    >>> profile = get_profile("aasm_v3_rec")
    >>> profile.hypopnea.flow_reduction_threshold
    0.30

    >>> list_profiles()
    ['aasm_v3_rec', 'aasm_v3_strict', 'aasm_v3_sensitive',
     'aasm_v2_rec', 'aasm_v1_rec', 'cms_medicare',
     'mesa_shhs', 'chicago_1999']

References:
    Iber et al. 2007 (v1) - AASM Manual 1st edition
    Berry et al. 2012 (v2.0) - Sleep Apnea Definitions Task Force update
    Berry et al. 2020 (v2.6) - Last v2 edition
    Troester et al. 2023 (v3) - Current mandated standard
    MESA Scoring Manual (Brigham Reading Center) - NSRR convention
    AASM Task Force 1999 - Chicago criteria (pre-AASM)

License: BSD-3-Clause
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional

__all__ = [
    "HypopneaRules",
    "ApneaRules",
    "SpO2Rules",
    "PostProcessingRules",
    "Profile",
    "get_profile",
    "list_profiles",
    "list_profile_groups",
    "resolve_profile_name",
    "PROFILES",
    "PROFILE_GROUPS",
    "LEGACY_ALIASES",
]


# ============================================================
# Data classes
# ============================================================

@dataclass
class HypopneaRules:
    """Hypopnea detection parameters."""

    flow_reduction_threshold: float
    """Minimum flow amplitude reduction (0.30 = 30% drop)."""

    sensor: str
    """Primary sensor for flow reduction measurement.
    Values: 'nasal_pressure', 'rip_bands_primary', 'nasal_pressure_or_flow'
    """

    min_duration_s: float = 10.0
    max_duration_s: float = 60.0

    desat_threshold: Optional[float] = None
    """SpO2 desaturation threshold (0.03 = 3%). None = no desat coupling."""

    desat_required: bool = True
    """If True, desat is mandatory for event qualification."""

    arousal_required: bool = False
    """If True, arousal is mandatory for event qualification."""

    desat_or_arousal: bool = False
    """If True, desat OR arousal is sufficient (AASM Rule 1A behaviour)."""

    nasal_pressure_fallback: bool = False
    """For dataset profiles: fall back to nasal pressure if RIP unreliable."""

    square_root_linearisation: bool = True
    """Apply Bernoulli correction to nasal pressure (standard practice)."""

    output_variants: List[str] = field(default_factory=list)
    """For dataset profiles: emit multiple AHI variants per detection pass."""


@dataclass
class ApneaRules:
    """Apnea detection parameters."""

    flow_reduction_threshold_thermistor: float | None = None
    """Aparte apneudrempel wanneer de THERMISTOR de apneus draagt.

    `None` = geen apart getal, dus `flow_reduction_threshold` voor beide
    sensoren. Dat is het gedrag van vóór 22-08-2026 en blijft de default.

    Waarom dit veld bestaat. De twee sensoren leven op verschillende schalen.
    Gemeten op 597 door mensen gescoorde apneus (NSRR-annotatie, niet onze
    detector): de neusdruk zakt mediaan 89,6 %, de thermistor 80,3 %, en van
    diezelfde apneus haalt 50 % de drempel van 0,90 op de druk tegen 13 % op
    de thermistor. De neusdruk vergroot de daling uit -- het signaal loopt
    ongeveer met het kwadraat van de flow -- en de AASM schrijft de thermische
    opnemer juist voor omdat de druk overdrijft.

    Het is geen informatieverschil: de AUC apneu-vs-hypopneu is gelijk
    (gepaard +0,009, binnen een vooraf vastgelegde band van 0,03). Eén drempel
    kan alleen niet beide schalen lezen.

    Zie docs/apneudrempel_sensorafhankelijk_bevinding.md en de bijbehorende
    preregistratie.
    """

    flow_reduction_threshold: float = 0.90
    sensor: str = "thermistor"
    min_duration_s: float = 10.0
    max_duration_s: float = 90.0


@dataclass
class SpO2Rules:
    """SpO2 coupling and baseline parameters."""

    baseline_window_s: float = 120.0
    nadir_search_s: float = 45.0
    global_p95_fallback: bool = True


@dataclass
class PostProcessingRules:
    """Post-processing and bias-correction parameters."""

    stability_filter_enabled: bool = True
    stability_filter_cv: float = 0.45
    """Coefficient of variation threshold for rejecting normal variability."""

    csr_reclassification: bool = True
    """Reclassify events as central based on Cheyne-Stokes periodicity."""

    local_baseline_validation: bool = True
    """Reject candidates driven solely by inflated rolling baseline."""

    flow_smoothing_s: float = 0.0
    """Flow envelope smoothing window (0 = off, default since v0.2.8).
    Was 3.0 prior to v0.2.8; removed because it caused +54 false hypopneas
    in PSG-IPA SN1 (severity drift Mild → Moderate)."""

    breath_level_detection: bool = True
    """Enable peak-based breath-level hypopnea detection."""

    peak_min_consecutive_breaths: int = 3
    """Minimum consecutive low-amplitude breaths for peak-based detection.
    Lowered to 2 in sensitive profile for UARS sensitivity."""

    artefact_flank_exclusion: bool = True

    mixed_apnea_decomposition: bool = True
    """Split mixed apneas into central + obstructive components."""

    unsure_as_hypopnea: bool = False
    """NSRR-specific: 'Unsure' tag = hypopnea with >50% reduction."""

    hypopnea_detector: str = "envelope"
    """v0.12.3+: welke hypopnee-detector draait.

    "envelope"       — bestaande pijplijn (default, ongewijzigd gedrag)
    "breath_graded"  — psgscoring.breath_scoring: ademteug als atoom,
                       patiëntkalibratie in twee passages, gegradeerde
                       AASM-predicaten en een kans per event.
    """

    hypopnea_strictness: float = 0.50
    """Operatiepunt op de kans-as van de gegradeerde detector. Lager =
    soepeler. Eén as in plaats van drie parametercombinaties."""

    hypopnea_force_linearisation: bool = False
    """Pas de wortellinearisatie (AASM Regel 3) ook toe wanneer hypopnee- en
    apneukanaal HETZELFDE kanaal zijn.

    Zonder dit veld hangt de voorbewerking af van de montage, niet van het
    profiel. Bij twee flowkanalen wordt de hypopnee-envelope uit de neusdruk
    berekend mét linearisatie; bij één kanaal hergebruikt de code de
    apneu-envelope, die zonder linearisatie is berekend. Dezelfde profielnaam
    draait dan een andere voorbewerking op PSG-IPA (één kanaal) dan op MESA
    (`Pres` + `Therm`).

    Dat verschil is niet neutraal: zonder linearisatie meet een werkelijke
    flowreductie van 50 % als 75 % amplitudereductie, dus reducties worden
    stelselmatig overschat en meer kandidaten halen het 30 %-criterium. Een
    operatiepunt dat op de ongelineariseerde conventie is gekalibreerd, is op
    de gelineariseerde te streng.

    GEMETEN (12-08-2026, PSG-IPA, zie CHANGELOG v0.16.0): het effect zit
    volledig in de regelcascade (`aasm_v3_rec`: bias +1,77 → −1,43, MAE
    1,84 → 1,43, binnen scoorderrange 3/5 → 5/5, hypopneus 208 → 116) en is op
    de gegradeerde tak **nul**. `score_hypopneas_breathwise` krijgt het RUWE
    signaal via `_run_breath_analysis` en filtert zelf, dus het ziet de
    envelope waarin de linearisatie leeft nooit. Deze asymmetrie verklaart
    daarom NIET de tegengestelde bias-richting van `aasm_v3_breath`/`_prob` op
    de twee cohorten — ze blijft een aannemelijke bijdrage voor de
    cascadeprofielen en is op zichzelf een defect.

    Default `False` = bestaand gedrag; elk bestaand profiel blijft
    byte-identiek. Aanzetten is GEEN losse wijziging: het verschuift de
    amplitudeschaal waarop `hypopnea_strictness` is geijkt, dus linearisatie
    aanzetten en strictness opnieuw afleiden zijn één experiment. Zie de
    CHANGELOG bij v0.16.0."""

    desat_low_baseline_relaxation: bool = False
    """Verschuif het desaturatiecentrum van 3 % naar 2 % bij een lage baseline.

    **Dit is een AFWIJKING van de AASM-manual, geen reparatie.** Regel 1A eist
    ≥ 3 %. Deze optie versoepelt dat, en mag daarom nooit stilzwijgend aanstaan.

    **Waarom hij bestaat.** Bij een baseline-SpO₂ van 85 % is de fysiologische
    ruimte voor een daling van 3 % beperkt: de dissociatiecurve is daar steil
    en een gelijke ventilatiedaling levert een kleinere procentuele dip op dan
    bij een baseline van 96 %. Een vaste eis onderschat de belasting dan
    stelselmatig — het probleem dat §S11.3 beschrijft.

    **Hoe, gegradeerd in plaats van binair.** Het CENTRUM van de
    desaturatie-sigmoid schuift van 3 % naar 2 %, uitsluitend voor events
    waarvan de lokale (2 min) baseline zelf onder 92 % ligt. De
    breedte van de sigmoid blijft gelijk. Er komt dus geen nieuwe mechaniek
    bij: het blijft één gegradeerde term in dezelfde conjunctie, alleen met een
    ander centrum. De binaire variant zou een tweede drempel introduceren en
    daarmee precies de trapfunctie terugbrengen die de gegradeerde detector
    vermijdt.

    Elk event dat hierdoor kwalificeert draagt `low_baseline_relaxed: true` in
    zijn criteria-dict, zodat het rapport bij de AHI-regel kan vermelden dat
    dit profiel van Regel 1A afwijkt — hetzelfde patroon als de bestaande
    experimenteel-markering.

    **De conditie is één keer herzien, op grond van een meting.** De
    oorspronkelijke specificatie las "pre-event-saturatie onder de lokale
    2-minuten-baseline". Zo geformuleerd is ze vacuum waar — de baseline is een
    90e percentiel en de pre-event-saturatie een mediaan uit datzelfde venster
    — en ze vuurde op PSG-IPA op **466 van de 466 events**. Dat zou de
    3 %-eis blanco naar 2 % hebben verlaagd voor iedereen, precies niet wat de
    optie beoogt. De conditie volgt nu de motivering: de baseline zelf laag
    (< 92 %), want daar is de dissociatiecurve steil.

    **Nooit default aan**, ook niet later, zonder een aparte beslissing die
    expliciet over de regelafwijking gaat en niet over de meetuitkomst.

    **Herkomst: CAISR-resp** (Nasiri et al., *Sleep* 48(8):zsaf134, 2025), dat
    een 2 %-daling als zachte 3 % telt wanneer de saturatie al onder de
    2-minuten-baseline ligt. Hun regel is binair; deze is gegradeerd. Zie
    `docs/third_party_comparison.md`."""

    split_events_longer_than_s: float | None = None
    """Splits events langer dan deze drempel op fysiologische ankers.

    De AASM kent geen bovengrens voor eventduur, maar een flowreductie-envelop
    van 90 s die drie desaturaties overspant is klinisch drie events, geen één.
    Op Cheyne-Stokes-nachten kan één envelop meerdere cycli beslaan.

    Ankers, in deze volgorde: de onsets van desaturaties binnen het event; is
    er geen desaturatie, dan arousal-onsets; is er geen van beide, dan blijft
    het event heel. Een anker in de eerste of laatste 10 s wordt genegeerd,
    anders ontstaat een fragment dat per constructie onder de duurdrempel valt.

    Elk deel gaat opnieuw door de duurtoets. Delen die daardoor afvallen gaan
    naar `rejected_hypopneas` (of de afgewezen apneus) met reden
    `split_fragment` — ze worden niet weggegooid, zodat de reviewer ze ziet.
    Elk deel draagt `split_from` en `split_anchor` in zijn criteria-dict.

    Subtype: elk deel erft het label van het oorspronkelijke event. Geen
    hersubtypering per fragment in deze ronde — dat zou de effortclassificatie
    op steeds kortere vensters draaien en dat is een aparte vraag.

    Default `None` = geen splitsing = bestaand gedrag; elk bestaand profiel
    blijft byte-identiek.

    **Herkomst: CAISR-resp** (Nasiri et al., *Sleep* 48(8):zsaf134, 2025),
    dat events > 60 s splitst op arousals of desaturaties. Alleen het idee en
    de startwaarde komen over; de implementatie is eigen werk vanaf deze
    specificatie. Zie `docs/third_party_comparison.md`."""

    event_boundaries: str = "envelope"
    """Waar de event-grenzen vandaan komen: `"envelope"` of `"breath"`.

    De cascadedetector legt een grens waar de envelope zijn drempel kruist —
    op de dalende flank, dus vóór de eerste duidelijk gereduceerde ademteug.
    Een menselijke scoorder markeert die ademteug zelf. Gemeten op PSG-IPA
    (blok 1B, 14-08-2026, gemiddelde over vijf opnames van de mediane
    getekende afwijking algoritme − scoorder):

    - apneus: onset +0,80 / −0,10 / −0,40 / +0,55 s — binnen de menselijke
      interkwartielafstand, dus in orde.
    - hypopneus op `aasm_v3_rec`: onset −3,30 / −5,30 / −5,15 / −3,15 / −2,10 s
      — dezelfde richting op alle vijf opnames, vier buiten de menselijke IQR.
      Systematische offset, niet ruis.
    - hypopneus op `aasm_v3_breath`: de helft daarvan, en de EINDEN vallen op
      alle vijf binnen de menselijke IQR. Die detector werkt al per ademteug;
      dat hij dichterbij landt is de bevestiging van de diagnose.

    `"breath"` snapt elke grens naar de dichtstbijzijnde ademteuggrens uit de
    bestaande breath-detector. **Nul gefitte parameters** — geen
    naschuifconstante, want die zou op deze vijf opnames gefit zijn en niet
    generaliseren. De ademteuggrenzen zijn het fysiologische anker zelf.

    Default `"envelope"` = bestaand gedrag; elk bestaand profiel blijft
    byte-identiek. Aanzetten beweegt duren → tellingen → AHI → elk
    koppelvenster, en breekt de golden-baseline per ontwerp. Zie CHANGELOG."""

    stability_filter_all_hypopnea_subtypes: bool = True
    """Laat het stabiele-ademhalingsfilter ook op gesubtypeerde hypopneus slaan.

    Het filter vergelijkt het eventtype EXACT met `"hypopnea"`, terwijl
    hypopneus een subtype dragen: `hypopnea` (obstructief), `hypopnea_central`,
    `hypopnea_mixed`, `hypopnea_uncertain`. Alles wat niet obstructief is,
    ontsnapt er dus aan. De rest van de codebase telt hypopneus juist op
    substring (`"hypopnea" in e["type"]` in `_compute_summary`).

    Hoeveel dat uitmaakt hangt af van het cohort, en daarmee van iets wat niets
    met de patiënt te maken heeft. Gemeten 12-08-2026 op `aasm_v3_rec`:

    - PSG-IPA: 200 van 208 hypopneus dragen het kale label (96,2 %), het filter
      wijst er 190 af.
    - MESA met de RIP-poort dicht: elke hypopnee heet `hypopnea_uncertain`, het
      filter wijst er NUL af.

    De keten loopt van de eenhedendeclaratie in het EDF via de RIP-poort naar
    het subtypelabel naar de vraag of dit filter überhaupt draait. Zie
    `rip_quality_scale_free` en `docs/rip_poort_reparatie_20260812.md`.

    Of de beperking tot obstructieve hypopneus ooit bedoeld was, valt uit de
    code niet op te maken: de subtypelabels bestonden al toen het filter
    geschreven werd (v0.2.2 tegen v0.2.8), en het criterium zelf — de
    variatiecoëfficiënt van de ademamplitude — heeft niets met het subtype te
    maken. Daarom een vlag en geen stille correctie.

    **Default `True` sinds 13-08-2026** (gebruikersbesluit). Het filter dekt nu
    alle hypopneus, ongeacht subtype, zodat de dekking niet meer afhangt van of
    de effortclassificatie toevallig lukte. `mesa_shhs` en `chicago_1999`
    blijven expliciet op `False` voor de reproduceerbaarheid van paper
    v31/v37 respectievelijk de historische Chicago-criteria."""

    single_channel_rhythm: bool = True
    """Beslis de eenkanaalsfallback op RITMIEK in plaats van alleen amplitude.

    Draait de effortclassificatie op één band, dan kent
    `single_channel_fallback_classify` maar EEN as: de mediane envelope tijdens
    het event tegen de P75 van de basislijn. Onder 0,20 centraal, boven 0,50
    obstructief, daartussen `uncertain` -- en die kale `uncertain` vallen
    buiten `ahi_total`.

    Die as gooit twee verschillende dingen op een hoop. Een band die KLEINER
    wordt: bij obstructie bewegen thorax en abdomen in tegenfase, dus het
    volume verplaatst zich en een enkele band zakt terwijl de inspanning
    doorgaat. En een band die AFWEZIG is: bij een centrale apneu ligt hij stil.
    Een scorer scheidt die twee moeiteloos, want die kijkt of de band nog
    meebeweegt op ademfrequentie.

    De fout is systematisch. Obstructief levert verminderde-maar-aanwezige
    inspanning en belandt in de dode zone; centraal levert een vlakke band en
    wordt met vertrouwen gelabeld. De regel is dus scherp waar hij centraal
    moet zeggen en vaag waar hij obstructief moet zeggen -- in een populatie
    waar obstructief overheerst wordt de meest voorkomende diagnose het vaakst
    weggegooid. Gemeten op een klinische opname met de paarpoort actief: 89
    centraal, 23 uncertain, 9 obstructief.

    Met `True` telt `detect_breaths` hoeveel van de uit de basislijn verwachte
    ademhalingen er tijdens het event nog zijn. Vanaf 50 % obstructief, onder
    15 % centraal, daartussen beslist de oude amplituderegel. Cyclustellen is
    schaalvrij, dus dit hangt niet af van de versterking van de band.

    **Default `True` sinds 19-08-2026.** Twee cohorten, twee referenties,
    hetzelfde teken. PSG-IPA met twaalf scorers per opname: 0,586-0,592
    tegen 0,609 bilateraal, waar de amplituderegel op 0,140-0,199 blijft.
    MESA tegen NSRR, gepaard over 12 opnames: mediaan delta +0,236
    (borstband) en +0,336 (buikband), p=0,002 beide. Het aandeel
    `uncertain` daalt van 15-19 % naar 2-5 %; die events vielen buiten
    `ahi_total`, dus dit verhoogt ook de gerapporteerde index.

    De historische profielen (`aasm_v2_rec`, `aasm_v1_rec`) en
    `cms_medicare` blijven expliciet op `False`: die bootsen oudere of
    regulatoire regelsets na. `mesa_shhs` en `chicago_1999` idem, voor de
    reproduceerbaarheid van paper v31/v37."""

    hypoxic_burden_cap_at_next_event: bool = False
    """Kap de integratie van elk event af bij de onset van het volgende.

    De herstelzoektocht loopt tot 120 s ná het eventeinde als de saturatie niet
    terugkeert tot `basislijn - 1 %`. Bij geclusterde events overlappen die
    vensters -- 73 van de 114 op mesa-sleep-1374 -- en telt dezelfde hypoxemie
    meermaals mee. Het aandeel events dat de 120 s-grens haalt verklaart de
    afwijking t.o.v. de gepubliceerde definitie met r = 0,89.

    Default `False`: dit verzet een gerapporteerde grootheid. Zie
    docs/hypoxic_burden_venster_preregistratie.md."""

    arousal_eog_reject: bool = False
    """Verwerp occipitaal-only arousals die samenvallen met een oogbeweging.

    Bedoeld als menselijke scoorder-heuristiek: een EOG-doorslag naar de
    occipitale elektroden lijkt op een arousal maar is er geen. Alleen actief
    in multi-derivatiemodus.

    **Default False sinds 22-08-2026, na meting.** Op de vijf PSG-IPA
    arousal-opnames, met de hybride aan: nul keer beter dan multi zonder
    reject, twee keer identiek (SN1, SN2) en drie keer slechter (SN3 0,390 ->
    0,371; SN4 0,568 -> 0,557; SN5 0,678 -> 0,665). De spreiding van de index
    verslechtert van 1,97 naar 2,16, en op SN3 zakt de recall van 0,383 naar
    0,350. Hij verwijdert dus echte events zonder er iets voor terug te geven.

    Stond tot dan default AAN zodra er een EOG-kanaal was. Zie
    docs/arousal_derivatie_preregistratie.md.

    Env-override: `PSGSCORING_AROUSAL_EOG_REJECT=1` zet hem terug aan."""

    arousal_lgbm_threshold: float | None = 0.80
    """Werkpunt van de arousal-classifier. `None` = de moduleconstante (0,60).

    **0,60 is gedomineerd.** De waarde is nooit onafhankelijk geijkt: de veeg
    die hem bevestigde liep op dezelfde vijf PSG-IPA-opnames waarop
    gerapporteerd werd, in de single-derivatie configuratie die niet meer
    draait.

    Herijkt op 23-08-2026 met gescheiden steekproeven (kalibratie MESA n=15,
    validatie MESA n=30, overlap 0), multi + hybride, artefactlijst genegeerd:

    | | MESA-kal. F1 | MESA-val. F1 | PSG-IPA F1 | telling/referentie |
    |---|---:|---:|---:|---|
    | 0,60 | 0,493 | 0,421 | 0,505 | 1,52 / 1,47 |
    | 0,80 | 0,562 | — | 0,514 (5/5 beter) | **1,07 / 1,01** |
    | 0,90 | **0,607** | **0,543** | 0,478 (3/5) | 0,60–0,81 |

    Op de validatieset gaat 0,90 van 0,421 naar 0,543 (gepaard +0,091, 24/30,
    p = 1,5e-06) — meer winst dan de classifier zelf opleverde.

    De keuze tussen 0,80 en 0,90 is klinisch, niet statistisch. 0,90 wint op
    event-F1; 0,80 wint op beide cohorten tegen 0,60 én geeft een **zuivere
    arousal-index**, terwijl 0,90 er ruim een derde te weinig telt — en die
    index staat in het rapport.

    **Default blijft `None`** tot die keuze gemaakt is. Voorwaardelijk op
    `arousal_uses_artifact_epochs`: wordt de lijst tóch gebruikt, dan moet de
    ijking over. Zie docs/arousal_drempel_herijking_preregistratie.md.
    """

    arousal_uses_artifact_epochs: bool = False
    """Slaat de arousalstap de als artefact gevlagde epochs over?

    `True` = huidig gedrag: de epochs uit `artifact_epochs` doen niet mee aan
    arousaldetectie. `False` = de arousalstap negeert de lijst.

    **Gemeten, en niet nipt.** MESA n=30 gepaard, drie armen op dezelfde
    opnames (arousal-F1 tegen de NSRR-annotatie, IoU 0,20):

    | | F1 | precisie | recall | gevlagd |
    |---|---:|---:|---:|---:|
    | geen lijst | **0,421** | 0,364 | 0,597 | 0 % |
    | huidige regel (500 µV) | 0,338 | 0,305 | 0,370 | 19,9 % |
    | `yasa.art_detect` | 0,356 | 0,304 | 0,484 | 2,1 % |

    Geen lijst wint van beide op **30 van de 30** opnames (p = 1,7e-06), en op
    PSG-IPA repliceert het teken (0,505 tegen 0,484).

    Twee dingen maken dit meer dan een drempelkwestie. De huidige regel vlagt
    op een ABSOLUTE 500 µV en gooit daarmee mediaan 19,9 % van de nacht weg,
    tot 53,6 % op één opname -- dezelfde schaalfout als de RIP-poort, die met
    een absolute MAD-drempel in feite de EDF-eenheid mat. En `art_detect`,
    die maar 2,1 % weggooit, is even slecht: hij selecteert de epochs met de
    grootste variantie, en dat zijn juist de epochs waar arousals zitten.

    Onderdrukken is dus het probleem, niet de detector. Zie
    docs/arousal_artefactlijst_preregistratie.md en
    docs/artefactdetector_vervangen_preregistratie.md.

    **Default blijft `True`** tot de gebruiker beslist: hem omzetten verandert
    de arousal-index en via de RERA-koppeling de RDI op de vier
    `arousal_limb_wired`-profielen.
    """

    arousal_lgbm: bool = True
    """Filter arousal-kandidaten met het op MESA getrainde LightGBM-model.

    Het kandidaatstadium draait dan op ruime drempels (ratio 1,2 / abrupt 1,0)
    en de classifier gooit weg. Gemeten op PSG-IPA, extern voor dit model:
    mediane event-F1 0,182 -> 0,463 tegen 0,692 scoorder-onderling, mediane
    eventduur 4,0 -> 6,2 s tegen 8,3 s menselijk, en de spreiding van
    index_algo/index_scoorder over vijf opnames 10,13 -> 2,10.

    **Default True sinds 22-08-2026** (gebruikersbeslissing, na de meting).
    Op MESA n=150 gepaard: `aasm_v3_rec` identiek op alle 150 (dat profiel
    gebruikt de arousals niet voor scoring), `aasm_v3_breath` bias
    -5,13 -> -5,28 (+0,14, grens 1,00) en F1 0,510 -> 0,514 met 101 van 150
    beter. Looptijd 1,03x.

    UIT op twee groepen profielen:

    1. Die een externe regelset of gepubliceerde dataset-analyse reproduceren:
       `aasm_v2_rec`, `aasm_v1_rec`, `cms_medicare`, `mesa_shhs`,
       `chicago_1999`. Een ML-classifier maakt geen deel uit van AASM v1/v2 of
       de CMS-regels, en `mesa_shhs`/`chicago_1999` moeten paper v31/v37
       reproduceren.
    2. Waar arousals doorwerken in de RDI, dus waar `arousal_limb_wired` aan
       staat: `aasm_v3_breath`, `aasm_v3_prob` en hun `_dual`-varianten.
       Op MESA n=50 verschoof de classifier daar de RDI-ernstklasse op 14/50
       opnames (28 %) tegen 5/50 voor de AHI, en er is geen RERA-referentie om
       te bepalen welke RDI juister is: MESA annoteert er geen en PSG-IPA
       bevat er 3 in de hele manuele set, hoewel `^RERA$` in de scoringsfilter
       stond. Zie test_profiles.py voor de invariant.

    LET OP (gecorrigeerd 22-08-2026): eerder stond hier dat
    `_compute_rera_rdi()` de arousallijst RECHTSTREEKS leest en dat de RDI dus
    op ELK profiel met deze vlag beweegt. Dat klopt niet. De functie krijgt de
    GEFILTERDE lijst door: de poort in `pipeline.py` levert arousals alleen
    door als `arousal_limb_wired` en de limb-stap aan staan. Daarom is
    `aasm_v3_rec` aantoonbaar onbewogen -- daar is RDI gelijk aan AHI."""

    flow_gap_scale_free: bool = False
    """Beoordeel signaaluitval op de flow schaalvrij in plaats van absoluut.

    `_detect_signal_gaps` markeert uitval bij `|x| < 1e-5` of `diff == 0`.
    Beide falen op echte data. De eerste is ABSOLUUT en wordt op het RUWE
    signaal toegepast: gemeten op `mesa-sleep-0001` vuurt hij op **0,00 %** van
    de `Pres`-samples en **6,44 %** van de `Therm`-samples — dezelfde opname,
    alleen een andere amplitudeschaal. De tweede vuurt nergens, want echte
    ADC-data heeft geen exact gelijke opeenvolgende samples.

    Voor het kanaal waarop apneus gescoord worden is het mechanisme daarmee
    dood. Dat is niet onschuldig: een **losgeraakte of verschoven canule**
    ruist rond nul, en dan ís er per definitie geen ademhaling — dus scoort de
    detector aaneengesloten apneus. De maximumduur van 90 s begrenst het losse
    event, niet het aantal.

    Schaalvrij vergelijkt de LOKALE activiteit met de eigen typische activiteit
    van het kanaal. Zelfde reparatiepatroon als bij de RIP-poort, die met een
    absolute MAD-drempel in feite de EDF-eenheid mat.

    Op een gesimuleerde losgeraakte canule (ruisend bijna-nul, 60 s) vangt de
    absolute regel **0 %** en de schaalvrije **96,7 %**, bij nul vals-positieven
    daarbuiten; het blijft schoon van fractie 0,10 tot 0,30.

    **Default `False`.** De fractie 0,10 komt van een synthetische fixture en
    is dus VOORLOPIG: hij hoort op echte opnames geijkt te worden met het gat
    tussen aantoonbaar dode en werkende kanalen, met gescheiden steekproeven.
    Aanzetten neemt bovendien events weg, en dat verlaagt de AHI.
    """

    plm_event_list_cap: int = 200
    """Hoeveel beenbewegingen er hoogstens in `plm["events"]` meegaan.

    Een transportgrens, geen scoringsregel. Geen enkele index hangt er nog van
    af: `plm_index` en `n_plm` komen uit de volledige lijst, en sinds
    22-08-2026 rekent ook `plm_arousal_index` over de hele nacht (de pipeline
    vraagt `analyze_plm` om de ongekapte lijst en kapt pas na de afgeleide
    indices af). Wat de grens nog raakt is wat een lezer ZIET: de EDF+-export
    in YASAFlaskified schrijft `events[]` als annotaties, dus in een viewer
    stoppen de markeringen midden in de nacht.

    **200 is laag.** Menselijke scoorders markeren op PSG-IPA tot 1033
    bewegingen in een nacht (SN4, hoogste van 60 scoorder-opnames), en onze
    detector vindt er 660 op SN1. Bij 200 valt dus het merendeel van de
    markeringen uit de export.

    Toch blijft 200 de default: hem verhogen vergroot de opgeslagen
    jobpayload, en die afweging hoort bij wie de opslag beheert. Zet hem
    hoger op een profiel, of via `PSGSCORING_PLM_EVENT_LIST_CAP`, als de
    export compleet moet zijn. `n_events_truncated` in de samenvatting zegt
    altijd hoeveel er wegviel.
    """

    plm_time_base: bool = True
    """Reken de tijd van een beenbeweging met de echte vensterlengte.

    `_detect_lm_channel` berekent RMS over vensters van `int(sf * 0.1)`
    STALEN en zette de vensterindex om met `idx * 0.1`. Bij 256 Hz duurt een
    venster 0,09766 s, dus liep elke gerapporteerde tijd 2,3 % voor --
    lineair oplopend tot +620 s aan het eind van een nacht van 7,4 u.

    Event-F1 op PSG-IPA tegen twaalf scoorders, per been: mediaan
    0,038 -> 0,692 (mens onderling 0,820). Zie
    docs/plm_tijdbasis_bevinding.md.

    **Default `True` sinds 21-08-2026** (gebruikersbeslissing). Dit is geen
    andere keuze maar een rekenfout: de omzetting van vensterindex naar tijd
    nam 0,1 s aan waar het venster 0,09766 s duurt. Daarom volgt deze vlag
    het patroon van `rip_quality_scale_free` -- overal aan, ook op de
    historische profielen, want geen enkele regelset schrijft een
    tijdsafwijking van 2,3 % voor -- en niet dat van `single_channel_rhythm`,
    dat een scoringscriterium verandert.

    Uitgezonderd blijven `mesa_shhs` en `chicago_1999`, die paper v31/v37
    reproduceren."""

    arousal_hysteresis: bool = False
    """Laat een arousal doorlopen zolang de activiteit verhoogd blijft.

    Fase 1 van `detect_arousals` bouwt haar mask per sample en labelt die
    direct: er wordt geen enkel gat gedicht. Bandvermogen fluctueert op
    subseconde-schaal, dus één arousal valt uiteen in scherven en de 3 s-eis
    gooit ze bijna allemaal weg. Gemeten op MESA: 1897 ruwe regio's, waarvan
    er 65 overblijven. De mediane eventduur (3,6 s) ligt daardoor op de
    ondergrens zelf, tegen 8,6 s (PSG-IPA) en 11,0 s (MESA) bij mensen.

    Met deze vlag geldt een tweede, lagere drempel voor het DOORLOPEN van een
    event. De instapdrempel verandert niet, dus de vlag bepaalt alleen waar
    een event eindigt. Zie docs/arousal_duration_preregistratie.md.

    Default `False` — bestaand gedrag is byte-identiek met de vlag uit."""

    arousal_spectral_shift: bool = False
    """Beslis arousals op een SPECTRALE VERSCHUIVING i.p.v. op vermogen.

    De regel tot nu toe vergelijkt het vermogen in alpha+theta+beta met een
    basislijn uit de opname zelf (`detect_arousals`, fase 1). De AASM
    beschrijft een verschuiving van de FREQUENTIE. Vermogen is onbegrensd en
    amplitude-gevoelig, dus een vaste verhouding betekent per opname iets
    anders — op PSG-IPA loopt de drempel die de scoordermediaan reproduceert
    van 1,2 tot 4,0 over vijf nachten, en `delta_pow` werd wel berekend maar
    nergens in de beslissing gebruikt.

    Met deze vlag beslist de detector op de FRACTIE van het spectrale
    vermogen in de snelle banden, `(alpha+theta+beta) / totaal`: dimensieloos,
    begrensd op [0,1] en invariant onder een amplitudeschaling. Zie
    docs/arousal_spectral_shift_preregistratie.md voor het vooraf vastgelegde
    acceptatiecriterium.

    Default `False` — bestaand gedrag is byte-identiek met de vlag uit."""

    rip_pair_scale_free: bool = False
    """Laat de PAAR-poort geen effortkanaal meer afkeuren dat zijn eigen
    kwaliteitstoets doorstaat.

    `rip_quality_scale_free` maakte de PER-KANAAL drempel schaalvrij. De
    paarregel bleef absoluut: `breath_energy` is de som van de PSD in de
    ademband, dus kwadratisch in de amplitude, en bij `ratio > 100` verklaart
    de poort het zwakste kanaal "disconnected". Twee banden met een andere
    versterking -- of in een andere eenheid geexporteerd -- halen die drempel
    zonder dat er iets mis is.

    GEMETEN op een klinische opname (19-08-2026, CSAS-of-OSAS-vraagstelling):
    ratio 1186x, thorax afgekeurd, terwijl de per-kanaal-toets voor diezelfde
    thorax `status=ok` en ademfractie 0,740 gaf en de band visueel bewoog. Het
    amplitudeverschil tussen de banden was ~23x.

    Zonder thorax is er geen paradoxale fase te meten, dus valt de
    classificatie terug op een eenkanaalsheuristiek. Wat dat kostte, gemeten
    door dezelfde opname twee keer te scoren met alleen deze beslissing
    verschillend -- 142 events in beide runs, Jaccard 1,000, dus de poort
    raakt geen detectie:

        49 central   -> obstructive
        16 uncertain -> obstructive
         7 uncertain -> central

    De verdeling ging van 89 centraal / 9 obstructief naar 47 / 74, en
    `ahi_total` van 20,0 naar 23,9 (de kale `uncertain` vallen buiten die
    index). Op die opname draait dat de diagnose om van centraal naar
    obstructief met centrale apneus.

    Met `True` mag een kanaal alleen "disconnected" heten als het zijn eigen,
    schaalvrije kwaliteitstoets NIET doorstaat. Een losgekoppelde band draagt
    geen ademhaling en valt daar vanzelf op. Halen beide kanalen `ok`, dan
    blijft de ratio een waarschuwing in plaats van een modusomschakeling.

    **Default `False`.** Dit verschuift subtypes en daarmee OAHI/CAHI op elke
    opname waar de poort nu afgaat; het hoort gemeten te worden op MESA en
    PSG-IPA voordat het aan gaat, niet stil gecorrigeerd."""

    rip_quality_scale_free: bool = True
    """Beoordeel de RIP-effortkanalen op signaalVORM in plaats van op absolute
    amplitude.

    De oude poort keurt af bij `MAD < 0.005` of `breath_energy < 0.001`. Die
    drempels zijn absoluut, en EDF-eenheden zijn per kanaal vrij: MESA schrijft
    RIP in mV en komt na omrekening naar V ~150x te laag binnen met een
    volstrekt normaal signaal, PSG-IPA schrijft `n/a` en komt er duizenden
    malen boven uit. De poort selecteerde dus op de eenhedendeclaratie.
    Gemeten spant MAD zeven ordes van grootte terwijl de ademfractie — de
    werkelijke kwaliteit — een factor 2,5 spant, in een andere volgorde.

    Aan: `rip_shape_metrics()` beslist op de ademfractie (vermogen 0,10–0,50 Hz
    gedeeld door 0,02–4,0 Hz) en op de vlakke fractie. Beide zijn verhoudingen
    binnen hetzelfde signaal en dus eenheidsvrij.

    **Default `True` sinds 13-08-2026** (gebruikersbesluit). Dit is een
    INDEXwijziging, niet alleen een typeringswijziging: kale `uncertain` valt
    buiten `ahi_total`, dus een werkende poort verhoogt de AHI. Tegen de
    NSRR-`aasm15`-referentie (n=40) halveert de bias — `aasm_v3_rec`
    −9,48 → −5,14, `aasm_v3_breath` −11,93 → −5,33 — bij op `breath`
    identieke F1, precisie en recall: zelfde events, andere boekhouding.

    `mesa_shhs` en `chicago_1999` blijven expliciet op `False`, anders breekt
    de reproductie van paper v31/v37. Zie CHANGELOG v0.17.0."""

    max_events_per_desaturation: int | None = None
    """Hoeveel hypopneus één desaturatie ten hoogste mag bevestigen.

    AASM Regel 1A vraagt per event een eigen desaturatie van ≥ 3 %. De
    detectoren toetsen dat per kandidaat afzonderlijk: elk event zoekt zelf een
    daling in zijn eigen nazoekvenster. Twee kandidaten die kort na elkaar
    liggen kunnen daardoor dezelfde daling als bevestiging opvoeren, en op een
    nacht met veel opeenvolgende reducties telt één diepe desaturatie meerdere
    keren mee. Dat is geen detectiefout maar een boekhoudfout, en hij werkt
    maar één kant op: hij telt te veel.

    Bij overschrijding blijft het event met de hoogste `p_scored` staan; de
    rest wordt GEDEGRADEERD naar `rejected_hypopneas` met
    `reject_reason="desat_reuse_limit"`, niet verwijderd. Dat houdt ze
    beschikbaar voor de ML-promotie en de visuele controle, en het houdt de
    ingreep omkeerbaar en te tellen.

    Raakt uitsluitend de desaturatietak: events die op een arousal zijn
    bevestigd (Regel 1A-arousal) worden niet begrensd, want daar bestaat het
    hergebruikprobleem niet in deze vorm.

    Default `None` = geen begrenzing = bestaand gedrag; elk bestaand profiel
    blijft byte-identiek. Idee overgenomen uit CAISR-resp (daar hard op 2);
    herkomst en meting staan in `docs/third_party_comparison.md`."""

    hypoxic_burden_local_baseline: bool = False
    """Laat de hypoxic burden de LOKALE basislijn volgen in plaats van
    `max(lokaal, globaal)`.

    De burden meet het oppervlak onder de basislijn vóór het event. Die
    basislijn is nu het hoogste van het lokale 90e percentiel (120 s vóór het
    event) en het globale 95e percentiel van de nacht. Dat globale plafond
    bestaat om te voorkomen dat een clusterend event de lokale basislijn omlaag
    trekt en de burden onderschat.

    Maar op een nacht waarin de saturatie daalt — COPD, obesitas-hypoventilatie —
    worden events laat in de nacht gemeten tegen een basislijn van vroeg in de
    nacht. Gemeten op een synthetische trend van 94% naar 82%: de burden gaat
    van 21,6 naar 41,2 zonder dat er iets aan de events verandert. Bij een
    VLAKKE lage basislijn is er geen probleem: 85% en 96% geven allebei 21,6.

    Standaard uit, want dit verschuift een gepubliceerde grootheid op elke
    opname met drift. De code waarschuwde hier zelf al voor in een
    v0.4.4-review, met het voorstel de override te poorten op een lage lokale
    basislijn; deze vlag is die poort, maar dan expliciet in plaats van als
    drempel op 88%.
    """

    thermistor_gate: str = "envelope_agreement"
    """Welke toets beslist of de thermistor de apneu-sensor mag zijn.

    "envelope_agreement"  — bestaande poort (default, ongewijzigd gedrag):
                            correlatie tussen de envelopes van thermistor en
                            neusdruk, drempel 0,40.
    "respiratory_band"    — ademband-vermogen van het thermistorkanaal ALLEEN,
                            drempel 0,70. Zie assess_thermistor_band_power.
    "breath_coherence"    — magnitude-squared coherentie tussen thermistor en
                            neusdruk over de ademband, drempel 0,015. Vraagt of
                            beide sensoren DEZELFDE ademteugen zien in plaats
                            van of hun amplitudes samen bewegen. Zie
                            assess_breath_coherence: een thermistor is
                            thermisch en verzadigt, neusdruk schaalt vóór
                            linearisatie als flow-kwadraat, dus hun
                            amplitudedynamiek verschilt legitiem. Gemeten op
                            MESA keurt de envelope-poort 14 van 25 opnames af
                            terwijl 24 van 25 thermistors aantoonbaar dezelfde
                            ademhaling volgen.

    De default blijft de bestaande poort, en dat is geen conservatisme uit
    gewoonte: deze keuze verschuift op elke opname of apneus op de thermistor
    of op de neusdruk gescoord worden, en dus de AHI. Dat raakt mesa_shhs en
    de gepubliceerde cijfers. Aanzetten hoort per profiel te gebeuren, met een
    meting erbij.
    """

    hypopnea_arousal_weight: float = 0.90
    """v0.14.4: gewicht van de arousal-tak in de gegradeerde bevestiging.

    De detector is gegradeerd op elke as behalve deze. ``p_confirm`` is een
    noisy-OR van desaturatie en arousal, en ``p_arousal`` springt naar deze
    waarde zodra er een arousal in het koppelvenster ligt. Bij 0,90 is
    ``p_confirm >= 0,90`` ongeacht de desaturatie — een drempel in een verder
    continu model, en dus een gratis toegangsbewijs.

    Op PSG-IPA SN1 markeert de detector tien events die GEEN van de twaalf
    scoorders markeerde; vijf daarvan hebben een desaturatie onder 1,2 %,
    terwijl het laagste consensus-event op 2,16 % zit. Die vijf kunnen alleen
    via deze tak gekwalificeerd zijn.

    Lager zetten maakt een arousal een argument in plaats van een bewijs: met
    0,55 haalt een arousal-only event de drempel van 0,50 alleen nog bij een
    overtuigende reductie en duur, terwijl een arousal mét gedeeltelijke
    desaturatie via de noisy-OR ruim boven de drempel blijft.

    Default 0,90 = ongewijzigd gedrag."""

    thermistor_agreement_weighting: bool = False
    """v0.14.4: gebruik de sensorovereenstemming als GEWICHT in plaats van poort.

    ``assess_flow_sensor_agreement`` rekent een continue envelope-correlatie
    tussen thermistor en neusdruk uit — op echte opnames 0,32 tot 0,71 — en die
    wordt gereduceerd tot ``usable`` ja/nee. Boven de drempel telt de sensor
    volledig mee, eronder helemaal niet. Dat is dezelfde vorm als de arousal-as
    vóór v0.14.4: een drempel in een verder continu model.

    Met deze vlag aan:

    * de thermistor wordt niet meer weggegooid bij lage overeenstemming maar
      ADDITIEF gebruikt — hij mag events toevoegen, nooit afwijzen, wat de
      veilige richting is (falsifiëren met een sensor die je niet vertrouwt is
      hoe een echte opname 83 centrale apneus verloor);
    * elk apneu-event krijgt ``sensor_agreement``, en events waarvan de
      thermistor de enige steun is krijgen hun confidence met die waarde
      geschaald. Een event uit een sensor met overeenstemming 0,32 komt dan
      binnen met 32 % van zijn oorspronkelijke zekerheid in plaats van
      helemaal niet, of volledig.

    ⚠︎ **Niet gevalideerd.** PSG-IPA heeft één flowkanaal en geen thermistor,
    dus deze as is daar principieel niet te meten — precies waarom
    ``aasm_v3_rec``, ``aasm_v3_dual`` en ``aasm_v3_pressure`` er tot op de
    decimaal identiek zijn. Toetsen kan alleen op MESA of op een eigen cohort
    met twee flowsensoren.

    Default False = ongewijzigd gedrag."""

    hypopnea_desat_width: float = 0.80
    """v0.14.4: breedte van de zachte stap op de desaturatie-as.

    ``graded(x, 3.0, w)`` is een sigmoïde die op de drempel 0,5 geeft en over
    ``w`` van ~0,12 naar ~0,88 loopt. Kleiner = dichter bij een harde drempel.

    Let op de meetkunde vóór je hieraan draait: de sigmoïde is GECENTREERD op
    de drempel, dus een event onder 3 % komt nooit boven p_desat 0,5, hoe breed
    hij ook staat. Verbreden tilt lage waarden op naar 0,5 en drukt hoge
    waarden omlaag naar 0,5 — het maakt de as minder beslissend, in beide
    richtingen. Het verplaatst de drempel niet.

    Op PSG-IPA SN1 liggen de vijf gemiste consensus-events op 2,15-3,15 %
    desaturatie met een plafond van p = 0,42-0,48 bij oneindige breedte; ze
    kunnen dus met geen enkele breedte het werkpunt van 0,50 halen.

    Default 0,80 = ongewijzigd gedrag."""

    hypopnea_arousal_latency_grading: bool = False
    """v0.14.4: gradeer de arousal-koppeling op latentie in plaats van binair.

    Een arousal is als gebeurtenis binair, maar hoe overtuigend hij bij dít
    event hoort niet: een arousal die tijdens of vlak na het event begint is
    sterker gekoppeld dan een aan de rand van het venster. De changelog van
    v0.13.0 meet deze variant als de enige van drie die wint — F1 0,440 tegen
    0,434 en ernstovereenkomst 5/5 tegen 4/5 — en zette hem desondanks uit,
    omdat aanzetten het geijkte werkpunt verlaat.

    Default False = ongewijzigd gedrag; de env-override
    ``PSGSCORING_BREATH_AROUSAL_LATENCY`` blijft werken en wint."""

    rule1a_arousal_enabled: bool = False
    """v0.12.3+: laat de AASM Rule 1A arousal-tak daadwerkelijk kwalificeren.

    De tak was door een doorgiftefout dood (issue #16). De doorgifte is nu
    gerepareerd, maar hem AANZETTEN is een gedragswijziging: op PSG-IPA gaat
    de bias van +1.77 naar +7.75/u zolang de koppeling niet gekalibreerd is.
    Default False, zodat paper- en NSRR-reproductie ongemoeid blijven; fase 4
    bepaalt de kalibratie en pas daarna kan dit aan.
    """

    rule1a_gap_max_breaths: int = 1
    """Maximaal aantal ademteugen tussen eventeinde en arousal (gap-check)."""

    baseline_mode: str = "rolling"
    """v0.12.3+: welke baseline de hypopnee-drempelbeoordeling gebruikt.

    "rolling"   — compute_dynamic_baseline(): gecentreerd venster van 5 min.
                  Huidig gedrag; default voor ALLE profielen zolang de
                  validatie geen omschakeling rechtvaardigt.
    "pre_event" — compute_pre_event_baseline(): AASM-conform, alleen de
                  ademhaling vóór het event, ademteug-gebaseerd.

    De rolling baseline blijft in beide standen bestaan voor signaalkwaliteit
    en de bestaande visualisaties; alleen de drempelbeoordeling verandert.
    """

    local_baseline_cv_threshold: float = 0.30
    """v0.4.2: CV threshold below which local-baseline reduction is enforced strictly.

    The local baseline validation (`_validate_local_reduction` in respiratory.py)
    applies a stricter flow-reduction requirement when surrounding breathing is
    stable (low CV). At CV < this threshold, the stricter reduction kicks in.

    Default 0.30 matches the legacy hardcoded behaviour. Lower values (e.g. 0.20)
    relax the strictness — useful for sensitive profiles where borderline events
    should not be rejected.
    """

    local_baseline_strict_reduction: float = 25.0
    """v0.4.2: Flow-reduction percentage required when local CV < threshold.

    When breathing is stable (CV < local_baseline_cv_threshold), candidate events
    must show this percentage of flow reduction (vs the default 20%) to be
    accepted as hypopneas.

    - Strict profile: 30.0 (extra strict — reject more borderline events)
    - Standard:       25.0 (moderate)
    - Sensitive:      20.0 (no extra strictness — accept borderline events)
    """

    local_baseline_min_reduction_pct: float = 20.0
    """v0.5.0: Floor for required local-baseline reduction percentage.

    Previously hard-coded as the `min_reduction_pct=20.0` parameter default in
    `respiratory._validate_local_reduction`. The MESA/SHHS validation in
    paper v34 §S5.6 found that this floor over-rejects events on dense-cluster
    OSA recordings; the `mesa_shhs` dataset profile lowers it to 15.0 to align
    with NSRR-scorer behaviour. Clinical profiles keep 20.0 to preserve
    paper v31 PSG-IPA reproducibility.
    """

    local_baseline_pre_win_s: float = 30.0
    """v0.5.0: Pre-event window (seconds) for local-baseline-reduction
    validation.

    Previously hard-coded as the `pre_win_s=30.0` parameter default in
    `respiratory._validate_local_reduction`. NSRR scorers use a 1-2 minute
    visual baseline; the `mesa_shhs` profile uses 60.0 to match. Clinical
    profiles keep 30.0.
    """

    rule1b_arousal_window_s: float = 15.0
    """v0.5.0: Window (seconds) within which a scored arousal must follow a
    rejected hypopnea candidate for AASM Rule~1B reinstatement.

    Previously hard-coded as `RULE1B_AROUSAL_WINDOW_S = 15.0` in constants.py.
    On dense-arousal recordings (e.g. MESA mesaid 519 with 121 reinstatements
    at 15s) the wide window allows one arousal to couple to many rejected
    events, inflating Rule 1B counts. The `mesa_shhs` profile uses 5.0
    (≈ one breath cycle) to enforce closer 1:1 coupling. Clinical profiles
    keep 15.0.
    """

    baseline_window_s: int = 300
    """v0.5.1: Sliding-window length (seconds) for `compute_dynamic_baseline`.

    Previously hard-coded as `BASELINE_WINDOW_S = 300` (5 min). Lazazzera
    et al. 2020 (Sleep Breath, PMID 33019189) and Koley & Dey 2014
    (Med Biol Eng Comput, PMID 24876133) report higher hypopnea recall
    on heterogeneous cohorts using a 2-3 min window — the shorter window
    tracks local quiet-breathing periods more responsively, which helps
    on Compumedics-type recordings where DC amplitude varies across the
    night. The `mesa_shhs` profile uses 120 (2 min); clinical profiles
    keep 300. Range typically 60-600s.
    """

    baseline_percentile: float = 95.0
    """v0.5.1: Percentile of envelope amplitude used as the local baseline
    in `compute_dynamic_baseline`.

    Previously hard-coded as `np.percentile(seg, 95)`. The 95th percentile
    captures the upper end of breathing amplitude (peak inspiration in
    quiet sleep), but on cohorts with intermittent arousal-driven peaks
    it can be inflated by spikes that are not representative of true
    baseline. The Lazazzera/Koley line of work uses 80-90th percentile
    over the local window. The `mesa_shhs` profile uses 85.0; clinical
    profiles keep 95.0.
    """

    ml_classifier_path: Optional[str] = None
    """v0.6.0: optional path to a LightGBM booster file for candidate-level
    re-classification. When set, after Rule 1B reinstatement the pipeline
    calls `psgscoring.ml_classifier.apply_ml_reclassification` to score
    every candidate (current accepted + current rejected) and keep only
    those with `booster.predict(...) >= ml_threshold`.

    The path may be absolute or relative to the package data directory
    (`psgscoring/data/`). The shipped model
    `data/lightgbm_v06_q7holdout.txt` is consumed by the `mesa_shhs`
    profile by default. Clinical profiles leave this `None` and skip
    the re-classification step entirely (PSG-IPA paper-v31
    reproducibility unaffected).
    """

    ml_threshold: float = 0.65
    """v0.6.0: probability threshold for the LightGBM re-classifier.
    Default 0.65 was the bias-near-zero operating point on the q=7
    holdout (paper v34 §S5.6 sweep table)."""

    dual_sensor_apnea: bool = False
    """v0.14.0: gebruik BEIDE flowsensoren voor apneus in plaats van te kiezen.

    Kiezen is een valse keuze - de neusdruk mist mondademhaling, de thermistor
    is te traag voor korte events. Apneus worden daarom op beide kanalen
    gedetecteerd en ontdubbeld op overlap, zodat dezelfde gebeurtenis niet
    twee keer in de AHI belandt.

    Wat NIET gebeurt is falsifieren: een event dat maar een sensor ziet wordt
    behouden, niet afgewezen. Zie ``dual_sensor_corroboration``."""

    dual_sensor_corroboration: bool = False
    """v0.14.0: mag de tweede sensor events AFWIJZEN?

    **Default False, en dat is een klinische keuze.** Overdetectie tegenhouden
    mag alleen bij een betrouwbaar signaal; bij twijfel wint goedkeuren. Een
    onbetrouwbare tweede sensor die events wegneemt is hoe een echte opname 83
    centrale apneus verloor en van matig CSAS naar mild SAS schoof.

    Op MESA (25 opnames, 1785 apneu-detecties) wordt slechts 19% door beide
    sensoren bevestigd; falsifieren zou 81% weggooien op een detector die
    tegenover de NSRR-referentie al onderdetecteert. En de envelope-
    overeenstemming voorspelt de bevestiging niet (r = +0,07), dus er is geen
    maat die de licentie zou kunnen verlenen. Aanzetten vraagt bewijs dat er
    nu niet is."""

    arousal_limb_wired: bool = False
    """v0.13.0: mag dit profiel REAGEREN op de gerepareerde arousal-lijst?

    Issue #16: ``run_arousal_respiratory_analysis()`` levert zijn events
    genest af, terwijl de scoringsstappen de platte sleutel lazen. Die was in
    het auto-detectiepad nooit gevuld, dus elke arousal-afhankelijke stap
    draaide op een lege lijst — Rule 1B-reinstatement, FRI-RERA, en de
    LightGBM-features.

    v0.13.0 repareert de plumbing zelf, zodat consumenten BUITEN de scoring
    (de EDF+-export en de event-API van YASAFlaskified) eindelijk arousals
    zien. Maar de scoringsstappen laten reageren verandert bestaande
    klinische en dataset-uitkomsten ingrijpend — op mesa-sleep-2408 loopt de
    AHI van 30,4 naar 46,3, en dat breekt de NSRR/paper-v31-reproductie.

    Daarom is dit een profielkeuze met het HUIDIGE gedrag als default.
    Alleen ``aasm_v3_breath`` staat aan; dat profiel is nieuw, dus daar valt
    niets te breken. Voor de overige profielen blijft de uitkomst
    byte-identiek aan v0.12.3 tot elk arousal-afhankelijk onderdeel
    afzonderlijk opnieuw is gevalideerd — inclusief het hertrainen van de
    booster, zie ``ml_arousal_features``."""

    ml_arousal_features: bool = False
    """v0.13.0: feed real arousal values to the LightGBM re-classifier?

    **Default False, deliberately.** The booster's highest-gain features are
    arousal-based (`n_arousals_per_h`, `n_arousals_within_30s`,
    `has_arousal_within_5s`), but issue #16 meant the arousal list never
    reached inference — they were *always zero* in production, while training
    saw real values. That is a train/serve mismatch predating this release.

    v0.13.0 repairs the plumbing, so real values now reach every consumer.
    Passing them to this booster changes its predictions substantially
    (mesa-sleep-2408: AHI 30.4 -> 46.3) on a model that has never seen a
    non-zero value at inference. Until it is retrained, it gets what it
    knows — which is also what keeps NSRR/paper-v31 reproduction exact.

    Set True only together with a booster retrained on the repaired
    features."""

    flow_reference: str = "apnea"
    """v0.14.1: which flow channel the *non-apnea* analyses read.

    Five consumers besides apnea detection take a flow signal: the AHI-interval
    robustness sweep, baseline anchoring, the arousal/RERA coupling,
    Cheyne-Stokes detection and the ventilatory burden. All of them read the
    apnea channel.

    Under a substitutive profile that is harmless — the apnea channel *is* the
    recording's flow. Under an additive one it is not: ``aasm_v3_dual`` keeps
    the thermistor as the nominal apnea channel even when it fails the quality
    check, because the second detection pass makes a bad channel unable to
    remove events. The five consumers got no such second pass, so they
    inherited exactly the substitutive failure the additive design avoids. On
    one recording the ventilatory burden read 20.4% instead of 42.6% — below
    the ≤25% reference — on a patient spending 94.7% of the night under 90%
    saturation.

    Options:
      ``"apnea"``    — the apnea channel (default; current behaviour).
      ``"hypopnea"`` — the hypopnea channel, i.e. nasal pressure where the
                       montage has one, falling back to the single available
                       channel otherwise.

    ``"hypopnea"`` is also the AASM-correct answer on its own terms: the manual
    assigns *quantitative* flow assessment to nasal pressure and treats the
    oronasal thermistor as qualitative, suitable for detecting the absence of
    flow but not for grading it."""

    summary_after_reclassification: bool = False
    """v0.14.2: recompute the event summary AFTER step 11 (issue #18).

    The last ``_compute_summary()`` runs at step 9; step 11 then reclassifies
    events (CSR-driven obstructive→central, mixed decomposition) and replaces
    the event list. Everything derived from the summary — ``n_obstructive``,
    ``n_central``, ``n_mixed``, the type indices and ``oahi`` — therefore
    describes the state *before* that reclassification, while the event list,
    the confidence table and every per-event figure describe the state after.

    Visible in the report as a mismatch between the ``n`` column and the
    confidence breakdown, exactly equal to ``n_csr_reclassified``. It is not
    cosmetic: ``oahi`` moved 32.2 → 28.6 on one real recording, i.e. across
    the severe/moderate boundary. ``ahi_total`` is unaffected — the
    reclassification relabels events, it does not add or drop any.

    **On** for every profile a clinician can select — the whole clinical family
    plus the two v3 arms in the exploratory family, which sit in the same
    dropdown. **Off** for the reproduction profiles (``mesa_shhs``,
    ``chicago_1999``), which must stay byte-identical to the published numbers.

    The dataclass default stays ``False`` on purpose: forgetting to switch it on
    for a new clinical profile leaves the old behaviour, which is visible and
    harmless, whereas forgetting to pin a new dataset profile would silently
    break NSRR reproduction. The safe failure mode is the default.

    Note that the robustness sweep calls ``detect_respiratory_events`` directly
    rather than the pipeline, so the strict/sensitive arms of
    ``profile_comparison`` never read this flag; only a run where such a profile
    is the primary one is affected.
    """

    flow_fallback_strategy: str = "none"
    """v0.5.1: Behaviour when the primary nasal-pressure flow channel
    appears low-quality on a recording.

    Options:
      ``"none"``                       — never fall back (legacy behaviour).
      ``"ripsum_on_nasal_failure"``    — if Pres signal amplitude-CV over
                                         the whole recording falls below a
                                         threshold (sensor dropout / flat
                                         line / extreme attenuation), use
                                         the thoracoabdominal RIP sum
                                         (thorax + abdomen) as the
                                         hypopnea-detection input
                                         instead. Vaquerizo-Villar et al.
                                         (DRIVEN, npj Dig Med 2024) and
                                         Nassi et al. (IEEE TBME 2022)
                                         both show that effort-belt-only
                                         scoring achieves r²~0.79 when
                                         airflow channels degrade.

    Clinical profiles default to ``"none"``; ``mesa_shhs`` sets
    ``"ripsum_on_nasal_failure"``.
    """

    envelope_method: str = "hilbert"
    """v0.19.0: how the amplitude envelope of flow and effort is built.

    Every published result to date uses ``"hilbert"``: the analytic signal over
    the full recording. That transform FFTs the whole night at once and returns
    complex128, so it peaks at roughly half a gigabyte per channel — the
    largest per-worker allocation in the library, and the one that makes a
    parallel sweep swap.

    Options (all measured against ``"hilbert"``, none of them free):

      ``"hilbert"``          — full-array analytic signal. Default.
      ``"hilbert_chunked"``  — same transform, blockwise with overlap-discard.
                               ~30 MB per channel instead of ~500 MB.
      ``"rectify_lowpass"``  — ``|x|`` through a 0.5 Hz zero-phase lowpass.
                               O(n), negligible memory, ripple at the breathing
                               rate and filter-shaped flanks.
      ``"breath_amplitude"`` — per-breath peak-to-trough amplitude, interpolated.
                               Closest to how AASM words the rules (excursions
                               per breath, not a continuous envelope), and puts
                               the sensitivity into peak detection at low
                               amplitudes — i.e. during events.

    ``"hilbert_chunked"`` was specified as a free implementation detail on the
    reasoning that a generous pad reproduces the full transform up to numerical
    noise. It does not: the 1/(pi t) kernel has no compact support, the
    interior residual floors around 1e-4 of the p95 envelope and does not
    converge as the pad grows, and the first and last samples differ by ~30 %
    because an FFT wraps circularly and the two versions wrap differently.
    1e-4 is far below any clinical threshold, but the envelope is compared to a
    baseline sample by sample, so it can still move a boundary. It is on this
    axis for that reason, not for symmetry.

    The default is ``"hilbert"`` for every profile, including the two
    reproduction profiles. A memory optimisation is never worth an unannounced
    change to a published number.
    """

    flow_wavelet_denoise: bool = False
    """v0.20.0: remove impulsive artefacts from the flow signal by soft
    wavelet thresholding, after the bandpass and before anything reads it.

    ``False`` = current behaviour, bit-for-bit. Requires PyWavelets
    (``pip install psgscoring[denoise]``); a profile that asks for denoising
    without it raises rather than scoring silently undenoised.

    **What it addresses, and why the bandpass does not already.** The bandpass
    removes what lies spectrally outside 0.05–3 Hz. A motion artefact of 1–2 s
    has power *inside* that band and passes straight through. Wavelet
    thresholding acts locally in time and scale, so it removes the spike without
    touching the breathing around it. The scope is deliberately narrow: short,
    high-energy disturbances within the respiratory band.

    Threshold: Donoho & Johnstone's universal ``T = sigma * sqrt(2 ln N)`` with
    ``sigma = MAD(d1)/0.6745`` from the finest detail scale — derived, not
    tuned, and conservative on coloured noise because it over-estimates T and
    therefore removes less.

    **The risk.** An apnea onset is also an abrupt amplitude change. Too
    aggressive a threshold smooths exactly the flanks the detection runs on and
    shifts event boundaries — the axis block 1B measures. The boundary-offset
    distribution is therefore the primary endpoint for this field, ahead of
    event-F1, and the pre-registered acceptance condition is in the CHANGELOG.

    Scope is the **flow** signal. RIP belts keep the analytic envelope on an
    undenoised waveform: the artefacts this targets are airflow-sensor
    artefacts, and extending it to effort would be an unmeasured change riding
    along on a measured one.

    Interacts with ``artefact_flank_exclusion`` (fix 5) rather than replacing
    it: thresholding damps the spike, fix 5 excludes the flank around it. Both
    stay on, and the number of times fix 5 fires is reported so a measurement
    can show whether they duplicate each other.
    """

    envelope_fs: float | None = None
    """v0.19.0: decimate to this rate before the analytic-signal transform.

    ``None`` = current behaviour (transform at the acquisition rate). The
    upstream bandpass passes nothing above 3 Hz while acquisition runs at
    200–512 Hz, so a 10 Hz envelope rate keeps Nyquist well clear of the band
    and shrinks the transform by a factor of 20–50.

    The envelope is interpolated back onto the original sample grid, so no
    downstream consumer has to learn a second sample rate. The saving is in the
    transform, not in the returned array.

    Not free either: the decimation anti-alias filter and the coarser raster
    both leave a fingerprint. Ignored by ``rectify_lowpass`` and
    ``breath_amplitude``, which never build an analytic signal.
    """


@dataclass
class Profile:
    """Complete scoring profile specification."""

    name: str
    display_name: str
    family: str  # "clinical" | "dataset" | "legacy" | "exploratory"
    aasm_version: str
    aasm_rule: str
    description: str

    hypopnea: HypopneaRules
    apnea: ApneaRules = field(default_factory=ApneaRules)
    spo2: SpO2Rules = field(default_factory=SpO2Rules)
    post_processing: PostProcessingRules = field(default_factory=PostProcessingRules)

    citation: str = ""
    """Primary reference citation for this profile."""

    deprecated: bool = False
    deprecated_alias_for: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dict for audit metadata."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        h = self.hypopnea
        rule = []
        rule.append(f"≥{int(h.flow_reduction_threshold * 100)}%")
        if h.desat_threshold:
            rule.append(f"≥{int(h.desat_threshold * 100)}%")
        if h.desat_or_arousal:
            rule.append("OR arousal")
        elif h.desat_required and h.desat_threshold:
            rule.append(f"desat ≥{int(h.desat_threshold * 100)}%")
        return f"{self.display_name} [{' '.join(rule)}]"


# ============================================================
# Profile definitions
# ============================================================

# ---- AASM v3 RECOMMENDED (Rule 1A) ----
# This is THE DEFAULT for all clinical use in AASM-accredited labs
# since Dec 31, 2023. All shared post-processing comes from here.
_aasm_v3_rec = Profile(
    name="aasm_v3_rec",
    display_name="AASM v3 — Recommended (3%-or-arousal)",
    family="clinical",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "Current clinical standard per AASM Scoring Manual v3 (Troester "
        "2023). Mandatory for AASM-accredited facilities since Dec 31, "
        "2023. Hypopneas require ≥30% nasal pressure reduction for "
        "≥10 seconds associated with EITHER a ≥3% oxygen desaturation "
        "OR an arousal."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
    ),
)

# ---- AASM v3 Rule 1A, ademteug-gebaseerd en gegradeerd (experimenteel) ----
# Zelfde regel, andere architectuur: de ademteug is het atoom, de patiënt
# kalibreert zichzelf in twee passages, en elk AASM-criterium is gegradeerd
# in plaats van binair. Elk event draagt een kans "welk deel van de scorers
# zou dit markeren". Zie psgscoring/breath_scoring.py.
_aasm_v3_breath = Profile(
    name="aasm_v3_breath",
    display_name="AASM v3 — Rule 1A, breath-graded",
    family="clinical",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED), graded",
    description=(
        "Same AASM v3 Rule 1A conjunction — >=30% reduction AND >=10 s AND "
        "(>=3% desaturation OR arousal) — evaluated per breath instead of "
        "per sample, with a per-patient template (including that patient's "
        "own SpO2 delay) and graded rather than binary criteria. Each event "
        "carries p_scored, readable as the fraction of scorers who would "
        "mark it. Strictness is a single axis on that probability."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        hypopnea_detector="breath_graded",
        summary_after_reclassification=True,
        # Als enige profiel reageert dit op de gerepareerde arousal-lijst
        # (issue #16). Dat kan hier veilig, want het profiel is nieuw: er is
        # geen bestaande uitkomst om te breken. Zie arousal_limb_wired.
        arousal_limb_wired=True,
        # De arousal-classifier staat sinds 23-08-2026 AAN op deze profielen
        # (gebruikersbeslissing). Hij stond hier een dag uit omdat de
        # RDI-impact ongemeten was; die meting is nu gedaan, gepaard op MESA
        # n=30 met werkpunt 0,80 en artefact-epochs zoals productie:
        #
        #   arousals 212 -> 107, tegen een MENSELIJKE referentie van 128:
        #   de ratio gaat van 1,47 naar 0,83 en de afwijking van een zuivere
        #   telling van 0,58 naar 0,26, dichter bij de referentie op 22/30.
        #   Respiratoire event-F1 0,44 -> 0,48 (+0,009 gepaard, p = 5,5e-04,
        #   beter op 23/30).
        #
        # De prijs: de RDI zakt mediaan 9,25/u en de RDI-ernstklasse verschuift
        # op 11/30 = 37 % van de opnames. Dat is NIET als correctie geboekt --
        # er is geen RERA-referentie (zie reference_no_rera_reference), dus het
        # is een verschuiving. De AHI-bias verslechtert licht (-2,09 -> -3,28)
        # omdat arousals op dit profiel ook hypopneus bevestigen.
        #
        # De afweging die de doorslag gaf: twee MEETBARE dingen worden beter
        # (arousaltelling, event-F1) tegen één onmeetbaar ding dat verschuift.
        # Bij de vorige beslissing lag er alleen dat laatste.
        # Operatiepunt, met de herkomst erbij omdat die het cijfer weegt:
        #
        #   0,50 kwam uit een VOORAF vastgelegde regel op PSG-IPA — de
        #   strictness waarbij de AHI-bias nul wordt, niet die waar de F1
        #   maximaal is. F1 en percentiel waren daar dus een niet-gefitte
        #   uitkomst.
        #
        #   Er is korte tijd 0,55 geprobeerd, om een Normal->Mild-drift te
        #   dempen die zichtbaar was tegen MESA's oahi45. Een tweede,
        #   disjuncte MESA-steekproef (n = 50, overlap 0) liet zien dat die
        #   drift een artefact was van de REFERENTIE: oahi45 crediteert geen
        #   hypopnee die alleen door een arousal kwalificeert, terwijl Rule 1A
        #   dat wel doet. Tegen nsrr_ahi_hp3r_aasm15 — letterlijk de regel die
        #   dit profiel implementeert — is het probleem juist forse
        #   ONDERdetectie, en dempt 0,50 die het sterkst: severity 24/50 tegen
        #   20/50 bij 0,55 en 15/50 bij aasm_v3_rec, en gepaard 36/50 beter
        #   (p = 0,0016) tegen 31/50 (p = 0,026).
        #
        #   0,50 wint daarmee op F1, recall, bias, MAE en severity, op alle
        #   drie de referenties. Zie CHANGELOG.
        hypopnea_strictness=0.50,
    ),
)


# ---- AASM v3 Rule 1A, volledig probabilistisch (experimenteel) ----
# aasm_v3_breath in alles behalve de arousal-as. Die was daar de enige
# ongegradeerde: p_arousal sprong naar 0,90 zodra er een arousal in het
# koppelvenster lag, waardoor de noisy-OR p_confirm >= 0,90 gaf ongeacht de
# desaturatie. Een drempel in een verder continu model.
#
# Aanleiding is een meting, geen ontwerpvoorkeur. Op PSG-IPA SN1 markeert
# aasm_v3_breath tien events die GEEN van de twaalf scoorders markeerde; vijf
# daarvan hebben een desaturatie onder 1,2 %, terwijl het laagste
# consensus-event op 2,16 % ligt. Desaturatie scheidt die twee groepen met
# AUC 0,82, de confidence met 0,68 — de as die de fout maakt is dus juist de
# as die niet meetelt.
_aasm_v3_prob = Profile(
    name="aasm_v3_prob",
    display_name="AASM v3 — Rule 1A, graded arousal axis (experimental)",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED), graded",
    description=(
        "Identical to aasm_v3_breath except on the arousal axis, which was "
        "the one criterion still evaluated as a threshold: p_arousal jumped "
        "to its full weight as soon as an arousal fell in the coupling "
        "window, so the noisy-OR confirmation could never drop below that "
        "weight however small the desaturation. Here an arousal is an "
        "argument rather than a proof: it is graded by coupling latency and "
        "carries a weight that leaves an arousal-only event dependent on a "
        "convincing reduction and duration, while an arousal WITH partial "
        "desaturation still clears the operating point comfortably. "
        "Experimental: the operating point has not been re-derived. "
        "NOTE ON THE NAME: this profile was called 'fully probabilistic', "
        "which overpromised. Only the HYPOPNEA axis is graded. Apneas still "
        "come from the same rule cascade as every other profile, and their "
        "confidence saturates at its per-rule cap (0.95 obstructive, 0.90 "
        "central), so within a class it barely discriminates. Grading the "
        "apnea axis is detector work that has not been done."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        hypopnea_detector="breath_graded",
        summary_after_reclassification=True,
        arousal_limb_wired=True,
        # De arousal-classifier staat UIT op de vier profielen waar arousals
        # doorwerken in de RDI. Gemeten op MESA n=50 (gepaard, seed 20260801):
        # met classifier daalt de RDI-mediaan 37,2 -> 30,0, gepaarde mediaan
        # -5,8/u, en de RDI-ERNSTKLASSE verschuift op 14/50 opnames (28 %) --
        # tegen 5/50 voor de AHI. De oorzaak is niet omstreden: de classifier
        # verwerpt ~een vijfde van de arousals (225 -> 179 mediaan) en elke
        # RERA hangt aan een arousal binnen 15 s, dus de RERA-index zakt mee
        # (17,0 -> 11,1) terwijl RERA's ~46 % van de RDI uitmaken.
        #
        # Waarom dat NIET als verbetering geboekt wordt: er is geen
        # RERA-referentie om tegen te toetsen. MESA annoteert er geen, en
        # PSG-IPA -- waar RERA WEL in de toegestane labelset zat
        # (`^RERA$` in de scoringsfilter) -- bevat er 3 in de hele manuele
        # set, uit 2 van de 60 scoorder-opnames. Zonder referentie is een
        # verschuiving van deze omvang geen correctie maar een onbewezen
        # herijking, en dan gaat de bestaande uitkomst voor.
        #
        # OMGEZET 23-08-2026 (gebruikersbeslissing): de classifier staat nu ook
        # hier AAN. De RDI-impact was toen ongemeten en is dat niet meer -- zie
        # de toelichting bij `aasm_v3_breath`. Kort: de arousaltelling gaat van
        # 47 % te veel naar 17 % te weinig tegen de menselijke MESA-referentie
        # en de event-F1 stijgt, tegen een RDI die mediaan 9,25/u zakt met
        # 37 % herklassering. Twee meetbare verbeteringen tegen een
        # onmeetbare verschuiving.
        hypopnea_strictness=0.50,
        # De twee assen die dit profiel onderscheiden.
        #
        # 0,70 komt uit een sweep over 0,90 / 0,70 / 0,55 / 0,40 / 0,00 op
        # PSG-IPA (n = 5, manueel hypnogram). Het aantal events dat met GEEN
        # ENKEL menselijk event overlapt daalt monotoon met het gewicht —
        # 69 / 60 / 47 / 37 / 29 / 26 — wat bevestigt dat een groot deel van
        # die events inderdaad via deze tak binnenkwam. Maar onder 0,70 gaat
        # het consensus-events kosten: de recall op clusters die >=6 van de
        # 12 scoorders deelden blijft 0,50 tot en met 0,70 en zakt daarna
        # naar 0,47 (0,55) en 0,44 (0,40).
        #
        # 0,70 is dus het laagste gewicht dat nog gratis is: F1 0,453 — de
        # hoogste van alle geteste configuraties, tegen 0,434 voor
        # aasm_v3_breath — precisie 0,72 -> 0,79, en de AHI blijft op 5/5
        # opnames binnen de scoorderspreiding.
        #
        # 0,55 was mijn eerste keuze, beredeneerd uit de rekensom en niet uit
        # data. De sweep wees hem af: hogere precisie (0,86) maar ten koste
        # van consensus-recall en met een bias van -1,47.
        hypopnea_arousal_weight=0.70,
        hypopnea_arousal_latency_grading=True,
    ),
)


# ---- AASM v2 RECOMMENDED (2012-2020) ----
# Functionally identical to v3 Rule 1A. Kept for explicit labeling
# of analyses performed during the v2 era.
_aasm_v2_rec = Profile(
    name="aasm_v2_rec",
    display_name="AASM v2 — Recommended (3%-or-arousal, legacy)",
    family="clinical",
    aasm_version="v2.0–v2.6 (2012–2020)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM Manual v2 Rule 1A, introduced by the Sleep Apnea Definitions "
        "Task Force (Berry 2012) and retained through v2.6 (Berry 2020). "
        "Functionally identical to v3 Rule 1A. This profile exists for "
        "explicit labeling of analyses performed under the v2 era; for "
        "current clinical use, prefer aasm_v3_rec."
    ),
    citation="Berry RB, et al. JCSM 2012;8:597-619; Berry RB, et al. AASM v2.6. 2020.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Ritmiek-as UIT: historisch/regulatoir dan wel bevroren.
        single_channel_rhythm=False,
        # Arousal-classifier om dezelfde reden UIT: een ML-filter maakt geen
        # deel uit van AASM v1/v2 of de CMS-regels, en de twee
        # dataset-profielen moeten paper v31/v37 reproduceren.
        arousal_lgbm=False,
        # Om dezelfde reden GEPIND op het artefactgedrag van vóór 23-08-2026.
        # De niet-gepinde v3-profielen negeren de artefactlijst nu (30/30 op
        # MESA), maar dat verschuift de arousaldetectie en dus de uitkomst van
        # een gereproduceerde regelset.
        arousal_uses_artifact_epochs=True,summary_after_reclassification=True),
)

# ---- AASM v1 RECOMMENDED (2007) ----
# Historical 2007 rule. Note: 4% desaturation, NO arousal alternative.
_aasm_v1_rec = Profile(
    name="aasm_v1_rec",
    display_name="AASM v1 (2007) — Historical (4% desat, no arousal)",
    family="clinical",
    aasm_version="v1 (2007)",
    aasm_rule="Original Recommended",
    description=(
        "AASM Manual v1 original recommended rule (Iber 2007). Requires "
        "≥30% flow reduction with ≥4% oxygen desaturation; does NOT allow "
        "arousal as alternative qualifier. Use for re-analysis of pre-2012 "
        "research cohorts. Typically yields lower AHI than v2/v3 rules."
    ),
    citation="Iber C, Ancoli-Israel S, Chesson AL, Quan SF. AASM Manual v1. 2007.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.04,
        desat_required=True,
        arousal_required=False,
        desat_or_arousal=False,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Ritmiek-as UIT: historisch/regulatoir dan wel bevroren.
        single_channel_rhythm=False,
        # Arousal-classifier om dezelfde reden UIT: een ML-filter maakt geen
        # deel uit van AASM v1/v2 of de CMS-regels, en de twee
        # dataset-profielen moeten paper v31/v37 reproduceren.
        arousal_lgbm=False,
        # Om dezelfde reden GEPIND op het artefactgedrag van vóór 23-08-2026.
        # De niet-gepinde v3-profielen negeren de artefactlijst nu (30/30 op
        # MESA), maar dat verschuift de arousaldetectie en dus de uitkomst van
        # een gereproduceerde regelset.
        arousal_uses_artifact_epochs=True,summary_after_reclassification=True),
)

# ---- CMS / Medicare (AASM v3 1B OPTIONAL) ----
# US insurance reimbursement criterion. Degraded from "acceptable" to
# "optional" in v3 (2023) but still widely used for coverage.
_cms_medicare = Profile(
    name="cms_medicare",
    display_name="CMS / Medicare (4% desat, no arousal)",
    family="clinical",
    aasm_version="v3 Rule 1B (OPTIONAL) / CMS",
    aasm_rule="1B (OPTIONAL)",
    description=(
        "CMS (Centers for Medicare & Medicaid Services) hypopnea criterion, "
        "corresponding to AASM Rule 1B. Requires ≥30% flow reduction with "
        "≥4% oxygen desaturation; arousals are not accepted as qualifier. "
        "Downgraded from 'acceptable' (v2) to 'optional' (v3) by AASM in "
        "2023 but retained for CMS reimbursement in the United States. "
        "Typically yields ~30% lower AHI than the v3 recommended rule."
    ),
    citation="CMS National Coverage Determination 240.4; Troester 2023 Rule 1B.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.04,
        desat_required=True,
        arousal_required=False,
        desat_or_arousal=False,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Ritmiek-as UIT: historisch/regulatoir dan wel bevroren.
        single_channel_rhythm=False,
        # Arousal-classifier om dezelfde reden UIT: een ML-filter maakt geen
        # deel uit van AASM v1/v2 of de CMS-regels, en de twee
        # dataset-profielen moeten paper v31/v37 reproduceren.
        arousal_lgbm=False,
        # Om dezelfde reden GEPIND op het artefactgedrag van vóór 23-08-2026.
        # De niet-gepinde v3-profielen negeren de artefactlijst nu (30/30 op
        # MESA), maar dat verschuift de arousaldetectie en dus de uitkomst van
        # een gereproduceerde regelset.
        arousal_uses_artifact_epochs=True,summary_after_reclassification=True),
)

# ---- MESA / NSRR convention ----
# Faithful reproduction of scoring as documented in the MESA scoring
# manual (Brigham Reading Center). Key differences from AASM:
#   1. Hypopneas use thoracoabdominal band amplitude as primary sensor
#   2. Events are marked independent of desaturation; multiple AHI
#      variants (3%, 3%+arousal, 4%) are generated post-hoc
#   3. "Unsure" tag = hypopnea with >50% reduction (not uncertainty)
_mesa_shhs = Profile(
    name="mesa_shhs",
    display_name="MESA / NSRR (nasal-pressure-primary, NSRR clinical AHI)",
    family="dataset",
    aasm_version="NSRR convention (R&K staging era)",
    aasm_rule="mesa_shhs",
    description=(
        "Reproduction of scoring as documented in the MESA Sleep "
        "Polysomnography Scoring Manual (Brigham Reading Center). "
        "Hypopneas are identified from airflow signals (nasal pressure "
        "primary, with thermistor and RIP-bands as supporting context). "
        "Gating follows the NSRR `nsrr_ahi_hp3u` clinical AHI definition: "
        "≥30% airflow reduction with ≥3% desaturation OR coupled arousal, "
        "functionally equivalent to AASM v2 Rule 1A. Local-baseline "
        "validator and Rule-1B arousal-coupling window are tuned for "
        "NSRR-cohort scoring conventions (paper v34 §S5.6). The 'Unsure' "
        "tag in MESA XML denotes a hypopnea with >50% reduction, NOT "
        "uncertainty. Use for reproduction of NSRR-dataset analyses "
        "(MESA, SHHS, CFS, CHAT)."
    ),
    citation="MESA Sleep PSG Scoring Manual, NSRR / Brigham Reading Center.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure_primary",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        nasal_pressure_fallback=False,
        square_root_linearisation=True,
        output_variants=["ahi_3pct", "ahi_3pct_arousal", "ahi_4pct"],
    ),
    post_processing=PostProcessingRules(
        # Ritmiek-as UIT: historisch/regulatoir dan wel bevroren.
        single_channel_rhythm=False,
        # Arousal-classifier om dezelfde reden UIT: een ML-filter maakt geen
        # deel uit van AASM v1/v2 of de CMS-regels, en de twee
        # dataset-profielen moeten paper v31/v37 reproduceren.
        arousal_lgbm=False,
        # Om dezelfde reden GEPIND op het artefactgedrag van vóór 23-08-2026.
        # De niet-gepinde v3-profielen negeren de artefactlijst nu (30/30 op
        # MESA), maar dat verschuift de arousaldetectie en dus de uitkomst van
        # een gereproduceerde regelset.
        arousal_uses_artifact_epochs=True,
        stability_filter_enabled=True,
        stability_filter_cv=0.45,
        csr_reclassification=True,
        local_baseline_validation=True,
        breath_level_detection=True,
        artefact_flank_exclusion=True,
        mixed_apnea_decomposition=True,
        unsure_as_hypopnea=True,  # ← NSRR-specific
        # GEPIND: ook de thermistorpoort blijft de oude, want die beslist of
        # apneus op de thermistor of op de neusdruk gescoord worden.
        thermistor_gate="envelope_agreement",
        # GEPIND op het gedrag van vóór 13-08-2026. Beide vlaggen staan
        # sindsdien default aan, maar dit profiel reproduceert paper v31/v37
        # en moet byte-identiek blijven. De poort aanzetten zou hier
        # `uncertain`-apneus in `ahi_total` trekken en de gepubliceerde
        # indices verschuiven; zie PostProcessingRules.rip_quality_scale_free.
        rip_quality_scale_free=False,
        # Idem voor de PLM-tijdbasis: die is per 21-08-2026 default aan omdat
        # het een rekenfout was, maar dit profiel moet reproduceerbaar blijven.
        plm_time_base=False,
        # En de arousal-classifier: paper v31/v37 draaide op het
        # regelgebaseerde pad.
        stability_filter_all_hypopnea_subtypes=False,
        local_baseline_cv_threshold=0.3,
        local_baseline_strict_reduction=25.0,
        # Paper v34 §S5.6 MESA tuning: relax local-baseline validator
        # and shrink Rule-1B arousal window for dense-arousal recordings.
        local_baseline_min_reduction_pct=15.0,
        local_baseline_pre_win_s=60.0,
        rule1b_arousal_window_s=5.0,
        # v0.5.1: shorter / lower-percentile baseline tracks local quiet
        # breathing more responsively on heterogeneous Compumedics
        # recordings (Lazazzera 2020, Koley 2014).
        baseline_window_s=120,
        baseline_percentile=85.0,
        # v0.5.1: when nasal pressure quality is poor, fall back to RIPsum.
        flow_fallback_strategy="ripsum_on_nasal_failure",
        # v0.6 prototype: targeted at the noisy-Pres-signal failure mode
        # documented on mesaid 6382 (paper v34 §S5.8). 3-second envelope
        # smoothing bridges brief noise excursions so that legitimate
        # ≥10s flow-reduction periods form continuous below-threshold
        # runs in the candidate-formation step. Removed from clinical
        # defaults in v0.2.8 because it caused +54 false hypopneas on
        # PSG-IPA SN1, but PSG-IPA uses aasm_v3_rec, not mesa_shhs;
        # this cohort-specific re-enablement is paper-defensible.
        flow_smoothing_s=3.0,
        # v0.6 prototype: lower the consecutive-breaths threshold for
        # peak-based detection on dense-event MESA recordings. Default
        # 3 matches AASM and the sensitive profile uses 2 already.
        peak_min_consecutive_breaths=2,
        # v0.6.0: LightGBM candidate re-classifier (paper v35 §S5.9).
        # Trained on q≥5∖q=7 stratum (n=653, ~210k candidates).
        # Threshold 0.65 = bias-near-zero operating point on q=7 holdout.
        ml_classifier_path="data/lightgbm_v06_q7holdout.txt",
        ml_threshold=0.65,
    ),
)

# ---- Chicago Criteria (1999, pre-AASM) ----
# Legacy criterion used in early epidemiological cohorts.
_chicago_1999 = Profile(
    name="chicago_1999",
    display_name="Chicago Criteria (1999, pre-AASM)",
    family="legacy",
    aasm_version="Chicago 1999 (pre-AASM)",
    aasm_rule="chicago",
    description=(
        "AASM Task Force 1999 'Chicago Criteria' for respiratory event "
        "scoring. Requires ≥50% flow reduction (stricter than modern "
        "AASM ≥30%), with ≥3% desaturation OR arousal. Used in the "
        "Wisconsin Sleep Cohort Study (WSCS) and other pre-2007 research "
        "cohorts. Use for historical re-analysis and cross-era cohort "
        "comparison."
    ),
    citation="AASM Task Force. Sleep 1999;22:667-689.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.50,
        sensor="nasal_pressure_or_flow",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Ritmiek-as UIT: historisch/regulatoir dan wel bevroren.
        single_channel_rhythm=False,
        # Arousal-classifier om dezelfde reden UIT: een ML-filter maakt geen
        # deel uit van AASM v1/v2 of de CMS-regels, en de twee
        # dataset-profielen moeten paper v31/v37 reproduceren.
        arousal_lgbm=False,
        # Om dezelfde reden GEPIND op het artefactgedrag van vóór 23-08-2026.
        # De niet-gepinde v3-profielen negeren de artefactlijst nu (30/30 op
        # MESA), maar dat verschuift de arousaldetectie en dus de uitkomst van
        # een gereproduceerde regelset.
        arousal_uses_artifact_epochs=True,
        # GEPIND op het gedrag van vóór 13-08-2026: dit profiel reproduceert
        # de Chicago-criteria van 1999 en hoort niet mee te bewegen met
        # reparaties aan de sensorpoorten. Zie
        # PostProcessingRules.rip_quality_scale_free.
        rip_quality_scale_free=False,
        plm_time_base=False,
        stability_filter_all_hypopnea_subtypes=False,
        thermistor_gate="envelope_agreement",
    ),
)

# ---- AASM v3 STRICT (exploratory) ----
# Conservative variant of v3_rec. Rejects borderline candidates.
_aasm_v3_dual = Profile(
    name="aasm_v3_dual",
    display_name="AASM v3 — Rule 1A, dual-sensor apneas",
    family="clinical",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED), dual-sensor",
    description=(
        "AASM v3 Rule 1A with apneas detected on BOTH flow sensors instead "
        "of choosing one. The nasal pressure misses mouth breathing; the "
        "thermistor is too slow for short events. Apneas seen on either "
        "sensor are kept, de-duplicated on overlap so the same event does "
        "not enter the AHI twice. Crucially the second sensor never REMOVES "
        "an event: when in doubt, the apnea stands. Hypopneas stay on the "
        "nasal pressure with their own baseline, per AASM."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Beide sensoren gebruiken, niet kiezen. Zie dual_sensor_apnea.
        dual_sensor_apnea=True,
        summary_after_reclassification=True,
        # Falsifieren staat UIT en dat is de kern van dit profiel. Op MESA
        # bevestigt maar 19% van 1785 apneu-detecties op beide sensoren;
        # afwijzen zou 81% weggooien op een detector die al onderdetecteert.
        # Bij twijfel wint goedkeuren.
        dual_sensor_corroboration=False,
        # De thermistor blijft hier het nominale apneukanaal, ook wanneer hij
        # de kwaliteitstoets niet haalt — de tweede detectiepas maakt dat
        # ongevaarlijk vóór de apneutelling. Maar de sweep, de anker-basislijn,
        # de arousal-koppeling, CSR en de ventilatory burden kennen die tweede
        # pas niet en zouden alsnog op een dood kanaal rekenen. Zij lezen de
        # neusdruk. Zie flow_reference.
        flow_reference="hypopnea",
    ),
)


# ---- AASM v3 dual-sensor met GEWOGEN fusie (experimenteel) ----
# aasm_v3_dual, maar de sensorovereenstemming telt als gewicht in plaats van
# als poort. `assess_flow_sensor_agreement` levert een continue
# envelope-correlatie — op echte opnames 0,32 tot 0,71 — en die werd
# gereduceerd tot usable ja/nee. Dat is dezelfde vorm als de arousal-as vóór
# v0.14.4: een drempel midden in een verder continu model.
#
# ⚠︎ NIET GEVALIDEERD. PSG-IPA heeft één flowkanaal en geen thermistor, dus
# deze as is daar principieel niet te meten — precies waarom aasm_v3_rec,
# aasm_v3_dual en aasm_v3_pressure er tot op de decimaal identiek zijn.
# Toetsen kan alleen op MESA of op een eigen cohort met twee flowsensoren.
_aasm_v3_fusion = Profile(
    name="aasm_v3_fusion",
    display_name="AASM v3 — dual-sensor, agreement-weighted (experimental)",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED), dual-sensor weighted",
    description=(
        "aasm_v3_dual with the sensor-agreement score used as a weight "
        "instead of a gate. The envelope correlation between thermistor and "
        "nasal pressure is already computed on every dual-sensor montage, "
        "then thrown away except for a usable yes/no. Here it stays: every "
        "apnea carries sensor_agreement, and an apnea whose only support is "
        "the thermistor has its confidence scaled by that value — so a "
        "detection from a sensor that agrees at 0.32 enters at a third of "
        "its confidence rather than either fully or not at all. Nothing is "
        "rejected; the thermistor remains additive. EXPERIMENTAL AND "
        "UNVALIDATED: this axis cannot be measured on PSG-IPA, which has a "
        "single flow channel."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        dual_sensor_apnea=True,
        dual_sensor_corroboration=False,
        summary_after_reclassification=True,
        flow_reference="hypopnea",
        thermistor_agreement_weighting=True,
    ),
)


# ============================================================
# Het kruisvlak: ademteug-gegradeerde hypopneeën + duale apneus
# ============================================================
# De profielen varieerden langs twee onafhankelijke assen — welke
# hypopneedetector draait, en op hoeveel flowsensoren apneus worden gezocht —
# en het vakje waar beide aan staan was leeg. Deze twee vullen dat.
#
# Afgeleid met `replace()` van hun ouder in plaats van gekopieerd, zodat een
# latere wijziging aan `_aasm_v3_breath` of `_aasm_v3_prob` automatisch
# doorwerkt. Twee handmatig gelijkgehouden kopieën lopen vroeg of laat uiteen,
# en dat zou hier stil gebeuren: het verschil zit in een operatiepunt, niet in
# een regel die een test vanzelf raakt.


def _with_dual_apneas(parent: Profile, *, name: str, display_name: str,
                      description: str) -> Profile:
    """Zelfde profiel, maar apneus op beide flowsensoren.

    Elke geneste dataclass wordt via ``replace()`` opnieuw gebouwd. Zonder dat
    zou het kind het HypopneaRules-object van de ouder delen — twee profielen
    met één mutabel object ertussen, en een mutatie die niemand verwacht.
    """
    return replace(
        parent,
        name=name,
        display_name=display_name,
        family="exploratory",
        aasm_rule=f"{parent.aasm_rule}, dual-sensor",
        description=description,
        hypopnea=replace(parent.hypopnea),
        apnea=replace(parent.apnea),
        spo2=replace(parent.spo2),
        post_processing=replace(
            parent.post_processing,
            # De tweede sensor is additief: hij voegt apneus toe en verwijdert
            # er nooit een. Zie corroborate_apnea_events.
            dual_sensor_apnea=True,
            dual_sensor_corroboration=False,
            # Niet de ademteugsegmentatie is hier de reden — die draait
            # ongeacht dit veld op de neusdruk (respiratory.py roept
            # _run_breath_analysis aan met hypop_flow). De reden is de
            # arousal-analyse: die leest ref_flow, en haar arousals zijn in de
            # gegradeerde detector de helft van de noisy-OR-bevestiging. Onder
            # een additieve thermistor wijst het apneukanaal naar de
            # thermistor, ook wanneer die de kwaliteitstoets niet haalt — de
            # tweede detectiepas dekt dat af vóór de apneutelling, maar de
            # arousal-koppeling kent die pas niet. Zie flow_reference.
            flow_reference="hypopnea",
            # De per-kanaal poort in plaats van de overeenstemmingsmaat.
            #
            # Waarom juist hier, en nergens anders. Deze twee profielen bestaan
            # om de tweede flowsensor te gebruiken, en de oude poort weigerde
            # die op 8 van 9 gemeten montages — waaronder drie waar de
            # thermistor 98% van zijn vermogen in de ademband heeft en zijn
            # ademfrequentie tot op 0,002 Hz met de neusdruk deelt. Met die
            # poort ervoor zijn deze profielen op vrijwel elke opname
            # identiek aan hun enkelsensor-ouder: dure no-ops.
            #
            # De klinische profielen houden de bestaande poort. Daar zou deze
            # keuze de AHI verschuiven op elke opname, en dat raakt mesa_shhs
            # en de gepubliceerde cijfers. Hier kan dat niet: beide profielen
            # zijn nieuw, exploratory en standaard uit.
            thermistor_gate="respiratory_band",
        ),
    )


_aasm_v3_breath_dual = _with_dual_apneas(
    _aasm_v3_breath,
    name="aasm_v3_breath_dual",
    display_name="AASM v3 — breath-graded + dual-sensor apneas (experimental)",
    description=(
        "aasm_v3_breath with apneas detected on BOTH flow sensors instead of "
        "choosing one. The two axes are independent: the breath-graded "
        "detector replaces only the hypopneas and takes its apneas unchanged "
        "from the envelope detector, so the second sensor operates on the "
        "list the graded detector does not touch. On a montage with a single "
        "flow channel this profile is identical to aasm_v3_breath, exactly as "
        "aasm_v3_dual is identical to aasm_v3_rec there. EXPERIMENTAL: the "
        "sensor axis cannot be measured on PSG-IPA, which has one flow "
        "channel and no thermistor, so this combination has not been "
        "validated against human scoring."
    ),
)


_aasm_v3_prob_dual = _with_dual_apneas(
    _aasm_v3_prob,
    name="aasm_v3_prob_dual",
    display_name="AASM v3 — graded arousal axis + dual-sensor apneas (experimental)",
    description=(
        "aasm_v3_prob with apneas detected on BOTH flow sensors. Keeps the "
        "graded arousal axis of its parent — arousal weight 0.70 with latency "
        "grading — and adds the second flow sensor to the apnea pass. Same "
        "independence and the same limitation as aasm_v3_breath_dual: "
        "identical to aasm_v3_prob on a single-flow montage, and unvalidated "
        "on the sensor axis. Both experimental axes at once, so this is the "
        "furthest from a measured operating point of any profile here."
    ),
)


# ---- AASM v3 NASAL-PRESSURE REFERENCE (clinical) ----
# aasm_v3_rec in every respect except which channel the derived analyses read.
_aasm_v3_pressure = Profile(
    name="aasm_v3_pressure",
    display_name="AASM v3 — Rule 1A, nasal-pressure reference",
    family="clinical",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM v3 Rule 1A, scored exactly as aasm_v3_rec — apneas on the "
        "oronasal thermistor where the montage has a usable one, hypopneas on "
        "nasal pressure. What differs is the FIVE analyses that take a flow "
        "signal without detecting apneas: the AHI robustness sweep, baseline "
        "anchoring, the arousal/RERA coupling, Cheyne-Stokes detection and the "
        "ventilatory burden. They read the nasal pressure here instead of the "
        "apnea channel. The AASM assigns quantitative flow assessment to nasal "
        "pressure and treats the thermistor as qualitative — able to show that "
        "flow stopped, not to grade how much of it is left. Identical to "
        "aasm_v3_rec on any montage without a usable thermistor."
    ),
    citation="Troester MM, Quan SF, Berry RB, et al. AASM Manual v3. 2023.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        # Apneudetectie ongewijzigd t.o.v. aasm_v3_rec — enkelsensor.
        dual_sensor_apnea=False,
        summary_after_reclassification=True,
        # De enige variabele die dit profiel verzet. Zie flow_reference.
        flow_reference="hypopnea",
    ),
)


_aasm_v3_strict = Profile(
    name="aasm_v3_strict",
    display_name="AASM v3 — Strict (conservative)",
    family="exploratory",
    aasm_version="v3 (2023), conservative variant",
    aasm_rule="1A strict",
    description=(
        "Conservative variant of AASM v3 Rule 1A. Retains the 3%-or-arousal "
        "qualifier but disables flow smoothing, breath-level detection, "
        "and uses a stricter stability filter (CV 0.30 instead of 0.45). "
        "Use for robustness-grade framework lower bound or for evaluators "
        "with a conservative manual scoring style."
    ),
    citation="Rombaut et al. 2026 (paper v31), §2.1.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    spo2=SpO2Rules(
        baseline_window_s=120.0,
        nadir_search_s=30.0,  # ← strict: conservative window (v0.3.2 behaviour)
        global_p95_fallback=True,
    ),
    post_processing=PostProcessingRules(
        stability_filter_enabled=True,
        stability_filter_cv=0.30,
        summary_after_reclassification=True,
        csr_reclassification=True,
        local_baseline_validation=True,
        flow_smoothing_s=0.0,
        breath_level_detection=False,
        artefact_flank_exclusion=True,
        mixed_apnea_decomposition=True,
        local_baseline_cv_threshold=0.3,
        local_baseline_strict_reduction=30.0,
    ),
)

# ---- AASM v3 SENSITIVE (exploratory, UARS) ----
# Permissive variant for UARS/borderline patients.
_aasm_v3_sensitive = Profile(
    name="aasm_v3_sensitive",
    display_name="AASM v3 — Sensitive (UARS)",
    family="exploratory",
    aasm_version="v3 (2023), sensitive variant",
    aasm_rule="1A sensitive",
    description=(
        "Sensitive variant of AASM v3 Rule 1A. Lowers the flow reduction "
        "threshold from 30% to 25%, extends flow smoothing to 5 s, "
        "and enables breath-level peak-based detection. Intended for "
        "use alongside aasm_v3_rec in symptomatic patients with "
        "borderline AHI where upper airway resistance syndrome (UARS) "
        "is suspected. Not for primary clinical scoring."
    ),
    citation="Rombaut et al. 2026 (paper v31), §2.1.",
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.25,  # ← key difference
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=90.0,  # v0.2.8: extended for UARS
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    apnea=ApneaRules(
        flow_reduction_threshold=0.90,
        sensor="thermistor",
        min_duration_s=10.0,
        max_duration_s=120.0,  # v0.2.8: extended for UARS
    ),
    post_processing=PostProcessingRules(
        stability_filter_enabled=True,
        stability_filter_cv=0.50,
        summary_after_reclassification=True,
        csr_reclassification=True,
        local_baseline_validation=True,
        flow_smoothing_s=5.0,
        breath_level_detection=True,
        peak_min_consecutive_breaths=2,  # ← v0.2.8: lower for UARS
        artefact_flank_exclusion=False,  # ← v0.3.2: CROSS_CONTAM_WINDOW_S = 0.0
        mixed_apnea_decomposition=True,
        local_baseline_cv_threshold=0.2,
        local_baseline_strict_reduction=20.0,
    ),
)


# ---- Envelope axis (exploratory) ----
# Four arms that differ from `aasm_v3_rec` in ONE respect: how the amplitude
# envelope of flow and effort is built. Every other rule is inherited verbatim
# from `aasm_v3_rec`, so a difference measured between any of these and the
# reference is attributable to the envelope and to nothing else.
#
# They exist because the analytic-signal transform is the largest per-worker
# allocation in the library (~500 MB per channel on an 8 h 256 Hz recording),
# and because "which envelope" turns out to be a methodological question in its
# own right: AASM words its rules as excursions per breath, while three of
# these four build a continuous envelope.
#
# None of them is a candidate for clinical use until it has been measured
# against human scoring. `aasm_v3_rec` remains the reference arm; nothing here
# changes it.

_ENVELOPE_ARM_HYPOPNEA = HypopneaRules(
    flow_reduction_threshold=0.30,
    sensor="nasal_pressure",
    min_duration_s=10.0,
    max_duration_s=60.0,
    desat_threshold=0.03,
    desat_required=False,
    arousal_required=False,
    desat_or_arousal=True,
    square_root_linearisation=True,
)

_aasm_v3_amplitude = Profile(
    name="aasm_v3_amplitude",
    display_name="AASM v3 — eenkanaals amplituderegel, pre-0.22 (experimental)",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "Identiek aan `aasm_v3_rec` op elke detectieregel; het enige verschil "
        "is dat de EENKANAALSFALLBACK op ritmiek beslist in plaats van op "
        "amplitude alleen. Raakt uitsluitend opnames waar de effort"
        "classificatie op één band terugvalt; waar de bilaterale analyse "
        "gewoon draait verandert er niets.\n\n"
        "De oude regel vergelijkt de mediane envelope tijdens het event met de "
        "P75 van de basislijn: onder 0,20 centraal, boven 0,50 obstructief, "
        "daartussen `uncertain` -- en die kale `uncertain` vallen buiten "
        "`ahi_total`. Die ene as gooit twee dingen op een hoop: een band die "
        "KLEINER wordt (bij obstructie bewegen thorax en abdomen in tegenfase, "
        "dus een enkele band zakt terwijl de inspanning doorgaat) en een band "
        "die AFWEZIG is.\n\n"
        "Gemeten op PSG-IPA met twaalf menselijke scorers per opname en "
        "geforceerde eenkanaalsmodus (de paarpoort gaat daar nooit af, dus "
        "anders meet je niets): subtype-overeenstemming 0,586-0,592 tegen "
        "0,609 bilateraal -- 97 % van het plafond met één band -- waar de "
        "amplituderegel op 0,140-0,199 blijft. Het aandeel `uncertain` gaat "
        "van 52-63 % naar 1-3 %."
    ),
    citation=(
        "Rombaut et al. 2026 — single-channel rhythm axis; zie "
        "docs/RITMIEK_AS_CRITERIUM.md voor de meting en het criterium, "
        "inclusief welke opnames verkennend en welke blind waren."
    ),
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        single_channel_rhythm=False,
    ),
)


_aasm_v3_pair_scalefree = Profile(
    name="aasm_v3_pair_scalefree",
    display_name="AASM v3 — RIP-paarpoort schaalvrij (experimental)",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "Identiek aan `aasm_v3_rec` op elke detectieregel; het enige verschil "
        "is dat de RIP-PAARPOORT geen effortkanaal meer afkeurt dat zijn "
        "eigen, schaalvrije kwaliteitstoets doorstaat. Bedoeld als meetarm "
        "voor `rip_pair_scale_free`, niet als klinisch profiel.\n\n"
        "De bestaande paarregel vergelijkt `breath_energy`, een absolute "
        "grootheid die kwadratisch meeschaalt met de amplitude, en verklaart "
        "bij `ratio > 100` het zwakste kanaal 'disconnected'. Twee banden met "
        "een andere versterking halen dat zonder defect. Op een klinische "
        "opname (19-08-2026) gaf dat ratio 1186x met een thorax die "
        "`status=ok` en ademfractie 0,740 had en zichtbaar bewoog.\n\n"
        "Wat de poort daar kostte, gemeten met dezelfde opname twee keer "
        "gescoord en alleen deze beslissing verschillend: 142 events in beide "
        "runs, Jaccard 1,000 -- geen detectie verandert -- maar 73 van de 142 "
        "kregen een ander label (49 central->obstructive, 16 "
        "uncertain->obstructive, 7 uncertain->central). De verdeling ging van "
        "89 centraal / 9 obstructief naar 47 / 74 en `ahi_total` van 20,0 "
        "naar 23,9."
    ),
    citation=(
        "Rombaut et al. 2026 — RIP pair gate scale dependence; zie de "
        "CHANGELOG-entry voor de meting en het vooraf vastgelegde criterium."
    ),
    hypopnea=HypopneaRules(
        flow_reduction_threshold=0.30,
        sensor="nasal_pressure",
        min_duration_s=10.0,
        max_duration_s=60.0,
        desat_threshold=0.03,
        desat_required=False,
        arousal_required=False,
        desat_or_arousal=True,
        square_root_linearisation=True,
    ),
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        rip_pair_scale_free=True,
    ),
)


_aasm_v3_env_chunked = Profile(
    name="aasm_v3_env_chunked",
    display_name="AASM v3 — envelope: chunked Hilbert",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM v3 Rule 1A with the analytic-signal envelope computed blockwise "
        "(30 min blocks, 60 s overlap discarded) instead of over the whole "
        "recording. Peak memory per channel drops from ~500 MB to ~30 MB. "
        "Intended as a low-memory arm for parallel sweeps, NOT as a "
        "replacement for the reference: the block transform is not identical "
        "to the full one — the interior residual floors around 1e-4 of the p95 "
        "envelope and does not shrink as the overlap grows, and the first and "
        "last samples differ by ~30% because an FFT wraps circularly."
    ),
    citation="Rombaut et al. 2026 — envelope axis (blockwise Hilbert); see the CHANGELOG entry for the measurement and the decision rule.",
    hypopnea=_ENVELOPE_ARM_HYPOPNEA,
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        envelope_method="hilbert_chunked",
    ),
)

_aasm_v3_env_rectify = Profile(
    name="aasm_v3_env_rectify",
    display_name="AASM v3 — envelope: rectify + lowpass",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM v3 Rule 1A with the envelope from AM demodulation — |x| through "
        "a 0.5 Hz zero-phase lowpass — instead of the analytic signal. Fully "
        "streamable and O(n) in memory. This is a different envelope, not a "
        "cheaper one: rectification leaves a ripple at the breathing rate and "
        "the flanks of a reduction are shaped by the filter, so event "
        "boundaries and the measured percentage reduction both move."
    ),
    citation="Rombaut et al. 2026 — envelope axis (rectify + lowpass); see the CHANGELOG entry for the measurement and the decision rule.",
    hypopnea=_ENVELOPE_ARM_HYPOPNEA,
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        envelope_method="rectify_lowpass",
    ),
)

_aasm_v3_env_breath = Profile(
    name="aasm_v3_env_breath",
    display_name="AASM v3 — envelope: breath amplitude",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM v3 Rule 1A with the envelope built from per-breath peak-to-"
        "trough amplitudes, interpolated onto the sample grid. Closest to how "
        "AASM words the rules — excursions per breath rather than a continuous "
        "envelope — and event boundaries fall on breaths by construction. The "
        "risk sits in peak detection at low amplitudes, i.e. during events: "
        "where breaths are missed the envelope interpolates straight across "
        "instead of dipping. Distinct from aasm_v3_breath, which changes the "
        "decision rule; this one changes only the envelope."
    ),
    citation="Rombaut et al. 2026 — envelope axis (breath amplitude); see the CHANGELOG entry for the measurement and the decision rule.",
    hypopnea=_ENVELOPE_ARM_HYPOPNEA,
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        envelope_method="breath_amplitude",
    ),
)

_aasm_v3_env_decimated = Profile(
    name="aasm_v3_env_decimated",
    display_name="AASM v3 — envelope: decimated Hilbert (10 Hz)",
    family="exploratory",
    aasm_version="v3 (2023)",
    aasm_rule="1A (RECOMMENDED)",
    description=(
        "AASM v3 Rule 1A with the signal decimated to 10 Hz before the "
        "analytic-signal transform, then interpolated back onto the original "
        "sample grid. The upstream bandpass passes nothing above 3 Hz while "
        "acquisition runs at 200–512 Hz, so 10 Hz keeps Nyquist well clear of "
        "the band while shrinking the transform 20–50×. The deviation from the "
        "reference is small but real: the anti-alias filter and the coarser "
        "raster each leave a fingerprint."
    ),
    citation="Rombaut et al. 2026 — envelope axis (decimated Hilbert); see the CHANGELOG entry for the measurement and the decision rule.",
    hypopnea=_ENVELOPE_ARM_HYPOPNEA,
    post_processing=PostProcessingRules(
        summary_after_reclassification=True,
        envelope_fs=10.0,
    ),
)


# ============================================================
# Registry
# ============================================================

PROFILES: Dict[str, Profile] = {
    # Clinical family
    "aasm_v3_rec":       _aasm_v3_rec,
    "aasm_v3_breath":    _aasm_v3_breath,
    "aasm_v3_prob":      _aasm_v3_prob,
    "aasm_v3_pressure":  _aasm_v3_pressure,
    "aasm_v3_dual":      _aasm_v3_dual,
    "aasm_v3_fusion":    _aasm_v3_fusion,
    "aasm_v3_breath_dual": _aasm_v3_breath_dual,
    "aasm_v3_prob_dual":   _aasm_v3_prob_dual,
    "aasm_v3_strict":    _aasm_v3_strict,
    "aasm_v3_sensitive": _aasm_v3_sensitive,

    # Envelope axis (exploratory) — identical to aasm_v3_rec except for how
    # the amplitude envelope is built.
    "aasm_v3_amplitude":     _aasm_v3_amplitude,
    "aasm_v3_pair_scalefree": _aasm_v3_pair_scalefree,
    "aasm_v3_env_chunked":   _aasm_v3_env_chunked,
    "aasm_v3_env_rectify":   _aasm_v3_env_rectify,
    "aasm_v3_env_breath":    _aasm_v3_env_breath,
    "aasm_v3_env_decimated": _aasm_v3_env_decimated,

    "aasm_v2_rec":       _aasm_v2_rec,
    "aasm_v1_rec":       _aasm_v1_rec,
    "cms_medicare":      _cms_medicare,

    # Dataset family
    "mesa_shhs":         _mesa_shhs,

    # Legacy family
    "chicago_1999":      _chicago_1999,
}

# ---- Profile groups (for confidence-interval / robustness-grade output) ----
PROFILE_GROUPS: Dict[str, List[str]] = {
    # Clinical robustness-grade: current paper v31 behaviour
    "clinical": ["aasm_v3_strict", "aasm_v3_rec", "aasm_v3_sensitive"],

    # Historical era comparison
    "aasm_era": ["aasm_v1_rec", "aasm_v2_rec", "aasm_v3_rec"],

    # Insurance / coverage awareness (US)
    "coverage": ["aasm_v3_rec", "cms_medicare"],

    # Dataset reproduction check (e.g., compare our v3_rec to NSRR-conv)
    "dataset":  ["aasm_v3_rec", "mesa_shhs"],

    # Envelope axis: the reference first, then the four arms. Everything except
    # the envelope is held constant across all five, so any difference measured
    # here is attributable to the envelope alone.
    "envelope": ["aasm_v3_rec", "aasm_v3_env_chunked", "aasm_v3_env_rectify",
                 "aasm_v3_env_breath", "aasm_v3_env_decimated"],

    # Maximum coverage: all 6 "standard" profiles
    "full_6":   ["aasm_v3_rec", "aasm_v2_rec", "aasm_v1_rec",
                 "cms_medicare", "mesa_shhs", "chicago_1999"],

    # Everything (research-grade sensitivity analysis)
    "all":      list(PROFILES.keys()),
}

# ---- Legacy name mapping (v0.3.x → v0.4.0) ----
LEGACY_ALIASES: Dict[str, str] = {
    "strict":    "aasm_v3_strict",
    "standard":  "aasm_v3_rec",
    "sensitive": "aasm_v3_sensitive",
}


# ============================================================
# Public API
# ============================================================

def get_profile(name: str) -> Profile:
    """
    Retrieve a scoring profile by name.

    Accepts legacy names (strict/standard/sensitive) with deprecation
    warning.

    Parameters
    ----------
    name : str
        Profile name. See list_profiles() for available options.

    Returns
    -------
    Profile
        Complete profile specification.

    Raises
    ------
    KeyError
        If the profile name is not recognised.

    Examples
    --------
    >>> profile = get_profile("aasm_v3_rec")
    >>> profile.hypopnea.flow_reduction_threshold
    0.3
    >>> profile = get_profile("standard")  # legacy alias → v3_rec
    >>> profile.name
    'aasm_v3_rec'
    """
    resolved = resolve_profile_name(name)
    if resolved not in PROFILES:
        raise KeyError(
            f"Unknown profile '{name}'. "
            f"Available: {', '.join(sorted(PROFILES.keys()))}"
        )
    return PROFILES[resolved]


def resolve_profile_name(name: str) -> str:
    """
    Resolve a profile name, handling legacy aliases.

    Emits a DeprecationWarning if a legacy name is used.
    """
    if name in LEGACY_ALIASES:
        new_name = LEGACY_ALIASES[name]
        warnings.warn(
            f"Profile name '{name}' is deprecated in psgscoring v0.4.0 "
            f"and will be removed in v0.5.0. Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return new_name
    return name


def list_profiles(family: Optional[str] = None) -> List[str]:
    """
    List available profile names.

    Parameters
    ----------
    family : str, optional
        Filter by family: 'clinical', 'dataset', 'legacy', 'exploratory'.
        If None, return all profiles.

    Returns
    -------
    list of str
        Profile names, sorted.
    """
    if family is None:
        return sorted(PROFILES.keys())
    return sorted(
        name for name, p in PROFILES.items() if p.family == family
    )


def list_profile_groups() -> Dict[str, List[str]]:
    """Return all predefined profile groups."""
    return {k: list(v) for k, v in PROFILE_GROUPS.items()}


def profile_metadata(name: str) -> Dict[str, Any]:
    """
    Return audit-ready metadata dict for a profile.

    Used by pipeline.py to attach profile information to every analysis
    output for full traceability.
    """
    p = get_profile(name)
    return {
        "profile_name":       p.name,
        "profile_display":    p.display_name,
        "profile_family":     p.family,
        "aasm_version":       p.aasm_version,
        "aasm_rule":          p.aasm_rule,
        "flow_threshold_pct": int(p.hypopnea.flow_reduction_threshold * 100),
        "desat_threshold_pct": (
            int(p.hypopnea.desat_threshold * 100)
            if p.hypopnea.desat_threshold else None
        ),
        "desat_or_arousal":   p.hypopnea.desat_or_arousal,
        "citation":           p.citation,
    }
