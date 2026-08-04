# AASM-scoringsregels — cheat sheet

*Eén pagina, voor intern gebruik en als bijlage voor externe centra.*
*Bron: AASM Manual for the Scoring of Sleep and Associated Events, v3 (2023),
Troester MM, Quan SF, Berry RB et al. Waar v2.6 afwijkt, staat dat erbij.*

Dit blad beschrijft de **regels**. Wat `psgscoring` er per profiel van
implementeert staat in `psgscoring/profiles.py`; die docstrings zijn de
normatieve bron voor het gedrag van de software.

---

## Apneu

| | |
|---|---|
| **Sensor** | oronasale thermistor (v3: ook PVDT of RIPsum als vervanger) |
| **Amplitude** | ≥ **90 %** daling t.o.v. pre-event baseline |
| **Duur** | ≥ **10 s**, gemeten over het deel dat aan het amplitudecriterium voldoet |
| **Desaturatie** | **niet vereist** |
| **Arousal** | **niet vereist** |

**Type** — bepaald door ademarbeid tijdens de gebeurtenis:

- **obstructief** — ademarbeid blijft gedurende de hele gebeurtenis aanwezig
- **centraal** — geen ademarbeid gedurende de hele gebeurtenis
- **gemengd** — eerst geen, daarna wél ademarbeid

De thermistor is **kwalitatief**: hij toont dát de flow stopt, niet hoeveel
ervan over is. Daarom scoort de AASM apneus erop en hypopneeën niet.

## Hypopneu

| | Rule 1A (RECOMMENDED) | Rule 1B (ACCEPTABLE / CMS) |
|---|---|---|
| **Sensor** | nasale druk (kwantitatief) | idem |
| **Amplitude** | ≥ 30 % daling | ≥ 30 % daling |
| **Duur** | ≥ 10 s | ≥ 10 s |
| **Gevolg** | ≥ **3 %** desaturatie **óf** een arousal | ≥ **4 %** desaturatie |

De keuze tussen 1A en 1B is geen detail: 1B levert stelselmatig een lagere AHI,
en de AHI is de drempel voor terugbetaling en behandeling. Een rapport hoort te
zeggen welke regel gebruikt is.

**Nasale druk is niet-lineair** met flow. De AASM staat √-linearisatie toe; doe
je dat, dan verandert de betekenis van "30 % daling" en moet elke daarop
gekalibreerde correctie opnieuw. Zie de openstaande architectuurschuld in de
changelog.

## Vervangsensoren

Faalt een primaire sensor, dan mag een vervanger gebruikt worden — maar de
substitutie hoort in het rapport te staan, want ze verandert de gevoeligheid:

| Primair | Vervanger (v3) |
|---|---|
| oronasale thermistor (apneu) | nasale druk (met of zonder √), PVDT, RIPsum |
| nasale druk (hypopneu) | oronasale thermistor, PVDT, RIPflow, RIPsum |

Apneus op de nasale druk scoren **overdetecteert** ten opzichte van de
thermistor: de druk daalt sneller naar nul dan de temperatuur.

## Arousal

| | |
|---|---|
| **Duur** | ≥ **3 s** abrupte EEG-frequentieverschuiving (alfa, thèta of > 16 Hz, geen spindels) |
| **Voorwaarde** | ≥ **10 s** stabiele slaap ervóór |
| **In REM** | daarbij ≥ **1 s** toename van de kin-EMG |

Zonder kin-EMG is REM-arousalscoring niet AASM-conform. Been-EMG (tibialis
anterior) is **geen** vervanger — dat is een andere spier met een ander doel.

## RERA en RDI

**RERA** — ≥ 10 s toenemende ademarbeid of afvlakking van de neusdrukcurve,
eindigend in een arousal, die niet aan de apneu- of hypopneucriteria voldoet.

```
AHI = (apneus + hypopneeën) / uur slaap
RDI = (apneus + hypopneeën + RERA's) / uur slaap
OAHI = (obstructieve + gemengde apneus + obstructieve hypopneeën) / uur slaap
```

Let op de noemer: **uur slaap** (TST), niet uur registratie. Verschuift de
staging, dan verschuift de AHI mee zonder dat er één event verandert.

## Baseline

De AASM definieert de baseline **pre-event**: het gemiddelde van de twee minuten
vóór de gebeurtenis, over stabiele ademteugen — of, bij instabiele ademhaling,
het gemiddelde van de drie grootste ademteugen in die periode.

`psgscoring` gebruikt standaard een **gecentreerd** venster van 5 minuten (P95
van de envelope). Dat is niet hetzelfde; de pre-event-variant zit achter
`baseline_mode="pre_event"` en is nog niet gevalideerd als default.

## Cheyne-Stokes-ademhaling

Crescendo-decrescendopatroon met ≥ 3 opeenvolgende cycli, én minstens één van:

- ≥ 5 centrale apneus/hypopneeën per uur slaap, of
- het patroon houdt ≥ 10 opeenvolgende minuten aan.

## Wat dit blad **niet** dekt

Hypoventilatie (transcutane of end-tidal CO₂), PLM-scoringsregels,
slaapstadiëring zelf, en pediatrische criteria — die wijken op meerdere punten
af van het bovenstaande.
