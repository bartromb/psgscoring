# Preregistratie — mag een EEG-artefact de ademhalingsdetectie schrappen?

*23 augustus 2026, vóór de meting.*

## Wat er nu gebeurt

`build_sleep_mask` sluit artefact-epochs uit, en dat masker poort **zowel de
apneu- als de hypopneudetectie** (`respiratory.py:503-504`). Op MESA gaat het
om mediaan **19,9 %** van de nacht, met uitschieters tot 53,6 %.

De vlag komt uit `run_artifact_detection(raw, all_eeg_channels)` — hij wordt
**op de EEG-kanalen** berekend, met een absolute drempel van 500 µV piek.

## De vraag

Bij arousals was uitsluiten nog verdedigbaar: slecht EEG, dus geen betrouwbare
arousalscoring. Die redenering geldt hier niet. Een zweetartefact, een losse
EEG-elektrode of tandenknarsen zet het epoch op artefact, en daarmee vervalt de
apneudetectie op de **neusdruk** en de desaturatie op de **saturatiemeter** —
twee signalen die van dat EEG niets merken.

Het raakt bovendien teller én noemer: de events vervallen, én de epochs gaan
uit de slaaptijd waardoor de AHI gedeeld wordt. Welke kant dat op werkt volgt
niet uit de redenering.

## Voorwerk dat eerst moest

`validate_mesa.py` gaf **helemaal geen** `artifact_epochs` door, terwijl
productie ze wél doorgeeft. Elke MESA-meting tot nu toe draaide dus op een
pipeline die op dit punt niet was wat er draait — en mijn RDI-meting van
23-08-2026 gaf daardoor stil nul verschil op 30/30, wat ik eerst voor een
uitkomst aanzag.

Gerepareerd achter `PSGSCORING_HARNESS_ARTIFACT_EPOCHS=1`, **default uit**,
zodat eerdere metingen vergelijkbaar blijven.

## Opzet

MESA n=30, zaad 20260824, gepaard, volle pipeline, profiel `aasm_v3_rec`
(waar RDI = AHI, dus de AHI staat zuiver in beeld) én `aasm_v3_breath`
(RERA-dragend, dus daar komt de RDI-impact van de zojuist omgezette
arousalvlag mee).

| arm | artefact-epochs aan de pipeline |
|---|---|
| A | geen (zoals alle eerdere metingen) |
| B | wél, volgens de productieregel |

Arm B is dus wat er **nu in productie draait**; arm A is wat we tot nu toe
gemeten hebben. Dat verschil op zichzelf is al een bevinding.

## Maten

Event-F1 tegen `aasm15` (IoU 0,20), AHI-bias, en het aandeel opnames met een
verschoven AHI-ernstklasse. Voor `aasm_v3_breath` daarnaast de RDI en de
RDI-ernstklasse.

## Beslisregel — vooraf

Dit is in de eerste plaats een **karakterisering**: hoe ver staat wat we meten
van wat we draaien? Daar hoort geen slaag/zak bij.

Eén vervolgvraag krijgt hem wel. Blijkt arm A (geen uitsluiting) beter op
event-F1 met mediane gepaarde ΔF1 ≥ **+0,010**, dan wordt voorgesteld de
**ademhalingsstap** de artefactlijst te laten negeren, achter een profielvlag
met het huidige gedrag als default en de vijf gepinde profielen ongemoeid —
dezelfde vorm als de arousalvlag.

**Weerlegd** bij ΔF1 ≤ 0: dan is uitsluiten daar wél zinnig en blijft het.

## Meldgrens

Verschuift de AHI-ernstklasse op meer dan een kwart van de opnames tussen A en
B, dan leg ik dat apart voor. Dat zou betekenen dat onze gepubliceerde
MESA-cijfers systematisch van de productie-uitkomst afwijken.

## Wat dit niet uitwijst

Of een artefactvlag **per signaalgroep** beter is — EEG-artefact alleen de
EEG-stappen laten raken, flow-artefact alleen de flow. Dat is de eigenlijke
reparatie als arm A wint, maar het is een andere en grotere ingreep dan wat
hier gemeten wordt.

---

# Uitkomst — 23 augustus 2026

**Niet aangenomen.** De ademhalingsstap blijft de artefactlijst gebruiken.

Gepaard, MESA n=30, zaad 20260824. Arm A = geen artefact-epochs (zoals alle
eerdere metingen), arm B = mét, volgens de productieregel.

## `aasm_v3_rec` (cascade, RDI = AHI)

| | A: geen | B: productie | verschil |
|---|---:|---:|---:|
| event-F1 | 0,362 | 0,316 | −0,046 |
| precisie | 0,456 | 0,474 | **+0,017** |
| recall | 0,348 | 0,304 | −0,044 |
| AHI mediaan | 16,0 | 16,9 | +0,9 |
| events mediaan | 93 | 74 | −19 |
| AHI-bias | −4,32 | **−3,89** | +0,43 |

Gepaarde ΔF1 (A − B) **+0,0084**, A beter op 19/30, p = 1,5·10⁻⁴.
Regel: aannemen vanaf +0,010 → **onbeslist**, net eronder.

## `aasm_v3_breath` (gegradeerd)

| | A: geen | B: productie | verschil |
|---|---:|---:|---:|
| event-F1 | 0,444 | 0,444 | 0,000 |
| AHI mediaan | 17,85 | 19,45 | +1,60 |
| events mediaan | 93 | 93 | 0 |
| RDI mediaan | 33,0 | 34,2 | +1,2 |

Gepaarde ΔF1 **0,0000**, p = 0,083 → **weerlegd**.

Dat de gegradeerde tak nauwelijks beweegt is consistent met wat al bekend was:
`score_hypopneas_breathwise` krijgt het RUWE signaal via `_run_breath_analysis`
en filtert zelf, dus het slaapmasker raakt die tak veel minder.

## Wat dit betekent, ook tegen mijn eigen verwachting in

Mijn analyse noemde het een categoriefout dat een EEG-artefact de
ademhalingsdetectie schrapt. Als **redenering** klopt dat nog steeds — maar de
**omvang** valt tegen. Anders dan bij arousals (F1 0,421 → 0,338, 30/30
slechter) is het hier ongeveer een wisselwerking: uitsluiten kost recall en
koopt precisie, en de **AHI-bias wordt er zelfs iets beter van** (−4,32 →
−3,89).

Dat is een eerlijker uitkomst dan de mijne. De vooraf vastgelegde regel wordt
niet gehaald en er verandert dus niets.

## De karakteriseringsvraag: hoe ver staat meten van draaien?

| | verschil A → B |
|---|---|
| AHI-ernstklasse `aasm_v3_rec` | 4/30 = **13 %** |
| AHI-ernstklasse `aasm_v3_breath` | 6/30 = **20 %** |
| RDI-ernstklasse `aasm_v3_breath` | 3/30 = 10 % |

Alle drie onder de meldgrens van 25 %, maar geen ervan is nul. Onze
MESA-cijfers zijn dus **representatief maar niet identiek** aan wat de kliniek
krijgt; de AHI-mediaan scheelt 0,9 tot 1,6/u. Dat hoort bij een paper vermeld
te worden, niet stilzwijgend weggelaten.

Vanaf nu kan het harnas beide: `PSGSCORING_HARNESS_ARTIFACT_EPOCHS=1` draait
zoals productie, default uit zodat eerdere metingen vergelijkbaar blijven.

## Wat overeind blijft van de flow-analyse

Deze meting zegt niets over **flow**-artefacten. Wat daar gemeten is, staat
los en blijft staan: `_detect_signal_gaps` drempelt op een absolute `1e-5` en
vuurt daardoor op de neusdruk van `mesa-sleep-0001` op 0,00 % van de samples,
tegen 6,44 % op de thermistor — dezelfde opname, alleen een andere
amplitudeschaal. Dat mechanisme is dood voor het kanaal waar apneus op gescoord
worden, en dat is een aparte reparatie met een eigen preregistratie.
