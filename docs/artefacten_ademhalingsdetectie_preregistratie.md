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
