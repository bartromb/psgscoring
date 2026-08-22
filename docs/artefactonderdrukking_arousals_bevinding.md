# Artefactonderdrukking kost arousals en levert geen precisie op

*22 augustus 2026. Aanleiding: de vraag of artefacten eerst onderdrukt moeten
worden. Antwoord: dat gebeurt in productie al, en op deze meting schaadt het.*

## Eerst de diagnose die de vraag scherp maakte

Drie ingrepen op de arousaldetector zijn dit jaar weerlegd zonder dat iemand
naar de FOUTSTRUCTUUR had gekeken. Die is nu bekeken, op PSG-IPA met twaalf
scoorders per opname. Recall per instemmingsniveau:

| scoorders het eens | mediane recall | gebieden |
|---|---:|---:|
| 1/12 | 0,171 | 337 |
| 3/12 | 0,333 | 121 |
| 6/12 | 0,500 | 60 |
| 9/12 | 0,769 | 68 |
| 11/12 | 0,857 | 116 |
| 12/12 | **0,862** | 177 |

**De detector vindt de arousals waar de scoorders het over eens zijn.** 86 %
bij 11 en 12 van de 12. Wat hij daarnaast doet is events produceren die
niemand markeerde: van onze events wordt **83,8 %** door minstens één scoorder
gedekt, 52,6 % door minstens zes. Het gat naar 0,692 is dus **precisie**, geen
localisatie — een belangrijke correctie op hoe ik het eerder omschreef.

## De vraag: helpt artefactonderdrukking daartegen?

De detector slaat artefact-epochs al volledig over, en de pipeline geeft ze
door: YASAFlaskified berekent ze bij het stadiëren
(`yasa_analysis.py:589` — piek > 500 µV of vlak) en `tasks.py:304` zet ze om
naar epoch-indices. **Productie doet dit dus nu al.**

Geen enkel PSG-IPA-harnas heeft ze ooit meegegeven. Alle arousalcijfers tot nu
toe — inclusief de 0,505 waarop de hybride is aangezet — zijn ZONDER
artefactonderdrukking gemeten.

## Uitkomst

Vijf opnames, dezelfde regel als productie:

| | F1 | precisie | recall | gedekt door ≥1 scoorder |
|---|---:|---:|---:|---:|
| zonder onderdrukking | **0,505** | 0,425 | 0,649 | 0,838 |
| met onderdrukking | **0,484** | 0,429 | 0,510 | 0,834 |

Gepaarde ΔF1 **−0,084**, slechter op **5 van 5**. Mediaan 6,9 % van de epochs
wordt onderdrukt.

Per opname, en het patroon is systematisch:

| | artefact | F1 | recall |
|---|---:|---:|---:|
| SN1 | 0,8 % | 0,505 → 0,504 | 0,649 → 0,642 |
| SN2 | 6,3 % | 0,355 → 0,221 | 0,508 → 0,308 |
| SN3 | 13,7 % | 0,390 → **0,186** | 0,383 → 0,142 |
| SN4 | 6,9 % | 0,568 → 0,484 | 0,691 → 0,510 |
| SN5 | 9,8 % | 0,678 → 0,654 | 0,834 → 0,720 |

Hoe meer epochs de regel onderdrukt, hoe groter de schade. Op SN1 (0,8 %)
verandert er niets; op SN3 (13,7 %) **halveert de F1**.

## Waarom het niet werkt

**De precisie beweegt niet.** 0,425 → 0,429, en het aandeel van onze events
dat geen enkele scoorder zag blijft op ~16 % (0,838 → 0,834). De onderdrukte
events waren dus **niet** de verzonnen events — het waren de goede.

Dat is mechanisch te begrijpen. De regel vlagt een epoch bij een piek boven
500 µV, en een arousal gáát vaak samen met spieractiviteit en beweging. De
regel selecteert daarmee juist de epochs waar arousals zitten. De scoorders
markeren daar wél events; wij kijken er niet meer.

## Wat dit betekent voor productie

Dit draait nu. Op PSG-IPA-achtige data onderdrukt de regel mediaan 6,9 % van
de epochs en kost dat arousal-recall zonder iets terug te geven. Dat raakt de
arousal-index, en via de RERA-koppeling de RDI op de vier
`arousal_limb_wired`-profielen.

**Niet eenzijdig gewijzigd.** De artefactvlag doet meer dan arousals: hij komt
ook in TST-noemers en in andere stappen terecht, en hem daar weghalen
verschuift indices die niets met dit probleem te maken hebben. Wat hier
voorligt is een gerichte keuze: **de arousalstap de artefactlijst niet laten
gebruiken**, of een nauwer criterium hanteren dat spierbursts niet met
elektrode-artefact verwart.

## Grenzen

Vijf opnames, één montagetype, en de drempel van 500 µV is op AZORG-data
gekalibreerd noch hier opnieuw bekeken. Wat vaststaat is de RICHTING —
slechter op 5 van 5, monotoon met het onderdrukte percentage — niet de
precieze omvang.

## Wat hierna hoort

Een preregistratie met één vraag: gebruikt de arousalstap de artefactlijst wel
of niet? Uitkomstmaat vooraf kiezen (F1 of recall), profielvlag met huidig
gedrag als default, en meten op MESA erbij — want daar is de referentie
onafhankelijk van deze vijf opnames.
