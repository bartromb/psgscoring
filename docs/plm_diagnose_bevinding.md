# Waar het PLM-gat zit: onderdetectie, niet valse events

*24 augustus 2026. Diagnose vóór enige ingreep — precies de stap die bij
arousals drie keer is overgeslagen.*

## Aanleiding

Na de tijdbasisreparatie staat de PLM-detectie op event-F1 **0,692** tegen
**0,820** scoorder-onderling. Dat gat is nooit uitgesplitst.

Bij arousals bleek diezelfde vraag beslissend: drie ingrepen sneuvelden omdat
niemand had gekeken wáár de fout zat, en toen de diagnose er lag bleek het
**precisie** te zijn — waarna twee reparaties wél werkten. Dus eerst kijken.

## Uitkomst — het spiegelbeeld van arousals

PSG-IPA, twaalf scoorders, volle lijst (de payloadgrens van 200 staat af).

| | gedetecteerd | scoordermediaan | precisie | recall |
|---|---:|---:|---:|---:|
| SN1 | 202 | 420 | 0,713 | 0,347 |
| SN2 | 135 | 204 | 0,785 | 0,510 |
| SN3 | \phantom{0}97 | 430 | 0,722 | 0,155 |

**Onze events kloppen; er zijn er te weinig.**

| onze events gedekt door | mediaan |
|---|---:|
| ≥ 1 scoorder | **1,000** |
| ≥ 3 scoorders | 1,000 |
| ≥ 6 scoorders | 0,985 |
| ≥ 9 scoorders | 0,938 |

Recall per instemmingsniveau:

| scoorders het eens | mediane recall |
|---|---:|
| 7/12 | 0,000 |
| 9/12 | 0,231 |
| 11/12 | 0,361 |
| **12/12** | **0,636** |

## Wat dat betekent

Bijna elk event dat wij markeren, markeert een scoorder ook — en 94 % wordt
door minstens negen van de twaalf gedekt. Er is dus **geen** valse-positieven-
probleem.

Maar zelfs waar **alle twaalf** scoorders het eens zijn, vinden we er maar
**64 %**, en bij 11 van 12 nog maar 36 %. Op SN3 detecteren we 97 bewegingen
waar de scoorders er 430 zien — minder dan een kwart.

**De detector is te ongevoelig.** Dat is de tegenovergestelde diagnose van
arousals, waar 16 % van onze events door geen enkele scoorder gedekt werd en
de reparatie in strengheid zat. Hier moet het de andere kant op, en een
ingreep die op arousals werkte zou hier averechts uitpakken.

## Waar te kijken

De AASM-definitie vraagt een EMG-toename van ≥ 8 µV boven de basislijn
gedurende 0,5–10 s. Drie kandidaten, in volgorde van vermoedelijke opbrengst:

1. **De amplitudedrempel zelf** — staat die op 8 µV, en waartegen wordt
   "basislijn" gemeten? Een te hoog geschatte basislijn maakt de effectieve
   drempel hoger dan de definitie.
2. **De duurgrenzen** — bewegingen korter dan de ondergrens vallen weg; de
   scoorders lijken kortere te accepteren.
3. **De basislijnschatting zelf** — als die per nacht globaal is in plaats van
   rollend, tilt een onrustige periode de drempel voor de hele nacht op.

Geen daarvan is nagekeken; dit document stelt alleen de richting vast.

## Wat deze diagnose NIET zegt

Dat meer gevoeligheid de F1 verbetert. Meer events verlagen de precisie, en
die staat nu op 0,71–0,79. Wat vaststaat is dat het gat aan de recall-kant zit
en dat er ruimte is: bij precisie 0,72 en recall 0,16 (SN3) is de afruil zeer
scheef verdeeld.

Een ingreep hoort dezelfde discipline te krijgen als de arousaldrempel:
vooraf vastgelegde uitkomstmaat, kalibratie- en validatiesteekproef
gescheiden, en replicatie op een tweede cohort.

---

# CORRECTIE — 24 augustus 2026, dezelfde nacht

**De diagnose hierboven overdrijft de onderdetectie, door een fout van mij in
de vergelijking.**

`result["events"]` is `plm_eligible`: alle bewegingen **minus waak** en minus
respiratoir-geassocieerde. De PSG-IPA-scoorders annoteren **alle**
beenbewegingen, ook in waak. Ik heb dus een deelverzameling naast een superset
gelegd, en het verschil dat ik als onderdetectie las is voor een groot deel het
waakaandeel.

Zelfde soort fout als de harnasteller die op de substring `"apnea"` zocht:
twee dingen vergeleken die niet hetzelfde zijn.

## De gecorrigeerde vergelijking

Beide kanten op slaap gefilterd, met hetzelfde hypnogram:

| | scoorder totaal | scoorder in slaap | onze | F1 | precisie | recall |
|---|---:|---:|---:|---:|---:|---:|
| SN1 | 420 | 221 | 202 | 0,670 | 0,710 | 0,640 |
| SN2 | 204 | 162 | 135 | 0,706 | 0,785 | 0,625 |
| SN3 | 430 | 139 | \phantom{0}97 | 0,581 | 0,722 | 0,458 |
| SN4 | 921 | 669 | 558 | 0,799 | 0,869 | 0,730 |
| **SN5** | 584 | 254 | \phantom{0}**36** | **0,197** | 0,833 | **0,112** |

Mediane F1 **0,670** — vier van de vijf opnames zitten op 0,58–0,80. Er is dus
**geen systemische ongevoeligheid**; die conclusie was een artefact van mijn
vergelijking.

## Wat er wél staat: SN5

Op SN5 vindt de detector **479** bewegingen in totaal tegen 594 bij scoorder 1
— vergelijkbaar. Maar in slaap: **36 tegen 217**. Terwijl 37 % van de
scoorder-events in slaap valt, valt maar 8 % van de onze daar.

Onze detecties zitten dus onevenredig in **waak**. De voor de hand liggende
verklaring is amplitude: in slaap, en zeker in REM-atonie, zijn bewegingen
kleiner, en de drempel is `10e percentiel van de RUMS over de HELE nacht + 8 µV`
— één globaal getal voor een grootheid die per stadium verschilt.

Een rollende basislijn over 120 s geeft over de hele nacht mínder events
(0,83–0,98×) en is dus niet de oplossing; wat niet gemeten is, is een
**stadium-afhankelijke** rustwaarde. Dat is de volgende toets, niet een
conclusie.

## Wat hiervan blijft staan

Dat de **precisie hoog is** (0,71–0,87) en dat vrijwel elk event dat wij
markeren door een scoorder gedekt wordt. Dat deel van de eerste diagnose is
niet geraakt door de fout: het rekende met onze events als noemer, niet met de
hunne.

---

## Twee hypotheses getoetst en beide WEERLEGD (24-08-2026, nacht)

**1. Rollende basislijn.** De arousaldetector gebruikt er sinds v0.8.11 één,
met de motivering dat een nachtgemiddelde bij gefragmenteerde slaap misleidt;
de PLM-detector gebruikt één globaal 10e percentiel. Getoetst over 120 s
vensters: de rollende variant geeft **minder** events (0,83–0,98×), niet meer.
Niet de verklaring.

**2. Stadium-afhankelijke rustwaarde.** Als de rust-EMG in slaap lager ligt dan
nachtbreed, staat de drempel daar te hoog. Gemeten per stadium:

| | W | N1 | N2 | N3 | R |
|---|---:|---:|---:|---:|---:|
| SN5 p10 (µV) | 3,11 | 3,26 | 3,46 | 3,78 | 3,74 |
| SN4 p10 (µV) | 5,65 | 5,38 | 5,42 | 5,69 | 5,55 |
| SN1 p10 (µV) | 2,08 | 1,29 | 1,28 | 1,16 | 1,40 |

De rustwaarde is vrijwel stadium-onafhankelijk, en een stadium-lokale drempel
verschuift de telling met enkele procenten. Niet de verklaring.

## Wat er nu precies openstaat op SN5

Op één been telt het ruwe criterium **411** bursts in waak en **136** in
slaapstadia — dus 25 % in slaap. Maar `analyze_plm` rapporteert voor SN5
`n_lm_total` 479 en `n_lm_sleep` **36**, ofwel 8 %.

Die twee cijfers zijn niet met elkaar te rijmen: bilaterale samenvoeging kan de
telling verlagen, maar niet de slaap/waak-**verhouding** van 25 % naar 8 %
duwen. Er gaat dus iets mis tussen burstdetectie en stadiumtoewijzing dat noch
de basislijn noch het stadium is.

**Dat is de volgende plek om te kijken**, en het is een gerichte vraag in
plaats van een vermoeden: reconstrueer voor SN5 de `all_lms`-lijst met hun
`onset_s`, en vergelijk hun stadiumverdeling met die van de ruwe bursts. Wijkt
die af, dan zit de fout in de tijd-naar-epoch-omzetting van `analyze_plm` en
niet in de detectie.

**Wat ik hieruit NIET concludeer:** dat er een defect is. Twee van mijn
verklaringen zijn vannacht al gesneuveld, en de derde is een discrepantie die
ik nog niet heb kunnen narekenen. De vier andere opnames halen F1 0,58–0,80 en
geven geen aanleiding tot zorg.
