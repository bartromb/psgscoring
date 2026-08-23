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
