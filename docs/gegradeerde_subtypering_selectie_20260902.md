# De uitgerolde s=0,25 stond op een set die op centrale apneus was uitgezocht

*Gemeten 2026-09-02 op Obelix, 161 opnames.*

## Wat er aan de hand is

De uitrol van `shape_evidence_scale = 0,25` rustte op een MESA-run die
"REPLICATIE: GESLAAGD" meldde. Die run gebruikte een `REC_LIST` van **32
opnames die zijn gekozen omdat ze veel centrale apneus bevatten**
(`mesa_centraal.txt`, begint met mesa-sleep-1790, -1863, -1604: 67, 67 en 62
centrale apneus).

Dat is geen fout op zichzelf — je hebt centrale apneus nodig om centrale
recall te meten. Maar het maakt de uitkomst niet generaliseerbaar naar een
onselecte populatie, en zo is hij wel gelezen.

## De twee runs naast elkaar

| | verrijkte set (32 opnames) | onselecte set (161 opnames) |
|---|---|---|
| huidig gedrag | 30,6 % / 88,9 %, κ 0,195 | 24,1 % / 93,4 %, κ 0,210 |
| **s = 0,25** | **61,4 % / 71,9 %, κ 0,260** | **39,8 % / 89,6 %, κ 0,281** |
| s = 0,3 | 68,1 % / 66,5 %, κ 0,254 | 46,1 % / 86,7 %, κ 0,280 |
| oordeel | GESLAAGD | criterium 2 niet gehaald |

*(recall centraal / recall obstructief)*

## Wat wél en niet repliceert

**Criterium 1 — hogere kappa: repliceert ruim.** 0,210 → 0,281 op de onselecte
set, een relatieve winst van 34 %. Dat is de kern van de wijziging en die staat.

**Criterium 2 — beide recalls ≥ 60 %: repliceert niet.** De centrale recall
komt op 39,8 % in plaats van 61,4 %. Geen enkel werkpunt haalt de 60 % op een
onselecte set; s = 0,3 komt tot 46,1 %.

De verklaring ligt voor de hand: opnames die zijn uitgezocht op véél centrale
apneus dragen vaak periodieke ademhaling, en daar is een centrale apneu
morfologisch overduidelijk. In een gewone populatie zijn centrale apneus
zeldzamer én dubbelzinniger.

## Waarom de conclusie ondanks 8 % overlap staat

13 van de 161 opnames zitten óók in de verrijkte set. Die overlap trekt de
centrale recall van vandaag **omhoog**, want het zijn juist de opnames met de
duidelijkste centrale apneus. De 39,8 % is dus eerder een bovengrens dan een
onderschatting, en de conclusie geldt a fortiori. Een volledig disjuncte run
van 200 verse opnames draait ter bevestiging.

## Wat dit betekent voor de uitrol

**Geen terugrol.** Op een onselecte populatie vindt s = 0,25 nog altijd 39,8 %
van de menselijke centrale apneus tegen 24,1 % daarvoor, en dat kost aan
obstructieve recall slechts 93,4 % → 89,6 %. Dat is een reële verbetering: meer
centrale apneus gevonden tegen een kleine prijs, met een hogere kappa.

Wat niet klopt is de **verantwoording**. De claim "beide recalls boven 60 %"
geldt alleen op een set die op centrale apneus is uitgezocht, en hoort zo
opgeschreven te staan — in de release-aantekening, in het geheugen en in elk
paperfragment dat dit werkpunt noemt.

## De les

Een `REC_LIST` die een cohort verrijkt is een legitiem instrument, maar hij
verandert wat "repliceert" betekent. Zo'n selectie hoort in de uitkomst te
staan, niet alleen in het bestand ernaast. Dit is dezelfde klasse fout als het
snurkkanaal en de kanaalnamen van vandaag: de meting mat iets anders dan waar
de conclusie over ging.
