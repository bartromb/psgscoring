# Wat de nacht van 1 op 2 september opleverde

**Doel:** detectie van apneus, hypopneus en arousals goed genoeg voor klinisch
gebruik.

Vier metingen, elk met een eigen conclusie. De derde is de belangrijkste en de
minst verwachte.

---

## 1. De subtypering: van κ 0,139 naar 0,431

**Het probleem.** Op PSG-IPA noemden wij 60 van de 75 menselijk-centrale apneus
obstructief — vier op de vijf fout. De omgekeerde richting klopte bijna perfect
(153 van 154). Geen classificatiefout dus, maar een eenzijdige bias.

**De oorzaak, per beslisregel uitgesplitst:**

| regel | n | wat er gebeurt |
|---|---:|---|
| `phase_angle` | 33 | fasehoek ≥ 45° → obstructief, zonder amplitudedrempel |
| `borderline_default` | 27 | geen regel vuurde; de restcategorie is obstructief |
| `truly_flat` | 10 | terecht centraal |
| `low_effort_default` | 5 | terecht centraal |

Fasehoek, paradoxcorrelatie en ruwe variabiliteit zijn **vormmaten**. Ze
beschrijven de structuur van een signaal en zeggen niets over of er signaal
*is*. Onder de effortdrempel meten ze ruis — en ruis heeft vorm.

**Wat niet werkte.** Een harde poort die de vormmaten uitzet onder
`EFFORT_ABSENT_RATIO` ruilt de ene bias voor de andere: centrale recall van
20 % naar 98,7 %, obstructieve van 99,4 % naar 44,8 %.

**Wat wel werkt.** Een gewicht dat meeloopt met het signaal eronder, precies
zoals dit pakket Rule 1A behandelt.

| arm | recall centraal | recall obstructief | gebalanceerd | κ |
|---|---:|---:|---:|---:|
| huidig | 20,0 % | 99,4 % | 59,7 % | 0,139 |
| harde poort | 98,7 % | 44,8 % | 71,7 % | 0,250 |
| **gegradeerd s=0,4** | **82,7 %** | **85,7 %** | **84,2 %** | **0,431** |
| gegradeerd s=1,0 | 96,0 % | 46,8 % | 71,4 % | 0,246 |
| gegradeerd s=0,1 | 26,7 % | 99,4 % | 63,0 % | 0,183 |

Zowel omlaag (0,1) als omhoog (1,0) is slechter, dus **0,4 is een optimum in
het midden en geen randartefact**.

**Klinisch.** Het onderscheid obstructief/centraal bepaalt de therapie: CPAP
tegen ASV. Van vier op de vijf centrale apneus fout naar vier op de vijf goed.

**Status: NIET uitgerold.** Geijkt op dezelfde vijf opnames waarop beoordeeld —
dat is een fit. De MESA-replicatie loopt, met vooraf vastgelegd criterium:
hogere κ én beide recalls boven 60 %.

### Drie iteraties, en waarom ze nodig waren

Een poort op regel 0 verplaatste de fout naar regel 1 (`paradox_corr=-0,997`),
en daarna naar regel 2 (`raw_movement_var=0,965`). Regel 2 en de centrale
regels lezen `raw_var_ratio` en `paradox_corr` *rechtstreeks*, buiten
`has_raw_move` en `is_paradox` om. De weging moest door élke regel die een
vormmaat leest.

En de ruis blokkeerde óók de centrale regel: die eist `no_paradox` en
`raw_var < 0,25`, allebei vormmaten. Zonder dat mee te wegen viel het event
alsnog door naar de restcategorie.

---

## 2. MESA traint, PSG-IPA kalibreert — het concept werkt

```
trainen op    54 635 kandidaten / 150 opnames  (1 scoorder)
kalibreren op  1 301 kandidaten /   5 opnames  (12 scoorders)

CV op MESA:                     AUC 0,839 – 0,858
overgezet naar PSG-IPA, VOOR:   AUC 0,943   kalibratiefout 0,172
                        NA:     AUC 0,923   kalibratiefout 0,063  (−63 %)
```

Het model generaliseert tussen twee cohorten met andere apparatuur en andere
scoorders (AUC 0,94), maar de **kansen** die het uitspreekt kloppen niet. Vijf
multi-scoorder opnames volstaan om dat recht te trekken.

Dat is de rolverdeling: MESA levert het volume om te leren *wat* een event is,
PSG-IPA de enige data die zegt *hoe zeker*. Een model schat 32 featuregewichten
en heeft daar honderden opnames voor nodig; een kalibratiekromme schat een
monotone afbeelding en heeft aan vijf genoeg.

**Wat kalibratie niet repareert:** een klasse die nooit gelabeld is. MESA
annoteert geen RERA's, dus RDI is er niet uit te leren. Dekkingsprobleem, geen
kalibratieprobleem.

---

## 3. De moeilijkheid zit in de lichte patiënt, niet in onze techniek

Dit is de bevinding die het beeld verandert, en ze is langs **drie
onafhankelijke wegen** gemeten.

**Mensen onderling** (PSG-IPA, 12 scoorders, 66 paren per opname):

| opname | events per scoorder | F1 mens–mens |
|---|---|---:|
| SN3 | 273–339 | 0,948 |
| SN4 | **1–38** | 0,553 |

Op de lichtste opname scoorde de ene expert één event en de andere
achtendertig, met κ 0,000 op het subtype.

**Onze regelketen** tegen diezelfde scoorders: 93 % van het plafond op de zware
opname, 52 % op de lichte.

**Een sequentiemodel** (1D-U-Net, 462k parameters, 113 opnames training,
37 held out, RTX A4000) dat geen enkele regel van ons erft:

| ziektelast | opnames | mediane F1 |
|---|---:|---:|
| < 20 events | 8 | **0,254** |
| 20–60 | 9 | 0,378 |
| 60–150 | 8 | 0,475 |
| ≥ 150 | 12 | **0,743** |

Mediaan over alles: 0,539.

**Drie systemen zonder gedeelde aannames, dezelfde helling.** Waar weinig te
vinden is, weten mensen het onderling niet, weten onze regels het niet, en weet
een netwerk dat de golfvorm leest het evenmin.

Dat verschuift de vraag. Niet *hoe maken we detectie beter*, maar **hoe
rapporteren we dat een AHI van 8 een andere betrouwbaarheid heeft dan een AHI
van 40**. De onzekerheid hoort in het rapport, niet in een voetnoot.

---

## 4. Onze arousalindex is geijkt op de mildste scoorder

MESA legt per opname vast wie hem scoorde (`scorerid5`). Drie scoorders dragen
het cohort; één deed 59 %. De toewijzing is niet willekeurig over sites
(χ² p = 1,1e-05), dus corrigeren was nodig.

Gecorrigeerd voor leeftijd, BMI, geslacht en site (n = 2030–2052):

| index | scoorder 928 | scoorder 939 |
|---|---:|---:|
| OAHI3 | +1,28/u (n.s.) | +2,55/u (p=0,03) |
| OAHI4 | +1,05/u (n.s.) | +1,84/u (n.s.) |
| **arousalindex** | **+3,03/u (p=3e-06)** | **+2,69/u (p=7e-04)** |

Respiratoir ontlopen ze elkaar nauwelijks. Op **arousals** scheelt het ruim
drie per uur op dezelfde populatie — een zesde van de mediane index.

Onze arousalclassifier is op MESA getraind (653 proefpersonen). 59 % van die
labels komt van de scoorder die het hoogst scoort. `--split-on scorer` in
`train_event_classifier.py` maakt nu meetbaar of we een fysiologie of een
persoon modelleren.

---

## Wat er NIET werkt, en dat is ook een uitkomst

**Waveletontruising** blijft uit, maar om een andere reden dan genoteerd stond.
Niet de σ-schatting: 58 % van de artefactenergie ligt in de schaal die 99,9 %
van de ademhaling draagt. Een artefact van 0,5–2 s en een ademteug van 3–6 s
liggen één octaaf uit elkaar; er is geen schaal die het ene bevat en het andere
niet.

**Snurken uit de neusdruk** is niet af te leiden. AUC 0,596 / 0,484 / 0,314
tegen flow-gematchte normale ademhaling — toeval. Criterium 1 van de
hypopneu-subtypering vraagt een echt snurkkanaal.

**De manualregel voor hypopneu-subtypering** (AASM v3 §6.1) is geïmplementeerd
maar onbruikbaar zonder dat kanaal: 70,2 % zou centraal heten tegen een
menselijk ijkpunt van 5,9 %. Met abstentie wordt het `uncertain` in plaats van
een verkeerd label — eerlijk, en niet nuttig.

---

## Openstaand

* MESA-replicatie van de gegradeerde subtypering (loopt)
* Fijnijking 0,15–0,35 (loopt)
* Scoordersplit: train op 916+939, test op 928
* De 5,9 %-vraag: waarom noemen wij 1,0 % van de hypopneus centraal
* Een rapportagevorm voor onzekerheid bij lage ziektelast
