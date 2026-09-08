# Overnames uit commerciële systemen — de vier metingen (2026-09-07/08)

*Vervolg op commerciele_autoscoring_20260907.md ("wat kunnen we hiervan
overnemen?"). Vier van de vijf overnames zijn dezelfde dag gebouwd of
gemeten; dit is het meetverslag met de beslissingen.*

## 1. FDA-valuta: PPA/NPA tegen 2/3-consensus (PSG-IPA, n=5)

Harnas: `meetscripts/consensus_ppa_psgipa.py`. Constructie identiek aan de
510(k)'s (gepoolde 30s-epochs, event raakt epoch, consensus = ≥2 van 3),
maar over ALLE C(12,3)=220 scoorderpanels in plaats van één drietal —
mediaan [2,5-97,5 pct] over panels. Detectie: respiratoir `aasm_v3_rec`
(psgscoring 0.32.0, scoorder-1-hypnogram), arousals via de klinische keten
(union F4/C4/O2, LGBM 0,70, 10s-regel). Opnameset: SN1-SN5 volledig,
ongeselecteerd.

| Domein       | PPA               | NPA               | OA                |
|--------------|-------------------|-------------------|-------------------|
| Respiratoir  | **78,8 %** [74,7-83,2] | 93,0 % [92,3-93,5] | 90,5 % [89,7-91,1] |
| Arousal      | **55,6 %** [49,1-60,9] | 84,7 % [84,2-85,1] | 80,1 % [78,1-81,5] |

Ter vergelijking (andere populaties/montages — indicatief, geen gepaarde
vergelijking): EnsoSleep K210034 (n=100): SDB-PA 75,4 %, hypopneu 66,3 %,
arousal 73,6 %; MICHELE K112102 (n=30, split-night-zwaar): hypopneu-PPA
76,3 %, arousal-PPA 60,0 %.

**Lezing.** Respiratoir zitten we in dezelfde valuta op/boven
510(k)-niveau, en dat op een míld gedomineerd spectrum (4/5 nachten
AHI<15) waar consensus-events juist schaars en ambigu zijn. Arousals
bevestigen het bekende gat: 55,6 % tegen 60-74 % bij de commerciëlen —
consistent met F1 0,546 tegen het menselijke plafond 0,679. De
spreidingskolom is zelf een bevinding: afhankelijk van wélk drietal
scoorders het panel vormt schuift de respiratoire PPA ruim 8 punten —
een 510(k) met drie scoorders rapporteert één trekking uit die verdeling.

## 2. Ernstklasse-overeenstemming (beschrijvend, n=5)

AHI≥15 en ≥30: **60/60 scoorderoordelen gelijk (100 %)**. AHI≥5: 45/60
(75 %) — en de 15 afwijkingen liggen exact op de twee nachten waar de
twaalf scoorders zélf over de 5/u-grens verdeeld zijn (SN2: 8 normaal/
4 mild; SN4: 7/5). Geen scherpe referentie bij lage AHI — zoals de
scoorder-verwachtingsnoot in het rapport al zegt. LR± bewust niet op n=5;
eindpuntdefinitie staat in draaiboek §9b voor de multicenter-n≥50.

## 3. Pleth fase-0: PWA-daling als tweede arousal-getuige (MESA, n=46)

Harnas: `meetscripts/pleth_fase0_mesa.py` (Obelix, 6 workers, ±45 min;
48 seeded-random VERSE id's, 2 pleth-onbruikbaar; id's geregistreerd).
Eenvoudige getuige: per-slag PWA, rollende mediaan 60 s, daling = <70 %
basislijn ≥3 s; coïncidentievenster [onset−5, +10] s; onze kandidaten via
de uitgerolde keten (EEG1/2/3-union, 0,80, 10s-regel), TP/FP tegen
NSRR-arousals met IoU ≥0,20.

| Groep                  | PWA-coïncidentie | n     |
|------------------------|------------------|-------|
| Onze TP's              | **24,6 %**       | 3814  |
| Onze FP's              | **16,9 %**       | 2509  |
| Door ons gemiste mens  | 26,2 %           | 3251  |
| Alle menselijke        | 29,5 %           | 7065  |
| Kansdekking (nul)      | 12,9 %           | —     |

Gepaard per opname: TP>FP op **36/46**, mediaan +0,047, Wilcoxon
**p = 0,0001**. Grote heterogeniteit: nachten met TP-coïncidentie 0,6+
naast nachten van ~0 (pleth-kwaliteit), en 10/46 nachten waar FP's júist
vaker coïncideren (bewegingsartefacten drijven zowel valse arousals als
PWA-dalingen).

**Vooraf vastgelegd kader, uitslag:** beide criteria gehaald —
discriminatie (36/46, p<0,05) én recall-potentieel (gemiste arousals 2×
kansniveau). MAAR de effectgrootte van déze eenvoudige getuige is klein:
hij ziet maar 29,5 % van alle menselijke arousals, dus als hard filter
zou hij 75 % van onze TP's meeslachten. **Beslissing die voorligt (fase
1):** PWA-/hartslagfeatures als extra INPUT van de LGBM-classifier
(zachte getuige, zoals Philips' DL-detector ze gebruikt), niet als poort
erachter. Philips haalt met een getrainde detector ICC 0,73 waar onze
regelgetuige blijft steken — de ruimte zit in de getuige, niet in het
principe.

## 4. Canule-uitval-prevalentie (MESA, n=100, seeded-random uit alle)

Harnas: `meetscripts/canule_uitval_mesa.py` (Z6, ±7 min). Partiële
uitval (Pres dood ≥10 min terwijl Thor leeft): **11/100 nachten**, samen
537 min = **0,83 % van de opnametijd**; nul volledig dode Pres-nachten.
In 12 van de 15 uitvalsegmenten is de **thermistor óók dood** — de
Noxturnal-failover zou dus vrijwel altijd op RIP moeten terugvallen.

**Beslissing: niet bouwen nu.** 0,83 % nachttijd tegen een ingreep in de
poortarchitectuur is geen ruil zolang detectiekwaliteit dé prioriteit is.
Dossier compleet; heropenen als klinische opnames een hogere
uitvalfractie tonen dan MESA.

## 5. CSA-bevestigingsnoot (gebouwd, YF commit f1432b1)

Productstance van EnsoSleep overgenomen: bij n_central>0 meldt het
rapport in vier talen dat het individuele subtype-label minder
betrouwbaar is dan de telling en handmatige bevestiging aanbevolen is.
Drie rendertests (eerst rood), suite 716 groen. Gaat mee met de volgende
YF-release; draaiboek §9b draagt de eindpuntdefinities voor multicenter.

## Openstaande beslissingen voor de gebruiker

1. **Pleth fase 1**: PWA/hartslag als classifier-features (afleiding +
   replicatie op disjuncte sets, kost een paar Obelix-nachten) — het
   vooraf vastgelegde fase-0-kader zegt "kandidaat", de effectgrootte
   van de eenvoudige getuige maant tot bescheiden verwachtingen.
2. **RIP-failover**: gearchiveerd op 0,83 %; heropenen op klinische data?
3. YF-release met de CSA-noot (geen indexverandering) wanneer de
   volgende uitrol toch plaatsvindt.
