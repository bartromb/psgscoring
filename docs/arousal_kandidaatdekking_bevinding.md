# De kandidaatdrempels zijn niet de oorzaak van SN3

Gemeten 2026-08-26. Punt 2 van de drie openstaande vragen in
`arousal_waar_het_gat_zit.md`. Aanleiding: drie modelvarianten (v4, v5, v6) en
twee drempelorakels zijn weerlegd, terwijl het pool-orakel op 0,896 staat tegen
onze 0,514 — en SN3 dekt maar 50 % van de menselijke arousals.

## Methode

Per scoorder de fractie van zijn arousals die door minstens één kandidaat wordt
geraakt (IoU ≥ 0,20); daarvan de mediaan over de twaalf scoorders. Dat is de
**bovengrens** van wat welk model dan ook op die opname kan halen: wat niet in
de pool zit, kan geen selectie terughalen.

Vijf instellingen van de twee kandidaatdrempels, als expliciete argumenten
meegegeven zodat de module niet gemuteerd wordt. `1,2/1,0` is wat productie
draait; `1,0/0,7` is zo los dat de arousalband de basislijn nauwelijks nog hoeft
te overschrijden.

## Dekking

| opname | mens | **1,2/1,0** | 1,1/1,0 | 1,2/0,85 | 1,1/0,85 | 1,0/0,7 |
|---|---:|---:|---:|---:|---:|---:|
| SN1 | 121 | 87 % | 89 % | 87 % | 90 % | 91 % |
| SN2 | 46 | 81 % | 84 % | 83 % | 84 % | 87 % |
| **SN3** | 142 | **50 %** | 55 % | 52 % | 57 % | **60 %** |
| SN4 | 104 | 74 % | 76 % | 76 % | 77 % | 77 % |
| SN5 | 202 | 90 % | 89 % | 92 % | 91 % | 90 % |

## Poolgrootte — de prijs

| opname | 1,2/1,0 | 1,1/1,0 | 1,2/0,85 | 1,1/0,85 | 1,0/0,7 |
|---|---:|---:|---:|---:|---:|
| SN1 | 1163 | 1246 | 1199 | 1298 | 1267 |
| SN2 | 1188 | 1271 | 1225 | 1306 | 1290 |
| SN3 | 1453 | 1675 | 1533 | 1745 | 1709 |
| SN4 | 1410 | 1651 | 1487 | 1723 | 1802 |
| SN5 | 1601 | 1739 | 1641 | 1799 | 1854 |

## Uitkomst

**De drempels zijn niet de oorzaak.** Ze helemaal loslaten tilt SN3 van 50 % naar
60 % en kost 18 % meer kandidaten om uit te selecteren — precies de stap waar het
al vastliep. Veertig procent van SN3's menselijke arousals blijft bij élke
drempel onzichtbaar voor de kandidaatstap.

**En het is geen kwestie van aantal.** SN3 krijgt met 1453 kandidaten al méér dan
SN1 met 1163, en dekt er de helft mee tegen SN1's 87 %. De kandidaten liggen op
de verkeerde plekken; er zijn er niet te weinig.

Daarmee is de hele familie "genereer meer kandidaten" begrensd op +10 punten voor
SN3, tegen een prijs in de selectiestap waarvan we al weten dat die het knelpunt
is. Bouw die richting niet.

## Wat er dan wél aan de hand kan zijn

Drie mogelijkheden, in volgorde van hoe makkelijk ze uit te sluiten zijn:

1. ~~**De tijdas.**~~ **WEERLEGD.** Dekking gemeten als functie van een
   kunstmatige verschuiving van −60 tot +60 s (stap 2 s), met de vier andere
   opnames als controle. Alle vijf pieken bij 0 of +2 s; SN3 gedraagt zich als
   de rest. Er is geen verschoven referentie. Ruwe cijfers in
   `arousal_uitlijning_lag.json`.
2. **De basislijn.** SN3 draagt 2,2× het absolute beta-achtergrondvermogen van
   SN1 (EEG p95 98,2 tegen 35,8 µV). Een detector die een VERHOUDING tegen een
   nachtbasislijn meet, ziet een vaste toevoeging op een hoge achtergrond als
   een kleine verhouding — zie het slot van
   [[project_arousal_detector_status]]. Maar `1,0/0,7` neemt die eis vrijwel
   weg en levert maar 10 punten op, dus dit verklaart hooguit een deel.
3. ~~**De afleidingen.**~~ **WEERLEGD.** SN3 draagt exact dezelfde montage als
   SN1, SN2 en SN5 — `EEG F4-M1`, `EEG C4-M1`, `EEG O2-M1` — en alle drie
   worden gelezen (alleen SN4 heeft `Cz-M1` in plaats van `C4-M1`). Er is geen
   kanaal dat wij op SN3 laten liggen.

Daarmee blijft **de basislijn** over als enige levende verklaring, en die
verklaart hooguit een deel: `1,0/0,7` neemt de verhoudingseis vrijwel weg en
levert maar 10 punten op.

Een aanwijzing dat het écht het signaal is en niet de boekhouding: SN3's
menselijke plafond is 0,692 — de mediaan van de vijf. De scoorders zijn het op
SN3 dus gewoon met elkaar eens. De 40 % die wij nooit zien zijn geen
twijfelgevallen.

## Bijvangst — de kandidaatonsets liggen systematisch ~2 s te vroeg

De piek ligt op vier van de vijf opnames bij **+2 s**, niet bij 0:

| | SN1 | SN2 | SN3 | SN4 | SN5 |
|---|---:|---:|---:|---:|---:|
| dekking bij 0 s | 87 % | 81 % | 50 % | 74 % | 90 % |
| dekking bij +2 s | 90 % | 81 % | 56 % | 77 % | 90 % |

Mediaan +3 punten. Dat past bij een links uitgelijnd 2 s-vermogensvenster: de
kandidaat begint waar het venster begint, de scoorder markeert waar hij de
verschuiving ziet. Een lead, geen conclusie — en hij moet op **F1** beoordeeld
worden, niet op dekking. Een maat die alleen vraagt of er iets in de buurt ligt,
beloont verschuiven ook als het niets oplevert; dat is precies waarop het
event-locked venster eerder strandde (koppeling +7,2 punt, F1 +0,018).

## Reproductie

`scripts/measure_arousal_candidate_coverage.py`, PSG-IPA `EEG_arousals`-subboom,
12 scoorders. Uitlijningscontrole: `scripts/diag_arousal_alignment_lag.py`.
Ruwe cijfers in `arousal_kandidaatdekking.json`.
