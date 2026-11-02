# De laatste tien eigen PSG's: machine tegen menselijke scoring en decompositie van de vier oktoberwijzigingen

Datum: 2026-11-02. Preregistratie: `docs/laatste10_humaan_vs_machine_preregistratie_20261102.md`
(ccc6613, aanvulling 7ad8689 vóór de run). Status: **meting klaar (10/10, 0 fouten), geverifieerd 02-11 (14 bevindingen, alle verwerkt;
besluit staat); de menselijke referentie is voor 1 van 10 nachten bekend (L01 ≈ 80 /u, mondeling, ernstige OSAS met centrale
component). De vergelijking machine − mens over de tien wacht op de overige negen cijfers.**
Scripts: `docs/laatste10_20261102/` (maak_mapping.py, run.py, analyse.py). Ruw (zonder koppeling):
`/srv/CODE/docs/laatste10_20261102/` (L01–L10.json, run.log, app_check.log, analyse_uitvoer.md,
samenvatting.json). Koppeling job-id ↔ L en het pseudoniem blijven op de server (mode 600).

## 1. Uitvoering

- Selectie: elf jobs bekeken, één duplicaat (zelfde opname twee keer opgeladen) uitgesloten, tien
  diagnostische PSG's zonder split-night, alle gescoord door productie met psgscoring 0.37.0 op
  `aasm_v3_breath_dual`, alle met `eog_ch` in de jobconfig (EOG bereikte het U-Net).
- Rookproef L01, arm A0: AHI 117,8 = opgeslagen 117,8; 151 s. Volledige run 13:03–14:06 (Z6-tijd; run.log
  staat in UTC, één uur eerder), container uit `yasaflaskified:0.39.2`, `--cpus 16`, 8 nachten
  parallel, 58 GB RSS volgens `docker stats` tijdens de run, 7 armen × 10 nachten in 21–40 min per
  nacht (L10 21, L03 40). App elke minuut bereikbaar (70 × 200, 0 anders), load ≤ 8,9.
- Incident zonder gevolg voor de meting: tussen 13:48 en 14:06 weigerde de server SSH (fail2ban
  na foute wachtwoordpogingen vanaf hetzelfde adres); de container liep door.

## 2. Identiteit (prereg §5.1)

A0 reproduceert de opgeslagen AHI op **10/10** nachten exact (Δ 0,0). De run meet dus wat
productie meet.

## 3. AHI per nacht en arm (/u)

De noemer van elke machine-AHI is `summary.tst_minutes` = slaap zónder artefact-epochs, niet de
TST uit het hypnogram; binnen een nacht is hij voor alle zeven armen gelijk, dus de verschillen
tussen armen (§4) hangen er niet van af. De laatste kolom geeft A0 herrekend op de volle TST, zoals
een menselijke scoorder deelt.

| L | mens | A0 productie | SB basislijn uit | ST strictness 0,50 | BR breath | LG lgbm | ALL alles terug | REC rec-route | TST (min) | noemer (min) | A0 op volle TST |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | ≈ 80 | 117,8 | 115,0 | 111,8 | 101,4 | 119,5 | 96,4 | 40,2 | 245,5 | 242,0 | 116,1 |
| L02 | — | 22,0 | 21,4 | 13,7 | 22,0 | 22,0 | 12,2 | 6,6 | 331,5 | 280,5 | 18,6 |
| L03 | — | 18,4 | 18,4 | 15,9 | 16,1 | 17,2 | 12,6 | 7,7 | 341,5 | 328,5 | 17,7 |
| L04 | — | 23,7 | 20,2 | 21,6 | 15,6 | 22,4 | 9,7 | 11,3 | 237,0 | 222,5 | 22,3 |
| L05 | — | 130,5 | 129,8 | 130,0 | 130,5 | 130,5 | 129,4 | 104,1 | 370,0 | 275,5 | 97,1 |
| L06 | — | 17,8 | 18,4 | 16,0 | 17,8 | 19,7 | 18,8 | 15,4 | 363,5 | 277,0 | 13,5 |
| L07 | — | 9,0 | 9,0 | 5,6 | 8,8 | 8,0 | 5,2 | 5,5 | 506,0 | 447,5 | 7,9 |
| L08 | — | 10,2 | 9,4 | 7,3 | 10,2 | 12,3 | 7,6 | 10,9 | 336,5 | 330,5 | 10,0 |
| L09 | — | 5,0 | 5,0 | 3,4 | 5,0 | 4,2 | 2,4 | 7,7 | 436,5 | 429,0 | 4,9 |
| L10 | — | 8,9 | 9,5 | 7,3 | 8,9 | 7,0 | 6,3 | 14,8 | 447,0 | 437,0 | 8,7 |

Ernstklassen A0: 2 ernstig (L01, L05), 4 matig (L02, L03, L04, L06), 4 licht (L07–L10).

**Noemer-effect (verificatiebevinding 3):** de artefact-uitsluiting maakt de noemer 1–26 % kleiner
dan de TST (234–511 artefact-epochs per nacht; L05 43 % van de epochs) en de AHI dus tot +34 %
hoger dan dezelfde telling op de volle TST (L05 130,5 → 97,1; L06 17,8 → 13,5; L02 22,0 → 18,6;
L07 9,0 → 7,9). Een menselijke scoorder deelt door de TST. Dit is een kandidaat voor prereg-vraag 3
(regel of noemer) die vóór de meting niet benoemd was, en hij staat los van de vier
oktoberwijzigingen. Of de machine events in artefact-epochs óók niet telt (dan is de herrekening een
ondergrens) is uit deze uitvoer niet af te leiden: open punt voor psgscoring.

## 4. Verschuiving tegen productie: A0 − arm per één-wijziging-terug-arm (beschrijvend)

De prereg-drager van §5.3 ("brengt de mediaan van A0 − mens het meest naar 0") is pas te bepalen
met de tien menselijke cijfers; op de enige nacht mét mens (L01) is dat BR, niet ST (§5). Wat hier
staat is de verschuiving tegen productie. Tellingen op de afgeronde AHI's, op |A0 − arm|, dus
inclusief nachten waar de arm hóger ligt (LG 3 van 7, REC L09/L10 en L10 bij ≥ 5, ALL L06).

| arm | wijziging | mediaan | gemiddeld | min | max | nachten \|Δ\| ≥ 1 /u | nachten \|Δ\| ≥ 5 /u |
|---|---|---|---|---|---|---|---|
| ST | 10-10 strictness 0,30 → 0,50 | **+2,3** | +3,1 | +0,5 | +8,3 (L02) | 9/10 | 2 |
| BR | 07-10 vereniging → breath | 0,0 | +2,7 | 0,0 | +16,4 (L01) | 3/10 | 2 (L01, L04) |
| SB | 12-10 stadiumbasislijn uit | +0,3 | +0,7 | −0,6 | +3,5 (L04) | 2/10 | 0 |
| LG | 26-10 unet_v1 → lgbm | +0,4 | 0,0 | −2,1 (L08) | +1,9 (L10) | 7/10 (6 op onafgeronde waarden: L07 0,98) | 0 |
| ALL | alle vier terug (op breath) | +3,2 | +6,3 | −1,0 | +21,4 (L01) | 10/10 | 4 |
| REC | rec + basislijn uit + lgbm | +7,1 | +13,9 | −5,9 (L10) | +77,6 (L01) | 9/10 | 6 |

Lezing:

1. **De breedste verschuiving komt van de hypopneeroute (strictness 0,30), niet van de
   stadiumbasislijn.** ST verlaagt de AHI op 10/10 nachten (9/10 met ≥ 1 /u), mediaan 2,3 /u; het
   hypopneeaantal gaat van mediaan 47,5 naar 36,5 (§6: op elke nacht lager, L02 102 → 63, L01 65 →
   41). **De vooraf verwachte drager SB is op deze tien weerlegd:** mediaan 0,3 /u, nergens 5 /u.
   Niet omdat de basislijn de drukapneus ongemoeid laat (SB verandert ze wél: L04 54 → 33, L01 410 →
   391, L08 27 → 20; L06 42 → 47, L10 30 → 34), maar omdat de verdwenen apneus grotendeels als
   hypopnee terugkomen (L04 hyp 34 → 42, L01 65 → 73, L08 29 → 32): herklassificatie, AHI-neutraal.
   Op MESA-0111 zat de foutmodus op de thermistor (alle 183), hier komen de apneus bijna uitsluitend
   van de druk (§6: 273/410, 53/54, 482/479, 32/46 …). Prereg §5.5: A0 − SB < 1 /u → **deze meting
   geeft géén steun aan `stage_baseline=False`**; het MESA-argument (3 normale nachten → matig/ernstig)
   staat op zichzelf en wordt hier niet tegengesproken, alleen niet versterkt.
2. **De vereniging (breath_dual) is de grootste post op twee nachten**: L01 −16,4 (273 alleen-drukapneus
   worden onder breath hypopneeën: 65 → 272) en L04 −8,1 (54 → 1 apneus, 34 → 57 hypopneeën), en
   L03 −2,3. BR wijkt alleen af waar breath's eigen enveloppe-poort de thermistor aanvaardt (L01 0,58,
   L03 0,53, L04 0,48); op L06/L07/L08 wijst die poort hem af en scoort breath óók op druk (0,0 / 0,2 /
   0,0), en op L02/L05/L09/L10 vond de thermistor onder A0 nul eigen apneus (L05/L10 dood, band 0,18
   en 0,14; L02/L09 lage overeenstemming maar gebruikt), zodat BR ≡ A0 (L02 op één hypopnee-label na).
   De prereg verwachtte BR als tweede "op nachten met een afgekeurde thermistor"; het is het
   omgekeerde: de vereniging telt juist waar beide poorten de thermistor goedkeuren.
3. **unet_v1 is AHI-neutraal** (mediaan +0,4, bereik −2,1…+1,9) maar niet arousal-neutraal: zie §7.
4. **Additiviteit:** A0 − ALL tegen de som van de vier enkelvoudige verschuivingen: L01 −2,1 (op
   onafgeronde waarden −1,98), L04 −1,0 (−1,08), L08 +1,0 (+1,09), overige |·| ≤ 0,9. Volgens de
   prereg-grens (> 1 /u = interactie) is er **een interactie op L01** (9 % van A0 − ALL = 21,4) en
   liggen L04 en L08 op de grens; in substantie tellen de vier wijzigingen op.
5. **De rec-route van vóór 07-10 is geen "oude waarheid":** mediaan 7,1 /u lager dan A0, maar met
   L01 −77,6 (40,2 tegen mens ≈ 80) én hoger dan A0 op drie van de vier lichtste nachten (L08 +0,7,
   L09 +2,7, L10 +5,9; op L07 juist −3,5) doordat rec's enveloppe-hypopneeroute daar meer telt (§6: hyp REC 40/55/74 tegen A0
   29/36/35). Wie "vroeger klopte het beter" zegt, bedoelt dus niet rec.

## 5. Machine tegen mens (prereg §5.2): alleen L01 beschikbaar

| L | mens | A0 | Δ | Δ % | SB | ST | BR | LG | ALL | REC |
|---|---|---|---|---|---|---|---|---|---|---|
| L01 | ≈ 80 | 117,8 | +37,8 | +47 % | +35,0 | +31,8 | +21,4 | +39,5 | +16,4 | −39,8 |

Op L01 brengt BR (→ +21,4) A0 − mens het dichtst bij 0, in de zin van prereg §5.3; de vereniging
verklaart 16 van de 38 /u, de hypopneeroute 6, de stadiumbasislijn 3, het U-Net −2; alle vier samen
21, en dan blijft +16 /u over (ALL 96,4). Wat overblijft zit in de hypopneeroute van breath zelf
(272 hypopneeën onder breath tegen 25 onder rec) en in de noemer (machine 242,0 min artefactvrije
slaap, TST 245,5; menselijke TST onbekend). Met één nacht is dit een illustratie, geen uitkomst;
de tekentoets, ernstconcordantie en TST-herrekening uit de prereg volgen zodra de negen andere
cijfers er zijn (`humaan.csv` op de server, dan `maak_mapping.py --cijfers` en `analyse.py`).

## 6. Per nacht: sensoren, hypopneeroute, arousaldetectoren

| L | TST | apneus A0 | druk-only | beide | th-only | thermistor bruikbaar | hyp A0 | hyp ST | hyp REC | hyp ≥ 3 % | hyp arousalvlag | arousals U-Net | arousals lgbm | AI U-Net | AI lgbm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | 246 | 410 | 273 | 136 | 1 | ja | 65 | 41 | 25 | 61 | 4 | **23** | **303** | 5,6 | 74,1 |
| L02 | 332 | 1 | 1 | 0 | 0 | nee | 102 | 63 | 29 | 81 | 21 | 156 | 195 | 28,2 | 35,3 |
| L03 | 342 | 46 | 32 | 15 | 0 | ja | 55 | 41 | 23 | 29 | 26 | 175 | 130 | 30,7 | 22,8 |
| L04 | 237 | 54 | 53 | 1 | 0 | ja | 34 | 26 | 42 | 17 | 17 | 84 | 62 | 21,3 | 15,7 |
| L05 | 370 | 479 | 482 | 0 | 0 | nee | 120 | 118 | 2 | 119 | 1 | 551 | 484 | 89,4 | 78,5 |
| L06 | 364 | 42 | 39 | 3 | 0 | ja | 40 | 32 | 24 | 15 | 25 | 75 | 100 | 12,4 | 16,5 |
| L07 | 506 | 1 | 0 | 0 | 1 | ja | 66 | 41 | 40 | 44 | 22 | 87 | 83 | 10,3 | 9,8 |
| L08 | 336 | 27 | 28 | 0 | 0 | ja | 29 | 13 | 40 | 25 | 3 | 33 | 72 | 5,9 | 12,8 |
| L09 | 436 | 0 | 0 | 0 | 0 | nee | 36 | 24 | 55 | 23 | 13 | 69 | 51 | 9,5 | 7,0 |
| L10 | 447 | 30 | 31 | 0 | 0 | nee | 35 | 23 | 74 | 13 | 22 | 109 | 50 | 14,6 | 6,7 |

(druk-only/beide/th-only = duale apneutelling vóór samenvoeging; "thermistor bruikbaar" =
`thermistor_check.usable` onder de respiratory_band-poort; hyp ≥ 3 % en arousalvlag = A0.)

## 7. Bijvangst: het U-Net mist op L01 vrijwel alle arousals

Op L01 geeft unet_v1 23 arousals (24 ruw vóór de 10 s-samenvoeging, arousalindex 5,6) waar lgbm er
303 vindt (74,1 /u) op een nacht met 410 apneus en menselijk ≈ 80 events/u. Op de andere negen
nachten ligt de verhouding U-Net/lgbm tussen 0,46 (L08) en 2,2 (L10). Dit is één nacht en zegt
niets over de replicatie op MESA/PSG-IPA, maar het is de tweede keer dat een detector op een echte
klinische nacht iets anders doet dan op de harnassen (vgl. de stadiumbasislijn). Te onderzoeken,
niet hier: de U-Net-kansreeks op L01 (is de uitvoer leeg of zit hij net onder τ 0,35?), de
EEG-amplitude na de 18-min-normalisatie (ernstige OSA met 410 apneus geeft een EEG vol
ontwaakreacties, wat de normalisatie kan afvlakken), en de waarde van `unet_fallback_reason`
(None op 10/10: geen terugval gemeld). RDI op L01 volgt de arousaltelling: `summary.rdi` 121,5 (U-Net) tegen 162,4 (lgbm). Datanotitie
voor psgscoring: `arousal_summary.rdi` is op elke nacht een ander getal (L01 102,9, L05 78,5 tegen
`summary.rdi` 186,5/166,7) en verschilt niet tussen de detectoren; en de lgbm-armen dragen
`arousal_summary.detector = None` in plaats van "lgbm" (dat lgbm liep blijkt uit de tellingen en
de "LGBM filter"-regels in run.log).

## 8. Wat dit betekent voor de open keuzes

- **Stadiumbasislijn (27-10):** deze tien nachten dragen niet bij: het effect is klein omdat de apneus
  hier van de drukcanule komen. De keuze blijft op het MESA-argument rusten (foutmodus op
  thermistor-nachten), of op de reparatie-prereg.
- **Hypopneeroute:** strictness 0,30 verhoogt de AHI op vrijwel elke nacht met 0,5–8 /u. Of dat
  "boven de mens" is, beslissen de negen ontbrekende cijfers; op MESA (n = 140) bracht 0,30 de bias
  van −3,3 naar −0,05 tegen NSRR-scoorders met de 3 %-of-arousal-regel. Scoort de kliniek met 4 %
  zonder arousal, dan verklaart de regel een deel per constructie (A0 telt mediaan 19 hypopneeën
  met arousalvlag per nacht).
- **Vereniging:** op nachten met bruikbare thermistor én veel alleen-drukapneus (L01, L04) is dit de
  grootste post; het dossier van 07-10 (93–94 % van de alleen-drukapneus is geen
  thermistor-apneu-equivalent) wordt hier niet weerlegd.
- **Noemer:** de artefact-uitsluiting uit de AHI-noemer (tot +34 % op L05, +32 % L06, +18 % L02)
  is een vijfde kandidaat voor "boven de mens", ouder dan oktober en niet door de vier armen gedekt.
  Hoort in de vergelijking als aparte kolom (§3) en in de vervolganalyse als herrekening op de
  menselijke TST met `summary.tst_minutes` als basis (analyse.py §3b aangepast).
- **Niets uitgerold, niets gewijzigd.** De drie koppen staan in één regel per profiel; welke
  terug gaat is Barts beslissing, en die hoort op de tien menselijke cijfers te wachten.

## 9. Beperkingen

n = 10, één scoorder, één nacht met menselijke referentie; de machine-TST komt uit YASA, de
menselijke uit de staging van de kliniek (noemer onbekend); de menselijke hypopneeregel is niet
vastgelegd; 58 GB RSS en het SSH-incident komen uit de sessie, niet uit de logs; de eigen nachten zijn zonder NSRR-achtige referentie, dus "drager" betekent hier
"draagt het verschil met productie", niet "draagt de fout". Geen inferentie buiten deze tien.

## 10. In situ na de uitrol (02-11, 17:09–17:35): YF 0.39.3 / psgscoring 0.38.0

Barts beslissing op §4/§5 ("AHI op volle TST benadert menselijke scoring. Pas toe.") is als
profielveld `index_denominator="sleep"` gebouwd (psgscoring 0.38.0, CHANGELOG) en om ±17:00
uitgerold (YASAFlaskified 0.39.3, rollback-image 0.39.2). Daarna zijn de vijftien meest recente
jobs op de productiewachtrij opnieuw gescoord langs hetzelfde pad als het uploadformulier
(`tasks.run_analysis_job`); de 0.39.2-uitvoer van die vijftien staat op de server bewaard. De
vijftien omvatten de tien nachten van §3 (de duplicaat-upload van §1 ditmaal twee keer, want in
de app zijn het twee jobs), één nieuwere upload en vier oudere. Alleen cijfers; de koppeling
blijft op de server.

**Detectie onveranderd, noemer alleen.** Het aantal respiratoire events is op 15/15 jobs identiek
aan de 0.37.0-uitvoer (bv. 602, 475, 174, 103, 0). De noemer in de samenvatting ging van de
artefactvrije slaap naar de volle slaaptijd (`index_denominator: sleep`), en het verschil is op
elke job precies het aantal artefact-epochs in slaap × 0,5 min.

**AHI (0.37.0 → 0.38.0) op de vijftien jobs**, gesorteerd op verschil: −44,5 (83,5 → 39,0; 656
artefact-epochs, 125 → 268 min), −33,4 (130,5 → 97,1 = L05, §4 voorspelde 97,1), −4,3 ×2 (17,8 →
13,5 = L06, voorspeld 13,5; de duplicaat-upload), −3,4 (22,0 → 18,6 = L02, voorspeld 18,6), −1,7
(117,8 → 116,1 = L01), −1,4, −1,1, −0,8, −0,7, −0,2 ×2, −0,1 ×2, 0,0 (0,0 → 0,0). Mediaan −0,8 /u;
op 9/15 minder dan 2 /u. **Ernstklasse veranderd op 2 van de 14 verschillende nachten**: matig →
licht (17,8 → 13,5) en licht → normaal (5,0 → 4,9, op de grens). De 83,5 → 39,0-nacht blijft
ernstig; daar stond 56 % van de slaap als artefact (656 van 1 319 epochs, 268 min slaap).

**Overige indices.** Arousal-index identiek op 15/15 (deelde al door de volle slaap). RDI volgt
de AHI (186,5 → 138,8; 88,8 → 41,5). PLM-index iets lager (12,8 → 9,6; 19,0 → 18,3), de
LM-index ook. **ODI beweegt niet alleen mee naar beneden**: `spo2_index_denominator="sleep"`
neemt alle slaapsamples in teller én noemer (de documentatie van het veld: incl. artefact-epochs
en uitval), zodat desaturaties in artefact-epochs nu meetellen — op de 83,5-nacht ODI3 59,5 →
45,0 (noemereffect wint), maar op nachten met weinig artefact-slaap stijgt de ODI licht (12,6 →
13,4; 9,0 → 9,8; 2,6 → 3,1) omdat de teller groeit (69 → 76, 67 → 83 desaturaties). T90/T80
delen voortaan door alle samples, dus sensoruitval verdunt ze weer. Dat is de prijs van één
noemer voor alle indices; wie de artefactvrije ODI terug wil, zet alleen
`spo2_index_denominator` terug.

**Uitvoering.** 14/15 jobs `finished` in 8–24 min onder acht parallelle workers; één job (36
events, 5,0 → 4,9) liep 16 min en werd na het schrijven van results/PDF/Excel/EDF+ door de
RQ-timeout van 900 s afgebroken ("Work-horse terminated unexpectedly") — zijn uitvoer stond al op
0.38.0; hij is daarna alleen op de lege wachtrij opnieuw gedraaid (7 min, `finished`, zelfde
cijfers). De uploads van 02-11 deden er op productie 3–4 min over; de parallelle herrekening
was de uitzondering, niet de regel.

