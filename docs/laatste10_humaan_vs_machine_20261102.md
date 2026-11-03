# De laatste tien eigen PSG's: machine tegen menselijke scoring en decompositie van de vier oktoberwijzigingen

Datum: 2026-11-02. Preregistratie: `docs/laatste10_humaan_vs_machine_preregistratie_20261102.md`
(ccc6613, aanvulling 7ad8689 vóór de run). Status: **meting klaar (10/10, 0 fouten), geverifieerd 02-11 (14 bevindingen, alle verwerkt;
besluit staat); de menselijke referentie is voor 1 van 10 nachten bekend (L01 ≈ 80 /u, mondeling, op 03-11 vervallen; ernstige OSAS met centrale
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
| L01 | — | 117,8 | 115,0 | 111,8 | 101,4 | 119,5 | 96,4 | 40,2 | 245,5 | 242,0 | 116,1 |
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
met de tien menselijke cijfers; de menselijke waarde van L01 is op 03-11 vervallen (§5). Wat hier
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
   L01 −77,6 (40,2 tegen A0 117,8; menselijke waarde onbekend) én hoger dan A0 op drie van de vier lichtste nachten (L08 +0,7,
   L09 +2,7, L10 +5,9; op L07 juist −3,5) doordat rec's enveloppe-hypopneeroute daar meer telt (§6: hyp REC 40/55/74 tegen A0
   29/36/35). Wie "vroeger klopte het beter" zegt, bedoelt dus niet rec.

## 5. Machine tegen mens (prereg §5.2–5.5): negen menselijke AHI's (02-11, 20:53; L01 vervallen 03-11)

Bart leverde op 02-11 de menselijke AHI van alle tien nachten; op 03-11 14:33 trok hij de waarde voor L01
(95) als invoerfout in, zodat L01 zonder menselijke referentie staat (het eerdere mondelinge "≈ 80" telt
evenmin). n = 9.
Hypopneeregel en menselijke TST ontbreken nog, dus de noemer-afsplitsing op de menselijke TST
(prereg §5.2, laatste zin) blijft open. Cijfers: `analyse.py` met `humaan_cijfers.json` op de server
(`analyse_uitvoer.md` §2–§3). De kolom **0.38.0** is A0 herrekend op de volle slaaptijd — precies wat
productie sinds 17u schrijft (§10); Δ = machine − mens.

| L | mens | A0 (0.37.0) | Δ | 0.38.0 | Δ | SB | ST | BR | LG | ALL | REC | klasse mens / A0 / 0.38.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L01 | — | 117,8 | — | 116,1 | — | 115,0 | 111,8 | 101,4 | 119,5 | 96,4 | 40,2 | — / ernstig / ernstig |
| L02 | 16,2 | 22,0 | +5,8 | 18,6 | +2,4 | 21,4 | 13,7 | 22,0 | 22,0 | 12,2 | 6,6 | matig / matig / matig |
| L03 | 18,9 | 18,4 | −0,5 | 17,7 | −1,2 | 18,4 | 15,9 | 16,1 | 17,2 | 12,6 | 7,7 | matig / matig / matig |
| L04 | 19,3 | 23,7 | +4,4 | 22,3 | +3,0 | 20,2 | 21,6 | 15,6 | 22,4 | 9,7 | 11,3 | matig / matig / matig |
| L05 | 102,0 | 130,5 | +28,5 | 97,1 | −4,9 | 129,8 | 130,0 | 130,5 | 130,5 | 129,4 | 104,1 | ernstig / ernstig / ernstig |
| L06 | 19,8 | 17,8 | −2,0 | 13,5 | −6,3 | 18,4 | 16,0 | 17,8 | 19,7 | 18,8 | 15,4 | matig / matig / **licht** |
| L07 | 7,5 | 9,0 | +1,5 | 7,9 | +0,4 | 9,0 | 5,6 | 8,8 | 8,0 | 5,2 | 5,5 | licht / licht / licht |
| L08 | 13,2 | 10,2 | −3,0 | 10,0 | −3,2 | 9,4 | 7,3 | 10,2 | 12,3 | 7,6 | 10,9 | licht / licht / licht |
| L09 | 4,5 | 5,0 | +0,5 | 4,9 | +0,4 | 5,0 | 3,4 | 5,0 | 4,2 | 2,4 | 7,7 | normaal / **licht** / normaal |
| L10 | 19,0 | 8,9 | −10,1 | 8,7 | −10,3 | 9,5 | 7,3 | 8,9 | 7,0 | 6,3 | 14,8 | matig / **licht** / **licht** |

Machine − mens per arm (n = 9; tekentoets exact, tweezijdig):

| arm | mediaan Δ | gemiddeld | mediaan \|Δ\| | MAE | min | max | +/− | p | klasse exact | binnen 1 | mediaan Δ % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 (0.37.0, productie tot 17u) | +0,5 | +2,8 | 3,0 | 6,3 | −10,1 | +28,5 | 5/4 | 1,000 | 7/9 | 9/9 | +11 |
| **0.38.0 (volle slaaptijd, productie nu)** | **−1,2** | **−2,2** | **3,0** | **3,6** | −10,3 | +3,0 | 4/5 | 1,000 | 7/9 | 9/9 | — |
| SB | +0,5 | +2,3 | | | −9,5 | +27,8 | 5/4 | 1,000 | 7/9 | 9/9 | +5 |
| ST | −2,5 | 0,0 | | | −11,7 | +28,0 | 2/7 | 0,180 | 7/9 | 9/9 | −19 |
| BR | −2,0 | +1,6 | | | −10,1 | +28,5 | 4/5 | 1,000 | 7/9 | 9/9 | −10 |
| LG | −0,1 | +2,5 | | | −12,0 | +28,5 | 4/5 | 1,000 | 8/9 | 9/9 | −1 |
| ALL | −4,0 | −1,8 | | | −12,7 | +27,4 | 1/8 | 0,039 | 5/9 | 9/9 | −33 |
| REC | −4,2 | −4,0 | | | −11,2 | +3,2 | 2/7 | 0,180 | 4/9 | 9/9 | −22 |

Lezing volgens de prereg:

1. **Vraag 1 (§5.2):** A0 zit mediaan +0,5 /u boven de mens, 5 nachten boven en 4 onder, p = 1,000 —
   geen richting. Ernstklasse exact 7/9, binnen één klasse 9/9. Het gemiddelde (+2,8) komt van de
   ernstige nacht L05 (+28,5); de indruk "psgscoring zit boven de mens" waarmee dit
   dossier begon, is dus een indruk van de zware nachten, niet van de mediane nacht. Op de volle
   slaaptijd (0.38.0) valt L05 naar −4,9 en L02 naar +2,4. Mediaan −1,2, MAE 6,3 → 3,6; 5 van de 9 onder de mens.
2. **Vraag 2, drager (§5.3):** SB laat de mediaan op +0,5 staan en A0 − SB is mediaan 0,3 /u
   (< 1): de stadiumbasislijn is op deze tien **geen drager** en de steun voor `stage_baseline=False`
   (optie 1 van 27-10) vervalt volgens §5.5 — SB "brengt" de mediaan niet binnen ±2, want A0 stond
   er al. De vooraf verwachte drager komt dus niet uit. BR verschuift L01 met 16,4 /u (§4) maar doet
   op de nachten met afgekeurde thermistor (L02, L05, L09, L10) per constructie niets.
3. **Vraag 3, regelverschil (§5.4):** REC − mens mediaan −4,2 en ALL − mens −4,0 liggen niet boven +2
   maar eronder: de oude rec-route en de vier terugdraaiingen samen scoren systematisch ónder de mens
   (2/7 en 1/8, ALL p = 0,039). Het restverschil is dus geen regelverschil in de zin van §5.4; `regel_h` blijft onbekend.
4. **Wat overblijft, per nacht:** één nacht waar de noemer het niet verklaart, plus L01 zonder referentie.
   **L01** (116 /u op 0.38.0, menselijke waarde sinds 03-11 onbekend): 410 apneus, 273 alleen op druk (§6),
   en het U-Net vindt 23 arousals tegen 303 voor lgbm (§7) — de arousalvraag blijft, de AHI-vraag is
   open. **L10** (−10 /u): 30 apneus
   en 35 hypopneeën tegen een menselijke 19 /u; alleen REC (14,8, enveloppe-hypopneeën) komt in de
   buurt — hier mist de breath-route hypopneeën. **L06** (−6,3) landt op 0.38.0 een klasse onder de
   mens (licht tegen matig); A0 lag op L09 een klasse boven (5,0 tegen 4,5).
5. **n = 9, één scoorder** (§5.6): richtingen, geen uitspraak. Alleen ALL (1/8, p = 0,039) haalt de tekentoets.

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

**Bijwerking met de negen menselijke cijfers (02-11, avond; L01 vervallen 03-11; §5):**

- *Stadiumbasislijn:* bevestigd — A0 − SB mediaan 0,3 /u, geen drager; de steun voor optie 1 vervalt (§5.2).
- *Hypopneeroute:* strictness 0,30 (A0) zit mediaan +0,5 /u boven de mens, de terugdraaiing naar 0,50 (ST)
  mediaan −2,5 eronder, beide 7/9 in klasse. Op deze tien is 0,30 dus niet "boven de mens"; de
  hypopneeregel van de kliniek (`regel_h`) is sinds 03-11 bekend: **AASM regel 1A** (≥ 30 %, ≥ 10 s,
  ≥ 3 % desaturatie óf arousal), dezelfde regel als `aasm_v3_breath_dual` in productie; de
  onderscoring op de matige nachten (L06/H07, L10/H12) is dus een detectieverschil, geen regelverschil.
- *Noemer:* gemeten en uitgerold (§10): op de volle slaaptijd mediaan −1,2 /u, MAE 6,3 → 3,6, L05 van
  +28,5 naar −4,9. De eerste bullet van §5.4 in de prereg (regel of noemer) wordt daarmee "noemer".
- *Wat niet door de noemer wordt gedekt:* L10 (−10 /u, hypopneeën gemist, alleen REC komt dichtbij) en de
  arousaltelling op L01 (U-Net 23 tegen 303; de menselijke AHI van L01 is sinds 03-11 onbekend). Dat zijn
  de twee detectievragen die overblijven.

## 9. Beperkingen

n = 9 (L01 zonder menselijke waarde), één scoorder, menselijke AHI zonder eventlijst, TST of hypopneeregel; de machine-TST komt uit YASA, de
menselijke uit de staging van de kliniek (noemer onbekend); de menselijke hypopneeregel is AASM 1A
(bevestigd 03-11, na de analyse); 58 GB RSS en het SSH-incident komen uit de sessie, niet uit de logs; de eigen nachten zijn zonder NSRR-achtige referentie, dus "drager" betekent hier
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

### 10b. Menselijke AHI's voor de vijftien herrekende jobs (02-11, 20:53; H02 vervallen 03-11)

Dertien patiënten (H07 = H08; H02 = L01 zonder menselijke waarde sinds 03-11) met een menselijke AHI; de
negen van §5 plus vier buiten de prereg-steekproef. Machine − mens, n = 13:

| | mediaan Δ | gemiddeld | MAE | mediaan \|Δ\| | max \|Δ\| | binnen ±5 | Pearson r | Lin CCC | klasse exact |
|---|---|---|---|---|---|---|---|---|---|
| 0.37.0 (artefactvrije noemer) | −0,5 | +4,4 | 8,2 | 3,0 | 41,6 | 9/13 | 0,950 | 0,900 | 11/13 |
| 0.38.0 (volle slaaptijd) | −2,9 | −2,5 | 3,4 | 2,9 | 10,3 | 10/13 | 0,992 | 0,987 | 11/13 |

De grootste afwijking van 0.37.0 (+41,6 /u: 83,5 tegen 41,9 mens, 656 artefact-epochs) wordt op 0.38.0
−2,9; de grootste op 0.38.0 is L10/H12 (−10,3). De klasseverschillen op 0.38.0 zijn de twee van §5 (L06/H07 en L10/H12, machine een klasse lager);
0.37.0 had L09/H11 (een klasse hoger) en L10. Vier nachten buiten §5: 8,4 tegen 6,8; 41,9 tegen 39,0;
2,9 tegen 0,0; 64 tegen 58,7 — alle vier in de klasse van de mens.

## 11. Herrekening van de 75 laatste diagnostische PSG's op 0.38.0, met artefactkolom en EDF+ (02-11 22:03 → 03-11 00:14)

**Selectie en uitvoering.** De 75 meest recente `diagnostic_psg`-jobs met resultaten én bron-EDF op
de server (nachten 07-10 → 02-11), H01–H75 in omgekeerde uploadvolgorde; H01–H15 zijn de vijftien van
§10. Drie nachten zijn dubbel geüpload (H07/H08, H35/H36, H68/H69), dus 72 patiënten; de cijfers
hieronder tellen per patiënt (n = 72), de tabel op de server telt per job. Dertien van de 72 hebben
een menselijke AHI (§10b, H02 vervallen 03-11); voor de overige 59 is er geen. De 60 jobs die nog niet op 0.38.0 stonden
(0.37.0 ×4, 0.36.2 ×19, 0.36.1 ×20, 0.35.1 ×6, 0.34.0 ×11) zijn om 22:03 met een job-timeout van
3600 s op de lege wachtrij gezet, acht workers, laatste klaar om 00:14 (2 u 11 min); 60/60 `finished`,
nul fouten (het failed-register bevat alleen entries van maart–oktober plus de eerste H11-poging
van §10). De oude uitvoer (results, PDF, Excel, EDF+, config; 300 bestanden, 22,7 GB) staat in
`metingen/herrekening75_20261102/backup_pre0380/`; tabel en CSV in dezelfde map, mode 600.

**Artefactkolom.** Per nacht `art_pct_slaap` = 100 · (slaapminuten − artefactvrije slaapminuten) /
slaapminuten uit `respiratory.summary` (voor oudere versies gereconstrueerd uit `tst_minutes` en
`n_artifact_epochs_excluded` × 0,5), met graad A < 10 %, B 10–25 %, C 25–50 %, D ≥ 50 %. Daarnaast de
signaalgraad uit `channel_quality` (per kanaal vlak/clip/ruis over de hele opname, regel: ≥ 2 kanalen
poor → poor, 1 poor of ≥ 3 acceptable → acceptable), hier herberekend **zonder het polskanaal**: de
vlakke-detector (flat > 20 % → poor) zet Pulse op 65/75 nachten op poor, omdat die integerreeks
(0–255, lage rate) lange reeksen gelijke samples bevat; met pols mee zou 56/75 "acceptable" en 15/75
"poor" zijn, zonder pols 34 good, 37 acceptable, 4 poor (per patiënt 31/37/4). Dat is een bevinding over de detector, niet
over de montages; de bibliotheek is niet aangepast (losse eind).

**Verschuiving voor → na (n = 72).** Mediaan −0,3 /u, gemiddeld −4,9; 45 lager, 3 gelijk, 24 hoger;
|Δ| ≤ 1 voor 28, ≤ 2 voor 43, > 5 voor 13. Mediane AHI na 17,0; ernst na: normaal 5, licht 28,
matig 16, ernstig 23 (voor: 6/25/17/24). De verschuiving volgt het artefactpercentage
(Spearman −0,70):

| artefactgraad | n | Δ mediaan | Δ bereik |
|---|---|---|---|
| A (< 10 %) | 52 | −0,1 | −8,1 … +5,1 |
| B (10–25 %) | 13 | −3,9 | −10,5 … +0,5 |
| C (25–50 %) | 4 | −33,3 | −41,4 … −3,6 |
| D (≥ 50 %) | 3 | −44,5 | −105,3 … −44,2 |

De dertien met |Δ| > 5: twaalf omlaag, waarvan zes met graad C/D (H06 130,5 → 97,1; H13 83,5 → 39,0;
H24 135,9 → 30,6 bij 77,5 % artefact; H37 96,2 → 63,0; H48 98,2 → 56,8; H55 59,2 → 15,0 bij 75,3 %),
vier met graad B (H20, H25, H34, H49: −5,6 … −10,5) en twee met graad A (H26 −6,0, H44 −8,1; beide
0.36.x, dus meer dan de noemer); één omhoog (H68 8,2 → 13,3, 0.34.0). Per oude versie: 0.37.0 n = 18
mediaan −0,8 en geen enkele hoger (zuiver noemereffect); 0.36.2 n = 18 −1,0; 0.36.1 n = 20 −0,8;
0.35.1 n = 6 −0,4; 0.34.0 n = 10 **+2,5 en alle tien hoger** — die nachten overbruggen vier releases
(arousaldetector, waakpoort, hypopneeroute) en zijn hier niet ontleed. Acht klassewissels: drie
matig → licht op B-nachten (H07 17,8 → 13,5; H34 20,4 → 14,8; H61 18,1 → 14,2), H55 ernstig → matig
(D), twee grensgevallen (H11 5,0 → 4,9; H41 4,8 → 5,3) en twee omhoog uit 0.34.0 (H72 11,5 → 15,1;
H73 2,3 → 6,9). Dunste noemer: H23 met 62,5 min slaap op 11 u opname (AHI 18,2).

**Signaalkolom.** Poor zonder pols: H03 (thermistor, thorax), H11, H25 (thermistor, abdomen), H30
(thermistor, thorax, abdomen). ECG poor op 9 (clipping), thorax 6, thermistor 4. H11 heeft 23,6 %
vlak op álle kanalen tegelijk: de recorder liep ~2,6 u door na het afkoppelen; in de slaap zelf is
1,7 % artefact. De RIP-paarpoort meldt op **43 van de 75 jobs** een verdacht paar (energieverhouding
thorax/abdomen in de orde 10³–10⁴: de thoraxband levert een fractie van de abdomenamplitude, zonder
vlakke stukken) en schakelt dan naar abdomen alleen; dat is een montage- of versterkingsvraag voor de
thoraxband in DOMINO, geen softwarevraag. `n_low_conf_noise` > 0 op 5 jobs.

**EDF+.** Alle 75 `_scored.edf` zijn geldig EDF+C met annotatiekanaal, 287,6–374,2 MB, 8,5–11,0 u;
hardlinks in `metingen/herrekening75_20261102/edfplus/<H>_<pseudoniem>_scored.edf` (27,5 GB zonder
extra schijf) met `edfplus_manifest.csv`; per job ook via `/results/<job_id>/edfplus`. Annotaties per
bestand: stadia per 30 s (1015–1320), respiratoire events 0–599, arousals 6–551, PLM 0–114 (37
bestanden met 0), artefact-epochs 23–1125, desaturaties ≤ 200 — de schrijver kapt op 200 en dat
raakt 16 van de 75 bestanden (losse eind: cap verhogen of weglaten). De bron-EDF's van DOMINO
dragen zelf alleen `Start`/`Lights Off`/`Lights On` (zie `somnomedics_domino_edfplus_20261102.md`).

**Beperkingen.** Voor 56 van de 60 herrekende jobs verschilt meer dan de noemer (0.34.0–0.36.2 →
0.38.0), dus Δ is daar een versieverschil, geen noemereffect; alleen de 18 van 0.37.0 isoleren de
noemer. De artefactgraad is het aandeel van de slaap dat de respiratoire artefactpoort op 0.38.0
uitsluit (wat 0.37.0 uit de noemer liet), niet de EEG-artefactdetector (die staat apart in de CSV als
`eeg_art_pct_tib`). Geen menselijke referentie voor 58 van de 72.

## 12. Mens tegen machine op 55 patiënten (03-11: Barts aanvulling van de 75-tabel; H02 verwijderd 14:33)

**Bron.** Bart vulde de kolom "mens" van de 75-tabel aan met de menselijke AHI's die hij ter
beschikking had (document van 03-11 12:45, correctie 14:33: de waarde voor H02 (95) was fout en is
verwijderd; regel 1A, één scoorder): 58 jobs = 55 patiënten (H08, H36 en H69 zijn dubbele uploads met
dezelfde waarde als hun partner); 17 jobs zonder (H02, H23, H24, H33, H37, H39–H48, H52, H75). Van de
14 van §10b blijven er 13. Eén bron: `metingen/herrekening75_20261102/humaan.csv` (58 rijen). Alle
cijfers hieronder per patiënt, n = 55, machine − mens.

| | 0.38.0 (na) | vorige versie (voor) |
|---|---|---|
| mediaan Δ | −1,2 | −1,1 |
| gemiddeld Δ (SD) | −1,2 (6,3) | +1,7 (11,2) |
| MAE / mediaan \|Δ\| / max \|Δ\| | 4,2 / 2,9 / 22,6 | 6,3 / 3,1 / 49,8 |
| binnen ±5 / binnen ±10 | 42/55 / 50/55 | 37/55 / 47/55 |
| lager / hoger dan de mens (tekentoets) | 36 / 19 (p = 0,030) | 35 / 20 (p = 0,058) |
| Pearson / Spearman / Lin CCC | 0,963 / 0,930 / 0,961 | 0,913 / 0,900 / 0,900 |
| Bland-Altman LoA | −13,5 … +11,1 | −20,3 … +23,7 |
| ernstklasse exact / binnen één klasse / gewogen κ | 47/55 / 55/55 / 0,86 | 45/55 / 54/55 / 0,80 |

"Vorige versie" is voor 38 van de 55 een oudere release (0.34.0–0.36.2), dus die kolom meet meer dan
de noemer; de winst van 0.38.0 zit vooral in de staart (max 49,8 → 22,6; LoA-breedte 24,7 tegen 44,0), de
mediaan beweegt niet.

**Klassetabel** (mens in de rij, 0.38.0 in de kolom): normaal 3/3; licht 18/20 + 2 naar matig
(H55 9,4 → 15,0 bij 75 % artefact; H56 12,3 → 22,1); matig 10/16 + **6 naar licht** (H07 19,8 → 13,5;
H12 19,0 → 8,7; H17 18,2 → 10,0; H19 15,2 → 11,2; H61 18,5 → 14,2; H74 15,4 → 12,4); ernstig 16/16.
Als beslisdrempel: AHI ≥ 15 sensitiviteit 26/32 (0,81), specificiteit 21/23 (0,91); ≥ 5 en ≥ 30
allebei 100 %. Per band: mens < 15 n = 23 mediaan Δ −0,4 (−4 %); 15–30 n = 16 **−2,8 (−14 %)**;
≥ 30 n = 16 −3,6 (−6 %). De absolute afwijking groeit met de AHI: de vier grootste zijn alle ernstig
en gaan beide kanten op (H49 −22,6; H58 +20,9; H25 −19,8; H26 +11,6), zonder klassegevolg; daarna
volgt H12 (−10,3). Een proportionele bias is er niet.

**Wat de zes matig-naar-licht-missers gemeen hebben.** Globaal hangt Δ met niets samen (Spearman:
arousal-index, ODI3, artefact, slaapminuten alle \|ρ\| < 0,15; alle correlaties in deze sectie zijn
Spearman). Maar bij de zes missers is de machine-ODI3 gedeeld door de menselijke AHI mediaan 0,45
(0,16–1,24; vier van de zes < 0,5) tegen 0,8 bij de tien treffers in dezelfde band (0,64–2,0); hun
arousal-index is 14,7 tegen 20,2, hun artefactaandeel 10,6 % tegen 3,1 %. De menselijke AHI op die
nachten wordt dus gedragen door hypopneeën zónder 3 %-desaturatie, onder regel 1A de arousal-gekoppelde;
daarvan scoort de machine er te weinig (duidelijkst H12: ODI3 3,1 tegen AHI 19). Of de arousaldetector
die arousals mist of de koppeling ze afwijst is alleen op eventniveau te zien (DOMINO-ASCII-export,
open vraag §11).

**Kwaliteit tegen afwijking (figuur `bland_altman_56.png` op de server in
`metingen/herrekening75_20261102/`; bestandsnaam ongewijzigd, inhoud n = 55).** De absolute afwijking
hangt zwak samen met het artefactaandeel (Spearman \|Δ\| tegen artefact % 0,39; tegen het
EEG-artefactaandeel over de opname 0,15); de getekende afwijking niet (−0,10 en −0,08). Per groep:

| groep | n | Δ mediaan | \|Δ\| mediaan | binnen ±5 | klasse gelijk |
|---|---|---|---|---|---|
| artefact A (< 10 %) | 41 | −1,2 | 2,4 | 34/41 | 37/41 |
| artefact B (10–25 %) | 11 | −0,2 | 4,8 | 6/11 | 8/11 |
| artefact C + D | 3 | −2,9 | 4,9 | 2/3 | 2/3 |
| signaal good | 23 | −1,0 | 2,4 | 17/23 | 19/23 |
| signaal acceptable | 28 | −1,4 | 3,0 | 22/28 | 24/28 |
| signaal poor | 4 | −0,3 | 1,7 | 3/4 | 4/4 |
| thoraxband normaal | 23 | −0,5 | 1,6 | 20/23 | 21/23 |
| thoraxband bijna vlak (paarpoort) | 32 | −1,2 | 3,1 | 22/32 | 26/32 |

Artefact B–D tegen A: \|Δ\| 4,9 tegen 2,4 (Mann-Whitney p ≈ 0,09); thoraxband bijna vlak tegen normaal
3,1 tegen 1,6 (p ≈ 0,04); signaal poor tegen de rest 1,7 tegen 2,9 (p ≈ 0,55, n = 4). Geen van de
kwaliteitsmaten verklaart de richting van de fout, en de zes matig-naar-licht-missers liggen op drie
A-nachten en drie B-nachten met de thoraxband vier keer wel en twee keer niet verdacht: de
onderscoring is geen kwaliteitsprobleem. Artefactrijke nachten (B–D, 14 van de 55) en nachten met een
bijna vlakke thoraxband zijn wel minder precies: de twee grootste negatieve afwijkingen (H49 −22,6;
H25 −19,8) zijn B-nachten.

**Beperkingen.** Eén scoorder, alleen de AHI (geen eventlijst, geen menselijke TST); 55 van de 72 naar
beschikbaarheid, niet naar toeval; de 15 van §10 waren de aanleiding voor de noemerwijziging (de
voorspellingen gingen wel aan de menselijke cijfers vooraf); de menselijke waarden zijn met de hand
overgenomen, zonder dubbele invoer (één invoerfout, H02, is door Bart zelf gevonden en verwijderd).

## 13. Herrekening van de resterende 178 PSG's op 0.38.0 (03-11 13:12 → 18:02): de telling stijgt, en een half-dode thermistor levert honderden apneus

**Selectie en uitvoering.** Alle overige jobs met `study_type` `diagnostic_psg` (170) of
`titration_psg_cpap` (8) én bron-EDF op de server, niet in de 75 van §11: uploads 04-04 → 03-11,
labels H76–H245 en T01–T08 in omgekeerde uploadvolgorde (vier `titration_pg_cpap`-jobs en vier
zonder studietype zijn niet meegenomen). Twee stonden al op 0.38.0 (H76, H77, upload 03-11
ochtend); de overige 176 — oude versie 0.34.0 ×87, 0.32.0 ×13, 0.31.x ×13, 0.30.0 ×7, 0.27.x ×7,
0.36.x ×2, en 47 zonder versie in de oude uitvoer — zijn vanaf 13:12 gedoseerd op de wachtrij
gezet (besturing op de server: hooguit zes tegelijk, twee workers vrij voor de kliniek, timeout
3600 s), laatste klaar 18:02 (4 u 50 min); 176/176 `finished`, nul fouten, daarna wachtrij 0/0,
site 200, schijf 268 GB gebruikt van 3,1 TB. De oude uitvoer (876 bestanden, 62,0 GB) staat in
`metingen/herrekening_rest_20261103/backup_pre0380/`; tabel, CSV's, docx/pdf en EDF+-hardlinks in
dezelfde map.

**Wat telt mee.** Zes van de 170 "diagnostische" jobs zijn geen patiënten: H239, H240 en H242–H244
zijn de PSG-IPA-opnames SN1–SN5 (testuploads 04/05-08) en H245 is het YASA-voorbeeldbestand van
04-04 (zeven EEG/EOG/EMG/ECG-kanalen, geen ademhaling, dus geen AHI). Vier nachten zijn exact
dubbel geüpload (H92/H93, H124/H125, H171/H172, H230/H231); H83/H84 zijn twee uploads van dezelfde
nacht met andere uitkomst (oude uitvoer al verschillend, 148 tegen 145 events; voor beide 17,9,
na 23,4 en 25,1) en tellen allebei. Drie hadden geen eerder
resultaat (H78: geen results.json; H218 en H219 van 12-08: 100 % artefact, AHI 1,8 en 2,7 zonder
arousal-index — onbruikbaar, niet verder nagekeken). Blijven 155 unieke kliniek-nachten met een
voor- én na-AHI (uploads 05-08 → 06-10) en acht titratie-PSG's. Geen enkele heeft een menselijke
AHI; de kolom "mens" in de tabel is leeg.

**Verschuiving voor → na (n = 155).** Mediaan +2,7 /u, gemiddeld +2,6 (SD 14,5), IQR +0,4 … +6,0,
bereik −105,1 … +68,0; 123 hoger, 32 lager (tekentoets p < 0,001); |Δ| ≤ 2 bij 48, ≤ 5 bij 97,
> 10 bij 27. Mediane AHI voor 12,7, na 17,7. Dat is het tegendeel van §11 (mediaan −0,3): daar
overbrugden de meeste nachten alleen 0.36.x/0.37.0 → 0.38.0, hier vier tot elf minor-releases
(0.34.0 → 0.38.0: 0.35–0.38; 0.27.0 → 0.38.0: 0.28–0.38). Ontleed
in telling en noemer (AHI = events · 60 / noemer; restterm ≤ 0,1): de telling geeft mediaan +4,0 /u
(140 van 155 nachten meer events, mediaan +23 events), de noemer mediaan −0,3 en op geen enkele
nacht positief (de volle slaaptijd is nooit korter dan de artefactvrije). Per eventtype, uit de
eventlijsten: hypopneeën mét desaturatie ≥ 3 % mediaan +1,6 /u (141 hoger, 8 lager; 4 526 → 6 406
events), hypopneeën zonder desaturatie +0,7 /u (97 hoger, 38 lager; 1 531 → 2 409), apneus 0,0
(71 hoger, 56 lager; 10 117 → 12 137, waarvan `thermistor_only` 692 → 1 900 — zie verderop). De
stijging is dus geen arousal-effect: het aandeel hypopneeën zonder desaturatie blijft 25 → 27 %,
ΔAI mediaan +0,5 (Spearman ΔAHI–ΔAI 0,32). Welke release de desaturatiehypopneeën erbij haalt is
niet ontleed; kandidaten uit de CHANGELOG van 0.36.0 zijn de SpO2-nadir op de tijdas en de continue
herstelbasislijn. Per oude versie (Δ telling en noemer als mediaan):

| versie voor | n | Δ mediaan | hoger/lager | Δ telling | Δ noemer |
|---|---|---|---|---|---|
| 0.27.x | 7 | +4,5 | 5/2 | +7,1 | −0,4 |
| 0.30.0 | 7 | +1,7 | 5/2 | +4,4 | −0,2 |
| 0.31.x | 13 | +2,2 | 12/1 | +3,7 | −0,5 |
| 0.32.0 | 12 | +4,1 | 11/1 | +5,5 | −0,3 |
| 0.34.0 | 79 | +3,7 | 62/17 | +4,4 | −0,3 |
| zonder versie | 37 | +1,6 | 28/9 | +1,7 | −0,2 |

De ODI-kolom is in deze batch voor en na gelijk (175 van 176; alleen T02, 0.36.1, 9,3 → 9,4): 0.34.0 en ouder deelden de ODI al
door de volle slaaptijd (de AHI niet), alleen 0.36.x–0.37.0 deelden door artefactvrije slaap.

**Artefact en klassen.** De noemer werkt alleen op artefactrijke nachten: graad A (n = 125)
Δ mediaan +3,6 (noemer −0,2), B (14) +1,6 (−2,3), C (10) −2,4 (−10,8), D (6) −17,3 (−22,4);
Spearman Δ–artefact% −0,32. De zes grootste dalingen zijn noemereffecten op C/D-nachten (H112
156,8 → 51,7 bij 72,6 % artefact; H162 153,7 → 90,0; H209 81,1 → 23,1; H192 49,5 → 20,9; H187
80,4 → 52,8; H146 38,3 → 23,5), zoals in §11; dunste noemer H111 met 66 min slaap (AHI 30,0).
Ernstklasse gelijk 110/155 (71 %), 41 hoger, 4 lager. Over de grens 15: 20 omhoog, 1 omlaag
(na ≥ 15: 84 tegen 65 voor); grens 30: 11 omhoog, 3 omlaag; grens 5: 14 omhoog, 0 omlaag.

| voor \ na | normaal | licht | matig | ernstig |
|---|---|---|---|---|
| normaal (37) | 23 | 12 | 2 | 0 |
| licht (53) | 0 | 35 | 16 | 2 |
| matig (25) | 0 | 1 | 15 | 9 |
| ernstig (40) | 0 | 0 | 3 | 37 |

**Apneus op een half-dode thermistor (bevinding over 0.38.0 zoals het draait).** De thermistorpoort
(`envelope_agreement`, sinds 0.36.2 met de dode-thermistorvloer) wijst op 0.38.0 de thermistor af
op 68 van de 178 jobs (oude uitvoer: 1); op de 57 kliniek-nachten met afgewezen thermistor is
Δ apneus mediaan 0,0 /u. Op negen nachten keurt de poort een thermistor goed die de kanaalkwaliteit
als poor graadt (vlak > 20 % van de opname): daar is Δ apneus mediaan +13,3 /u en ΔAHI +16,0
(tegen 0,0 en +1,6 … +3,5 in de drie andere groepen: afgewezen 57, goedgekeurd good 68,
goedgekeurd acceptable 21), en op zes van de negen stijgt het apneuaantal met meer dan 5 /u:
H97 97 → 240, H98 4 → 492, H100 429 → 509, H107 129 → 268, H108 219 → 291, H157 89 → 196 —
op H100 na vrijwel allemaal `thermistor_only` (H98 1 → 490, H97 88 → 230, H107 98 → 246, H108
30 → 111, H157 42 → 142). H98 is het zuiverste geval: de thermistor ligt 277 min (42 % van de
opname) in één blok vlak; alle 492 nieuwe apneus liggen in dat blok (uren 7–10 van de opname:
120, 138, 157, 77), mediaan 10,0 s (de minimumduur) en 93 % ≤ 10,5 s, 465 van 466 zijn `borderline`, en
de AHI gaat van 6,1 naar 74,1 bij ODI3 4,1 en AI 13,9. Op H97, H107 en H157 (63, 67 en 18 losse
vlakke stukken) zijn het lange events (mediaan 44, 37 en 33 s); of die op de vlakke stukken
liggen is niet toetsbaar, want de intervallijst kapt op tien (H97: 0 van 240 binnen de gelijste,
H157: 9 van 196). De vereniging van de twee sensoren op `aasm_v3_breath_dual` is additief; de
vloer van 0.36.2 (bandvermogen; het venster staat niet in de CHANGELOG) laat kennelijk een
thermistor door die een deel van de nacht dood is; de kanaalkwaliteitsmodule ziet het (interval
gelijst) maar is informatief. Het spiegelbeeld staat als keerzijde in de CHANGELOG van 0.35.0
(426 druk-apneus op één nacht die een goedgekeurde thermistor niet zag); de daar gebouwde
`dual_sensor_confirmation` (default uit) eist een gevolg voor alleen-druk-apneus en voor
thermistor-only apneus alleen bij een afgekeurde thermistor, en dekt dit geval zoals beschreven
dus niet. In de 75 van §11 komt het niet
voor (Δ apneus mediaan 0,0; thermistor poor op vier nachten zonder apneusprong). Vijf AHI's uit
deze batch zijn daarmee niet bruikbaar zoals ze staan (H97, H98, H107, H108, H157; in de tabel met
een sterretje). Niet verklaard: H136 (apneus 123 → 333, 303 `pressure_only`) en H153 (66 → 193: 86
`pressure_only`, 82 `both`, 25 `thermistor_only`), beide thermistor acceptable, ODI3 15,0 en
11,5, en H170 (`aasm_v3_breath`, 25 → 96
op goede signalen, ODI3 4,5, 30 events met lage zekerheid). De bibliotheek is niet aangepast.

**Signaal, titratie, EDF+.** Signaalgraad zonder pols: good 89, acceptable 57, poor 9 (|Δ|
mediaan 3,7, 3,9 en 7,0); pols poor op 136 van 155 (dezelfde vlakdetector-bevinding als §11);
RIP-paarpoort verdacht op 105 van 155, single-channel op 111. Profielen in de configs:
`aasm_v3_breath_dual` 94, `aasm_v3_breath` 58 (uploads tot begin september), `aasm_v3_rec` 3 —
de herrekening gebruikt het profiel van de config, niet de huidige standaard. Titratie-PSG's
(T01–T08): Δ mediaan −0,2 (T05 −12,7 bij 18,3 % artefact, verder −1,5 … +1,9), vier hoger, vier
lager, klasse 8/8 gelijk. EDF+: 177 van 178 geldig EDF+C met annotaties, 61,8 GB hardlinks in
`edfplus/`; H159 faalt in alle drie de schrijvers (digitaal bereik 8 388 607 uit een 24-bits
BDF-bron: pyedflib weigert, MNE struikelt over een headerveld, edfio laat een bestand van 7 kB
zonder data achter) terwijl results, PDF en Excel gewoon klaar zijn — losse eind naast de
desaturatiecap van 200, die hier 32 van de 178 bestanden raakt.

**Beperkingen.** Geen menselijke referentie. Δ is een versieverschil over vier tot elf minor-releases;
alleen de ontleding in telling en noemer is exact, de toewijzing van de extra
desaturatiehypopneeën aan een release niet. Vier uploads dragen geen ANON-pseudoniem: twee (H159,
H229) staan alleen in de CSV op de server, twee (H199, H212) dragen een studielabel. De vlakke-intervallijst van de kanaalkwaliteit kapt op tien
intervallen, zodat de toets "apneus binnen vlakke stukken" alleen op H98 (één interval) sluitend
is.
