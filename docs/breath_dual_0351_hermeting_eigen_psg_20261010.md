# Hermeting van de 20 eigen PSG's na de uitrol van psgscoring 0.35.1 (strictness 0,30)

Datum: 2026-10-10, 06:19–07:28 CEST. Preregistratie
`docs/breath_dual_0351_hermeting_eigen_psg_preregistratie_20261010.md` (64ef149, geschreven vóór
de meting). In situ op de Hetzner-server in een losse container uit het productie-image
YASAFlaskified 0.38.10 (psgscoring **0.35.1**), uploads read-only, 16 van 32 CPU's, `nice`, 8
analyses parallel, 69 min; de app antwoordde op alle 66 controles met 200. Dezelfde 20 opnames
(R01–R20) en dezelfde invoer per opname als op 07-10 (opgeslagen YASA-hypnogram, opgeslagen
artefact-epochs, dezelfde kanaalkeuze en inleesroute); koppeling naar de jobs blijft op de
server (`/data/slaapkliniek/metingen/breath_dual_0351_20261010/`, mode 600). Vier armen per
opname: A `aasm_v3_breath` en B `aasm_v3_breath_dual` op de nieuwe default (strictness 0,30),
C en D dezelfde profielen met de strictness in-process teruggezet op 0,50 (driftcontrole; de
gelezen waarde is per arm vastgelegd en klopt op 20/20 × 4). Ruwe uitvoer zonder
patiëntgegevens: `/srv/CODE/docs/breath_dual_0351_20261010/R*.json`, analyse `analyse.py`,
samenvatting `samenvatting.json`. **Verschilmeting zonder referentie**: er is geen menselijke
scoring van deze nachten; wat hieronder "beter" of "slechter" zou zijn, kan dit verslag niet
zeggen.

## 1. Driftcontrole eerst (preregistratie §1): niets anders dan de strictness is veranderd

- D (`breath_dual` op 0.35.1 met strictness 0,50) tegen de dual-arm van 07-10 (0.34.0): identiek
  op AHI, apneus O/C/M, hypopneeën, eventtelling, RDI, RERA-telling, OAHI en AHI-incl-uncertain op
  **20/20**.
- C (`breath` op 0.35.1 met strictness 0,50) tegen de breath-arm van 07-10: idem, **20/20**.
- De arousaltelling is identiek over de vier armen op 20/20 (zelfde detector, zelfde hypnogram);
  07-10 legde geen arousaltelling vast, dus een vergelijking met 07-10 op de arousal-index is niet
  mogelijk — de identiteit van D met 07-10 op RDI en RERA-telling zegt wel dat de koppeling niet
  verschoven is.

De CHANGELOG-claim van 0.35.0 ("de strictness en de familie-vlag zijn de enige
standaardveranderingen") houdt op deze 20 nachten stand: wat B van 07-10 scheidt, is uitsluitend
de strictness. De vergelijking "B tegen D binnen 0.35.1" geeft dan ook op elk veld dezelfde
getallen als "B tegen dual 07-10" (zie `analyse_uitvoer.md`).

## 2. Primair — productiestandaard `breath_dual` op 0,30 tegen dezelfde opnames op 0,50 (07-10)


| R | tertiel | AHI oud → nieuw | ΔAHI | ernst oud → nieuw | apneus O/C/M oud → nieuw | hypopneeën oud → nieuw | RDI oud → nieuw | ΔRDI | RERA-index oud → nieuw | VB oud → nieuw |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | hoog | 55.9 → 58.5 | +2.6 | ernstig → ernstig | 202/0/1 → 202/0/1 | 15 → 25 | 57.4 → 59.8 | +2.4 | 1.5 → 1.3 | 62.6 → 62.6 |
| R02 | hoog | 23.2 → 25.1 | +1.9 | matig → matig | 105/2/0 → 105/2/0 | 61 → 75 | 28.3 → 29.8 | +1.5 | 5.1 → 4.7 | 32.0 → 32.0 |
| R03 | midden | 14.5 → 24.9 | +10.4 | licht → matig | 3/0/0 → 3/0/0 | 29 → 52 | 31.3 → 41.2 | +9.9 | 16.8 → 16.3 | 23.3 → 23.3 |
| R04 | midden | 7.8 → 9.0 | +1.2 | licht → licht | 28/0/0 → 28/0/0 | 11 → 17 | 9.2 → 10.4 | +1.2 | 1.4 → 1.4 | 17.8 → 17.8 |
| R05 | laag | 1.4 → 3.7 | +2.3 | normaal → normaal | 0/0/0 → 0/0/0 | 11 → 28 | 3.4 → 5.5 | +2.1 | 2.0 → 1.8 | 8.8 → 8.8 |
| R06 | laag | 4.6 → 8.0 | +3.4 | normaal → licht | 2/0/0 → 2/0/0 | 6 → 12 | 8.6 → 11.4 | +2.8 | 4.0 → 3.4 | 21.3 → 21.3 |
| R07 | laag | 2.5 → 4.4 | +1.9 | normaal → normaal | 0/0/0 → 0/0/0 | 4 → 7 | 8.1 → 9.4 | +1.3 | 5.6 → 5.0 | 14.8 → 14.8 |
| R08 | midden | 8.3 → 12.3 | +4.0 | licht → licht | 22/0/0 → 22/0/0 | 29 → 53 | 12.1 → 15.4 | +3.3 | 3.8 → 3.1 | 19.4 → 19.4 |
| R09 | midden | 29.9 → 34.2 | +4.3 | matig → ernstig | 36/0/0 → 36/0/0 | 26 → 35 | 34.2 → 38.5 | +4.3 | 4.3 → 4.3 | 33.8 → 33.8 |
| R10 | midden | 17.9 → 21.2 | +3.3 | matig → matig | 101/0/0 → 101/0/0 | 44 → 71 | 22.0 → 24.9 | +2.9 | 4.1 → 3.7 | 36.8 → 36.8 |
| R11 | midden | 22.1 → 26.1 | +4.0 | matig → matig | 65/3/0 → 65/3/0 | 39 → 58 | 28.3 → 31.9 | +3.6 | 6.2 → 5.8 | 53.5 → 53.5 |
| R12 | hoog | 47.5 → 55.1 | +7.6 | ernstig → ernstig | 27/7/0 → 27/7/0 | 53 → 67 | 58.4 → 64.9 | +6.5 | 10.9 → 9.8 | 70.2 → 70.2 |
| R13 | midden | 10.0 → 11.4 | +1.4 | licht → licht | 23/0/0 → 23/0/0 | 5 → 9 | 17.1 → 18.5 | +1.4 | 7.1 → 7.1 | 39.0 → 39.0 |
| R14 | hoog | 90.7 → 92.2 | +1.5 | ernstig → ernstig | 158/0/0 → 158/0/0 | 145 → 150 | 98.5 → 100.0 | +1.5 | 7.8 → 7.8 | 74.3 → 74.3 |
| R15 | laag | 2.6 → 3.9 | +1.3 | normaal → normaal | 6/0/0 → 6/0/0 | 7 → 14 | 5.5 → 6.5 | +1.0 | 2.9 → 2.6 | 14.2 → 14.2 |
| R16 | hoog | 24.5 → 32.2 | +7.7 | matig → ernstig | 71/2/0 → 71/2/0 | 48 → 86 | 30.0 → 36.4 | +6.4 | 5.5 → 4.2 | 41.1 → 41.1 |
| R17 | laag | 5.1 → 8.9 | +3.8 | licht → licht | 4/0/0 → 4/0/0 | 27 → 50 | 5.6 → 9.4 | +3.8 | 0.5 → 0.5 | 15.5 → 15.5 |
| R18 | hoog | 103.3 → 110.9 | +7.6 | ernstig → ernstig | 439/2/0 → 439/2/0 | 72 → 110 | 109.9 → 117.3 | +7.4 | 6.6 → 6.4 | 68.2 → 68.2 |
| R19 | laag | 1.3 → 2.2 | +0.9 | normaal → normaal | 1/0/0 → 1/0/0 | 11 → 19 | 3.4 → 3.8 | +0.4 | 2.1 → 1.6 | 7.4 → 7.4 |
| R20 | laag | 6.0 → 9.4 | +3.4 | licht → licht | 12/0/0 → 12/0/0 | 23 → 43 | 7.7 → 10.6 | +2.9 | 1.7 → 1.2 | 12.7 → 12.7 |

- ΔAHI: mediaan +3.3, gemiddeld +3.73, bereik +0.9 … +10.4 /u; gelijk op 0/20; niet-nul 20, positief 20, exacte tekentoets p = 1.91e-06.
- ΔRDI: mediaan +2.8, gemiddeld +3.33.
- Ernstklasse-wissels: 4/20: R03 licht→matig, R06 normaal→licht, R09 matig→ernstig, R16 matig→ernstig
- Per tertiel (ΔAHI gemiddeld / mediaan / max): hoog n=6 +4.82 / +5.1 / +7.7; laag n=7 +2.43 / +2.3 / +3.8; midden n=7 +4.09 / +4.0 / +10.4
- Vlaggen (preregistratie §4): R03: ΔAHI +10.4

Lezing:

- **Elke nacht gaat omhoog, nooit omlaag:** ΔAHI mediaan +3.3, gemiddeld +3.73, bereik
  +0.9 … +10.4 /u; 20/20 positief (exacte tekentoets p = 1.9e-06). De MESA-verwachting
  (bias −2,74 → +0,52, dus ≈ +3,3 /u gemiddeld) komt uit; de mediaan ligt in het vooraf genoemde
  venster 0…+5.
- **Alleen hypopneeën bewegen.** Apneus O/C/M identiek op 20/20 (zoals verwacht: de strictness
  raakt alleen de hypopneu-gradering). Hypopneeën onder `breath_dual`: 666 → 981 over de 20
  nachten (+47 %). De ventilatoire last is op 20/20 identiek (ze hangt niet van de
  eventlijst af), de RERA-index daalt licht op 15/20 (events die RERA waren, worden nu hypopneu),
  en de RDI stijgt daardoor iets minder dan de AHI (mediaan +2.8 tegen +3.3).
- **Ernstklasse wisselt op 4/20, telkens één klasse omhoog:** R03 licht→matig, R06 normaal→licht, R09 matig→ernstig, R16 matig→ernstig.
  Geen sprong van twee klassen.
- **Per tertiel van de opgeslagen AHI** (ΔAHI gemiddeld / mediaan / max): laag n=7 +2.43 / +2.3 / +3.8;
  midden n=7 +4.09 / +4.0 / +10.4; hoog n=6 +4.82 / +5.1 / +7.7. De verschuiving groeit met de
  ernst in absolute zin; relatief is ze het grootst op de lichte nachten (R05 1,4 → 3,7, R19 1,3 → 2,2).
- **Vlag volgens preregistratie §4:** R03 (ΔAHI +10,4, licht → matig: 29 → 52 hypopneeën bij 3
  apneus en een RERA-index van 16,8 die nauwelijks beweegt — de extra hypopneeën komen dus niet uit
  de RERA-lijst maar uit kandidaten die op 0,50 nergens meetelden). Verder geen nacht boven +10, geen
  daling, geen afwijking van de verwachtingen.

## 3. Secundair — `breath` op 0,30 tegen `breath` 07-10


| R | tertiel | AHI oud → nieuw | ΔAHI | ernst oud → nieuw | apneus O/C/M oud → nieuw | hypopneeën oud → nieuw | RDI oud → nieuw | ΔRDI | RERA-index oud → nieuw | VB oud → nieuw |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | hoog | 55.9 → 58.5 | +2.6 | ernstig → ernstig | 201/0/0 → 201/0/0 | 17 → 27 | 56.9 → 59.5 | +2.6 | 1.0 → 1.0 | 62.6 → 62.6 |
| R02 | hoog | 17.3 → 19.9 | +2.6 | matig → matig | 23/0/0 → 23/0/0 | 102 → 121 | 22.1 → 24.2 | +2.1 | 4.8 → 4.3 | 3.8 → 3.8 |
| R03 | midden | 14.5 → 24.9 | +10.4 | licht → matig | 3/0/0 → 3/0/0 | 29 → 52 | 36.2 → 46.6 | +10.4 | 21.7 → 21.7 | 23.3 → 23.3 |
| R04 | midden | 6.6 → 8.0 | +1.4 | licht → licht | 21/0/0 → 21/0/0 | 12 → 19 | 8.0 → 9.4 | +1.4 | 1.4 → 1.4 | 15.4 → 15.4 |
| R05 | laag | 1.4 → 3.7 | +2.3 | normaal → normaal | 0/0/0 → 0/0/0 | 11 → 28 | 8.5 → 10.1 | +1.6 | 7.1 → 6.4 | 8.8 → 8.8 |
| R06 | laag | 4.6 → 8.0 | +3.4 | normaal → licht | 2/0/0 → 2/0/0 | 6 → 12 | 13.7 → 17.1 | +3.4 | 9.1 → 9.1 | 21.3 → 21.3 |
| R07 | laag | 2.5 → 4.4 | +1.9 | normaal → normaal | 0/0/0 → 0/0/0 | 4 → 7 | 8.1 → 9.4 | +1.3 | 5.6 → 5.0 | 0.0 → 0.0 |
| R08 | midden | 8.3 → 12.3 | +4.0 | licht → licht | 22/0/0 → 22/0/0 | 29 → 53 | 17.1 → 20.5 | +3.4 | 8.8 → 8.2 | 19.4 → 19.4 |
| R09 | midden | 29.9 → 34.2 | +4.3 | matig → ernstig | 36/0/0 → 36/0/0 | 26 → 35 | 32.8 → 37.1 | +4.3 | 2.9 → 2.9 | 33.8 → 33.8 |
| R10 | midden | 17.9 → 21.2 | +3.3 | matig → matig | 101/0/0 → 101/0/0 | 44 → 71 | 26.6 → 29.2 | +2.6 | 8.7 → 8.0 | 36.8 → 36.8 |
| R11 | midden | 13.4 → 18.2 | +4.8 | licht → matig | 0/0/0 → 0/0/0 | 65 → 88 | 19.2 → 23.6 | +4.4 | 5.8 → 5.4 | 6.9 → 6.9 |
| R12 | hoog | 47.5 → 55.1 | +7.6 | ernstig → ernstig | 27/7/0 → 27/7/0 | 53 → 67 | 55.1 → 62.7 | +7.6 | 7.6 → 7.6 | 70.2 → 70.2 |
| R13 | midden | 9.3 → 10.7 | +1.4 | licht → licht | 20/1/0 → 20/1/0 | 5 → 9 | 17.9 → 18.9 | +1.0 | 8.6 → 8.2 | 39.0 → 39.0 |
| R14 | hoog | 90.7 → 92.2 | +1.5 | ernstig → ernstig | 158/0/0 → 158/0/0 | 145 → 150 | 98.8 → 100.3 | +1.5 | 8.1 → 8.1 | 74.3 → 74.3 |
| R15 | laag | 2.6 → 3.9 | +1.3 | normaal → normaal | 6/0/0 → 6/0/0 | 7 → 14 | 9.3 → 10.0 | +0.7 | 6.7 → 6.1 | 14.2 → 14.2 |
| R16 | hoog | 24.5 → 32.2 | +7.7 | matig → ernstig | 71/2/0 → 71/2/0 | 48 → 86 | 31.8 → 38.7 | +6.9 | 7.3 → 6.5 | 41.1 → 41.1 |
| R17 | laag | 4.9 → 8.7 | +3.8 | normaal → licht | 3/0/0 → 3/0/0 | 27 → 50 | 5.6 → 9.4 | +3.8 | 0.7 → 0.7 | 15.5 → 15.5 |
| R18 | hoog | 66.0 → 80.7 | +14.7 | ernstig → ernstig | 30/2/0 → 30/2/0 | 296 → 369 | 71.0 → 85.5 | +14.5 | 5.0 → 4.8 | 27.3 → 27.3 |
| R19 | laag | 1.3 → 2.2 | +0.9 | normaal → normaal | 1/0/0 → 1/0/0 | 11 → 19 | 7.2 → 7.3 | +0.1 | 5.9 → 5.1 | 7.4 → 7.4 |
| R20 | laag | 4.8 → 8.2 | +3.4 | normaal → licht | 5/0/0 → 5/0/0 | 23 → 43 | 11.8 → 14.4 | +2.6 | 7.0 → 6.2 | 12.7 → 12.7 |

- ΔAHI: mediaan +3.3, gemiddeld +4.17, bereik +0.9 … +14.7 /u; gelijk op 0/20; niet-nul 20, positief 20, exacte tekentoets p = 1.91e-06.
- ΔRDI: mediaan +2.6, gemiddeld +3.81.
- Ernstklasse-wissels: 7/20: R03 licht→matig, R06 normaal→licht, R09 matig→ernstig, R11 licht→matig, R16 matig→ernstig, R17 normaal→licht, R20 normaal→licht
- Per tertiel (ΔAHI gemiddeld / mediaan / max): hoog n=6 +6.12 / +5.1 / +14.7; laag n=7 +2.43 / +2.3 / +3.8; midden n=7 +4.23 / +4.0 / +10.4
- Vlaggen (preregistratie §4): R03: ΔAHI +10.4; R18: ΔAHI +14.7

Hier is de verschuiving groter op de nachten waar `breath` zelf de meeste hypopneeën draagt
(R18: 296 → 369, ΔAHI +14,7; onder `breath_dual` is dat +7,6 omdat de vereniging daar al ruim 400
apneus extra telde (30 → 439 obstructief)). Ernstklasse wisselt op 7/20. Dit profiel is niet de productiestandaard;
de tabel staat hier omdat `breath` voor scoorders selecteerbaar blijft.

## 4. Wat dit betekent

1. **De uitrol van 0.35.1 verhoogt de AHI van elke nacht met ongeveer 3 /u** (bereik +0,9 … +10,4),
   uitsluitend via hypopneeën, en zet op 4 van 20 nachten de ernstklasse één stap omhoog. Dat is
   het gedrag dat de MESA-kalibratie voorspelde (daar ging de onderschatting van −2,74 naar +0,52
   en verbeterde de ernstklasse-overeenstemming van 78 naar 83 op 140), maar op deze 20 nachten
   kan niemand zeggen of de 4 wissels juist zijn — er is geen referentie. De dossiers R03, R06,
   R09 en R16 zijn de kandidaten om in de review-interface na te kijken; de koppeling staat op de
   server.
2. **Er is geen drift.** 0.35.1 met de oude strictness reproduceert 07-10 byte-voor-byte op elk
   vastgelegd veld. Wie een rapport van vóór 10-10 wil vergelijken met een nieuw, vergelijkt
   alleen de strictness.
3. **Verslagen van vóór en na de uitrol zijn niet direct vergelijkbaar** op AHI en RDI; het
   rapport draagt de bibliotheekversie en het profiel in de Herkomst-tabel, en `breath_dual` op
   0,50 is reproduceerbaar door de parameter expliciet te zetten.
4. Looptijd: 498 CPU-minuten over 20 nachten × 4 armen (mediaan 25.3 min per nacht), 69 min wandkloktijd
   op 16 CPU's; productie bleef bereikbaar.

## 5. Verificatie

Nagerekend 10-10 met een apart script rechtstreeks uit de 40 ruwe JSON's, los van `analyse.py`
(de onafhankelijke verificatie-agent viel uit op een gebruikslimiet; dit is dus een tweede
berekening door dezelfde auteur, geen onafhankelijke review). Klopt op elk punt:

- versie 0.35.1 op 20/20; gelezen strictness per arm 0,3/0,3 en 0,5/0,5 op 20/20; arousaltelling
  gelijk over de vier armen op 20/20;
- driftcontrole 20/20 voor beide profielen (AHI, O/C/M, hypopneeën, eventtelling, RDI, n_rera,
  OAHI, AHI-incl-uncertain);
- primair: ΔAHI mediaan +3,35 (afgerond +3,3), gemiddeld +3,73, bereik +0,9…+10,4, 20/20 positief,
  exacte tekentoets p = 1,9·10⁻⁶; hypopneeën 666 → 981; RERA-index lager op 15/20; VB en apneus
  gelijk op 20/20; ΔRDI mediaan +2,85 (afgerond +2,8); vier wissels R03/R06/R09/R16; tertielen
  zoals in §2;
- secundair: mediaan +3,35, gemiddeld +4,17, bereik +0,9…+14,7, zeven wissels; hypopneeën 960 → 1320;
- looptijd 498 CPU-minuten, mediaan 24,7 min per nacht (vier armen);
- preregistratie gecommit 06:18:02, start van de run 06:19:10; daarna niet gewijzigd.

Niet verifieerbaar uit de ruwe data: de 66 app-controles (alleen in de wachterlog op het werkstation).
