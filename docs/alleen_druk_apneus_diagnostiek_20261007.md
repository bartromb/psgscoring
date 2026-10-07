# Diagnostiek van de alleen-druk-apneus onder `breath_dual` — stap 1

Datum: 2026-10-07. Preregistratie `docs/alleen_druk_apneus_diagnostiek_preregistratie_20261007.md`
(15825c8, geschreven vóór de run). In situ op de Hetzner-server, zelfde 20 opnames, zelfde
container uit het productie-image (psgscoring 0.34.0), beide armen opnieuw mét eventlijsten,
arousallijst en per-apneu signaalkenmerken; 4 parallel, 40 min. Ruwe uitvoer zonder
patiëntgegevens: `/srv/CODE/docs/breath_dual_20261007/diag/R*_diag.json` (+ `_diag2.json`
voor de post-hoc variant), lezing `analyse_diag.py`, `samenvatting_diag.json`. AHI's per arm
identiek aan de meting van vanmiddag (20/20).

## Vraag
Welke foutmodus dragen de apneus die alleen de neusdruk ziet: **A** thermistor te ongevoelig
voor 0,90 (thermistordaling d_th ≥ 0,72), **C** hypopneu-niveau op de thermistor en de druk
overdrijft (0,30 ≤ d_th < 0,72), of **B** de thermistor ademt door — mond/canule
(d_th < 0,30; gesplitst op effortdaling < 0,50 = behouden)?

## Resultaat op de poort-aan-nachten (R02, R04, R07, R11, R18)

| | alleen-druk-apneus | waarvan al hypopneu onder `breath` | ΔAHI-dragend (onder `breath` niets) | dragend zonder gevolg |
|---|---:|---:|---:|---:|
| totaal | 586 | 325 (55 %) | 261 | 147 (56 % van dragend) |
| klasse A (d_th ≥ 0,72) | 41 (7 %) | | 17 (7 %) | 8 (5 %) |
| klasse C (0,30–0,72) | 339 (58 %) | | 136 (52 %) | 75 (51 %) |
| klasse B (< 0,30) | 206 (35 %) | | 108 (41 %) | 64 (44 %) |
| … B met effort behouden | 190 | | 99 | 59 |

Per nacht (ΔAHI-dragend → klassen; gevolg = SpO2-daling ≥ 3 % met nadir in [t0, t1 + 40 s] of arousal-onset in [t0, t1 + 15 s]):

| R | ΔAHI | alleen-druk | al hypopneu | dragend | dragend A / C / B | zonder gevolg | met desat / arousal / beide |
|---|---:|---:|---:|---:|---|---:|---|
| R02 | +5,9 | 85 | 41 | 44 | 8 / 12 / 24 | 39 | 3 / 2 / 0 |
| R04 | +1,2 | 7 | 0 | 7 | 0 / 4 / 3 | 5 | 2 / 0 / 0 |
| R07 | 0,0 | 0 | 0 | 0 | — | 0 | — |
| R11 | +8,7 | 68 | 27 | 41 | 1 / 21 / 19 | 31 | 4 / 6 / 0 |
| R18 | +37,3 | 426 | 257 | 169 | 8 / 99 / 62 | 72 | 56 / 20 / 21 |

- De dragende events zonder gevolg en mét gevolg zien er op de signalen **hetzelfde** uit:
  duur 12,8 tegen 14,3 s, d_th 0,34 tegen 0,36, drukdaling 0,76 tegen 0,74, effortdaling 0,38
  tegen 0,36 (medianen). Het gevolg onderscheidt dus geen signaalklasse; het is een
  onafhankelijke tweede as.
- Van de 586 alleen-druk-"apneus" was **meer dan de helft onder `breath` al een hypopneu**
  (AHI-neutrale herklassering). Het AHI-verschil zit in de 261 andere, en daarvan heeft 44 %
  wél een gevolg zonder dat de ademteug-gegradeerde hypopneedetector ze scoorde — een gat in
  de hypopneeroute van `breath` dat los staat van de sensorvraag (de strictness-0,30-arm van
  stap 2 raakt het).

## Lezing volgens de preregistratie
- Op de vooraf vastgelegde maat: A ≥ 2/3? **Nee** (7 %). C ≥ 2/3? **Nee** (52 %). B ≥ 1/3? **Ja** (41 %). → **gemengd**, met
  een forse B-component: op de nachten waar de thermistorpoort de thermistor goedkeurt, zijn
  de apneus die de neusdruk erbij zet voor 93 % géén thermistor-apneu-equivalent; bij vier
  op de tien ademt de thermistor gewoon door met behouden effort — het beeld van
  mondademhaling of een canule die niet alles ziet.
- Volgens de vooraf opgeschreven regel komt daarmee de **voorwaardelijke vereniging**
  (d_th ≥ 0,72 óf gevolg) in stap 2, en bij B ≥ 1/3 "bovendien een effortcriterium en een
  artefactvlag". **Afwijking, benoemd:** het effortcriterium als *bevestiging* was in de
  preregistratie verkeerd bedacht — behouden effort mét doorademende thermistor pleit juist
  tégen een apneu (mondademhaling), niet ervoor; het kan alleen B splitsen, niet bevestigen.
  Het wordt niet gebouwd. De artefactvlag (aantal alleen-druk-apneus met doorademende
  thermistor en behouden effort, per nacht in het rapport) is een rapportfunctie en staat
  op de lijst, niet in de scoring.
- **Wat de regel op deze nachten zou doen** (vervallen = dragend zonder gevolg én
  d_th < 0,72; ΔAHI minus vervallen / indexnoemer van de opname): R02 +5,9 → ≈ +1,3
  (33 vervallen, 7,24 u); R04 +1,2 → ≈ +0,2 (5; 5,01 u); R11 +8,7 → ≈ +2,5 (30; 4,83 u);
  R18 +37,3 → ≈ +23,0 (71; 4,97 u). De vereniging houdt dus op R18 nog ruim twintig per
  uur over tegenover `breath`, omdat 97 van de 169 dragende events een desaturatie of
  arousal hebben.
  Of die 97 juist zijn, zegt deze meting niet; de per-event-koppeling tegen NSRR in stap 2
  wel (op één MESA-smoke-nacht matchten gevolg-bevestigde alleen-druk-apneus in 89–91 % een
  NSRR-event, vrijwel altijd een hypopneu).

## Controle en grenzen
- **`both`-controle (150 events):** d_th mediaan 0,85; 75 % ≥ 0,72, maar slechts **39 % ≥ 0,90**
  terwijl de detector ze op ≥ 0,90 zette. De Hilbert-maat met de mediaan van de 60 s ervoor
  loopt dus 0,05–0,10 lager dan de eigen `flow_norm` van de detector, vooral in dichte
  clusters (R04: lange events van 19,6–90 s achter elkaar, mediaan d_th 0,34 op `both`). De
  klassegrenzen zijn **niet** verschoven (zelfde maat als het 0,72-dossier); het betekent dat
  klasse C ten dele A kan zijn, niet dat B kleiner is — d_th < 0,30 is ook met deze bias
  doorademen.
- **Post-hoc, buiten de regel — robuuste basislijn** (`diag2.py`, `analyse_diag2.py`): dezelfde
  omhullende, maar basislijn = 90e percentiel over [t0 − 120, t0) ∪ (t1, t1 + 120] in plaats
  van de mediaan van de 60 s ervoor. Op de `both`-controle klopt die maat wél met de detector:
  mediaan **0,96, 99 % ≥ 0,72, 95 % ≥ 0,90** (tegen 0,85 / 75 % / 39 % vooraf). De
  vooraf vastgelegde maat is dus in dichte clusters gecontamineerd: de "basislijn" vóór een
  event ligt zelf in het vorige event. Met de robuuste maat verschuift de verdeling van de
  261 ΔAHI-dragende events van A 17 / C 136 / B 108 naar **A 60 (23 %) / C 185 (71 %) /
  B 16 (6 %)**; 90 van de 108 "B"-events worden C en 41 van de 136 "C"-events worden A
  (overgang A→A 17, B→B 16, B→C 90, B→A 2, C→C 95, C→A 41). Zonder gevolg: A 28 / C 107 /
  B 12. Per nacht robuust A/C/B: R02 11/29/4, R04 2/4/1, R11 2/35/4, R18 45/117/7;
  "zou vervallen" wordt R02 31, R04 3, R11 29, R18 56 (R18 ΔAHI ≈ +26 i.p.v. +23).
  **Wat dat betekent — na de onafhankelijke verificatie voorzichtiger gesteld:** de
  verschuiving B → C is in hoofdzaak een **schaaleffect** (p90 ligt per constructie boven
  de mediaan, en d = 1 − ev/bl is juist in het B/C-bereik het gevoeligst voor de basislijn),
  geen contaminatie-effect: op de 65 dragende events met een schóón 60 s-venster gaat B
  net zo goed van 27 naar 6 (B→C 78 %) als op de gecontamineerde (85 %). Contaminatie
  bestaat (op de 44 events met ≥ 50 % overlap in het venster gaat d 0,20 → 0,60) maar
  draagt het beeld niet. De `both`-controle valideert de robuuste maat alleen waar de
  basislijn er weinig toe doet (ev ≈ 0) en lijkt op de detector omdat die zelf een hoog
  percentiel gebruikt (`compute_dynamic_baseline`, p95 over 300 s). **Conclusie:** hoe groot
  B is, hangt af van de schaal; "vier op de tien ademt door de mond" is dus geen robuuste
  bevinding, en op de robuuste schaal is het beeld **C-dominant** (zeven van de tien op
  hypopneu-niveau, bijna een kwart A). Wat op beide schalen staat: de alleen-druk-apneus
  zijn voor 93–94 % géén thermistor-apneu-equivalent volgens de grens die bij hun schaal
  hoort, en de druk overdrijft (het dossier van 22-08: thermistor 80 % waar de druk 90 %
  zakt). De vooraf vastgelegde lezing "gemengd" blijft de formele uitkomst; een drempel op
  de robuuste schaal moet opnieuw worden afgeleid, 0,72 hoort bij de mediaanschaal.
- **Gevolg voor de gebouwde regel (be78002):** `postprocess.envelope_drop` gebruikt de
  maat van het 0,72-dossier (mediaan 60 s ervoor), omdat 0,72 dáárop gekalibreerd is. Die
  maat ligt lager dan de detectorstatistiek en zal in de MESA-run een deel van de events
  die op de robuuste schaal A zijn als onbevestigd laten vervallen waar geen gevolg is. De
  run loopt zoals vooraf vastgelegd en wordt niet aangeraakt; de per-event-koppeling tegen
  NSRR laat zien of die events menselijke apneus zijn.
  Een robuuste maat vraagt een opnieuw afgeleide drempel (0,72 hoort bij de oude maat): dat
  wordt een aparte preregistratie ná de run, met de 140 nachten gesplitst in afleiding en
  validatie, want verse MESA-nachten zijn er niet meer.
- Alleen-thermistor-apneus op poort-af-nachten (12): drukdaling mediaan 0,06, 3 met gevolg —
  de regel laat er 9 vervallen; dat is de kleine kant van het verhaal (ΔAHI ≤ 1,2).
- Geen referentie; de klassen zijn signaallezingen. R07 heeft geen apneus op geen van beide
  sensoren en draagt niets bij.

## Gevolg
Stap 2 draait met de armen zoals vooraf vastgelegd (`breath`, `breath_dual`,
`breath_dual + thermistor_or_consequence`, elk op strictness 0,50 en 0,30, plus `rec`),
bibliotheek be78002 (opt-in gebouwd, default uit, golden onveranderd). Niets aan productie
gewijzigd.

## Verificatie
Onafhankelijk nagerekend (meting-verificatie, 07-10) uit R*_diag.json en R*_diag2.json:
prereg ongewijzigd na de start, klassen/definities/leesregel letterlijk, alle gepoolde
cijfers, de controle, de medianen en de post-hoc verdelingen kloppen. Verwerkt: de
per-nacht A/C/B-rijen van R02/R11/R18 (waren foutief uit de "zonder gevolg"-verdeling
afgeleid), de "zou vervallen"-benadering (nu rechtstreeks via de indexnoemer), het
SpO2-venster in de kop van de tweede tabel (40 s, niet 15 s), de R04-duren, en de
attributie van de post-hoc verschuiving (schaaleffect, geen contaminatie-artefact). R07
telt als poort-aan-nacht (usable 0,42) maar draagt niets bij. Niet verifieerbaar vanaf het
werkstation: container- en servertijden; `diag2.py` gebruikt Hilbert met `next_fast_len`-
padding en `diag.py` zonder (verschil verwaarloosbaar, niet gemeten).
