# hPG-simulatie op MESA — uitslag (preregistratie 25-10-2026; meting nacht 25/26-10)

Preregistratie `docs/hpg_simulatie_preregistratie_20261025.md`, sha256 `5468abd180e23405…` (in
`run.log` bij de start, ongewijzigd). Script `scripts/hpg_simulatie_mesa.py`, profiel
`aasm_v3_breath_dual`, psgscoring 0.36.2-werkboom (git 58c3e06 + de ongecommitte wijzigingen van
25-10: `unet_v1` als standaarddetector, agreement, hr_surge; de bibliotheek is tijdens de meting
niet bewerkt). 150 nachten van de standaard-n150-validatieset, 0 fouten, drie workers onder
`CPUQuota=300 %` naast de U-Net-schaalketen, 25-10 20:44:32 → 26-10 04:54:01 (8 u 10 min;
24,4 rekenuren, mediaan 578 s per nacht voor drie armen). Bewakers: CPU max 63 °C, 0
overschrijdingen, MemAvailable min 41 GB, geen ingreep. Uitvoer in `docs/hpg_simulatie_20261025/`:
`rapport.md`, `samenvatting.json`, `rijen.json` (9,8 MB, met eventlijsten), `run.log`,
`thermal_guard.log`, `memguard.log`; post hoc `decompositie.py` → `decompositie.json`.
De psg-arm draaide op alle 150 nachten `unet_v1` zonder terugval.

## Uitslag in drie zinnen

1. **Zonder EEG/EOG/EMG maar mét hypnogram** (`hpg_hypno`) verliest de scoring precies de
   hypopneeën die op een arousal steunen: 2 709 van 10 213 (26,5 %; 2 605 ervan waren in de
   psg-arm via Rule 1A arousal-bevestigd), apneus identiek (11 288). De AHI daalt op **150 van
   150** nachten, mediaan **−2,55 /u** (gemiddeld −3,04; bereik −12,2 … −0,2), op 23 nachten
   ≥ 5 /u; **27 ernstwissels**, alle naar beneden (17 matig → licht, 7 licht → normaal, 3
   ernstig → matig), 24 ervan in het lage en middelste tertiel; F1 −0,014 (39/109).
2. **De polygrafieroute** (`hpg`: zelfde kanalen, hypnogram = alles N2, registratietijd als
   noemer — exact `tasks.py` van YASAFlaskified) is niet "iets lager" maar **per nacht
   onbetrouwbaar**: ΔAHI mediaan +2,40 /u maar bereik **−29,3 … +33,1**, 56 nachten lager / 91
   hoger, **85 van 150 ≥ 5 /u af**, 66 ernstwissels in beide richtingen, ernst gelijk aan de
   referentie 94 → **68** van 150, F1 0,562 → **0,310** (precisie 0,24), apneus 11 288 →
   **30 628**.
3. **De oorzaak is geen detectorfout maar de ontbrekende waak/slaap-poort.** 56 % van de
   hpg-events (23 604 van 42 246) begint in een W-epoch van het NSRR-hypnogram: 19 137 apneus en
   4 467 hypopneeën. Twee derde daarvan (15 334) ligt **ná het laatste slaapepoch**, vrijwel
   allemaal apneus (14 643, mediane duur 40 s): de recorder loopt door nadat de slaap — en vaak
   de sensor — voorbij is. Mijn lezing vooraf ("`hpg` verliest door de noemer, ΔAHI ≈
   −(1 − TST/TIB) × AHI") was de helft van het verhaal en **verkeerd van teken**: het
   noemereffect is er (mediaan −10,7 /u) maar wordt overstemd door de waak-events (+14,7 /u).

## Per arm (uit `rapport.md`, referentie `aasm15`, matcher `LEGACY_MATCHER`)

| arm | F1 med | F1 gem | prec | rec | AHI-bias | MAE | r | ernst = ref | hypopneeën | via Rule 1A | apneus | detector | s/nacht |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `psg` | 0,562 | 0,534 | 0,601 | 0,608 | +0,09 | 8,07 | 0,782 | 94/150 | 10 213 | 2 605 | 11 288 | unet_v1 | 210 |
| `hpg_hypno` | 0,547 | 0,513 | 0,630 | 0,526 | −2,95 | 8,47 | 0,784 | 93/150 | 7 504 | 0 | 11 288 | geen | 171 |
| `hpg` | 0,310 | 0,329 | 0,239 | 0,534 | +2,51 | 11,33 | 0,630 | 68/150 | 11 618 | 0 | 30 628 | geen | 183 |

| gepaard t.o.v. `psg` | ΔAHI med | gem | min … max | lager/hoger | \|Δ\| ≥ 5 | Wilcoxon p | ΔF1 med | beter/slechter | ernstwissels |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| `hpg_hypno` | −2,55 | −3,04 | −12,2 … −0,2 | 150/0 | 23 | < 10⁻⁴ | −0,014 | 39/109 | 27 (matig→licht 17, licht→normaal 7, ernstig→matig 3) |
| `hpg` | +2,40 | +2,42 | −29,3 … +33,1 | 56/91 | 85 | 0,0009 | −0,208 | 1/147 | 66 (matig→ernstig 20, licht→matig 19, normaal→licht 8, matig→licht 7, ernstig→matig 6, licht→ernstig 2, licht→normaal 2, normaal→ernstig 1, normaal→matig 1) |

Per tertiel van de referentie-AHI (grenzen 13,3 / 28,5 /u; n = 50 per tertiel):

| arm × tertiel | ΔAHI med | ΔAHI gem | ΔF1 med | ernstwissels |
|---|---:|---:|---:|---:|
| `hpg_hypno` laag / midden / hoog | −2,50 / −3,20 / −2,30 | −2,80 / −3,72 / −2,60 | −0,014 / −0,017 / −0,013 | **12 / 12** / 3 |
| `hpg` laag / midden / hoog | +4,40 / +3,25 / −2,15 | +5,91 / +3,30 / −1,95 | −0,231 / −0,236 / −0,183 | 30 / 24 / 12 |

TIB/TST mediaan 1,73 (p10–p90 1,39–2,48); aandeel hypopneeën zonder ≥ 3 % desaturatie in de
psg-arm 26,5 % (de vooraf benoemde bovengrens van wat een hPG kán verliezen — en vrijwel exact
wat `hpg_hypno` verliest).

## Decompositie van de `hpg`-arm (post hoc, beschrijvend — `decompositie.py`)

- **Waar de events liggen** (onset-epoch van het NSRR-hypnogram): slaap 18 642 (11 491 apneus,
  7 151 hypopneeën); waak 23 604 (19 137 apneus, 4 467 hypopneeën). Van de waak-events: 2 904
  vóór het eerste slaapepoch, 5 366 in WASO, **15 334 na het laatste slaapepoch** (14 643
  apneus, mediane duur 40 s; slaapevents mediaan 19,5 s). Per nacht mediaan 56 % van de events in
  waak; op 91 van 150 nachten is het de meerderheid.
- **Waakuren**: vóór slaapbegin mediaan 0,88 u, WASO 1,60 u, na het laatste slaapepoch 1,52 u;
  eventfrequentie in waak 36,4 /u tegen 21,9 /u in slaap (psg-AHI).
- **ΔAHI(hpg − psg) = noemereffect + waak-events/TIB**: mediaan −10,7 /u + 14,7 /u = +2,4 /u.
  Twee grote, tegengestelde termen die in de mediaan bijna wegvallen en per nacht niet.
- **Tegenproef**: hpg-slaapevents / TST tegen de AHI van `hpg_hypno` (zelfde kanalen, wél
  hypnogram): mediaan −0,09 /u, −1 event per nacht (Wilcoxon p = 0,03; klein en systematisch —
  de epoch-afhankelijke stappen zoals de slaappoort op eventgrenzen). `hpg` is dus `hpg_hypno`
  plus waak-events plus noemer; de detector zelf doet in slaap hetzelfde.
- **Per TIB/TST-tertiel** (grenzen 1,57 / 1,90): ΔAHI mediaan +0,9 / +3,35 / +3,7 /u; |Δ| ≥ 5
  op 19 / 34 / 32 nachten. Hoe langer de staart, hoe erger — maar ook het laagste tertiel heeft
  op 19 van 50 nachten ≥ 5 /u af.

## Lezing tegen de preregistratie

- **`hpg_hypno`**: "kan alleen verliezen" → klopt, 150/150, nergens ΔAHI > 0. De grootte
  (−2,55 /u, 24 van de 27 ernstwissels in het lage en middelste tertiel = 24 % van die 100
  nachten) valt onder de vooraf geschreven regel "een verlies dat ernstklassen verschuift in het
  lage en middelste tertiel maakt [de arousalpoot op RIP/Pleth] nodig". Dat voedt
  roadmap-kandidaat C stap 2 positief; of hij er komt is Barts beslissing.
- **`hpg`**: de prereg verwachtte een daling door de noemer; gemeten is een stijging met enorme
  spreiding. De fout zit in mijn lezing vooraf (de waak-events niet voorzien), niet in de meting
  of de vastgelegde rapportage.
- **Wat dit voor YASAFlaskified betekent.** De `hpg`-arm ís de polygrafieroute van de app
  (`tasks.py`: geen EEG → hypnogram `["N2"] × epochs`, REI over de registratietijd). Elke
  polygrafie-upload wordt dus van de eerste tot de laatste seconde gescoord zonder enig
  waakfilter; de REI die eruit komt hangt af van hoe lang de recorder vóór en na de slaap
  doorliep. Op MESA (thuis-PSG met lange staart, TIB/TST 1,73) geeft dat op 85 van 150 nachten
  een afwijking ≥ 5 /u; de grootte op klinische hPG's hangt van hun staart af en is **niet
  gemeten** — de richting en het mechanisme wel. De 44,5 %-bevinding uit de literatuurnotitie
  (hPG-autoscoring zonder nalezing faalt) past hierbij: de nalezing is in de praktijk ook het
  afknippen van de staart.

## Wat NIET uit deze meting volgt, en wat erna kan

- Niet gemeten: echte hPG-opnames (andere sensorset, PAT/pleth-slaapschatting, patiënt zet de
  recorder zelf aan/uit), RERA/RDI (geen arousals in de hPG-armen), het effect van een
  handmatige keuze van het analysevenster, de eigen PSG's R01–R20 (Hetzner, autorisatie nodig).
- Twee vervolgen, geen van beide gebouwd (bibliotheek en app ongewijzigd):
  1. **Een analysevenster/waakpoort voor de polygrafieroute** — begin en einde van de bruikbare
     opname op signaalgeldigheid (flow/RIP/SpO2 aanwezig en niet vlak) en bewegings-/houdings-
     kanaal, met WASO-schatting (hartslag/actigrafie/RIP-onrust) als tweede stap. Zou het
     grootste deel van de 23 604 waak-events wegnemen (15 334 liggen in de staart). Preregistratie
     met de n150 als afstelset en verse MESA-nachten als toets; eigen hPG's als externe toets.
  2. **De arousalpoot op RIP/Pleth** (kandidaat C stap 2) tegen de −2,55 /u van `hpg_hypno`
     (HR-stijging droeg de winst in de pleth-re-ranker van 0.33.0).
- Registratie: de n150 was al geregistreerd en is als hergebruik vermeld in de sectie van 25-10
  in `/srv/DATA/MESA/gebruikte_mesa_ids.txt`. `rijen.json` en `rijen.partial.jsonl` dragen
  volledige eventlijsten (16 MB samen); of die in git gaan of naast de U-Net-uitvoer buiten git,
  is een keuze bij de commit.

## Nameting 26-10 (ochtend): hartslagstijging als arousalproxy — NIET bouwen

Preregistratie `docs/hpg_arousalpoot_hr_surge_preregistratie_20261026.md` (sha256 `24eeed94…`,
geschreven vóór de berekening), script `hpg_simulatie_20261025/hr_surge_proxy.py`, uitvoer
`hr_surge_proxy.{json,md}`. ΔHR per event (Huang-definitie: piek in [onset, einde + 15 s] min het
gemiddelde van de 60 s ervoor) op het MESA-oximeterkanaal `HR` (1 Hz, 150/150 nachten), klassen uit
de `psg`-arm, controlevensters in slaap zonder event binnen ± 30 s.

| klasse | n | mediaan | p25 | p75 | ≥ 5 bpm | ≥ 7 | ≥ 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A hypopneu ≥ 3 % desaturatie | 7 499 | 3,8 | 1,7 | 6,8 | 0,38 | 0,24 | 0,12 |
| B hypopneu < 3 % (arousal-gedragen; 2 603 van 2 707 via Rule 1A) | 2 693 | 3,7 | 1,8 | 6,9 | 0,38 | 0,25 | 0,12 |
| C controle (slaap, geen event) | 2 703 | 2,1 | 1,1 | 3,7 | 0,16 | 0,09 | 0,03 |
| D apneu | 11 276 | 3,7 | 1,4 | 7,2 | 0,38 | 0,26 | 0,14 |

AUC B tegen C **0,648** (A tegen C 0,637; D tegen C 0,623; B tegen A 0,506). Per nacht (141 met
≥ 3 per klasse) ligt de mediaan van B 1,5 bpm boven die van C, hoger op 126/141, p = 2·10⁻²⁰:
het effect is echt en systematisch, maar klein ten opzichte van de spreiding. **Volgens de
vooraf vastgelegde lezing (AUC < 0,70): de poot kan de −2,55 /u niet terughalen zonder veel
bijvangst — niet bouwen.** Twee kanttekeningen: arousal-gedragen hypopneeën hebben precies
dezelfde autonome respons als desaturatie-hypopneeën (AUC 0,506), dus de maat ziet het event, niet
de bevestigingsroute; en MESA-`HR` is een gefilterde oximeterafleiding op 1 Hz — een beat-to-beat
pulsfrequentie uit het pleth-signaal (zoals de re-ranker van 0.33.0 gebruikte) kan scherper zijn.
Dat is een nieuwe meting met een nieuwe prereg, geen reden om deze uitslag te herzien.

