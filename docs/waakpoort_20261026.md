# Waakpoort-meting — uitslag (preregistratie 26-10-2026; meting 26-10 06:26–08:23)

Preregistratie `docs/waakpoort_preregistratie_20261026.md`, sha256 `347abcb05d5c87ba…` (in `run.log`
bij de start, ongewijzigd). Script `scripts/waakpoort_meting_mesa.py`, bibliotheek git 5e44caf
(`psgscoring/polygraphy.py` zoals gecommit vóór de meting, drempels uit stap 0 niet bijgesteld),
profiel `aasm_v3_breath_dual`, EEG/EOG/EMG-kanalen verwijderd zoals de `hpg`-arm van de
hPG-simulatie. 100 MESA-testnachten (`docs/waakpoort_20261026/ids_fresh100.txt`; vers voor de
vensterregel, zie prereg), 0 fouten, 8 workers, CPU max 61 °C, geen ingreep. Ruwe rijen buiten git
in `/srv/CODE/docs/waakpoort_20261026_raw/`; `rapport.md` en `samenvatting.json` hier.

**Uitslag in één zin.** Het analysevenster op signaalgeldigheid (arm **V**) haalt alle vier de
vooraf vastgelegde voorwaarden: tegen het orakel (arm **O**, zelfde kanalen met NSRR-hypnogram)
daalt het aantal nachten met ≥ 5 /u afwijking van **65 naar 30**, de mediane |ΔAHI| van **7,35 naar
2,40 /u** (gepaard p = 6·10⁻¹⁰), het verlies aan orakel-events is **0,48 %** (55 van 11 510; grens
0,5 %), en de ernstklasse klopt met het orakel op **78** in plaats van **52** van 100 nachten.
Volgens de prereg wordt V de polygrafieroute in psgscoring 0.37.0 / YASAFlaskified 0.39.0, met het
handmatige venster als override.

## Per arm

| arm | AHI med | F1 med (NSRR `aasm15`) | events | apneus | events met onset in NSRR-W | ernst = ref | \|Δ\| vs O med | \|Δ\| ≥ 5 | ernst = O | s/nacht |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T huidige route (alles N2, TIB) | 22,80 | 0,250 | 25 593 | 18 429 | 14 217 | 48 | 7,35 | **65** | 52 | 181 |
| **V venster** | 16,85 | 0,402 | 17 074 | 9 947 | 5 734 | 58 | **2,40** | **30** | **78** | 177 |
| O orakel (NSRR-hypnogram) | 15,45 | 0,491 | 11 510 | 6 839 | 0 | 62 | — | — | — | 171 |

ΔAHI tegen het orakel: T mediaan +5,8 /u (−33,7 … +34,5), V mediaan **0,0** (−34,7 … +13,7).
Ernstwissels tegen O: T 48 (licht→matig 14, normaal→matig 9, normaal→licht 7, matig→ernstig 6,
ernstig→matig 6, licht→ernstig 4), V 22 (normaal→licht 7, ernstig→matig 6, licht→matig 5,
matig→licht 3, normaal→matig 1). Vier nachten wijken onder V ≥ 5 /u af waar T dat niet deed.

**Wat het venster wegknipt.** Op 80 van 100 nachten sluit het iets uit (mediaan daar 95 min;
over alle nachten mediaan 78,5 min, vrijwel geheel ná het venster: 55,8 min mediaan, vóór 0,0,
binnen 0,0). Events met onset in een NSRR-W-epoch dalen van 14 217 naar 5 734 (−60 %); de
overblijvende 5 734 liggen op geldig signaal (inslapen, WASO) — precies de 41 % uit stap 0 waar
dit venster niet voor gebouwd is. Over-knippen: 71 NSRR-slaapminuten in totaal, > 5 min op 3
nachten (`mesa-sleep-0902` 24 min; `mesa-sleep-1127` 27 min; `mesa-sleep-4273` 6 min).

Per TIB/TST-tertiel (grenzen 1,47 / 1,83): T |Δ| 5,20 → V 2,10 (≥ 5: 17 → 5); 7,40 → 2,20 (22 → 10);
12,00 → 4,50 (26 → 15). Hoe langer de staart, hoe groter de winst én hoe groter de rest.

De drie slechtste nachten onder V:

| nacht | AHI O | AHI T | AHI V | uitgesloten min | slaap weggeknipt min | V-events in NSRR-W | TIB/TST |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mesa-sleep-1041` | 77,6 | 43,9 | 42,9 | 10 | 0 | 83 | 2,28 |
| `mesa-sleep-2239` | 61,9 | 46,6 | 46,6 | 0 | 0 | 72 | 1,49 |
| `mesa-sleep-2377` | 4,0 | 24,7 | 17,7 | 57 | 0 | 118 | 1,50 |

## Beslisregel (vooraf) — uitslag

| voorwaarde | eis | gemeten | |
|---|---|---|---|
| 1 | nachten ≥ 5 /u (V) ≤ ½ × (T) | 30 ≤ 32,5 | **JA** |
| 2 | mediaan \|Δ\| lager, gepaarde Wilcoxon p < 0,01 | 2,40 < 7,35; mediaan verschil 2,50; p = 6·10⁻¹⁰ | **JA** |
| 3 | orakel-events buiten V ≤ 0,5 % | 55 / 11 510 = 0,48 % | **JA** (krap) |
| 4 | ernst = O: V ≥ T + 10 | 78 ≥ 62 | **JA** |

Afstelset (n150, stap 0) voorspelde 92 → 38 van 150 (61 % → 25 %) en 6 van 18 642 verloren
slaapevents (0,03 %); de testset geeft 65 → 30 van 100 (65 % → 30 %) en 0,48 % verlies. De
richting en grootte repliceren; het verlies is op de testset hoger dan op de afstelset en zit
tegen de grens — zie kanttekeningen.

## Kanttekeningen

- **Voorwaarde 3 is krap gehaald** (0,48 % tegen 0,5 %). De 55 verloren orakel-events liggen op
  3 nachten met > 5 min weggeknipte slaap plus kleine randen; een hogere `min_dead_run` zou het
  verlies verkleinen maar is niet vooraf vastgelegd en wordt hier niet gekozen. Rapporteren, niet
  bijstellen.
- **De testnachten zijn vers voor de vensterregel, niet onaangeraakt** (MESA is op; zie prereg).
  Een replicatie op een onaangeraakt cohort vraagt SHHS1 (thermistor-flow, andere oximeter) en
  een aangepast harnas.
- **De resterende fout is waak met geldig signaal**: 5 734 events, 30 nachten ≥ 5 /u. Dat is het
  volgende project (slaap/waak-proxy: houding/beweging, hartslag, RIP-onrust), apart te
  pre-registreren; dit venster is er niet voor gebouwd en claimt het niet.
- **MESA-staarten zijn lang** (TIB/TST 1,61): de absolute winst op klinische hPG's hangt van hun
  staart af. Het mechanisme (SpO2 buiten bereik, flow vlak = sensor los) is recorder-onafhankelijk,
  de grootte niet gemeten op eigen opnames.
- Geen RERA/RDI (geen arousals zonder EEG), geen echte hPG-recorders, geen handmatig venster in
  deze meting (alleen de automatische arm).

## Gevolg volgens de preregistratie

V wordt de polygrafieroute: psgscoring 0.37.0 levert `polygraphy_hypnogram_from_raw`
(gecommit 5e44caf); YASAFlaskified bouwt het venster in `tasks.py` in plaats van "N2" × epochs, met
het handmatige venster (van/tot, minuten na start) als override en
`YASAFLASKIFIED_POLYGRAPHY_WINDOW=off` als terugweg (gecommit bd1984f, actief zodra de pin 0.37.0
is). PDF en resultatenpagina tonen het venster en de uitgesloten tijd. In productie geldt dit voor
elke polygrafie-upload vanaf de uitrol van YF 0.39.0; PSG's met hypnogram merken er niets van.

## Aanvulling (zelfde ruwe rijen, nadere decompositie)

- **Per nacht wint V op 58, is gelijk (binnen 0,5 /u) op 30 en verliest op 12.** De richting van
  de restfout verandert: T ligt systematisch te hoog (52 nachten ≥ +5 /u, 13 ≤ −5), V is
  gecentreerd (13 ≥ +5, 17 ≤ −5).
- **De noemer is nu de dominante restfout.** V haalt 60 % van de waak-events weg, maar zijn noemer
  ligt nog mediaan 2,62 u (IQR 1,97–3,57) boven de TST van het orakel (T: 3,88 u) — waak op
  geldig signaal telt mee als analysetijd. De 12 verliesnachten zijn nachten waar de oude route
  per ongeluk compenseerde: haar waak-apneus in de teller maskeerden de verdunning in de noemer.
  Duidelijkst `mesa-sleep-4366` (TIB/TST 3,1; 298 waak-epochs buiten V, 149 min uitgesloten):
  T −3,5 /u door compensatie, V −13,0 zonder. Een zichtbaar gemaakte fout, geen nieuwe.
- **Over-knippen is één mechanisme**: een sensor die tijdens slaap losraakt. `mesa-sleep-1127`
  verliest 27 slaapminuten en 33 orakel-events aan twee dode runs (4 en 23,5 min) ín het venster;
  `mesa-sleep-0902` 23,5 min en 8 events aan vier dode runs rond 3,5–4 u in de opname. Samen
  41 van de 55 verloren events (voorwaarde 3). Zonder SpO2 of flow kan de keten die minuten toch
  niet scoren; het orakel scoort ze op de kanalen die wél werkten.
- Uitgesloten tijd per reden over alle nachten: ná het venster op 78 nachten (mediaan 55,8 min,
  max 378), vóór het venster op 6 (max 110 min), dode runs binnen het venster op 24 (90e
  percentiel 33 min, max 193). Methode `signal_validity_window` op 100/100, handmatig venster
  nergens.
