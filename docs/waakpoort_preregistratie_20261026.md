# Preregistratie — analysevenster op signaalgeldigheid voor de polygrafieroute (waakpoort, stap 1)

Datum: 2026-10-26, ochtend. **Geschreven vóór de bouw en vóór de meting op verse nachten.**
Aanleiding: de hPG-simulatie (`docs/hpg_simulatie_20261026.md`) — de polygrafieroute van
YASAFlaskified (geen EEG → hypnogram alles "N2", registratietijd als noemer) scoort 56 % van haar
events in waak, twee derde daarvan ná het laatste slaapepoch. Stap 0
(`docs/hpg_simulatie_20261025/waakpoort_stap0.py`, post hoc op de n150) zegt: 59 % van die
waak-events ligt op los signaal (SpO2 buiten bereik 13 769 van 23 604, flow vlak 5 695), 41 % op
geldig signaal. Deze stap bouwt alleen het eerste: een analysevenster op signaalgeldigheid. Een
slaap/waak-proxy voor de overige 41 % is een apart project.

## Wat gebouwd wordt (vastgelegd vóór de meting)

- **Bibliotheek** `psgscoring/polygraphy.py` (nieuw):
  - `epoch_validity(signalen, sf, n_epochs)` → per 30 s-epoch: `flow_dead` (Pres én Therm — of
    het enige aanwezige flowkanaal — met IQR ≤ 2 % van de **mediane epoch-IQR van de hele
    opname** van dat kanaal, of exact 0; een kanaal waarvan die mediaan 0 is, telt niet mee),
    `spo2_invalid` (epochmediaan buiten 50–100 % of < 50 % van de samples in dat bereik),
    `effort_dead` (informatief, zelfde regel op Thor/Abdo), `sensor_off = flow_dead ∨
    spo2_invalid`. In productie is er geen hypnogram, vandaar de mediaan over alle epochs (stap 0
    gebruikte de slaapepochs; op de n150 verschilt dat weinig omdat > 50 % van de epochs geldig is).
  - `analysis_window(sensor_off, min_valid_run=4, min_dead_run=6)` → regel B uit stap 0: het
    venster loopt van het begin van de eerste tot het einde van de laatste run van ≥ 4 geldige
    epochs (2 min); binnen het venster vallen runs van ≥ 6 opeenvolgende `sensor_off`-epochs
    (3 min) eruit (een centrale apneu haalt nooit 3 min). Uitvoer: per epoch `analysed`, plus
    provenance (vensterbegin/-einde in s, uitgesloten minuten per reden, drempels, kanalen).
  - `polygraphy_hypnogram(raw_of_signalen, ..., manual_window_s=None)` → lijst met "N2" voor
    geanalyseerde epochs en "W" voor uitgesloten epochs; met `manual_window_s=(t0, t1)` (lights
    off/on van de technoloog) vallen epochs buiten dat venster óók op "W". De bestaande
    slaappoort en TST-noemer van de keten doen de rest: geen tweede poortmechanisme.
  - Drempels 2 %, 4 epochs, 6 epochs, 50–100 %, 50 % zijn die van stap 0 en worden **niet** op de
    verse set bijgesteld.
- **YASAFlaskified**: de polygrafieroute in `tasks.py` gebruikt `polygraphy_hypnogram` in plaats
  van "N2" × epochs; twee optionele velden op het uploadformulier (analyse van / tot, hh:mm);
  PDF en resultatenpagina tonen het analysevenster, de uitgesloten tijd per reden en de REI over
  de geanalyseerde tijd. Rollback: de velden leeg én `YASAFLASKIFIED_POLYGRAPHY_WINDOW=off` geeft
  de oude route.

## Meting

- **Testset**: 100 MESA-nachten (seed 20261026, `docs/waakpoort_20261026/ids_fresh100.txt`),
  geregistreerd vóór de meting. **MESA heeft geen onaangeraakte nachten meer** (2 055 van 2 055
  staan in het register sinds de U-Net-schaaltraining van 25-10); gekozen is daarom uit de
  trainingspool van die schaalmeting, met uitsluiting van elke nacht uit de n150, de 140 van 07-10,
  de 40 kandidaatpoort-nachten, MESA-76, de validatie-40 en de hPG-simulatie — dus **vers voor de
  vensterregel** (die alleen op de n150 is afgesteld) en voor elke meting van oktober, maar
  eerder aangeraakt door arousal-training en door respiratoire campagnes van juni–september. Een
  echt onaangeraakte replicatie vraagt SHHS1 (≈ 4 200 nachten over) en een aangepast harnas; dat
  staat buiten deze meting. De n150 is de **afstelset** van stap 0 en wordt apart en als zodanig
  gerapporteerd.
- **Armen per nacht** (`aasm_v3_breath_dual`, EEG/EOG/EMG-kanalen verwijderd zoals de `hpg`-arm):
  **T** = huidige polygrafieroute (alles "N2", registratietijd); **V** = `polygraphy_hypnogram`
  (automatisch venster, geen handmatig venster); **O** = orakel: zelfde kanalen met het
  NSRR-hypnogram (= `hpg_hypno`, het beste wat zonder EEG haalbaar is). Script
  `scripts/waakpoort_meting_mesa.py`, hergebruik van `hpg_simulatie_mesa.py`.
- **Beslisregel (vooraf), alle vier op de 100 verse nachten:**
  1. nachten met |AHI_V − AHI_O| ≥ 5 /u ≤ **de helft** van de nachten met |AHI_T − AHI_O| ≥ 5 /u;
  2. mediaan |AHI_V − AHI_O| < mediaan |AHI_T − AHI_O|, gepaarde Wilcoxon op |Δ| p < 0,01;
  3. verlies van orakel-events: O-events met onset in een epoch dat V uitsluit ≤ 0,5 % van alle
     O-events;
  4. ernstklasse gelijk aan O: V ≥ T + 10 nachten.
  Verwachting uit de afstelset (n150, regel B): |Δ| ≥ 5 /u 92 → 38; mediaan |Δ| 6,88 → 2,84 /u;
  verlies 6 van 18 642; ernst gelijk aan orakel 79 → 114 van 150.
- **Rapportage daarnaast**: event-F1 tegen NSRR `aasm15` voor T/V/O; aantal events met onset in
  een NSRR-W-epoch T tegen V; uitgesloten minuten per reden; per tertiel van TIB/TST; nachten
  waarop het venster korter is dan de NSRR-slaap (over-knippen) met de verloren slaapminuten.
- **Gevolg bij slagen**: V wordt de polygrafieroute in 0.37.0 / YF 0.39.0, met het handmatige
  venster als override. **Bij falen**: alleen het handmatige venster gaat mee; het automatische
  venster blijft uit (vlag) en de uitslag gaat in CHANGELOG en notitie.

## Wat NIET uit deze meting volgt

Waak met geldig signaal (vóór slaapbegin, WASO; 41 % van de waak-events op de n150) — daarvoor is
een slaap/waak-proxy nodig (actigrafie/houding, hartslag, RIP-onrust), apart te pre-registreren.
Geen uitspraak over echte hPG-recorders (andere sensorset, patiënt start/stopt zelf), noch over
RERA/RDI. De grootte op klinische hPG's hangt van hun staart af; het mechanisme niet.
