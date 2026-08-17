# Ideeën uit externe systemen — herkomst en meting

Dit is het auditspoor waar de docstrings van de betrokken profielvelden en §7.4
van de paper naar verwijzen. Eén rij per overgenomen idee.

**De regel.** CAISR-App staat onder CC BY-NC 4.0, psgscoring onder BSD-3.
Ideeën en parameterwaarden mogen over, code niet. Alles hieronder is
geïmplementeerd vanaf een specificatie, niet vanaf hun broncode. Geen
bestandsnaam, functienaam of kolomnaam uit dat project komt in deze codebase
voor.

**Wat bewust NIET is overgenomen** — en waarom dat het punt is. CAISR gebruikt
`drop_h = 0.41` op de ademtrace, `0.48` op de effortbanden, duren van 7–10 s en
aparte waarden voor titratiestudies. psgscoring houdt 30 % en 10 s als *centra
van gegradeerde termen*. Hun hyperparameters zijn met random search afgesteld;
de onze zijn de AASM-drempels zelf. Dat contrast ís het
traceerbaarheidsargument, dus die waarden overnemen zou het weggooien. Ook niet
overgenomen: hun basislijnconstructie (een ontwerp, geen parameter — het eigen
spoor is `baseline_mode="pre_event"`) en hun regel om te accepteren bij kapotte
oximetrie (filosofieverschil, hoort in de discussie).

---

| datum | idee | bron | waarde daar | waarde hier | meting | besluit |
|---|---|---|---|---|---|---|
| 2026-08-12 | koppelvenster tussen event-einde en arousal verruimen | CAISR-resp | 25 s | veld bestond al (`rule1b_arousal_window_s`, default 15 s); bereikte de gegradeerde tak niet | sweep 15/20/25/30 op PSG-IPA, `aasm_v3_breath`, precisie = mediaan over 12 scoorders | **NIET overgenomen — venster blijft 15 s** (de doorbedrading blijft) |
| 2026-08-12 | begrenzen hoe vaak één desaturatie mag bevestigen | CAISR-resp | hard op 2 | `max_events_per_desaturation`, default `None` = huidig gedrag | `None`/2/3 op MESA n=40 tegen `aasm15`, plus PSG-IPA n=5 | **NIET aangezet** — limiet 3 doet niets, limiet 2 kost 8 van 4303 events en maakt bias en F1 marginaal slechter |
| 2026-08-14 | lange events splitsen op fysiologische ankers | CAISR-resp (`split_long_events`) | > 60 s, splits op arousals of desaturaties | `split_events_longer_than_s`, default `None`; desaturatie-onsets eerst, arousals als terugval, ankers binnen 10 s van een rand genegeerd | duurverdeling MESA n=150 | **niet aangezet** — blijkt een ander anker voor de bestaande splitser `_split_long_region`; 0 van 9880 hypopneus haalt 60 s, alleen 593 apneus zijn kandidaat |
| 2026-08-14 | 2 %-desaturatie onder een lage baseline | CAISR-resp (`SpO2_drop_analysis`) | binair: 2 % telt als 3 % onder de 2-min-baseline | `desat_low_baseline_relaxation`, default `False`; **gegradeerd** — het sigmoidcentrum schuift 3 % → 2 %, breedte gelijk | vuurfrequentie op PSG-IPA | **niet aangezet** — de gespecificeerde conditie was vacuum waar (466/466); herzien naar 'lokale baseline < 92 %', vuurt nu op 1,8 %. **Regelafwijking**, nooit default aan |
| — | effortbanden als ventilatiebron | **geen CAISR-idee** — AASM-erkende alternatieve sensor, terug te voeren op de Chicago-criteria; psgscoring heeft al een fallback-pad | n.v.t. | eigen drempels, eigen sweep | alleen op een cohort met bruikbare effortbanden | *nog niet begonnen* |
| 2026-08-16 | enveloppe uit piekdetectie per ademteug in plaats van een continue enveloppe | CAISR-resp (piekdetectie + kubische interpolatie) | continu spoor per ademteug | `envelope_method="breath_amplitude"`, default `"hilbert"`; eigen implementatie vanaf de beschrijving, **lineaire** interpolatie — een kubische fit kan tussen een normale en een apneïsche ademteug onder nul schieten, en een negatieve enveloppe heeft tegen een basislijnratio geen betekenis | PSG-IPA n=5, mediaan over 12 scoorders, anker `aasm_v3_rec` in dezelfde run: bias +1,69 → **+3,19**, MAE 1,76 → **4,11**, F1 0,462 → **0,414**; events verloren aan de IoU-drempel 63 → **205**; absolute onsetfout van het 66e naar het 83e percentiel van de mens-tegen-mens-verdeling. SN1 geeft AHI 18,00 tegen een scoordermediaan van 5,96 (bereik 4,66–6,56) | **NIET aangezet, profiel `aasm_v3_env_breath` (exploratory)** — gemeten en afgewezen op TWEE cohorten. MESA n=150, gepaard: ΔF1 −0,0206, slechter op 96 van 150 opnames, **p = 2,4·10⁻⁶**, severity-overeenstemming 87 → 79. De voorspelde faalmodus (gemiste ademteugen bij lage amplitude, dus tijdens events) is precies wat de IoU-kolom laat zien: een derde van de extra events overlapt een menselijk event maar is anders afgebakend |
| — | drie overige enveloppevarianten (blokgewijze Hilbert, gelijkrichten+laagdoorlaat, decimeren) | **geen CAISR-idee** — standaard signaalverwerking (overlap-discard; AM-demodulatie, zie Oppenheim & Schafer) | n.v.t. | `envelope_method`, `envelope_fs`, alle drie default uit | blokgewijze Hilbert is **niet** gratis (interieurresidu ~1e-4 van de p95, convergeert niet met de pad). Op PSG-IPA n=5 blijft daar niets van over: AHI, eventaantal, F1 en élke grensstatistiek gelijk aan het anker; 116 van 8350 grenzen bewegen, allemaal één stap van 0,1 s. `rectify_lowpass` mat op PSG-IPA **beter** (bias +1,69 → +0,01, F1 +0,025) maar dat **repliceerde niet**: op MESA n=150 draait het teken om (bias −5,30 → −5,67, ΔF1 −0,0015, 61 beter tegen 79 slechter, p = 0,10) | **geen enkele aangezet**, profielen `aasm_v3_env_chunked` / `_rectify` / `_decimated`. De beslisregel stond vóór de meting vast en houdt stand: `rectify_lowpass` is een methodewijziging en n=5, dus een betere bias was een reden voor MESA n=150 en geen reden om te wisselen. MESA heeft dat bevestigd. `hilbert_chunked` haalt zijn criterium nu op beide cohorten (gelijk op 129/150 opnames, p=0,70) en is daarmee een kandidaat, maar promotie is een besluit over de gepubliceerde cijfers — niet over de code |

---

## Aantekeningen per rij

### Koppelvenster (rij 1)

Het profielveld bestond al maar was **halfaangesloten**: `RULE1B_AROUSAL_WINDOW_S`
bereikte alleen de Rule 1B-herstelpas, terwijl `score_hypopneas_breathwise` op
zijn eigen hardgecodeerde 15 s draaide. Dezelfde grootheid stond dus op twee
plekken en kon uiteenlopen zonder dat iets dat merkte.

Doorbedraden is gedaan vóór enige waardewijziging en is byte-identiek: het enige
profiel met een afwijkend venster is `mesa_shhs` (5 s), en dat draait de
envelope-detector, niet deze tak. Golden 8/8 ongewijzigd, 686 tests groen.

Motivatie voor de sweep: uit de v0.13.0-meting heeft slechts ~17 % van de
afgewezen kandidaten een arousal binnen 15 s. Als gemiste consensus-events een
arousal nét buiten het venster hebben, verklaart dat recall zonder dat de
detector iets mankeert.

**Beslisregel, vooraf vastgelegd:** de grootste waarde die de precisie niet
onder de huidige brengt. Het venster blijft eenzijdig (ná event-einde); de
richting verandert niet.

#### Uitkomst: 25 s draagt niet over

Gemeten 12 augustus 2026, `aasm_v3_breath`, PSG-IPA n=5,
`PSGSCORING_AROUSAL_DERIVATION=single`, psgscoring 0.16.0. Precisie/recall/F1
zijn per opname de mediaan over de twaalf scoorders (IoU ≥ 0,20, typeloos),
daarna gemiddeld over de vijf opnames.

| venster | precisie | recall | F1 | bias | MAE | in range | severity | hypopneus |
|---|---|---|---|---|---|---|---|---|
| **15 s** | **0,538** | 0,513 | **0,519** | −0,29 | 0,52 | 5/5 | 5/5 | 150 |
| 20 s | 0,533 | 0,513 | 0,518 | −0,15 | 0,39 | 5/5 | 5/5 | 154 |
| 25 s | 0,515 | 0,513 | 0,509 | +0,07 | 0,30 | 5/5 | 5/5 | 161 |
| 30 s | 0,519 | 0,516 | 0,511 | +0,13 | 0,28 | 5/5 | 5/5 | 163 |

Apneus onveranderd (316 in alle vier de standen), zoals verwacht: het venster
raakt alleen de hypopneutak.

**Elke verruiming kost precisie, dus de regel wijst 15 s aan.** Het argument is
sterker dan de regel alleen. **Recall staat stil** — 0,513 / 0,513 / 0,513 /
0,516 — terwijl er dertien hypopneus bijkomen. Die events matchen dus bij geen
enkele van de twaalf scoorders.

Dat zet de AHI-kolommen in hun juiste licht: bias en MAE verbeteren monotoon
(−0,29 → +0,13 en 0,52 → 0,28) omdat de toegevoegde events een negatieve bias
opheffen, niet omdat de detectie beter wordt. Het venster koopt een beter
AHI-getal met events die geen mens gescoord heeft. Precies daarom stond de
beslisregel vóór de sweep vast: op MAE gekozen was 30 s eruit gekomen.

De hypothese die de sweep motiveerde — gemiste consensus-events hebben een
arousal nét buiten het venster — is hiermee **weerlegd** voor dit cohort. De
gemiste events liggen niet net buiten 15 s.

De doorbedrading zelf blijft staan: die repareerde een half aangesloten veld en
is byte-identiek. Alleen de waarde 25 s wordt niet overgenomen. Meting:
`docs/venster_psgipa_20260812.{json,log}`,
`scripts/sweep_arousal_window_psgipa.py`.

### Desaturatiehergebruik (rij 2)

`max_events_per_desaturation` groepeert de op desaturatie bevestigde hypopneus
per werkelijke desaturatie-episode (`detect_desaturations`), houdt er
`max_events` met de hoogste `p_scored` en degradeert de rest naar
`rejected_hypopneas` met `reject_reason="desat_reuse_limit"`.

**Degraderen, niet verwijderen** — de kandidaten blijven beschikbaar voor de
ML-promotie en de visuele controle, en de ingreep blijft telbaar en
omkeerbaar.

#### PSG-IPA: het probleem bestaat hier niet

`aasm_v3_breath`, 12 augustus 2026, `PSGSCORING_AROUSAL_DERIVATION=single`,
psgscoring 0.16.0, gemeten bij limiet 1 (de scherpst mogelijke):

| opname | hypopneus | desaturaties | aan een desat gekoppeld | groepen | grootste groep | gedegradeerd |
|---|---|---|---|---|---|---|
| SN1 | 16 | 48 | 8 | 8 | 1 | 0 |
| SN2 | 21 | 80 | 9 | 9 | 1 | 0 |
| SN3 | 42 | 322 | 28 | 25 | **2** | 3 |
| SN4 | 18 | 61 | 4 | 4 | 1 | 0 |
| SN5 | 50 | 135 | 22 | 22 | 1 | 0 |
| **totaal** | **147** | 646 | **71** | 68 | **2** | **3** |

**Geen enkele groep haalt drie, dus CAISR's harde limiet van 2 degradeert op
PSG-IPA precies nul events.** Zelfs limiet 1 raakt drie van de 147 hypopneus,
alle drie op SN3.

Dat dit een échte nul is en geen niet-aangesloten koppeling, is af te lezen aan
`n_events_grouped`: 71 van de 147 hypopneus (48 %) zijn wel degelijk aan een
desaturatie toegewezen. De overige 52 % is op een arousal bevestigd en valt
buiten deze regel — zoals bedoeld. De statistiek draagt daarom `n_groups`,
`n_events_grouped` en `max_group_size` náást `n_degraded`; zonder die drie is
"nul gedegradeerd" niet te onderscheiden van "er is nooit iets gegroepeerd".

#### MESA: hergebruik bestaat wél, maar verandert het besluit niet

`aasm_v3_breath`, n=30 (30 van 30 bruikbaar), staging uit de NSRR-annotatie:

| grootste groep op één desaturatie | opnames |
|---|---|
| 1 event | 9 |
| 2 events | 15 |
| 3 events | 5 |
| 4 events | 1 |

2704 hypopneus, 4129 desaturaties, 1523 (56,3 %) aan een desaturatie
gekoppeld. Een limiet van 2 zou op **6 van de 30** opnames iets doen, een
limiet van 3 op 1.

Anders dan op PSG-IPA bestaat het hergebruik hier dus. Toch blijft de default
`None`, en het argument daarvoor hangt niet aan deze cijfers — het stond al
vast vóór de meting:

> De limiter kan uitsluitend events WEGNEMEN. Op MESA is de AHI-bias −11 tot
> −15/u, dus daar wordt al te weinig geteld; op PSG-IPA valt de AHI op 5 van
> de 5 opnames binnen de scoorderrange. Er is op geen van beide cohorten een
> tekort aan strengheid dat deze knop zou verhelpen.

**Eerlijkheidshalve:** voor blok 2B is, anders dan voor 2A, géén beslisregel
vooraf vastgelegd. Elke regel die ik nú op deze getallen zou formuleren is
post-hoc. Daarom rust het besluit op het bovenstaande argument — dat wél vooraf
stond — en niet op de gemeten verdeling.

**Wat dit besluit kan omgooien:** de gerepareerde RIP-poort verhoogt `ahi_total`
op MESA met gemiddeld +4,75/u, waardoor de onderdetectie kleiner wordt en het
argument "er wordt al te weinig geteld" verzwakt. Gaat die poort default aan,
dan hoort 2B opnieuw gemeten — mét een vooraf vastgelegde beslisregel. Zie
`docs/rip_poort_reparatie_20260812.md`.

Meting: `docs/desat_limit_mesa_final.{json,log}` (MESA n=30),
`scripts/sweep_desat_limit_mesa.py`.

*Twee eerdere MESA-runs (`desat_limit_mesa_20260812`, `_20260813`) verloren 16
respectievelijk 14 opnames aan een `AttributeError`: ik bewerkte `profiles.py`
terwijl de run ertegen liep, zodat workers van vóór en ná de edit naast elkaar
draaiden. De overgebleven opnames vormden een steekproef bepaald door timing.
Ze zijn vervangen, niet gerapporteerd.*
