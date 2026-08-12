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
| 2026-08-12 | begrenzen hoe vaak één desaturatie mag bevestigen | CAISR-resp | hard op 2 | `max_events_per_desaturation`, default `None` = huidig gedrag | `None`/2/3 op beide cohorten | *nog niet begonnen* |
| — | effortbanden als ventilatiebron | **geen CAISR-idee** — AASM-erkende alternatieve sensor, terug te voeren op de Chicago-criteria; psgscoring heeft al een fallback-pad | n.v.t. | eigen drempels, eigen sweep | alleen op een cohort met bruikbare effortbanden | *nog niet begonnen* |

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
