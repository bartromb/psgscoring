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
| 2026-08-12 | koppelvenster tussen event-einde en arousal verruimen | CAISR-resp | 25 s | veld bestond al (`rule1b_arousal_window_s`, default 15 s); bereikte de gegradeerde tak niet | sweep 15/20/25/30 op PSG-IPA, daarna MESA tegen `aasm15` | *in uitvoering* |
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
