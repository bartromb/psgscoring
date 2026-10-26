# Preregistratie — hPG-simulatie op MESA: wat verliest de scoring zonder EEG, EOG en kin-EMG?

Datum: 2026-10-25, geschreven vóór de meting. Beschrijvende verschilmeting, geen beslisregel
(zelfde soort als `docs/unet_doorwerking_preregistratie_20261009.md`). Aanleiding: de
slaapapneuconventie 2027 stuurt een deel van de verwijzingen naar thuispolygrafie (hPG), en de
onafhankelijke literatuur (`docs/literatuur_update_20261025.md` §3: Roncero 2025 concordantie
44,5 %, Khadadah 2026) zegt dat ingebouwde hPG-autoscoring rond de klinische drempels faalt.
Vraag 1 van roadmap-kandidaat C: hoeveel van de AHI gaat verloren als psgscoring op een
hPG-kanaalset draait, en bij wie verschuift de ernstklasse. De eigen 20 PSG's (R01–R20) staan
op de productieserver en vragen een aparte autorisatie; deze meting draait lokaal op MESA.

## Opzet

- **Nachten:** de standaard n150-validatieset (seed 20260801; registersectie "orakel-decompositie
  Rule-1A-tak 2026-09-17 — standaard n150-validatieset"), hergebruik, niets wordt afgesteld.
  Alle 150; bij tijdnood een seeded steekproef (`--n`, seed 20261025) — vooraf: alle 150.
- **Profiel:** `aasm_v3_breath_dual` (productiestandaard sinds 07-10-2026), bibliotheek zoals
  ze op het moment van de meting in `main` staat (werkboom na de wijzigingen van 25-10:
  `arousal_detector` default `unet_v1`; versie en git-stand in het logboek). De bibliotheek
  wordt tijdens de meting niet bewerkt.
- **Drie armen per nacht** (`scripts/hpg_simulatie_mesa.py`):
  1. `psg` — volledige kanaalset, NSRR-hypnogram (zoals `validate_mesa.py`).
  2. `hpg_hypno` — alle kanalen waarvan de naam met EEG, EOG of EMG begint verwijderd
     (ook `*_Off`), NSRR-hypnogram. Isoleert de arousalpoot (noemer gelijk aan `psg`).
  3. `hpg` — zelfde kanaalset als 2, hypnogram = alle epochs "N2" (registratietijd als
     noemer), exact de polygrafieroute van YASAFlaskified (`tasks.py`: "Polygrafie — geen
     slaapstaging", REI).
- **Referentie:** NSRR-events gereconstrueerd door `validate_mesa.parse_nsrr`; primair `aasm15`
  (3 % óf arousal, alle apneus), daarnaast `desat3_all`, `oahi3`, `oahi4`. Matcher
  `LEGACY_MATCHER` (validate_psgipa). Ernstbanden 5/15/30.

## Wat gerapporteerd wordt (vooraf vastgelegd)

Per arm: mediane/gemiddelde event-F1, precisie, recall, AHI-bias, MAE, r, ernstklasse gelijk aan
de referentie, sommen van hypopneeën (totaal, met ≥ 3 % desaturatie, via Rule 1A-arousal) en
apneus, welke arousaldetector liep (verwacht: `unet_v1` op `psg`, geen arousalstap op de
hPG-armen). Gepaard t.o.v. `psg`: ΔAHI (mediaan, gemiddelde, bereik, lager/hoger, |Δ| ≥ 5 /u,
Wilcoxon), ΔF1 (mediaan, beter/slechter, Wilcoxon), ernstwissels met richting, per tertiel van
de referentie-AHI. Daarnaast de TIB/TST-verhouding en het aandeel hypopneeën zonder ≥ 3 %
desaturatie in de `psg`-arm (de bovengrens van wat een hPG kán verliezen).

## Lezing vooraf

- `hpg_hypno` kan alleen verliezen (hypopneeën met alleen een arousal als gevolg), tenzij de
  afwezigheid van arousals elders iets kantelt; een ΔAHI > 0 zou een boekhoudkundige bijwerking
  zijn en wordt dan als zodanig uitgezocht.
- `hpg` verliest daarbovenop door de noemer (TIB > TST): verwacht ΔAHI ≈ −(1 − TST/TIB) × AHI.
- De klinische lezing gaat over de ernstwissels rond 5/15/30 per tertiel, niet over de mediaan:
  Pitkänen 2025 vond 22 % van de AHI's binnen ±1 /u van een drempel.
- Uitkomst voedt de beslissing over roadmap-kandidaat C stap 2 (arousalpoot op RIP/Pleth): een
  klein verlies maakt die poot overbodig; een verlies dat ernstklassen verschuift in het lage
  en middelste tertiel maakt hem nodig.

## Registratie

De 150 id's staan al in `/srv/DATA/MESA/gebruikte_mesa_ids.txt` (sectie van 17-09); bij de start
wordt een sectie "hPG-simulatie 2026-10-25 — standaard-n150, hergebruik (150)" toegevoegd.
Ruwe uitvoer: `docs/hpg_simulatie_20261025/` (rijen.json, samenvatting.json, rapport.md, log).
