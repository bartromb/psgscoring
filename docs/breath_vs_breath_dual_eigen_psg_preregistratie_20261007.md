# Preregistratie — `aasm_v3_breath` tegen `aasm_v3_breath_dual` op eigen PSG's (slaapkliniek.be)

Datum: 2026-10-07. **Geschreven vóór de meting.** Aanleiding: `breath` is sinds vandaag de
standaard voor alle scoorders (YASAFlaskified 0.38.9); twee technici hadden `breath_dual`
ingesteld. Op PSG-IPA zijn de twee identiek omdat die montage één flowkanaal heeft. Op de
eigen opnames (thermistor én nasale druk) kan de duale apneuregel wél verschil maken. Dit is
een **verschilmeting zonder referentie**: er is geen menselijke scoring van deze nachten, dus
de uitkomst zegt hoe groot en in welke richting het verschil is, niet welk profiel juist is.

## Materiaal
Productieopnames op de Hetzner-server, in situ (niets wordt gekopieerd, niets verlaat de
server behalve geaggregeerde cijfers per anonieme index). Kandidaten: jobs met
`study_type = diagnostic_psg`, een kanaalkeuze met thermistor én druk als verschillende
kanalen, EDF en resultaten nog aanwezig, een opgeslagen hypnogram en een opgeslagen AHI, geen
split-night (n = 159 op het moment van schrijven). **Steekproef: 20**, seed 20261007,
gestratificeerd op de opgeslagen AHI in tertielen (grenzen 7,1 en 23,1 /u): 7 / 7 / 6. De
koppeling job-id ↔ index R01–R20 blijft op de server
(`/data/slaapkliniek/metingen/breath_dual_20261007/`); het rapport noemt alleen R-indices,
geen datums, namen of identificatienummers.

## Opzet (vast)
Per opname twee armen met **identieke invoer**: hetzelfde opgeslagen hypnogram (YASA, uit de
oorspronkelijke job), dezelfde artefact-epochs uit de opgeslagen resultaten, dezelfde
kanaalkeuze (`pneumo_channels` + kin-EMG) en dezelfde inleesroute (`signal_io.read_raw_signal`
zoals de worker). Alleen `scoring_profile` verschilt: `aasm_v3_breath` tegen
`aasm_v3_breath_dual`. psgscoring 0.34.0 (de versie in het productie-image 0.38.9), niet
0.34.2 — beide profielen bestaan in beide versies; gemeld in het rapport. Geen re-ranker-
of arousalvlaggen gewijzigd; split-night uit.

## Uitkomsten (vooraf)
Per opname en arm: AHI (`ahi_total`), aantallen per eventtype (obstructief, centraal, gemengd,
hypopneu), ernstklasse (< 5 / 5–15 / 15–30 / ≥ 30), en uit de `breath_dual`-arm de
provenance van de duale regel (hoeveel apneukandidaten door beide sensoren bevestigd zijn,
als psgscoring dat rapporteert).
- **Primair:** gepaarde ΔAHI (dual − breath): mediaan, bereik, Wilcoxon; aantal opnames met
  een andere ernstklasse; aantal opnames met |ΔAHI| ≥ 5 /u.
- **Per tertiel** van de opgeslagen AHI (laag / midden / hoog), zoals de huisregel vraagt.
- **Mechanisme:** Δ apneus per type tegen Δ hypopneeën (de duale regel degradeert niet-
  bevestigde apneus naar hypopneu of laat ze vallen).

## Lezing (vooraf, geen beslisregel)
- |ΔAHI| mediaan < 1 /u én geen ernstklasse-wissel → de keuze tussen `breath` en
  `breath_dual` is op deze montages klinisch onverschillig; de standaard blijft wat Bart
  gekozen heeft.
- Anders: de grootte en richting van het verschil per tertiel worden gerapporteerd; of
  `breath_dual` de standaard moet worden op dual-sensor-montages is **Barts beslissing**, niet
  een uitkomst van deze meting (geen referentie).
- Geen bibliotheek-, profiel- of configuratiewijziging volgt automatisch uit deze meting.

## Rekenplan
Een losse container uit het productie-image (`docker run --rm`, uploads read-only, 16 CPU's,
`nice`), 4 analyses parallel, `OMP_NUM_THREADS=2`; naar schatting 45 min. Vooraf: jobqueue
leeg; de app- en workercontainers blijven onaangeraakt. Uitvoer per opname als JSON zonder
`patient_info`; rapport `docs/breath_vs_breath_dual_eigen_psg_20261007.md`, daarna
meting-verificatie.
