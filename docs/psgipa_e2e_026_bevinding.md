# End-to-end controle op PSG-IPA van de uitgerolde 0.26.0

*23 augustus 2026, ná de uitrol. Volle pipeline, profiel `aasm_v3_rec`,
artefact-epochs berekend zoals productie ze berekent.*

## Waarom deze controle nodig was

Alle PSG-IPA-cijfers waarop de twee beslissingen rusten kwamen uit **directe
aanroepen** van `detect_arousals_multi` met losse parameters. Geen enkel cijfer
kwam door de volle keten met de profielen zoals ze uitgerold zijn. Een
verbetering in de bibliotheek is pas geleverd als de keten hem doorgeeft.

Bovendien waren de twee vlaggen nooit **samen** op PSG-IPA gemeten.

## Uitkomst

Vijf opnames, twaalf scoorders elk, mediane F1 over de scoorders.

| | F1 | precisie | recall | telling/referentie |
|---|---:|---:|---:|---:|
| 0.25.0 (oud) | 0,465 | 0,436 | 0,454 | 1,13 |
| **0.26.0 (uitgerold)** | **0,487** | **0,608** | 0,437 | **0,88** |

Gepaarde ΔF1 **+0,1040**, beter op **5 van 5**, bereik +0,022 tot +0,238.

Per opname:

| | artefact | F1 | events | ratio |
|---|---:|---|---|---|
| SN1 | 0,8 % | 0,465 → 0,487 | 141 → 94 | 1,17 → 0,78 |
| SN2 | 6,3 % | 0,236 → **0,414** | 61 → 41 | 1,31 → 0,88 |
| SN3 | 13,7 % | 0,181 → **0,419** | 67 → 67 | 0,47 → 0,47 |
| SN4 | 6,9 % | 0,489 → 0,593 | 88 → 97 | 0,84 → 0,93 |
| SN5 | 9,8 % | 0,655 → 0,697 | 227 → 203 | 1,13 → 1,01 |

De winst is het grootst waar de artefactregel het meest wegvlagde (SN2 en SN3),
precies zoals de MESA-meting voorspelde.

**De precisie is waar het vandaan komt**: 0,436 → 0,608, tegen een recall die
licht zakt (0,454 → 0,437). Dat was de diagnose — 16 % van onze events werd
door geen enkele scoorder gedekt — en dit is de correctie ervan.

De telling komt dichter bij de referentie: afwijking van 1,00 gaat van 0,17
naar 0,12.

## De nuance die erbij hoort

De **absolute** F1 door de pipeline (0,487) ligt LAGER dan de losse
componentmetingen suggereerden (0,505 voor de artefactvlag, 0,514 voor de
drempel). De richting en de omvang van de winst kloppen, het niveau niet.

Waarom precies, is **niet vastgesteld**. Het meest waarschijnlijke verschil is
de afleidingskeuze: de componentmetingen gaven expliciet drie afleidingen mee,
terwijl de pipeline zelf kiest welke kanalen hij als afleiding gebruikt. Dat is
een hypothese, geen bevinding.

**Wat het wél betekent, en dat is de les:** de losse componentcijfers zijn geen
voorspelling van wat productie haalt. Ze zijn bruikbaar om ARMEN te vergelijken
— dat is waar ze voor gebruikt zijn — maar niet om een absoluut niveau te
claimen. Als er een niveau in een paper komt, hoort dat uit de volle keten te
komen.

## Wat deze meting niet is

Geen onafhankelijke bevestiging dat 0,80 de juiste drempel is. PSG-IPA heeft
die keuze mede bepaald (0,80 won daar 5/5 tegen 0,60), dus dit cohort kan de
keuze niet meer toetsen. Daarvoor is een derde, ongebruikte MESA-steekproef
nodig.
