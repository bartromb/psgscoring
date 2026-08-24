# `plm_offset_aasm` op de PLM-INDEX: weerlegd op 6/6

**Datum:** 2026-08-24
**Cohort:** MESA n=6, via autodetectie (kanaal `Leg`), profiel `aasm_v3_breath`
**Uitkomstmaat:** PLM-telling tegenover de NSRR-annotatie, gepaard
**Status:** de vlag blijft **default uit**. Deze meting is een reden om hem
uit te houden, niet om hem aan te zetten.

---

## Waarom deze meting pas nu kon

MESA-EDF's dragen één kaal kanaal `Leg`, dat tot psgscoring 0.27.3 geen enkele
rol matchte. De PLM-stap draaide op dat cohort helemaal niet. De eerdere
MESA-validatie van deze vlag liep daarom via een **handgeschreven
`channel_map`**; dit is de eerste meting langs het normale pad.

## Het cijfer

| opname | referentie | vlag uit | vlag aan | \|uit−ref\| | \|aan−ref\| |
|---|---:|---:|---:|---:|---:|
| 0001 |  28 |  37 |  41 |   9 |  13 |
| 0002 |  52 | 244 | 330 | 192 | 278 |
| 0006 | 204 | 189 | 566 |  15 | 362 |
| 0010 |  27 |  73 | 114 |  46 |  87 |
| 0012 |   0 |  72 | 317 |  72 | 317 |
| 0014 |   0 |  74 | 677 |  74 | 677 |

**Mediane afwijking 59 tegen 297,5. De huidige default ligt dichter bij de
referentie op 6 van de 6** (gepaarde Wilcoxon, p = 0,031).

PLM-index, mediaan: referentie **6,62/u**, vlag uit **23,80/u**, vlag aan
**66,15/u**.

Tellingen in plaats van indices, want onze TST en die van de annotatie lopen
uiteen (0010: 1,82 u volgens de annotatie). De richting is daar niet gevoelig
voor — beide armen delen door dezelfde noemer.

## Het mechanisme

De AASM-offset beëindigt een beweging pas onder 2 µV in plaats van 8 µV, dus
elke LM wordt **langer**. Naburige bewegingen smelten samen (LM-tellingen
halveren: 1864 → 590, 1654 → 402, 2152 → 1383) en de resterende intervallen
vallen vaker binnen het 5–90 s-serievenster, waardoor er véél méér LM's als
PLM-serielid kwalificeren. Netto: minder bewegingen, veel meer PLM's.

## Waarom de eerdere validatie het tegenovergestelde leek te zeggen

In het geheugen stond deze vlag als "gevalideerd op TWEE cohorten"
(PSG-IPA +0,0516 op 4/5; MESA +0,1091 op 13/16). Dat waren **event-F1's**, en
die meten de temporele overeenstemming van de bewegingen die je detecteert. De
offset verbetert precies dat: hij zet het EINDE van een LM waar de AASM hem
legt.

De PLM-**serie**detectie stroomafwaarts versterkt diezelfde verandering tot
een heel andere grootte, en de index is wat de kliniek leest. Dat is dezelfde
afruil als bij de graderingsvraag in paper v38: **event-F1 omhoog, index
omlaag**, en de F1 laat dat niet zien.

Les: een vlag die de eventvorm verandert moet gemeten worden op de grootheid
die eruit volgt, niet alleen op de overeenstemming van de events zelf.

## Wat deze meting ook laat zien, en wat zwaarder weegt

Op 0012 en 0014 telde de menselijke scoorder **nul** beenbewegingen — en die
nachten zijn wél volledig gescoord (965 en 956 events, inclusief arousals en
respiratoire events). Onze huidige default vindt daar 72 en 74 PLM's, met de
vlag 317 en 677.

De PLM-detector **overdetecteert dus fors in beide armen**. De mediane
index ligt 3,6× boven de referentie met de vlag uit. Dat is het eigenlijke
optimalisatiedoel voor PLM's, en het is sinds 0.27.3 voor het eerst meetbaar.

## Beperkingen

- n = 6. Klein, maar het teken is 6/6 en het effect is groot.
- MESA annoteert alleen het **linkerbeen** (`PLM (Left)`); wij detecteren op
  één ongezijderd kanaal. Beide zijn "één been waard", maar niet identiek.
- Afwezigheid van annotatie is niet hetzelfde als afwezigheid van beweging.
  Voor 0012 en 0014 is dat nagetrokken: beide nachten dragen bijna duizend
  andere events, dus er is wel degelijk gescoord.
