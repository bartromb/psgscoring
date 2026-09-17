# Kandidaatpoort voor arousal-kwalificatie — afleiding en replicatie

*17 september 2026. Pre-registratie: `orakel_rule1a_poort_preregistratie_20260917.md`
(regel én bevroren werkpunt vóór de replicatie). Keten psgscoring 0.34.2 +
commit `f9ce415` (poort, default uit; golden 9/9). Ruwe uitvoer:
`/srv/CODE/docs/orakel_rule1a_20260917/`.*

## Afleiding (40 verse MESA-nachten, orakel-arm)

Sweep over 7 × 4 × 4 cellen op de geëxporteerde herstellingen (per nacht
gecontroleerd: A ∪ herstellingen = C, dus post-hoc = in-pipeline).
Zonder poort: 341 herstellingen, R1 0,455, ΔF1 +0,027. Keuzeregel
(hoogste ΔF1 onder R1 ≥ 0,60) → **min_red 55 %, max_dur 45 s,
min_local 30 %**: 169 herstellingen, R1 0,604, ΔF1 +0,020 (25/40). De
lokale daling is de dragende knop.

## Replicatie (n149, bevroren werkpunt, in-pipeline via env; A hergebruikt uit 17-09 en op 5 nachten bit-identiek hercontroleerd)

| arm | F1 | precisie | recall | bias | herstellingen | **R1** | gat-recall | ernst = ref |
|---|---|---|---|---|---|---|---|---|
| A | 0,438 | 0,520 | 0,412 | −5,61 | 0 | — | 0,238 | 87 |
| C zonder poort (17-09) | 0,461 | 0,523 | 0,463 | −4,38 | 1082 | 0,436 | 0,366 | 84 |
| **C_poort** | 0,450 | 0,530 | 0,453 | −4,88 | 637 | **0,504** | 0,308 | 85 |
| B zonder poort (17-09) | 0,447 | 0,508 | 0,450 | −4,03 | 1372 | 0,254 | 0,329 | 81 |
| **B_poort** | 0,444 | 0,510 | 0,444 | −4,64 | 833 | **0,295** | 0,286 | 83 |

ΔF1(C_poort−A): gemiddeld +0,019, beter op 79/149, Wilcoxon p = 6,8·10⁻¹⁰
(zonder poort: +0,024, 89/149). ΔF1(B_poort−A): +0,002, 60/149, p = 0,22.

Per tertiel: C_poort wint in elk tertiel (T1 +0,037 p=0,01; T2 +0,015
p=9·10⁻⁷; T3 +0,004 p=4·10⁻⁶); B_poort blaast T1 nog steeds op (bias
+2,5 → +3,8; zonder poort +4,8; R1 in T1 0,02).

## Besluit volgens de vooraf vastgelegde regel

* **Plafond niet gehaald**: R1(C_poort) = 0,504 < 0,60. De afleiding (0,604
  op 40 nachten) repliceerde niet op n150 — de cel is de beste van 112 op
  40 nachten en dat is optimistisch, precies waarvoor de replicatie dient.
* **Praktijk niet gehaald**: B_poort is F1-neutraal (p = 0,22), R1 0,295
  < 0,40, en de T1-bias stijgt +1,3/u (grens +1,0).
* Gevolg: **de poort blijft gebouwd, gemeten en uit**; de tak blijft uit.

## Wat de poort wél laat zien

De poort verwijdert 41 % van de orakel-herstellingen; wat weggaat had een
precisie van 0,34, wat blijft 0,50. Ze onderscheidt, maar bescheiden: de
overgebleven helft "foute" orakel-herstellingen zijn kandidaten die NSRR
niet als hypopneu labelde, met een echte arousal erbij.

Eén observatie die de lat in perspectief zet, en die ik pas ná de meting
zag (dus geen onderdeel van de regel): de gewone precisie van arm A tegen
`aasm15` is **0,520** — ook desaturatie-bevestigde events matchen de
referentie maar in de helft van de gevallen. Een herstelprecisie van 0,50
(C_poort) is dus *pariteit* met de detector zelf; de vooraf gekozen lat van
0,60 lag boven wat het eigen basispad haalt. Dat verandert het besluit niet
(de regel stond vast), maar wel de lezing: de eligibility is met deze poort
niet "te toegeeflijk" ten opzichte van de rest van de detector — ze is er
even goed of slecht als. Wat overblijft is het algemene event-niveau-
meningsverschil met de referentie (grenzen, IoU 0,20), niet iets dat een
arousalpoort kan oplossen.

## Rekenkundig

Afleiding 40 × 2 armen in 17 min; replicatie 150 × 2 armen in 56 min, 20
workers, piek 80 °C (bewaker herstart om 11:57 na een stille uitval —
zie `thermal_guard_poort.log`). Sets geregistreerd in
`gebruikte_mesa_ids.txt`.
