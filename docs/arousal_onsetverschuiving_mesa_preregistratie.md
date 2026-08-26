# Voorregistratie — MESA-replicatie van de onsetverschuiving

Vastgelegd 2026-08-26, **vóór** de meting. Repliceert
`arousal_onsetverschuiving_bevinding.md` (PSG-IPA n=5, +0,0123 op 5/5).

## Waarom een replicatie nodig is

Δ = +2 s is gekozen op de **dekkingsmeting** op precies de vijf opnames waarop
hij daarna op F1 is gemeten. Dat cohort kan de keuze niet ook nog bevestigen.
MESA is onafhankelijk: ander apparatuurpark, andere scoorders, één scoorder per
opname in plaats van twaalf, en **menselijke arousals duren er 11,0 s tegen 8,6 s
op PSG-IPA**. Als de verschuiving een eigenschap van ónze detector is, hoort ze
daar terug te komen. Is ze een eigenschap van de PSG-IPA-scoorders, dan niet.

## Opzet

n = 30 MESA-opnames, productieconfiguratie (`detect_arousals_multi`, werkpunt
0,80, afleidingen zoals `_pick_eeg_multi` ze kiest). Referentie: de
`Arousal`-events uit de NSRR-XML, met onset én duur.

Uitkomstmaat: event-F1 bij IoU 0,20, dezelfde matcher als op PSG-IPA.

De detectie draait **één keer per opname**; de verschuivingen worden daarna op
dezelfde eventlijst toegepast. Δ ∈ {−2, −1, 0, +1, +2, +3, +4, +6} s.

## Criteria — beide moeten gehaald

1. **Primair.** Bij Δ = +2 s: gemiddelde paarsgewijze ΔF1 > 0 **én** een
   tweezijdige tekentoets over de 30 opnames met **p < 0,05**. Bij n = 30 is dat
   ≥ 21 van de 30 beter.
2. **Vorm.** Het maximum van de reeks ligt op **+1, +2 of +3 s**.

**Bewaker.** De telling moet over alle verschuivingen gelijk zijn. Wijkt hij af,
dan zit er een fout in het harnas en telt de meting niet.

## Vooraf vastgelegd: hoe ik een half resultaat lees

- **Beide gehaald** → gerepliceerd. Dan is een profielvlag gerechtvaardigd
  (default uit), en pas daarna komt de vraag of hij ergens default aan mag.
- **Primair gehaald, vorm niet** (bijv. maximum op +4 of +6) → dan verschilt de
  optimale verschuiving per cohort en is een vaste waarde niet houdbaar. Geen
  vlag met een vast getal.
- **Primair niet gehaald** → niet gerepliceerd. Dan was de PSG-IPA-winst een
  eigenschap van die vijf opnames of van hun scoorders, en gaat er niets mee
  gebeuren. Ik ga in dat geval **niet** op zoek naar een subgroep waar het wél
  werkt.

## Wat er hoe dan ook niet gebeurt

Geen versiebump, geen uitrol, geen defaultwijziging op grond van deze meting
alleen. Een verschuiving verandert de gerapporteerde arousal-onsets in het
klinische rapport.
