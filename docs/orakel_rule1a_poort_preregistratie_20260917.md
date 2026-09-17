# Preregistratie — kandidaatpoort voor arousal-kwalificatie (Rule 1A)

Datum: 2026-09-17, **geschreven vóór de afleidingsrun.** Vervolg op de
orakel-decompositie (`orakel_rule1a_20260917.md`): met perfecte arousals is
44 % van de herstellingen een referentie-hypopneu; elke koppeling heeft een
echte arousal; de bijvangst zit in de kandidaat (lang, zwak). Vlag gebouwd
in psgscoring (default uit, golden 9/9 byte-identiek):
`rule1a_arousal_min_flow_reduction_pct`, `rule1a_arousal_max_duration_s`,
`rule1a_arousal_min_local_reduction_pct` (env
`PSGSCORING_RULE1A_GATE_{MIN_RED,MAX_DUR,MIN_LOCAL_RED}`), gelezen in
`reinstate_rule1a_arousal_hypopneas` vóór de koppeling; kandidaten dragen
sinds deze commit `flow_reduction_pct` en `local_reduction_pct`.

## Afleiding (verse nachten, orakel-arm)

* Set: 40 MESA-nachten, seed 20260917, uit de 516 nog nooit geregistreerde
  (`/tmp/afleiding40_ids.txt`, wordt geregistreerd); disjunct van de n150.
* Armen A (tak uit) en C (tak aan, referentie-arousals) in-pipeline, zonder
  poort; per herstelling worden debietdaling, lokale daling en duur
  geëxporteerd (`reinstated_fields`).
* Post-hoc poortsweep op die export (exact equivalent aan in-pipeline: de
  poort is een pure filter vóór de koppeling op precies deze velden; de
  koppeling zelf verandert niet). Rooster: min_red ∈ {uit, 35, 40, 45, 50,
  55, 60} × max_dur ∈ {uit, 60, 45, 30} × min_local ∈ {uit, 20, 30, 40}.
* Per cel: R1 (gepoolde herstelprecisie vs `aasm15`), gepaarde ΔF1(C_poort−A)
  vs `aasm15` (met de referentie-eventlijsten hermatched op de opgeslagen
  events), aantal herstellingen.
* **Keuzeregel:** de cel met de hoogste gemiddelde ΔF1 onder de cellen met
  R1 ≥ 0,60; bij gelijkspel de eenvoudigste (minste actieve knoppen, dan
  laagste drempels). Haalt géén cel R1 ≥ 0,60, dan de cel met de hoogste
  R1 onder ΔF1 > 0, en het plafondcriterium geldt als niet gehaald.
  Eén werkpunt wordt bevroren vóór de replicatie.

## Replicatie (n150, bevroren werkpunt, in-pipeline via env)

Armen: C_poort (referentie-arousals + poort) en B_poort (eigen arousals +
poort); A wordt hergebruikt uit de run van 17-09 (zelfde keten; de poort
raakt het pad met tak uit niet — golden 9/9) en op vijf nachten
in-pipeline hercontroleerd op identieke eventlijsten.

**Beslisregel:**
* *Plafond (C_poort):* R1 ≥ 0,60 ÉN ΔF1(C_poort−A) > 0 op de meerderheid met
  Wilcoxon p < 0,05 → de poort repareert de eligibility.
* *Praktijk (B_poort):* ΔF1(B_poort−A) > 0 op de meerderheid met p < 0,05
  ÉN R1(B_poort) ≥ 0,40 ÉN bias in het laagste AHI-tertiel niet meer dan
  +1,0/u boven arm A → B_poort wordt promotiekandidaat (beslissing aan
  Bart, niet aan deze meting). Anders: gebouwd, gemeten, uit.
* Rapporteer altijd per AHI-tertiel; bewaker: ernstklasse-verschuivingen.

## Rekenplan

Afleiding 40 × 2 armen (~15 min, 20 workers); replicatie 150 × 2 armen
(~65 min). Temperatuurbewaker 85 °C; 20 workers is het plafond (piek 82 °C
op 17-09).
