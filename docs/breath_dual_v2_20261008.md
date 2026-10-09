# `aasm_v3_breath_dual_v2` (zuivere vereniging) — identiteitstoets op 12 MESA-nachten

Datum: 2026-10-08/09. Profiel gebouwd 08-10 (03a59ae): `aasm_v3_breath` plus alleen de
vereniging van thermistor- en druk-apneus, met poort (`envelope_agreement`, blokkerend) en
referentiekanaal (`flow_reference="apnea"`) van de ouder. Doel: AHI gelijk aan `breath_dual`,
RDI (en ventilatoire last) gelijk aan `breath`, omdat de verificatie van 07-10 de RDI-
verschuiving onder `breath_dual` toeschreef aan de twee neven-schakelaars
(`flow_reference="hypopnea"`, `thermistor_gate="respiratory_band"`). Toets: de eerste 12
nachten van de stap-2-lijst (`opnames.txt`), `scripts/validate_mesa.py` met
`aasm_v3_breath`, `aasm_v3_breath_dual`, `aasm_v3_breath_dual_v2`, drie workers; de run
werd door de bevriezing van 08-10 21:48 onderbroken (6/12) en 09-10 hervat vanaf
checkpoint. Uitvoer `identiteit.json` (buiten git).

| | AHI v2 = dual | RDI v2 = dual | RDI v2 = breath | F1 v2 = dual | apneus v2 = dual |
|---|---:|---:|---:|---:|---:|
| 12 nachten (5 met poort open, 7 dicht) | **12/12** | **12/12** | 1/12 | 12/12 | 12/12 |

**Uitkomst: het doel is niet gehaald, en de toeschrijving van 07-10 klopt niet op MESA.**
v2 is op alle twaalf nachten op AHI, RDI, F1 en apneutelling identiek aan `breath_dual`,
ook op de vijf nachten waar de poort de thermistor goedkeurt en de primaire pas dus op een
andere sensor draait. De RDI-verschuiving tegenover `breath` (bv. mesa-sleep-0537 RDI 20,8
→ 27,5, -0554 25,6 → 32,3) komt dus van de **vereniging zelf** — meer apneus in de
eventlijst veranderen de RERA/FRI-kandidaten die de RDI voeden — en niet van de twee
neven-schakelaars. De ventilatoire last is hier niet vergeleken; de VB-sprong op de vijf
poort-aan-nachten van de eigen PSG's (07-10) kan nog wel aan `flow_reference` liggen.

**Gevolg:** `breath_dual_v2` blijft bestaan als exploratory profiel (default uit) en is op
deze twaalf nachten een duur duplicaat van `breath_dual`; de "RDI gelijk aan breath"-belofte
uit het denkstuk (§3.3) vervalt. Wie de RDI onder de vereniging wil begrijpen, moet in de
RERA-/FRI-koppeling kijken (hoe apneuvensters RERA-kandidaten uitsluiten), niet in de
profielschakelaars. De mechanismeparagraaf in
`docs/breath_vs_breath_dual_eigen_psg_20261007.md` (verificatiebevinding 2) krijgt een
verwijzing hiernaar.
