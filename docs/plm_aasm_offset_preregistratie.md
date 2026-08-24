# Preregistratie — de AASM-einde-regel voor beenbewegingen

*24 augustus 2026, vóór de meting. De vlag is gebouwd, default uit.*

## Wat er getoetst wordt

`plm_offset_aasm`: het einde van een beweging volgens AASM regel 4.A (eerste
periode van ≥0,5 s onder +2 µV) in plaats van het einde van de periode boven
+8 µV.

Dat de huidige regel **niet conform** is, staat vast en is geen onderdeel van
deze toets. Wat getoetst wordt is of conform maken de overeenstemming met
menselijke scoring **verbetert**.

## Waarom dat niet vanzelf spreekt

Per been in slaap vermenigvuldigt de regel de telling met 9,22× (SN5), 1,40×
(SN4) en 1,79× (SN3). Op SN3 en SN4 telt de huidige regel per been al méér dan
de scoorders. Meer events kunnen daar dus schaden.

Die per-been-cijfers gaan bovendien niet door de bilaterale samenvoeging en de
overige filters, en die doen aantoonbaar veel werk: SN3 gaat van 556 per been
naar 97 uiteindelijk. De meting hieronder draait wél door de hele keten.

## Opzet

PSG-IPA, 5 opnames, twaalf scoorders, `analyze_plm` door de volle keten met
`event_list_cap=None`. Beide kanten op **slaap** gefilterd met hetzelfde
hypnogram — de correctie op mijn eerdere meetfout, waar ik `plm_eligible`
naast álle geannoteerde bewegingen legde.

| arm | einde-regel |
|---|---|
| A | huidig (8 µV voor begin én einde) |
| B | AASM (begin 8 µV, einde 2 µV) |

## Maten en beslisregel — vooraf

**Primair: event-F1** tegen de twaalf scoorders, mediaan per opname.
**Secundair, met vetorecht: de telling** ten opzichte van de scoordermediaan.

De vlag gaat aan **alleen als beide**:

1. mediane **gepaarde** ΔF1 ≥ **+0,010**;
2. de mediane afwijking van een zuivere telling (|ratio − 1|) verslechtert
   **niet**.

**Weerlegd** bij ΔF1 ≤ 0. Daartussen: onbeslist, vlag blijft uit.

Het veto staat er omdat conformiteit hier tegen overdetectie kan inruilen: een
regel die de F1 verbetert maar de telling verdubbelt, levert een PLM-index op
die niet te rapporteren is.

## Wat dit niet uitwijst

Of de niet-conformiteit elders schaadt. Dit meet één cohort van vijf opnames
met één montagetype. Blijft de uitkomst onbeslist, dan is dat een argument om
de vlag te behouden en de vraag op MESA te herhalen, niet om hem te vergeten.
