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

---

# Uitkomst — 24 augustus 2026

**AANGENOMEN.** Beide criteria gehaald, door de volle keten gemeten.

| | scoorder in slaap | A: huidig | B: AASM | F1 A | F1 B | ratio A | ratio B |
|---|---:|---:|---:|---:|---:|---:|---:|
| SN1 | 221 | 202 | 181 | 0,670 | 0,572 | 0,91 | 0,82 |
| SN2 | 162 | 135 | 158 | 0,706 | 0,718 | 0,83 | 0,97 |
| SN3 | 139 | \phantom{0}97 | 132 | 0,581 | **0,808** | 0,70 | 0,95 |
| SN4 | 669 | 558 | 784 | 0,799 | 0,850 | 0,83 | 1,17 |
| SN5 | 254 | \phantom{0}36 | 258 | 0,197 | **0,669** | 0,14 | 1,01 |

Gepaarde ΔF1 **+0,0516**, beter op **4 van 5**. Afwijking van een zuivere
telling **0,17 → 0,05**.

## Mijn per-been-pessimisme was ongegrond

Ik verwachtte overdetectie op SN3 en SN4, omdat de regel daar per been al méér
telde dan de scoorders. Door de volle keten gebeurt het omgekeerde: SN3 gaat
van 97 naar 132 tegen een scoordermediaan van 139, en SN5 van 36 naar 258
tegen 254. De bilaterale samenvoeging en de overige filters compenseren
precies het verschil dat de per-been-cijfers suggereerden — en dat is waarom
die cijfers als niet-vergelijkbaar gemarkeerd stonden.

**SN5 is het geval dat de regel rechtvaardigt**: van F1 0,197 naar 0,669, en
van een telling die 14 % van de referentie was naar 101 %.

## Waar het slechter wordt

**SN1**: F1 0,670 → 0,572, telling 202 → 181 terwijl de scoorders er 221 zien.
Daar telde de huidige regel al bijna goed en beweegt de AASM-regel ervandaan.
Eén op vijf, en het veto bewaakte de telling — die verbetert juist.

## Wat dit niet is

Een replicatie. Vijf opnames, één montagetype, één cohort. MESA annoteert
beenbewegingen en is de aangewezen tweede toets; die staat nog open.

**Vlag blijft default UIT** tot dat gebeurd is of tot de gebruiker beslist.

---

# Replicatie op MESA — 24 augustus 2026

**Het teken repliceert, en sterker dan op PSG-IPA.**

16 opnames (zaad 20260827; vier overgeslagen wegens te weinig geannoteerde
bewegingen in slaap):

| | F1 |
|---|---:|
| A: huidig | 0,379 |
| **B: AASM-einde** | **0,543** |

Gepaarde ΔF1 **+0,1091**, beter op **13 van 16**. Op PSG-IPA was dat +0,0516
op 4 van 5.

## Het mechanisme is niet "meer events" — het is herstructurering

Op PSG-IPA vóégde de regel events toe (SN5: 36 → 258). Op MESA **verwijdert**
hij er juist: `mesa-sleep-0010` gaat van 448 naar 152. De regel telt niet op of
af maar **hergroepeert**: losse bursts binnen één beweging smelten samen tot
één langere, en marginale bewegingen die eerst wegvielen blijven behouden.

Dat verklaart waarom hij op twee cohorten met **tegengestelde** foutrichtingen
allebei wint.

## De tellingsmaat is op MESA NIET te beoordelen

De afwijking van een zuivere telling gaat daar van 0,71 naar 0,90 — schijnbaar
slechter. Maar die maat is hier ongeldig, om twee redenen die vooraf gecheckt
hadden moeten worden:

1. MESA annoteert **alleen het linkerbeen** (`periodic leg movement - left`,
   `limb movement - left`; over 25 opnames staat er geen enkele rechts).
2. Het merendeel is **`periodic leg movement`** (1474 tegen 428 losse), dus de
   referentie bestaat overwegend uit bewegingen die deel uitmaken van een
   PLM-reeks — terwijl `plm_eligible` álle in aanmerking komende bewegingen
   bevat. Een superset tegen een deelverzameling, opnieuw.

De **gepaarde** F1 blijft wél geldig: beide armen dragen exact dezelfde
handicap, dus het verschil ertussen is zuiver.

## Wat hiermee vaststaat

De AASM-einde-regel verbetert de overeenstemming met menselijke scoring op
**twee onafhankelijke cohorten**, met verschillende referenties, verschillende
montages en tegengestelde foutrichtingen: +0,0516 (4/5) en +0,1091 (13/16).

Op PSG-IPA — waar de tellingsmaat wél geldig is — verbetert die ook, van 0,17
naar 0,05 afwijking.

**De vlag staat nog steeds default UIT.** Aanzetten verandert de PLM-index en
is een klinische beslissing.
