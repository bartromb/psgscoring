# PSG-IPA-replicatie van de classifierbeslissing op de RERA-dragende profielen

*23 augustus 2026, ná de uitrol van 0.27.0. Volle pipeline, `aasm_v3_breath`,
artefact-epochs zoals productie.*

## Waarom deze toets iets kon weerleggen

De beslissing van vanavond rust op MESA, en die referentie heeft **één
scoorder per opname**. Eerder is vastgesteld dat dat de optima naar mínder
detecteren trekt: MESA wees werkpunt 0,90 aan waar PSG-IPA 0,80 aanwees.

De claim "de classifier brengt de arousaltelling van 47 % te veel naar 17 % te
weinig" kon dus een artefact van die enkele scoorder zijn. PSG-IPA heeft er
**twaalf** en kan tegenspreken.

## Uitkomst — hij repliceert, en sterker

| | F1 | precisie | recall | telling/referentie | RDI |
|---|---:|---:|---:|---:|---:|
| classifier uit | 0,325 | 0,293 | 0,423 | 1,39 | 19,9 |
| **classifier aan (0,80)** | **0,487** | **0,608** | 0,437 | **0,88** | 14,8 |

Gepaarde ΔF1 **+0,2020**, beter op **5 van 5**. Afwijking van een zuivere
telling 0,39 → 0,12.

Per opname:

| | scoorder-mediaan | F1 | events | ratio | RDI |
|---|---:|---|---|---|---|
| SN1 | 121 | 0,325 → 0,487 | 169 → 94 | 1,40 → 0,78 | 14,7 → 10,0 |
| SN2 | 46 | 0,160 → 0,414 | **257 → 41** | **5,53 → 0,88** | 19,9 → 4,9 |
| SN3 | 142 | 0,154 → 0,419 | 110 → 67 | 0,78 → 0,47 | 51,8 → 47,3 |
| SN4 | 104 | 0,391 → 0,593 | 106 → 97 | 1,01 → 0,93 | 15,9 → 14,8 |
| SN5 | 202 | 0,518 → 0,697 | 281 → 203 | 1,39 → 1,01 | 32,5 → 26,5 |

**SN2 is het geval dat de beslissing rechtvaardigt.** Zonder classifier 257
events tegen een scoordermediaan van 46 — ruim **vijf keer** overdetectie op
een profiel waarvan de RDI in het klinische rapport belandt. Met classifier 41,
ratio 0,88, en de RDI zakt van 19,9 naar 4,9.

Die RDI-daling van driekwart ziet er alarmerend uit tot je ziet waar hij vandaan
komt: de uitgangswaarde was fout, niet de nieuwe.

## Waar het NIET beter wordt, en dat hoort erbij

**SN3 en SN4 gaan van bijna zuiver naar te weinig.** SN4 zat op 1,01 en gaat
naar 0,93; SN3 zat al op 0,78 en zakt naar 0,47 — daar onderdetecteert de
classifier duidelijk. De telling is daarom "dichterbij" op maar **3 van 5**
opnames, terwijl de F1 op 5 van 5 verbetert.

Dat is geen tegenspraak maar een verfijning: de classifier wint vooral door
**precisie** (0,293 → 0,608), en op opnames die al niet overdetecteerden koopt
hij die precisie met recall. Op MESA was hetzelfde patroon zichtbaar (0,83
tegen een referentie van 1,00).

## Wat dit bevestigt en wat niet

**Bevestigd op een onafhankelijk cohort met twaalf scoorders:** de classifier
aanzetten op deze profielen verbetert de arousaldetectie, en de grote
RDI-verschuivingen komen voort uit posities die aantoonbaar fout waren.

**Niet bevestigd:** dat de RDI zelf juister wordt. Er is nog steeds geen
RERA-referentie — PSG-IPA bevat er 3 in de hele manuele set. De RDI-kolom
hierboven is beschrijvend.

**Openstaand:** de onderdetectie op SN3 (ratio 0,47). Dat is dezelfde
onderkant die het werkpunt-onderzoek liet zien bij 0,90, nu bij 0,80 op één
opname. Als er een volgende ronde komt, is dat de plek om te kijken.
