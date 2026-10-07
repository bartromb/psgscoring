# Het menselijk plafond voor respiratoire event-F1 op PSG-IPA: 0,667

**Datum:** 2026-10-07. **Cohort:** PSG-IPA, 5 opnames, 12 scoorders, **330 scoorderparen**.
**Maat:** event-level F1, `validate_psgipa.human_baseline` met `LEGACY_MATCHER` (IoU 0,20,
typeonbewust, greedy), signaalduur `raw.times[-1]` (preload=True, de harnasconventie).
Cijfers: `docs/respiratoir_menselijk_plafond_psgipa_20261007.json`. Tegenhanger van het
arousal-plafond 0,679 (`docs/arousal_menselijk_plafond.md`, 25-08); dit getal ontbrak voor
respiratoire events (claim-trace 07-10) en is voorwerk voor een geleerde respiratoire
detector (`docs/breath_dual_vervolg_denkstuk_20261007.md` §4.1).

| opname | mediaan | p25–p75 | bereik | events per scoorder |
|---|---:|---|---|---|
| SN1 | 0,826 | 0,775–0,849 | 0,719–0,928 | 27–38 |
| SN2 | 0,549 | 0,431–0,640 | 0,286–0,755 | 8–33 |
| SN3 | 0,948 | 0,935–0,959 | 0,876–0,982 | 273–339 |
| SN4 | 0,553 | 0,462–0,637 | 0,051–0,817 | 1–38 |
| SN5 | 0,556 | 0,487–0,640 | 0,248–0,734 | 25–101 |
| **alle 330 paren** | **0,667** | 0,529–0,849 | | |

- SN3 en SN4 reproduceren de getallen uit `docs/nacht_20260901_bevindingen.md` (0,948 /
  0,553): zelfde harnas, zelfde conventie.
- De spreiding is bijna volledig **ziektelast**: op de zware nacht (SN3, ~300 events) zijn
  mensen het voor 95 % eens, op de drie lichte nachten (SN2, SN4, SN5; 8–101 events)
  voor ~55 %, en op SN4 scoort de ene expert één event en de andere achtendertig. Een
  algoritme-F1 op PSG-IPA is daarom alleen leesbaar per opname náást dit plafond, en een
  gepoolde F1 hangt af van welke nachten meetellen.
- **Gebruik:** elke respiratoire event-F1 op PSG-IPA voortaan met dit plafond ernaast
  rapporteren (per opname en 0,667 gepoold), zoals bij arousals met 0,679. Voor een
  geleerde detector is "boven 0,667 gepoold" géén zinvol doel — op SN3 ligt de lat op
  0,95 en op SN4 op 0,55; het doel is het gat tot het plafond per opname.
- Niet gedaan: geen algoritme gemeten, geen nieuwe beslissing. De huidige
  profielvergelijking van 07-10 (`/srv/CODE/docs/profielvergelijking_psgipa_20261007.md`)
  rapporteert AHI's tegen de scoordermediaan, geen event-F1; de eerstvolgende
  PSG-IPA-F1-meting zet dit plafond ernaast.
