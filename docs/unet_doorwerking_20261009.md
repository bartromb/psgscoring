# Doorwerking van `unet_v1` op `aasm_v3_breath_dual` — AHI, RDI, arousal-index, koppeling

Datum: 2026-10-09/10. Preregistratie `docs/unet_doorwerking_preregistratie_20261009.md`
(0da5152, vóór de meting; beschrijvend, geen beslisregel). Bibliotheek bevroren op cfd75fc
(main; de release-branch 0.35.0 is pas ná deze meting gemerged). Arm lgbm = productieketen;
arm unet = `PSGSCORING_AROUSAL_DETECTOR=unet_v1` (τ 0,35), verder identiek. Ruwe uitvoer
`docs/unet_doorwerking_20261009/` (mesa_lgbm.json, mesa_unet.json, psgipa_rows.json,
analyse.py, psgipa_run.py).

## 1. PSG-IPA SN1–5 (hypnogram van de arousal-replicatie; referenties: 12 scoorders respiratoir, EEG-arousal-referentie van de replicatie)

| nacht | AHI lgbm → unet (scoordermediaan) | RDI | hypopneeën | RERA | arousal-index | arousals (n) | resp-F1 (scoorder-mediaan) | arousal-F1 tegen referentie |
|---|---|---|---|---|---|---|---|---|
| SN1 | 5,4 → 5,2 (6,0) | 9,5 → 9,2 | 16 → 15 | 24 → 23 | 7,1 → 6,4 | 41 → 37 | 0,678 → 0,657 | 0,641 → 0,595 |
| SN2 | 4,9 → 4,3 (4,3) | 20,8 → 13,6 | 21 → 18 | 77 → 45 | 28,2 → 17,9 | 137 → 87 | 0,388 → 0,389 | 0,524 → 0,623 |
| SN3 | 53,5 → 52,5 (54,0) | 58,3 → 58,9 | 43 → 37 | 29 → 39 | 33,0 → 41,4 | 200 → 251 | 0,900 → 0,903 | 0,630 → 0,833 |
| SN4 | 2,0 → 2,2 (3,8) | 6,2 → 6,5 | 12 → 13 | 25 → 26 | 24,1 → 16,6 | 145 → 100 | 0,252 → 0,271 | 0,335 → 0,392 |
| SN5 | 9,7 → 9,5 (10,0) | 12,6 → 12,9 | 51 → 50 | 20 → 24 | 16,7 → 16,5 | 117 → 116 | 0,476 → 0,488 | 0,598 → 0,643 |

- **AHI beweegt nauwelijks** (−0,6 tot +0,2 /u; de koppeling raakt alleen arousal-gegradeerde
  hypopneeën: −1 tot −6 hypopneeën per nacht, +1 op SN4).
- **RDI volgt de arousaltelling**: op SN2 halveert de LGBM-overtelling (137 → 87 arousals,
  RERA 77 → 45, RDI 20,8 → 13,6); op SN3 vindt het U-Net 51 arousals méér (200 → 251,
  arousal-F1 0,63 → 0,83) en stijgt de RDI 0,6.
- **Arousal-F1 tegen de referentie stijgt op 4 van 5 nachten** (SN2 +0,10, SN3 +0,20,
  SN4 +0,06, SN5 +0,05) en daalt op SN1 (−0,05); de arousal-index gaat op SN2/SN4 fors
  omlaag en op SN3 omhoog — het U-Net corrigeert in beide richtingen.
- Respiratoire F1 tegen de scoorders: ±0,02, met SN1 −0,02 als grootste verschuiving.

## 2. MESA-val (100 nachten, NSRR-referentie)
MESA_PLACEHOLDER

## 3. Lezing
LEZING_PLACEHOLDER
