# Doorwerking van `unet_v1` op `aasm_v3_breath_dual` — AHI, RDI, arousal-index, koppeling

Datum: 2026-10-09/10. Preregistratie `docs/unet_doorwerking_preregistratie_20261009.md`
(0da5152, vóór de meting; beschrijvend, geen beslisregel). Bibliotheek bevroren op cfd75fc
(main, strictness 0,50; de release-branch 0.35.0 met strictness 0,30 is pas ná deze meting
gemerged — de absolute AHI's hier zijn dus die van 0.34.x). Arm lgbm = productieketen;
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

## 2. MESA-val (100 nachten, NSRR-referentie; `scripts/validate_mesa.py`, 8 workers, lgbm-arm 22:57–00:40, unet-arm 00:40–02:21, 0 fouten)

| maat | lgbm mediaan | unet mediaan | Δ unet − lgbm mediaan / gemiddeld | bereik | hoger / lager / gelijk | Wilcoxon p | Δ per NSRR-AHI-tertiel (laag / midden / hoog) |
|---|---:|---:|---:|---|---:|---:|---|
| AHI | 16,35 | 16,05 | −0,20 / −0,22 | −3,1 … +2,2 | 35 / 57 / 8 | 0,014 | −0,19 / −0,32 / −0,16 |
| RDI | 29,75 | 27,70 | −0,90 / −1,28 | −11,7 … +6,8 | 30 / 66 / 4 | 2,4e-4 | −2,41 / −1,72 / +0,32 |
| arousal-index | 22,25 | 20,15 | −1,80 / −2,04 | −22,5 … +18,8 | 26 / 73 / 1 | 1,6e-5 | −4,04 / −2,54 / +0,52 |
| arousals (n) | 149 | 129 | −12 / −14 | −184 … +80 | 26 / 73 / 1 | 1,5e-5 | −28,9 / −14,6 / +1,8 |
| hypopneeën (n) | 41 | 39 | −1 / −1,3 | −20 … +13 | 35 / 57 / 8 | 0,021 | −1,35 / −1,67 / −0,94 |
| respiratoire F1 | 0,511 | 0,506 | +0,003 / +0,010 | −0,05 … +0,14 | 62 / 32 / 6 | 8,6e-4 | +0,030 / +0,002 / −0,001 |

AHI-bias tegen NSRR: lgbm −4,95 (MAE 10,24), unet −5,18 (MAE 10,36).

- **AHI:** mediaan −0,2 /u, bereik −3,1 tot +2,2; op 92 van 100 nachten verandert hij en op
  57 daalt hij — de koppeling verliest enkele arousal-gegradeerde hypopneeën (−1 mediaan).
- **RDI en arousal-index dalen** op 66 en 73 van 100 nachten: het U-Net vindt op MESA-val
  mediaan 20 arousals minder dan de LGBM-keten, vooral op lichte nachten (laag tertiel
  −29 arousals, −2,4 RDI), en meer op zware (hoog tertiel +1,8 / +0,3). Dat is dezelfde
  richting als de replicatie van 27-09 (MESA: index 24,4 → 19,8; de NSRR-referentie ligt
  lager dan de LGBM-telling).
- **Respiratoire F1 stijgt licht** (62 beter, 32 slechter, p = 9e-4), met de winst in het
  lage tertiel (+0,03) — waar een arousal vaker het enige criterium van een hypopneu is.

## 3. Lezing
1. De doorwerking op de **AHI is klein maar niet nul**: mediaan −0,2 /u op MESA en ±0,6 /u op
   PSG-IPA, met uitschieters tot ±3 /u op nachten waar de koppeling een handvol hypopneeën
   kantelt. De respiratoire F1 wordt er eerder beter dan slechter van (62/32 op MESA,
   4/5 ±0,02 op PSG-IPA).
2. De **RDI en de arousal-index** bewegen wél merkbaar, in beide richtingen, en volgen de
   arousaltelling: minder waar de LGBM-keten overtelt (SN2: 137 → 87 arousals, RDI 20,8 →
   13,6; MESA laag tertiel −29), meer waar ze ondertelt (SN3: 200 → 251, arousal-F1
   0,63 → 0,83). De arousal-F1 tegen de referentie stijgt op 4 van 5 PSG-IPA-nachten.
3. Wat volgt: niets automatisch. `unet_v1` blijft opt-in; aanzetten verandert vooral
   arousal-index en RDI in de rapporten en moet, zoals bij de re-ranker, met een klinische
   aan/uit-controle en Barts beslissing over default en werkpunt (0,35 tellingsneutraal op
   PSG-IPA; op NSRR-cohorten telt het 0,85).
