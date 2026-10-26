# hPG-simulatie op MESA — 150 nachten (fouten 0), referentie `aasm15`

| arm | F1 med | F1 gem | prec | rec | AHI-bias | MAE | r | ernst = ref | hyp (som) | hyp rule1a | apneus | detector | s/nacht |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| psg | 0,562 | 0,534 | 0,601 | 0,608 | 0,09 | 8,07 | 0,782 | 94/150 | 10213 | 2605 | 11288 | unet_v1 | 210 |
| hpg_hypno | 0,547 | 0,513 | 0,630 | 0,526 | -2,95 | 8,47 | 0,784 | 93/150 | 7504 | 0 | 11288 | None | 171 |
| hpg | 0,310 | 0,329 | 0,239 | 0,534 | 2,51 | 11,33 | 0,630 | 68/150 | 11618 | 0 | 30628 | None | 183 |

| gepaard t.o.v. psg | ΔAHI med | gem | min … max | lager/hoger | \|Δ\| ≥ 5 | Wilcoxon p | ΔF1 med | beter/slechter | p | ernstwissels |
|---|---:|---:|---|---|---:|---:|---:|---|---:|---|
| hpg_hypno | -2,55 | -3,04 | -12,2 … -0,2 | 150/0 | 23 | 0,0000 | -0,014 | 39/109 | 0,0000 | 27 (Moderate->Mild: 17, Mild->Normal: 7, Severe->Moderate: 3) |
| hpg | 2,40 | 2,42 | -29,3 … 33,1 | 56/91 | 85 | 0,0009 | -0,208 | 1/147 | 0,0000 | 66 (Moderate->Severe: 20, Mild->Moderate: 19, Normal->Mild: 8, Moderate->Mild: 7, Severe->Moderate: 6, Mild->Severe: 2, Mild->Normal: 2, Normal->Severe: 1, Normal->Moderate: 1) |

Tertielgrenzen (NSRR-aasm15-AHI): 13,3 / 28,5 /u

| arm × tertiel | n | ΔAHI med | ΔAHI gem | ΔF1 med | ernstwissels |
|---|---:|---:|---:|---:|---:|
| hpg_hypno laag | 50 | -2,50 | -2,80 | -0,014 | 12 |
| hpg_hypno midden | 50 | -3,20 | -3,72 | -0,017 | 12 |
| hpg_hypno hoog | 50 | -2,30 | -2,60 | -0,013 | 3 |
| hpg laag | 50 | 4,40 | 5,91 | -0,231 | 30 |
| hpg midden | 50 | 3,25 | 3,30 | -0,236 | 24 |
| hpg hoog | 50 | -2,15 | -1,95 | -0,183 | 12 |

TIB/TST mediaan 1,73; aandeel hypopneeën zonder ≥ 3 % desaturatie in de psg-arm 26,5 %.
