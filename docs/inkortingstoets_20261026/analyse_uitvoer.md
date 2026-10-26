# Analyse-uitvoer inkortingstoets (n = 30, fouten 0)

## 1. Controle: H0_old tegen H0_new (dekkend hypnogram)

- eventlijst + AHI gelijk op 30/30; stage_baseline_used True in beide op 30/30

## 2_implementatie_H1: H1_new tegen H0_new in het H1_new-venster

| nacht | tertiel | T_cut (h) | apneus ref → new | Δ% apneus | events ref → new | Δ% events | Jaccard | sbu ref/new |
|---|---|---|---|---|---|---|---|---|
| mesa-sleep-0001 | midden | 11.99 | 80 → 80 | +0,0 | 184 → 184 | +0,0 | 1,000 | True/True |
| mesa-sleep-0111 | laag | 10.99 | 183 → 183 | +0,0 | 186 → 186 | +0,0 | 1,000 | True/True |
| mesa-sleep-0195 | midden | 8.99 | 46 → 46 | +0,0 | 89 → 89 | +0,0 | 1,000 | True/True |
| mesa-sleep-0225 | laag | 8.99 | 220 → 220 | +0,0 | 233 → 233 | +0,0 | 1,000 | True/True |
| mesa-sleep-0286 | hoog | 8.99 | 0 → 0 | +0,0 | 256 → 256 | +0,0 | 1,000 | True/True |
| mesa-sleep-0392 | midden | 9.99 | 29 → 29 | +0,0 | 121 → 121 | +0,0 | 1,000 | True/True |
| mesa-sleep-0460 | laag | 8.99 | 45 → 45 | +0,0 | 84 → 84 | +0,0 | 1,000 | True/True |
| mesa-sleep-0468 | hoog | 9.99 | 258 → 258 | +0,0 | 360 → 360 | +0,0 | 1,000 | True/True |
| mesa-sleep-0537 | hoog | 8.99 | 53 → 53 | +0,0 | 108 → 108 | +0,0 | 1,000 | True/True |
| mesa-sleep-0554 | midden | 12.99 | 120 → 120 | +0,0 | 195 → 195 | +0,0 | 1,000 | True/True |
| mesa-sleep-0583 | midden | 11.99 | 39 → 39 | +0,0 | 70 → 70 | +0,0 | 1,000 | True/True |
| mesa-sleep-0651 | hoog | 9.99 | 168 → 168 | +0,0 | 315 → 315 | +0,0 | 1,000 | True/True |
| mesa-sleep-0899 | laag | 9.99 | 5 → 5 | +0,0 | 19 → 19 | +0,0 | 1,000 | True/True |
| mesa-sleep-0910 | midden | 11.99 | 152 → 152 | +0,0 | 197 → 197 | +0,0 | 1,000 | True/True |
| mesa-sleep-0917 | midden | 9.99 | 38 → 38 | +0,0 | 56 → 56 | +0,0 | 1,000 | True/True |
| mesa-sleep-0934 | hoog | 9.99 | 119 → 119 | +0,0 | 232 → 232 | +0,0 | 1,000 | True/True |
| mesa-sleep-0979 | midden | 8.99 | 94 → 94 | +0,0 | 187 → 187 | +0,0 | 1,000 | True/True |
| mesa-sleep-1122 | laag | 9.99 | 13 → 13 | +0,0 | 81 → 81 | +0,0 | 1,000 | True/True |
| mesa-sleep-1224 | laag | 9.99 | 62 → 62 | +0,0 | 95 → 95 | +0,0 | 1,000 | True/True |
| mesa-sleep-1227 | laag | 8.99 | 76 → 76 | +0,0 | 96 → 96 | +0,0 | 1,000 | True/True |
| mesa-sleep-1305 | laag | 11.99 | 98 → 98 | +0,0 | 125 → 125 | +0,0 | 1,000 | True/True |
| mesa-sleep-1374 | hoog | 8.99 | 1 → 1 | +0,0 | 82 → 82 | +0,0 | 1,000 | True/True |
| mesa-sleep-1395 | laag | 8.99 | 9 → 9 | +0,0 | 75 → 75 | +0,0 | 1,000 | True/True |
| mesa-sleep-1589 | hoog | 9.99 | 150 → 150 | +0,0 | 212 → 212 | +0,0 | 1,000 | True/True |
| mesa-sleep-1735 | laag | 8.99 | 42 → 42 | +0,0 | 112 → 112 | +0,0 | 1,000 | True/True |
| mesa-sleep-1862 | midden | 9.99 | 13 → 13 | +0,0 | 106 → 106 | +0,0 | 1,000 | True/True |
| mesa-sleep-1892 | hoog | 11.99 | 58 → 58 | +0,0 | 268 → 268 | +0,0 | 1,000 | True/True |
| mesa-sleep-1998 | hoog | 9.99 | 106 → 106 | +0,0 | 220 → 220 | +0,0 | 1,000 | True/True |
| mesa-sleep-2083 | hoog | 11.99 | 90 → 90 | +0,0 | 197 → 197 | +0,0 | 1,000 | True/True |
| mesa-sleep-2149 | midden | 8.67 | 3 → 3 | +0,0 | 45 → 45 | +0,0 | 1,000 | True/True |

- mediaan |Δ% apneus| 0,0 (getekend +0,0), nachten > 5 %: 0 []; mediaan |Δ% events| 0,0; Jaccard mediaan 1,000, min 1,000 → **STABIEL** (regel: ≤ 2 % én ≤ 3/30 > 5 % én Jaccard ≥ 0,95)

## 2b_implementatie_H5: H5_new tegen H0_new in het H5_new-venster

| nacht | tertiel | T_cut (h) | apneus ref → new | Δ% apneus | events ref → new | Δ% events | Jaccard | sbu ref/new |
|---|---|---|---|---|---|---|---|---|
| mesa-sleep-0001 | midden | 11.40 | 80 → 80 | +0,0 | 184 → 184 | +0,0 | 1,000 | True/True |
| mesa-sleep-0111 | laag | 10.45 | 183 → 183 | +0,0 | 186 → 186 | +0,0 | 1,000 | True/True |
| mesa-sleep-0195 | midden | 8.55 | 46 → 46 | +0,0 | 89 → 89 | +0,0 | 1,000 | True/True |
| mesa-sleep-0225 | laag | 8.55 | 220 → 220 | +0,0 | 233 → 233 | +0,0 | 1,000 | True/True |
| mesa-sleep-0286 | hoog | 8.55 | 0 → 0 | +0,0 | 256 → 256 | +0,0 | 1,000 | True/True |
| mesa-sleep-0392 | midden | 9.50 | 29 → 29 | +0,0 | 121 → 121 | +0,0 | 1,000 | True/True |
| mesa-sleep-0460 | laag | 8.55 | 45 → 50 | +11,1 | 82 → 87 | +6,1 | 0,496 | True/True |
| mesa-sleep-0468 | hoog | 9.50 | 258 → 258 | +0,0 | 360 → 360 | +0,0 | 1,000 | True/True |
| mesa-sleep-0537 | hoog | 8.55 | 52 → 52 | +0,0 | 106 → 106 | +0,0 | 0,752 | True/True |
| mesa-sleep-0554 | midden | 12.35 | 120 → 120 | +0,0 | 195 → 195 | +0,0 | 1,000 | True/True |
| mesa-sleep-0583 | midden | 11.40 | 39 → 39 | +0,0 | 70 → 70 | +0,0 | 1,000 | True/True |
| mesa-sleep-0651 | hoog | 9.50 | 168 → 168 | +0,0 | 315 → 315 | +0,0 | 1,000 | True/True |
| mesa-sleep-0899 | laag | 9.50 | 5 → 5 | +0,0 | 19 → 19 | +0,0 | 1,000 | True/True |
| mesa-sleep-0910 | midden | 11.40 | 152 → 152 | +0,0 | 197 → 197 | +0,0 | 1,000 | True/True |
| mesa-sleep-0917 | midden | 9.50 | 38 → 38 | +0,0 | 56 → 56 | +0,0 | 1,000 | True/True |
| mesa-sleep-0934 | hoog | 9.50 | 119 → 119 | +0,0 | 232 → 232 | +0,0 | 1,000 | True/True |
| mesa-sleep-0979 | midden | 8.55 | 84 → 84 | +0,0 | 172 → 173 | +0,6 | 0,835 | True/True |
| mesa-sleep-1122 | laag | 9.50 | 13 → 13 | +0,0 | 81 → 81 | +0,0 | 1,000 | True/True |
| mesa-sleep-1224 | laag | 9.50 | 62 → 62 | +0,0 | 95 → 95 | +0,0 | 1,000 | True/True |
| mesa-sleep-1227 | laag | 8.55 | 58 → 58 | +0,0 | 76 → 76 | +0,0 | 0,854 | True/True |
| mesa-sleep-1305 | laag | 11.40 | 98 → 98 | +0,0 | 125 → 126 | +0,8 | 0,992 | True/True |
| mesa-sleep-1374 | hoog | 8.55 | 1 → 1 | +0,0 | 82 → 82 | +0,0 | 1,000 | True/True |
| mesa-sleep-1395 | laag | 8.55 | 9 → 9 | +0,0 | 75 → 75 | +0,0 | 1,000 | True/True |
| mesa-sleep-1589 | hoog | 9.50 | 150 → 144 | -4,0 | 202 → 200 | -1,0 | 0,436 | True/True |
| mesa-sleep-1735 | laag | 8.55 | 42 → 41 | -2,4 | 112 → 111 | -0,9 | 0,874 | True/True |
| mesa-sleep-1862 | midden | 9.50 | 13 → 13 | +0,0 | 106 → 106 | +0,0 | 1,000 | True/True |
| mesa-sleep-1892 | hoog | 11.40 | 58 → 58 | +0,0 | 268 → 268 | +0,0 | 1,000 | True/True |
| mesa-sleep-1998 | hoog | 9.50 | 106 → 106 | +0,0 | 220 → 220 | +0,0 | 1,000 | True/True |
| mesa-sleep-2083 | hoog | 11.40 | 90 → 90 | +0,0 | 197 → 197 | +0,0 | 1,000 | True/True |
| mesa-sleep-2149 | midden | 8.24 | 3 → 3 | +0,0 | 45 → 45 | +0,0 | 1,000 | True/True |

- mediaan |Δ% apneus| 0,0 (getekend +0,0), nachten > 5 %: 1 [('mesa-sleep-0460', 45, 50, 11.1)]; mediaan |Δ% events| 0,0; Jaccard mediaan 1,000, min 0,436 → **STABIEL** (regel: ≤ 2 % én ≤ 3/30 > 5 % én Jaccard ≥ 0,95)

## 3_oud_pad_H1_old_vs_H1_new: H1_old tegen H1_new in het H1_old-venster

| nacht | tertiel | T_cut (h) | apneus ref → new | Δ% apneus | events ref → new | Δ% events | Jaccard | sbu ref/new |
|---|---|---|---|---|---|---|---|---|
| mesa-sleep-0001 | midden | 11.99 | 80 → 71 | -11,2 | 184 → 170 | -7,6 | 0,383 | True/False |
| mesa-sleep-0111 | laag | 10.99 | 183 → 3 | -98,4 | 186 → 9 | -95,2 | 0,026 | True/False |
| mesa-sleep-0195 | midden | 8.99 | 46 → 47 | +2,2 | 89 → 87 | -2,2 | 0,285 | True/False |
| mesa-sleep-0225 | laag | 8.99 | 220 → 17 | -92,3 | 233 → 32 | -86,3 | 0,056 | True/False |
| mesa-sleep-0286 | hoog | 8.99 | 0 → 0 | +0,0 | 256 → 256 | +0,0 | 0,984 | True/False |
| mesa-sleep-0392 | midden | 9.99 | 29 → 28 | -3,4 | 121 → 120 | -0,8 | 0,575 | True/False |
| mesa-sleep-0460 | laag | 8.99 | 45 → 34 | -24,4 | 84 → 77 | -8,3 | 0,309 | True/False |
| mesa-sleep-0468 | hoog | 9.99 | 258 → 257 | -0,4 | 360 → 354 | -1,7 | 0,155 | True/False |
| mesa-sleep-0537 | hoog | 8.99 | 53 → 52 | -1,9 | 108 → 102 | -5,6 | 0,382 | True/False |
| mesa-sleep-0554 | midden | 12.99 | 120 → 86 | -28,3 | 195 → 162 | -16,9 | 0,284 | True/False |
| mesa-sleep-0583 | midden | 11.99 | 39 → 10 | -74,4 | 70 → 49 | -30,0 | 0,384 | True/False |
| mesa-sleep-0651 | hoog | 9.99 | 168 → 159 | -5,4 | 315 → 312 | -1,0 | 0,326 | True/False |
| mesa-sleep-0899 | laag | 9.99 | 5 → 4 | -20,0 | 19 → 18 | -5,3 | 0,682 | True/False |
| mesa-sleep-0910 | midden | 11.99 | 152 → 87 | -42,8 | 197 → 144 | -26,9 | 0,258 | True/False |
| mesa-sleep-0917 | midden | 9.99 | 38 → 38 | +0,0 | 56 → 56 | +0,0 | 0,287 | True/False |
| mesa-sleep-0934 | hoog | 9.99 | 119 → 91 | -23,5 | 232 → 223 | -3,9 | 0,323 | True/False |
| mesa-sleep-0979 | midden | 8.99 | 94 → 96 | +2,1 | 187 → 187 | +0,0 | 0,331 | True/False |
| mesa-sleep-1122 | laag | 9.99 | 13 → 9 | -30,8 | 81 → 77 | -4,9 | 0,681 | True/False |
| mesa-sleep-1224 | laag | 9.99 | 62 → 40 | -35,5 | 95 → 83 | -12,6 | 0,211 | True/False |
| mesa-sleep-1227 | laag | 8.99 | 76 → 87 | +14,5 | 96 → 106 | +10,4 | 0,116 | True/False |
| mesa-sleep-1305 | laag | 11.99 | 98 → 13 | -86,7 | 125 → 41 | -67,2 | 0,194 | True/False |
| mesa-sleep-1374 | hoog | 8.99 | 1 → 1 | +0,0 | 82 → 82 | +0,0 | 0,976 | True/False |
| mesa-sleep-1395 | laag | 8.99 | 9 → 12 | +33,3 | 75 → 78 | +4,0 | 0,719 | True/False |
| mesa-sleep-1589 | hoog | 9.99 | 150 → 125 | -16,7 | 212 → 198 | -6,6 | 0,206 | True/False |
| mesa-sleep-1735 | laag | 8.99 | 42 → 43 | +2,4 | 112 → 111 | -0,9 | 0,439 | True/False |
| mesa-sleep-1862 | midden | 9.99 | 13 → 12 | -7,7 | 106 → 107 | +0,9 | 0,775 | True/False |
| mesa-sleep-1892 | hoog | 11.99 | 58 → 10 | -82,8 | 268 → 225 | -16,0 | 0,638 | True/False |
| mesa-sleep-1998 | hoog | 9.99 | 106 → 94 | -11,3 | 220 → 213 | -3,2 | 0,392 | True/False |
| mesa-sleep-2083 | hoog | 11.99 | 90 → 27 | -70,0 | 197 → 134 | -32,0 | 0,471 | True/False |
| mesa-sleep-2149 | midden | 8.67 | 3 → 2 | -33,3 | 45 → 44 | -2,2 | 0,894 | True/False |

- mediaan |Δ% apneus| 18,3 (getekend -14,0), nachten > 5 %: 21 [('mesa-sleep-0001', 80, 71, -11.2), ('mesa-sleep-0111', 183, 3, -98.4), ('mesa-sleep-0225', 220, 17, -92.3), ('mesa-sleep-0460', 45, 34, -24.4), ('mesa-sleep-0554', 120, 86, -28.3), ('mesa-sleep-0583', 39, 10, -74.4), ('mesa-sleep-0651', 168, 159, -5.4), ('mesa-sleep-0899', 5, 4, -20.0), ('mesa-sleep-0910', 152, 87, -42.8), ('mesa-sleep-0934', 119, 91, -23.5), ('mesa-sleep-1122', 13, 9, -30.8), ('mesa-sleep-1224', 62, 40, -35.5), ('mesa-sleep-1227', 76, 87, 14.5), ('mesa-sleep-1305', 98, 13, -86.7), ('mesa-sleep-1395', 9, 12, 33.3), ('mesa-sleep-1589', 150, 125, -16.7), ('mesa-sleep-1862', 13, 12, -7.7), ('mesa-sleep-1892', 58, 10, -82.8), ('mesa-sleep-1998', 106, 94, -11.3), ('mesa-sleep-2083', 90, 27, -70.0), ('mesa-sleep-2149', 3, 2, -33.3)]; mediaan |Δ% events| 5,1; Jaccard mediaan 0,356, min 0,026 → **NIET STABIEL** (regel: ≤ 2 % én ≤ 3/30 > 5 % én Jaccard ≥ 0,95)

## 4_kwaliteit_H1: H1_new tegen H1_old tegen `aasm15` (venster H1_new)

| nacht | tertiel | AHI ref | AHI old → new | F1 old → new | ΔF1 | apneu-F1 old → new | apneus old → new (ratio) | ernst ref / old / new | sbu old/new |
|---|---|---|---|---|---|---|---|---|---|
| mesa-sleep-0001 | midden | 28,5 | 29,7 → 32,1 | 0,721 → 0,680 | -0,041 | 0,400 → 0,364 | 71 → 80 (0,89) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-0111 | laag | 0,0 | 2,3 → 48,4 | 0,000 → 0,000 | +0,000 | 0,000 → 0,000 | 3 → 183 (0,02) | Normal / Normal / Severe | False/True |
| mesa-sleep-0195 | midden | 18,3 | 16,1 → 16,5 | 0,538 → 0,521 | -0,016 | 0,412 → 0,328 | 47 → 46 (1,02) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0225 | laag | 1,7 | 4,8 → 35,3 | 0,279 → 0,057 | -0,222 | 0,105 → 0,009 | 17 → 220 (0,08) | Normal / Normal / Severe | False/True |
| mesa-sleep-0286 | hoog | 83,8 | 34,4 → 34,4 | 0,553 → 0,553 | +0,000 | 0,000 → 0,000 | 0 → 0 (—) | Severe / Severe / Severe | False/True |
| mesa-sleep-0392 | midden | 17,3 | 16,8 → 16,9 | 0,713 → 0,710 | -0,003 | 0,000 → 0,000 | 28 → 29 (0,97) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0460 | laag | 11,5 | 15,5 → 16,9 | 0,582 → 0,596 | +0,014 | 0,111 → 0,085 | 34 → 45 (0,76) | Mild / Moderate / Moderate | False/True |
| mesa-sleep-0468 | hoog | 56,1 | 53,1 → 54,0 | 0,819 → 0,809 | -0,009 | 0,650 → 0,643 | 257 → 258 (1,00) | Severe / Severe / Severe | False/True |
| mesa-sleep-0537 | hoog | 33,6 | 22,7 → 24,0 | 0,545 → 0,517 | -0,028 | 0,172 → 0,169 | 52 → 53 (0,98) | Severe / Moderate / Moderate | False/True |
| mesa-sleep-0554 | midden | 20,7 | 20,1 → 24,1 | 0,456 → 0,425 | -0,031 | 0,000 → 0,017 | 86 → 120 (0,72) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0583 | midden | 19,3 | 7,3 → 10,5 | 0,382 → 0,432 | +0,050 | 0,000 → 0,000 | 10 → 39 (0,26) | Moderate / Mild / Mild | False/True |
| mesa-sleep-0651 | hoog | 62,7 | 60,0 → 60,6 | 0,853 → 0,858 | +0,005 | 0,738 → 0,743 | 159 → 168 (0,95) | Severe / Severe / Severe | False/True |
| mesa-sleep-0899 | laag | 3,1 | 2,9 → 3,1 | 0,270 → 0,263 | -0,007 | 0,000 → 0,000 | 4 → 5 (0,80) | Normal / Normal / Normal | False/True |
| mesa-sleep-0910 | midden | 26,1 | 26,7 → 36,5 | 0,625 → 0,562 | -0,062 | 0,426 → 0,369 | 87 → 152 (0,57) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-0917 | midden | 28,2 | 8,5 → 8,5 | 0,365 → 0,365 | +0,000 | 0,500 → 0,500 | 38 → 38 (1,00) | Moderate / Mild / Mild | False/True |
| mesa-sleep-0934 | hoog | 52,8 | 37,6 → 39,2 | 0,701 → 0,705 | +0,003 | 0,544 → 0,488 | 91 → 119 (0,76) | Severe / Severe / Severe | False/True |
| mesa-sleep-0979 | midden | 24,0 | 26,1 → 26,1 | 0,769 → 0,774 | +0,006 | 0,713 → 0,723 | 96 → 94 (1,02) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1122 | laag | 15,2 | 14,7 → 15,4 | 0,599 → 0,609 | +0,010 | 0,000 → 0,000 | 9 → 13 (0,69) | Moderate / Mild / Moderate | False/True |
| mesa-sleep-1224 | laag | 15,9 | 16,1 → 18,4 | 0,606 → 0,610 | +0,004 | 0,551 → 0,571 | 40 → 62 (0,65) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1227 | laag | 17,0 | 23,6 → 21,4 | 0,725 → 0,756 | +0,031 | 0,224 → 0,230 | 87 → 76 (1,14) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1305 | laag | 0,6 | 4,9 → 15,0 | 0,130 → 0,046 | -0,084 | 0,143 → 0,020 | 13 → 98 (0,13) | Normal / Normal / Moderate | False/True |
| mesa-sleep-1374 | hoog | 31,6 | 18,3 → 18,3 | 0,554 → 0,554 | +0,000 | 0,200 → 0,200 | 1 → 1 (1,00) | Severe / Moderate / Moderate | False/True |
| mesa-sleep-1395 | laag | 12,6 | 11,5 → 11,1 | 0,380 → 0,400 | +0,020 | 0,000 → 0,000 | 12 → 9 (1,33) | Mild / Mild / Mild | False/True |
| mesa-sleep-1589 | hoog | 34,1 | 31,3 → 33,5 | 0,739 → 0,757 | +0,018 | 0,797 → 0,766 | 125 → 150 (0,83) | Severe / Severe / Severe | False/True |
| mesa-sleep-1735 | laag | 16,0 | 16,5 → 16,6 | 0,758 → 0,745 | -0,013 | 0,771 → 0,722 | 43 → 42 (1,02) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1862 | midden | 28,7 | 21,5 → 21,3 | 0,552 → 0,554 | +0,002 | 0,375 → 0,353 | 12 → 13 (0,92) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1892 | hoog | 66,9 | 38,2 → 45,5 | 0,656 → 0,689 | +0,033 | 0,137 → 0,397 | 10 → 58 (0,17) | Severe / Severe / Severe | False/True |
| mesa-sleep-1998 | hoog | 32,0 | 39,4 → 40,7 | 0,694 → 0,702 | +0,008 | 0,210 → 0,188 | 94 → 106 (0,89) | Severe / Severe / Severe | False/True |
| mesa-sleep-2083 | hoog | 29,4 | 26,4 → 38,9 | 0,580 → 0,538 | -0,042 | 0,651 → 0,283 | 27 → 90 (0,30) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-2149 | midden | 20,9 | 9,3 → 9,5 | 0,434 → 0,431 | -0,003 | 0,000 → 0,000 | 2 → 3 (0,67) | Moderate / Mild / Mild | False/True |

- ΔF1 (H1_new − H1_old): mediaan +0,0000, gemiddeld -0,0119, beter/slechter/gelijk 13/13/4, Wilcoxon p = 0,5090; apneu-ΔF1 mediaan +0,0000
- bias (AHI − ref) gemiddeld old -5,07 → new -0,53 (mediaan -0,83 → -0,21); ernst gelijk aan ref old 23/30 → new 18/30; apneus old/new ratio mediaan 0,83 (bereik 0,02 … 1,33); stage_baseline_used: old False op 30/30, new True op 30/30
- per tertiel van de referentie-AHI (grenzen 17.3 / 29.4): laag n=10 ΔF1 mediaan +0,0021 beter/slechter 5/4; midden n=10 ΔF1 mediaan -0,0030 beter/slechter 3/6; hoog n=10 ΔF1 mediaan +0,0015 beter/slechter 5/3
- **Lezing (huisregel): MESA beslist niet**

## 4b_kwaliteit_H5: H5_new tegen H5_old tegen `aasm15` (venster H5_new)

| nacht | tertiel | AHI ref | AHI old → new | F1 old → new | ΔF1 | apneu-F1 old → new | apneus old → new (ratio) | ernst ref / old / new | sbu old/new |
|---|---|---|---|---|---|---|---|---|---|
| mesa-sleep-0001 | midden | 28,5 | 29,7 → 32,1 | 0,721 → 0,680 | -0,041 | 0,400 → 0,364 | 71 → 80 (0,89) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-0111 | laag | 0,0 | 2,3 → 48,4 | 0,000 → 0,000 | +0,000 | 0,000 → 0,000 | 3 → 183 (0,02) | Normal / Normal / Severe | False/True |
| mesa-sleep-0195 | midden | 18,3 | 16,1 → 16,5 | 0,538 → 0,521 | -0,016 | 0,412 → 0,328 | 47 → 46 (1,02) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0225 | laag | 1,7 | 4,8 → 35,3 | 0,279 → 0,057 | -0,222 | 0,105 → 0,009 | 17 → 220 (0,08) | Normal / Normal / Severe | False/True |
| mesa-sleep-0286 | hoog | 83,8 | 34,4 → 34,4 | 0,553 → 0,553 | +0,000 | 0,000 → 0,000 | 0 → 0 (—) | Severe / Severe / Severe | False/True |
| mesa-sleep-0392 | midden | 17,4 | 16,8 → 16,9 | 0,713 → 0,710 | -0,003 | 0,000 → 0,000 | 28 → 29 (0,97) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0460 | laag | 12,4 | 16,6 → 19,0 | 0,586 → 0,583 | -0,003 | 0,111 → 0,077 | 34 → 50 (0,68) | Mild / Moderate / Moderate | False/True |
| mesa-sleep-0468 | hoog | 56,1 | 53,1 → 54,0 | 0,819 → 0,809 | -0,009 | 0,650 → 0,643 | 257 → 258 (1,00) | Severe / Severe / Severe | False/True |
| mesa-sleep-0537 | hoog | 37,2 | 24,6 → 26,1 | 0,550 → 0,521 | -0,028 | 0,175 → 0,172 | 51 → 52 (0,98) | Severe / Moderate / Moderate | False/True |
| mesa-sleep-0554 | midden | 20,7 | 20,1 → 24,1 | 0,456 → 0,425 | -0,031 | 0,000 → 0,017 | 86 → 120 (0,72) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-0583 | midden | 19,3 | 7,3 → 10,5 | 0,382 → 0,432 | +0,050 | 0,000 → 0,000 | 10 → 39 (0,26) | Moderate / Mild / Mild | False/True |
| mesa-sleep-0651 | hoog | 62,7 | 60,0 → 60,6 | 0,853 → 0,858 | +0,005 | 0,738 → 0,743 | 159 → 168 (0,95) | Severe / Severe / Severe | False/True |
| mesa-sleep-0899 | laag | 3,1 | 2,9 → 3,1 | 0,270 → 0,263 | -0,007 | 0,000 → 0,000 | 4 → 5 (0,80) | Normal / Normal / Normal | False/True |
| mesa-sleep-0910 | midden | 26,1 | 26,7 → 36,5 | 0,625 → 0,562 | -0,062 | 0,426 → 0,369 | 87 → 152 (0,57) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-0917 | midden | 28,2 | 8,5 → 8,5 | 0,365 → 0,365 | +0,000 | 0,500 → 0,500 | 38 → 38 (1,00) | Moderate / Mild / Mild | False/True |
| mesa-sleep-0934 | hoog | 52,8 | 37,6 → 39,2 | 0,701 → 0,705 | +0,003 | 0,544 → 0,488 | 91 → 119 (0,76) | Severe / Severe / Severe | False/True |
| mesa-sleep-0979 | midden | 22,7 | 25,3 → 25,5 | 0,755 → 0,765 | +0,010 | 0,687 → 0,692 | 85 → 84 (1,01) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1122 | laag | 15,2 | 14,7 → 15,4 | 0,599 → 0,609 | +0,010 | 0,000 → 0,000 | 9 → 13 (0,69) | Moderate / Mild / Moderate | False/True |
| mesa-sleep-1224 | laag | 15,9 | 16,1 → 18,4 | 0,606 → 0,610 | +0,004 | 0,551 → 0,571 | 40 → 62 (0,65) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1227 | laag | 14,6 | 20,9 → 18,2 | 0,716 → 0,745 | +0,028 | 0,182 → 0,185 | 70 → 58 (1,21) | Mild / Moderate / Moderate | False/True |
| mesa-sleep-1305 | laag | 0,6 | 5,1 → 15,2 | 0,128 → 0,046 | -0,082 | 0,143 → 0,020 | 13 → 98 (0,13) | Normal / Mild / Moderate | False/True |
| mesa-sleep-1374 | hoog | 31,6 | 18,3 → 18,3 | 0,554 → 0,554 | +0,000 | 0,200 → 0,200 | 1 → 1 (1,00) | Severe / Moderate / Moderate | False/True |
| mesa-sleep-1395 | laag | 12,6 | 11,5 → 11,1 | 0,380 → 0,400 | +0,020 | 0,000 → 0,000 | 12 → 9 (1,33) | Mild / Mild / Mild | False/True |
| mesa-sleep-1589 | hoog | 35,6 | 32,0 → 34,2 | 0,754 → 0,775 | +0,020 | 0,797 → 0,776 | 125 → 144 (0,87) | Severe / Severe / Severe | False/True |
| mesa-sleep-1735 | laag | 17,1 | 17,6 → 17,6 | 0,758 → 0,749 | -0,009 | 0,771 → 0,729 | 43 → 41 (1,05) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1862 | midden | 28,7 | 21,5 → 21,3 | 0,552 → 0,554 | +0,002 | 0,375 → 0,353 | 12 → 13 (0,92) | Moderate / Moderate / Moderate | False/True |
| mesa-sleep-1892 | hoog | 66,9 | 38,2 → 45,5 | 0,656 → 0,689 | +0,033 | 0,137 → 0,397 | 10 → 58 (0,17) | Severe / Severe / Severe | False/True |
| mesa-sleep-1998 | hoog | 32,0 | 39,4 → 40,7 | 0,694 → 0,702 | +0,008 | 0,210 → 0,188 | 94 → 106 (0,89) | Severe / Severe / Severe | False/True |
| mesa-sleep-2083 | hoog | 29,4 | 26,4 → 38,9 | 0,580 → 0,538 | -0,042 | 0,651 → 0,283 | 27 → 90 (0,30) | Moderate / Moderate / Severe | False/True |
| mesa-sleep-2149 | midden | 21,0 | 9,3 → 9,5 | 0,434 → 0,431 | -0,003 | 0,000 → 0,000 | 2 → 3 (0,67) | Moderate / Mild / Mild | False/True |

- ΔF1 (H5_new − H5_old): mediaan +0,0000, gemiddeld -0,0122, beter/slechter/gelijk 12/14/4, Wilcoxon p = 0,4164; apneu-ΔF1 mediaan +0,0000
- bias (AHI − ref) gemiddeld old -5,14 → new -0,57 (mediaan -0,83 → -0,21); ernst gelijk aan ref old 21/30 → new 17/30; apneus old/new ratio mediaan 0,87 (bereik 0,02 … 1,33); stage_baseline_used: old False op 30/30, new True op 30/30
- per tertiel van de referentie-AHI (grenzen 17.3 / 29.4): laag n=10 ΔF1 mediaan -0,0016 beter/slechter 4/5; midden n=10 ΔF1 mediaan -0,0030 beter/slechter 3/6; hoog n=10 ΔF1 mediaan +0,0015 beter/slechter 5/3
- **Lezing (huisregel): MESA beslist niet**

- rekentijd per nacht (6 armen): mediaan 33,9 min, max 47,3 min; psgscoring ['0.37.0']
