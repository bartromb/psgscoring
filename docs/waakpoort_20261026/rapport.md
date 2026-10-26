# Waakpoort-meting — 100 nachten (fouten 0), profiel aasm_v3_breath_dual, kanalen zonder EEG/EOG/EMG

| arm | AHI med | F1 med (aasm15) | events | apneus | events in NSRR-W | ernst = ref | \|Δ\| vs O med | \|Δ\| ≥ 5 | ernst = O | s/nacht |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 22,80 | 0,250 | 25593 | 18429 | 14217 | 48 | 7,35 | 65 | 52 | 181 |
| V | 16,85 | 0,402 | 17074 | 9947 | 5734 | 58 | 2,40 | 30 | 78 | 177 |
| O | 15,45 | 0,491 | 11510 | 6839 | 0 | 62 | — | — | — | 171 |

Wilcoxon |Δ_T| − |Δ_V|: mediaan 2,50, p = 6.1e-10. O-events buiten V: 55 van 11510 (0,48 %). NSRR-slaapminuten weggeknipt: 71 totaal, > 5 min op 3 nachten.
Uitgesloten minuten mediaan: vóór 0,0, na 55,8, binnen 0,0, totaal 78,5; venstermethode {'signal_validity_window': 100}.
Per TIB/TST-tertiel (grenzen 1,47 / 1,83): laag: T |Δ| 5,20 (≥5: 17) → V 2,10 (≥5: 5); midden: T |Δ| 7,40 (≥5: 22) → V 2,20 (≥5: 10); hoog: T |Δ| 12,00 (≥5: 26) → V 4,50 (≥5: 15)

Beslisregel: 1_ge5_halveert: JA, 2_mediaan_lager_p<0.01: JA, 3_orakelverlies_le_0.5pct: JA, 4_ernst_gelijk_O_plus10: JA, alle: JA
