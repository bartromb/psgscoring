# Complementariteit psgscoring × MSED op PSG-IPA (referentie-arousals, IoU 0,20)

| opname | n_ref | beide | alleen psgscoring | alleen MSED | geen | F1 MSED↔psgscoring | mediane duur ref / psgscoring / MSED (s) | mediane onsetfout psgscoring / MSED (s) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| SN1 | 37 | 15 | 10 | 6 | 6 | 0,517 | 7,8 / 9,7 / 12,1 | +0,0 / -1,1 |
| SN2 | 96 | 39 | 22 | 9 | 26 | 0,493 | 6,5 / 9,2 / 11,4 | -0,2 / +0,1 |
| SN3 | 241 | 109 | 30 | 61 | 41 | 0,576 | 9,7 / 7,2 / 11,2 | +0,4 / -0,1 |
| SN4 | 58 | 23 | 11 | 5 | 19 | 0,413 | 6,5 / 10,5 / 14,0 | +0,6 / -0,1 |
| SN5 | 164 | 59 | 25 | 16 | 64 | 0,584 | 8,5 / 11,4 / 12,0 | -0,1 / +0,6 |
| **totaal** | 596 | 245 | 98 | 97 | 156 | | | |

Dekking van de referentie: psgscoring 0,576, MSED 0,574, unie 0,738 (bovengrens van een ensemble-recall, zonder rekening te houden met de fout-positieven van beide),
