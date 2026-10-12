# Hermeting reviewvelden — analyse-uitvoer

Automatisch gegenereerd door `analyse.py`; preregistratie `docs/reviewvelden_hermeting_preregistratie_20261012.md`.

## MESA run 1: `aasm_v3_breath_dual` (held-out nachten, referentie NSRR aasm15, matcher IoU 0,20 typeonbewust)

Opnames met resultaat: **30** (van 30 gevraagd).

### `aasm_v3_breath_dual`: elke arm tegen `nieuw` (ΔF1 = nieuw − arm)

| arm | n | ΔF1 mediaan | gemiddeld | beter/slechter/gelijk | p | bias nieuw → arm | ernst nieuw → arm | events nieuw → arm | oordeel |
|---|---|---|---|---|---|---|---|---|---|
| alles_oud | 30 | -0.0014 | -0.0036 | 13/16/1 | 0.795 | -0.53 → -0.44 | 18 → 18 | 4606 → 4625 | blijft default |
| oud_nadir | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_csr | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_plm | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_odi | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_spectraal | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_herstel | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_mixed | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_stadium | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| oud_fase | 30 | -0.0014 | -0.0036 | 13/16/1 | 0.795 | -0.53 → -0.44 | 18 → 18 | 4606 → 4625 | blijft default |
| oud_antialias | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft default |
| aan_masker | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft uit (MESA-deel NEE) |
| aan_lineariseer | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -0.53 → -0.53 | 18 → 18 | 4606 → 4606 | blijft uit (MESA-deel NEE) |

Let op het teken: de tabel vergelijkt `nieuw` met de arm; voor de `oud_*`-armen is een positieve ΔF1 winst van de reparatie, voor de `aan_*`-armen is een **negatieve** ΔF1 winst van het inschakelen.

### `aasm_v3_rec`: alles_oud tegen nieuw (in deze run)
n = 30: ΔF1 mediaan +0.0000 (gemiddeld +0.0022), beter/slechter/gelijk 14/10/6, p = 0.577; bias -6.00 → -5.84 /u; ernst 15 → 16; events 3600 → 3625. Oordeel: blijft default
Anker: `aasm_v3_rec` (harnasbasislijn) vs `aasm_v3_rec|nieuw` max |ΔF1| = 0.000000 over 30 nachten (hoort 0 te zijn).

### Netto-effect van de review per AHI-tertiel van de referentie (`aasm_v3_breath_dual`, nieuw − alles_oud)

| tertiel | n | ΔF1 mediaan | beter/slechter/gelijk | bias oud → nieuw | ernst oud → nieuw |
|---|---|---|---|---|---|
| laag | 10 | -0.0044 | 3/6/1 | +10.73 → +10.81 | 6 → 6 |
| midden | 10 | +0.0001 | 5/5/0 | -2.79 → -3.01 | 5 → 5 |
| hoog | 10 | +0.0013 | 5/5/0 | -9.26 → -9.38 | 7 → 7 |

### Apneu-subtypes (tellingen over alle nachten; nieuw → arm)

| arm | obstructief | centraal | gemengd | uncertain |
|---|---|---|---|---|
| alles_oud | 2114 → 2114 | 222 → 209 | 34 → 47 | 0 → 0 |
| oud_csr | 2114 → 2114 | 222 → 222 | 34 → 34 | 0 → 0 |
| oud_spectraal | 2114 → 2114 | 222 → 222 | 34 → 34 | 0 → 0 |
| oud_mixed | 2114 → 2114 | 222 → 209 | 34 → 47 | 0 → 0 |

## MESA run 2: `aasm_v3_rec` (held-out nachten, referentie NSRR aasm15, matcher IoU 0,20 typeonbewust)

Opnames met resultaat: **30** (van 30 gevraagd).

### `aasm_v3_rec`: elke arm tegen `nieuw` (ΔF1 = nieuw − arm)

| arm | n | ΔF1 mediaan | gemiddeld | beter/slechter/gelijk | p | bias nieuw → arm | ernst nieuw → arm | events nieuw → arm | oordeel |
|---|---|---|---|---|---|---|---|---|---|
| alles_oud | 30 | +0.0000 | +0.0022 | 14/10/6 | 0.577 | -5.84 → -6.00 | 16 → 15 | 3625 → 3600 | blijft default |
| oud_nadir | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -5.84 → -5.84 | 16 → 16 | 3625 → 3625 | blijft default |
| oud_csr | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -5.84 → -5.84 | 16 → 16 | 3625 → 3625 | blijft default |
| oud_spectraal | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -5.84 → -5.84 | 16 → 16 | 3625 → 3625 | blijft default |
| oud_herstel | 30 | +0.0000 | +0.0005 | 10/9/11 | 0.794 | -5.84 → -5.97 | 16 → 15 | 3625 → 3605 | blijft default |
| oud_mixed | 30 | +0.0000 | +0.0000 | 0/0/30 | — | -5.84 → -5.84 | 16 → 16 | 3625 → 3625 | blijft default |
| oud_fase | 30 | +0.0000 | +0.0017 | 11/11/8 | 1.000 | -5.84 → -5.80 | 16 → 16 | 3625 → 3635 | blijft default |
| aan_masker | 30 | -0.0754 | -0.1044 | 2/27/1 | 6.2e-06 | -5.84 → -0.37 | 16 → 17 | 3625 → 4626 | MESA-deel JA — PSG-IPA (binnen bereik) beslist mee |
| aan_lineariseer | 30 | +0.0000 | -0.0057 | 6/7/17 | 0.442 | -5.84 → -6.88 | 16 → 13 | 3625 → 3430 | blijft uit (MESA-deel NEE) |

Let op het teken: de tabel vergelijkt `nieuw` met de arm; voor de `oud_*`-armen is een positieve ΔF1 winst van de reparatie, voor de `aan_*`-armen is een **negatieve** ΔF1 winst van het inschakelen.

### `aasm_v3_rec`: alles_oud tegen nieuw (in deze run)
n = 30: ΔF1 mediaan +0.0000 (gemiddeld +0.0022), beter/slechter/gelijk 14/10/6, p = 0.577; bias -6.00 → -5.84 /u; ernst 15 → 16; events 3600 → 3625. Oordeel: blijft default
Anker: `aasm_v3_rec` (harnasbasislijn) vs `aasm_v3_rec|nieuw` max |ΔF1| = 0.000000 over 30 nachten (hoort 0 te zijn).

### Netto-effect van de review per AHI-tertiel van de referentie (`aasm_v3_rec`, nieuw − alles_oud)

| tertiel | n | ΔF1 mediaan | beter/slechter/gelijk | bias oud → nieuw | ernst oud → nieuw |
|---|---|---|---|---|---|
| laag | 10 | +0.0008 | 5/2/3 | +6.69 → +6.99 | 3 → 4 |
| midden | 10 | +0.0000 | 4/4/2 | -6.97 → -6.87 | 5 → 5 |
| hoog | 10 | +0.0011 | 5/4/1 | -17.73 → -17.64 | 7 → 7 |

### Apneu-subtypes (tellingen over alle nachten; nieuw → arm)

| arm | obstructief | centraal | gemengd | uncertain |
|---|---|---|---|---|
| alles_oud | 1512 → 1512 | 206 → 202 | 8 → 12 | 0 → 0 |
| oud_csr | 1512 → 1512 | 206 → 206 | 8 → 8 | 0 → 0 |
| oud_spectraal | 1512 → 1512 | 206 → 206 | 8 → 8 | 0 → 0 |
| oud_mixed | 1512 → 1512 | 206 → 202 | 8 → 12 | 0 → 0 |

## PSG-IPA (SN1–SN5, 12 scoorders, hypnogram scoorder 1; plafond respiratoire F1 0,667)

| arm | bias | MAE | binnen bereik | ernst | event-F1 mediaan (min–max over opnames) |
|---|---|---|---|---|---|
| aasm_v3_breath_dual|aan_lineariseer | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|aan_masker | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|alles_oud | +2.61 | 2.61 | 2/5 | 4/5 | 0.476 (0.320–0.894) |
| aasm_v3_breath_dual|nieuw | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_antialias | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_csr | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_fase | +2.61 | 2.61 | 2/5 | 4/5 | 0.476 (0.320–0.894) |
| aasm_v3_breath_dual|oud_herstel | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_mixed | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_nadir | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_odi | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_plm | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_spectraal | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_breath_dual|oud_stadium | +2.95 | 2.95 | 2/5 | 3/5 | 0.479 (0.299–0.893) |
| aasm_v3_rec|aan_lineariseer | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|aan_masker | +8.95 | 8.95 | 1/5 | 2/5 | 0.387 (0.296–0.849) |
| aasm_v3_rec|alles_oud | +1.69 | 1.76 | 3/5 | 4/5 | 0.349 (0.286–0.886) |
| aasm_v3_rec|nieuw | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_antialias | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_csr | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_fase | +1.71 | 1.71 | 3/5 | 4/5 | 0.355 (0.286–0.887) |
| aasm_v3_rec|oud_herstel | +1.51 | 1.51 | 3/5 | 4/5 | 0.327 (0.294–0.884) |
| aasm_v3_rec|oud_mixed | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_nadir | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_odi | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_plm | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_spectraal | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |
| aasm_v3_rec|oud_stadium | +1.51 | 1.51 | 3/5 | 4/5 | 0.318 (0.299–0.884) |

Regel 2 (default-uit-velden): `aan_masker`/`aan_lineariseer` mogen het aantal binnen het bereik niet verlagen t.o.v. `nieuw`.

