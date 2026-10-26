# Arousal-U-Net en LGBM-keten onder twee koppelregels, met 1 s-κ en grensverschuiving

Datum: 2026-10-25. Aanleiding: de literatuurcheck van dezelfde dag
(`docs/literatuur_update_20261025.md` §1): Pitkänen e.a. (Sleep 2026) en Ehrlich e.a. (Sci Rep
2024) rapporteren event-F1 met **elke overlap** als treffer, plus een seconde-per-seconde κ en
de verschuiving van begin en einde. Onze replicatie van 27-09 koppelt op **IoU 0,20**. Zonder
dezelfde regel is "0,745 tegen 0,61" geen vergelijking. Dit is een **herscoring van bewaarde
eventlijsten**: geen EDF gelezen, geen model gedraaid, niets afgesteld.

## Wat gelezen en hoe gekoppeld

- Voorspellingen en referenties van de U-Net-replicatie, ongewijzigd:
  `/srv/CODE/docs/arousal_unet_20260927/out/{psgipa,shhs,mesa}/{rec}_{ref,base,unet}.csv` en
  `{rec}_hypno.json` (hypnogram, opnameduur, TST). `base` = de LGBM-productieketen van 27-09,
  `unet` = het bevroren model (sha256 cc91bb86…, τ 0,35, samenvoeging 10 s). SHHS1: 147 van 150
  nachten (drie zonder referentie-arousals, zoals in de replicatie). MESA: 76. PSG-IPA: SN1–5
  tegen de gedeelde EEG-arousalannotatie van de `Resp_events`-subboom (`bench/eeg/ref/`).
- Scoorderplafond: `/srv/DATA/PSG-IPA/EEG_arousals/Annotations/manual/SNx_EEGarousals_manual_scorer{1..12}.txt`,
  alle 66 paren per nacht, 330 paren; opnameduur uit de EDF-kop; duur 0 → 3 s (zelfde lezer als
  `scripts/sweep_event_locked_window._psgipa_spans`). Dit is de bron van
  `docs/arousal_menselijk_plafond.md` (25-08-2026).
- Koppeling met `psgscoring.agreement.compare_event_sets` (bibliotheek, vandaag uitgebreid; niet
  nagebouwd): `matcher="iou"` (IoU ≥ 0,20, symmetrisch greedy, hoogste IoU eerst) en
  `matcher="any"` (overlap > 0 s, grootste overlap eerst). F1 = 2·gepaard / (n_ref + n_pred).
  `kappa_1s` = Cohen's κ op 1 s-maskers over de hele opname; "κ slaap" = hetzelfde, beperkt tot de
  seconden in N1/N2/N3/R van het hypnogram (bij het plafond: meerderheidshypnogram van de twaalf
  scoorders). Verschuiving = voorspelling − referentie, mediaan over de gekoppelde paren (onder
  "any"); positief = de voorspelling begint/eindigt later.
- Controle: de IoU-0,20-cijfers van de bibliotheekmatcher zijn op alle drie de cohorten gelijk aan
  die van de replicatie (`eval.json`, `project`-matcher = `validate_psgipa` legacy): mediaan 0,598 /
  0,709 (PSG-IPA), 0,525 / 0,667 (SHHS1), 0,548 / 0,670 (MESA); gepoold 0,555 / 0,745, 0,543 / 0,676,
  0,556 / 0,687. Het plafond 0,679 en de vijf per-nacht-medianen (0,642 / 0,492 / 0,692 / 0,767 /
  0,766) reproduceren exact.
- Scripts en ruwe uitvoer: `docs/arousal_unet_matcher_vergelijking_20261025/{herscoor.py,
  plafond.py, herscoor.json, plafond.json}`. Geen PHI (NSRR-id's, PSG-IPA-nummers).

## 1. Detector tegen referentie, per cohort

Mediaan per nacht (gepoold tussen haakjes). P/R = gepoolde precisie/recall.

| cohort | arm | F1 IoU 0,20 | F1 elke overlap | P / R (elke overlap) | κ 1 s | κ slaap | begin / einde (s) | duur pred vs ref (s) | count-ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSG-IPA n=5 | lgbm | 0,598 (0,555) | 0,605 (0,591) | 0,57 / 0,61 | 0,393 | 0,389 | +0,05 / +2,90 | 9,7 vs 7,8 | 1,11 |
| | **U-Net** | **0,709 (0,745)** | **0,709 (0,760)** | 0,76 / 0,76 | **0,588** | 0,628 | +0,24 / +1,14 | 9,6 vs 7,8 | 1,02 |
| SHHS1 n=147 | lgbm | 0,525 (0,543) | 0,542 (0,562) | 0,81 / 0,43 | 0,393 | 0,471 | +0,30 / +1,70 | 9,5 vs 8,8 | 0,51 |
| | **U-Net** | **0,667 (0,676)** | **0,682 (0,684)** | 0,75 / 0,63 | **0,549** | 0,628 | +0,16 / +0,52 | 8,8 vs 8,8 | 0,85 |
| MESA n=76 | lgbm | 0,548 (0,556) | 0,588 (0,608) | 0,64 / 0,58 | 0,363 | 0,459 | +0,08 / +1,20 | 8,5 vs 9,1 | 0,92 |
| | **U-Net** | **0,670 (0,687)** | **0,692 (0,702)** | 0,83 / 0,61 | **0,540** | 0,670 | −0,14 / +0,08 | 9,1 vs 9,1 | 0,72 |

Gepaard verschil U-Net − lgbm per nacht (mediaan; beter/slechter; Wilcoxon):

| cohort | ΔF1 IoU | ΔF1 elke overlap | Δκ 1 s |
|---|---|---|---|
| PSG-IPA | +0,136; 5/0; p = 0,0625 | +0,115; 5/0; p = 0,0625 | +0,116; 5/0; p = 0,0625 |
| SHHS1 | +0,125; 134/12; p = 2,0e-22 | +0,116; 130/16; p = 2,2e-21 | +0,151; 139/8; p = 4,3e-24 |
| MESA | +0,124; 72/4; p = 1,3e-13 | +0,093; 69/7; p = 2,1e-12 | +0,192; 74/2; p = 4,4e-14 |

## 2. Het scoorderplafond op PSG-IPA onder dezelfde maten

| nacht | arousals (mediaan per scoorder) | duur (mediaan) | F1 IoU 0,20 | F1 elke overlap | κ 1 s | κ slaap |
|---|---:|---:|---:|---:|---:|---:|
| SN1 | 121 | 7,6 s | 0,642 | 0,656 | 0,540 | 0,536 |
| SN2 | 46 | 7,9 s | 0,492 | 0,506 | 0,479 | 0,494 |
| SN3 | 142 | 8,3 s | 0,692 | 0,692 | 0,599 | 0,610 |
| SN4 | 104 | 10,3 s | 0,767 | 0,767 | 0,676 | 0,697 |
| SN5 | 202 | 9,1 s | 0,766 | 0,778 | 0,644 | 0,649 |
| **alle 330 paren** | | | **0,679** (0,192–0,869) | **0,692** (0,272–0,871) | **0,587** | 0,589 |

Verschuiving tussen scoorders (|mediaan| per paar, elke overlap): begin 0,33 s, einde 0,77 s.

Per nacht, U-Net (tegen de `Resp_events`-referentie) naast het plafond (`EEG_arousals`, andere
export — zie beperkingen):

| nacht | U-Net IoU | plafond IoU | U-Net elke overlap | plafond elke overlap | U-Net κ | plafond κ |
|---|---:|---:|---:|---:|---:|---:|
| SN1 | 0,709 | 0,642 | 0,709 | 0,656 | 0,588 | 0,540 |
| SN2 | 0,678 | 0,492 | 0,678 | 0,506 | 0,567 | 0,479 |
| SN3 | 0,883 | 0,692 | 0,903 | 0,692 | 0,705 | 0,599 |
| SN4 | 0,471 | 0,767 | 0,510 | 0,767 | 0,350 | 0,676 |
| SN5 | 0,713 | 0,766 | 0,720 | 0,778 | 0,625 | 0,644 |

## 3. Naast de literatuur

| | koppelregel | model | scoorders onderling | κ model | κ scoorders | duur model vs mens | begin / einde |
|---|---|---:|---:|---:|---:|---:|---:|
| Pitkänen 2026, Sleep Revolution 50 PSG × 10 scoorders | elke overlap | 0,61 | 0,57 | 0,50 | 0,46 | 9,0 vs 6,9 s | −1,3 / +0,9 s |
| Pitkänen 2026, MESA-test 205 (in-distributie, 1847 trainingsnachten) | elke overlap | 0,77 | — | 0,67 | — | | |
| Ehrlich 2024, MESA-test | elke overlap | 0,81 | — | — | — | | |
| **wij, PSG-IPA 5 × 12 scoorders** | elke overlap | **0,709** (gepoold 0,760) | **0,692** | **0,588** | **0,587** | 9,6 vs 7,8 s | +0,24 / +1,14 s |
| wij, PSG-IPA | IoU 0,20 | 0,709 (0,745) | 0,679 | | | | |
| wij, SHHS1 147 (extern, nooit gezien) | elke overlap | 0,682 (0,684) | geen plafond | 0,549 | — | 8,8 vs 8,8 s | +0,16 / +0,52 s |
| wij, MESA 76 (317 trainingsnachten) | elke overlap | 0,692 (0,702) | geen plafond | 0,540 | — | 9,1 vs 9,1 s | −0,14 / +0,08 s |

## 4. Lezing

1. **De koppelregel verplaatst de U-Net-cijfers nauwelijks** (≤ +0,02 F1 mediaan op elk cohort):
   zijn events liggen waar de referentie ligt. De LGBM-keten wint méér bij "elke overlap"
   (+0,02 tot +0,05): een deel van haar fouten is lokalisatie, zichtbaar in haar late einde
   (+1,2 tot +2,9 s) en langere events. Het verschil U-Net − lgbm blijft onder beide regels
   overeind, met dezelfde p-waarden-orde.
2. **Ook het plafond beweegt weinig** (0,679 → 0,692). Scoorders zijn het oneens over het
   *bestaan* van arousals, niet over hun grenzen (onderlinge verschuiving 0,3 / 0,8 s). De regel
   "IoU 0,20" was dus wel strenger, maar niet veel strenger; de vergelijking met de literatuur
   mag nu op dezelfde regel.
3. **1 s-κ**: het U-Net haalt over de hele nacht κ 0,54–0,59, op PSG-IPA gelijk aan de
   scoorders onderling (0,587); binnen de slaap 0,63–0,67. Pitkänen's model haalt 0,50 tegen
   scoorders 0,46 op hun 50-PSG-set en 0,67 in-distributie op MESA. Onze MESA-κ (0,54 heel,
   0,67 slaap) staat daar met 317 trainingsnachten tegenover hun 1847 — argument voor de
   schaalarm (roadmap punt 1).
4. **Grenzen**: het U-Net eindigt 0,1–1,1 s later dan de referentie en begint ±0,2 s; Pitkänen
   meldt −1,3 / +0,9 s en events een derde langer dan de mens. Onze duur is gelijk aan de
   referentie op SHHS1 en MESA (8,8 / 9,1 s) en langer op PSG-IPA (9,6 vs 7,8 s).
5. **Per nacht blijft het beeld van 27-09**: boven het plafond op SN1–SN3 onder elke maat,
   eronder op SN4 (0,51 vs 0,77) en SN5. SN4 is de nacht waar ook de LGBM-keten het laagst
   stond (0,38).

## 5. Voor paper v45

De rij "arousal-F1 0,745 tegen plafond 0,679 (IoU 0,20)" krijgt ernaast: "0,760 tegen 0,692
met elke overlap als treffer; 1 s-κ 0,59 tegen 0,59; verschuiving +0,2 / +1,1 s" — en de
koppelregel staat voortaan bij elk cijfer. De literatuurrij (Pitkänen 0,61 / 0,57, κ 0,50 /
0,46) kan dan in dezelfde tabel, met de kanttekening dat hun 50 PSG's uit zeven centra komen
en onze vijf uit één dataset.

## 6. Beperkingen

- PSG-IPA: onze detectorcijfers staan tegen de `Resp_events`-export (scoorder-1-annotatie, op
  tien van twaalf kopieën identiek), het plafond tegen de `EEG_arousals`-export (12 unieke
  scoringen, andere opnameduur: SN3 8,13 u tegen 9,7 u referentieduur hier). Zelfde vijf
  nachten, niet hetzelfde bestand; de per-nacht-tabel in §2 is een richtpunt, geen gepaarde meting.
- n = 5 voor het plafond; SHHS1 en MESA hebben één NSRR-scoorder en dus geen plafond.
- κ over de hele nacht hangt af van de slaapfractie (wake telt als overeenstemming "geen
  arousal"); daarom staat κ-slaap ernaast. Of Pitkänen κ over de hele opname of over de slaap
  berekent, staat niet in de samenvatting die wij lazen — hun 0,50 / 0,46 hoort bij "heel"
  of "slaap" en dat maakt hier ~0,05 uit.
- Dit is een herscoring: τ 0,35, samenvoeging 10 s en slaappoort zijn die van 27-09; er is
  niets opnieuw gekozen.
