# Respiratoire 1D-U-Net (bench/resp) — eerste evaluatie volgens de preregistratie

Datum: 2026-10-07/08 (nacht). Preregistratie `docs/resp_unet_preregistratie_20261007.md`
(a46a228 → 9f24c48 formulering → 2be0f06 bevriezing, alle vóór de eerste evaluatie buiten
de validatienachten). Bench `bench/resp/` (README, data, model, postproc, train, eval_cohort,
baseline, analyse). Bevroren model sha256 39b1c095… (epoch 14 van 21, vroegtijdig gestopt
op MESA-val), kopie `/srv/CODE/docs/resp_unet_20261007/model_frozen_39b1c095.pt`; ruwe
uitvoer in diezelfde map (rows-JSON's per cohort, ids, consensus-diagnostiek). Werkpunt
τ = 0,20 (uit de checkpoint; MESA-val vlak 0,79 over τ 0,15–0,30).

**Status: STATUS_PLACEHOLDER**

## 1. Training (geen bewijs, keuzeset)
399 MESA-trainingsnachten / 100 validatienachten buiten de standaard-n150-set en buiten de
140 van de breath_dual-MESA-run (seed 20261007, geregistreerd), 4,58 M parameters, 21
epochs à ~20 s + ~65 s validatie op de A4000; tijdfractie apneu 2,0 % / hypopneu 7,1 %.
Gepoolde event-F1 op MESA-val (IoU 0,20, typeonbewust, slaappoort): epoch 0 al 0,751,
beste 0,795 (epoch 14). Variantie: seeds 20261008 en 20261009 op dezelfde nachten halen
0,794 (epoch 17, τ 0,25) en 0,793 (epoch 16, τ 0,30) — binnen 0,002 van de bevroren run; de
bewaker (elk > +0,05 tegen `breath_dual` op MESA-val) wordt in §2 afgerekend zodra de
baseline er is.

## 2. MESA-validatienachten (keuzeset, beschrijvend)
U-Net n=100: F1 mediaan 0,753 (p25 0,653), gepoold 0,795, count-ratio mediaan 1,02,
AHI-bias gemiddeld +0,10 /u. Gepaard tegen psgscoring `aasm_v3_breath_dual@0,50` en `rec`
op dezelfde nachten: MESAVAL_BASELINE_PLACEHOLDER

**Ablaties (MESA-val, kanaal op nul bij inferentie):**

| zonder | F1 mediaan | count-ratio | AHI-bias |
|---|---:|---:|---:|
| niets (volledig) | 0,753 | 1,02 | +0,10 |
| neusdruk | 0,741 | 0,98 | −0,58 |
| thermistor | 0,755 | 1,01 | −0,24 |
| effortbanden | 0,738 | 1,03 | −0,01 |
| **SpO2** | **0,394** | **0,29** | **−14,90** |

De kanaal-uitval-augmentatie werkt voor de flowsensoren (zonder druk of zonder thermistor
nauwelijks verlies — het model heeft de sensorarbitrage geleerd die de regelketen met een
poort en een vereniging doet), maar het model **leunt op de SpO2**: zonder desaturatie
vindt het minder dan een derde van de events. Dat is de referentie zelf (NSRR `aasm15`:
hypopneu = ≥ 3 % desaturatie óf arousal, en arousals ziet dit model niet) en dus de
bibliotheekregel voor een inbouw: zonder SpO2 terug naar de regelketen.

## 3. SHHS1, 150 verse nachten (beslissend; thermokoppel + RIP + SaO2, geen neusdruk)
Ids `/srv/CODE/docs/resp_unet_20261007/shhs1_ids.txt` (seed 20261007, geregistreerd in
`gebruikte_shhs_ids.txt`). U-Net: F1 mediaan **0,583** (p25 0,410), gepoold 0,650,
count-ratio mediaan 0,91, AHI-bias gemiddeld −1,37 /u; per NSRR-AHI-tertiel laag (AHI 4,3)
F1 0,384 / count-ratio 0,76 / bias −0,29, midden (12,9) 0,562 / 0,95 / −0,94, hoog (30,4)
0,694 / 0,93 / −2,88 — dezelfde ziektelasthelling als mensen en regels.
**Primaire regel (gepaard tegen `rec` én tegen `breath`, psgscoring op hetzelfde
NSRR-hypnogram, thermokoppel op de drukplaats zoals `SHHS-validation/score_shhs.py` en de
arousal-replicatie van 27-09):**

| | F1 mediaan | ΔF1 mediaan / gemiddeld | beter / slechter | Wilcoxon p | AHI-bias gemiddeld | ΔF1 per tertiel laag / midden / hoog |
|---|---:|---:|---:|---:|---:|---|
| U-Net | 0,583 | | | | −1,37 | |
| `aasm_v3_rec` | 0,081 | +0,372 / +0,406 | 146 / 0 | 1,0e-25 | −10,79 | +0,31 / +0,45 / +0,46 |
| `aasm_v3_breath` | 0,200 | +0,262 / +0,289 | 142 / 6 | 1,3e-24 | −9,55 | +0,21 / +0,34 / +0,32 |

Count-ratio mediaan 0,91 (in [0,80; 1,25]), |bias| 1,37 < 10,79, geen tertiel onder −0,02:
**alle vier de onderdelen van de primaire regel gehaald, tegen beide baselines.** Kanttekening
die de lezing kleurt: de regelketen is op dit cohort zelf zwak — een thermokoppel uit de
jaren negentig op de drukplaats haalt de apneugrens van 0,90 zelden (het dossier van
22-08: 13 % op een thermistor) en de hypopneeroute mist evenveel, vandaar AHI-bias −10
en F1 0,08–0,20. Dat is de werkelijke stand van de bibliotheek op SHHS1 (de paper-
reproductie daar gebruikte het dataset-profiel `mesa_shhs`); daarom staat hieronder
post-hoc ook `mesa_shhs` als derde baseline: MESASHHS_PLACEHOLDER

## 4. PSG-IPA SN1–5 (12 scoorders; neusdruk + RIP + SaO2, geen thermistor)
Scoorder-mediaan F1 (IoU 0,20) per nacht, náást het menselijk plafond
(`docs/respiratoir_menselijk_plafond_20261007.md`) en de psgscoring-baselines op hetzelfde
hypnogram:

| nacht | U-Net | `breath_dual` | `rec` | plafond | U-Net/plafond | AHI U-Net | AHI scoordermediaan |
|---|---:|---:|---:|---:|---:|---:|---:|
| SN1 | 0,486 | **0,678** | 0,470 | 0,826 | 0,59 | 8,8 | 6,0 |
| SN2 | **0,557** | 0,388 | 0,317 | 0,549 | 1,01 | 5,8 | 4,3 |
| SN3 | 0,882 | **0,900** | 0,886 | 0,948 | 0,93 | 45,6 | 54,0 |
| SN4 | **0,471** | 0,252 | 0,286 | 0,553 | 0,85 | 4,8 | 3,8 |
| SN5 | **0,580** | 0,476 | 0,349 | 0,556 | 1,04 | 12,0 | 10,0 |

- **Regel "op ≥ 4/5 nachten niet lager dan `breath_dual`": NIET gehaald (3/5).** Het model
  wint fors op de drie lichte nachten SN2/SN4/SN5 (+0,17 / +0,22 / +0,10; op SN2 en SN5
  zit het op of boven het menselijk plafond) en verliest op SN1 (−0,19) en nipt op SN3
  (−0,02).
- **SN1 ontleed** (consensus = events die ≥ 6 van 12 scoorders markeren, 34 stuks): het
  U-Net zet 51 events waarvan **22 door geen enkele scoorder** worden gezien (precisie
  0,41 tegen consensus, recall 0,62); `breath_dual` zet 31 events met 7 zonder steun
  (precisie 0,71, recall 0,65). Op SN3 is het omgekeerd: U-Net precisie 0,97 maar recall
  0,82 (276 tegen 326 consensus-events; AHI 45,6 tegen 54,0), `breath_dual` 0,91 / 0,90.
  Op SN5 wint het U-Net op beide (0,56/0,72 tegen 0,54/0,57). Het model telt op de
  lichte nachten te veel (AHI +1,5 tot +2,8 boven de scoordermediaan) en op de zware te
  weinig; de apneukop blijft op deze druk-alleen-montage vrijwel leeg (0 apneus op
  SN1/SN2/SN4, 1 op SN5, 91 op SN3) — alles wordt hypopneu, wat de typeonbewuste F1 niet
  raakt maar de rapportage wel zou raken.
- Duur: U-Net-events mediaan 18,8–30,1 s tegen scoorders 16,2–26,5 s (iets lang).

## 5. Bewakers
- CPU-inferentie (4 threads, PSG-IPA): voorwaarts 0,6–2,1 s, totaal 7–14 s per nacht
  (inclusief inlezen en resamplen) — ruim onder de 60 s. Identieke events als op de GPU.
- Variantie: zie §1. Ablaties: zie §2 (SpO2 is de kritische ingang).
- Niet gemeten: subtypering (het model levert alleen apneu/hypopneu), RERA, desaturatie-
  koppeling per event; menselijk plafond alleen op PSG-IPA.

## 6. Lezing
LEZING_PLACEHOLDER

## 7. Wat dit niet is
Geen bibliotheekcode, niets uitgerold, geen beslissing. De DUA-vraag (gewichten uit
MESA/SHHS verspreiden) is niet beantwoord. De vergelijking op PSG-IPA gebruikt het
hypnogram van de arousal-replicatie (scoorder 1) voor alle drie de systemen.
