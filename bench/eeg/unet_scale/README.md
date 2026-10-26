# Schaalmeting arousal-U-Net — M / MS / MSF tegen het bevroren model (25-10-2026)

Preregistratie: `docs/arousal_unet_schaal_preregistratie_20261025.md` (geschreven vóór de
start; sha256 in `chain.log`). Vraag: brengt méér trainingsdata (1278 MESA i.p.v. 317), een
tweede cohort (1278 SHHS1) en een flowkanaal als vierde ingang de event-F1 omhoog op een
nooit gezien SHHS1-cohort?

| bestand | rol |
|---|---|
| `select_ids.py` | trekt de sets (seed 20261025) uit de pools van de preregistratie en schrijft de registerregels (idempotent) |
| `ids_mesa_train.txt` (1278), `ids_mesa_val.txt` (40), `ids_shhs_train.txt` (1278), `ids_shhs_val.txt` (40), `ids_shhs_test.txt` (150, SHHS1-150b), `ids_mesa76_test.txt` (76) | de sets |
| `data_scale.py` | laden MESA/SHHS1/PSG-IPA met alle kandidaatkanalen (EEG-varianten, EOG L/R, EMG, flow druk/thermistor) en een indexkaart; voorbewerking uit `../unet50/data.py` |
| `train_scale.py` | de keten: laden → bevroren τ op val-80 → arm M → MS → MSF → inferentie op de drie testcohorten; hervatbaar per arm (`KLAAR`) |
| `run_chain.sh` | start de keten onder `setsid` met `thermal_pauze.sh` (78/66 °C) op de procesgroep |

Uitvoer buiten git in `/srv/CODE/docs/arousal_unet_schaal_20261025/`: `chain.log`,
`thermal.log`, `laadfouten.json`, per arm `M/ MS/ MSF/` met `model_best.pt`, `train_log.json`,
`tau.json`, `KLAAR`; `frozen/tau.json`; `out/<shhs150b|mesa76|psgipa>/` met per nacht
`_ref.csv`, `_hypno.json` en per model `_prob10hz.npz`, `_pred.csv` (val-τ, zonder
10 s-samenvoeging), `_grid.json` (events op τ 0,15 … 0,85); `infer_log.jsonl`; marker
`ALLES_KLAAR`.

De evaluatie staat in `evaluate_scale.py` (projectmatcher IoU 0,20 én 'elke overlap', 10 s-samenvoeging
uit de bibliotheek, Wilcoxon, tertielen, REM/NREM, de beslisregel uit de preregistratie als JSON; schrijft
`evaluatie.json` + `evaluatie.md` in de uitvoermap). Ze gebruikt `psgscoring` en draait pas als de
bibliotheek niet meer in bewerking is én `ALLES_KLAAR` er staat; op 25-10 rookgetest op een synthetische
fixture in dezelfde lay-out. De keten zelf importeert de bibliotheek niet.

Rookproef (3+2 / 3+2 nachten, 1 epoch van 3 stappen) draaide op 25-10 20:36 foutloos door
alle stadia; de echte run is gestart met `run_chain.sh` (zie `chain.log`).

## Uitslag 26-10 en export

Uitslag in `docs/arousal_unet_schaal_replicatie_20261026.md` (MS haalt de regel; flow niet).
`export_onnx.py --arm MS` exporteert `MS/model_best.pt` naar `MS/arousal_unet_MS_kandidaat.onnx`
(+ `.json` met sha256's, τ en de afwijking tegen torch) met de conventies van
`psgscoring/data/arousal_unet_v1.json` (opset 17, sigmoid in de graaf, n deelbaar door 128).
Kandidaat, niet in de bibliotheek.
