# Baseline — psgscoring 0.34.2 zoals geïnstalleerd (`aasm_v3_rec`)

**Wat draait hier:** `psgscoring.run_pneumo_analysis(raw, hypno, scoring_profile="aasm_v3_rec")`
op de vijf PSG-IPA-opnames (`/srv/DATA/PSG-IPA/Resp_events/PSG/SNx_Respiration.edf`,
256 Hz, 13 kanalen) met het hypnogram van scoorder 1
(`validate_psgipa.parse_scorer_file`). De arousals komen uit
`result["arousal"]["events"]` (`onset_s`, `duration_s`), ongewijzigd.
Dat is de productiestand: multi-derivatie-union (F4-M1 ∪ C4-M1/Cz-M1 ∪ O2-M1)
→ LightGBM-classifier `arousal_classifier_v3` op werkpunt 0,70 →
10 s-minimuminterval → onset-offset +2,0 s; de autonome re-ranker staat op deze
montage uit (geen Pleth/HR). Geen parameter is aangeraakt; geen aangepaste kopie.

Bron/licentie: dit project (BSD-3). Looptijd 92–146 s per opname (1 thread).

- `../common/run_baseline.py SNx` → `SNx_pred.csv` (onset_s, offset_s, type) + `SNx_summary.json`
- `../common/export_ref.py` → `../ref/SNx_ref_arousal.csv` (de vaste EEG-arousalannotatie
  uit het scoorder-1-bestand van `Resp_events`; 37/96/241/58/164 arousals) en `SNx_hypno.json`
- `../common/run_baseline_mesa.py <rec>` → dezelfde aanroep op de 80 MESA-validatienachten
  van `../unet50/ids_val.txt` (`../baseline_mesa/`), zodat de eigen U-Net op MESA tegen
  dezelfde detector en dezelfde NSRR-referentie staat.

Cijfers: `../results/tabel.md`, `../RAPPORT.md`.
