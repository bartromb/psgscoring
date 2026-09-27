# Kandidaat 1 — DeepSleep 2.0 (voorgetraind, PhysioNet/CinC 2018)

| | |
|---|---|
| Bron | Fonod R. *DeepSleep 2.0: Automated Sleep Arousal Segmentation via Deep Learning.* AI 3(1):164–179, 2022, doi:10.3390/ai3010010; code+gewichten github.com/rfonod/deepsleep2 (Zenodo 10.5281/zenodo.13964322, 2024). Compacte opvolger van DeepSleep (Li & Guan, Commun Biol 2020), de winnaar van de PhysioNet-2018-challenge en daarmee de referentiestandaard voor arousal-segmentatie — de reden dat een model van 2022 hier toch meedoet. |
| Licentie | **MIT** (upstream/LICENSE) — bruikbaar voor een BSD-3-bibliotheek. |
| Claim | gross AUPRC 0,450 / AUROC 0,901 op 249 held-out PhysioNet-2018-opnames (per-sample; het originele DeepSleep-ensemble 0,55). |
| Doelvariabele | PhysioNet-2018-"target arousals": **niet-apneu-arousals** (RERA-spannen inbegrepen); arousals bij apneus/hypopneus zijn in de training gemaskeerd (label −1). Dat is NIET de AASM-EEG-arousal die PSG-IPA annoteert. |
| Invoer | 13 kanalen à 200 Hz: F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2, Chin, ABD, Chest, Airflow, SaO2, ECG; z-score per kanaal over de hele opname; gecentreerd nul-gevuld tot 2^23 samples. |
| Architectuur | 1D-U-Net, 5 niveaus, pooling 4/8/16/32, kernel 7, 740 551 parameters; één sigmoïde per sample. |

## Wat hier precies gedaan is

- `upstream/` (niet in git, zie `../.gitignore`): alleen wat nodig is uit de MIT-repo, opgehaald
  op 2026-09-27 van github.com/rfonod/deepsleep2 @ `main` = commit `9d27032adc57e3043f5cad5e687ca229024f111a`
  (Zenodo-archief v1.0.0, doi:10.5281/zenodo.13964322); ophalen:
  `for f in LICENSE README.md architectures/architecture_v1.py utils.py losses.py score2018.py CITATION.cff
  models/model_2/hyperparameters.txt models/model_2/my_checkpoint_21.pth.tar models/model_2/records.csv; do
  curl -sfL -o upstream/$f https://raw.githubusercontent.com/rfonod/deepsleep2/9d27032/$f; done` (mappen vooraf aanmaken).
  Upstream is niet gewijzigd — `architectures/architecture_v1.py`,
  `utils.py`, `losses.py`, `score2018.py`, `LICENSE`, `README.md`, en van `models/model_2`
  (beste configuratie, Z-norm + MagScale + RandShuffle) het beste checkpoint
  `my_checkpoint_21.pth.tar` (epoch 21, laagste validatieverlies).
- `run_deepsleep2.py`: PSG-IPA-Resp_events-EDF → 13 slots; de ontbrekende linker
  homologen (F3-M2, C3-M2, O1-M2) worden gevuld met het rechter kanaal (het model is met
  RandShuffle over de zes EEG-kanalen getraind); Airflow := nasale druk. Inferentie op de
  GPU over de hele nacht in één voorwaartse pas (7–9 s per opname). Kans per sample →
  `../common/postproc.prob_to_events` (1 s-gemiddelde, drempel, gaten ≤ 1 s dicht, ≥ 3 s)
  → slaappoort (scoorder-1-hypnogram). Primair werkpunt **0,50** (vooraf gekozen; het model
  levert geen eigen drempel), drempelraster 0,10–0,80 als orakelcurve.
- `calibrate_mesa.py`: eerlijke ijking ZONDER PSG-IPA — op de **59** bruikbare van de 80
  MESA-validatienachten van `../unet50/ids_val.txt` (MESA-kanalen: EEG1=Fz-Cz, EEG2=Cz-Oz,
  EEG3=C4-M1, EOG-L, EMG, Abdo, Thor, Flow, SpO2, EKG; 21 nachten langer dan 2^23 samples
  bij 200 Hz = 11,65 u overgeslagen — ids in `calibratie_mesa.json`): beste drempel op gepoolde event-F1 (IoU 0,20) en de mediane onset-/
  offsetverschuiving van overlappende paren, daarna ongewijzigd toegepast op PSG-IPA
  (`out_mesacal_raw/`, `out_mesacal_corr/`; `calibratie_mesa.json`).
- Sanity-check (bench/eeg/RAPPORT.md §DeepSleep2): sample-AUROC 0,71–0,82 op SN1/2/4/5,
  0,47 op SN3; de kans piekt systematisch **~15 s vóór** de menselijke EEG-arousal
  (kruiscorrelatie-lag −14…−22 s op alle vijf) — consistent met RERA-spannen als doel.

Afhankelijkheden: `requirements.txt` (torch, mne, numpy) in `../_venv` (aparte uv-venv,
niets in de projectomgeving geïnstalleerd).
