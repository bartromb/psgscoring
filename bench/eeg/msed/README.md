# Kandidaat 2 — MSED, multimodale sleep-event-detector (voorgetraind op MrOS)

| | |
|---|---|
| Bron | Zahid AN, Jennum P, Mignot E, Sørensen HBD. *MSED: A Multi-Modal Sleep Event Detection Model for Clinical Sleep Analysis.* IEEE Trans Biomed Eng 70(9):2508–2518, 2023, doi:10.1109/TBME.2023.3252368 (arXiv 2101.02530); code+gewichten github.com/neergaard/msed. |
| Licentie | **MIT** (upstream/LICENSE, © 2024 A. N. Zahid) — bruikbaar. |
| Claim | gezamenlijke detectie van arousals, beenbewegingen en SDB-events op MrOS (1653 train / 1000 hold-out): arousal-F1 **0,70** (precisie 0,76, recall 0,67) bij hun eigen matching (IoU 0,5 in de trainer-config), `results_test.json`. |
| Doelvariabele | NSRR-arousals (MrOS), dus AASM-EEG-arousals — hetzelfde doel als PSG-IPA. |
| Invoer | 10 kanalen à 128 Hz: C3, C4, EOG-L, EOG-R, kin, been L/R, nasale druk, thorax, abdomen; bandpass 0,3–35 Hz (EEG/EOG), hoogdoorlaat 10 Hz (EMG), 0,03 Hz (druk), 0,1–15 Hz (banden); z-score per kanaal; vensters van 120 s met 50 % overlap. |
| Architectuur | SplitStreamNet: per eventtype een eigen convolutionele stroom (DOSED-achtig, SSD-stijl ankers van 3/15/30 s), samengevoegd door een bi-GRU + additieve aandacht; NMS 0,5; classificatiedrempel per klasse uit `results_eval.json` (arousal **0,64**, door upstream op hún evaluatieset geoptimaliseerd — hier ongewijzigd overgenomen, dus géén drempelkeuze op PSG-IPA). |

## Wat hier precies gedaan is

- `upstream/` (niet in git, zie `../.gitignore`): volledige MIT-repo zonder .git, opgehaald op
  2026-09-27 van github.com/neergaard/msed @ `main` = commit `9b6eedc59affb737a4ac49df888b2ae64ad7a209`
  (`git clone --depth 1 https://github.com/neergaard/msed.git upstream && rm -rf upstream/.git`),
  incl. `models/splitstream/weights.pth` (14 MB). **Upstream is niet gewijzigd.**
  Het pakket is
  als pakket in `../_venv` geïnstalleerd (`uv pip install --no-deps -e upstream`; upstream pint
  mne 1.8.0/numpy 2.1.1, hier draait mne 1.13.2/numpy 2.x zonder aanpassing aan hun code).
- `run_msed.py`: gebruikt upstream `process_file` (voorbewerking) en `initialize_model`
  ongewijzigd. Alleen de kanaalkeuze en de inferentielus staan hier, omdat upstream zijn
  `channel_map.json` naast de EDF's (dus onder /srv/DATA) zou schrijven en interactief vraagt.
  PSG-IPA heeft één centrale afleiding; MSED wil er twee (C3-A2, C4-A1):
  - **variant `dup` (primair, vooraf gekozen)**: C3 := C4-M1 (Cz-M1 op SN5), C4 := C4-M1;
  - variant `f4` (gevoeligheid): C3 := F4-M1.
  EOG-L/R := E1-M2/E2-M2, Chin := EMG chin, LegL/R := EMG LAT/RAT, NasalP := Resp nasal,
  Thor := Resp chest, Abdo := Resp abdomen. Daarna alleen de slaappoort van
  `../common/postproc.py` (events met onset in een W-epoch weg); MSED's eigen eventgrenzen
  blijven staan.
- **Gevonden in upstream `predict_events.py:149`** (ongewijzigd gelaten): `Detection.forward`
  levert klasse-index − 1 (arousal = 0, lm = 1, sdb = 2), maar `predict_events` schrijft de
  maskerrij `ev[-1] - 1`, waardoor arousals in de LAATSTE rij ("sdb") belanden en de
  "arousal"-CSV beenbewegingen bevat (mediane duur 1,8 s). De correctie zit uitsluitend in de
  bench-eigen inferentielus (`run_msed.py:73` en `run_msed_sweep.py:85`: `mask[ev[-1]]`),
  gecontroleerd op SN5:
  klasse-0-events vallen op de referentie-arousals (bv. 515–528 s vs 517–533 s) en hebben
  een mediane duur van ~12 s. De eerste run (vóór de correctie) staat als waarschuwing in
  het rapport; de gerapporteerde cijfers zijn van ná de correctie.

Afhankelijkheden: `requirements.txt` (torch, mne, einops, xmltodict, rich, pandas, h5py).
