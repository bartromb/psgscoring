# Kandidaat 3 — U-Net op 50 Hz (Ehrlich e.a. 2024), schone herimplementatie, getraind op ongebruikte MESA-nachten

| | |
|---|---|
| Bron | Ehrlich J, Sehr J, Brandt A, et al. *State-of-the-art sleep arousal detection evaluated on a comprehensive clinical dataset.* Sci Rep 14, 2024, doi:10.1038/s41598-024-67022-9 (PMC11247076). Upstream code+gewichten: gitlab.com/sleep-is-all-you-need/arousaldetector. |
| Licentie upstream | **GNU GPL → NIET-PORTEERBAAR** voor deze BSD-3-bibliotheek; er is geen regel code of gewicht van hen gebruikt. Alles hier is vanaf de publicatie geschreven (`model.py`, `data.py`, `train.py`, `infer_psgipa.py`) en dus BSD-3. |
| Claim | U-Net (8 dubbele convoluties, kernel 21) op 3 kanalen à 50 Hz (EEG, EOG, kin-EMG), getraind op 3423 DSDS-PSG's + SHHS1 1698 + MESA 616 + MrOS 815; AUPRC 0,82 / **F1 0,81 op MESA** (elke overlap = TP; niet vergelijkbaar met IoU 0,20), 0,83/0,80 SHHS, 0,71/0,74 DSDS. |
| Doelvariabele | NSRR-/klinische EEG-arousals — hetzelfde doel als PSG-IPA. |

## Wat hier precies gedaan is (en waar het afwijkt van het artikel)

- **Data**: 400 MESA-nachten die in `/srv/DATA/MESA/gebruikte_mesa_ids.txt` in GEEN
  eerdere sectie stonden (476 beschikbaar), geschud met seed 20260927; 320 train
  (`ids_train.txt`), 80 validatie (`ids_val.txt`); bijgeschreven in het register onder
  `## dsp-scout eeg 2026-09-27 (400)`. Vier nachten vielen af (0 NSRR-arousals: mesa-sleep-4854,
  -5623, -5721, -6476; zie `train_log.json`): 317 train / 79 validatie. Signalen alleen in het geheugen; niets van MESA is naar bench/ gekopieerd.
- **Voorbewerking** (artikel): anti-alias + decimatie naar 50 Hz (`mne.resample`), per kanaal
  gemiddelde en RMS over een lopend venster van 18 min verwijderd, clip ±20.
  MESA: EEG3 (C4-M1) / EOG-L / EMG; PSG-IPA: C4-M1 (Cz-M1 op SN5) / E1-M2 / EMG chin.
- **Augmentatie** (artikel): willekeurig kanaal per signaaltype per iteratie (EEG3 50 %,
  EEG1 Fz-Cz 25 %, EEG2 Cz-Oz 25 %; EOG-L/R 50/50), amplitudefactor U(0,8; 1,3) per kanaal.
- **Model**: `model.py` — encoder 16-32-64-128-256, pooling (2,4,4,4), kernel 21, BN+ReLU,
  lineaire upsampling met skip-concatenatie, 4,58 M parameters. Het artikel geeft geen
  kanaalaantallen/poolingfactoren; dit is een eigen keuze binnen hun beschrijving.
- **Training**: BCE-with-logits (artikel), AdamW lr 1e-3 met warm-up en cosinus (artikel:
  cyclische LR 0,003–0,00075; hier niet gereproduceerd), vensters van 10 min rond de
  slaapperiode, batch 32, 600 stappen per epoch, ≤ 30 epochs, bf16-autocast, RTX A4000.
- **Afwijkingen, bewust**: (1) géén labelverlenging −2 s/+10 s (het artikel doet dat voor
  autonome respons; onze maat is IoU tegen EEG-arousalgrenzen, dus dat zou de matching
  schaden); (2) werkpunt en vroegtijdig stoppen op de **gepoolde event-F1 (IoU 0,20) op de
  MESA-validatienachten**, niet op AUPRC, omdat dat de doelmaat van dit project is;
  (3) nabewerking = `../common/postproc.py` (1 s-gemiddelde, drempel, gaten ≤ 1 s, ≥ 3 s,
  slaappoort) i.p.v. hun "patience 10 s"; (4) ~320 nachten training tegen 6552 bij hen.
- **Uitkomst training**: 23 epochs (vroegtijdig gestopt), beste epoch 16 met gepoolde
  validatie-F1 0,682 op drempel **0,20**; ~2,3 min per epoch op de A4000, 5 min laden.
  Eén trainingsrun, seed 20260927 — de variantie tussen runs is niet gemeten.
- **Inferentie** (`infer_psgipa.py`): hele nacht in één pas; primair werkpunt = de op MESA
  gekozen drempel 0,20 (`model_best.pt["thr"]`), drempelraster 0,15–0,85 als orakelcurve.
  CPU-kost: 1,7 s voorbewerken + 2,1 s inferentie (8 threads) voor een nacht van 8,4 u.
- `infer_mesa_val.py`: dezelfde detector op de 79 validatienachten (`out_mesa/`), voor de
  gepaarde MESA-tabel in `../results/mesa_val.md`.
- Op dezelfde 80 validatienachten draait `../common/run_baseline_mesa.py` psgscoring
  0.34.2, zodat U-Net en huidige detector op MESA tegen dezelfde NSRR-referentie staan.

`model_best.pt` (18 MB) staat niet in git (`../.gitignore`); opnieuw maken met
`OMP_NUM_THREADS=1 ../_venv/bin/python train.py --n-train 320 --n-val 80 --epochs 30 --steps 600 --workers 8`
(zelfde seed 20260927 en dezelfde id-lijsten; de registerregel wordt niet dubbel geschreven).

Afhankelijkheden: `requirements.txt` (torch, mne, scipy, scikit-learn, numpy).
