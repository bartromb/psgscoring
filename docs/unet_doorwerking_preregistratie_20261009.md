# Preregistratie — doorwerking van `unet_v1` op `aasm_v3_breath_dual` (AHI, RDI, koppeling)

Datum: 2026-10-09 22:55, **vóór de meting.** Vervolg op `docs/arousal_unet_replicatie_20260927.md`
(§"niet gemeten": RERA/RDI en hypopneu-koppeling) en de inbouw van `arousal_detector="unet_v1"`
(cfc90c1, opt-in). Vraag: wat verandert er aan de respiratoire uitkomst van het
productieprofiel als de arousals van het U-Net komen in plaats van de LGBM-keten?

## Opzet
- Bibliotheek bevroren op cfd75fc (main); env-override `PSGSCORING_AROUSAL_DETECTOR=unet_v1`
  tegen de default (lgbm), verder identiek (profiel `aasm_v3_breath_dual`, τ 0,35).
- **MESA-val, 100 nachten** (`bench/resp/ids_val.txt`; neusdruk + thermistor + EEG/EOG/EMG):
  `scripts/validate_mesa.py --recordings … --profiles aasm_v3_breath_dual`, NSRR-hypnogram,
  artefact-epochs leeg; twee runs (lgbm / unet). Referentie `aasm15`, matcher IoU 0,20.
- **PSG-IPA SN1–5**: `run_pneumo_analysis` per nacht met het hypnogram van de
  arousal-replicatie, beide armen; respiratoire F1 tegen de 12 scoorders (mediaan), AHI tegen
  de scoordermediaan, arousal-F1 tegen de EEG-arousal-referentie waar die er is.
- Uitkomsten per nacht en arm: AHI, RDI, n_hypopnea, n_rera, arousal-index, respiratoire
  F1, aandeel hypopneeën met arousalkoppeling.

## Lezing (vooraf)
- Beschrijvend, **geen beslisregel** — het gaat om de grootte en richting van de doorwerking.
  Rapporteren: gepaarde ΔAHI, ΔRDI, Δarousal-index (mediaan, bereik, p), ΔF1 respiratoir,
  per NSRR-AHI-tertiel. Verwachting uit de replicatie: AHI vrijwel ongewijzigd (koppeling
  raakt alleen arousal-gegradeerde hypopneeën), RDI omhoog waar het U-Net meer arousals
  vindt (SHHS: index 12 → 19), arousal-index naar de referentie toe.
- Wat volgt: niets automatisch; de default (lgbm) en het werkpunt blijven Barts beslissing.
