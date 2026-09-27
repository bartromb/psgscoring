# Preregistratie — bevroren U-Net-arousaldetector als kandidaat-default

Datum: 2026-09-27. **Geschreven vóór enige meting; ter goedkeuring aan Bart
vóór er iets draait.** Aanleiding: dsp-scout-bench `bench/eeg/RAPPORT.md`
(27-09, onafhankelijk geverifieerd): eigen BSD-schone 1D-U-Net (à la Ehrlich
e.a. 2024; EEG+EOG+kin-EMG @ 50 Hz, 4,58 M par., getraind op 317 verse
MESA-nachten, seed 20260927) haalt op PSG-IPA arousal-F1 0,733 tegen 0,555
(5/5), MESA-val 0,568→0,682 (74/79, p=1,3·10⁻¹²); tellingsneutraal τ 0,35
geeft 0,743. Een bench is geen meting: één seed, MESA-val was keuzeset, en
PSG-IPA-cijfers op alle τ zijn al gezien.

## Bevroren artefact
- `bench/eeg/unet50/model_best.pt` (sha256 vastleggen bij goedkeuring),
  voorbewerking en nabewerking exact als in `bench/eeg/unet50/README.md` en
  `bench/eeg/common/postproc.py`; daarna **10 s-samenvoeging en
  slaappoort** zoals de huidige keten. Werkpunt **τ = 0,35** (tellings-
  neutraal op MESA-val, ratio 1,01) — niet 0,20. Geen hertraining, geen
  hertuning vóór de replicatie; wie iets wijzigt, begint opnieuw.

## Cohorten (in deze volgorde)
1. **SHHS1, 150 verse nachten**, seed 20260927 uit `shhs1/` (5792 nachten;
   montage EEG/EEG(sec), EOG(L/R), EMG @ 125 Hz → resample 50 Hz;
   NSRR-`Arousal`-annotaties als referentie; hypnogram uit de nsrr-xml).
   Registratie in nieuw `/srv/DATA/SHHS/gebruikte_shhs_ids.txt`. SHHS is
   een ánder cohort (jaren 90, andere sensoren) en nooit door dit model
   gezien: dit is de beslissende externe replicatie.
2. **MESA, de 76 resterende verse nachten** (tweede, zwakkere replicatie —
   zelfde cohort als de training, ander sample).
3. **PSG-IPA n=5**: bevestiging, niet beslissend (τ-veeg al gezien).
Baseline overal: psgscoring 0.34.2, `aasm_v3_rec`, productie-aanroep,
zelfde hypnogram; matcher `validate_psgipa.LEGACY_MATCHER` (IoU 0,20,
typeonbewust) plus onset ±5 s als tweede maat.

## Beslisregel (vooraf)
- **Primair (SHHS1 n=150):** gepaarde ΔF1(U-Net−baseline) > 0 op ≥ 90/150
  én Wilcoxon p < 0,05, ÉN mediane count-ratio (onze events / NSRR) in
  [0,80; 1,25], ÉN winst niet beperkt tot één arousallast-tertiel (in geen
  tertiel gemiddelde ΔF1 < −0,02).
- **Tweede cohort (MESA 76):** ΔF1 > 0 op de meerderheid, p < 0,05.
- **PSG-IPA:** F1 op ≥ 4/5 opnames niet lager dan baseline.
- **Bewakers:** (a) variantie — twee extra trainingsruns (seeds 20260928/29,
  zelfde 317 nachten) moeten op MESA-val elk ΔF1 > +0,05 halen, anders is
  de bevroren run een uitschieter; (b) CPU-inferentie zonder GPU ≤ 60 s per
  nacht (productie op Hetzner heeft geen GPU) — anders geen default;
  (c) montage-ablaties (zonder kin-EMG; zonder EOG; alleen EEG) op MESA-val
  gerapporteerd, met als regel: zonder EMG of EOG valt het model terug op de
  huidige detector (geen stille degradatie); (d) REM- en NREM-F1 apart.
- **Gevolg bij slagen:** de detector komt als profielveld
  `arousal_detector = "unet_v1"` (default `"lgbm"`) in psgscoring, ONNX-
  export + `onnxruntime` onder de `[ml]`-extra (geen torch in de bibliotheek),
  gewichten in `psgscoring/data/` met sha256-wacht in de tests, provenance-
  rij in het rapport; bevroren profielen (`mesa_shhs`, `chicago_1999`)
  gepind op `lgbm`; golden 9/9 byte-identiek met de vlag uit. Default
  aanzetten op `aasm_v3_rec` is daarna een aparte gebruikersbeslissing met
  klinische aan/uit-controle (zoals bij de re-ranker).
- **Gevolg bij falen:** gebouwd als opt-in, gemeten, uit — met de cijfers in
  CHANGELOG en `docs/third_party_comparison.md`.

## Doorwerking die meegemeten wordt (rapporteren, geen criterium)
Arousal-index-bias per tertiel; RERA/RDI (FRI-koppeling leest de
arousallijst); hypopneu-arousalkoppeling (`coupled_arousal`; de Rule-1A-tak
blijft uit — zie `project_rule1a_arousal_limb`); PLM-arousal-associatie;
interactie met de autonome re-ranker (die werkt op LGBM-kandidaten: bij
`unet_v1` staat hij UIT en de provenance zegt waarom — herontwerp is een
aparte meting).

## Rekenplan
SHHS 150 + MESA 76 nachten: baseline ~3 min/nacht (20 workers ≈ 35 min),
U-Net-inferentie GPU (RTX A4000) enkele minuten totaal; twee extra
trainingsruns ≈ 2 × 50 min GPU; CPU-kostmeting op 5 nachten. Temperatuur-
bewaker gebonden aan de pgid, logregel geverifieerd vóór de start.
Harnas: uitbreiding van `bench/eeg/common/` naar `scripts/`, met dezelfde
matcher en referentie-export als de bench (`bench/export_ref.py`).
