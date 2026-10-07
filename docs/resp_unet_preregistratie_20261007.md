# Preregistratie — respiratoire 1D-U-Net (bench/resp) als kandidaat-kandidaatgenerator

Datum: 2026-10-07, **geschreven tijdens de training en vóór enige evaluatie buiten de
validatienachten.** Bench: `bench/resp/README.md` (model, invoer, augmentatie, data). Eerste
epoch op de 100 MESA-validatienachten: gepoolde event-F1 0,751 op τ 0,25 — een keuzeset, geen
bewijs. Het bevroren artefact is `bench/resp/model_best.pt` na afloop van de training
(sha256 wordt hier bijgeschreven **vóór** de eerste evaluatie hieronder; wie daarna iets aan
model, nabewerking of werkpunt wijzigt, begint opnieuw).

## Werkpunt
τ = de beste drempel op de MESA-validatienachten (`model_best.pt["thr"]`), nabewerking exact
`bench/resp/postproc.py` (1 s-gemiddelde, gaten ≤ 2 s, ≥ 10 s, type = kop met hoogste
gemiddelde, slaappoort). Geen hertuning op de cohorten hieronder; het drempelraster wordt
wél gerapporteerd als orakelcurve.

## Cohorten (in deze volgorde) en baselines
1. **SHHS1, 150 verse nachten** (seed 20261007 uit `shhs1/`, uitgesloten: de 150 van de
   arousal-replicatie van 27-09 en alles in `/srv/DATA/SHHS/gebruikte_shhs_ids.txt`;
   registratie daar). Montage **thermokoppel + RIP + SaO2, geen neusdruk**: dit toetst de
   kanaal-uitval-augmentatie op een cohort dat het model nooit zag (andere jaren, andere
   sensoren). Referentie NSRR `aasm15` via `validate_mesa.parse_nsrr`-equivalent voor
   SHHS, matcher IoU 0,20 typeonbewust. Baselines: psgscoring `aasm_v3_rec` en
   `aasm_v3_breath` (op SHHS zijn de duale profielen gelijk aan hun ouder), zelfde
   hypnogram, zelfde harnasconventie (artefact-epochs leeg).
2. **PSG-IPA SN1–5**: event-F1 tegen elke scoorder (mediaan over 12), náást het
   menselijk plafond per nacht (`docs/respiratoir_menselijk_plafond_20261007.md`: 0,826 /
   0,549 / 0,948 / 0,553 / 0,556; gepoold 0,667) en de AHI tegen de scoordermediaan;
   baseline `aasm_v3_breath_dual` (= `breath` op één flowkanaal) en `rec`.
3. **MESA 100 validatienachten**: keuzeset; gepaard tegen `aasm_v3_breath_dual@0,50` en
   `rec` (via `scripts/validate_mesa.py --recordings`), beschrijvend.

## Beslisregel (vooraf)
- **Primair (SHHS1 n=150):** gepaarde ΔF1 (U-Net − `aasm_v3_rec`, het productie-anker;
  dezelfde toets apart tegen `aasm_v3_breath`, beide moeten slagen) > 0 op ≥ 90/150 én
  Wilcoxon p < 0,05 (nullen weggelaten), ÉN mediane count-ratio (onze events /
  NSRR) in [0,80; 1,25], ÉN AHI-bias: |gemiddelde bias| niet groter dan die van `rec`,
  ÉN in geen NSRR-AHI-tertiel gemiddelde ΔF1 < −0,02.
- **PSG-IPA:** mediane F1 (over 12 scoorders) op ≥ 4/5 nachten niet lager dan
  `breath_dual`; rapporteren als fractie van het plafond per nacht.
- **Bewakers:** (a) variantie — twee extra trainingsruns (seeds 20261008/09, zelfde
  nachten) halen op MESA-val elk ΔF1 > +0,05 tegen `breath_dual`; (b) CPU-inferentie
  ≤ 60 s/nacht (4 threads); (c) ablaties op MESA-val: alleen druk, alleen thermistor,
  zonder SpO2, zonder effort — met als regel: zonder SpO2 of zonder beide flowkanalen valt
  een bibliotheekinbouw terug op de regelketen; (d) F1 per type (apneu / hypopneu,
  typebewust) en per NSRR-AHI-tertiel apart.
- **Gevolg bij slagen:** bouw als opt-in kandidaatgenerator (`respiratory_detector =
  "unet_v1"`, default regelketen), ONNX + onnxruntime, gewichten + sha256 in
  `psgscoring/data/`, met de AASM-regels als nabewerking (duur, subtypering op effort,
  desaturatiekoppeling, arousalkoppeling) zodat elk event een reden houdt; bevroren
  profielen gepind; golden 9/9. Default aanzetten en werkpunt: Barts beslissing met
  klinische aan/uit-controle. DUA-vraag (gewichten uit MESA/SHHS verspreiden) vóór enige
  release beantwoorden.
- **Gevolg bij falen:** bench blijft, cijfers in `docs/third_party_comparison.md`.

## Rekenplan
SHHS1 150: inferentie GPU enkele minuten; baselines `rec`/`breath` ~3 min/nacht → 2 armen
× 150 / 20 workers ≈ 45 min CPU-tijd ná de lopende MESA-run (de CPU is tot dan bezet).
PSG-IPA: minuten. MESA-val-baselines: 2 armen × 100 nachten ≈ 30 min. Extra seeds: 2 ×
~40 min GPU. Verslag `docs/resp_unet_replicatie_<datum>.md`, daarna meting-verificatie.
