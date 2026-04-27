# Session summary — Apr 26, 2026

## Achievements
- TST parsing bug found and fixed (clip to signal duration + duration>0)
- Sleep-only AHI methodology implemented (events during wake excluded per scorer)
- v0.4.0 validated on PSG-IPA: Pearson r=0.994, MAE=2.90, severity 4/5
- 3-profile extraction working (validate_psgipa.py v3.4)
- Figure 1 generated with real data

## Numbers (v0.4.0 vs scorer median, sleep-only AHI)
- SN1: ref 5.6, algo 9.2, Δ +3.6, Mild=Mild ✓
- SN2: ref 3.8, algo 10.2, Δ +6.4, Normal→Mild ✗
- SN3: ref 49.2, algo 50.5, Δ +1.3, Severe ✓
- SN4: ref 3.6, algo 4.1, Δ +0.5, Normal ✓
- SN5: ref 11.2, algo 13.9, Δ +2.7, Mild ✓
- Bias +2.90, MAE 2.90, r 0.994, kappa 0.800

## Open question for tomorrow
Standard AHI (aasm_v3_rec) lies OUTSIDE the [strict, sensitive] interval
for all 5 recordings. Three options:
  A) Accept as transparent (rewrite paper text)
  B) Revise profiles in v0.4.1 (monotone strict<=std<=sens)
  C) Use OAHI confidence sweep instead (richer continuum)

Cool finding: psgscoring already exposes oahi_thresholds dict
{0.00, 0.40, 0.60, 0.85} - ready-made uncertainty visualization.

## Files
- validate_psgipa.py v3.4 (with 3-profile extraction)
- validation_results_*.json (real data dump)
- diagnose_results.json (FP/FN patterns)
- Figure1_v040_REAL.png/pdf (current figure)
