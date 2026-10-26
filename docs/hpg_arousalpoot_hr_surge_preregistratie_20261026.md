# Preregistratie — kan een hartslagstijging de arousalpoot van een hPG dragen? (meting vóór bouw)

Datum: 2026-10-26, ochtend. **Geschreven vóór de berekening.** Aanleiding: de hPG-simulatie
(`docs/hpg_simulatie_20261026.md`) verliest zonder EEG precies de 2 709 arousal-gedragen
hypopneeën (−2,55 /u, 27 ernstwissels). Roadmap-kandidaat C stap 2 is een arousalpoot op
RIP/Pleth; sinds 25-10 staat de hartslagstijging per event (Huang 2026) in de bibliotheek. Vraag
hier: **onderscheidt die maat de arousal-gedragen hypopneeën van ademhaling zonder event?** Dit is
een beschrijvende meting op de al gescoorde `psg`-arm (n150, een al gebruikte set); er wordt niets
gebouwd en geen drempel gekozen.

## Opzet

- Per nacht het MESA-kanaal `HR` (oximeter-puls, 1 Hz; geldig 20–250 bpm) en het NSRR-hypnogram
  (`waakpoort_stap0.stadia`, identiek aan de meting).
- ΔHR per event = max HR in [onset, einde + 15 s] − gemiddelde HR in de 60 s vóór de onset, met
  ≥ 30 geldige basislijnseconden; anders ongeldig (dezelfde definitie als
  `ancillary.compute_event_hr_surge`, hier nagerekend per event zodat elke klasse zijn eigen lijst
  houdt).
- Klassen uit de `psg`-arm (`aasm_v3_breath_dual`, `unet_v1`): **A** hypopneeën met ≥ 3 %
  desaturatie; **B** hypopneeën met < 3 % desaturatie (de arousal-gedragen, die een hPG verliest);
  **D** apneus; **C** controlevensters: per B-event één venster van dezelfde duur op een
  willekeurige slaapseconde (seed 20261026) waarvan basislijn en venster in slaap liggen en geen
  psg-event binnen ±30 s ligt (≤ 50 pogingen).
- Rapportage: per klasse n, mediaan/p25/p75 van ΔHR, aandeel ≥ 5 / ≥ 7 / ≥ 10 bpm; AUC (Mann-
  Whitney) B tegen C en A tegen C; per nacht de mediaan van B en van C met een gepaarde Wilcoxon;
  aandeel B met `rule1a_arousal`.

## Lezing vooraf

- AUC(B, C) ≥ 0,80: de hartslagstijging draagt de arousalinformatie grotendeels; een poot op het
  pulskanaal is het bouwen en pre-registreren waard (met kandidaatpool uit de hPG-keten, niet uit
  deze lijst).
- 0,70 ≤ AUC < 0,80: deels; alleen in combinatie met een tweede autonome maat (PWA-daling, RIP-
  onrust) te overwegen.
- AUC < 0,70: de poot kan de −2,55 /u niet terughalen zonder veel bijvangst; niet bouwen.
- Wat deze meting NIET zegt: de precisie van een echte kandidaatpoort (de afgewezen kandidaten van
  de hPG-keten zitten niet in de rijen), het effect op de AHI, en de overdracht naar klinische
  pulsoximeters (MESA-`HR` is een oximeterafleiding op 1 Hz).
