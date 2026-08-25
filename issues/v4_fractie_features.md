# v4-model: vervang ratio-features door begrensde fracties

**Aangemaakt:** 2026-08-25
**Aanleiding:** `docs/arousal_sn3_amplitude_bevinding.md`
**Type:** modelhertraining (het T7-spoor uit `docs/arousal-recall-diagnose.md`)
**Blokkeert:** niets — het huidige model draait, dit is een verbetering

---

## Het probleem in één alinea

`beta_ratio` is het zwaarste feature van `arousal_classifier_v3.txt`
(gain 2 533 753; 3,6× boven nummer twee). Het is een VERHOUDING tegenover de
NREM-basislijn en veronderstelt daarmee dat arousals multiplicatief schalen met
de achtergrond. Op PSG-IPA SN3 — 2,2× het absolute beta-achtergrondvermogen
van SN1 — leest het model daardoor te laag: keep-rate 5,9 %, telling 0,52 van
de menselijke, recall 0,31 tegen een menselijk plafond van 0,71, terwijl de
precisie normaal is (0,61 tegen 0,72).

Vier alternatieve verklaringen zijn getoetst en weerlegd: de
pre-slaapvoorwaarde (verwerpt 0 % van de menselijke arousals), de
kandidaatgeneratie (regelgebaseerd 135 tegen 142 menselijk), signaalvervuiling
(deltagedreven en fysiologisch, 30–45 Hz juist het laagst) en spectrale
lekkage (0,0 % met en zonder hoogdoorlaat).

## Voorstel

Train v4 met **begrensde fractie-features** naast of in plaats van de ratio's:

    r_fast = (alpha + theta + beta) / (delta + alpha + theta + beta + sigma)

per venster (pre / cand / post), plus de DELTA daarvan tussen pre en cand. Die
grootheid ligt per constructie in [0, 1] en is invariant onder een
amplitudeschaling van het EEG. De formule staat al in de codebase: zie de
`spectral_shift`-tak in `arousal.py` en
`docs/arousal_spectral_shift_preregistratie.md`.

**Let op de eerdere weerlegging en waarom die hier niet geldt.** Als vervanging
van het REGELGEBASEERDE criterium was `spectral_shift` slechter (PSG-IPA
mediane F1 0,182 → 0,146). Dat zegt niets over de waarde als MODELFEATURE: een
regel moet met één drempel beslissen, een model mag een feature wegen naast
vijftig andere. Het risico is dus niet dat de fractie "al weerlegd is" maar dat
ze correleert met bestaande features en niets toevoegt — en dat is een
gain-vraag, meetbaar na de training.

## Tweede kandidaat: EMG-dropout-augmentatie

Uit `docs/arousal_emg_transport_bevinding.md`: het model splitst 486 keer op
`emg_var_ratio`, alle drempels boven nul, en degenereert zonder kin-EMG. Nu
opgevangen met een guard (regelgebaseerd pad). Beter is een model dat een
EMG-loos beslispad heeft geleerd: 30–50 % van de trainingskandidaten met
EMG-features op nul, plus een expliciet `emg_present`-feature.

## Acceptatiecriteria

- [ ] De **spreiding over opnames** krimpt. Nu: telling/mens 0,52–1,10 over
      vijf PSG-IPA-opnames bij een vast werkpunt. Dat is de grootheid die dit
      issue moet verbeteren, niet de mediane F1 — die kan gelijk blijven
      terwijl de uitschieters verdwijnen.
- [ ] SN3-recall omhoog van 0,31 **zonder** dat SN5 (nu 0,73, telling 1,10)
      over 1,10 gaat.
- [ ] Menselijk plafond blijft de maat: 0,679 mediane scoorder-tegen-scoorder
      F1 over 330 paren. Wij staan op 0,514.
- [ ] Beslisregel vooraf in de CHANGELOG, en replicatie op MESA n≥20.
- [ ] Golden 9/9; de vijf reproductieprofielen hebben `arousal_lgbm=False` en
      blijven byte-identiek.
- [ ] Het oude model blijft gebundeld tot v4 op beide cohorten wint — een
      modelwissel is niet terug te draaien met een profielvlag.

## Wat NIET het antwoord is

**Het werkpunt verzetten.** Verlagen repareert SN3 en bederft SN5. Gemeten:
op PSG-IPA gaf 0,65 wel +7,2 procentpunt koppeling maar slechts +0,018 F1, en
0,60 duwde de telling over de grens van 1,10
(`arousal_event_locked_bevinding.md`).

**Een lastafhankelijk werkpunt.** Weerlegd op MESA n=20: Spearman tussen
arousallast en count-ratio rho = −0,128, p = 0,59.
