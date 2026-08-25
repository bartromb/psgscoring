# Preregistratie — v5: features gestandaardiseerd BINNEN de opname

**Datum:** 2026-08-25, **vóór** enige training
**Vervolg op:** `docs/arousal_v4_preregistratie.md` (criteria gehaald, hypothese níét)

---

## Wat v4 leerde, en waarom dat de opzet hier bepaalt

v4 haalde alle vier de vooraf vastgelegde criteria — en loste niets op. SN3,
de opname waarvoor de hypothese bestond, ging van count-ratio 0,52 naar 0,54.
De krimp van het bereik (0,58 → 0,54) kwam van SN1 en SN5.

**Mijn criterium was speelbaar.** "Het bereik moet krimpen" laat zich halen
door opnames die al goed zaten iets dichter bij elkaar te brengen, zonder de
uitschieter aan te raken. Dat is hier rechtgezet: de primaire maat noemt SN3
bij naam.

**En mijn mechanisme was onnauwkeurig.** Ik schreef dat `beta_ratio` faalt
omdat arousals multiplicatief zouden schalen. Maar `beta_ratio` is
`seg_beta / beta_bl_nrem`; schaal het hele EEG en teller én noemer schalen mee
— de ratio is invariant. De fracties van v4 losten daarom een probleem op dat
er niet was, en droegen 2,5 % van de gain.

Wat er op SN3 werkelijk anders is, is het **contrast**: `beta_ratio` mediaan
1,28 tegen 1,57–1,60 elders. Het onderscheid tussen arousal en achtergrond is
daar kleiner. De hele featureverdeling van die nacht ligt verschoven, en een
vast werkpunt op een ABSOLUTE featurewaarde kan dat niet opvangen.

## Hypothese

Een scoorder beoordeelt of een kandidaat opvalt **binnen déze nacht**, niet of
hij een populatiedrempel haalt. Features die per opname gestandaardiseerd zijn
(z-score binnen `mesaid`) maken die vergelijking expliciet en zouden de
verschoven verdeling van SN3 moeten opvangen.

## Wat er verandert

De 39 spectrale/vermogens/complexiteitsfeatures worden **VERVANGEN** door hun
z-score binnen de opname. Elf features blijven ruw omdat ze intrinsiek of
categorisch zijn en absolute betekenis hebben: `duration_s`, `stage_code`,
`stage_n1/n2/n3`, `stage_rem`, `dom_band_code`, `emg_confirmed`, `cvr_boost`,
`hr_shift_rel`, `pos_in_night`.

**Vervangen, niet toevoegen** — dat is de les van v4: ernaast gezet blijft het
model de ruwe features gebruiken en verandert er niets.

Runtime-consequentie: `_filter_candidates_with_lgbm` krijgt de volledige
kandidatenlijst van de opname al binnen, dus de z-score is daar berekenbaar
over exact dezelfde verzameling als bij de training (de rijen van één
`mesaid`). Een guard is nodig voor te weinig kandidaten of nulvariantie.

Trainingsopzet verder identiek: GroupKFold op `mesaid`, 5 folds, 500 bomen,
31 bladeren, lr 0,05, q7 als holdout.

## Beslisregel — VOORAF, en deze keer niet speelbaar

1. **Poort — geen regressie op MESA.** Holdout AP ≥ die van v3.
2. **PRIMAIR — de uitschieter beweegt.** SN3's count-ratio moet van 0,52 naar
   **≥ 0,75**. Dit is de opname die het probleem definieert; een model dat hem
   niet raakt heeft de hypothese niet gesteund, hoe de aggregaten er ook uit
   zien.
3. **Bewaking — de rest loopt niet weg.** De count-ratio van SN1, SN2, SN4 en
   SN5 blijft binnen **[0,85, 1,15]**.
4. **Bewaking — geen verlies op de mediaan.** Mediane PSG-IPA-F1 ≥ 0,514.

Werkpunt op MESA-OOF gekozen, zoals bij v3 en v4, en pas daarna op PSG-IPA
geëvalueerd.

**Faalt criterium 2, dan is ook deze hypothese weerlegd** en blijft v3
gebundeld. Twee weerleggingen op rij zouden betekenen dat het probleem niet in
de features zit maar in de kandidaatgeneratie of in de labels — en dan is dát
de volgende plek om te kijken, niet een derde featurevariant.

## Bekende beperkingen, vooraf

- **n = 5** op PSG-IPA; SN3 is één opname. Een positieve uitkomst vraagt
  replicatie per-opname op MESA vóór er iets uitgerold wordt.
- Z-scoren binnen de opname maakt het model **afhankelijk van de
  kandidatenverdeling van die nacht**. Bij weinig kandidaten of een nacht met
  uitsluitend zware arousals verschuift het referentiekader zelf. De guard
  vangt het extreme geval; het principe blijft een aandachtspunt.
- Het verlies van absolute informatie is echt: een model dat alleen relatieve
  posities ziet, kan een nacht zonder enige arousal niet als zodanig herkennen.
