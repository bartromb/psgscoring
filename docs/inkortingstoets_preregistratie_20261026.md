# Preregistratie — inkortingstoets: de stadium-basislijn op een te kort hypnogram (MESA, 30 nachten)

Datum: 2026-10-26, geschreven vóór de meting. Opdracht van Bart: "doe de inkortingstoets op MESA, je
hebt permissie".

## Vraag

De decompositie van 26-10 (`docs/drift_036x_decompositie_20261026.md`) wees
`stage_baseline_partial_epochs` aan als drager van de apneu-verschuiving op de eigen PSG's (R04 ×3).
Het mechanisme: de stadium-specifieke basislijn gooide vóór 0.36.0 een `IndexError` zodra het
hypnogram korter was dan het signaal en viel dan stil terug op de dynamische basislijn alleen
(`signal.compute_stage_baseline`, `respiratory._apply_stage_baseline` →
`stage_baseline_used=False`). Gecontroleerd vandaag: MESA-EDF's duren 43 199 s (geen heel aantal
epochs), maar het harnas (`validate_mesa.parse_nsrr`) rondt het NSRR-hypnogram **naar boven** af
(`ceil`), zodat het het signaal dekt en de basislijn op MESA altijd draaide; YASA in de kliniek
levert een **vloerlengte**-hypnogram, zodat hij daar tot 0.36.0 nooit draaide. Twee vragen:

1. **Implementatie.** Doet de partial-epoch-route (0.36.0, `True`) op een vloerlengte-hypnogram
   hetzelfde als de basislijn op een dekkend hypnogram? Zo ja, dan zijn de nieuwe eigen-PSG-cijfers
   de gevalideerde detector; zo nee, dan is de implementatie het probleem.
2. **Detectiekwaliteit.** Scoort de detector mét stadium-basislijn (gevalideerd pad) beter of
   slechter tegen de NSRR-referentie dan zonder (het pad waarop de kliniek tot 12-10 scoorde)? Op
   MESA kon dat nooit worden gemeten omdat beide paden daar samenvielen; met een ingekort hypnogram
   wordt het oude pad reproduceerbaar.

## Steekproef en invoer

De 30 MESA-nachten van de reviewvelden-hermeting van 12-10 (de eerste 30 van de standaard-n150,
seed 20260801; `ids.txt`), hergebruik, niets afgesteld — geregistreerd in
`/srv/DATA/MESA/gebruikte_mesa_ids.txt`. NSRR-hypnogram en referentie `aasm15` via `parse_nsrr`,
EDF volledig geladen zoals in het harnas. psgscoring 0.37.0 (werkboom = tag v0.37.0 plus docs),
profiel `aasm_v3_breath_dual`, `PSGSCORING_AROUSAL_DETECTOR=lgbm` (zelfde keten als de
decompositie), werkstation Z6: 6 workers onder `systemd-run` met CPUQuota 600 %, één thread per
proces, thermische bewaker 78 °C / 3 metingen / 10 s (recept van 25-10).

## Armen (per nacht, zelfde raw)

| arm | hypnogram | `stage_baseline_partial_epochs` | rol |
|---|---|---|---|
| H0_new | NSRR, dekkend (ceil, zoals het harnas) | True | referentie: de gevalideerde detector |
| H0_old | idem | False | controle: moet byte-gelijk zijn aan H0_new (geen IndexError bij dekking) |
| H1_new | laatste epoch weg (vloerlengte, de YASA-situatie) | True | implementatietoets |
| H1_old | idem | False | het kliniekpad tot 0.36.0 (basislijn faalt stil) |
| H5_new | laatste 5 % van de epochs weg | True | gevoeligheid voor de mate van inkorting |
| H5_old | idem | False | idem, oud pad |

Veldwissel in-process via `validate_mesa._apply_fields` (zelfde recept als 12-10); de vlag
`respiratory.stage_baseline_used` wordt per arm vastgelegd en is de controle dat het oude pad
werkelijk faalde (verwacht: True in alle `new`-armen en in H0_old, False in H1_old en H5_old).

## Vergelijking binnen het gemeenschappelijke venster

Een ingekort hypnogram neemt ook slaap weg aan het einde; alle tellingen en F1's worden daarom
berekend op events (algoritme én referentie) met onset vóór `T_cut = len(hypnogram_arm) · 30 s`,
met voor H0 hetzelfde venster als de arm waarmee vergeleken wordt. Matcher: `LEGACY_MATCHER` uit
`validate_mesa` (projectstandaard), referentie `aasm15`.

## Vooraf vastgelegde lezing

1. **Controle.** H0_old ≡ H0_new op eventlijst, AHI en `stage_baseline_used=True` op 30/30.
   Anders stopt de meting (dan faalt de basislijn ook op dekkende hypnogrammen en klopt de lezing
   van de decompositie niet).
2. **Implementatie (vraag 1), H1_new tegen H0_new in het H1-venster:** per nacht de relatieve
   verandering van de apneutelling en van de totale eventtelling, en de Jaccard-overlap van de
   eventlijsten. **Stabiel** = mediaan |Δapneus| ≤ 2 % én hoogstens 3/30 nachten > 5 % én Jaccard
   mediaan ≥ 0,95. Stabiel → de partial-epoch-route reproduceert de dekkende basislijn; de nieuwe
   eigen-PSG-cijfers zijn dan de gevalideerde detector. Niet stabiel → de implementatie wijkt af en
   wordt per nacht beschreven (welke events verschuiven, waar in de nacht). H5_new idem,
   beschrijvend (meer inkorting = meer verloren slaap in de basislijnschatting; geen criterium).
3. **Oud pad (de kliniek tot 12-10), H1_old tegen H1_new:** `stage_baseline_used` False op 30/30
   verwacht; relatieve apneuverandering per nacht (de eigen PSG's gaven −33 … +200 %), met
   tertielen van de referentie-AHI.
4. **Detectiekwaliteit (vraag 2), H1_new tegen H1_old tegen `aasm15` in het H1-venster:** ΔF1
   (new − old) per nacht, mediaan, beter/slechter/gelijk, Wilcoxon (`wilcoxon_signed_rank`); bias
   (AHI − referentie) per arm; ernstconcordantie; apneu-F1 apart (de basislijn raakt de apneus).
   Lezing volgens de huisregel: mediaan ΔF1 > 0 met p < 0,05 en |bias| niet groter → de
   stadium-basislijn is op MESA beter; het omgekeerde → slechter; anders → MESA beslist niet, en
   de eigen-PSG-verschuiving blijft een niet-beoordeelbare verschuiving tot de EC-referentie er is.
   Ook H5 op dezelfde manier (secundair).
5. **Prominent:** elke nacht waar H1_new > 5 % van H0_new afwijkt; de ernstwissels onder H1_old;
   de grootste apneuverhoudingen H1_old/H1_new (ter vergelijking met R04 ×3).
6. **Geen beslissing uit deze meting.** De uitkomst zegt of het nieuwe pad de gevalideerde
   detector is en of die detector op MESA beter scoort dan het oude kliniekpad; of
   `stage_baseline_partial_epochs` in de kliniek aan blijft, beslist Bart.

## Uitvoer en verificatie

`docs/inkortingstoets_20261026/`: `ids.txt`, `rijen.partial.jsonl` (hervatbaar), `rijen.json`
(per nacht per arm: tellingen, vensterstatistieken, F1/precisie/recall/bias, `stage_baseline_used`,
eventlijsten als onset/duur/type), `run.log`, `thermal_guard.log`, `KLAAR`. Analyse `analyse.py`
in dezelfde map; daarna de meting-verificatie-agent. Verslag `docs/inkortingstoets_20261026.md`.
