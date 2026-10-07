# Preregistratie — diagnostiek van de alleen-druk-apneus onder `breath_dual` (stap 1)

Datum: 2026-10-07, **geschreven vóór de run.** Vervolg op
`docs/breath_vs_breath_dual_eigen_psg_20261007.md` en stap 1 van
`docs/breath_dual_vervolg_denkstuk_20261007.md`. Vraag: welke foutmodus dragen de apneus
die alleen de neusdruk ziet — A (thermistor te ongevoelig voor 0,90), C (hypopneu-niveau op
de thermistor, de druk overdrijft) of B (de thermistor ademt door: mond/canule)?

## Invoer en recept
Dezelfde 20 opnames (R01–R20, mapping op de server), zelfde container uit het
productie-image (psgscoring 0.34.0), zelfde hypnogram/artefacten/kanalen/inleesroute.
Beide armen (`aasm_v3_breath`, `aasm_v3_breath_dual`) opnieuw, nu mét eventlijsten
(type, onset_s, duration_s, corroboration, min_spo2) en de arousallijst van de dual-arm.
Geen patiëntgegevens: R-codes en tijden relatief tot de opnamestart.

## Kenmerken per apneu van de dual-arm (alle corroboratieklassen; `both` en
`thermistor_only` dienen als controle)
- **Amplitudedaling** op thermistor en druk: bandfilter 0,10–0,70 Hz, Hilbert-omhullende,
  mediaan tijdens het event gedeeld door de mediaan in de 60 s ervoor — **identiek aan de
  maat van `docs/apneudrempel_sensorafhankelijk_bevinding.md`**, waar 0,72 uit komt.
- **Effortdaling**: zelfde maat op RIP thorax en abdomen, gemiddeld over beide.
- **SpO2-daling**: mediaan in de 30 s vóór onset minus minimum in [onset; einde + 40 s]
  (waarden < 50 % genegeerd als uitval).
- **Arousal**: een arousal uit de dual-arm met onset in [onset; einde + 15 s].
- **Status onder `breath`**: hoogste IoU met een `breath`-event (apneu of hypopneu);
  "al gescoord" bij IoU ≥ 0,20.

## Klassen (vooraf vastgelegd, op de thermistordaling d_th)
- **A** d_th ≥ 0,72: apneu-equivalent volgens de gekalibreerde thermistordrempel, gemist
  door 0,90.
- **C** 0,30 ≤ d_th < 0,72: hypopneu-niveau op de thermistor; de druk overdrijft tot apneu.
- **B** d_th < 0,30: de thermistor ademt door; gesplitst in B-effort-behouden (effortdaling
  < 0,50: mond/canule) en B-effort-weg (verdacht voor sensorfout aan beide kanten).
- **ΔAHI-dragend** = alleen-druk-apneu die onder `breath` niet gescoord was. Daarbinnen
  "zonder gevolg" = geen SpO2-daling ≥ 3 % én geen arousal.

## Lezing en gevolg voor het ontwerp van stap 2 (vooraf)
Per poort-aan-nacht (R02, R04, R11, R18) en gepoold: de verdeling A/B/C van de
alleen-druk-apneus en van de ΔAHI-dragende subset.
- A ≥ 2/3 van de ΔAHI-dragende events → de vereniging is in de kern juist; stap 2 meet
  `breath_dual` zoals hij is, met de thermistordrempel 0,72 alleen als arm om de FRI-bron
  te zuiveren.
- C ≥ 2/3 → voorwaardelijke vereniging: alleen-druk-apneu telt als d_th ≥ 0,72 óf er is
  een gevolg (SpO2 ≥ 3 % of arousal); anders hypopneu-kandidaat via de gewone route.
- B ≥ 1/3 → bovendien een effortcriterium en een artefactvlag (mond/canule) in het rapport.
- Gemengd → de volledige regel (d_th ≥ 0,72 óf gevolg óf effortpatroon).
Niets verandert automatisch aan bibliotheek of productie; dit bepaalt alleen welke arm in
de MESA-preregistratie van stap 2 komt.

## Controle en grenzen
`both`-apneus zijn door de detector op ≥ 0,90 thermistordaling gezet; komt de
Hilbert-maat daar systematisch lager uit, dan rapporteer ik die afwijking en verschuif ik
de klassegrenzen **niet** (maat en grenzen komen uit hetzelfde dossier). Geen referentie:
de klassen zijn fysiologische lezingen van de signalen, geen juistheidsoordeel; de
MESA-arm van stap 2 koppelt dezelfde kenmerken aan NSRR-labels.
