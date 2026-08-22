# Preregistratie — de eigen artefactregel vervangen door `yasa.art_detect`

*22 augustus 2026, vóór enige meting. Spoor 2; spoor 1 (de arousalstap de
lijst laten negeren) loopt nog en is hier bewust van gescheiden.*

## Wat er nu staat

YASAFlaskified detecteert artefacten met twaalf regels eigen code
(`yasa_analysis.py:579`, `run_artifact_detection`): een epoch is artefact bij
een **piek boven 500 µV** of een **vlak signaal** (< 0,5 µV maximale stap).
Die vlaggen gaan als epoch-indices naar `run_pneumo_analysis` en van daar naar
de arousalstap, die de epochs volledig overslaat.

**YASA levert `art_detect`** — covariantie-gebaseerde outlierdetectie per
venster, met het hypnogram erbij zodat elk stadium tegen zijn eigen
achtergrond wordt beoordeeld. Wij gebruiken die nergens. Dat is niet
principieel besloten; het staat er gewoon niet.

## Waarom dit ertoe doet

Gemeten op PSG-IPA (n=5): de huidige regel onderdrukt mediaan 6,9 % van de
epochs en dat kost arousal-F1 0,505 → 0,484, slechter op 5 van 5, met de
schade oplopend met het onderdrukte percentage (SN3: 13,7 % onderdrukt, F1
halveert naar 0,186). De **precisie beweegt niet** (0,425 → 0,429) en het
aandeel events dat geen enkele scoorder zag blijft op 16 %.

De regel haalt dus de goede events weg en laat de verzonnen events staan. Het
mechanisme ligt voor de hand: een arousal gáát samen met spieractiviteit, en
een drempel op 500 µV piekamplitude selecteert precies die epochs. Dat is geen
artefactdetectie maar amplitudedetectie.

## De eerlijkheidsbeperking die het ontwerp bepaalt

**Er bestaat geen artefactreferentie.** Noch PSG-IPA noch MESA annoteert
artefacten. Ik kan dus NIET meten welke van de twee regels artefacten juister
aanwijst, en ik ga ook niet doen alsof.

Wat wel meetbaar is, is het **gevolg**: welke regel levert betere
arousaldetectie tegen een menselijke referentie. Dat is een indirecte maat en
ze kan een betere artefactdetector afstraffen die toevallig meer
arousal-epochs raakt. Die beperking staat hier vooraf, niet achteraf.

## Opzet

Drie armen, dezelfde detectorstand (multi + hybride, EOG-reject uit):

| arm | artefactlijst |
|---|---|
| A | geen (niets onderdrukt) |
| B | huidige regel (piek > 500 µV of vlak) |
| C | `yasa.art_detect`, venster 30 s, `method="covar"`, `threshold=3`, hypnogram mee |

**Cohorten, beide:**

- PSG-IPA n=5, twaalf scoorders per opname, mediane F1 over de scoorders;
- MESA n=30, zaad 20260824 (dezelfde opnames als spoor 1, zodat de armen
  vergelijkbaar zijn), NSRR-arousals als referentie.

**Maten:** event-F1 bij IoU 0,20 als primaire uitkomst, met precisie en recall
apart — het effect van spoor 1 liep volledig via recall en dat hoort zichtbaar
te blijven. Daarnaast beschrijvend: percentage gevlagde epochs per regel en de
**overlap** tussen B en C, want als ze vrijwel hetzelfde vlaggen is er niets
te kiezen.

## Beslisregel — vooraf

`art_detect` vervangt de eigen regel **alleen als beide**:

1. mediane **gepaarde** ΔF1 (C − B) ≥ **+0,010** op MESA, én
2. het teken repliceert op PSG-IPA.

**Weerlegd** bij mediane gepaarde ΔF1 ≤ 0 op MESA. Daartussen: onbeslist,
blijft opt-in.

Arm A staat er als ijkpunt bij, niet als kandidaat: als A beter is dan zowel B
als C, dan is de vraag niet welke artefactdetector maar óf de arousalstap er
een moet gebruiken — en dat is precies wat spoor 1 meet.

## Reikwijdte als het doorgaat

De artefactvlag voedt méér dan arousals: TST-noemers en andere stappen lezen
dezelfde lijst. `art_detect` vlagt vrijwel zeker een ander aantal epochs, dus
**indices die niets met arousals te maken hebben schuiven mee**. Aannemen
betekent daarom: achter een schakelaar met het huidige gedrag als default, en
apart nagaan wat er met TST en de afgeleide indices gebeurt vóór het aan gaat.

Dat laatste is geen formaliteit. `n_artifact_epochs` staat in het
Excel-rapport en artefact-epochs vallen uit de slaaptijd; een andere regel
verandert dus gerapporteerde getallen bij ongewijzigde opname.

## Wat dit niet uitwijst

Of `art_detect` met andere parameters beter zou zijn. `threshold=3` en
`method="covar"` zijn de YASA-defaults en worden hier **niet** afgesteld —
afstellen op het meetcohort is precies de fout die deze validaties moeten
vermijden. Blijkt de richting goed maar de instelling verkeerd, dan is dat een
apart spoor met een eigen kalibratieset.

---

## Wijziging vóór de meting: `method="std"` in plaats van `"covar"`

Bij het bouwen bleek `yasa.art_detect(method="covar")` `pyriemann` te vereisen
— een optionele afhankelijkheid (`yasa[art]`) die niet in de omgeving zit en
niet in `requirements.txt` staat.

Ik wissel naar **`method="std"`**, en schrijf de reden op in plaats van het
stil te doen:

- een productie-afhankelijkheid toevoegen vóór er enig bewijs is dat de
  richting deugt, is de verkeerde volgorde;
- `"std"` is geen noodgreep maar een echte artefactdetector: z-scores van de
  signaalstandaarddeviatie per venster, **per stadium** genormaliseerd als het
  hypnogram meegaat. Het verschil met de huidige regel blijft daarmee intact —
  relatief tegen de eigen nacht in plaats van een vaste drempel van 500 µV.

**`"covar"` blijft ongetest.** Wint `"std"` niet, dan is dat geen uitspraak
over `"covar"`; die vraagt een extra pakket in de image en hoort een eigen
afweging te krijgen.

Alle overige afspraken — drie armen, beide cohorten, ΔF1 ≥ +0,010 op MESA met
replicatie op PSG-IPA, parameters op de YASA-defaults — blijven ongewijzigd.
`threshold=3` is ook voor `"std"` de default.
