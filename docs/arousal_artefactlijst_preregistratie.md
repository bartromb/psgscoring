# Preregistratie — moet de arousalstap de artefactlijst gebruiken?

*22 augustus 2026, vóór de MESA-meting. De PSG-IPA-uitkomst is bekend en staat
hieronder; de MESA-arm is nog niet gedraaid.*

## Wat er nu gebeurt

YASAFlaskified vlagt een epoch als artefact bij een piek boven 500 µV of een
vlak signaal (`yasa_analysis.py:589`), `tasks.py:304` zet dat om naar
epoch-indices, en `run_pneumo_analysis` geeft ze door aan de arousalstap. De
detector slaat die epochs **volledig** over — geen kandidaten, geen events.

Dat draait in productie sinds de arousaldetectie bestaat, en het is nooit
gemeten: geen enkel PSG-IPA-harnas gaf de lijst mee, dus alle arousalcijfers
die we hebben — de 0,505 waarop de hybride is aangezet incluis — zijn zonder
onderdrukking tot stand gekomen.

## Wat al gemeten is (PSG-IPA, n=5)

| | F1 | precisie | recall | gedekt door ≥1 scoorder |
|---|---:|---:|---:|---:|
| zonder onderdrukking | 0,505 | 0,425 | 0,649 | 0,838 |
| met onderdrukking | 0,484 | 0,429 | 0,510 | 0,834 |

Gepaarde ΔF1 −0,084, slechter op 5 van 5, en de schade schaalt met het
onderdrukte percentage (SN1 0,8 % → onveranderd; SN3 13,7 % → F1 halveert).
De precisie beweegt niet en het aandeel events dat geen scoorder zag blijft op
16 %: wat wegvalt zijn de goede events, niet de verzonnen events.

**Waarom dat niet volstaat om te beslissen.** Vijf opnames, één montagetype,
en de drempel van 500 µV is nooit tegen deze data geijkt. Een klinische
omslag hoort niet op n=5 te rusten.

## De MESA-arm

- **Cohort:** MESA, arousal-annotaties uit de NSRR-XML (onset én duur), zaad
  20260824, n=30. Onafhankelijk van de vijf PSG-IPA-opnames.
- **Armen:** dezelfde detectorstand (multi + hybride, EOG-reject uit), met en
  zonder artefactlijst volgens exact de YASAFlaskified-regel.
- **Maat:** event-F1 met greedy IoU-koppeling op 0,20, plus precisie en recall
  apart — de PSG-IPA-uitkomst zegt dat het effect via recall loopt, en dat
  hoort zichtbaar te blijven.

## Beslisregel — vooraf

De arousalstap negeert de artefactlijst (nieuw gedrag) **alleen als beide**:

1. mediane **gepaarde** ΔF1 (negeren − gebruiken) ≥ **+0,010** op MESA, en
2. het teken komt overeen met PSG-IPA, dus negeren is óók daar beter.

Punt 2 staat er omdat één cohort een toevalstreffer kan zijn en de twee
cohorten verschillende EEG-amplitudes hebben. Repliceert het niet, dan blijft
het gedrag zoals het is — dezelfde regel die `rectify_lowpass` op de
enveloppe-as terecht heeft tegengehouden.

**Weerlegd** bij mediane gepaarde ΔF1 ≤ 0 op MESA. Daartussen: opt-in, en dat
rapporteer ik als onbeslist.

## Reikwijdte als het doorgaat

Alleen de **arousalstap**. De artefactlijst blijft doen wat ze elders doet —
TST-noemers, andere stappen — want die weghalen verschuift indices die niets
met dit probleem te maken hebben. Achter een profielvlag met het huidige
gedrag als default, en de gepinde reproductieprofielen (`mesa_shhs`,
`chicago_1999`, `cms_medicare`, `aasm_v1_rec`, `aasm_v2_rec`) blijven daar hoe
dan ook op staan.

## Wat dit niet uitwijst

Of de artefactregel zélf deugt. 500 µV is een amplitudedrempel, en een arousal
gaat samen met spieractiviteit; de regel kan een échte
artefactdetectie-methode zijn die hier alleen verkeerd wordt toegepast. Deze
meting kiest tussen gebruiken en negeren, niet tussen goede en slechte
artefactdetectie. Een betere regel is een apart spoor.

---

# Uitkomst — 23 augustus 2026

**AANGENOMEN.** Beide vooraf vastgelegde criteria gehaald, en niet nipt.

MESA, n=30, zaad 20260824, gepaard:

| | F1 | precisie | recall |
|---|---:|---:|---:|
| lijst **negeren** | **0,421** | 0,364 | 0,597 |
| lijst gebruiken (huidig) | 0,338 | 0,305 | 0,370 |

Gepaarde ΔF1 (negeren − gebruiken): mediaan **+0,0685**, **beter op 30 van
30**, Wilcoxon **p = 1,7·10⁻⁶**.

| criterium | uitkomst |
|---|---|
| 1. mediane gepaarde ΔF1 ≥ +0,010 | +0,0685 → JA |
| 2. teken repliceert op PSG-IPA (+0,084) | JA |

Anders dan bij PSG-IPA verbetert hier **ook de precisie** (0,305 → 0,364),
niet alleen de recall. Het onderdrukken kostte dus op MESA aan beide kanten.

## Het getal dat het verklaart

**Mediaan 19,9 % van de epochs wordt gevlagd** — tegen 6,9 % op PSG-IPA, met
uitschieters tot 53,6 % op één opname. Een vijfde tot de helft van de nacht
gaat weg.

Dat is geen artefactpercentage, dat is een schaalprobleem. De regel vlagt op
een **absolute** drempel van 500 µV piekamplitude, en die drempel betekent
iets anders bij elke opname met een andere versterking of eenheid. Precies de
fout die de RIP-poort maakte: daar mat een absolute MAD-drempel in feite de
eenhedendeclaratie van het EDF, en het repareren ervan halveerde de MESA-bias
bij identieke F1.

## Reikwijdte

Alleen de **arousalstap**. De artefactlijst blijft doen wat ze elders doet.
Achter een profielvlag met het huidige gedrag als default; de gepinde
reproductieprofielen (`mesa_shhs`, `chicago_1999`, `cms_medicare`,
`aasm_v1_rec`, `aasm_v2_rec`) blijven daar hoe dan ook op staan.

**Nog niet geïmplementeerd**, en bewust. Spoor 2 (`yasa.art_detect` in plaats
van de eigen regel) draait op ditzelfde cohort. Blijkt daar dat een fatsoenlijke
artefactdetector de arousals wél ongemoeid laat, dan is negeren de verkeerde
reparatie — dan moet de regel weg, niet de lijst. Die uitkomst hoort eerst
bekeken te worden; de bibliotheek wordt niet aangeraakt zolang die meting
loopt.
