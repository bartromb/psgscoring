# Waveletontruising: de reden was te beperkt opgeschreven

**Datum:** 2026-09-01
**Vlag:** `flow_wavelet_denoise` (ingevoerd v0.20.0, default `False`, nooit aan geweest)
**Status:** blijft uit — maar om een andere en fundamentelere reden dan tot nu toe genoteerd

---

## Wat er tot nu toe stond

De vlag werd geparkeerd met deze verklaring:

> De universele drempel degenereert **in deze positie van de keten**. `σ = MAD(d₁)/0,6745`
> veronderstelt dat de fijnste detailschaal ruis draagt. Die schaal beslaat sf/4 tot sf/2
> — 16 tot 32 Hz bij 64 Hz — en het bandfilter heeft alles boven 3 Hz al weggehaald.

Dat klopt, en het is netjes gemeten: σ = 5,5·10⁻⁷ tegen een signaalschaal van 9,7·10⁻¹,
en 0 % verwijderde piekenergie.

**Maar de formulering "in deze positie van de keten" nodigt uit tot een reparatie die niet
werkt.** Ze suggereert dat een andere positie, of een andere σ-schatting, het probleem
oplost. Dat is nu gemeten en het is onwaar.

## Drie reparaties, alle drie gemeten, alle drie onvoldoende

Synthetisch: 10 min ademhaling op 0,25 Hz bij 64 Hz, met drie ingespoten pieken
(0,5–2 s, amplitude 10×). Poort: **>90 % onderdrukking én flankverschuiving <0,25 s**.

### 1. σ schatten uit de doorlaatband in plaats van uit d₁

De schaalladder laat zien dat er wél een bruikbare schaal is:

| schaal j | band | σ = MAD/0,6745 |
|---|---|---:|
| 1 | 16–32 Hz | 1,4·10⁻⁶ |
| 4 | 2–4 Hz | 3,0·10⁻² |
| **5** | **1–2 Hz** | **4,8·10⁻²** |
| 7 | 0,25–0,5 Hz | 1,1·10¹ |

Schaal 5 is de fijnste die binnen 0,05–3 Hz valt en boven de ademfrequentie ligt. σ is daar
een echte ruisvloer.

| σ uit | σ | T | onderdrukking |
|---|---:|---:|---:|
| d₁ (huidig) | 1,4·10⁻⁶ | 0,000 | **0,0 %** |
| schaal 5 | 4,8·10⁻² | 0,219 | **0,7 %** |
| schaal 4 | 3,0·10⁻² | 0,137 | 0,4 % |

**Waarom dit niet kan werken.** Zachte drempeling trekt T van élke coëfficiënt af. De grootste
piekcoëfficiënt is 55,8; 55,8 − 0,219 = 55,6. De universele drempel is ontworpen om een
*ruisvloer* weg te halen, niet om uitschieters te verwijderen. Een correcte σ maakt de drempel
niet groter, alleen minder absurd klein.

### 2. Uitschieterdrempel per schaal (`med + k·MAD` op alle schalen)

**−79 %**: de fout wordt bijna twee keer zo groot. Deze drempel raakt ook de grove schalen,
en daar zit de ademhaling zelf.

### 3. Uitschieterdrempel alleen bóven de ademfrequentie

Fysiologisch de juiste variant: laat de schalen met ademhaling ongemoeid.

| k | onderdrukking | flankverschuiving |
|---:|---:|---:|
| 3 | 0,1 % | 0,000 s |
| 8 | 0,4 % | 0,016 s |
| 12 | 0,6 % | 0,016 s |

De flanken blijven nu keurig staan — ruim binnen de 0,25 s. De onderdrukking niet.

## De werkelijke reden: er is geen schaalscheiding

Energieverdeling over de schalen, ademhaling tegen artefact:

| schaal | band | ademhaling | artefact |
|---|---|---:|---:|
| **7** | **0,25–0,5 Hz** | **99,9 %** | **57,5 %** |
| 6 | 0,5–1 Hz | 0,1 % | 28,5 % |
| 5 | 1–2 Hz | 0,0 % | 10,7 % |
| 4 | 2–4 Hz | 0,0 % | 3,3 % |

**58 % van de artefactenergie ligt in de schaal waar 99,9 % van de ademhaling zit.**

De premisse van de methode — *"wavelet thresholding acts locally in time and scale, so it can
take out the spike without touching the breathing"* — veronderstelt dat artefact en signaal in
verschillende schalen leven. Bij de gespecificeerde artefactduren (0,5–2 s) tegen normale
ademhaling (3–6 s per teug) liggen ze **één octaaf** uit elkaar. Er bestaat geen schaal die het
ene bevat en het andere niet.

Dat is geen kwestie van drempelkeuze. Elke drempel die 58 % van het artefact raakt, raakt de
ademhaling even hard. Wavelets zijn hier het verkeerde gereedschap, niet verkeerd afgesteld
gereedschap.

## Gevolg

* De vlag blijft `False`, op elk profiel. Ongewijzigd.
* De **reden** in docstring en CHANGELOG wordt gecorrigeerd van "σ degenereert in deze positie
  van de keten" naar "geen schaalscheiding bij deze artefactduren". Het verschil is niet
  cosmetisch: de eerste formulering laat iemand — mij, over drie weken — opnieuw aan de σ
  sleutelen.
* Een methode die hier wél zou kunnen werken, scheidt niet op schaal maar op **vorm**: een
  artefact is niet-periodiek terwijl ademhaling dat wel is. Denk aan afwijking van een lokaal
  ademhalingssjabloon. Dat is een ander voorstel, met een eigen poort.
* `artefact_flank_exclusion` (fix 5) blijft de werkende aanpak: die repareert de flank niet maar
  sluit hem uit, en omzeilt daarmee precies het probleem dat hierboven onoplosbaar blijkt.

## Waar dit in de code staat

`psgscoring/signal.py`, in het commentaar bij de degeneratiecheck, met de kop
`DO NOT "FIX" THIS BY MOVING THE ESTIMATOR` en de drie gemeten getallen erbij.
De vier tests in `tests/test_wavelet_denoise.py` die de mislukking vastpinnen
blijven ongewijzigd geldig: ze meten de huidige σ-schatting, en die verandert
niet.

## Reproductie

De drie metingen staan in de commit die dit document toevoegt; de synthetische opzet is
identiek aan `tests/test_wavelet_denoise.py` (`_breathing`, `_with_spikes`, SF = 64 Hz).
