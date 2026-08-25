# SN3: de classifier verwerpt de helft, omdat zijn hoofdfeature een verhouding is

**Datum:** 2026-08-25
**Cohort:** PSG-IPA, 5 opnames, 12 scoorders, arousal-F1 bij IoU 0,20
**Status:** oorzaak vastgesteld, **niet gerepareerd** — de reparatie is
modelhertraining (T7), zie `issues/v4_fractie_features.md`.

---

## De observatie

Tegen de twaalf scoorders (subtree `EEG_arousals`, 12 unieke scoorderssets):

| opname | wij | plafond | gat | precisie | recall |
|---|---:|---:|---:|---:|---:|
| SN5 | 0,71 | 0,766 | 0,06 | 0,67 | 0,73 |
| SN2 | 0,40 | 0,492 | 0,09 | 0,43 | 0,42 |
| SN1 | 0,51 | 0,642 | 0,13 | 0,53 | 0,52 |
| SN4 | 0,60 | 0,767 | 0,17 | 0,60 | 0,64 |
| **SN3** | **0,42** | **0,692** | **0,27** | **0,61** | **0,31** |

SN3 is **puur een recall-probleem**: de precisie (0,61) ligt normaal tegenover
de menselijke 0,72, maar de recall is 0,31 tegen 0,71. Wij vinden 74 arousals
waar twaalf scoorders er 142 zien; op alle andere opnames komt onze telling
binnen 10 % uit.

## Vier verklaringen getoetst en weerlegd

1. **Pre-slaapvoorwaarde** (Check A, ≥ 60 % slaap in de voorafgaande 10 s).
   Verwerpt **0 %** van de menselijke arousals, op alle vijf de opnames. SN3
   heeft bovendien het mínst gefragmenteerde hypnogram (3,4 % W).
2. **Kandidaatgeneratie.** Regelgebaseerd vindt SN3 er **135 tegen 142**
   menselijk. De arousals wórden gevonden; de filter gooit ze weg.
3. **Signaalvervuiling.** SN3's hoge amplitude is deltagedreven en fysiologisch
   (89,3 % van het vermogen in 0,5–4 Hz), en juist de 30–45 Hz-band is er het
   laagst (0,5 %) — dus geen breedbandige EMG- of bewegingsruis.
4. **Spectrale lekkage** van delta in beta. Gemeten met en zonder
   hoogdoorlaat op 4 Hz, met exact het 2 s-venster van de detector:
   **0,0 % op alle vijf.**

## De oorzaak

| | SN1 | SN2 | **SN3** | SN4 | SN5 |
|---|---:|---:|---:|---:|---:|
| EEG p95 (µV) | 35,8 | 64,0 | **98,2** | 45,3 | 59,7 |
| **absoluut betavermogen** | 6,80 | 11,99 | **15,05** | 11,35 | 13,86 |
| `beta_ratio` mediaan | 1,57 | 1,60 | **1,28** | 1,48 | 1,60 |
| keep-rate bij 0,80 | 11,8 % | 3,7 % | **5,9 %** | 8,0 % | 15,6 % |
| telling / mens | 0,93 | 1,04 | **0,52** | 1,02 | 1,10 |

SN3 draagt **2,2× het absolute beta-achtergrondvermogen van SN1**. Een arousal
is een corticaal event van ruwweg vaste grootte; de achtergrond verschilt per
persoon. Voegt zo'n event een min of meer VASTE hoeveelheid beta toe, dan is de
VERHOUDING op een hoog-amplitude opname kleiner bij dezelfde arousal.

`beta_ratio` is met afstand het belangrijkste feature van het model:
gain 2 533 753, **3,6× boven nummer twee** (`delta_beta_pre_cand`, 707 436).
Dat feature veronderstelt dat arousals **multiplicatief** schalen met de
achtergrond. Op SN3 houdt die aanname geen stand, het model leest te laag, en
een vast werkpunt vertaalt dat in stille onderdetectie.

Dezelfde faalwijze als de EMG-bug van 24-08 (`emg_var_ratio` constant nul), nu
op de amplitude-as: **het model degradeert buiten zijn trainingsverdeling en
een vaste drempel maakt daar een klinische fout van.**

## Waarom het werkpunt dit niet oplost

Verlagen repareert SN3 en bederft SN5, waar de keep-rate al 15,6 % is en de
telling 1,10 van de menselijke. Dat is dezelfde afruil die het event-locked
venster liet zien: de koppeling verbeterde, maar de telling ging over de
vooraf vastgelegde grens (`arousal_event_locked_bevinding.md`).

## De richting die wél bij de wortel aangrijpt

De oplossing bestaat conceptueel al in deze codebase. De `spectral_shift`-vlag
(v0.23.0) verving het vermogenscriterium door een **begrensde fractie**

    r = (alpha + theta + beta) / (delta + alpha + theta + beta + sigma)

die per constructie invariant is onder amplitudeschaling. Die vlag is op het
REGELGEBASEERDE pad weerlegd — maar het **model** gebruikt nog steeds ratio's.

Zie `issues/v4_fractie_features.md`.

## Beperkingen

- **n = 5.** SN3 is een uitschieter in zowel amplitude als `beta_ratio`; met
  vijf opnames is een verband tussen die twee niet vast te stellen, alleen de
  samenloop.
- Het mechanisme (arousal voegt een vaste absolute hoeveelheid beta toe) is
  **beredeneerd, niet gemeten**. Toetsbaar: de absolute beta-toename tijdens
  menselijk gescoorde arousals vergelijken tussen opnames met een verschillende
  achtergrondamplitude.
