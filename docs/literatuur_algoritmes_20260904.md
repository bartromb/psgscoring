# Literatuurnotitie: algoritmes voor apneu-, hypopneu- en arousaldetectie

*2026-09-04, geschreven terwijl de poort-replicatie draait. Doel: onze
getallen naast de stand van het veld leggen en de roadmap ijken.*

## De directe vergelijkingspunten

| domein | veld (beste gepubliceerd) | wij (psgscoring 0.32.0) |
|---|---|---|
| respiratoire events, event-F1 | ABED 0,78 (apneus totaal); OA 0,71 / **CA 0,51** / HYP 0,65 | harnasafhankelijk; menselijk plafond PSG-IPA mediaan 0,556 |
| arousals | PhysioNet 2018 winnaar (DeepSleep, U-Net): AUPRC 0,55; CAISR: κ 0,45 ≈ ervaren technoloog | F1 0,556 = **82 % van het menselijke plafond** (0,679) met drie afleidingen op 0,80 |
| AASM-ISR-benchmark | geautomatiseerde analyse: resp. κ 0,34, arousal κ 0,45 — "experienced technician level", onder multi-expertconsensus | — |

**Waarschuwing bij elke kruisvergelijking:** F1-definities verschillen
(overlapcriteria, eventfusie, welke eventtypen meetellen). Onze 0,277 op de
verse MESA-range is met een IoU-0,20-koppeling op álle respiratoire events
gemeten en is niet naast ABED's 0,78 te leggen zonder hetzelfde harnas. De
enige eerlijke externe vergelijking is er een op dezelfde opnames met dezelfde
matcher — daarvoor bestaat `compare_caisr.py` al.

## Wat de literatuur ons vertelt

### 1. Centrale apneus zijn ook voor de grote modellen het zwakke punt

ABED — getraind op **5456 PSG's** uit vier cohorten (MESA, MrOS, WSC, CFS) —
haalt CA-F1 0,51 tegen OA 0,71. Onze worsteling met de centrale subtypering
is dus geen lokaal defect maar de moeilijkste hoek van het veld. De
basiskans/Simpson-analyse van deze week (winst alleen boven 15 % prevalentie)
heb ik in deze literatuur nérgens zo gerapporteerd gezien; dat kan een
publiceerbare bevinding zijn.

### 2. Probabilistische events zijn de richting van het veld

ABED geeft per event een kansverdeling over typen ("apnotyping") en een
onzekerheidsbewuste AHI (r² 0,84, helling 0,90), en kent een aparte klasse
voor respiratoire events zónder arousal/desaturatie (IRE, F1 0,47 — RERA-achtig).
Dat is precies de filosofie van onze gegradeerde evidentie en van variant 2
(p_scored → verwachtingswaarde-AHI). De literatuur bevestigt de richting;
onze twaalf-scoorderkalibratie per event zou er iets aan toevoegen wat ABED
niet heeft: een expliciete afbeelding op de menselijke scoorderfractie.

### 3. De waveform-modellen winnen pas op duizenden opnames

PhysioNet 2018: 994 getrainde opnames, beste AUPRC 0,55. ABED: 5456. Onze
U-Net-poging faalde op ~100 opnames (F1 0,287 tegen 0,443 regelgebaseerd) —
consistent met het veld, geen weerlegging van de aanpak. Wil dit pad ooit,
dan is de geplande MESA-training (2056 opnames) met PSG-IPA-kalibratie de
juiste schaal; eronder is het verspilde rekentijd.

### 4. Het consensusplafond is overal het referentiepunt

De AASM-ISR-analyse en CAISR rapporteren beide "op het niveau van een
ervaren technoloog, onder de multi-expertconsensus". Ons project meet
hetzelfde plafond expliciet (0,679 arousals; 0,556 respiratoir; κ 0,000 voor
hypopneu-subtypen) en rapporteert er consequent tegen — dat is
methodologisch in lijn met de besten, en de hypopneu-subtype-bevinding
(geen stabiel doel) is scherper dan wat ik in deze bronnen aantref.

## Concreet voor de roadmap

1. **Externe benchmark op gelijke voet**: CAISR draaien op onze 375-run-opnames
   met onze matcher (`compare_caisr.py`) — één run, en dan staat er eindelijk
   een externe referentie naast onze cijfers.
2. **Variant 2 wint aan prioriteit**: het veld beweegt naar probabilistische
   events; onze scoorderfractie-kalibratie is daarbinnen een niche die niemand
   bezet.
3. **Waveform-pad alleen op volle MESA-schaal** — en met de ABED-les:
   arousal-/wake-kansen als extra invoerkanaal, niet alleen ruwe flow.
4. **De Simpson/basiskans-analyse van de subtypering opschrijven** — mogelijk
   zelfstandig publiceerbaar naast de hoofdpaper.

## Bronnen

- ABED: *Expert-level probabilistic breathing event detector informs
  phenotyping of sleep apnea*, Nat Commun 2026 (PMC12999980)
- PhysioNet/CinC Challenge 2018 ("You Snooze, You Win"); DeepSleep,
  Commun Biol 2020
- CAISR: *achieving human-level performance in automated sleep analysis*,
  SLEEP 2025 (zsaf134)
- *Automated analysis of the AASM Inter-Scorer Reliability gold-standard
  polysomnogram dataset*, JCSM 2025
- ALPEC: evaluatieframework arousaldetectie (arXiv 2409.13367)
