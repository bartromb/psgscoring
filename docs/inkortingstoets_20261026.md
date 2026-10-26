# Inkortingstoets: de stadium-basislijn op een vloerlengte-hypnogram (MESA, 30 nachten)

Datum: 2026-10-26, 19:17–21:41 CET (rookproef 18:37–19:17 op mesa-sleep-0001). Preregistratie
`docs/inkortingstoets_preregistratie_20261026.md` (d0a0a16, sha256 197cee5e…, geschreven vóór de
meting; prereg-sha in `run.log`). Werkstation Z6, `systemd-run` CPUQuota 800 %, 8 workers (de
prereg zei 6; verhoogd vóór de start voor de looptijd), één thread per proces, thermische bewaker
78 °C: 854 metingen, piek 61 °C. psgscoring **0.37.0** (werkboom 65388b3), profiel
`aasm_v3_breath_dual`, LGBM-arousalketen. De 30 MESA-nachten van de reviewvelden-hermeting van
12-10 (hergebruik, geregistreerd). Zes armen per nacht: NSRR-hypnogram dekkend (H0, ceil zoals het
harnas), laatste epoch weg (H1, vloerlengte zoals YASA in de kliniek), laatste 5 % weg (H5), elk met
`stage_baseline_partial_epochs` True (new) en False (old). Alle tellingen en F1's in het
gemeenschappelijke venster (onset < len(hypnogram)·30 s), referentie NSRR `aasm15`, matcher
`LEGACY_MATCHER`. Runner `scripts/inkortingstoets_mesa.py`, ruwe uitvoer
`docs/inkortingstoets_20261026/rijen.json` (lokaal, 30 × 6 armen met eventlijsten), analyse
`analyse.py` → `analyse_uitvoer.md`, `samenvatting.json`. 0 fouten; 34 min per nacht (max 47).

## 1. Controle en de vlag

- **H0_old ≡ H0_new op 30/30** (eventlijst én AHI) met `stage_baseline_used = True` in beide: bij een
  dekkend hypnogram zijn de twee codepaden identiek, zoals de decompositie afleidde.
- **`stage_baseline_used`: True in alle `new`-armen (90/90) en in H0_old (30/30), False in H1_old en
  H5_old (60/60).** De afleiding van de decompositie ("faalde stil") is hiermee waargenomen: op een
  hypnogram dat één epoch korter is dan het signaal gooit het oude pad de `IndexError` en draait de
  stadium-basislijn niet.

## 2. Vraag 1 — implementatie: de partial-epoch-route reproduceert de dekkende basislijn

| vergelijking | mediaan |Δ% apneus| | nachten > 5 % | Jaccard mediaan / min | oordeel (regel) |
|---|---|---|---|---|
| H1_new tegen H0_new (−1 epoch) | **0,0** | **0/30** | **1,000 / 1,000** | **STABIEL** |
| H5_new tegen H0_new (−5 %) | 0,0 | 1/30 (mesa-sleep-0460: 45 → 50, +11 %) | 1,000 / 0,436 | STABIEL |

Op een vloerlengte-hypnogram (de YASA-situatie) is de uitkomst van de nieuwe route **identiek** aan
die van de dekkende basislijn: dezelfde eventlijst op 30/30. De nieuwe eigen-PSG-cijfers sinds
0.36.0 zijn dus de cijfers van de gevalideerde detector, geen implementatieartefact. Eén
kanttekening uit H5: op mesa-sleep-0460 verandert 5 % minder slaap in de basislijnschatting de
eventlijst voor meer dan de helft (Jaccard 0,44, apneus 45 → 50) — de stadium-basislijn is op
sommige nachten gevoelig voor wélke epochs in de schatting zitten (beschrijvend, geen criterium).

## 3. Het oude kliniekpad (tot 12-10) op MESA, en wat het tegen de referentie doet

**H1_old tegen H1_new** (oud pad = basislijn faalt): mediaan |Δ% apneus| 18 % (getekend −14 %),
> 5 % op **21/30** nachten, Jaccard mediaan 0,36 (min 0,03) → NIET stabiel. Zonder stadium-basislijn
telt de detector op MESA meestal mínder apneus (ratio oud/nieuw mediaan 0,83, bereik 0,02 … 1,33);
over 30 nachten 1 555 tegen 2 370 apneus (referentie-apneus in het venster 852; totale events oud
3 857, nieuw 4 606, referentie 4 806).

**Vraag 2 — detectiekwaliteit, H1_new tegen H1_old tegen `aasm15` (prereg punt 4):**

| maat | oud pad (zonder basislijn) | nieuw pad (met basislijn) |
|---|---|---|
| F1 mediaan Δ (new − old) | — | **0,000**; beter/slechter/gelijk 13/13/4; Wilcoxon p = 0,51 |
| apneu-F1 mediaan Δ | — | 0,000 |
| bias AHI − referentie, gemiddeld (mediaan) | −5,07 (−0,83) | **−0,53 (−0,21)** |
| ernstklasse = referentie | **23/30** | 18/30 |
| per tertiel ΔF1 mediaan (laag/midden/hoog) | — | +0,002 / −0,003 / +0,002 |
| H5 (secundair) | — | zelfde beeld: ΔF1 0,000 (12/14/4, p = 0,42), bias −5,14 → −0,57, ernst 21 → 17 |

**Lezing volgens de huisregel: MESA beslist niet** (ΔF1 mediaan 0, p = 0,51). De bias halveert
ruimschoots mét basislijn, maar de ernstconcordantie daalt van 23 naar 18 van 30, en dat komt uit
een klein aantal nachten waar de stadium-basislijn honderden apneus toevoegt die de referentie niet
kent:

| nacht | tertiel | AHI ref | AHI oud → nieuw | apneus oud → nieuw | apneu-F1 oud → nieuw | ernst ref / oud / nieuw |
|---|---|---|---|---|---|---|
| mesa-sleep-0111 | laag | 0,0 | 2,3 → **48,4** | 3 → **183** | 0,000 → 0,000 | Normal / Normal / **Severe** |
| mesa-sleep-0225 | laag | 1,7 | 4,8 → **35,3** | 17 → **220** | 0,105 → 0,009 | Normal / Normal / **Severe** |
| mesa-sleep-1305 | laag | 0,6 | 4,9 → **15,0** | 13 → **98** | 0,143 → 0,020 | Normal / Normal / **Moderate** |
| mesa-sleep-2083 | hoog | 29,4 | 26,4 → 38,9 | 27 → 90 | 0,651 → 0,283 | Moderate / Moderate / Severe |
| mesa-sleep-0910 | midden | 26,1 | 26,7 → 36,5 | 87 → 152 | 0,426 → 0,369 | Moderate / Moderate / Severe |
| mesa-sleep-0001 | midden | 28,5 | 29,7 → 32,1 | 71 → 80 | 0,400 → 0,364 | Moderate / Moderate / Severe |
| mesa-sleep-1892 | hoog | 66,9 | 38,2 → **45,5** | 10 → **58** | 0,137 → **0,397** | Severe / Severe / Severe |
| mesa-sleep-0583 | midden | 19,3 | 7,3 → 10,5 | 10 → 39 | 0,000 → 0,000 | Moderate / Mild / Mild |

Twee gezichten van dezelfde basislijn: op nachten mét veel echte apneus haalt hij er meer binnen
(1892: apneu-F1 0,14 → 0,40; 0583, 1224, 1589 bias omlaag), op drie nachten met een referentie-AHI
onder 2 /u maakt hij van een normale nacht een ernstige (0111: 183 apneus waar de referentie in het
venster níéts telt). Het F1-gemiddelde is daardoor −0,012 bij een mediaan van 0,000. Mechanisme
(hypothese, niet gemeten): de stadium-basislijn is één constante per stadium (P90 van de stabiele
samples) die met de dynamische basislijn wordt gemiddeld; op een nacht met een korte periode van
grote ademamplitude in één stadium ligt die constante hoog en zakt de genormaliseerde flow in de rest
van dat stadium onder de apneudrempel. R04 (eigen PSG, 28 → 85 apneus, beide sensoren 9 → 54, van
licht naar matig) past in dat beeld — en kan daar zonder referentie niet van worden onderscheiden.

## 4. Wat dit betekent

- **Vraag 1 is beantwoord: de implementatie klopt.** Het nieuwe pad op een vloerlengte-hypnogram
  is op 30/30 identiek aan de dekkende basislijn; de eigen-PSG-cijfers sinds 0.36.0 zijn die van de
  detector die op MESA en PSG-IPA is gevalideerd.
- **Vraag 2: op MESA is mét basislijn niet beter dan zonder.** F1 gelijk (p = 0,51), bias beter,
  ernstconcordantie slechter, en drie van dertig normale nachten worden ernstig of matig. De
  MESA-validaties van de detector (F1, bias) zijn gemeten mét deze basislijn en mét deze
  foutmodus erin; de n150-mediaan verbergt hem zoals de mediaan hier.
- **Voor de kliniek** (Barts beslissing, prereg punt 6): (a) `stage_baseline_partial_epochs` op de
  klinische profielen terugzetten op False — dat herstelt de apneutellingen van vóór 12-10 (R04
  9,0 /u), is op MESA F1-gelijk met betere ernstconcordantie en slechtere bias, en is één regel in
  `profiles.py`; (b) laten staan en de foutmodus repareren (basislijn per stadium begrenzen aan de
  dynamische, of een minimum aan stabiele ademhaling per stadium eisen) achter een nieuw profielveld,
  gemeten op dezelfde 30 nachten plus de 20 eigen PSG's vóór uitrol. Mijn voorkeur: (a) nu, (b) als
  aparte prereg — een detector die op 3/30 nachten Normal → Severe geeft, hoort niet in een
  klinisch verslag, en de eigen nachten hebben geen referentie om de R04-kant te checken.
- `breath_phase_split` (de tweede drager uit de decompositie) is hier niet gemeten; MESA 12-10 vond
  hem neutraal.
- Niets aan bibliotheek, profielen of productie is gewijzigd (productie: 0.39.2 / 0.37.0, beide
  velden op de 0.36.0-waarde).

## 5. Verificatie

[Meting-verificatie volgt; bevindingen worden hieronder verwerkt.]
