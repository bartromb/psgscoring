# Preregistratie — hermeting van de 20 eigen PSG's na de uitrol van psgscoring 0.35.1

Datum: 2026-10-10, geschreven vóór de meting. Vraag van Bart: "test dit nu op de eigen psgs
op Hetzner en vergelijk met vorige resultaten."

## Vraag

Wat verandert de uitrol van YASAFlaskified 0.38.10 / psgscoring **0.35.1** (hypopneu-strictness
0,30 op `aasm_v3_breath` en daarmee op `aasm_v3_breath_dual`) aan de productie-uitkomsten op
dezelfde 20 eigen diagnostische PSG's, vergeleken met de meting van 07-10
(`docs/breath_vs_breath_dual_eigen_psg_20261007.md`, psgscoring 0.34.0, strictness 0,50)?

## Steekproef en invoer

Identiek aan 07-10: dezelfde 20 opnames (R01–R20, `mapping.json` van 07-10 wordt ongewijzigd
gekopieerd; koppeling job-id ↔ R blijft op de server in
`/data/slaapkliniek/metingen/breath_dual_0351_20261010/`, mode 600), dezelfde invoer per opname
(opgeslagen YASA-hypnogram, opgeslagen artefact-epochs, dezelfde kanaalkeuze en inleesroute als
de worker), `run.py` van 07-10 met als enige wijzigingen de map, het aantal armen en de
driftcontrole hieronder. Losse container uit het productie-image `yasaflaskified:0.38.10`,
uploads read-only, 16 van 32 CPU's, `nice -n 10`, 8 analyses parallel, `OMP_NUM_THREADS=2`.
Jobqueue leeg bij de start; app- en workercontainers blijven onaangeraakt.

## Armen (per opname, zelfde `raw`, zelfde hypnogram)

| arm | profiel | strictness | rol |
|---|---|---|---|
| A | `aasm_v3_breath` | 0,30 (nieuwe default) | nieuwe breath-arm |
| B | `aasm_v3_breath_dual` | 0,30 (nieuwe default) | **productiestandaard na de uitrol — primaire arm** |
| C | `aasm_v3_breath` | 0,50 (teruggezet) | driftcontrole tegen breath 07-10 |
| D | `aasm_v3_breath_dual` | 0,50 (teruggezet) | driftcontrole tegen dual 07-10 |

De strictness wordt in de pipeline per aanroep gelezen uit
`psgscoring.constants.SCORING_PROFILES[profiel]["HYPOPNEA_STRICTNESS"]` (`pipeline.py`, regel
1073; er is geen env-override). Voor C en D zet `run.py` die sleutel voor **beide** profielen op
0,50 vlak vóór de aanroep en daarna terug op 0,30; de gelezen waarde wordt per arm in de
uitvoer vastgelegd (`strictness_gelezen`). Dat is de enige ingreep; verder draait 0.35.1 zoals
uitgerold.

## Vooraf vastgelegde lezing

1. **Driftcontrole eerst.** C moet op AHI, apneu- en hypopneetelling gelijk zijn aan de
   breath-arm van 07-10 en D aan de dual-arm van 07-10, op 20/20. Is dat niet zo, dan is er
   méér veranderd tussen 0.34.0 en 0.35.1 dan de strictness, en wordt dat eerst uitgezocht
   vóór B wordt geïnterpreteerd. (De CHANGELOG van 0.35.0/0.35.1 claimt dat de strictness en de
   familie-vlag de enige standaardveranderingen zijn; deze arm toetst die claim.)
2. **Primaire vergelijking: B tegen de dual-arm van 07-10**, per opname: ΔAHI (nieuw − oud),
   ernstklasse (normaal/licht/matig/ernstig), hypopneeën, apneus O/C/M, RDI, RERA-index,
   arousal-index, ventilatoire last. Gerapporteerd per opname én per tertiel van de opgeslagen
   AHI (laag/midden/hoog, grenzen van 07-10).
3. **Verwachtingen** (geen slaagcriteria; dit is een verschilmeting zonder menselijke referentie):
   - apneutellingen identiek op 20/20 (de strictness raakt alleen de hypopneu-gradering);
   - arousal-index identiek op 20/20 (de arousaldetector is niet veranderd; `unet_v1` staat uit);
   - AHI gelijk of hoger op elke nacht, nooit lager; op MESA n=140 verschoof de bias onder dual
     van −2,74 naar +0,52 /u, dus gemiddeld rond +3 /u verwacht, mediaan tussen 0 en +5 /u;
   - RDI verschuift minder dan de AHI waar RERA's hypopneeën worden.
4. **Wat prominent in het rapport komt, voor Barts beoordeling:** elke nacht met ΔAHI > +10 /u,
   elke ernstklasse-sprong van twee klassen of meer, elke nacht waar de AHI daalt, en elke
   afwijking van de verwachtingen hierboven.
5. Statistiek: mediaan, gemiddelde, bereik; exacte tekentoets op de niet-nulverschillen
   (zoals op 07-10: bij weinig niet-nulverschillen zegt de toets alleen iets over het teken).
   Ook A tegen breath 07-10 wordt gegeven (secundair, zelfde opmaak).

## Uitvoer en verificatie

Per opname `R*.json` zonder patiëntgegevens, kopie naar
`/srv/CODE/docs/breath_dual_0351_20261010/`; analyse met een lokaal `analyse.py` dat de
07-10-uitvoer (`/srv/CODE/docs/breath_dual_20261007/R*.json`) naast de nieuwe legt. Daarna de
meting-verificatie-agent. Verslag: `docs/breath_dual_0351_hermeting_eigen_psg_20261010.md`.
