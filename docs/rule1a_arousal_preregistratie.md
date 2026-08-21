# Preregistratie — de arousal-tak van Rule 1A aanzetten

Datum: 2026-08-21. **Geschreven vóór enige meting**, terwijl de
AHI-impactmeting van de arousalhybride op MESA n=150 draait. Deze twee
vragen hangen samen en de volgorde is met opzet: eerst weten of de arousals
zelf beter worden, dan pas of het loont ze te gebruiken.

## Wat er nu gebeurt

AASM v3 Rule 1A kwalificeert een hypopnee op een debietdaling van ≥30 %
gedurende ≥10 s **met desaturatie ≥3 % OF een arousal**. In psgscoring staan
twee poorten tussen de gedetecteerde arousals en die regel:

| poort | waar | stand |
|---|---|---|
| `arousal_limb_wired` | `pipeline.py:905` — bereiken arousals Rule 1B-reinstatement? | **uit** op 17 van 21 profielen, waaronder `aasm_v3_rec` |
| `rule1a_arousal_enabled` | `pipeline.py:918` — draait de Rule 1A-arousaltak? | **uit op alle 21** |

Gevolg: `aasm_v3_rec` heeft `DESAT_OR_AROUSAL=True`, maar er komt geen enkele
arousal bij de scoring aan. In de praktijk scoort het profiel hypopneus
**alleen op desaturatie**.

Dat is geen vergissing maar een expliciet uitstel. De CHANGELOG bij de
reparatie van issue #16 zegt het met zoveel woorden: de plumbing is heel, maar
welk profiel erop reageert is een profielkeuze met het bestaande gedrag als
default, omdat Rule 1B-reinstatement, FRI-RERA en de LightGBM-kenmerken
tegelijk zouden verschuiven zonder dat elk onderdeel apart opnieuw gevalideerd
is. Het uitstel had een naam — "fase 4" — en die is er nooit gekomen.

## Waarom dit nu de moeite is

Drie dingen zijn sinds dat uitstel veranderd.

**1. De richting klopt.** Met de limb aan is op PSG-IPA gemeten:

| opname | AHI uit | AHI aan | hypopneus | getest | gekoppeld | gekwalificeerd |
|---|---:|---:|---|---:|---:|---:|
| SN1 | 8,1 | 9,3 | 32 → 39 | 74 | 11 | 7 |
| SN3 | 53,8 | 56,0 | 45 → 58 | 70 | 20 | 13 |
| SN5 | 11,4 | 14,8 | 63 → 87 | 170 | 38 | 24 |

De AHI gaat **omhoog**. Op MESA is onze bias juist negatief: −5,30/u
(`aasm_v3_rec`) en −5,18/u (`aasm_v3_breath`) na de poortreparaties van
augustus, tegen een referentie (`aasm15`) die arousal-gekwalificeerde
hypopneus wél meetelt. We meten dus systematisch te laag tegen een maat die
een tak crediteert die wij niet draaien.

**2. De arousals zelf zijn beter geworden.** Het hybride pad haalt op PSG-IPA
event-F1 0,463 tegen 0,182 regelgebaseerd (mens onderling 0,692), met de
index op elke opname binnen ±40 % van de scoordermediaan in plaats van 0,43
tot 4,36. Een tak aanzetten die op slechte arousals draait, voegt ruis toe;
op deze arousals is dat een andere afweging.

**3. Het meetharnas is nu herstartbaar** en legt zijn git-SHA vast, dus een
run van vijftien uur is niet langer alles-of-niets.

## De ingreep

Twee vlaggen, in deze volgorde, elk apart gemeten:

- **A. `arousal_limb_wired=True` op `aasm_v3_rec`** — Rule 1B-reinstatement
  krijgt de arousals. Raakt alleen hypopneus die al afgewezen waren wegens
  ontbrekende desaturatie.
- **B. `rule1a_arousal_enabled=True`** — de Rule 1A-arousaltak zelf.

Env-overrides bestaan al (`PSGSCORING_RULE1A_AROUSAL`), dus de 2×2 kan
gemeten worden zonder profielen te muteren.

## Acceptatiecriterium (vastgelegd vóór de meting)

Gemeten op **MESA n = 150**, seed 20260801, gepaard, referentie `aasm15`,
profielen `aasm_v3_rec` en `aasm_v3_breath`, met de arousalhybride in de
stand die uit de lopende meting als beste komt.

**Primair.** De absolute AHI-bias daalt met **≥ 1,5/u** op `aasm_v3_rec`.
Dat is de grootheid waar het defect zich uit; een tak aanzetten die de
onderdetectie niet aanraakt, heeft geen aanleiding.

**Secundair.** De event-F1 daalt niet met meer dan **0,01**. Meer events
vinden die de referentie niet heeft, is geen winst maar bias-cosmetica; dit
criterium scheidt die twee.

**Bewaking.** Het aandeel opnames waarop de AHI-ernstklasse verschuift wordt
gerapporteerd, ook als het criterium gehaald wordt. Een reparatie die de
gemiddelde bias verbetert maar op een kwart van de opnames de indeling
omgooit, is een andere beslissing dan een die dat niet doet.

**Randvoorwaarden voor promotie:**
1. Primair en secundair gehaald.
2. `mesa_shhs` en `chicago_1999` blijven gepind (paper v31/v37).
3. `cms_medicare` en `aasm_v1_rec` blijven eraf: die hebben
   `DESAT_OR_AROUSAL=False` en zouden door een arousaltak een AHI krijgen die
   hun eigen regelset niet kent.
4. Golden 9/9 met de vlaggen uit.
5. Toestemming van de gebruiker — dit verandert de AHI op elke opname.

## Wat dit expliciet NIET aanneemt

Dat de bias volledig door deze tak verklaard wordt. De poortreparaties van
augustus haalden hem al van −11,20 naar −5,30 zonder dat de F1 meebewoog,
en dat bleek boekhouding te zijn in plaats van detectie. Het kan hier
opnieuw zo liggen. Daarom staat de F1-eis erbij, en daarom is het
primaire criterium een drempel en geen richting.
