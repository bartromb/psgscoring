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

*Herzien op 21-08-2026, vóór enige meting. De eerste versie hiervan
beschreef twee vlaggen als losse stappen A en B. Dat klopt niet, en het
verschil is niet cosmetisch — zie hieronder.*

De aanroep in `pipeline.py:940` luidt:

```python
if rejected and arousals and allow_arousal and limb_enabled:
```

`arousals` is leeg zonder `arousal_limb_wired`, en `limb_enabled` is False
zonder `rule1a_arousal_enabled`. **Beide vlaggen moeten dus tegelijk om**;
alleen `wired` aanzetten op `aasm_v3_rec` doet aantoonbaar niets. Was dit
niet opgemerkt, dan had de eerste arm van de meting nul verschil laten zien
en had die uitkomst gelezen kunnen worden als "de tak levert niets op".

Daar komt bij dat de twee profielen de arousals langs verschillende wegen
gebruiken, en dat verklaart de n=6-voorproef:

| | hypopneedetector | arousals bereiken de scoring via |
|---|---|---|
| `aasm_v3_rec` | `envelope` | **niets** — geen arousaltak in de detector, en Rule 1B-reinstatement staat dicht |
| `aasm_v3_breath` | `breath_graded` | de detector zelf (stap 7b leest `output["arousal"]["events"]` rechtstreeks, gewicht 0,9), plus Rule 1B als beide vlaggen om gaan |

Het verschil dat de hybride op `aasm_v3_breath` gaf, kwam dus **niet** van de
Rule 1A-tak — die staat overal uit — maar van de weging in de
breath-graded detector. Op `aasm_v3_rec` speelden de arousals geen enkele rol,
en dat is waarom die zes opnames tot op het cijfer identiek waren.

De ingreep is daarmee één ding, geen twee: **`arousal_limb_wired=True` én
`rule1a_arousal_enabled=True`**, gemeten als 2×2 tegen de bestaande stand.
Env-overrides bestaan al (`PSGSCORING_RULE1A_AROUSAL`), dus dat kan zonder
profielen te muteren.

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

---

# Uitkomst — 22 augustus 2026

**Weerlegd, en niet nipt. Beide criteria falen, in de tegenovergestelde
richting van wat verwacht werd.**

Gepaard over 150 MESA-opnames, seed 20260801, referentie `aasm15`, sha
`fed3786`. Twee armen: beide vlaggen uit tegen beide vlaggen aan.

| | `aasm_v3_rec` | `aasm_v3_breath` |
|---|---|---|
| AHI-bias | −5,26 → **+8,01**/u | −5,13 → **+8,13**/u |
| \|bias\| | verslechtert 2,75 | verslechtert 3,00 |
| event-F1 | 0,438 → **0,382** (−0,056) | 0,510 → **0,418** (−0,092) |
| F1 per opname | 37 beter, 106 slechter | 12 beter, 132 slechter |
| events | **+79,4** per opname | +79,4 per opname |
| ernstklasse verschuift | **94/150** | 94/150 |

- **Primair (|bias| daalt ≥ 1,5/u): NIET GEHAALD** — hij stijgt met 2,75.
- **Secundair (F1 daalt ≤ 0,01): NIET GEHAALD** — hij daalt met 0,056.

De tak schiet niet tekort maar ver door: van onderdetectie (−5,26) naar
overdetectie (+8,01), met bijna tachtig extra events per opname.

## De controle-arm bevestigde de opzet

De arm met beide vlaggen uit reproduceert de eerdere baseline exact — 150 van
150 identiek op béide profielen. Dat bevestigt en passant wat de correctie van
21 augustus stelde: `arousal_limb_wired` in zijn eentje doet niets, want zonder
`rule1a_arousal_enabled` wordt de arousallijst nooit gebruikt. De
"uit"-arm is dus het productiegedrag, en het contrast is schoon.

## Waarom de verwachting mis was

`docs/koppelvenster_bevinding.md` stelde de verwachting bij naar "1 tot 3/u
van het gat". Dat getal kwam uit de vaststelling dat slechts ~20 procentpunt
van de hypopneus boven toeval aan een arousal koppelt.

**Die 20 % is gemeten op MENSELIJK gescoorde hypopneus tegen MENSELIJK
gescoorde arousals.** De tak werkt op ONZE afgewezen kandidaten tegen ONZE
gedetecteerde arousals — allebei veel talrijker en veel slechter
gelokaliseerd. Een koppelpercentage uit een menselijke annotatie begrenst dus
niet het effect van een regel die op algoritmische kandidaten draait. De
schatting had nooit uit de ene naar de andere populatie overgezet mogen
worden.

## Wat dit wél zegt

De tak is niet fout; de invoer is het. De regelgebaseerde arousaldetector
haalt op PSG-IPA event-F1 **0,182** tegen 0,692 scoorder-onderling en
lokaliseert slecht (mediaan 184 s van het dichtstbijzijnde referentie-event op
een MESA-opname). Die als tweede bevestigingsroute gebruiken voegt ruis toe in
plaats van signaal — en Rule 1A vermenigvuldigt die ruis met het aantal
afgewezen kandidaten.

Daarmee is het uitstel in de CHANGELOG bij issue #16 achteraf terecht
gebleken, en krijgt het een concreet getal: aanzetten kost 0,056 F1 en
verschuift de ernstklasse op 63 % van de opnames.

## Volgorde die hieruit volgt

Rule 1A kan pas opnieuw beoordeeld worden **nadat** de arousaldetectie zelf op
orde is. De hybride haalde op 21 augustus zijn criteria (event-F1 0,182 →
0,463, index binnen ±40 % van de scoorder op elke opname); zou die default
gaan, dan is Rule 1A daarbovenop een zinnige hermeting. Nu niet.

Beide vlaggen blijven uit. `PSGSCORING_AROUSAL_LIMB_WIRED` blijft bestaan als
meetinstrument.
