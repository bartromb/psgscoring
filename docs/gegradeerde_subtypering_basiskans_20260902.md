# De uitgerolde s=0,25 keert om op een schone set — basiskans is de verklaring

*Gemeten 2026-09-02 op Obelix, 133 volledig disjuncte opnames.*

## Drie sets, drie antwoorden

| set | opnames | menselijke centrale apneus | κ oud | κ s=0,25 | oordeel |
|---|---:|---:|---:|---:|---|
| **verrijkt** (uitgezocht op centrale apneus) | 32 | 241 van 2108 (11,4 %) | 0,195 | **0,260** | GESLAAGD → uitgerold |
| eerste-200 (13 overlap met verrijkt) | 161 | 241 van 2099 (11,5 %) | 0,210 | **0,281** | recall < 60 % |
| **volledig disjunct** | 133 | **54 van 1592 (3,4 %)** | 0,132 | **0,115** | **κ DAALT** |

Op de schone set daalt de kappa bovendien **monotoon** met de gradering:
0,132 → 0,123 → 0,115 → 0,107 voor s = 0 / 0,20 / 0,25 / 0,30.

## De verklaring staat in de matrices

Wat de gradering koopt en kost, s = 0,25 tegen oud gedrag:

| set | juiste centrale erbij | valse centrale erbij | ruil |
|---|---:|---:|---|
| eerste-200 | **+38** | +70 | 1 : 1,8 |
| volledig disjunct | **+6** | **+81** | **1 : 13,5** |

Op de schone set levert de gradering zes echte centrale apneus op en
eenentachtig valse. Onze centrale telling wordt daar **207 waar de mens er 54
ziet** — een factor 3,8, tegen 2,2 zonder gradering.

Dit is basiskansafhankelijkheid, het klassieke precisieprobleem bij een zeldzame
klasse. Waar centrale apneus 11 % van de apneus uitmaken is de ruil gunstig;
waar ze 3 % zijn is hij dat niet. **Een gewone klinische populatie lijkt op de
tweede, niet op de eerste.**

## Waarom dit klinisch telt

De centrale-apneu-index onderscheidt obstructieve slaapapneu van centrale
slaapapneu en periodieke ademhaling, en stuurt de therapiekeuze. Een index die
bijna vier keer te hoog uitkomt wijst de verkeerde kant op.

Dat is een zwaardere zorg dan een kappa-getal: de AHI verandert niet, maar de
**samenstelling** wel, en die samenstelling is waar een clinicus op beslist.

## Wat ik NIET heb gedaan

De instelling staat in productie sinds 2026-08-30. Ik heb hem **niet
teruggerold** — dat is een productiewijziging en vraagt uitdrukkelijke
toestemming. Er ligt ook nog geen gepaarde toets: alle drie de runs bewaarden
alleen de optelsom, dus een stratificatie naar centrale prevalentie was
achteraf onmogelijk.

## Wat er nu draait

Een run van 450 verse opnames op Obelix, met `repl_obelix_perrec.py` die **per
opname** wegschrijft. Dat geeft eindelijk:

1. een gepaarde toets op de kappa in plaats van één optelsom;
2. stratificatie naar centrale prevalentie, zodat de omslag te lokaliseren is;
3. de vraag of er een werkpunt bestaat dat bij lage prevalentie niet schaadt.

Tot die er is, is het eerlijke oordeel: **de winst van s = 0,25 is aangetoond
op een verrijkte set en omgekeerd op een schone. Terugrol is een reële optie en
de beslissing ligt bij de gebruiker.**
