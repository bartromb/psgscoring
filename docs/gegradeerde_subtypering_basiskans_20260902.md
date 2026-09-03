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

---

# De 450-run: per opname is de tegenspraak opgelost

*Toegevoegd 2026-09-03 (avond). 375 verwerkte opnames, 4551 gekoppelde apneus,
per opname bewaard — de eerste run waarop een gepaarde toets kán.*

## Gepoold gaf ook deze run weer een ander antwoord

| arm | centraal | obstr. | κ (gepoold) |
|---|---:|---:|---:|
| oud | 25,1 % | 92,9 % | 0,197 |
| s=0,25 | 56,2 % | 85,7 % | 0,310 |
| s=0,3 | 66,5 % | 82,6 % | **0,323 — GESLAAGD** |

Vier gepoolde runs, vier oordelen. De verklaring staat hieronder, en ze maakt
alle vier de uitkomsten begrijpelijk.

## Per opname, gestratificeerd op centrale prevalentie

Wat s=0,25 per stratum koopt (+juist centraal) en kost (+vals centraal):

| prevalentie | opnames | centrale apneus | +juist | +vals | ruil |
|---|---:|---:|---:|---:|---|
| < 2 % | 58 | 12 | +4 | +86 | **1 : 21,5** |
| 2–5 % | 8 | 12 | +1 | +15 | 1 : 15,0 |
| 5–15 % | 20 | 57 | +11 | +121 | 1 : 11,0 |
| **> 15 %** | 12 | **316** | **+115** | +57 | **1 : 0,5** |

De twaalf opnames met > 15 % centrale apneus — periodieke ademhaling — dragen
316 van de ~400 centrale apneus en domineren élke gepoolde matrix. Dáár wint
de gradering ruim (twee juiste per valse). Overal daaronder verliest hij met
1 : 11 tot 1 : 21. De vier gepoolde runs verschilden simpelweg in hoeveel van
zulke opnames de steekproef trof: Simpson's paradox, klassiek.

## De gepaarde toets

Op opnames met ≥ 3 apneus van beide klassen (n = 22): κ 0,088 → 0,131,
mediaan Δ **+0,003, beter op 11/22, p = 0,29**. Gepaard is er GEEN bewijs
dat de gradering opnames beter subtypeert.

## De gewone kliniek (prevalentie < 5 %, n = 66)

| | menselijk | oud gedrag | s=0,25 |
|---|---:|---:|---:|
| centrale telling | **24** | 128 | **234** |

Twee dingen tegelijk: de gradering **verdubbelt** de valse centrale telling
(121 → 222), én het oude gedrag zat er zélf al een factor 5 boven. Het
centrale-tellingprobleem bij lage prevalentie is dus ouder dan s=0,25 — de
gradering maakt een bestaand probleem twee keer zo groot.

## Wat dit betekent

1. **De uitgerolde s=0,25 heeft gepaard geen aantoonbaar voordeel en
   verdubbelt de valse centrale apneus in het klinisch gewone stratum.**
2. **Kaal terugrollen lost het niet op** — ook het oude gedrag telt bij lage
   prevalentie vijf keer te veel centrale apneus.
3. Het mechanisme wijst precies naar de conditionele variant uit het
   denkstuk: gradering wáár periodieke ademhaling is gedetecteerd (de
   CSR-detector bestaat), oud gedrag of nog strenger daarbuiten. De winst
   boven 15 % prevalentie is reëel (+115 juiste, ruil 1:0,5) en het verlies
   daarbuiten is even reëel.

De beslissing — terugrollen, laten staan, of de conditionele poort bouwen en
meten — ligt bij de gebruiker.
