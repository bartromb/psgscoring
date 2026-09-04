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

---

# De poortzoektocht: VLF-autocorrelatie faalt, de tweepassagepoort werkt

*Toegevoegd 2026-09-03 (avond), offline op de per-opname-matrices van de 450-run.*

## Orakelplafond

Gradering exact waar de menselijke prevalentie > 15 % is: **κ 0,371** — boven
beide uitersten (oud 0,158; s=0,25 overal 0,287), met 94 % van de winst en
235 van de 297 extra valse centrale apneus weggehaald. Een prevalentiepoort
is dus geen compensatieknop maar een echte verbetering — mits realiseerbaar.

## De CSR-detector kan hem niet dragen

`detect_cheyne_stokes` (VLF-autocorrelatie) vlagt op zijn eigen drempel 0,3
het merendeel van de opnames; de drempelveeg over de ruwe piek laat zien dat
géén werkpunt helpt:

| drempel | gevlagd | κ | vals |
|---|---:|---:|---:|
| 0,30 | 228 | 0,264 | 507 |
| 0,40 | 111 | 0,262 | 383 |
| 0,50 | 22 | 0,264 | 302 |
| 0,60 | 7 | 0,154 | 290 |

De kappa is vlak terwijl het aantal gevlagde opnames een factor tien varieert:
de piek meet iets anders dan centrale-apneuprevalentie. (Nevenbevinding: de
eerste validatie telde 332 gevlagd op 0,3, de piekreplicatie 228 — onverklaard
verschil, genoteerd; de conclusie staat op elk werkpunt.)

De gebouwde `shape_evidence_csr_gate` blijft bestaan en default uit; hij is
correct gebouwd maar zijn sensor deugt niet voor dit doel.

## De tweepassagepoort: onze eigen eerste passage als prevalentiesensor

De centrale fractie die de ÓNGEgradeerde classificatie zelf al vindt, volgt de
menselijke prevalentie met **ρ = 0,456 (p = 2,3e-06)** — beter dan de
autocorrelatie. Poort: gradeer alleen wanneer passage 1 (oud gedrag) op deze
opname > 15 % centraal ziet (en er ≥ 5 apneus zijn):

| poort | gevlagd | κ | juist | vals |
|---|---:|---:|---:|---:|
| s=0,25 overal | — | 0,287 | 260 | 548 |
| **passage-1 > 0,15** | 36 | **0,300** | 218 | **367** |
| orakel | — | 0,371 | 244 | 313 |

Wint op beide assen van het uitgerolde gedrag: +0,013 kappa én −33 % valse
centrale apneus, met 84 % van de graderingswinst behouden. Geen externe
detector, geen extra signaal — twee keer classificeren.

**Statuut: afleiding op deze 375 opnames.** De drempel 0,15 en de winst moeten
op een verse set repliceren voor er iets gebouwd of veranderd wordt; de
replicatie is geketend achter de rec-tegen-breath-run op Obelix.

---

# Replicatie van de tweepassagepoort: MISLUKT op de kappa, gehaald op de valse tellingen

*Toegevoegd 2026-09-04. Verse set: 235 opnames met data uit 400 aangeboden
(mesa-sleep-2903..4223), vier armen, per opname bewaard. Drempel 0,15 en
minimum 5 apneus lagen VAST uit de afleiding.*

| arm | κ | juist-centraal | vals-centraal | gevlagd |
|---|---:|---:|---:|---:|
| oud | 0,099 | 66/247 | 195 | — |
| s=0,25 overal (uitgerold) | **0,191** | 119 | 344 | — |
| poort (passage-1 > 0,15) | 0,180 | 102 | **271** | 18 |
| orakel (mens > 15 %) | 0,215 | 114 | 272 | — |

Criterium (b) — minder valse centrale — ruim gehaald (−21 %). Criterium (a) —
hogere kappa — **niet**: 0,180 tegen 0,191. Zelfde beeld als de tussenstand op
171; het oordeel is stabiel.

## De eerlijke samenvatting van de poort

Op de afleidingsset won de poort op beide assen; op de verse set is hij een
**ruil**: ~0,01 kappa inleveren voor ~20 % minder valse centrale apneus. Merk
op dat zelfs het orakel hier maar 0,215 haalt en op valse tellingen (272)
niet beter is dan de poort (271): deze ID-range draagt wéér een andere
prevalentiemix. Er bestaat op onselecte MESA géén instelling die overal
domineert — dat is de structurele les van vijf runs subtypering.

## De beslissing die nu op tafel ligt (drie gemeten opties)

| optie | κ (verse set) | valse centrale | karakter |
|---|---:|---:|---|
| terugrollen naar oud | 0,099 | **195** | minste valse, minste juiste |
| **laten staan** (s=0,25) | **0,191** | 344 | beste kappa, meeste valse |
| tweepassagepoort | 0,180 | 271 | middenweg, gerepliceerd als ruil |

Wat de keuze klinisch weegt: de valse centrale tellingen zitten vrijwel
allemaal in laag-prevalentie-opnames, waar ze de indruk van centrale
slaapapneu kunnen wekken. De kappa-winst van s=0,25 komt vrijwel volledig uit
de periodieke-ademhalingsnachten. Wie vooral vals-centraal-in-de-gewone-
kliniek vreest kiest de poort of terugrol; wie de CSR-nachten het zwaarst
weegt laat s=0,25 staan.

De gebouwde `shape_evidence_csr_gate` (VLF-detector) blijft default uit en is
door deze uitkomsten niet meer de kandidaat; een eventuele poort hoort op de
tweepassagefractie te werken. Die is NIET gebouwd — eerst deze
gebruikersbeslissing.
