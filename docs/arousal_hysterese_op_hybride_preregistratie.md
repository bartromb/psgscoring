# Preregistratie — hysterese bovenop de hybride

Datum: 2026-08-22. **Geschreven vóór de meting.**

## De redenering

Hysterese is op 21 augustus alleen gemeten, op het regelgebaseerde pad, en
weerlegd: de mediane F1 zakte van 0,182 naar 0,111 en de index verviervoudigde
(SN1 van 21,0 naar 83,3 tegen een scoordermediaan van 24,2). De duur bewoog
nauwelijks mee (4,0 → 4,6 s tegen 8,3 s menselijk), waardoor het
mechanisme-criterium terecht faalde.

**Waarom dat mislukte is nu duidelijk: er was niets dat filterde.** Hysterese
voegt scherven samen die daarvoor onder de 3 s-eis vielen, dus het aantal
kandidaten explodeert. Op het regelpad blijven die allemaal staan.

De classifier is precies dat filter. En de drempelveeg van de hybride wijst
dezelfde kant op:

| classifierdrempel | F1 | mediane duur |
|---|---:|---:|
| 0,30 | 0,409 | 5,5 s |
| 0,60 | **0,463** | 6,2 s |
| 0,85 | 0,409 | **7,4 s** |
| *mensen* | *0,692* | *8,3 s* |

De classifier houdt bij hogere drempels systematisch de LANGERE events over.
Langere events zijn dus vaker echt — wat de fragmentatiediagnose bevestigt en
suggereert dat betere eventmorfologie hem beter voer geeft.

## De ingreep

`hysteresis=True` én `lgbm=True` samen, tegen de hybride alleen. Beide
bestaan al; er verandert niets aan de code.

## Correctie op de opzet, vóór de meting

De eerste opname legde een fout in mijn meetopzet bloot: hysterese gaf exact
hetzelfde als de hybride alleen (SN1: index 26,6, duur 6,2 s, F1 0,463 in
beide armen).

Oorzaak: in hybride modus zet `detect_arousals` de instapdrempel op
`AROUSAL_LGBM_CAND_RATIO = 1.2`. Een `exit_ratio` van 1,2 is daar gelijk aan,
dus de doorloop-mask is identiek aan de instap-mask en de ingreep is een
no-op. Waarden BOVEN 1,2 zijn erger: dan is de doorloop-mask smaller dan de
instap-mask en keert de logica om.

Hysterese betekent per definitie een LAGERE uitstap dan instap. Op het
regelpad was dat 1,2 tegen een instap van 2,0 — een verhouding van 0,60. De
overeenkomstige waarden op het hybride pad zijn dus **onder de 1,2**. De veeg
wordt `0,7 · 0,8 · 0,9 · 1,0`, met **0,72** als vooraf vastgelegde waarde:
dezelfde verhouding tot de instap (0,60 × 1,2) als de 1,2 die op het regelpad
was vastgelegd.

De criteria hieronder blijven ongewijzigd; alleen de as waarop gemeten wordt
is gecorrigeerd.

## Acceptatiecriterium (vastgelegd vóór de meting)

Vijf PSG-IPA arousal-opnames, twaalf scoorders, event-F1 bij IoU 0,20,
single-derivatie — hetzelfde harnas als de eerdere metingen.

**Primair.** De mediane event-F1 stijgt boven de **0,463** van de hybride
alleen. Blijft hij daaronder, dan voegt hysterese ook mét filter niets toe en
gaat ze definitief van tafel.

**Secundair (mechanisme).** De mediane eventduur schuift van 6,2 s richting de
menselijke 8,3 s, en komt in **[7,0, 10,0] s** te liggen. Stijgt de F1 zonder
dat de duur meebeweegt, dan werkt er iets anders dan de veronderstelde
oorzaak en telt de winst niet — dezelfde regel die de hysterese op het
regelpad terecht afkeurde.

**Bewaking.** De spreiding van `index_algoritme / index_scoordermediaan` blijft
binnen de 0,61–1,29 die de hybride alleen haalt.

**Randvoorwaarde.** Ook bij succes gaat er niets default zonder een meting van
de respiratoire gevolgen op MESA n=150, zoals bij de hybride zelf. Arousals
voeden Rule 1B en de RDI.

---

# Uitkomst — 22 augustus 2026

**Primair gefaald op elke waarde. Hysterese is definitief van tafel — en het
mechanisme-criterium legt uit waarom dat ertoe doet.**

| | spreiding | mediane F1 | mediane duur | q-bereik |
|---|---:|---:|---:|---|
| exit 0,72 *(preregistratie)* | 3,74 | **0,302** | 12,0 s | 0,18–0,69 |
| exit 0,80 | 2,80 | 0,384 | 11,7 s | 0,35–0,99 |
| exit 0,90 | 2,14 | **0,404** | 9,4 s | 0,55–1,18 |
| exit 1,00 | 1,96 | 0,395 | 7,5 s | 0,69–1,35 |
| **hybride alleen** | 2,10 | **0,463** | 6,2 s | 0,61–1,29 |
| *mensen* | — | *0,692* | *8,3 s* | — |

- **Primair (F1 > 0,463): NIET GEHAALD** op alle vier de waarden. De beste
  komt op 0,404.
- **Secundair (duur in [7,0–10,0] s): GEHAALD** bij exit 0,90 (9,4 s) en
  exit 1,00 (7,5 s).
- **Bewaking (spreiding ≤ 2,10): GEHAALD** bij exit 0,90 en 1,00.

## Wat dit weerlegt

Niet alleen de ingreep, maar de aanname eronder.

De hele dag droeg ik de gedachte mee dat de versnipperde fase-1 mask de reden
was voor de lage overeenstemming: 1897 ruwe regio's waarvan er 65 de 3 s-eis
halen, mediane eventduur 3,6 s tegen 8,6 s menselijk. Repareer de morfologie,
zo luidde de redenering, en de F1 volgt.

**Bij exit 1,00 klopt de morfologie en zakt de F1 toch.** De duur landt op
7,5 s tegen 8,3 s menselijk, het aantal ligt binnen 0,69–1,35 van de
scoordermediaan, de spreiding is met 1,96 béter dan de hybride alleen — en de
overeenstemming daalt van 0,463 naar 0,395.

De events hebben dus de juiste lengte, het juiste aantal en de juiste
verdeling over opnames, en vallen nog steeds niet samen met wat mensen
markeren. **Morfologie en lokalisatie zijn onafhankelijke problemen, en alleen
het tweede bepaalt de overeenstemming.**

Waarom samenvoegen actief schaadt, is ook zichtbaar: op SN4 gaat de duur netjes
van 5,8 naar 9,6 s (mens 10,3) terwijl het aantal instort van 13,5 naar 4,2
(scoorder 14,3). Vijf korte kandidaten worden één lange, de classifier ziet er
één in plaats van vijf, en gooit daar vervolgens ook nog van weg. Je wint
morfologie en verliest dekking.

En het duurprobleem is niet eens uniform: de hybride zit op SN2 al op 8,3 s
(mens 7,9) maar op SN3 op 5,6 s (mens 8,3). Een correctie die op elke opname
hetzelfde doet, kan een probleem dat per opname verschilt niet oplossen.

## Waar dit naartoe wijst

De resterende kloof — 0,463 tegen 0,692 — is een LOKALISATIEprobleem. Twee
richtingen blijven over, en de eerste is meetbaar zonder nieuw werk:

1. **Single versus multi-derivatie tegen dezelfde referentie** (punt A2 uit
   `docs/PAPER_REVISIE_v40.md`). Klinische profielen draaien default multi,
   maar die keuze is nooit tegen een referentie gemeten. Menselijke scoorders
   zien de volle montage; wij meten hier op één centraal kanaal.
2. **De montage die de scoorder zag.** EOG en kin-EMG zitten in de
   kenmerkvector van de classifier, maar niet in het kandidaatstadium. Een
   arousal die op O2 zichtbaar is en op C4 niet, bestaat voor ons niet.

Wat NIET meer geprobeerd hoeft te worden: drempels en eventmorfologie in het
regelpad. Dat is nu driemaal gemeten — spectrale verschuiving, hysterese
alleen, hysterese mét filter — en driemaal weerlegd.
