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
