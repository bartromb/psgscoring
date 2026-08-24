# Een werkpunt dat per opname meebeweegt: getoetst op zijn plafond, en verworpen

*24 augustus 2026, nacht. Geen implementatie gebouwd — de vraag is beantwoord
vóór er code was.*

## De gedachte

Op twee assen kwam hetzelfde patroon terug. De arousaldrempel en de
hypopneu-strictness winnen allebei **fors waar het werkpunt verkeerd stond** en
verliezen **licht waar het al klopte**:

- arousals: SN2 5,53× overdetectie → 0,88, maar SN3 0,78 → 0,47;
- hypopneu: SN4 12 → 25 tegen een scoordermediaan van 23, maar SN5 69 → 101
  tegen 70.

Een vaste drempel kan dat per constructie niet oplossen. De voor de hand
liggende conclusie is dat de **vorm** niet deugt, niet de waarde: een werkpunt
dat per opname meebeweegt met iets dat vóór het scoren bekend is, zou beide
staarten sluiten.

## De toets: eerst het plafond, dan pas bouwen

Uit de kalibratieveeg (15 opnames × 7 werkpunten) is per opname het beste
werkpunt af te lezen. Dat geeft een **orakel** — de bovengrens van wat elke
adaptieve regel ooit kan halen.

| | mediane F1 | winst |
|---|---:|---:|
| vast 0,50 (huidig) | 0,534 | — |
| vast 0,30 (voorgesteld) | 0,588 | +0,054 |
| **per-opname orakel** | **0,607** | **+0,072** |

**Het plafond is +0,072, en een vaste 0,30 pakt daar al +0,054 van — 75 %.**
Er is dus hooguit 0,019 te winnen met een perfecte adaptieve regel, en een
perfecte regel bestaat niet.

## En de realiseerbare regel is slechter dan de vaste

Het per-opname optimum spreidt van 0,20 tot 0,70 en correleert met de
referentie-AHI (r = −0,514). Maar die is bij het scoren niet bekend. De
waarneembare proxy is het aantal events bij een neutraal werkpunt, en die
correleert zwakker: **r = −0,412**.

Negen tweetrapsregels op die proxy, alle **in-sample** gemeten en dus
optimistisch:

| regel | F1 | t.o.v. 0,50 |
|---|---:|---:|
| events > 45 → 0,20, anders 0,40 | 0,562 | +0,028 |
| events > 45 → 0,30, anders 0,50 | 0,556 | +0,022 |
| events > 63 → 0,20, anders 0,50 | 0,516 | −0,018 |
| events > 124 → 0,30, anders 0,50 | 0,534 | 0,000 |

De beste haalt 0,562. **Vast 0,30 haalt 0,588** — en dat zonder de
optimistische vertekening van in-sample meten.

## Uitkomst

Verworpen. Niet omdat het idee onlogisch is, maar omdat het plafond te laag
ligt en de waarneembare voorspeller te zwak. Een adaptief werkpunt zou meer
complexiteit, een extra scoringsronde en een tweede te ijken parameter kosten
voor minder resultaat dan één getal veranderen.

## Wat dit wél leert

De resterende fout is **niet** een kwestie van het werkpunt. Met een perfect
per-opname gekozen drempel blijft de mediane F1 op 0,607 steken, tegen een
menselijke bovengrens die op deze as veel hoger ligt. Wat overblijft zit dus in
de **detector** — welke kandidaten hij überhaupt genereert en hoe hij ze
weegt — niet in waar de streep ligt.

Dat is een bruikbare afbakening: elke volgende poging op deze as die aan de
drempel draait, draait aan de verkeerde knop.
