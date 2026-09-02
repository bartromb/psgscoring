# Hypopnee-subtypering: de twee cohorten vullen elkaar precies verkeerd aan

*Vastgelegd 2026-09-02.*

## De vondst in één tabel

AASM v3 §6.1 noemt een hypopnee obstructief bij ten minste één van drie
kenmerken: snurken tijdens het event, toegenomen inspiratoire afvlakking, of
paradox tijdens maar niet ervóór. Centraal is wat overblijft.

Om die regel te toetsen heb je twee dingen tegelijk nodig: een snurkkanaal en
menselijke hypopnee-subtypes. Wij hebben ze nooit samen.

| cohort | snurkkanaal | menselijke hypopnee-subtypes | gevolg |
|---|---|---|---|
| **MESA** (200 opnames) | ja, `Snore` op elke opname | **nee** — 21366 kale `Hypopnea`, nul centraal | de regel antwoordt (62,2 % centraal) maar er is niets om tegen te leggen |
| **PSG-IPA** (5 opnames, 12 scoorders) | **nee** — alleen `Resp nasal` | ja — 95 centrale van 1601 (5,9 %) | de regel onthoudt zich (70,2 % `uncertain`) |

MESA subtypeert wél apneus (4286 obstructief, 464 centraal). Alleen hypopneus
niet.

## Wat dit ongeldig maakt

Elk eerder getal van de vorm "wij zeggen X % centraal tegen een menselijk
ijkpunt van 5,9 %" vergelijkt een MESA-uitkomst met een PSG-IPA-referentie.
Dat zijn twee cohorten, twee montages en twee scoortradities. De vergelijking
zegt niets over juistheid.

## En het ijkpunt zelf is geen punt

De 5,9 % is een gemiddelde over twaalf scoorders die het onderling zeer oneens
zijn:

| opname | hypopneus | gemiddeld % centraal | spreiding over 12 scoorders |
|---|---:|---:|---|
| SN1 | 177 | 5,2 % | 0,0 – 37,5 % |
| SN2 | 213 | 16,3 % | **0,0 – 93,3 %** |
| SN3 | 213 | 2,1 % | 0,0 – 17,1 % |
| SN4 | 287 | 10,9 % | 0,0 – 50,0 % |
| SN5 | 711 | 0,4 % | 0,0 – 5,0 % |

Op SN2 noemt de ene scoorder geen enkele hypopnee centraal en een andere
negen van de tien. Paarsgewijs over tijdelijk gekoppelde hypopneus is de ruwe
overeenstemming 1,000 maar de kappa **0,000**: de gekoppelde paren bevatten
vrijwel geen centrale labels, dus de overeenstemming komt volledig uit de
meerderheidsklasse en niet uit kunde.

Dit is hetzelfde patroon als bij de arousals, waar het plafond 0,679 bleek en
niet 1,0 — maar scherper. Voor hypopnee-subtypering is er niet eens een stabiel
doel.

## De snurkmaat zelf: twee armen, beide weerlegd

Op MESA, 24 opnames, tegen menselijke hypopnee-labels en flow-gematchte normale
ademhaling:

| snurkmaat | tijdens hypopneus | normale ademhaling |
|---|---:|---:|
| absolute drempel (60e percentiel van de nacht) | 30,1 % | 39,7 % |
| lokale referentie (mediaan RMS tegen de 120 s ervóór, ratio 1,30) | 10,2 % | 11,3 % |

De absolute drempel markeerde snurken váker buiten de events dan erin. De
lokale referentie — die de contrastvorm van de andere twee criteria volgt, en
schaalvrij is — haalt dat weg maar scheidt nog steeds niets.

De lokale maat blijft in de bibliotheek: hij is aantoonbaar iets anders dan de
absolute (er staat een test die een nacht construeert die in de tweede helft
luider wordt, waar de absolute alles markeert en de lokale niets), en hij is de
enige van de twee die de manual's contrastvorm volgt. Maar hij is niet
gevalideerd.

## Wat dit betekent voor de vlag

`hypopnea_subtype_aasm` blijft **default uit**, nu met een reden die sterker is
dan "nog niet gemeten": op de data die wij hebben is hij niet te valideren.
De code zegt dat zelf al eerlijk — `criteria_unavailable`, `complete`, en
`uncertain` in plaats van een restbak — en dat gedrag hoort zo te blijven.

## Wat het wél zou kunnen valideren

De Somnomedics-opnames van de slaapkliniek dragen een snurkkanaal **en** worden
door mensen gescoord. Dat is het enige cohort in bereik waar beide kanten
tegelijk bestaan. Voor die validatie is nodig:

1. hypopneus die door de scoorder als obstructief of centraal zijn gelabeld
   (niet alleen kale hypopneus);
2. bij voorkeur meer dan één scoorder, anders is er opnieuw geen plafond;
3. het ruwe snurkkanaal, niet een afgeleide index.

Zonder punt 1 verschuift het probleem alleen van MESA naar de kliniek.
