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
| **MESA** (200 opnames) | een kanaal dat `Snore` heet, maar op **32 Hz** — Nyquist 16 Hz, onder de snurkband | **nee** — 21366 kale `Hypopnea`, nul centraal | de regel antwoordde (62,2 % centraal) op een kanaal dat snurken niet kan bevatten |
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

## De snurkmaat: NIET weerlegd, nooit getoetst

Op MESA, 24 opnames, tegen menselijke hypopnee-labels en flow-gematchte normale
ademhaling:

| snurkmaat | tijdens hypopneus | normale ademhaling |
|---|---:|---:|
| absolute drempel (60e percentiel van de nacht) | 30,1 % | 39,7 % |
| lokale referentie (mediaan RMS tegen de 120 s ervóór, ratio 1,30) | 10,2 % | 11,3 % |

Ik las dat eerst als een weerlegging van criterium 1. Dat was fout, en de
correctie is belangrijker dan de meting.

**Het MESA-kanaal dat `Snore` heet is bemonsterd op 32 Hz.** Dat staat in de
EDF-kop, niet in wat MNE teruggeeft: bij een gemengde-frequentie-EDF tilt MNE
alle kanalen naar de hoogste (hier 256 Hz), waardoor het kanaal breed lijkt.
Nyquist ligt op **16 Hz**.

Snurken is akoestische/vibratie-energie met een grondtoon rond 30 tot 250 Hz.
Een kanaal met een Nyquist van 16 Hz kan dat niet representeren — niet slecht,
maar principieel niet. Alles wat hierboven staat gaat dus over een
laagfrequente envelope van onbekende herkomst.

De sensortest bevestigt dat onafhankelijk. Op 37 opnames met **menselijke
apneu-subtypes** (930 centraal, 2938 obstructief, beide binnen dezelfde nacht,
schaalvrij genormaliseerd):

| venster | AUC | gepaard per opname |
|---|---:|---|
| vóór het event | **0,486** | obstructief luider op 14/37, p = 0,32 |
| tijdens | 0,566 | 27/37, mediaan verschil **+0,002**, p = 0,0001 |
| herstel | 0,543 | 24/37, mediaan verschil +0,020, p = 0,009 |

Juist in het venster waar de hypothese het sterkst is — een obstructieve apneu
zit ingebed in snurkende ademhaling — staat AUC 0,486, oftewel niets. En het
significante verschil "tijdens" is +0,002 op een schaal waar de mediaan 0,95
is: met bijna vierduizend events wordt tweetiende procent significant. Dat is
significantie zonder effect.

Dit is dezelfde fout als bij de RIP-poort, die de EDF-eenheid mat in plaats van
de ademhaling, en bij de vlakke-detectie, die het bestand mat in plaats van de
patiënt. **De meting mat het bestand.**

### Wat daarop is gebouwd

`_snore_during` weigert nu een kanaal waarvan de Nyquist onder de snurkband
ligt: `SNORE_MIN_SF_HZ = 60.0`, want om 30 Hz te representeren moet Nyquist
daarboven liggen. Geen gekozen marge maar de bemonsteringsstelling. De functie
geeft dan `None` — "niet gemeten" — en niet een berekende onwaarde die als
"niet gesnurkt" zou doorwerken.

Gevolg: op MESA-klasse hardware onthoudt de subtypering zich voortaan, net als
op PSG-IPA. Dat is het eerlijke antwoord, en het maakt de eerdere 62,2 % op
MESA ongeldig — dat getal kwam uit een criterium dat op een 32 Hz-kanaal
"niet gesnurkt" concludeerde.

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
