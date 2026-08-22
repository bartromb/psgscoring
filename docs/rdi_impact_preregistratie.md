# Preregistratie — hoeveel beweegt de RDI door de arousal-classifier?

Datum: 2026-08-22. **Geschreven vóór de meting.**

## Waarom, en waarom pas nu

De v0.24.0-release noemde de RDI-impact ongemeten. Twee dingen moesten eerst
recht:

1. **De reikwijdte was verkeerd opgeschreven.** Ik schreef dat
   `_compute_rera_rdi()` de arousallijst rechtstreeks leest en dat de RDI dus
   op elk profiel met de classifier beweegt. Onjuist: hij krijgt dezelfde
   `arousals`-variabele die `arousal_limb_wired` afknijpt. Op de zeventien
   profielen waar die vlag uit staat — `aasm_v3_rec` incluis — is die lijst
   leeg en geldt **RDI = AHI**. Gemeten op drie MESA-opnames: 10,5 · 5,3 ·
   18,5 met RDI gelijk aan AHI, bij 9, 59 en 320 gedetecteerde arousals.
2. **Op `aasm_v3_rec` viel er dus niets te meten.** Een meting daar had per
   constructie nul verschil opgeleverd, en dat had ik als geruststelling
   kunnen lezen.

Waar de vlag WEL aan staat is de blootstelling juist groot: `aasm_v3_breath`
geeft op diezelfde drie opnames 13,6 → 14,5 · 6,5 → 9,2 · 22,2 → **48,4**.
RERA's kunnen de index dus meer dan verdubbelen.

## Wat hier NIET gemeten kan worden

**MESA annoteert geen RERA's.** De referentiesets (`aasm15`, `oahi3`,
`oahi4`) bestaan uit apneus en hypopneus; er is geen RDI-referentie. Er valt
dus geen bias of F1 tegen een waarheid te berekenen.

Dat maakt het criterium dat ik vanochtend vastlegde — "de absolute RDI-bias
verslechtert niet met meer dan 1,0/u" — **onberekenbaar**. Het veronderstelde
een referentie die niet bestaat. Ik vervang het hier in plaats van er achteraf
een passende lezing bij te zoeken.

## Wat er wel gemeten wordt

Een KARAKTERISERING van de verschuiving, gepaard, `aasm_v3_breath`,
classifier uit tegen aan, op MESA:

1. de gepaarde ΔRDI per opname — mediaan, spreiding, richting;
2. het aandeel opnames waarop de RDI-gebaseerde ernstklasse verschuift;
3. het aandeel van de RDI dat uit RERA's bestaat, in beide armen.

**Geen slagen of zakken.** Er is niets om juist tegen te zijn, dus de uitkomst
is een beschrijving van wat er in de kliniek veranderd is — niet een oordeel
of het beter werd. Wie de RDI op deze profielen gebruikt, hoort te weten hoe
groot de verschuiving is.

**Wanneer ik alarm sla:** verschuift de ernstklasse op meer dan een kwart van
de opnames, dan leg ik dat als apart punt voor in plaats van het in een
verslag te vermelden. Dat is een grens voor de MELDING, niet voor de
aanvaarding.
