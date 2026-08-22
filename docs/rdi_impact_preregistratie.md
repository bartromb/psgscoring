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

---

# Uitkomst — 22 augustus 2026

**De meldgrens is gehaald: de RDI-ernstklasse verschuift op 28 % van de
opnames. Apart voorgelegd aan de gebruiker.**

Gepaard over 50 MESA-opnames, `aasm_v3_breath`, seed 20260801.

| | classifier uit | classifier aan | verschil |
|---|---:|---:|---:|
| AHI mediaan | 16,4 | 15,9 | −0,4 |
| **RDI mediaan** | **37,2** | **30,0** | **−7,2** |
| arousals mediaan | 225,5 | 179,0 | −46,5 |
| RERA-index mediaan | 17,0 | 11,1 | −5,9 |
| RERA-aandeel van RDI | 47 % | 45 % | |

Gepaarde ΔRDI: mediaan **−5,8/u**, 10e percentiel −17,0, 90e +5,3. Omlaag op
34 van 50, omhoog op 16.

| | verschuift |
|---|---|
| **RDI-ernstklasse** | **14/50 (28 %)** |
| AHI-ernstklasse | 5/50 (10 %) |

## Het mechanisme is consistent met de rest van de dag

De classifier verwerpt ongeveer een vijfde van de arousals (225 → 179), en dat
slaat door naar de RERA's (17,0 → 11,1). Bijna de helft van de RDI bestond uit
RERA's, en elke RERA vereist een arousal binnen 15 s. Op PSG-IPA ging de
precisie van 0,248 naar 0,425 — grofweg de helft van wat het regelpad
markeerde bleek geen arousal — en die vallen nu uit de RERA-telling.

## Wat hier NIET mee bewezen is

Dat de nieuwe RDI juister is. **MESA annoteert geen RERA's**, dus er is geen
waarheid om tegen te toetsen. Dat de verdwenen arousals precies degene zijn
die tegen twaalf scoorders niet standhielden, is een redenering — geen meting.

Wat wél vaststaat is de omvang: op ruim een kwart van de patiënten valt de RDI
in een andere categorie dan voorheen.

## Reikwijdte en terugweg

Het raakt de vier profielen met `arousal_limb_wired=True` — `aasm_v3_breath`,
`aasm_v3_prob` en hun duale varianten. **Niet** `aasm_v3_rec`: daar is
RDI = AHI, aantoonbaar onbewogen.

De terugweg is één vlag: `arousal_lgbm=False` op die profielen zet de oude RDI
terug zonder de arousalverbetering elders op te geven.

## Openstaand

Een RERA-referentie. Zonder die is elke RDI-uitspraak een uitspraak over
verschuiving, niet over juistheid. PSG-IPA `Resp_events/` bevat twaalf
onafhankelijke respiratoire scoringen — of daar RERA's in zitten is niet
onderzocht en is de goedkoopste volgende stap.
