# Preregistratie — wat doet de arousal-classifier met de RDI op de RERA-dragende profielen?

*23 augustus 2026, ná een geslaagde pre-flight en vóór de uitkomst.*

## De openstaande beslissing

De classifier staat sinds 22-08 **uit** op de vier profielen waar arousals de
RDI dragen (`aasm_v3_breath`, `aasm_v3_prob`, + duals). Reden toen: hij
verschoof de RDI-ernstklasse op 28 % van de opnames en er is geen
RERA-referentie om te bepalen welke RDI juister is.

Sindsdien is er iets veranderd: het **werkpunt** is herijkt van 0,60 naar 0,80,
en 0,60 was aantoonbaar gedomineerd. De vraag is dus opnieuw open — maar nu met
een classifier die op een ander punt draait dan toen de beslissing viel.

Deze opnames draaien op deze profielen ook in de kliniek; het is de configuratie
waar de RDI aan hangt.

## Waarom de vorige poging niets mat

Twee oorzaken, allebei van mij:

1. `aasm_v3_breath` heeft `arousal_lgbm=False`, dus de drempel deed daar niets;
2. de pipeline las `AROUSAL_LGBM` **zonder env-override** — als enige
   arousalvlag — zodat de armen niet te scheiden waren.

Resultaat: 30/30 identiek, wat eruitzag als "geen effect". De override is nu
toegevoegd, met een test die faalt als de armen gelijk uitkomen.

**Pre-flight gedaan** op `mesa-sleep-0407` vóór deze preregistratie werd
afgerond: classifier uit → AHI 66,5 / RDI 67,0 / 42 arousals; aan → AHI 64,9 /
RDI 65,4 / 30 arousals. De armen verschillen dus aantoonbaar.

## Opzet

- MESA n=30, zaad 20260824, gepaard.
- Profiel `aasm_v3_breath` — RERA-dragend, en de klinische werkpaard.
- **Artefact-epochs meegegeven** (`PSGSCORING_HARNESS_ARTIFACT_EPOCHS=1`),
  zodat de meting draait zoals productie draait. Dat was tot vandaag niet zo.
- Overige stand = uitgerold: arousalstap negeert de lijst, drempel 0,80.

| arm | classifier |
|---|---|
| A | uit (huidige stand op deze profielen) |
| B | aan, werkpunt 0,80 |

## Karakterisering, geen slaag/zak

Er is geen RERA-referentie, dus "juister" is niet meetbaar
([[reference_no_rera_reference]]). Wat gemeten wordt is de **omvang**: AHI,
RDI, arousal-index, RERA-index, en de gepaarde verschuivingen.

## Meldgrens — vooraf

Verschuift de **RDI-ernstklasse** op meer dan een kwart van de opnames, dan leg
ik dat als apart punt voor in plaats van het in een verslag te vermelden.
Dezelfde grens en dezelfde reden als bij de vorige RDI-karakterisering, die
28 % gaf.

Ik noteer de AHI-ernstklasse er apart bij: op dit profiel bevestigen arousals
ook hypopneus, dus de AHI beweegt mee — de pre-flight liet 66,5 → 64,9 zien.
Dat maakt dit géén zuivere RDI-vraag, en dat hoort vooraf te staan.

## Wat dit niet uitwijst

Welke arm de betere RDI geeft. Alleen hoe groot het verschil is, zodat de
beslissing met dat cijfer erbij genomen wordt in plaats van erna.
