# Preregistratie — wat doen de twee openstaande arousalvlaggen met de RDI?

*23 augustus 2026, vóór de meting. Karakterisering, geen slaag/zak.*

## Waarom dit vóór de beslissing hoort

Er liggen twee besluiten klaar:

1. **`arousal_uses_artifact_epochs = False`** — de arousalstap negeert de
   artefactlijst. Bewijs: 30/30 MESA (p = 1,7e-06), repliceert op PSG-IPA,
   verslaat óók `yasa.art_detect`.
2. **`arousal_lgbm_threshold` van 0,60 naar 0,80 of 0,90** — 0,60 is
   gedomineerd. Bewijs: validatie 0,421 → 0,543 bij 0,90 (24/30, p = 1,5e-05);
   0,80 wint op beide cohorten én houdt de eventtelling zuiver.

Beide zijn gemeten op **arousal-F1**. Geen van beide is gemeten op wat er in
het rapport belandt. Op de vier `arousal_limb_wired`-profielen voeden arousals
de RERA-detector, en RERA's waren daar ~46 % van de RDI. Toen de classifier
aanging verschoof de RDI-ernstklasse op **28 %** van de opnames — dat was de
vorige keer reden voor een apart gesprek, en het is geen reden om het deze keer
niet te meten.

De twee vlaggen duwen bovendien **tegengesteld** op de arousaltelling: de lijst
negeren geeft er méér, de drempel verhogen minder. Het netto-effect is niet af
te leiden uit de losse metingen en moet gemeten worden.

## Opzet

- MESA n=30, zaad 20260824 (dezelfde opnames als de arousalmetingen), gepaard.
- Profiel `aasm_v3_breath` — RERA-dragend, dus waar de RDI beweegt.
- Volle pipeline, want AHI, RDI en de ernstklassen komen daaruit.

| arm | artefactlijst | drempel |
|---|---|---|
| A | gebruiken (nu) | 0,60 (nu) |
| B | negeren | 0,80 |
| C | negeren | 0,90 |

## Wat gerapporteerd wordt

Mediane AHI, RDI, arousal-index en RERA-index per arm; gepaarde delta's; en
het aandeel opnames waarop de **AHI-ernstklasse** en de **RDI-ernstklasse**
verschuiven.

## Geen slaag/zak, één meldgrens

Dit is een karakterisering: er bestaat geen RERA-referentie, dus "juister" is
niet te meten (zie `reference_no_rera_reference`). Er is één drempel:

**verschuift de RDI-ernstklasse op meer dan een kwart van de opnames, dan leg
ik dat als apart punt voor** in plaats van het in een verslag te vermelden.
Zelfde grens en zelfde reden als bij de classifier.

## Wat dit niet uitwijst

Welke arm de betere RDI geeft. Dat kan niet zonder RERA-referentie. Wat het
wél geeft is de **omvang** van de verschuiving, zodat de keuze tussen 0,80 en
0,90 met dat cijfer erbij gemaakt wordt in plaats van erna.

`aasm_v3_rec` is bewust niet meegemeten: daar staat `arousal_limb_wired` uit
en geldt RDI = AHI, dus daar is de RDI per constructie onbewogen.
