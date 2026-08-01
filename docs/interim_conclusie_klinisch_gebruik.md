# Interim-conclusie — is `aasm_v3_breath` klaar voor de dagelijkse praktijk?

*Status per 2026-08-01. Branch `diag/arousal-branch`, niets gepusht, productie
draait onveranderd op `aasm_v3_rec`.*

## Kort antwoord

**Klaar om te testen op eigen opnames, nog niet om de productie-pin te
verzetten.** De verbetering is op drie onafhankelijke metingen bevestigd, maar
de twee validatiesets spreken elkaar op het absolute niveau zo sterk tegen dat
geen van beide een uitspraak toelaat over deze praktijkpopulatie.

## Wat vaststaat

Het gepaarde voordeel van de ademteug-detector is drie keer los van elkaar
gemeten en houdt elke keer stand:

| meting | opzet | uitkomst |
|---|---|---|
| PSG-IPA | 5 opnames, 12 scorers, werkpunt via vooraf vastgelegde regel | F1 0,343 → 0,434; percentiel p6 → p17 |
| MESA ronde 1 | 50 opnames, held-out, niets afgesteld | 31/50 beter, p = 0,0069 |
| MESA ronde 2 | 50 andere opnames, overlap 0, juiste referentie | 36/50 beter, p = 0,0016 |

Op PSG-IPA is de AHI-nauwkeurigheid het overtuigendst: de afwijking tot de
scorermediaan zakt van gemiddeld 1,84 naar **0,29 AHI-punt**, en geen enkele
opname wijkt meer dan 0,9 af (was 5,0).

| opname | scorers (mediaan) | spreiding | `aasm_v3_rec` | breath@0,50 |
|---|---|---|---|---|
| SN1 | 6,0 | 4,7 – 6,6 | 8,1 (+2,1) | **6,2 (+0,2)** |
| SN2 | 4,3 | 1,6 – 6,8 | 9,3 (+5,0) | **5,2 (+0,9)** |
| SN3 | 54,0 | 45,1 – 56,0 | 53,8 (−0,2) | **54,0 (+0,0)** |
| SN4 | 3,8 | 0,2 – 6,3 | 4,3 (+0,5) | **3,7 (−0,1)** |
| SN5 | 10,0 | 3,6 – 14,4 | 11,4 (+1,4) | **9,8 (−0,2)** |

## De tegenspraak die dit tegenhoudt

PSG-IPA zegt: de AHI klopt op een derde punt na. MESA zegt: de AHI ligt er
**16,5 punten** naast (bias −16,5 tegen `nsrr_ahi_hp3r_aasm15`). Dat kan niet
allebei waar zijn over dezelfde software.

De vermoedelijke oorzaak is de arousal-tak. Tegen een referentie die
arousal-gekwalificeerde hypopneeën meetelt, mist een detector die vooral op
desaturatie afgaat structureel een groot deel van de events — en MESA-opnames
zijn arousal-rijk. Op PSG-IPA scoren de twaalf humane scorers zelf
conservatiever, waardoor het gat niet zichtbaar wordt. Zolang die verklaring
niet is aangetoond, is onbekend welke van de twee cijfers voor déze praktijk
geldt.

Wat de richting bepaalt is dus welke populatie en welke scoortraditie de
referentie vormt. Geen van beide sets is die van deze kliniek.

## Overige beperkingen

- **Beide validatiesets zijn MESA of PSG-IPA.** Cross-cohort overdraagbaarheid
  is ongetoetst; SHHS is overwogen en afgevallen op datakwaliteit.
- **SN4 gaat achteruit** (F1 0,286 → 0,184) en geen enkele instelling repareert
  dat: van de 24 consensus-events worden er 12 nooit kandidaat, omdat daar op
  ademteugniveau geen daling van ≥15% is die 10 s aanhoudt.
- **Waar scorers het eens zijn, haalt het algoritme ze niet in.** Op SN1 gaat
  het naar F1 0,662 terwijl de mediane scorer 0,826 haalt — nog steeds onder
  alle 66 menselijke paren.
- **Hypopnee-subtypering vervalt.** `hypopnea_central` en `hypopnea_mixed`
  komen niet meer voor; de detector vervangt alleen de hypopneeën en levert
  geen effort-gebaseerde subtypering. Apneus behouden die wel.
- **De severity-concordantie is een grove maat.** Op SN2 geeft de detector 5,2
  bij een referentie van 4,3 — formeel fout, terwijl de twaalf scorers zelf
  tussen 1,6 en 6,8 liggen en dus zelf over de grens van 5 heen en weer vallen.

## Advies

1. **Productie-pin niet verzetten** zolang de tegenspraak tussen de twee sets
   niet is opgehelderd. YASAFlaskified geeft `scoring_profile="standard"`
   expliciet mee, dus het gedeployde gedrag verandert vanzelf niet.
2. **Wel draaien op eigen opnames**, naast het huidige profiel, en een
   handvol gevallen visueel nakijken — met name gevallen rond de AHI-grens van
   5 en 15, waar een verschuiving van klasse verandert. Het EC-protocol
   AZORG-YASA-2026-001 is goedgekeurd en biedt daarvoor de route.
3. **Eerst uitzoeken waarom MESA en PSG-IPA 16 AHI-punten uit elkaar liggen.**
   Zolang dat onverklaard is, is elke uitspraak over klinische
   nauwkeurigheid afhankelijk van welke set je toevallig citeert.
