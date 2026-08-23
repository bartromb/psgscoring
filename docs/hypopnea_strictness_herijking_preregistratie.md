# Preregistratie — `hypopnea_strictness` opnieuw ijken, met gescheiden steekproeven

*23 augustus 2026, vóór enige kalibratie.*

## Waarom

`hypopnea_strictness = 0,50` is het werkpunt van de gegradeerde
hypopneedetector op vier profielen, en het bepaalt rechtstreeks de AHI. Onze
eigen `scripts/validate_mesa.py` zegt er in regel 6–9 over:

> *"Het werkpunt van `aasm_v3_breath` (`hypopnea_strictness` 0,50) is gekozen
> op PSG-IPA — vijf opnames, en dezelfde vijf waarop het resultaat
> gerapporteerd werd. **Dat is een fit, geen validatie.**"*

Dat is woordelijk dezelfde situatie als `AROUSAL_LGBM_THRESHOLD = 0,60`, die
op 23-08 herijkt bleek te moeten worden: die was óók op de evaluatieset
gekozen, en de herijking gaf op de validatieset F1 0,421 → 0,543.

Dit is de laatste bekende knop van dat type, en hij zit op de respiratoire as.

## Twee lessen van vandaag, hier vooraf toegepast

**1. Het grid moet ruim.** De arousaldrempelveeg (0,50–0,90) eindigde op zijn
eigen grenspunt, en pas na uitbreiding bleek 0,90 een echt maximum. Daarom
hier meteen **0,20 t/m 0,80** in stappen van 0,10, zeven punten.

**2. De uitkomstmaat moet vooraf vastliggen én verdedigd worden.** Bij de
arousaldrempel bleken event-F1 en de eventtelling verschillende optima aan te
wijzen (0,90 tegen 0,80), en dat werd pas ná de meting zichtbaar.

Voor de hypopneedetector kies ik **event-F1 als primaire maat, met de AHI-bias
als vetorecht** — dezelfde vorm die bij de sensorafhankelijke apneudrempel
correct weigerde een winst te boeken die alleen op bias rustte. Reden: de AHI
is de klinische uitkomst, maar een werkpunt dat de bias sluit door events op
de verkeerde plek te tellen verbetert niets. F1 vraagt dat de events kloppen;
de bias bewaakt dat de index niet wegloopt.

## Opzet

- **Kalibratie:** MESA n=15, zaad **20260826**, disjunct van 20260824 (n=30)
  en 20260825 (n=15) die al gebruikt zijn.
- **Validatie:** MESA n=30, zaad 20260824 — de set waarop vandaag ook de
  arousalbeslissing is afgerekend, en waarop het huidige werkpunt 0,50 al
  gemeten is.
- **Replicatie:** PSG-IPA n=5, tekencontrole met twaalf scoorders.
- Profiel `aasm_v3_breath`, volle pipeline, artefact-epochs zoals productie,
  huidige uitgerolde stand (classifier aan, 0,80).

## Keuzeregel — vooraf

Gekozen wordt de strictness met de **hoogste mediane event-F1 op de
kalibratieset**. Bij een verschil kleiner dan 0,005 tussen de beste twee wint
de waarde die de **|AHI-bias| kleiner** maakt.

Landt het optimum op een grenspunt van het grid, dan wordt het grid op
dezelfde kalibratieset uitgebreid — vastgelegd vóór de validatie, nooit erna.

## Beslisregel — vooraf

De nieuwe waarde vervangt 0,50 **alleen als alle drie**:

1. mediane **gepaarde** ΔF1 op de validatieset ≥ **+0,010**;
2. de mediane |AHI-bias| verslechtert **niet** met meer dan **1,0/u**;
3. het teken van ΔF1 repliceert op PSG-IPA.

**Weerlegd** bij ΔF1 ≤ 0. Daartussen: onbeslist, 0,50 blijft.

## Meldgrens

Verschuift de **AHI-ernstklasse** op meer dan een kwart van de
validatieopnames, dan leg ik dat apart voor.

## Wat dit niet uitwijst

Of de gegradeerde detector zelf de juiste vorm heeft. Een werkpunt verschuift
alleen langs een bestaande curve. En het blijft één vaste waarde voor alle
opnames — terwijl de arousalmeting van vandaag liet zien dat de fout per
opname in beide richtingen loopt. Een werkpunt dat per opname meebeweegt is
een andere en grotere vraag; die staat hier expliciet buiten.
