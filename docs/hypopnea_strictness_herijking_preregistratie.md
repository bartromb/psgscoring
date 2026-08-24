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

---

# Uitkomst — 24 augustus 2026

**Alle drie de criteria gehaald. 0,50 is aantoonbaar niet het juiste
werkpunt.**

## Kalibratie (MESA n=15, zaad 20260826, disjunct)

| strictness | F1 | AHI-bias | events |
|---:|---:|---:|---:|
| 0,20 | 0,578 | +3,30 | 95 |
| **0,30** | **0,588** | **+0,40** | 78 |
| 0,40 | 0,556 | −0,82 | 73 |
| 0,50 (huidig) | 0,534 | −3,50 | 63 |
| 0,60 | 0,500 | −3,50 | 54 |
| 0,70 | 0,481 | −5,26 | 48 |
| 0,80 | 0,424 | −5,96 | 38 |

Binnenoptimum, geen randgeval — het ruime grid van de les van gisteren werkte.

## Validatie (MESA n=30, zaad 20260824, ongezien)

| | 0,50 | 0,30 | verschil |
|---|---:|---:|---:|
| event-F1 mediaan | 0,481 | **0,539** | +0,057 |
| AHI-bias mediaan | −3,28 | **−0,05** | +3,23 |
| \|AHI-bias\| mediaan | 7,20 | **4,50** | −2,71 |
| events mediaan | 89 | 117,5 | +28,5 |

Gepaarde ΔF1 **+0,0346**, beter op **26 van 30**, Wilcoxon **p = 2,6·10⁻⁷**.
AHI-ernstklasse verschuift op 7/30 = **23 %**, onder de meldgrens.

Dit is een **beide-kanten-winst**: de events kloppen beter én de systematische
onderschatting van de AHI verdwijnt vrijwel.

## Replicatie (PSG-IPA n=5, twaalf scoorders)

| | scoorder | F1 0,50 → 0,30 | events | AHI |
|---|---:|---|---|---|
| SN1 | 34 | 0,667 → 0,621 | 29 → 40 | 5,0 → 6,9 |
| SN2 | 21 | 0,388 → 0,413 | 24 → 42 | 4,9 → 8,7 |
| SN3 | 327 | 0,898 → 0,891 | 323 → 338 | 53,3 → 55,8 |
| SN4 | 23 | 0,252 → 0,333 | 12 → 25 | 2,0 → 4,2 |
| SN5 | 70 | 0,461 → 0,470 | 69 → 101 | 9,8 → 14,4 |

Gepaarde ΔF1 **+0,0090**, beter op **3 van 5**. Het teken repliceert, dus
criterium 3 is gehaald.

**Maar zwak, en dat hoort erbij.** Op MESA is de winst +0,035 op 26 van 30; op
PSG-IPA +0,009 op 3 van 5, met SN1 en SN3 licht slechter. Waar 0,50 al goed
zat — SN5 telde 69 tegen een scoordermediaan van 70 — schiet 0,30 door naar
101.

Dat is hetzelfde patroon als bij de arousaldrempel: de winst is groot waar het
werkpunt verkeerd stond en licht negatief waar het al klopte. Een vaste
drempel kan dat niet oplossen.

## Uitkomst volgens de regel

Aangenomen. Maar het is een klinische verschuiving — de AHI stijgt en de
ernstklasse beweegt op 23 % van de opnames — en dat is geen beslissing die uit
een regel volgt.

**Niet geïmplementeerd, niet uitgerold.** Ligt voor.
