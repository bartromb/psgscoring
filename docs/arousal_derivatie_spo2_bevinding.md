# De multi-derivatie draaide op de saturatiecurve

**Datum:** 2026-08-25
**Aanleiding:** `docs/arousal-recall-diagnose.md` (D4), casus ANON_D056849C
**Status:** gerepareerd, niet uitgerold. Productie (0.27.3) draagt de bug.

---

## Wat de provenance zei

```
n_derivations: 2
derivations:   ['C3', 'SpO2']
n_per_derivation: {'C3': 142, 'SpO2': 0}
```

`_pick_eeg_multi` zoekt de occipitale afleiding met onder meer een **kaal
`"O2"`**, en `"SPO2"` bevat `"O2"`. De tweede EEG-afleiding was dus de
saturatiecurve. Dezelfde val die `_ROLE_MAY_NOT_TAKE["eeg"]` in
`detect_channels` al afvangt — deze functie had haar eigen zoektocht en die
guard ontbrak.

Twee gevolgen, en het tweede weegt zwaarder:

1. De arousaldetectie draaide een volledige analyse op een saturatiecurve. Dat
   het daar nul events gaf is **geluk, geen ontwerp**: een SpO2-curve met
   dalingen levert "arousals" die als onzin niet herkenbaar zijn.
2. **`C4` stond in dezelfde raw en werd nooit overwogen.** De picker zocht
   alleen occipitaal en frontaal, nooit een tweede CENTRALE afleiding. De
   gangbaarste klinische montage (C3 + C4) kreeg dus geen union, terwijl de
   provenance `n_derivations: 2` meldde alsof multi gewerkt had.

## Wat de reparatie oplevert

Zelfde opname, exact de productiekanalen, alleen de picker verschilt:

| | afleidingen | AI | resp-arousals / events | fractie |
|---|---|---:|---|---:|
| vóór (klinische run) | C3 + SpO2 | 19,5 | 76 / 373 | 0,204 |
| ná, werkpunt 0,80 | **C3 + C4** | **24,5** | 97 / 377 | **0,257** |

Beide afleidingen dragen nu bij (C3: 142, C4: 115 op 0,80). De kandidatenpool
verdubbelt van 1020 naar 2020.

Het volledige beeld op deze opname ná de reparatie:

| arm | AI | koppelingsfractie |
|---|---:|---:|
| regelgebaseerd | 87,8 | 0,893 |
| 0,50 | 47,7 | 0,468 |
| 0,60 | 40,7 | 0,416 |
| 0,70 | 32,1 | 0,339 |
| 0,80 (huidig) | 24,5 | 0,257 |

## Wat dit NIET oplost

De koppelingsfractie gaat van 20,4 % naar 25,7 % — een echte verbetering, maar
het regelgebaseerde pad haalt op dezelfde opname 89,3 %. De filter verwijdert
nog steeds het merendeel.

En 89,3 % is zelf verdacht: dat ligt bóven het fysiologisch verwachte bereik
van 60–80 %. Het regelgebaseerde pad is bovendien per opname onvoorspelbaar —
op PSG-IPA geeft het 203 tegen een menselijke mediaan van 46 (SN2) en 61 tegen
142 (SN3). "Zet de classifier uit" is dus geen route.

## Gevolg voor de werkpuntkeuze

De PSG-IPA-sweep (12 scoorders, vijf opnames) draaide op **één** afleiding en
gaf: werkpunt 0,80 haalt 3/5 binnen de scoordersspreiding met count-ratio
0,68; 0,50 haalt 5/5 met 1,26; 0,70 is het best gecentreerd (0,93, 4/5).

Die sweep is **niet meer geldig voor productie** nu de derivatiekeuze
gerepareerd is: met een werkende union stijgen de tellingen, en het optimale
werkpunt schuift mee. Een drempel kiezen op de oude configuratie zou
kalibreren op iets dat niet meer bestaat.

**Volgorde:** eerst deze reparatie uitrollen of in elk geval vastzetten, dan de
sweep opnieuw op de multi-configuratie, en pas dan een werkpuntadvies.
