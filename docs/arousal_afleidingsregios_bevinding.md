# Drie hersenregio's, niet twee kanalen uit dezelfde

**Datum:** 2026-08-25
**Cohort:** PSG-IPA, 5 opnames, 12 scoorders, arousal-F1 (IoU 0,20)
**Status:** gerepareerd in psgscoring 0.27.6 / YASAFlaskified 0.34.6

---

## Wat er misging

De pneumo-raw wordt gebouwd uit `detect_channels`, dat **één kanaal per rol**
teruggeeft. Op de klinische montage stonden daar `C3` en `C4` in — twee
kanalen uit **dezelfde regio** — terwijl hetzelfde EDF ook `O1`/`O2` en
`F3`/`F4` droeg. De arousalstap kiest zijn afleidingen uit wat er ÍS, dus een
frontale of occipitale afleiding bereikte hem nooit.

AASM V.A **Note 1** schrijft expliciet voor dat arousal-scoring frontale,
centrale én occipitale informatie betrekt. Wij scoorden op één van de drie.

## De meting

| combinatie | regio's | F1 | n |
|---|---:|---:|---:|
| **F+C+O** | 3 | **0,514** | 106 |
| F+C | 2 | 0,501 | 88 |
| F+O | 2 | 0,485 | 94 |
| C+O | 2 | 0,460 | 97 |
| F | 1 | 0,442 | 65 |
| C | 1 | 0,439 | 71 |
| O | 1 | 0,316 | 44 |

Van één naar twee regio's: **+0,06**. De derde doet er nog **+0,013** bij.
Menselijk plafond: 0,679 (330 scoorderparen).

**Het argument staat niet in de mediaan maar in de spreiding:**

```
SN4:  O=0,59  C=0,48  F=0,44    occipitaal wint
SN5:  C=0,66  F=0,61  O=0,61    centraal wint
SN2:  F=0,41  C=0,36  O=0,32    frontaal wint
```

Geen enkele regio wint overal. Occipitaal is gemiddeld de zwakste (0,316) maar
op SN4 de sterkste, en hij voegt in élke combinatie waarde toe (C 0,439 →
C+O 0,460; F 0,442 → F+O 0,485). Je weet vooraf niet welke afleiding déze
arousal het duidelijkst toont; dát is waarom AASM alle drie vraagt.

## De reparatie

`arousal_derivation_channels(ch_names, channel_map)` is publiek en werkt op
**namen**, niet op een geladen `raw`: een aanroeper moet kunnen bepalen welke
kanalen hij moet inlezen vóórdat hij ze inleest. `_pick_eeg_multi` gebruikt
diezelfde functie, zodat de opgevraagde set per constructie gelijk is aan wat
de detector gebruikt — met een test die faalt zodra de twee uiteenlopen.

De kennis hoort in psgscoring en niet in de app. Laat de app raden welke
kanalen de picker straks kiest, en het loopt mis zodra die verandert; dat is
precies wat er met de SpO2-afleiding gebeurde
(`arousal_derivatie_spo2_bevinding.md`).

## Een weerlegde hypothese onderweg

De scoorders koppelden 78 % van hun arousals aan `Cz-O2` en `Fpz-Cz` — twee
lange bipolaire ketens die niet in de gepubliceerde EDF zitten. Die zijn te
reconstrueren door aftrekken (`C4-O2 = (C4-M1) − (O2-M1)`), en het lag voor de
hand dat ze beter zouden detecteren.

Gemeten: **niet waar.** C4-O2 komt exact uit op C4-M1 (0,400), F4-C4 is
slechter (0,369) en produceert twee keer zoveel events.

De les: waar een scoorder een arousal MARKEERT zegt niets over waar het signaal
het beste te zien is. Waarschijnlijk is het gewoon de montage die zijn software
prominent toont.

## Wat nog niet gemeten is

De klinische winst zelf. Op PSG-IPA is de klinische situatie — twee kanalen uit
dezelfde regio — niet na te bootsen, want dat cohort heeft er één per regio.
De verwachte winst ligt tussen **+0,013 en +0,06**, afhankelijk van hoeveel
C3+C4 al als twee regio's telt. Te meten op een verse opname: de
provenance-regel toont sinds 0.34.5 de volledige afleidingsset.
