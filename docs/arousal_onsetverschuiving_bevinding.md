# Onsetverschuiving +2 s — gemeten op F1, criteria gehaald

Gemeten 2026-08-26 tegen `arousal_onsetverschuiving_preregistratie.md`.

## Uitkomst

Event-F1 tegen de twaalf scoorders, mediaan per opname. De detectie draaide
**één keer** per opname; de verschuivingen zijn daarna op dezelfde eventlijst
toegepast, dus elk verschil is de verschuiving en niets anders.

| verschuiving | SN1 | SN2 | SN3 | SN4 | SN5 | gem. Δ | beter |
|---|---:|---:|---:|---:|---:|---:|---:|
| −2 s | 0,444 | 0,341 | 0,295 | 0,501 | 0,684 | −0,0745 | 0/5 |
| −1 s | 0,494 | 0,377 | 0,375 | 0,550 | 0,698 | −0,0287 | 0/5 |
| **0 s** *(nu)* | 0,514 | 0,396 | 0,419 | 0,601 | 0,708 | — | — |
| +1 s | 0,525 | 0,417 | 0,428 | 0,606 | 0,710 | +0,0097 | 5/5 |
| **+2 s** | **0,524** | **0,417** | **0,437** | **0,611** | **0,710** | **+0,0123** | **5/5** |
| +3 s | 0,510 | 0,428 | 0,433 | 0,606 | 0,701 | +0,0081 | 3/5 |
| +4 s | 0,482 | 0,419 | 0,419 | 0,591 | 0,674 | −0,0107 | 2/5 |
| +6 s | 0,379 | 0,373 | 0,354 | 0,538 | 0,581 | −0,0826 | 0/5 |

**Criterium 1 gehaald:** gemiddelde paarsgewijze Δ = **+0,0123** op **5/5**.
**Criterium 2 gehaald:** het maximum van de reeks ligt op **+2 s**, binnen het
vooraf genoemde venster (+1/+2/+3). De curve is glad en eentoppig.

**Bewaker gehaald:** de telling is over alle verschuivingen gelijk (112 / 48 /
74 / 106 / 222), zoals het hoort — een verschuiving verplaatst events en maakt
er geen.

Dit is de **eerste ingreep in deze reeks die zijn voorregistratie haalt**. Ter
vergelijking: v4, v5 en v6 haalden hem geen van drieën, en het beste
drempelorakel gaf +0,008 (tegen +0,012 hier, zonder orakel).

## Correctie: mijn verklaring was fout

In de voorregistratie schreef ik dat het vermogensvenster **links uitgelijnd**
is en dat de kandidaat daarom begint waar het venster begint. Dat is onjuist.
`_bandpower_instant()` zet de waarde op het **midden** van het venster:

```python
centers[i] = s + win // 2
power_full = np.interp(t_full, centers, powers)
```

Het venster is dus gecentreerd. Wat er wél gebeurt: bij een gecentreerd
2 s-Hanningvenster weerspiegelt het vermogen op tijdstip *t* het interval
[*t*−1, *t*+1], dus het begint ongeveer **1 s vóór** de werkelijke
frequentieverschuiving te stijgen. De onset is `indices[0] / sf` — het eerste
sample waar het gladgestreken vermogen de drempel passeert — en die passage
gebeurt daardoor ~1 s te vroeg. Dat verklaart ongeveer de helft.

De andere ~1 s is **niet verklaard**. Een plausibele kandidaat is dat scoorders
markeren waar zij de verschuiving herkennen, wat na het fysieke begin ligt. Dat
is niet gemeten en blijft dus een vermoeden.

**Gevolg voor de bewijskracht.** Criterium 2 was bedoeld om te voorkomen dat ik
een toevallige piek zou binnenhalen, en ik onderbouwde het met een mechanisme
dat de richting vooraf zou voorspellen. Dat mechanisme klopt maar half. De
empirische vorm — glad, eentoppig, maximum op +2, symmetrisch verval naar beide
kanten — staat overeind en is op zichzelf al lastig als toeval te lezen. Maar de
voorspelling was zwakker dan ik hem opschreef.

## Wat dit niet is

- **Geen uitrol.** De ingreep hoort achter een profielvlag met het huidige
  gedrag als default. Een verschuiving verandert de gerapporteerde onsets in het
  klinische rapport.
- **Nog niet gerepliceerd.** n = 5, en +2 s is gekozen op de dekkingsmeting op
  ditzelfde cohort. MESA is de replicatieset; zonder die replicatie is dit een
  aanwijzing, geen bevinding.
- **Geen oplossing voor SN3.** SN3 gaat van 0,419 naar 0,437 — de grootste
  winst van de vijf, maar het gat naar het plafond 0,692 blijft 0,25, en de
  pooldekking blijft 50 %.

## Reproductie

`scripts/sweep_arousal_onset_offset_psgipa.py`; ruwe cijfers in
`arousal_onsetverschuiving_f1.json`.
