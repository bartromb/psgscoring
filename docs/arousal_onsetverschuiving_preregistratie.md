# Voorregistratie — onsetverschuiving op F1

Vastgelegd 2026-08-26, **vóór** de meting. Aanleiding: bijvangst van
`arousal_kandidaatdekking_bevinding.md`.

## Wat er gemeten wordt

De kandidaatdekking piekt op vier van de vijf opnames bij **+2 s** in plaats van
0 (SN1 87→90 %, SN3 50→56 %, SN4 74→77 %, SN2 en SN5 vlak). Mechanisme dat
daarbij past: het vermogensvenster is 2 s en links uitgelijnd, dus de kandidaat
begint waar het vénster begint terwijl de scoorder markeert waar hij de
verschuiving ziet.

Ingreep: alle gedetecteerde arousals over een vaste verschuiving Δ opschuiven,
en de event-F1 tegen de twaalf scoorders opnieuw meten. De detectie zelf draait
**één keer** per opname; de verschuivingen worden op dezelfde events toegepast,
zodat elk verschil de verschuiving is en niets anders.

Δ ∈ {−2, −1, 0, +1, +2, +3, +4, +6} s. De hele reeks wordt gerapporteerd om de
vorm te tonen; de **beslissing gaat uitsluitend over Δ = +2 s**.

## Waarom dit geen vrijbrief is

Δ = +2 s is gekozen op dekking, op ditzelfde cohort van vijf. Dat is geen
onafhankelijke set. Twee dingen beperken de schade:

1. Dekking en F1 zijn verschillende grootheden — dekking vraagt alleen of er
   íets in de buurt ligt en telt precisie niet mee. Het event-locked venster
   liet zien dat die twee uiteen kunnen lopen: koppeling +7,2 punt bij F1
   +0,018.
2. Er is een mechanisme dat de richting vóóraf voorspelt (links uitgelijnd 2
   s-venster). Een verschuiving die de andere kant op wint, is een toevalstreffer.

Een positieve uitkomst hier is daarom een **aanwijzing**, geen bewijs, en moet
op MESA repliceren voordat er iets default gaat.

## Criteria — beide moeten gehaald

Na de v6-les gebruik ik niet meer de mediaan van vijf opnames (die ÍS één
opname), maar:

1. **Primair.** Gemiddelde paarsgewijze ΔF1 bij Δ = +2 s > 0, **én** verbetering
   op **≥ 4 van de 5** opnames (tekentoets).
2. **Vorm.** Het maximum van de reeks ligt op +1, +2 of +3 s. Ligt het op −2 of
   +6, dan meet ik ruis of iets anders dan het veronderstelde mechanisme, en
   telt +2 niet mee ook al haalt hij criterium 1.

**Bewaker.** De telling mag niet veranderen. Een verschuiving verplaatst events
en maakt er geen; verandert het aantal wél, dan zit er een fout in het harnas.

## Vooraf: wat een positief resultaat betekent

**Niet** dat er iets uitgerold wordt. De ingreep zou achter een profielvlag gaan
met het huidige gedrag als default, en zou eerst op MESA moeten repliceren.
Een verschuiving verandert de gerapporteerde onsets in het klinische rapport.
