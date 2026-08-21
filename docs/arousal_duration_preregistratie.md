# Preregistratie — eventmorfologie in de arousaldetectie

Datum: 2026-08-21. **Geschreven vóór enige meting van de ingreep.**
Volgt op `docs/arousal_spectral_shift_preregistratie.md`, waarvan de
uitkomst dit defect blootlegde.

## Het defect

`detect_arousals` bouwt in fase 1 een mask per SAMPLE:

    arousal_mask[s:e] = arousal_pow[s:e] > ratio_thresh * local_bl

en labelt die daarna direct met `scipy.ndimage.label`. Er wordt **geen enkel
gat gedicht**. Bandvermogen fluctueert op subseconde-schaal, dus de mask
flikkert en één fysiologische arousal valt uiteen in losse stukken.

Gemeten op twee MESA-opnames:

| | ruwe regio's | halen ≥3 s | mediane duur |
|---|---:|---:|---:|
| mesa-sleep-1020 (referentie 63 events, mediaan 11,0 s) | 1897 | **65** | 3,7 s |
| — gaten < 5 s gedicht | 612 | 373 | 8,9 s |
| mesa-sleep-6157 (referentie 77 events, mediaan 8,0 s) | 2837 | **185** | 3,9 s |
| — gaten < 5 s gedicht | 825 | 540 | 9,4 s |

Ruim 96 % van de verhoogde regio's wordt weggegooid door de 3 s-eis, en wat
overblijft is niet de sterkste maar de toevallig langste aaneengesloten
scherf. Vandaar dat het AANTAL ruwweg klopt (57 tegen 63 op 1020) terwijl de
events op de verkeerde plek staan: mediaan 184 s van de dichtstbijzijnde
referentie.

Door mensen gescoorde arousals duren mediaan **8,6 s** (PSG-IPA, 7528 events
over twaalf scoorders) en **11,0 s** (MESA). De detector geeft **3,6 s** en
plakt daarmee tegen `AROUSAL_MIN_DUR_S = 3.0`. Een detector waarvan de
mediaan op zijn eigen ondergrens ligt, meet die ondergrens, niet de arousal.

Merk op dat dezelfde module dit elders wél goed doet: de
flowlimitatie-detectie sluit haar mask met `binary_closing`/`binary_opening`.
Het arousalpad niet.

## De ingreep

**Hysterese** in plaats van één drempel — de standaardvorm voor een
eventdetector, en dichter bij wat een scoorder doet: de arousal begint waar
de activiteit duidelijk verhoogd is en loopt door zolang ze verhoogd blijft.

    binnenkomen : arousal_pow > ratio_thresh * local_bl        (2,0 — ONGEWIJZIGD)
    doorlopen   : arousal_pow > exit_ratio  * local_bl         (NIEUW)

Achter profielvlag `arousal_hysteresis`, **default False**. Eén nieuwe
constante, geen tweede geregelde knop: de instapdrempel blijft exact 2,0,
zodat de ingreep alleen bepaalt waar een event EINDIGT, niet of het begint.

Vastgelegd vóór de meting, gekozen uit de vorm en niet uit de data:

| constante | waarde | betekenis |
|---|---|---|
| `AROUSAL_EXIT_RATIO` | **1,2** | een event loopt door zolang het vermogen 20 % boven de lokale rustige vloer blijft |

## Acceptatiecriterium (vastgelegd vóór de meting)

Gemeten op de vijf PSG-IPA arousal-opnames, twaalf scoorders, event-F1 met
greedy IoU-matching op 0,20 — hetzelfde harnas als de vorige preregistratie,
zodat de getallen naast elkaar liggen.

**Primair.** De mediane event-F1 stijgt van **0,182** naar **≥ 0,25**.

**Secundair (mechanisme).** De mediane eventduur van het algoritme komt in
**[6, 14] s** te liggen. Valt de F1-winst uit zonder dat de duur meebeweegt,
dan werkt er iets anders dan de veronderstelde oorzaak en telt de winst niet.

**Bewaking.** De spreiding `max(q)/min(q)` van `q = index_algoritme /
index_scoordermediaan` wordt niet slechter dan de huidige **10,13**.

**Replicatie — bindend.** Haalt een configuratie het primaire criterium op
PSG-IPA, dan gaat ze NIET default voordat ze op **MESA n = 150** repliceert:
F1-winst in dezelfde richting, gepaard over de opnames, p < 0,05. Dit is de
regel die `rectify_lowpass` in augustus tegenhield en die daar precies deed
waarvoor ze bedoeld was.

**Randvoorwaarden voor promotie:**
1. Primair, secundair en bewaking gehaald op PSG-IPA.
2. Replicatie op MESA gehaald.
3. Golden 9/9 byte-identiek met de vlag uit.
4. `mesa_shhs` en `chicago_1999` blijven gepind op het oude gedrag.

---

# Uitkomst — 21 augustus 2026

**Weerlegd. De vlag gaat niet default.**

| | spreiding max(q)/min(q) | mediane F1 | mediane duur |
|---|---:|---:|---:|
| huidig gedrag | 10,13 | 0,182 | 4,0 s |
| exit = 1,00 | 4,43 | 0,100 | 5,3 s |
| exit = 1,10 | 3,93 | 0,107 | 4,9 s |
| **exit = 1,20 (preregistratie)** | **3,86** | **0,111** | **4,6 s** |
| exit = 1,40 | 4,19 | 0,150 | 4,2 s |
| exit = 1,60 | 5,31 | 0,183 | 4,1 s |
| exit = 1,80 | 8,03 | 0,183 | 3,8 s |

Mensen: mediane duur 8,3 s, scoorder-tegen-scoorder F1 0,692.

- **Primair (F1 ≥ 0,25): NIET GEHAALD** — 0,111 tegen 0,182 nu.
- **Secundair (duur in [6,14] s): NIET GEHAALD** — 4,6 s.
- **Bewaking (spreiding ≤ 10,13): gehaald** — 3,86.

Het secundaire criterium deed hier zijn werk. De duur beweegt nauwelijks mee,
dus het veronderstelde mechanisme klopt niet: samenvoegen van scherven maakt
niet één lang event maar laat véél regio's die eerst onder de 3 s-eis vielen
alsnog toe. Het aantal verviervoudigt (index 21 → 83 op SN1) terwijl de events
kort blijven. Was alleen de F1 beoordeeld, dan had de gunstige
spreidingsdaling van 10,13 naar 3,86 makkelijk als winst kunnen doorgaan.

## Wat hier onder ligt

Twee ingrepen op rij, beide op het KANDIDAATstadium, beide weerlegd. Het
patroon dat overblijft is scherper dan elk van beide:

**De detector lokaliseert niet.** Hij levert ongeveer het juiste AANTAL events
van ongeveer de juiste soort, op de verkeerde momenten. Op één MESA-opname
ligt een gedetecteerd event mediaan 184 s van het dichtstbijzijnde
referentie-event. Op PSG-IPA haalt hij F1 0,15–0,50 waar scoorders onderling
0,49–0,77 halen.

Bij de PLM-module is exact hetzelfde gemeten, en daar nog schriller:

| | algoritme | mens (mediaan scoorder) | F1 |
|---|---:|---:|---:|
| SN2 links | 152 events, 1,10 s | 135 events, 1,61 s | 0,014 |
| SN3 links | 341 events, 0,80 s | 332 events, 1,61 s | 0,045 |

Aantal en duur kloppen, de momenten niet. Dat de REFERENTIE deugt is apart
getoetst: in de menselijk geannoteerde intervallen ligt de EMG-RMS 6,3× (SN1)
en 10,2× (SN2) boven het niveau daarbuiten, tegen 1,44 en 1,03 voor dezelfde
events 60 s verschoven. De annotaties liggen dus exact op het signaal.

Een drempel verschuiven kan dit niet repareren. Wat een detector nodig heeft
die het juiste AANTAL maar de verkeerde MOMENTEN vindt, is een andere
selectieregel — niet een andere drempel.
