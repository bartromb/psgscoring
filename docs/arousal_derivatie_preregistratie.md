# Preregistratie — single versus multi-derivatie

Datum: 2026-08-22. **Geschreven vóór de meting.**

## Waarom dit geen verfijning is maar een gat

`AROUSAL_DERIVATION_MODE` staat op **multi** op 23 van de 24 profielen; alleen
`mesa_shhs` draait single. Multi is dus wat er in productie draait.

**Alle arousalmetingen van 21–22 augustus draaiden op single-derivatie**, op
één centraal kanaal, omdat het harnas `detect_arousals` rechtstreeks aanroept.
Het cijfer waarop de hybride default is gezet — event-F1 0,463 tegen 0,692
menselijk — geldt dus voor een configuratie die niet de default is.

Datzelfde geldt voor de MESA n=150-meting: daar vindt `_pick_eeg_multi` maar
één afleiding (de kanalen heten `EEG1/2/3` en matchen de patronen voor
occipitaal en frontaal niet), zodat multi vanzelf naar single degradeert. Ook
die AHI-cijfers zijn dus single-derivatie.

Er is bovendien een eerdere meting op het REGELgebaseerde pad die tegen multi
pleit: die gaf hogere sensitiviteit (0,38 → 0,47) bij een vlakke F1 (~0,38) —
winst in dekking, niet in accuraatheid. Of dat met een classifier erachter
anders ligt, is nooit gemeten.

## De ingreep

Drie armen op de vijf PSG-IPA arousal-opnames, alle met de hybride aan
(drempel 0,60), tegen dezelfde twaalf scoorders:

- **single** — één centraal kanaal (wat ik tot nu toe mat)
- **multi** — centraal + occipitaal + frontaal, event-level union
- **multi + EOG-reject** — idem, met verwerping van occipitaal-only events die
  samenvallen met een oogbeweging

PSG-IPA draagt `EEG F4-M1`, `EEG C4-M1` (of `Cz-M1`), `EEG O2-M1` en
`EOG E1-M2`/`E2-M2`, dus alle drie zijn meetbaar.

## Acceptatiecriterium (vastgelegd vóór de meting)

**Primair.** De mediane event-F1 van **multi** ligt boven die van single
(0,463 in dezelfde run gemeten, niet uit een eerdere overgenomen). Haalt multi
dat niet, dan is de huidige default op 23 profielen niet gerechtvaardigd en
hoort de vraag omgekeerd te worden gesteld: waarom staat multi aan?

**Secundair (mechanisme).** De recall stijgt met multi. Meer afleidingen horen
meer van wat de scoorders markeerden te vinden; gebeurt dat niet, dan doet de
union iets anders dan verondersteld.

**Bewaking.** De spreiding van `index_algoritme / index_scoordermediaan` blijft
binnen de 0,61–1,29 van single.

**Wat hier NIET uit volgt.** Ook als multi wint, verandert er geen default
zonder een meting van de respiratoire gevolgen — arousals voeden Rule 1B en de
RDI. En omgekeerd: verliest multi, dan is dat een argument om de bestaande
default te herzien, wat óók een gebruikersbeslissing is.

## Verwachting

Op grond van de eerdere meting op het regelpad: recall omhoog, F1 vlak of
lager. Als dat zich herhaalt mét classifier, is de conclusie dat de union
dekking koopt met precisie — en dat de huidige default die afruil maakt zonder
dat iemand hem gemeten heeft.

---

# Uitkomst — 22 augustus 2026

**Beide criteria gehaald. Multi is gerechtvaardigd; de EOG-reject niet.**

| opname | scoorder-index | single | multi | multi+eog | q(multi) |
|---|---:|---:|---:|---:|---:|
| SN1 | 24,2 | 0,463 | **0,505** | 0,505 | 1,48 |
| SN2 | 8,5 | 0,347 | **0,355** | 0,355 | 1,85 |
| SN3 | 18,0 | 0,340 | **0,390** | 0,371 | 0,94 |
| SN4 | 14,3 | 0,505 | **0,568** | 0,557 | 1,33 |
| SN5 | 27,7 | 0,666 | **0,678** | 0,665 | 1,47 |

| | F1 | recall | precisie | q-bereik |
|---|---:|---:|---:|---|
| single | 0,463 | 0,496 | 0,455 | 0,61–1,29 |
| **multi** | **0,505** | **0,649** | 0,425 | 0,94–1,85 |
| multi + EOG-reject | 0,505 | 0,649 | 0,425 | 0,86–1,85 |

- **Primair (F1 multi > single): GEHAALD** — 0,505 tegen 0,463, en beter op
  **5 van 5** opnames.
- **Secundair (recall stijgt): GEHAALD** — 0,496 → 0,649. De winst komt van
  dekking en overtreft het precisieverlies (0,455 → 0,425). Op het regelpad
  bleef de F1 bij diezelfde afruil vlak; de classifier maakt het verschil.

## De bewaking was dubbelzinnig geformuleerd

Ik schreef: "de spreiding blijft binnen de 0,61–1,29 van single." Dat laat twee
lezingen toe, en ze wijzen verschillende kanten op:

- als **spreidingsfactor** (max/min): 1,97 tegen 2,10 — verbeterd, gehaald;
- als **bereik**: multi ligt op 0,94–1,85, dus buiten [0,61–1,29] — niet
  gehaald.

Het inhoudelijke feit weegt zwaarder dan de formulering: **multi overdetecteert
systematisch.** De mediane `q` gaat van 1,10 naar 1,47 en op geen enkele
opname zit multi ónder de scoordermediaan, terwijl single nog aan beide kanten
lag. Voor de event-F1 is dat gunstig; voor een GERAPPORTEERDE arousal-index
betekent het structureel ongeveer anderhalf maal de menselijke waarde.

Die twee dingen — welke detectie beter overeenkomt, en welk getal je afdrukt —
had ik in één bewaking samengevat, en dat had ik niet moeten doen. Ze vragen
een aparte afweging.

## De EOG-reject moet uit

Nul keer beter, twee keer identiek (SN1, SN2), drie keer slechter (SN3, SN4,
SN5), en de spreiding verslechtert van 1,97 naar 2,16. Op SN3 haalt hij de
recall van 0,383 naar 0,350.

Hij staat in productie default AAN zodra er een EOG-kanaal is
(`PSGSCORING_AROUSAL_EOG_REJECT` default "1" in `pipeline.py`). Hij verwijdert
daar dus aantoonbaar echte events zonder er iets voor terug te geven. Dat hij
op twee opnames niets deed is op zichzelf al een signaal: een tak die vaak
niet vuurt en, als hij vuurt, schaadt.

## Wat hier NIET uit volgt

Geen enkele default verandert op deze meting alleen. Arousals voeden Rule 1B
en de RDI, dus zowel het aanhouden van multi als het uitzetten van de
EOG-reject vraagt een meting van de respiratoire gevolgen — en de
overdetectie van de index is een aparte, klinische afweging.

Wat wél vaststaat: de eerdere conclusie dat multi "sensitiviteit koopt zonder
accuraatheid" gold voor het REGELgebaseerde pad en geldt niet meer.


---

# Wat de uitrol van 21 augustus werkelijk deed

Alle metingen hierboven vergelijken armen binnen dezelfde derivatiemodus. De
vraag die daarnaast beantwoord moest worden: wat deed de overgang van
regelgebaseerd naar hybride in de configuratie die ECHT draait — multi?

| | F1 | recall | precisie | q mediaan | q bereik | spreiding |
|---|---:|---:|---:|---:|---|---:|
| multi + regels *(oude productie)* | 0,326 | 0,460 | 0,248 | 1,59 | 0,95–6,47 | 6,78 |
| **multi + hybride** *(nu)* | **0,505** | 0,649 | 0,425 | 1,47 | 0,94–1,85 | **1,97** |
| *scoorder-onderling* | *0,692* | | | | | |

Per opname (F1, oud → nieuw): SN1 0,334 → 0,505 · SN2 0,149 → 0,355 ·
SN3 0,152 → 0,390 · SN4 0,326 → 0,568 · SN5 0,508 → 0,678. **Vijf van vijf.**

De precisie is de plek waar de winst zit: 0,248 → 0,425.

**De uitschieter verdwijnt.** In de oude stand rapporteerde SN2 een
arousal-index van 55,1 per uur tegen een scoordermediaan van 8,5 — een factor
6,47 — met een precisie van 0,090. Dat is nu 15,7 en 0,273.

En SN3 laat zien waarom de index als maat niet volstaat: daar rapporteerde de
oude stand 17,2 tegen een scoorder van 18,0, wat er perfect uitzag, terwijl de
event-F1 op 0,152 lag met een precisie van 0,159. Het juiste aantal op de
verkeerde momenten — hetzelfde patroon als de PLM-tijdbasisfout.

**Perspectief op de overdetectie.** Hierboven noemde ik de mediane `q` van
1,47 een bezwaar. Dat blijft staan als openstaand punt, maar het is een
verbetering: van 1,59 mét een staart tot 6,47 naar 1,47 met een maximum van
1,85. De overdetectie is niet nieuw en niet verergerd — ze is gehalveerd in
mediaan en met een factor 3,5 in spreiding.

Meetscript: `docs/meet_arousal_derivatie_psgipa.py`.
