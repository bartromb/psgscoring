# Event-locked werkpunt: gebouwd, gemeten, niet aangezet

**Datum:** 2026-08-25
**Vlag:** `arousal_event_locked_threshold`, **default `None` (uit)** op alle profielen
**Vervolg op:** `docs/arousal-recall-diagnose.md` (D3)

---

## Waarom bij de filter en niet bij de kandidaatgeneratie

D3 stelde voor de KANDIDAATdrempels te verlagen rond een respiratoir
event-einde. Gemeten op PSG-IPA SN3 (327 menselijke events):

| | |
|---|---:|
| kandidaat in het venster (ruime drempels) | 283 (87 %) |
| overleeft de filter op 0,80 | 89 (27 %) |
| **events zonder énige kandidaat** — wat D3 kan winnen | **44** |
| **events mét kandidaat, weggefilterd** — filterverlies | **194** |

SN5 geeft hetzelfde beeld (11 tegen 40). Het filterverlies is ruim vier keer
zo groot; kandidaten toevoegen waar er al een ligt die weggegooid wordt,
verandert niets. De ingreep hoort bij de filter.

**Waarom dat verdedigbaar is:** het model levert een kans, geen beslissing. De
drempel waarop je die afkapt hoort van de prior af te hangen, en die is vlak na
een event-einde aantoonbaar anders. Het venster is exact dat van
`correlate_arousals_to_respiratory` (event-onset tot 15 s na het einde), zodat
detectie en koppeling dezelfde geometrie delen.

## Twee aannames uit het diagnosedocument die niet klopten

**De two-pass is niet nodig.** Het document ging ervan uit dat arousals vóór de
respiratoire scoring draaien. Ze draaien erná (events regel 376, arousals regel
740) en `resp_events` staat al in de signatuur.

**Maar er is een ander volgordeprobleem.** Op de `breath_graded`-profielen — de
klinische defaults — vervangt stap 7b de hypopneeën **ná** de arousalstap. Het
venster ziet daar alleen de apneus: op de motiverende opname 235 van de 377
events (63 %). Repareren vraagt een tweede pas ná stap 7b, en dat is een aparte
ingreep: de breath-detector gebruikt de arousals zelf
(`HYPOPNEA_AROUSAL_WEIGHT`), dus er is een echte cyclische afhankelijkheid.
Vastgelegd in `tests/test_arousal_event_locked_threshold.py`, die faalt zodra
de volgorde verandert.

## De meting

Beslisregel vooraf: de koppelingsfractie `ons × mens` moet stijgen ÉN de
telling mag niet boven ratio 1,10 uitkomen (grens uit het diagnosedocument).

PSG-IPA, volle pijplijn, `aasm_v3_breath`, gepoold op eventaantal:

| vensterdrempel | koppeling `ons × mens` | telling / referentie |
|---|---:|---:|
| **uit** | **36,4 %** | **0,99** |
| 0,60 | 44,4 % | 1,11 |
| 0,50 | 47,4 % | 1,15 |
| 0,40 | 51,6 % | 1,20 |
| 0,30 | 54,2 % | 1,25 |
| *mens × mens* | *60,4 %* | *1,00* |

**Het mechanisme werkt**: de koppeling stijgt monotoon en dicht driekwart van
het gat naar de mens. **De bewaking faalt**: zelfs de mildste stand komt op
1,11, net boven de grens. Elke stand die de koppeling noemenswaardig verbetert,
koopt dat met events die de referentie niet kent.

**Geen enkele stand haalt beide regels. De vlag blijft uit.**

De afruil is echt en is een gebruikersbeslissing, geen meetuitkomst: op 0,60
win je 8 procentpunt koppeling voor 11 % meer arousals. Gaat die keuze ooit
door, dan hoort er een MESA-replicatie vóór de default-flip.

## De volledige meting, mét F1 (25-08, beide cohorten)

De tabel hierboven mist de maat die er het meest toe doet. Koppeling vraagt
"volgt er een arousal op een respiratoir event"; **F1 vraagt of onze arousals
staan waar de MENSELIJKE staan**. Een vlag die arousals bij event-eindes zet
kan de eerste optillen zonder de tweede te raken -- en dat is precies wat er
gebeurt.

| cohort | arm | koppeling | **F1** | telling / ref |
|---|---|---:|---:|---:|
| PSG-IPA (n=5) | uit | 36,4 % | **0,546** | 0,99 |
| | 0,75 | 39,5 % | 0,546 | 1,02 |
| | 0,70 | 41,2 % | 0,554 | 1,04 |
| | 0,65 | 43,6 % | **0,564** | 1,09 |
| | 0,60 | 44,4 % | 0,559 | 1,11 |
| MESA (n=20) | uit | 24,2 % | **0,409** | 0,71 |
| | 0,75 | 25,1 % | 0,406 | 0,73 |
| | 0,70 | 25,6 % | 0,414 | 0,74 |
| | 0,65 | 26,2 % | **0,420** | 0,76 |
| | 0,60 | 26,6 % | 0,420 | 0,77 |

**+0,018 F1 op PSG-IPA, +0,011 op MESA.** De koppeling steeg met 7,2
respectievelijk 2,4 procentpunt. Het verschil tussen die twee getallen ís de
bevinding: de toegelaten arousals landen bij event-eindes waar de mens een
EVENT had, niet waar hij een AROUSAL scoorde.

Per opname is het bovendien inconsistent: op PSG-IPA SN2 gaat F1 achteruit
(0,51 → 0,50), op SN4 verandert niets (het venster laat er vijf toe), en op
MESA 0002/0014/0028/0035/0054 beweegt niets.

Tegen het menselijk plafond van **0,679** (zie `arousal_menselijk_plafond.md`)
dicht de vlag 13 % van het resterende gat op PSG-IPA en 8 % op MESA.

**Blijft uit, nu op drie maten gemeten in plaats van twee.**

## Correctie op eerdere formuleringen

De arousals in de `Resp_events`-subtree zijn een **gedeelde
referentie-annotatie**, geen scoring per scoorder: 12 bestanden bevatten
**3 unieke arousalsets** (241/242/242…). In `EEG_arousals` zijn het er 12 van
12. De formulering "mediaan over 60 scoorder-nachten" in de 0.27.4-changelog is
daarmee onjuist — het zijn 12 verschillende EVENTsets tegen in feite één
arousalset.

De vergelijking `mens × mens` tegenover `ons × mens` blijft geldig: beide
gebruiken dezelfde arousalbron en dezelfde eventsets, dus als maat voor
PLAATSING deugt ze. De spreiding die erbij genoemd werd was die van de events,
niet van de arousals.

**Tweede val:** de subtrees zijn andere exports met andere duur (SN3: 6,57 u in
`Resp_events`, 8,13 u in `EEG_arousals`). Tellingen uit het ene bestand naast
spreidingen uit het andere leggen is betekenisloos.
