# Het arousal-koppelvenster, gemeten in plaats van gekozen

*21 augustus 2026. Uit W35b van `docs/PAPER_REVISIE_v40.md`: `Resp_events/`
draagt twaalf onafhankelijke respiratoire scoringen tegen ÉÉN vaste
arousal-annotatie die de scoorders zagen tijdens het scoren. Daarmee is de
koppeling die mensen maakten direct af te lezen.*

## Opzet

Voor elke door een mens gescoorde hypopnee: de tijd van het eventEINDE tot de
eerstvolgende arousal-onset. Nul-verdeling: dezelfde events 120 s verschoven,
zodat de arousal-dichtheid gelijk blijft maar de koppeling verdwijnt. Alleen
annotaties, geen signaalverwerking.

Eerst geverifieerd dat de opzet klopt: de arousal-annotatie is per opname
identiek over de twaalf scoorders (SN1 en SN5 exact één set; SN2, SN3 en SN4
twee à drie sets die één event verschillen), terwijl de respiratoire
scoringen alle twaalf verschillen — op SN4 van 1 tot 38 events, op SN5 van 25
tot 101.

## Resultaat

1601 hypopnee-scoringen over vijf opnames en twaalf scoorders.

| venster | waargenomen | nul | overschot |
|---:|---:|---:|---:|
| 2 s | 10,6 % | 2,0 % | 8,6 |
| 5 s | 21,4 % | 4,1 % | 17,3 |
| **10 s** | 29,1 % | 8,1 % | **21,0** |
| **15 s** | 32,6 % | 12,6 % | **20,0** |
| 20 s | 33,5 % | 17,6 % | 15,9 |
| 30 s | 37,3 % | 27,7 % | 9,6 |
| 60 s | 50,2 % | 41,1 % | 9,1 |

Het overschot boven toeval piekt op **10 s** en is vlak tot 15 s. Daarboven
groeit het waargenomen aandeel nog wel, maar het overschot krimpt: het venster
vult zich dan met koppelingen die net zo goed toeval kunnen zijn.

## Conclusie

**`RULE1B_AROUSAL_WINDOW_S = 15.0` is goed gekozen.** Het vangt praktisch alle
koppeling die boven toeval uitkomt (20,0 tegen een piek van 21,0), en is
daarmee hooguit een fractie ruim. 10 s zou marginaal specifieker zijn; het
verschil is te klein om een gescoorde waarde voor te verzetten.

Dit is een parameter die tot nu toe gekozen was en nu gemeten is. Dat de
uitkomst "laat staan" luidt, maakt hem niet minder waard: het haalt hem van
de lijst met vrije parameters waarover een reviewer terecht een vraag stelt
(W34c/W34e).

## Wat het over de Rule 1A-tak zegt

Slechts **32,6 %** van de door mensen gescoorde hypopneus heeft überhaupt een
arousal binnen 15 s, waarvan ongeveer 20 procentpunt boven toeval. De
arousaltak kan dus hoogstens een vijfde van de hypopneus raken — en dan
alleen díé zonder kwalificerende desaturatie.

Dat komt overeen met wat destijds op PSG-IPA gemeten is toen de tak aanstond:
11 van 74, 20 van 70 en 38 van 170 afgewezen kandidaten koppelden aan een
arousal (15–22 %), goed voor +1,2 tot +3,4 AHI.

**Bijstelling van de verwachting in
`docs/rule1a_arousal_preregistratie.md`:** de tak kan plausibel 1 tot 3/u van
het gat van −5,26/u dichten, niet het hele gat. Het primaire criterium daar
(≥1,5/u biasdaling) ligt daarmee middenin het haalbare bereik in plaats van
er ruim onder — dat is scherper dan ik wist toen ik het opschreef, en het is
maar goed dat het getal vóór deze meting vastlag.

Meetscript: `docs/meet_koppelvenster_psgipa.py`.
