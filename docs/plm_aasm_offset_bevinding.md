# De duurbepaling van beenbewegingen volgt de AASM-regel niet

*24 augustus 2026, nacht. Bevinding met een gemeten omvang, GEEN voorstel.*

## De regel en de implementatie

AASM 2020, regel 4.A voor beenbewegingen:

- **onset**: het EMG stijgt ≥ **8 µV** boven de rustbasislijn;
- **einde**: het begin van een periode van ≥ 0,5 s waarin het EMG **niet boven
  2 µV** boven rust komt;
- duur tussen 0,5 en 10 s.

Onset en einde hebben dus **verschillende** drempels. `psgscoring/plm.py`
gebruikt er één:

```python
threshold = resting + LM_AMPLITUDE_UV      # 8 µV
labeled, n_bursts = label(rms > threshold)
dur_s = len(idx) * step_s                  # tijd BOVEN 8 µV
```

De gemeten duur is daarmee de tijd boven 8 µV in plaats van de tijd tot het
signaal onder 2 µV zakt. Die is stelselmatig **korter**, en een beweging die
maar net boven de drempel uitkomt zakt daardoor onder het minimum van 0,5 s.

## Waarom dit opviel

Op SN5 markeren de scoorders 217 bewegingen in slaap en vindt de detector er
36. Dat is geen amplitudeprobleem: **90 %** van die 217 haalt onze drempel wel
(mediane piek 12,6 µV boven rust). Ze zijn alleen **marginaal** — net boven de
8 µV — en brengen dus weinig tijd bóven die drempel door. Ter vergelijking:
op SN4, waar de detectie wél werkt, is die mediaan 45,0 µV.

## De omvang, gemeten

Per been geteld, in slaap, zonder bilaterale samenvoeging of de overige
pipelinefilters:

| | scoordermediaan | huidige regel | AASM-einde | factor |
|---|---:|---:|---:|---:|
| SN5 | 254 | 204 | 1880 | **9,22×** |
| SN4 | 669 | 949 | 1330 | 1,40× |
| SN3 | 139 | 556 | \phantom{0}996 | 1,79× |

## Wat dit wel en niet zegt

**Wel:** de niet-conformiteit is echt en de gevolgen zijn groot — de
duurbepaling verandert de telling met een factor 1,4 tot 9,2.

**Niet:** dat de AASM-regel toepassen de overeenstemming verbetert. Op SN5 zou
hij de onderdetectie ruimschoots wegnemen en waarschijnlijk doorschieten; op
SN3 en SN4 telt de huidige regel per been al méér dan de scoorders, en de
AASM-regel zou dat verergeren.

**Let op bij het lezen van die tabel:** dit zijn per-been-sommen zonder
bilaterale samenvoeging, terwijl de pipeline die wel doet (SN3 levert
uiteindelijk 97 events, niet 556). De verhoudingen zijn informatief, de
absolute aantallen niet vergelijkbaar met de scoorders.

## Wat er zou moeten gebeuren

De regel conform maken is verdedigbaar op zichzelf — een implementatie hoort
te doen wat het handboek zegt — maar het is een gedragswijziging die de
PLM-index fors verschuift. Dus: achter een profielvlag met het huidige gedrag
als default, en een preregistratie met de uitkomstmaat vooraf gekozen en
gescheiden kalibratie- en validatiesteekproeven.

Wat daarbij hoort en hier nog ontbreekt: de bilaterale samenvoeging en de
overige filters meenemen, want die doen aantoonbaar veel werk en de omvang
hierboven is daar niet doorheen gerekend.
