# PLM draait niet op MESA, en niet op elke enkelvoudige montage

*24 augustus 2026, 06:20. Bevinding, geen wijziging.*

## Wat er aan de hand is

`CHANNEL_PATTERNS` eist voor de beenrollen een **zijde-aanduiding**:

```
leg_l: leg l, lleg, emg leg l, tibial l, left leg, lat l, ta l, ...
leg_r: leg r, rleg, emg leg r, tibial r, right leg, lat r, ta r, ...
```

MESA levert één kanaal dat simpelweg **`Leg`** heet. Dat matcht geen van beide
rollen, dus `leg_l` en `leg_r` blijven leeg, en de pipeline neemt de tak

```python
output["plm"] = {"success": False, "error": "No leg-EMG channels", "summary": {}}
```

**Gevolg: op MESA wordt de PLM-analyse nooit uitgevoerd.**

## Hoe het opviel

De gecombineerde meting van vannacht gaf voor elk PLM-veld `None`, ook nadat
het harnas ze was gaan meeschrijven. Niet omdat het harnas ze miste, maar omdat
er niets te schrijven viel.

Dat verklaart ook waarom de gecombineerde uitkomst **exact** gelijk is aan de
strictness-alleen-uitkomst: de PLM-vlag kan op MESA per constructie niets doen.

## Wat dit betekent

1. **Elke PLM-uitspraak uit `validate_mesa.py` is onmogelijk**, niet
   onbetrouwbaar. De MESA-replicatie van de AASM-regel (+0,1091 op 13/16) is
   daar niet door geraakt: die riep `analyze_plm` rechtstreeks aan met een
   handmatig gekozen kanaal, buiten de pipeline om.
2. **De gecombineerde vraag is op MESA niet te beantwoorden.** Wat wél
   vaststaat is dat de strictness-validatie exact reproduceert op een
   onafhankelijke run.
3. **Het raakt meer dan MESA.** Een montage met één ongezijd beenkanaal —
   `Leg`, `EMG Leg`, `Tibialis` — verliest de PLM-analyse volledig, stil, met
   een foutmelding die "geen been-EMG" zegt terwijl het kanaal er wel is. Dat
   is dezelfde klasse als de thermistor die "niet in montage" heette terwijl
   hij in het EDF stond.

## Wat er zou moeten gebeuren

Een ongezijd beenkanaal hoort herkend te worden en op één been te scoren, met
in het rapport de vermelding dat er maar één afleiding was. Dat is een
gedragswijziging — opnames die nu geen PLM-index krijgen, krijgen er een — dus
achter een vlag met het huidige gedrag als default, en een meting die laat zien
dat de eenbenige index bruikbaar is.

**Niet gebouwd.** Dit document stelt alleen vast wat er mis is en hoe het
opviel.
