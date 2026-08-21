# De PLM-detector rekende met de verkeerde klok

*21 augustus 2026. Gevonden tijdens het onderzoek naar de arousaldetector,
omdat beide modules hetzelfde symptoom vertoonden: het juiste AANTAL events
van de juiste DUUR, op de verkeerde MOMENTEN.*

## De fout

`psgscoring/plm.py`, `_detect_lm_channel`:

```python
win = max(1, int(sf * 0.1))                     # aantal STALEN per venster
n_w = len(filt) // win
rms = np.array([... for i in range(n_w)])       # één RMS per venster
...
dur_s = len(idx) * 0.1                          # <-- neemt 0,1 s aan
"onset_s": idx[0] * 0.1,                        # <-- idem
```

`win` is een geheel aantal stalen. Bij 256 Hz is `int(256 × 0,1) = 25`, en
25 stalen duren **0,09766 s**, niet 0,1 s. De omzetting van vensterindex naar
tijd vermenigvuldigt toch met 0,1.

De fout is niet constant maar **stapelt lineair over de nacht**:

| opname | sf | venster | drift aan het eind |
|---|---:|---:|---:|
| PSG-IPA SN1 (7,4 u) | 256 Hz | 0,09766 s | **+620 s** (10,3 min) |
| PSG-IPA SN3 (7,8 u) | 256 Hz | 0,09766 s | +657 s (10,9 min) |
| MESA (12 u) | 256 Hz | 0,09766 s | +1013 s (16,9 min) |

De richting is altijd dezelfde: events worden **te laat** gerapporteerd.

Het treft de gangbare sample rates 256 Hz (2,3 % fout) en 128 Hz (6,3 %
fout: `int(12,8) = 12`). Bij 100, 200 en 500 Hz is `sf × 0,1` toevallig
geheel en is er geen fout — wat verklaart waarom dit nooit is opgevallen.

## Wat het kostte

Event-F1 tegen twaalf menselijke scoorders op PSG-IPA, greedy IoU-matching
op 0,20, linkerbeen tegen `limb movement@@emg lat`:

| | huidig | tijdbasis gecorrigeerd | mens tegen mens |
|---|---:|---:|---:|
| SN1 | 0,038 (431 ev) | **0,592** (372 ev) | 0,745 |
| SN2 | 0,014 (152 ev) | **0,699** (137 ev) | 0,820 |
| SN3 | 0,045 (341 ev) | **0,692** (278 ev) | 0,909 |

De detector was dus niet slecht in wat hij vindt — hij zette het op de
verkeerde tijd. Met de klok recht komt hij op ~0,15 van de menselijke
overeenstemming.

## Wat er verderop van afhangt

De drift blijft niet binnen de PLM-module. `pipeline.py:1968-1979` berekent
`plm_arousal_index` door een PLM-onset te koppelen aan een arousal-onset
binnen **−0,5 tot +3 s**. Met onsets die door de nacht tot tien minuten
verschuiven is die koppeling toeval, en `plm_arousal_index` staat in het
klinische PDF-rapport (`generate_pdf_report.py:2351`). Hetzelfde geldt voor
`_exclude_resp_associated`, dat beenbewegingen uitsluit die binnen enkele
seconden na een respiratoir event vallen: die uitsluiting werkt nu op
verschoven tijden.

Twee dingen die daar bovenop komen en apart aandacht verdienen:

- `result["events"]` is `plm_eligible[:200]` — alleen bewegingen in slaap
  zonder respiratoire koppeling, afgekapt op 200. Op SN1 zijn dat er 200 van
  660 gedetecteerde bewegingen. Alles wat verderop `output["plm"]["events"]`
  leest — inclusief `plm_arousal_index` en de signaalweergave — ziet dus
  hooguit de eerste 200 van de nacht.
- YASAFlaskified schrijft deze onsets door naar de **EDF+-export**
  (`generate_edfplus.py:155-161`). Wie dat bestand in een viewer opent, ziet
  de PLM-markering tot tien minuten naast de beweging staan. Dat is de
  afleverkant: de bibliotheek kan het getal intern consistent hebben, maar
  wat de gebruiker ziet is verschoven.
- De menselijke PSG-IPA-annotaties staan PER BEEN
  (`limb movement@@emg lat` en `@@emg rat`), terwijl de module bilateraal
  samenvoegt binnen 0,5 s. Voor de validatie hierboven is daarom per been
  vergeleken.

## Waarom dit niet eerder opviel

Het AANTAL klopte al. `n_lm_total` en de PLM-index worden per uur berekend en
zijn ongevoelig voor een verschuiving; alleen een event-voor-event
vergelijking legt het bloot. Die vergelijking bestond niet: de PLM-module is
tot 20 augustus 2026 nooit tegen menselijke scoorders gelegd.

Dat de referentie deugt, is apart getoetst voordat er iets gerepareerd werd:
in de menselijk geannoteerde intervallen ligt de EMG-RMS 6,3× (SN1) en 10,2×
(SN2) boven het niveau daarbuiten, tegen 1,44 en 1,03 voor dezelfde events
60 s verschoven. De annotaties liggen exact op het signaal.

## De reparatie

`stap = win / sf` in plaats van de aangenomen 0,1, voor zowel `onset_s` als
`duration_s`. Achter profielvlag `plm_time_base`, default uit, conform de
staande regel dat gedragswijzigingen achter een vlag gaan met het huidige
gedrag als default.

**Aanbeveling: default aanzetten.** Het huidige gedrag is niet een andere
keuze maar een rekenfout, en de gerapporteerde tijden lopen tot een kwartier
uit de pas — zichtbaar zodra iemand een beenbeweging opzoekt in de
signaalweergave van `/review/<job_id>`. De PLM-index zelf schuift mee (het
duurfilter werkt nu op de juiste duur), dus het is geen zuiver cosmetische
wissel en de knop hoort bij de gebruiker te liggen.
